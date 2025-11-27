import asyncio
import websockets
import numpy as np
import json
import time
import torch

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

SAMPLE_RATE = 16000

SILENCE_MS = 300
ENERGY_THRESHOLD = 150
MIN_SPEECH_MS = 300

SILERO_THRESHOLD = 0.5
MIN_SILERO_SAMPLES = SAMPLE_RATE // 10

MAX_TURN_MS = 3500
MAX_AUDIO_BYTES = int(SAMPLE_RATE * 2 * (MAX_TURN_MS / 1000.0))


class SmartTurnDetector:
    def __init__(self):
        try:
            self.model = LocalSmartTurnAnalyzerV3(
                params=SmartTurnParams(
                    stop_secs=3.0,
                    pre_speech_ms=0,
                    max_duration_secs=8.0,
                ),
                cpu_count=1,
            )
        except Exception as e:
            print("Model load error:", e)
            raise

        try:
            torch.set_num_threads(1)
            self.silero_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self.silero_model.eval()
        except Exception:
            self.silero_model = None

        self.reset()

    def reset(self):
        self.audio_buffer = bytearray()
        self.is_speaking = False
        self.last_speech_ms = None
        self.speech_start_ms = None

    @staticmethod
    def rms(pcm: bytes) -> float:
        if not pcm:
            return 0.0
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0

    def silero_prob(self, pcm_data: bytes) -> float:
        if not self.silero_model or not pcm_data:
            return 1.0
        try:
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_np) < 512:
                return 0.0
            probs = []
            with torch.no_grad():
                for i in range(0, len(audio_np) - 512 + 1, 512):
                    chunk = audio_np[i:i + 512]
                    tensor = torch.from_numpy(chunk).unsqueeze(0).float()
                    probs.append(self.silero_model(tensor, SAMPLE_RATE).item())
            return float(np.mean(probs)) if probs else 0.0
        except Exception:
            return 1.0

    def finalize(self):
        if not self.is_speaking or not self.last_speech_ms or not self.speech_start_ms:
            self.reset()
            return None

        duration = self.last_speech_ms - self.speech_start_ms
        if duration < MIN_SPEECH_MS or not self.audio_buffer:
            self.reset()
            return None

        pcm_bytes = bytes(self.audio_buffer)
        if len(pcm_bytes) >= MIN_SILERO_SAMPLES * 2:
            prob = self.silero_prob(pcm_bytes)
        else:
            prob = 1.0

        if prob < SILERO_THRESHOLD:
            self.reset()
            return None

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            if hasattr(self.model, "predict_endpoint"):
                result = self.model.predict_endpoint(audio)
            else:
                result = self.model._predict_endpoint(audio)
        except Exception:
            self.reset()
            return None

        self.reset()
        return result

    def process_audio(self, chunk: bytes):
        if not chunk or len(chunk) % 2 != 0:
            return None

        now = time.time() * 1000.0
        energy = self.rms(chunk)

        if energy > ENERGY_THRESHOLD:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_ms = now
                self.audio_buffer = bytearray()

            self.last_speech_ms = now
            self.audio_buffer.extend(chunk)

            if len(self.audio_buffer) >= MAX_AUDIO_BYTES:
                return self.finalize()

            return None

        if self.is_speaking and self.last_speech_ms:
            silence = now - self.last_speech_ms
            duration = self.last_speech_ms - self.speech_start_ms

            if duration >= MAX_TURN_MS or silence >= SILENCE_MS:
                return self.finalize()

        return None


async def handle_client(ws):
    detector = SmartTurnDetector()
    print("Client connected:", ws.remote_address)

    try:
        async for message in ws:
            if isinstance(message, (bytes, bytearray)):
                result = detector.process_audio(message)
                if result:
                    prob = float(result.get("probability", 0.0))
                    await ws.send(json.dumps({
                        "type": "turn_complete",
                        "probability": prob
                    }))
            else:
                continue

    except Exception as e:
        print("Client error:", e)

    finally:
        print("Client disconnected:", ws.remote_address)


async def main():
    async with websockets.serve(handle_client, "0.0.0.0", 9001):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
