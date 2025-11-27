import asyncio
import websockets
import numpy as np
import json
import time
import torch

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

SAMPLE_RATE = 16000
ENERGY_THRESHOLD = 120
SILENCE_MS = 80
MIN_SPEECH_MS = 200
SILERO_THRESHOLD = 0.25
MAX_TURN_MS = 2500
MAX_AUDIO_BYTES = int(SAMPLE_RATE * 2 * (MAX_TURN_MS / 1000))


class SmartTurnDetector:
    def __init__(self):
        self.model = LocalSmartTurnAnalyzerV3(
            params=SmartTurnParams(
                stop_secs=2.0,
                pre_speech_ms=0,
                max_duration_secs=5.0,
            ),
            cpu_count=1,
        )

        try:
            torch.set_num_threads(1)
            self.silero_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self.silero_model.eval()
        except:
            self.silero_model = None

        self.reset()

    def reset(self):
        self.audio_buffer = bytearray()
        self.is_speaking = False
        self.speech_start_ms = None
        self.last_speech_ms = None

    def rms(self, pcm):
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(arr * arr))) if arr.size else 0

    def silero_prob(self, pcm):
        if not self.silero_model or not pcm:
            return 1.0

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        if len(audio) < 512:
            return 1.0

        with torch.no_grad():
            probs = []
            for i in range(0, len(audio) - 512 + 1, 512):
                chunk = audio[i:i + 512]
                t = torch.from_numpy(chunk).unsqueeze(0)
                probs.append(self.silero_model(t, SAMPLE_RATE).item())

        return float(np.mean(probs)) if probs else 1.0

    def finalize(self):
        if not self.is_speaking or not self.speech_start_ms or not self.last_speech_ms:
            self.reset()
            return None

        duration = self.last_speech_ms - self.speech_start_ms
        if duration < MIN_SPEECH_MS:
            self.reset()
            return None

        pcm = bytes(self.audio_buffer)

        if self.silero_prob(pcm) < SILERO_THRESHOLD:
            self.reset()
            return None

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768

        try:
            result = self.model.predict_endpoint(audio)
        except:
            result = self.model._predict_endpoint(audio)

        self.reset()
        return result

    def process(self, chunk):
        if not chunk or len(chunk) % 2:
            return None

        now = time.time() * 1000
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

            if silence > SILENCE_MS or duration > MAX_TURN_MS:
                return self.finalize()

        return None


async def handle_client(ws):
    detector = SmartTurnDetector()
    print("Client connected:", ws.remote_address)

    try:
        try:
            async for chunk in ws:
                if isinstance(chunk, (bytes, bytearray)):
                    res = detector.process(chunk)

                    if res:
                        pred = res.get("prediction")
                        if isinstance(pred, dict):
                            pred = pred.get("value", 0)

                        completed = int(
                            bool(pred)
                            or bool(res.get("is_endpoint"))
                            or bool(res.get("endpoint"))
                        )

                        await ws.send(json.dumps({
                            "type": "turn_complete",
                            "completed": completed
                        }))
                        
                        print("completed: ",completed)

        except websockets.exceptions.ConnectionClosedError:
            pass
        except websockets.exceptions.ConnectionClosedOK:
            pass
        except Exception:
            pass

    finally:
        print("Client disconnected:", ws.remote_address)


async def main():
    async with websockets.serve(handle_client, "0.0.0.0", 9001):
        print("SmartTurn running at ws://localhost:9001")
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            print("SmartTurn server shutting down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
