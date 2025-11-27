import asyncio
import websockets
import numpy as np
import json
import time
import torch

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

SAMPLE_RATE = 16000

# Energy-based VAD params
SILENCE_MS = 300          # how long of silence to consider end of speech
ENERGY_THRESHOLD = 150    # RMS threshold, tune for your mic
MIN_SPEECH_MS = 300       # minimum speech duration

# Silero params
SILERO_THRESHOLD = 0.5
MIN_SILERO_SAMPLES = SAMPLE_RATE // 10  # at least 100ms

# Hard caps to avoid huge turns / buffers
MAX_TURN_MS = 3500  # 3.5s of speech per turn
MAX_AUDIO_BYTES = int(SAMPLE_RATE * 2 * (MAX_TURN_MS / 1000.0))  # 16kHz, int16


class SmartTurnDetector:
    def __init__(self):
        print("🔄 Initializing SmartTurn Detector...")

        # SmartTurn model
        self.model = LocalSmartTurnAnalyzerV3(
            params=SmartTurnParams(
                stop_secs=3.0,       # SmartTurn internal stop
                pre_speech_ms=0,
                max_duration_secs=8.0,  # still keep a safety here
            ),
            cpu_count=1,
        )

        print("📦 Loading Silero VAD model...")
        try:
            torch.set_num_threads(1)
            self.silero_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self.silero_model.eval()
            print("✓ Silero VAD model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading Silero VAD: {e}")
            self.silero_model = None

        self.reset()

    def reset(self):
        self.audio_buffer = bytearray()
        self.is_speaking = False
        self.last_speech_ms = None
        self.speech_start_ms = None

    @staticmethod
    def calculate_rms(pcm: bytes) -> float:
        if not pcm:
            return 0.0
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(arr * arr)))

    def get_speech_probability(self, pcm_data: bytes) -> float:
        if not self.silero_model or not pcm_data:
            # If Silero is not available, just accept as speech
            return 1.0

        try:
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_np) < 512:
                return 0.0

            probabilities = []

            with torch.no_grad():
                for i in range(0, len(audio_np) - 512 + 1, 512):
                    chunk = audio_np[i:i + 512]
                    tensor = torch.from_numpy(chunk).float().unsqueeze(0)  # shape [1, 512]
                    prob = self.silero_model(tensor, SAMPLE_RATE).item()
                    probabilities.append(prob)

            return float(np.mean(probabilities)) if probabilities else 0.0

        except Exception as e:
            print(f"⚠️ Silero error: {e}")
            # Fail-open: treat as speech so we don't drop everything
            return 1.0

    def _finalize_turn(self, reason: str):
        """
        Finalize the current speech turn:
        - Check minimum duration
        - Run Silero verification (if available and long enough)
        - Run SmartTurn model
        """
        if not self.is_speaking or self.speech_start_ms is None or self.last_speech_ms is None:
            self.reset()
            return None

        speech_duration = self.last_speech_ms - self.speech_start_ms

        if speech_duration < MIN_SPEECH_MS or not self.audio_buffer:
            print(f"ℹ️ Turn dropped ({reason}): too short ({speech_duration:.0f}ms)")
            self.reset()
            return None

        pcm_bytes = bytes(self.audio_buffer)
        byte_len = len(pcm_bytes)

        # Silero verification, only if we have enough samples
        if byte_len >= MIN_SILERO_SAMPLES * 2:  # samples * 2 bytes
            speech_prob = self.get_speech_probability(pcm_bytes)
        else:
            speech_prob = 1.0  # trust energy-based VAD for very short chunks

        print(
            f"📊 [{reason}] Speech probability: {speech_prob:.3f}, "
            f"Duration: {speech_duration:.0f}ms, Bytes: {byte_len}"
        )

        if speech_prob < SILERO_THRESHOLD:
            print(f"❌ Rejected ({reason}): speech probability too low ({speech_prob:.3f})")
            self.reset()
            return None

        # Convert to float32 [-1, 1] for SmartTurn
        float_audio = (
            np.frombuffer(pcm_bytes, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )

        # SmartTurn inference (handle both public/private API)
        try:
            if hasattr(self.model, "predict_endpoint"):
                result = self.model.predict_endpoint(float_audio)
            else:
                # fallback for older versions
                result = self.model._predict_endpoint(float_audio)  # type: ignore
        except Exception as e:
            print(f"❌ SmartTurn prediction error: {e}")
            self.reset()
            return None

        self.reset()
        return result

    def process_audio(self, pcm_chunk: bytes):
        if not pcm_chunk:
            return None

        # Safety: ensure proper int16 alignment
        if len(pcm_chunk) % 2 != 0:
            print("⚠️ Received odd-length PCM chunk, dropping")
            return None

        now = time.time() * 1000.0  # ms
        energy = self.calculate_rms(pcm_chunk)

        # --- ACTIVE SPEECH ---
        if energy > ENERGY_THRESHOLD:
            if not self.is_speaking:
                # Start of new speech segment
                self.is_speaking = True
                self.speech_start_ms = now
                self.audio_buffer = bytearray()

            self.last_speech_ms = now
            self.audio_buffer.extend(pcm_chunk)

            # HARD CAP: prevent unbounded growth
            if len(self.audio_buffer) >= MAX_AUDIO_BYTES:
                print("⚠️ Max audio buffer reached, finalizing turn early")
                return self._finalize_turn(reason="max_buffer")

            return None

        # --- SILENCE WHILE IN SPEECH ---
        if self.is_speaking and self.last_speech_ms is not None:
            silence_duration = now - self.last_speech_ms

            # Also cap by wall-clock speech duration, in case silence never fully drops
            current_speech_ms = self.last_speech_ms - self.speech_start_ms
            if current_speech_ms >= MAX_TURN_MS:
                print("⚠️ Max turn duration reached, finalizing turn")
                return self._finalize_turn(reason="max_duration")

            if silence_duration >= SILENCE_MS:
                # Consider the turn ended due to silence
                return self._finalize_turn(reason="silence")

        return None


async def handle_client(websocket: websockets.WebSocketServerProtocol):
    detector = SmartTurnDetector()
    print(f"👤 Client connected: {websocket.remote_address}")

    try:
        async for message in websocket:
            # Expecting raw PCM bytes
            if not isinstance(message, (bytes, bytearray)):
                print("⚠️ Received non-bytes message, skipping")
                continue

            result = detector.process_audio(message)

            if result:
                probability = float(result.get("probability", 0.0))
                response = {
                    "type": "turn_complete",
                    "probability": probability,
                }
                print(f"✓ Turn complete detected - Probability: {probability:.3f}")
                await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosedOK:
        print(f"📱 Client disconnected normally: {websocket.remote_address}")
    except websockets.exceptions.ConnectionClosedError:
        print(f"❌ Client disconnected abruptly: {websocket.remote_address}")
    except Exception as e:
        print(f"❌ Error handling client: {e}")
    finally:
        print(f"🔌 Connection closed: {websocket.remote_address}\n")


async def main():
    async with websockets.serve(handle_client, "0.0.0.0", 9001):
        print("\n" + "=" * 60)
        print("🚀 Hybrid SmartTurn + Silero VAD Server")
        print("📡 Running on ws://0.0.0.0:9001")
        print("=" * 60 + "\n")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
