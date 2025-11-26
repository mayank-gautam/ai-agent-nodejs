import asyncio
import websockets
import numpy as np
import json
import time

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

SAMPLE_RATE = 16000
SILENCE_MS = 700
ENERGY_THRESHOLD = 150
MIN_SPEECH_MS = 300

class SmartTurnDetector:
    def __init__(self):
        self.model = LocalSmartTurnAnalyzerV3(
            params=SmartTurnParams(
                stop_secs=3.0,
                pre_speech_ms=0,
                max_duration_secs=8.0
            ),
            cpu_count=1
        )
        self.reset()

    def reset(self):
        self.audio_buffer = bytearray()
        self.is_speaking = False
        self.last_speech_ms = None
        self.speech_start_ms = None

    @staticmethod
    def calculate_rms(pcm):
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        return np.sqrt(np.mean(arr * arr)) if len(arr) > 0 else 0

    def process_audio(self, pcm_chunk):
        now = time.time() * 1000
        energy = self.calculate_rms(pcm_chunk)

        if energy > ENERGY_THRESHOLD:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_ms = now
                self.audio_buffer = bytearray()
            
            self.last_speech_ms = now
            self.audio_buffer.extend(pcm_chunk)
            return None

        if self.is_speaking and (now - self.last_speech_ms) >= SILENCE_MS:
            duration = self.last_speech_ms - self.speech_start_ms

            if duration >= MIN_SPEECH_MS:
                float_audio = np.frombuffer(
                    self.audio_buffer, 
                    dtype=np.int16
                ).astype(np.float32) / 32768.0
                
                result = self.model._predict_endpoint(float_audio)
                self.reset()
                return result

            self.reset()

        return None

async def handle_client(websocket):
    detector = SmartTurnDetector()
    print(f"Client connected: {websocket.remote_address}")

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue

            result = detector.process_audio(message)
            
            if result:
                await websocket.send(json.dumps({
                    "type": "turn_complete",
                    "probability": result["probability"]
                }))

    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError:
        print(f"Client disconnected abruptly: {websocket.remote_address}")
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        print(f"Connection closed: {websocket.remote_address}")

async def main():
    async with websockets.serve(handle_client, "0.0.0.0", 9001):
        print("SmartTurn service running on ws://localhost:9001")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")