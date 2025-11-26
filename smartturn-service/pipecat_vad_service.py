import asyncio
import websockets
import numpy as np
import json
import time

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

SAMPLE_RATE = 16000
SILENCE_MS = 700            # silence required to end turn
ENERGY_THRESHOLD = 150      # RMS threshold for speech
MIN_SPEECH_MS = 300

FRAME_MS = 20               # your chunk size

class SmartTurnRealtime:
    def __init__(self):
        self.model = LocalSmartTurnAnalyzerV3(
            params=SmartTurnParams(
                stop_secs=3.0,
                pre_speech_ms=0,
                max_duration_secs=8.0
            ),
            cpu_count=1
        )

        self.audio_buffer = bytearray()
        self.is_speaking = False
        self.last_speech_ms = None
        self.speech_start_ms = None

    def rms(self, pcm):
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if len(arr) == 0: return 0
        return np.sqrt(np.mean(arr * arr))

    def feed(self, pcm_chunk):
        now = time.time() * 1000
        energy = self.rms(pcm_chunk)

        if energy > ENERGY_THRESHOLD:
            # Speech detected
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_ms = now
                self.audio_buffer = bytearray()
            self.last_speech_ms = now

            self.audio_buffer.extend(pcm_chunk)
            return None  # no turn end yet

        # Silence detected while speaking
        if self.is_speaking and (now - self.last_speech_ms) >= SILENCE_MS:
            duration = self.last_speech_ms - self.speech_start_ms

            if duration >= MIN_SPEECH_MS:
                # VALID TURN — run SmartTurn once
                float_audio = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                result = self.model._predict_endpoint(float_audio)

                self.is_speaking = False
                self.audio_buffer = bytearray()
                return result  # return SmartTurn result

            # too short to count
            self.is_speaking = False
            self.audio_buffer = bytearray()

        return None
                

async def main():
    vad = SmartTurnRealtime()

    async def handler(ws):
        print("🔗 Connected")
        try:
            async for msg in ws:
                if not isinstance(msg, bytes):
                    continue

                result = vad.feed(msg)
                if result:
                    await ws.send(json.dumps({
                        "type": "turn_complete",
                        "probability": result["probability"]
                    }))

        except websockets.exceptions.ConnectionClosedOK:
            print("🔌 Client closed normally")

        except websockets.exceptions.ConnectionClosedError:
            print("⚠️ Client disconnected abruptly")

        except Exception as e:
            print("❌ Unexpected:", e)

        finally:
            print("🔚 Connection ended")

    server = await websockets.serve(handler, "0.0.0.0", 9001)
    print("🚀 SmartTurn Real-Time WS running on ws://localhost:9001")

    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
