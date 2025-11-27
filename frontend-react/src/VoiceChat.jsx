import React, { useEffect, useRef, useState } from "react";

const VoiceChat = () => {
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [recording, setRecording] = useState(false);
  const [partialTranscript, setPartialTranscript] = useState("");
  const [finalTranscript, setFinalTranscript] = useState("");
  const [response, setResponse] = useState("");
  const [log, setLog] = useState([]);

  // -----------------------------------------------------------
  // WebSocket AUTO-CONNECT + SAFE RECONNECT
  // -----------------------------------------------------------
  const connectWS = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket("ws://localhost:8080");
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setConnected(true);
      setLog((p) => [...p, "🟢 Connected to backend"]);
    };

    ws.onclose = () => {
      setConnected(false);
      setLog((p) => [...p, "🔴 Disconnected from backend"]);

      // Retry connection after 1 sec
      reconnectTimerRef.current = setTimeout(connectWS, 1000);
    };

    ws.onerror = (err) => {
      setLog((p) => [...p, "⚠️ WebSocket error"]);
      console.error("WS error:", err);
      ws.close();
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        switch (msg.type) {
          case "stt_partial":
            setPartialTranscript(msg.text);
            setLog((p) => [...p, `📝 Partial: ${msg.text}`]);
            break;

          case "stt_final":
            setFinalTranscript(msg.text);
            setLog((p) => [...p, `🗣 Final: ${msg.text}`]);
            break;

          case "llm_response":
            setResponse(msg.text);
            setLog((p) => [...p, `🤖 LLM: ${msg.text}`]);
            break;

          case "turn_complete":
            setLog((p) => [
              ...p,
              `🔊 Speech complete (p=${msg.probability.toFixed(2)})`,
            ]);
            break;

          case "error":
            setLog((p) => [...p, `❌ Error: ${msg.message}`]);
            break;

          default:
            break;
        }
      } catch (e) {
        console.error("Failed parsing WS message:", e);
      }
    };

    wsRef.current = ws;
  };

  useEffect(() => {
    connectWS();
    return () => {
      clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, []);

  // -----------------------------------------------------------
  // START RECORDING
  // -----------------------------------------------------------
  const startRecording = async () => {
    if (!connected) {
      alert("Not connected to backend!");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: 48000 });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(2048, 1, 1);

      processor.onaudioprocess = (e) => {
        const float32 = e.inputBuffer.getChannelData(0);
        const resampled = resampleTo16k(float32, audioContext.sampleRate);
        const pcm16 = floatToPCM16(resampled);

        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(new Uint8Array(pcm16.buffer));
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
      processorRef.current = processor;

      setRecording(true);
      setPartialTranscript("");
      setFinalTranscript("");
      setResponse("");
      setLog((p) => [...p, "🎙 Listening…"]);
    } catch (err) {
      console.error(err);
      setLog((p) => [...p, "❌ Could not access microphone"]);
    }
  };

  // -----------------------------------------------------------
  // STOP RECORDING
  // -----------------------------------------------------------
  const stopRecording = () => {
    setRecording(false);
    setLog((p) => [...p, "⏹ Stopped"]);

    processorRef.current?.disconnect();
    audioContextRef.current?.close();

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
    }

    processorRef.current = null;
    audioContextRef.current = null;
    streamRef.current = null;

    wsRef.current?.send(JSON.stringify({ type: "reset" }));
  };

  // -----------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------
  function resampleTo16k(float32Audio, sourceRate) {
    const targetRate = 16000;
    if (sourceRate === targetRate) return float32Audio;

    const ratio = sourceRate / targetRate;
    const newLength = Math.floor(float32Audio.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const idx = i * ratio;
      const left = Math.floor(idx);
      const right = Math.min(left + 1, float32Audio.length - 1);
      const frac = idx - left;
      result[i] = float32Audio[left] * (1 - frac) + float32Audio[right] * frac;
    }
    return result;
  }

  function floatToPCM16(float32) {
    const pcm = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return pcm;
  }

  // -----------------------------------------------------------
  // UI
  // -----------------------------------------------------------
  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="w-full max-w-2xl bg-white/10 backdrop-blur-xl rounded-2xl p-8 shadow-lg border border-white/10">
        <h1 className="text-3xl font-bold mb-6 text-center text-white">
          🎧 AI Voice Assistant
        </h1>

        <p className="text-center mb-4 text-slate-300">
          WebSocket: {connected ? "🟢 Connected" : "🔴 Not connected"}
        </p>

        {/* Mic Button */}
        <div className="flex justify-center mb-6">
          {!recording ? (
            <button
              onClick={startRecording}
              disabled={!connected}
              className="px-6 py-3 text-lg font-semibold bg-emerald-500 hover:bg-emerald-600 text-white rounded-full shadow-lg transition-all"
            >
              🎙 Start Talking
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="px-6 py-3 text-lg font-semibold bg-rose-500 hover:bg-rose-600 text-white rounded-full shadow-lg"
            >
              ⏹ Stop
            </button>
          )}
        </div>

        {/* Transcript */}
        <div className="mb-4">
          <h2 className="text-xl font-semibold text-white mb-2">🗣 Transcript</h2>
          <div className="bg-white/10 border border-white/10 rounded-xl p-4 min-h-[70px] text-slate-200">
            {finalTranscript || partialTranscript || (
              <span className="text-slate-500">Say something…</span>
            )}
          </div>
        </div>

        {/* LLM Reply */}
        <div className="mb-4">
          <h2 className="text-xl font-semibold text-white mb-2">🤖 Assistant Reply</h2>
          <div className="bg-white/10 border border-white/10 rounded-xl p-4 min-h-[70px] text-slate-200">
            {response || <span className="text-slate-500">Waiting…</span>}
          </div>
        </div>

        {/* Logs */}
        <div>
          <h2 className="text-xl font-semibold text-white mb-2">📜 Logs</h2>
          <div className="h-32 overflow-auto bg-black/30 border border-white/10 rounded-xl p-2 text-sm text-slate-300">
            {log.map((item, idx) => (
              <div key={idx}>{item}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VoiceChat;
