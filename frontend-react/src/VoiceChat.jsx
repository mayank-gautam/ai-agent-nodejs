import React, { useEffect, useRef, useState } from 'react';

const VoiceChat = () => {
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [log, setLog] = useState([]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8080');
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      setConnected(true);
      setLog((prev) => [...prev, '🟢 Connected to backend']);
    };

    ws.onclose = () => {
      setConnected(false);
      setLog((prev) => [...prev, '🔴 Disconnected']);
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      setLog((prev) => [...prev, '⚠️ WebSocket error']);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === 'transcript') {
          setTranscript(msg.text);
          setLog((prev) => [...prev, `🗣 Transcript: ${msg.text}`]);
        }

        if (msg.type === 'llm_response') {
          setResponse(msg.text);
          setLog((prev) => [...prev, `🤖 LLM: ${msg.text}`]);
        }

        if (msg.type === 'error') {
          setLog((prev) => [...prev, `❌ Error: ${msg.message}`]);
        }
      } catch (e) {
        console.error(e);
      }
    };

    wsRef.current = ws;

    return () => ws.close();
  }, []);

  // ---------------------------------------------------
  // START / STOP RECORDING
  // ---------------------------------------------------

  const startRecording = async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      alert("WebSocket not connected yet!");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 48000,
      });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const pcm16 = floatToPCM16(resampleTo16k(input, audioContext.sampleRate));

        if (wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(pcm16);
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      processorRef.current = processor;

      setRecording(true);
      setTranscript('');
      setResponse('');
      setLog(prev => [...prev, "🎙 Listening…"]);
    } catch (err) {
      console.error(err);
      setLog(prev => [...prev, "❌ Could not access microphone"]);
    }
  };

  const stopRecording = () => {
    setRecording(false);
    setLog(prev => [...prev, "⏹ Stopped"]);

    if (processorRef.current) processorRef.current.disconnect();
    if (audioContextRef.current) audioContextRef.current.close();
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());

    processorRef.current = null;
    audioContextRef.current = null;
    streamRef.current = null;

    // optional reset signal
    if (wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "reset" }));
    }
  };

  // ---------------------------------------------------
  // AUDIO HELPERS
  // ---------------------------------------------------

  function resampleTo16k(input, inputSR) {
    const targetSR = 16000;
    const ratio = inputSR / targetSR;
    const newLength = Math.round(input.length / ratio);
    const result = new Float32Array(newLength);

    let offset = 0;
    for (let i = 0; i < newLength; i++) {
      result[i] = input[Math.floor(offset)];
      offset += ratio;
    }
    return result;
  }

  function floatToPCM16(float32) {
    const buffer = new ArrayBuffer(float32.length * 2);
    const view = new DataView(buffer);
    let offset = 0;

    for (let i = 0; i < float32.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, float32[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return buffer;
  }

  // ---------------------------------------------------
  // UI
  // ---------------------------------------------------

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="w-full max-w-2xl bg-white/10 backdrop-blur-xl rounded-2xl p-8 shadow-lg border border-white/10">
        
        <h1 className="text-3xl font-bold mb-6 text-center text-white">
          🎧 AI Voice Assistant
        </h1>

        <p className="text-center mb-4 text-slate-300">
          WebSocket: {connected ? "🟢 Connected" : "🔴 Not connected"}
        </p>

        {/* Microphone Button */}
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

        {/* Transcript Section */}
        <div className="mb-4">
          <h2 className="text-xl font-semibold text-white mb-2">🗣 Transcript</h2>
          <div className="bg-white/10 border border-white/10 rounded-xl p-4 min-h-[70px] text-slate-200">
            {transcript || <span className="text-slate-500">Say something…</span>}
          </div>
        </div>

        {/* LLM Reply */}
        <div className="mb-4">
          <h2 className="text-xl font-semibold text-white mb-2">🤖 Assistant Reply</h2>
          <div className="bg-white/10 border border-white/10 rounded-xl p-4 min-h-[70px] text-slate-200">
            {response || <span className="text-slate-500">Waiting for response…</span>}
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
