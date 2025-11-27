import "dotenv/config";
import WebSocket, { WebSocketServer } from "ws";
import sdk from "microsoft-cognitiveservices-speech-sdk";
import OpenAI from "openai";
import http from "http";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const speechConfig = sdk.SpeechConfig.fromSubscription(
  process.env.AZURE_SPEECH_KEY,
  process.env.AZURE_SPEECH_REGION
);
speechConfig.speechRecognitionLanguage = "en-US";

const pcmFormat = sdk.AudioStreamFormat.getWaveFormatPCM(16000, 16, 1);

const PORT = 8080;
const SMARTTURN_URL = "ws://localhost:9001";

let smartTurnWS = null;
let frontendConn = null;

let sttPushStream = null;
let sttRecognizer = null;

let lastFinalTranscript = "";
let isProcessingLLM = false;

async function getLLMResponse(text) {
  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "You are a concise voice assistant." },
        { role: "user", content: text },
      ],
    });
    return completion.choices?.[0]?.message?.content?.trim() || null;
  } catch (err) {
    console.error("LLM:", err.message);
    return null;
  }
}

function connectSmartTurn() {
  if (smartTurnWS) {
    try {
      smartTurnWS.close();
    } catch {}
  }

  smartTurnWS = new WebSocket(SMARTTURN_URL);

  smartTurnWS.on("close", () => {
    setTimeout(connectSmartTurn, 1000);
  });

  smartTurnWS.on("error", (e) => console.error("SmartTurn:", e.message));

  smartTurnWS.on("message", async (msg) => {
    try {
      const data = JSON.parse(msg.toString("utf8"));

      // 🔥 UPDATED: Using completed only (no probability)
      console.log("Data", data)
      if (data.type === "turn_complete") {
        if (data.completed === 1) {
          console.log("🟢 SmartTurn: USER FINISHED SPEAKING");
          console.log("User said:", lastFinalTranscript || "(no transcript)");
          console.log("--------------------------------------");

          if (frontendConn) {
            frontendConn.send(
              JSON.stringify({
                type: "turn_complete",
                completed: 1,
              })
            );
          }
        } else {
          console.log("🟡 SmartTurn: user still speaking...");
        }
      }
    } catch (err) {
      console.error("SmartTurn msg:", err.message);
    }
  });
}

function createStreamingRecognizer() {
  sttPushStream = sdk.AudioInputStream.createPushStream(pcmFormat);
  const audioConfig = sdk.AudioConfig.fromStreamInput(sttPushStream);
  sttRecognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);

  lastFinalTranscript = "";

  sttRecognizer.recognizing = (_, e) => {
    if (e.result.text) {
      sendToFrontend({ type: "stt_partial", text: e.result.text });
    }
  };

  sttRecognizer.recognized = async (_, e) => {
    if (e.result.reason === sdk.ResultReason.RecognizedSpeech) {
      lastFinalTranscript = e.result.text;
      sendToFrontend({ type: "stt_final", text: e.result.text });

      console.log("Final:", e.result.text);

      // Auto-process LLM as before
      if (!isProcessingLLM && lastFinalTranscript.trim()) {
        isProcessingLLM = true;
        sendToFrontend({ type: "turn_state", state: "processing" });

        const reply = await getLLMResponse(lastFinalTranscript);

        if (reply) {
          sendToFrontend({ type: "llm_response", text: reply });
        } else {
          sendToFrontend({
            type: "error",
            message: "Failed to generate response.",
          });
        }

        isProcessingLLM = false;
        sendToFrontend({ type: "turn_state", state: "ready" });
      }
    }
  };

  sttRecognizer.canceled = (_, e) =>
    console.error("STT canceled:", e.errorDetails || e.reason);

  sttRecognizer.startContinuousRecognitionAsync(
    () => {},
    (err) => console.error("STT:", err)
  );
}

function stopStreamingRecognizer() {
  if (sttRecognizer) {
    sttRecognizer.stopContinuousRecognitionAsync(
      () => {
        sttRecognizer.close();
        sttRecognizer = null;
      },
      () => {
        sttRecognizer.close();
        sttRecognizer = null;
      }
    );
  }

  if (sttPushStream) {
    try {
      sttPushStream.close();
    } catch {}
    sttPushStream = null;
  }
}

function sendToFrontend(data) {
  if (frontendConn?.readyState === WebSocket.OPEN) {
    frontendConn.send(JSON.stringify(data));
  }
}

const server = http.createServer();
const wss = new WebSocketServer({ server });

wss.on("connection", (ws) => {
  frontendConn = ws;

  if (!sttRecognizer) createStreamingRecognizer();

  ws.on("message", (msg) => {
    if (typeof msg === "string") {
      try {
        const data = JSON.parse(msg);
        if (data.type === "reset") {
          lastFinalTranscript = "";
          sendToFrontend({ type: "reset_ack" });
        }
      } catch {}
      return;
    }

    if (!Buffer.isBuffer(msg) || msg.length % 2 !== 0) return;

    if (sttPushStream) sttPushStream.write(msg);

    if (smartTurnWS?.readyState === WebSocket.OPEN) smartTurnWS.send(msg);
  });

  ws.on("close", () => {
    if (frontendConn === ws) frontendConn = null;

    setTimeout(() => {
      if (!frontendConn) stopStreamingRecognizer();
    }, 5000);
  });

  ws.on("error", (e) => console.error("Frontend WS:", e.message));
});

server.listen(PORT, () => connectSmartTurn());

process.on("SIGINT", () => {
  try {
    smartTurnWS?.close();
  } catch {}
  wss.close();
  stopStreamingRecognizer();
  server.close(() => process.exit(0));
});
