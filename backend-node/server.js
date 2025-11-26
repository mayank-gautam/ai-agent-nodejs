import "dotenv/config";
import WebSocket, { WebSocketServer } from "ws";
import sdk from "microsoft-cognitiveservices-speech-sdk";
import OpenAI from "openai";
import http from "http";

/* -------------------- OpenAI -------------------- */
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function askLLM(text) {
  if (!text || !text.trim()) return "I didn't hear anything clearly.";

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "You are a concise voice assistant." },
      { role: "user", content: text },
    ],
  });

  return completion.choices[0].message.content;
}

/* -------------------- Azure STT -------------------- */
const speechConfig = sdk.SpeechConfig.fromSubscription(
  process.env.AZURE_SPEECH_KEY,
  process.env.AZURE_SPEECH_REGION
);
speechConfig.speechRecognitionLanguage = "en-US";

const pcmFormat = sdk.AudioStreamFormat.getWaveFormatPCM(16000, 16, 1);

function azureSTTFromPCM(buffer) {
  return new Promise((resolve) => {
    const pushStream = sdk.AudioInputStream.createPushStream(pcmFormat);
    pushStream.write(buffer);
    pushStream.close();

    const audioCfg = sdk.AudioConfig.fromStreamInput(pushStream);
    const recognizer = new sdk.SpeechRecognizer(speechConfig, audioCfg);

    recognizer.recognizeOnceAsync(
      (result) => {
        resolve(result.text || "");
        recognizer.close();
      },
      (err) => {
        console.error("Azure STT error:", err);
        recognizer.close();
        resolve("");
      }
    );
  });
}

/* -------------------- SmartTurn Python WebSocket Client -------------------- */
let smartTurnWS = null;
let sttBuffer = [];

function connectSmartTurn() {
  smartTurnWS = new WebSocket("ws://localhost:9001");

  smartTurnWS.on("open", () => {
    console.log("🧠 SmartTurn connected");
  });

  smartTurnWS.on("close", () => {
    console.log("❌ SmartTurn disconnected — retrying in 1s...");
    setTimeout(connectSmartTurn, 1000);
  });

  smartTurnWS.on("message", async (msg) => {
    const data = JSON.parse(msg.toString());

    if (data.type === "probability") {
      console.log("   SmartTurn probability:", data.probability.toFixed(3));
    }

    if (data.type === "turn_complete") {
      console.log("🎤 SmartTurn: TURN COMPLETE");

      const audio = Buffer.concat(sttBuffer);
      sttBuffer = [];

      await processTurn(audio);
    }
  });
}

connectSmartTurn();

/* -------------------- STT + LLM Processing -------------------- */
let frontendConn = null;

async function processTurn(audioBuffer) {
  if (!frontendConn) return;

  frontendConn.send(JSON.stringify({
    type: "turn_state",
    state: "processing"
  }));

  try {
    // Azure STT
    const transcript = await azureSTTFromPCM(audioBuffer);
    console.log("📄 Transcript:", transcript);

    frontendConn.send(JSON.stringify({
      type: "transcript",
      text: transcript
    }));

    // LLM
    if (transcript.trim()) {
      const reply = await askLLM(transcript);
      console.log("🤖 LLM:", reply);

      frontendConn.send(JSON.stringify({
        type: "llm_response",
        text: reply
      }));
    }
  } catch (err) {
    console.error("Turn processing error:", err);
    frontendConn.send(JSON.stringify({
      type: "error",
      message: "Turn processing failed."
    }));
  }

  frontendConn.send(JSON.stringify({
    type: "turn_state",
    state: "ready"
  }));
}

/* -------------------- Node WebSocket Server -------------------- */
const server = http.createServer();
const wss = new WebSocketServer({ server });
const PORT = 8080;

wss.on("connection", async (ws) => {
  console.log("🎧 Frontend connected");
  frontendConn = ws;

  ws.on("message", (msg) => {
    if (!(msg instanceof Buffer)) return;

    // Store this chunk for STT later
    sttBuffer.push(msg);

    // Forward to SmartTurn Python service
    if (smartTurnWS?.readyState === WebSocket.OPEN) {
      smartTurnWS.send(msg);
    }
  });

  ws.on("close", () => {
    console.log("Frontend disconnected");
    frontendConn = null;
    sttBuffer = [];
  });
});

server.listen(PORT, () => {
  console.log(`🚀 Node server: ws://localhost:${PORT}`);
  console.log("🎙️ Using Pipecat SmartTurn V3 for turn detection");
});
