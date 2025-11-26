import "dotenv/config";
import WebSocket, { WebSocketServer } from "ws";
import sdk from "microsoft-cognitiveservices-speech-sdk";
import OpenAI from "openai";
import http from "http";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const speechConfig = sdk.SpeechConfig.fromSubscription(
  process.env.AZURE_SPEECH_KEY,
  process.env.AZURE_SPEECH_REGION
);
speechConfig.speechRecognitionLanguage = "en-US";

const pcmFormat = sdk.AudioStreamFormat.getWaveFormatPCM(16000, 16, 1);
const PORT = 8080;
const SMARTTURN_URL = "ws://localhost:9001";
const RECONNECT_DELAY = 1000;

let smartTurnWS = null;
let frontendConn = null;
let sttBuffer = [];

async function getLLMResponse(text) {
  if (!text?.trim()) return null;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "You are a concise voice assistant." },
        { role: "user", content: text },
      ],
    });
    return completion.choices[0].message.content;
  } catch (error) {
    console.error("LLM error:", error.message);
    return null;
  }
}

async function transcribeAudio(buffer) {
  return new Promise((resolve) => {
    const pushStream = sdk.AudioInputStream.createPushStream(pcmFormat);
    pushStream.write(buffer);
    pushStream.close();

    const audioConfig = sdk.AudioConfig.fromStreamInput(pushStream);
    const recognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);

    recognizer.recognizeOnceAsync(
      (result) => {
        recognizer.close();
        resolve(result.text || "");
      },
      (error) => {
        console.error("STT error:", error.message);
        recognizer.close();
        resolve("");
      }
    );
  });
}

function connectSmartTurn() {
  smartTurnWS = new WebSocket(SMARTTURN_URL);

  smartTurnWS.on("open", () => {
    console.log("SmartTurn connected");
  });

  smartTurnWS.on("close", () => {
    console.log("SmartTurn disconnected, reconnecting...");
    setTimeout(connectSmartTurn, RECONNECT_DELAY);
  });

  smartTurnWS.on("error", (error) => {
    console.error("SmartTurn error:", error.message);
  });

  smartTurnWS.on("message", async (msg) => {
    try {
      const data = JSON.parse(msg.toString());

      if (data.type === "turn_complete") {
        console.log(`Turn detected - Probability: ${data.probability.toFixed(3)}`);
        const audio = Buffer.concat(sttBuffer);
        sttBuffer = [];
        await processTurn(audio);
      }
    } catch (error) {
      console.error("SmartTurn message error:", error.message);
    }
  });
}

async function processTurn(audioBuffer) {
  if (!frontendConn || frontendConn.readyState !== WebSocket.OPEN) return;

  sendToFrontend({ type: "turn_state", state: "processing" });

  try {
    const transcript = await transcribeAudio(audioBuffer);
    
    if (transcript) {
      sendToFrontend({ type: "transcript", text: transcript });
      
      const reply = await getLLMResponse(transcript);
      if (reply) {
        sendToFrontend({ type: "llm_response", text: reply });
      }
    }
  } catch (error) {
    console.error("Turn processing error:", error.message);
    sendToFrontend({ 
      type: "error", 
      message: "Processing failed. Please try again." 
    });
  }

  sendToFrontend({ type: "turn_state", state: "ready" });
}

function sendToFrontend(data) {
  if (frontendConn?.readyState === WebSocket.OPEN) {
    frontendConn.send(JSON.stringify(data));
  }
}

const server = http.createServer();
const wss = new WebSocketServer({ server });

wss.on("connection", (ws) => {
  console.log("Frontend connected");
  frontendConn = ws;

  ws.on("message", (msg) => {
    if (!(msg instanceof Buffer)) return;

    sttBuffer.push(msg);

    if (smartTurnWS?.readyState === WebSocket.OPEN) {
      smartTurnWS.send(msg);
    }
  });

  ws.on("close", () => {
    console.log("Frontend disconnected");
    frontendConn = null;
    sttBuffer = [];
  });

  ws.on("error", (error) => {
    console.error("Frontend WS error:", error.message);
  });
});

server.listen(PORT, () => {
  console.log(`Server running on ws://localhost:${PORT}`);
  connectSmartTurn();
});

process.on("SIGINT", () => {
  console.log("\nShutting down gracefully...");
  smartTurnWS?.close();
  wss.close();
  server.close();
  process.exit(0);
});