import "dotenv/config";
import WebSocket, { WebSocketServer } from "ws";
import sdk from "microsoft-cognitiveservices-speech-sdk";
import OpenAI from "openai";
import http from "http";

// ---------------- OpenAI ----------------
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ---------------- Azure STT Config ----------------
const speechConfig = sdk.SpeechConfig.fromSubscription(
  process.env.AZURE_SPEECH_KEY,
  process.env.AZURE_SPEECH_REGION
);
speechConfig.speechRecognitionLanguage = "en-US";

// 16 kHz, 16-bit, mono PCM
const pcmFormat = sdk.AudioStreamFormat.getWaveFormatPCM(16000, 16, 1);

// ---------------- Server constants ----------------
const PORT = 8080;
const SMARTTURN_URL = "ws://localhost:9001";
const RECONNECT_DELAY = 1000;
const MAX_STT_CHUNKS = 200; // ≈ few seconds of audio

// ---------------- State ----------------
let smartTurnWS = null;
let frontendConn = null;
let sttBuffer = [];
let isProcessing = false;

// ---------------- LLM ----------------
async function getLLMResponse(text) {
  const content = text && text.trim();
  if (!content) return null;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "You are a concise voice assistant." },
        { role: "user", content },
      ],
    });

    const reply = completion.choices?.[0]?.message?.content ?? "";
    return reply.trim() || null;
  } catch (error) {
    console.error("LLM error:", error?.message || error);
    return null;
  }
}

// ---------------- STT ----------------
async function transcribeAudio(buffer) {
  return new Promise((resolve) => {
    try {
      const pushStream = sdk.AudioInputStream.createPushStream(pcmFormat);
      pushStream.write(buffer);
      pushStream.close();

      const audioConfig = sdk.AudioConfig.fromStreamInput(pushStream);
      const recognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);

      recognizer.recognizeOnceAsync(
        (result) => {
          if (!result) {
            console.warn("STT: empty result object");
            recognizer.close();
            return resolve("");
          }

          console.log("STT reason:", result.reason);

          if (result.reason === sdk.ResultReason.RecognizedSpeech) {
            console.log("STT text:", result.text);
            recognizer.close();
            return resolve(result.text || "");
          } else if (result.reason === sdk.ResultReason.NoMatch) {
            console.warn("STT NoMatch:", sdk.NoMatchDetails.fromResult(result).reason);
          } else if (result.reason === sdk.ResultReason.Canceled) {
            const details = sdk.CancellationDetails.fromResult(result);
            console.error("STT Canceled:", details.reason, details.errorDetails);
          }

          recognizer.close();
          resolve("");
        },
        (error) => {
          console.error("STT error:", error?.message || error);
          recognizer.close();
          resolve("");
        }
      );
    } catch (error) {
      console.error("Transcription setup error:", error?.message || error);
      resolve("");
    }
  });
}

// ---------------- SmartTurn WS ----------------
function connectSmartTurn() {
  if (smartTurnWS) {
    try {
      smartTurnWS.close();
    } catch {
      // ignore
    }
  }

  smartTurnWS = new WebSocket(SMARTTURN_URL);

  smartTurnWS.on("open", () => {
    console.log("✓ SmartTurn connected");
  });

  smartTurnWS.on("close", () => {
    console.log("✗ SmartTurn disconnected, reconnecting...");
    setTimeout(connectSmartTurn, RECONNECT_DELAY);
  });

  smartTurnWS.on("error", (error) => {
    console.error("SmartTurn error:", error?.message || error);
  });

  smartTurnWS.on("message", async (msg) => {
    try {
      const str = msg.toString("utf8");
      const data = JSON.parse(str);

      if (data.type === "turn_complete" && data.probability > 0.5) {
        if (isProcessing) {
          console.log("⏳ Already processing, ignoring detected turn");
          return;
        }

        if (sttBuffer.length === 0) {
          console.log("⚠ Empty audio buffer at turn, skipping");
          return;
        }

        console.log(
          `📍 Turn detected - Probability: ${data.probability.toFixed(
            3
          )}, chunks: ${sttBuffer.length}`
        );

        const audio = Buffer.concat(sttBuffer);
        console.log("STT audio size (bytes):", audio.length);
        sttBuffer = [];

        await processTurn(audio);
      }
    } catch (error) {
      console.error("SmartTurn message error:", error?.message || error);
    }
  });
}

// ---------------- Turn Processing ----------------
async function processTurn(audioBuffer) {
  if (!audioBuffer || audioBuffer.length === 0) {
    console.log("⚠ Empty audio buffer, skipping processing");
    return;
  }

  if (!frontendConn || frontendConn.readyState !== WebSocket.OPEN) {
    console.log("⚠ Frontend not connected, dropping turn");
    return;
  }

  isProcessing = true;
  sendToFrontend({ type: "turn_state", state: "processing" });

  try {
    console.log(`🔄 Transcribing audio (${audioBuffer.length} bytes)...`);
    const transcript = (await transcribeAudio(audioBuffer)).trim();

    if (transcript) {
      console.log(`👤 [USER] ${transcript}`);
      sendToFrontend({ type: "transcript", text: transcript });

      console.log("🤖 Generating response...");
      const reply = await getLLMResponse(transcript);

      if (reply) {
        console.log(`🤖 [ASSISTANT] ${reply}`);
        sendToFrontend({ type: "llm_response", text: reply });
      } else {
        console.log("⚠ Failed to generate LLM response");
        sendToFrontend({
          type: "error",
          message: "Failed to generate response. Please try again.",
        });
      }
    } else {
      console.log("⚠ Empty transcript received from STT");
      sendToFrontend({
        type: "error",
        message: "Could not transcribe audio. Please try again.",
      });
    }
  } catch (error) {
    console.error("❌ Turn processing error:", error?.message || error);
    sendToFrontend({
      type: "error",
      message: "Processing failed. Please try again.",
    });
  } finally {
    isProcessing = false;
    console.log("✓ Ready for next turn\n");
    sendToFrontend({ type: "turn_state", state: "ready" });
  }
}

// ---------------- Frontend WS ----------------
function sendToFrontend(data) {
  if (frontendConn?.readyState === WebSocket.OPEN) {
    frontendConn.send(JSON.stringify(data));
  }
}

const server = http.createServer();
const wss = new WebSocketServer({ server });

wss.on("connection", (ws) => {
  console.log("📱 Frontend connected");
  frontendConn = ws;
  sttBuffer = [];

  ws.on("message", (msg) => {
    // Text messages (e.g., reset)
    if (typeof msg === "string") {
      try {
        const data = JSON.parse(msg);
        if (data.type === "reset") {
          console.log("🔄 Reset received from frontend, clearing STT buffer");
          sttBuffer = [];
        }
      } catch {
        console.warn("⚠ Received text message but could not parse JSON");
      }
      return;
    }

    // Binary PCM
    if (!Buffer.isBuffer(msg)) {
      console.warn("⚠ Received non-buffer message from frontend");
      return;
    }

    if (msg.length % 2 !== 0) {
      console.warn("⚠ Received odd-length PCM buffer, dropping chunk");
      return;
    }

    if (sttBuffer.length > MAX_STT_CHUNKS) {
      console.warn("⚠ STT buffer too large, resetting");
      sttBuffer = [];
    }

    sttBuffer.push(msg);

    if (smartTurnWS?.readyState === WebSocket.OPEN) {
      smartTurnWS.send(msg);
    }
  });

  ws.on("close", () => {
    console.log("📱 Frontend disconnected");
    if (frontendConn === ws) {
      frontendConn = null;
    }
    sttBuffer = [];
  });

  ws.on("error", (error) => {
    console.error("❌ Frontend WS error:", error?.message || error);
  });
});

server.listen(PORT, () => {
  console.log(`\n🚀 Server running on ws://localhost:${PORT}\n`);
  connectSmartTurn();
});

// Graceful shutdown
process.on("SIGINT", () => {
  console.log("\n🛑 Shutting down gracefully...");
  try {
    smartTurnWS?.close();
  } catch {
    // ignore
  }
  wss.close();
  server.close(() => {
    process.exit(0);
  });
});
