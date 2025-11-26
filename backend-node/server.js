import "dotenv/config";
import WebSocket, { WebSocketServer } from "ws";
import sdk from "microsoft-cognitiveservices-speech-sdk";
import OpenAI from "openai";
import http from "http";
import { spawn } from "child_process";

// -------------------- OpenAI --------------------
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function askLLM(text) {
  if (!text || !text.trim()) return "I did not hear anything clearly.";

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "You are a concise voice assistant." },
      { role: "user", content: text },
    ],
  });

  return completion.choices[0].message.content;
}

// -------------------- Azure STT --------------------
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

// -------------------- Simple VAD Implementation --------------------
class SimpleVAD {
  constructor(options = {}) {
    this.silenceThreshold = options.silenceThreshold || 700; // ms
    this.minSpeechDuration = options.minSpeechDuration || 300; // ms
    this.energyThreshold = options.energyThreshold || 500; // adjust based on your audio
    
    this.isSpeaking = false;
    this.speechStartTime = null;
    this.lastSpeechTime = null;
    this.silenceTimer = null;
    this.audioChunks = [];
  }

  // Calculate audio energy (simple RMS)
  calculateEnergy(buffer) {
    let sum = 0;
    const samples = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.length / 2);
    
    for (let i = 0; i < samples.length; i++) {
      sum += samples[i] * samples[i];
    }
    
    return Math.sqrt(sum / samples.length);
  }

  processAudio(buffer, onTurnStart, onTurnEnd) {
    const energy = this.calculateEnergy(buffer);
    const now = Date.now();
    
    // Store audio chunk
    this.audioChunks.push(buffer);

    if (energy > this.energyThreshold) {
      // Speech detected
      this.lastSpeechTime = now;

      if (!this.isSpeaking) {
        // Speech just started
        this.isSpeaking = true;
        this.speechStartTime = now;
        this.audioChunks = [buffer]; // Start fresh
        
        if (onTurnStart) onTurnStart();
      }

      // Clear any pending silence timer
      if (this.silenceTimer) {
        clearTimeout(this.silenceTimer);
        this.silenceTimer = null;
      }
    } else if (this.isSpeaking) {
      // Currently speaking but current chunk is silence
      
      // Start silence timer if not already started
      if (!this.silenceTimer) {
        this.silenceTimer = setTimeout(() => {
          // Check if speech was long enough
          const speechDuration = this.lastSpeechTime - this.speechStartTime;
          
          if (speechDuration >= this.minSpeechDuration) {
            // Valid speech turn ended
            this.isSpeaking = false;
            const audio = Buffer.concat(this.audioChunks);
            this.audioChunks = [];
            
            if (onTurnEnd) onTurnEnd(audio);
          } else {
            // Too short, discard
            this.isSpeaking = false;
            this.audioChunks = [];
          }
          
          this.silenceTimer = null;
        }, this.silenceThreshold);
      }
    }
  }

  reset() {
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }
    this.isSpeaking = false;
    this.audioChunks = [];
  }
}

// -------------------- WebSocket Server --------------------
const server = http.createServer();
const wss = new WebSocketServer({ server });
const PORT = 8080;

wss.on("connection", async (frontend) => {
  console.log("Frontend connected");

  const vad = new SimpleVAD({
    silenceThreshold: 700,    // 700ms of silence ends turn
    minSpeechDuration: 300,   // Minimum 300ms of speech
    energyThreshold: 500,     // Adjust based on your audio levels
  });

  let isProcessing = false;

  // ----- Turn start callback -----
  const onTurnStart = () => {
    console.log("🎤 Turn started - User speaking");
    frontend.send(JSON.stringify({ 
      type: "turn_state", 
      state: "speaking" 
    }));
  };

  // ----- Turn end callback -----
  const onTurnEnd = async (audioBuffer) => {
    console.log("✋ Turn ended");
    
    if (isProcessing) return; // Prevent overlapping processing
    isProcessing = true;

    frontend.send(JSON.stringify({ 
      type: "turn_state", 
      state: "processing" 
    }));

    try {
      // STT
      const transcript = await azureSTTFromPCM(audioBuffer);
      console.log("📄 Transcript:", transcript);
      
      frontend.send(JSON.stringify({ 
        type: "transcript", 
        text: transcript 
      }));

      // LLM
      if (transcript.trim()) {
        const reply = await askLLM(transcript);
        console.log("🤖 LLM Response:", reply);
        
        frontend.send(JSON.stringify({ 
          type: "llm_response", 
          text: reply 
        }));
      }
    } catch (error) {
      console.error("Error processing turn:", error);
      frontend.send(JSON.stringify({ 
        type: "error", 
        message: "Failed to process audio" 
      }));
    } finally {
      isProcessing = false;
      frontend.send(JSON.stringify({ 
        type: "turn_state", 
        state: "ready" 
      }));
    }
  };

  // ----- Audio from frontend -----
  frontend.on("message", (msg) => {
    if (msg instanceof Buffer) {
      // Process through VAD
      vad.processAudio(msg, onTurnStart, onTurnEnd);
    }
  });

  frontend.on("close", () => {
    console.log("Frontend disconnected");
    vad.reset();
  });
});

server.listen(PORT, () => {
  console.log(`🚀 Server running on ws://localhost:${PORT}`);
  console.log(`Using built-in VAD (no Pipecat required)`);
});