import Fastify from 'fastify';
import websocket from '@fastify/websocket';
import twilio from 'twilio';
import sdk from 'microsoft-cognitiveservices-speech-sdk';
import OpenAI from 'openai';
import { Buffer } from 'buffer';
import dotenv from "dotenv";
dotenv.config();
// ==================== CONFIGURATION ====================
const config = {
  port: process.env.PORT || 3000,
  host: '0.0.0.0',
  
  // Twilio
  twilioAccountSid: process.env.TWILIO_ACCOUNT_SID || 'YOUR_TWILIO_ACCOUNT_SID',
  twilioAuthToken: process.env.TWILIO_AUTH_TOKEN || 'YOUR_TWILIO_AUTH_TOKEN',
  twilioPhoneNumber: process.env.TWILIO_PHONE_NUMBER || '+1234567890',
  
  // Azure Speech
  azureSpeechKey: process.env.AZURE_SPEECH_KEY || 'YOUR_AZURE_SPEECH_KEY',
  azureSpeechRegion: process.env.AZURE_SPEECH_REGION || 'eastus',
  azureTtsVoice: process.env.AZURE_TTS_VOICE || 'en-US-JennyNeural',
  
  // OpenAI
  openaiApiKey: process.env.OPENAI_API_KEY || 'YOUR_OPENAI_API_KEY',
  openaiModel: process.env.OPENAI_MODEL || 'gpt-4o-mini',
  
  // Server
  serverUrl: process.env.SERVER_URL || 'wss://your-server.com',
  
  // System prompt for AI assistant
  systemPrompt: `You are a helpful AI voice assistant. Keep responses concise and conversational, 
suitable for phone conversations. Respond naturally as if speaking to someone on the phone.`
};

// Initialize clients
const twilioClient = twilio(config.twilioAccountSid, config.twilioAuthToken);
const openai = new OpenAI({ apiKey: config.openaiApiKey });

// ==================== AUDIO CONVERSION FUNCTIONS ====================

/**
 * Decode base64 µ-law to PCM16 Int16Array
 * @param {string} base64 - Base64 encoded µ-law data
 * @returns {Int16Array} - PCM16 audio data
 */
function decodeUlaw(base64) {
  const ulawBuffer = Buffer.from(base64, 'base64');
  const pcmInt16Array = new Int16Array(ulawBuffer.length);
  
  for (let i = 0; i < ulawBuffer.length; i++) {
    pcmInt16Array[i] = alawmulaw.mulaw.decode(ulawBuffer[i]);
  }
  
  return pcmInt16Array;
}

/**
 * Encode PCM16 Int16Array to base64 µ-law
 * @param {Int16Array} int16PcmArray - PCM16 audio data
 * @returns {string} - Base64 encoded µ-law data
 */
function encodeUlaw(int16PcmArray) {
  const ulawBuffer = Buffer.alloc(int16PcmArray.length);
  
  for (let i = 0; i < int16PcmArray.length; i++) {
    ulawBuffer[i] = alawmulaw.mulaw.encode(int16PcmArray[i]);
  }
  
  return ulawBuffer.toString('base64');
}

/**
 * Convert PCM16 Buffer to Int16Array
 * @param {Buffer} buffer - PCM16 buffer
 * @returns {Int16Array}
 */
function bufferToInt16Array(buffer) {
  return new Int16Array(buffer.buffer, buffer.byteOffset, buffer.length / 2);
}

// ==================== AZURE SPEECH SERVICES ====================

/**
 * Create Azure Speech-to-Text recognizer with push stream
 * @returns {Object} - { recognizer, pushStream }
 */
function createSpeechRecognizer() {
  const speechConfig = sdk.SpeechConfig.fromSubscription(
    config.azureSpeechKey,
    config.azureSpeechRegion
  );
  speechConfig.speechRecognitionLanguage = 'en-US';
  
  const pushStream = sdk.AudioInputStream.createPushStream(
    sdk.AudioStreamFormat.getWaveFormatPCM(8000, 16, 1)
  );
  
  const audioConfig = sdk.AudioConfig.fromStreamInput(pushStream);
  const recognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);
  
  return { recognizer, pushStream };
}

/**
 * Synthesize speech using Azure TTS
 * @param {string} text - Text to synthesize
 * @returns {Promise<Buffer>} - PCM16 audio buffer
 */
async function synthesizeSpeech(text) {
  return new Promise((resolve, reject) => {
    const speechConfig = sdk.SpeechConfig.fromSubscription(
      config.azureSpeechKey,
      config.azureSpeechRegion
    );
    
    speechConfig.speechSynthesisVoiceName = config.azureTtsVoice;
    speechConfig.speechSynthesisOutputFormat = sdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm;
    
    const synthesizer = new sdk.SpeechSynthesizer(speechConfig, null);
    
    synthesizer.speakTextAsync(
      text,
      result => {
        if (result.reason === sdk.ResultReason.SynthesizingAudioCompleted) {
          const audioData = Buffer.from(result.audioData);
          synthesizer.close();
          resolve(audioData);
        } else {
          synthesizer.close();
          reject(new Error(`Speech synthesis failed: ${result.errorDetails}`));
        }
      },
      error => {
        synthesizer.close();
        reject(error);
      }
    );
  });
}

// ==================== OPENAI INTEGRATION ====================

/**
 * Get AI response from OpenAI
 * @param {string} userMessage - User's message
 * @param {Array} conversationHistory - Previous messages
 * @returns {Promise<string>} - AI response
 */
async function getAIResponse(userMessage, conversationHistory = []) {
  const messages = [
    { role: 'system', content: config.systemPrompt },
    ...conversationHistory,
    { role: 'user', content: userMessage }
  ];
  
  const completion = await openai.chat.completions.create({
    model: config.openaiModel,
    messages: messages,
    temperature: 0.7,
    max_tokens: 150
  });
  
  return completion.choices[0].message.content;
}

// ==================== CALL SESSION MANAGEMENT ====================

class CallSession {
  constructor(callId, connection) {
    this.callId = callId;
    this.connection = connection;
    this.streamSid = null;
    this.conversationHistory = [];
    this.isProcessing = false;
    this.audioBuffer = [];
    
    // Azure STT setup
    const { recognizer, pushStream } = createSpeechRecognizer();
    this.recognizer = recognizer;
    this.pushStream = pushStream;
    
    this.setupRecognizer();
  }
  
  setupRecognizer() {
    // Recognized event - final transcription
    this.recognizer.recognized = async (s, e) => {
      if (e.result.reason === sdk.ResultReason.RecognizedSpeech && e.result.text) {
        console.log(`[${this.callId}] User said: ${e.result.text}`);
        await this.handleUserSpeech(e.result.text);
      }
    };
    
    // Recognizing event - interim results (optional)
    this.recognizer.recognizing = (s, e) => {
      if (e.result.text) {
        console.log(`[${this.callId}] Recognizing: ${e.result.text}`);
      }
    };
    
    // Error handling
    this.recognizer.canceled = (s, e) => {
      console.error(`[${this.callId}] Recognition canceled: ${e.errorDetails}`);
    };
    
    // Start continuous recognition
    this.recognizer.startContinuousRecognitionAsync(
      () => console.log(`[${this.callId}] Speech recognition started`),
      err => console.error(`[${this.callId}] Failed to start recognition:`, err)
    );
  }
  
  async handleUserSpeech(text) {
    if (this.isProcessing) {
      console.log(`[${this.callId}] Already processing, skipping...`);
      return;
    }
    
    this.isProcessing = true;
    
    try {
      // Get AI response
      const aiResponse = await getAIResponse(text, this.conversationHistory);
      console.log(`[${this.callId}] AI response: ${aiResponse}`);
      
      // Update conversation history
      this.conversationHistory.push(
        { role: 'user', content: text },
        { role: 'assistant', content: aiResponse }
      );
      
      // Keep only last 10 messages
      if (this.conversationHistory.length > 10) {
        this.conversationHistory = this.conversationHistory.slice(-10);
      }
      
      // Synthesize speech
      const pcmAudio = await synthesizeSpeech(aiResponse);
      
      // Convert PCM to µ-law and send to Twilio
      await this.sendAudioToTwilio(pcmAudio);
      
    } catch (error) {
      console.error(`[${this.callId}] Error processing speech:`, error);
    } finally {
      this.isProcessing = false;
    }
  }
  
  async sendAudioToTwilio(pcmBuffer) {
    if (!this.streamSid) {
      console.error(`[${this.callId}] No streamSid available`);
      return;
    }
    
    // Convert PCM buffer to Int16Array
    const pcmInt16 = bufferToInt16Array(pcmBuffer);
    
    // Convert to µ-law and encode as base64
    const ulawBase64 = encodeUlaw(pcmInt16);
    
    // Split into chunks (Twilio expects ~20ms chunks, 160 bytes for 8kHz µ-law)
    const chunkSize = 160;
    const ulawBuffer = Buffer.from(ulawBase64, 'base64');
    
    for (let i = 0; i < ulawBuffer.length; i += chunkSize) {
      const chunk = ulawBuffer.slice(i, i + chunkSize);
      const chunkBase64 = chunk.toString('base64');
      
      const mediaMessage = {
        event: 'media',
        streamSid: this.streamSid,
        media: {
          payload: chunkBase64
        }
      };
      
      this.connection.send(JSON.stringify(mediaMessage));
      
      // Small delay to simulate real-time streaming
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    
    console.log(`[${this.callId}] Sent ${ulawBuffer.length} bytes of audio`);
  }
  
  handleMediaMessage(message) {
    const { payload } = message.media;
    
    // Decode µ-law to PCM16
    const pcmInt16 = decodeUlaw(payload);
    
    // Convert to Buffer for Azure
    const pcmBuffer = Buffer.from(pcmInt16.buffer, pcmInt16.byteOffset, pcmInt16.byteLength);
    
    // Push to Azure STT
    this.pushStream.write(pcmBuffer);
  }
  
  cleanup() {
    console.log(`[${this.callId}] Cleaning up session`);
    
    if (this.recognizer) {
      this.recognizer.stopContinuousRecognitionAsync(
        () => {
          this.recognizer.close();
          console.log(`[${this.callId}] Recognizer stopped`);
        },
        err => console.error(`[${this.callId}] Error stopping recognizer:`, err)
      );
    }
    
    if (this.pushStream) {
      this.pushStream.close();
    }
  }
}

// ==================== FASTIFY SERVER SETUP ====================

const fastify = Fastify({
  logger: {
    level: 'info',
    transport: {
      target: 'pino-pretty'
    }
  }
});

// Register WebSocket plugin
await fastify.register(websocket);

// Store active sessions
const activeSessions = new Map();

// ==================== REST ROUTES ====================

/**
 * POST /make-call
 * Trigger an outbound call
 */
fastify.post('/make-call', async (request, reply) => {
  const { to } = request.body;
  
  if (!to) {
    return reply.code(400).send({ error: 'Missing "to" phone number' });
  }
  
  try {
    const call = await twilioClient.calls.create({
      from: config.twilioPhoneNumber,
      to: to,
      url: `${config.serverUrl.replace('wss://', 'https://')}/call-answer`,
      method: 'POST'
    });
    
    fastify.log.info(`Call initiated: ${call.sid} to ${to}`);
    
    return {
      success: true,
      callSid: call.sid,
      to: to,
      status: call.status
    };
  } catch (error) {
    fastify.log.error('Error making call:', error);
    return reply.code(500).send({ error: error.message });
  }
});

/**
 * POST /call-answer
 * Webhook for when call is answered - returns TwiML
 */
fastify.post('/call-answer', async (request, reply) => {
  const callSid = request.body.CallSid;
  fastify.log.info(`Call answered: ${callSid}`);
  
  const twiml = `<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Hello! I am your AI assistant. How can I help you today?</Say>
  <Start>
    <Stream url="${config.serverUrl}/media-stream/${callSid}" />
  </Start>
  <Pause length="3600"/>
</Response>`;
  
  reply.type('text/xml').send(twiml);
});

/**
 * GET /health
 * Health check endpoint
 */
fastify.get('/health', async (request, reply) => {
  return {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    activeSessions: activeSessions.size
  };
});

// ==================== WEBSOCKET ROUTE ====================

/**
 * WebSocket route for Twilio Media Streams
 */
fastify.register(async (fastify) => {
  fastify.get('/media-stream/:callId', { websocket: true }, (connection, req) => {
    const callId = req.params.callId;
    fastify.log.info(`[${callId}] WebSocket connection established`);
    
    let session = null;
    
    connection.on('message', (message) => {
      try {
        const data = JSON.parse(message.toString());
        
        switch (data.event) {
          case 'start':
            fastify.log.info(`[${callId}] Stream started:`, data.start);
            session = new CallSession(callId, connection);
            session.streamSid = data.start.streamSid;
            activeSessions.set(callId, session);
            break;
            
          case 'media':
            if (session) {
              session.handleMediaMessage(data);
            }
            break;
            
          case 'stop':
            fastify.log.info(`[${callId}] Stream stopped`);
            if (session) {
              session.cleanup();
              activeSessions.delete(callId);
            }
            break;
            
          default:
            fastify.log.debug(`[${callId}] Unknown event: ${data.event}`);
        }
      } catch (error) {
        fastify.log.error(`[${callId}] Error handling message:`, error);
      }
    });
    
    connection.on('close', () => {
      fastify.log.info(`[${callId}] WebSocket connection closed`);
      if (session) {
        session.cleanup();
        activeSessions.delete(callId);
      }
    });
    
    connection.on('error', (error) => {
      fastify.log.error(`[${callId}] WebSocket error:`, error);
      if (session) {
        session.cleanup();
        activeSessions.delete(callId);
      }
    });
  });
});

// ==================== START SERVER ====================

const start = async () => {
  try {
    await fastify.listen({ 
      port: config.port, 
      host: config.host 
    });
    
    console.log('\n=================================================');
    console.log('🚀 Twilio AI Voice Assistant Server Running');
    console.log('=================================================');
    console.log(`📞 Server URL: ${config.serverUrl}`);
    console.log(`🔌 HTTP Port: ${config.port}`);
    console.log(`🎤 Azure Region: ${config.azureSpeechRegion}`);
    console.log(`🤖 OpenAI Model: ${config.openaiModel}`);
    console.log('=================================================\n');
    
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

// Handle graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n\nShutting down gracefully...');
  
  // Cleanup all active sessions
  for (const [callId, session] of activeSessions.entries()) {
    console.log(`Cleaning up session: ${callId}`);
    session.cleanup();
  }
  
  await fastify.close();
  process.exit(0);
});

start();