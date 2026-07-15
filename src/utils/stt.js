const { sendToRenderer } = require('./gemini');

// ── VAD configuration ──

const VAD_MODES = {
    NORMAL: { energyThreshold: 0.01, speechFramesRequired: 3, silenceFramesRequired: 30 },
    LOW_BITRATE: { energyThreshold: 0.008, speechFramesRequired: 4, silenceFramesRequired: 35 },
    AGGRESSIVE: { energyThreshold: 0.015, speechFramesRequired: 2, silenceFramesRequired: 20 },
    VERY_AGGRESSIVE: { energyThreshold: 0.02, speechFramesRequired: 2, silenceFramesRequired: 15 },
};

function calculateRMS(pcm16Buffer) {
    const samples = pcm16Buffer.length / 2;
    if (samples === 0) return 0;
    let sumSquares = 0;
    for (let i = 0; i < samples; i++) {
        const sample = pcm16Buffer.readInt16LE(i * 2) / 32768;
        sumSquares += sample * sample;
    }
    return Math.sqrt(sumSquares / samples);
}

// ── Speech segmentation: 24kHz mono PCM in → speech utterances (16kHz PCM) out ──
// Each instance keeps its own resampling/VAD state, so multiple audio sources
// (e.g. system audio and microphone) can be segmented independently.

class SpeechSegmenter {
    constructor({
        vadConfig = VAD_MODES.VERY_AGGRESSIVE,
        minSpeechBytes = 16000, // ~0.5 seconds at 16kHz, 16-bit
        label = 'STT',
        onSpeechStart = null,
        onSpeechEnd = null,
        onSpeechDiscarded = null,
    } = {}) {
        this.vadConfig = vadConfig;
        this.minSpeechBytes = minSpeechBytes;
        this.label = label;
        this.onSpeechStart = onSpeechStart;
        this.onSpeechEnd = onSpeechEnd;
        this.onSpeechDiscarded = onSpeechDiscarded;
        this.reset();
    }

    reset() {
        this.isSpeaking = false;
        this.speechBuffers = [];
        this.silenceFrameCount = 0;
        this.speechFrameCount = 0;
        this.resampleRemainder = Buffer.alloc(0);
    }

    push24k(monoChunk24k) {
        const pcm16k = this._resample24kTo16k(monoChunk24k);
        if (pcm16k.length > 0) {
            this._processVAD(pcm16k);
        }
    }

    _resample24kTo16k(inputBuffer) {
        // Combine with any leftover samples from previous call
        const combined = Buffer.concat([this.resampleRemainder, inputBuffer]);
        const inputSamples = Math.floor(combined.length / 2); // 16-bit = 2 bytes per sample
        // Ratio: 16000/24000 = 2/3, so for every 3 input samples we produce 2 output samples
        const outputSamples = Math.floor((inputSamples * 2) / 3);
        const outputBuffer = Buffer.alloc(outputSamples * 2);

        for (let i = 0; i < outputSamples; i++) {
            // Map output sample index to input position
            const srcPos = (i * 3) / 2;
            const srcIndex = Math.floor(srcPos);
            const frac = srcPos - srcIndex;

            const s0 = combined.readInt16LE(srcIndex * 2);
            const s1 = srcIndex + 1 < inputSamples ? combined.readInt16LE((srcIndex + 1) * 2) : s0;
            const interpolated = Math.round(s0 + frac * (s1 - s0));
            outputBuffer.writeInt16LE(Math.max(-32768, Math.min(32767, interpolated)), i * 2);
        }

        // Store remainder for next call
        const consumedInputSamples = Math.ceil((outputSamples * 3) / 2);
        const remainderStart = consumedInputSamples * 2;
        this.resampleRemainder = remainderStart < combined.length ? combined.slice(remainderStart) : Buffer.alloc(0);

        return outputBuffer;
    }

    _processVAD(pcm16kBuffer) {
        const rms = calculateRMS(pcm16kBuffer);
        const isVoice = rms > this.vadConfig.energyThreshold;

        if (isVoice) {
            this.speechFrameCount++;
            this.silenceFrameCount = 0;

            if (!this.isSpeaking && this.speechFrameCount >= this.vadConfig.speechFramesRequired) {
                this.isSpeaking = true;
                this.speechBuffers = [];
                console.log(`[${this.label}] Speech started (RMS:`, rms.toFixed(4), ')');
                if (this.onSpeechStart) this.onSpeechStart();
            }
        } else {
            this.silenceFrameCount++;
            this.speechFrameCount = 0;

            if (this.isSpeaking && this.silenceFrameCount >= this.vadConfig.silenceFramesRequired) {
                this.isSpeaking = false;
                console.log(`[${this.label}] Speech ended, accumulated`, this.speechBuffers.length, 'chunks');

                const audioData = Buffer.concat(this.speechBuffers);
                this.speechBuffers = [];

                if (audioData.length < this.minSpeechBytes) {
                    console.log(`[${this.label}] Audio too short, skipping`);
                    if (this.onSpeechDiscarded) this.onSpeechDiscarded();
                } else if (this.onSpeechEnd) {
                    this.onSpeechEnd(audioData);
                }
                return;
            }
        }

        // Accumulate audio during speech
        if (this.isSpeaking) {
            this.speechBuffers.push(Buffer.from(pcm16kBuffer));
        }
    }
}

// ── Whisper transcription (shared singleton so switching providers doesn't reload the model) ──

let whisperPipeline = null;
let isWhisperLoading = false;
let transcribeChain = Promise.resolve();

async function loadWhisperPipeline(modelName) {
    if (whisperPipeline) return whisperPipeline;
    if (isWhisperLoading) return null;

    isWhisperLoading = true;
    console.log('[STT] Loading Whisper model:', modelName);
    sendToRenderer('whisper-downloading', true);
    sendToRenderer('update-status', 'Loading Whisper model (first time may take a while)...');

    try {
        // Dynamic import for ESM module
        const { pipeline, env } = await import('@huggingface/transformers');
        // Cache models outside the asar archive so ONNX runtime can load them
        const { app } = require('electron');
        const path = require('path');
        env.cacheDir = path.join(app.getPath('userData'), 'whisper-models');

        // Default to CPU. The GPU (CUDA) execution provider in onnxruntime-node is
        // fragile with mismatched CUDA/cuDNN: it can load fine but then segfault during
        // inference, which kills the whole Electron process uncatchably. CPU is a bit
        // slower but reliable for the short utterances we transcribe. Advanced users with
        // a working CUDA/cuDNN can opt into GPU via CHEATING_DADDY_WHISPER_DEVICE=auto|cuda.
        const device = process.env.CHEATING_DADDY_WHISPER_DEVICE || 'cpu';
        try {
            whisperPipeline = await pipeline('automatic-speech-recognition', modelName, {
                dtype: 'q8',
                device,
            });
        } catch (accelError) {
            if (device === 'cpu') throw accelError;
            console.warn('[STT] Whisper device "' + device + '" failed to load, falling back to CPU:', accelError.message);
            sendToRenderer('update-status', 'GPU unavailable, loading Whisper on CPU...');
            whisperPipeline = await pipeline('automatic-speech-recognition', modelName, {
                dtype: 'q8',
                device: 'cpu',
            });
        }

        console.log('[STT] Whisper model loaded successfully');
        sendToRenderer('whisper-downloading', false);
        isWhisperLoading = false;
        return whisperPipeline;
    } catch (error) {
        console.error('[STT] Failed to load Whisper model:', error);
        sendToRenderer('whisper-downloading', false);
        sendToRenderer('update-status', 'Failed to load Whisper model: ' + error.message);
        isWhisperLoading = false;
        return null;
    }
}

function pcm16ToFloat32(pcm16Buffer) {
    const samples = pcm16Buffer.length / 2;
    const float32 = new Float32Array(samples);
    for (let i = 0; i < samples; i++) {
        float32[i] = pcm16Buffer.readInt16LE(i * 2) / 32768;
    }
    return float32;
}

async function doTranscribe(pcm16kBuffer, language) {
    if (!whisperPipeline) {
        console.error('[STT] Whisper pipeline not loaded');
        return null;
    }

    try {
        const float32Audio = pcm16ToFloat32(pcm16kBuffer);

        // Whisper expects audio at 16kHz which is what we have
        const result = await whisperPipeline(float32Audio, {
            sampling_rate: 16000,
            language, // ISO 639-1 code (e.g. 'ru', 'en'); undefined => auto-detect
            task: 'transcribe',
        });

        const text = result.text?.trim();
        console.log('[STT] Transcription:', text);
        return text;
    } catch (error) {
        console.error('[STT] Transcription error:', error);
        return null;
    }
}

// Concurrent calls into one transformers.js pipeline interleave badly — serialize them
function transcribe(pcm16kBuffer, language = 'ru') {
    const task = transcribeChain.then(() => doTranscribe(pcm16kBuffer, language));
    transcribeChain = task.then(
        () => {},
        () => {}
    );
    return task;
}

module.exports = {
    VAD_MODES,
    SpeechSegmenter,
    loadWhisperPipeline,
    transcribe,
};
