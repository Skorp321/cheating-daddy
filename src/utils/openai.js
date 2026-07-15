const { getSystemPrompt } = require('./prompts');
const { sendToRenderer, initializeNewSession, saveConversationTurn } = require('./gemini');
const { SpeechSegmenter, loadWhisperPipeline, transcribe, VAD_MODES } = require('./stt');
const { pcmToWavBuffer } = require('../audioUtils');

// ── State ──

let cfg = null; // { baseUrl, apiKey, model, language, sttLang, sttMode, sttBaseUrl, sttApiKey, sttModel }
let isOpenAIActive = false;
let currentSystemPrompt = null;
let conversationHistory = []; // text-only {role, content}, trimmed to last 20
let systemSegmenter = null; // system audio → Interviewer
let micSegmenter = null; // microphone → Candidate
let turnBuffer = []; // [{speaker, text}] accumulated since the last generation
let activeAbort = null; // AbortController of the in-flight chat request

const DEFAULT_IMAGE_PROMPT = 'Analyze the screen and give me the most helpful, complete answer for what is shown. No fluff.';

function stripThinkingTags(text) {
    return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

// Sleep that rejects promptly if the request is aborted mid-backoff
function abortableSleep(ms, signal) {
    return new Promise((resolve, reject) => {
        if (signal?.aborted) {
            reject(Object.assign(new Error('aborted'), { name: 'AbortError' }));
            return;
        }
        const timer = setTimeout(resolve, ms);
        if (signal) {
            signal.addEventListener(
                'abort',
                () => {
                    clearTimeout(timer);
                    reject(Object.assign(new Error('aborted'), { name: 'AbortError' }));
                },
                { once: true }
            );
        }
    });
}

// POST the chat request, retrying transient failures (5xx / network) before any
// streaming has started. Third-party gateways return intermittent 500s; a couple
// of quick retries turn a flaky hiccup into a normal answer instead of a failed turn.
async function fetchChatWithRetry(url, options, { maxRetries = 2 } = {}) {
    let lastError;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        if (options.signal?.aborted) throw Object.assign(new Error('aborted'), { name: 'AbortError' });
        try {
            const response = await fetch(url, options);
            if (response.status >= 500 && attempt < maxRetries) {
                const errorText = await response.text().catch(() => '');
                console.warn(
                    `[OpenAI] Server ${response.status} (attempt ${attempt + 1}/${maxRetries + 1}), retrying:`,
                    errorText.slice(0, 120)
                );
                sendToRenderer('update-status', `Server busy (${response.status}), retrying...`);
                await abortableSleep(500 * (attempt + 1), options.signal);
                continue;
            }
            return response;
        } catch (error) {
            if (error.name === 'AbortError') throw error;
            lastError = error;
            if (attempt < maxRetries) {
                console.warn(`[OpenAI] Network error (attempt ${attempt + 1}/${maxRetries + 1}), retrying:`, error.message);
                sendToRenderer('update-status', 'Connection issue, retrying...');
                await abortableSleep(500 * (attempt + 1), options.signal);
                continue;
            }
            throw error;
        }
    }
    throw lastError;
}

// ── Speech handling ──

async function handleSpeechEnd(pcm16k, source) {
    if (!isOpenAIActive) return;

    sendToRenderer('update-status', 'Transcribing...');
    const text = cfg.sttMode === 'remote' ? await transcribeRemote(pcm16k) : await transcribe(pcm16k, cfg.sttLang);

    if (!isOpenAIActive) return;

    if (!text || text.trim().length < 2) {
        console.log('[OpenAI] Empty transcription, skipping');
        sendToRenderer('update-status', 'Listening...');
        return;
    }

    turnBuffer.push({
        speaker: source === 'mic' ? 'Candidate' : 'Interviewer',
        text: text.trim(),
    });

    // Any completed utterance (interviewer via system audio, or you via the mic)
    // triggers a response. Speaker labels are preserved in the transcript, and if a
    // second utterance lands mid-generation the abort-and-resend logic merges them.
    const transcript = turnBuffer.map(t => `[${t.speaker}]: ${t.text}`).join('\n');
    turnBuffer = [];

    sendToRenderer('update-status', 'Generating response...');
    await sendToOpenAIChat(transcript, { isSpeech: true });
}

async function transcribeRemote(pcm16kBuffer) {
    try {
        const wav = pcmToWavBuffer(pcm16kBuffer, 16000, 1, 16);

        const form = new FormData();
        form.append('file', new Blob([wav], { type: 'audio/wav' }), 'audio.wav');
        form.append('model', cfg.sttModel || 'whisper-1');
        form.append('language', cfg.sttLang);

        // No manual Content-Type: fetch must set the multipart boundary itself
        const headers = {};
        const key = cfg.sttApiKey || cfg.apiKey;
        if (key) headers['Authorization'] = `Bearer ${key}`;

        const response = await fetch(`${cfg.sttBaseUrl}/audio/transcriptions`, {
            method: 'POST',
            headers,
            body: form,
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('[OpenAI] Transcription API error:', response.status, errorText.substring(0, 200));
            return null;
        }

        const json = await response.json();
        const text = json.text?.trim();
        console.log('[OpenAI] Remote transcription:', text);
        return text;
    } catch (error) {
        console.error('[OpenAI] Remote transcription error:', error);
        return null;
    }
}

// ── Chat completions (OpenAI-compatible, SSE streaming) ──

async function sendToOpenAIChat(userText, { contentOverride = null, isSpeech = false } = {}) {
    if (!cfg) return null;

    // Single in-flight request: a newer send supersedes the current one.
    // If the interviewer kept talking, merge transcripts and answer the fuller question.
    if (activeAbort) {
        const priorTranscript = activeAbort.speechTranscript;
        activeAbort.abort();
        if (isSpeech && priorTranscript) {
            userText = priorTranscript + '\n' + userText;
        }
    }

    const controller = new AbortController();
    controller.speechTranscript = isSpeech ? userText : null;
    activeAbort = controller;

    const userMessage = { role: 'user', content: userText.trim() };
    conversationHistory.push(userMessage);
    if (conversationHistory.length > 20) {
        conversationHistory = conversationHistory.slice(-20);
    }

    const messages = [
        { role: 'system', content: currentSystemPrompt || 'You are a helpful assistant.' },
        ...conversationHistory,
    ];
    if (contentOverride) {
        // Rich content (e.g. vision) goes only into the request; history keeps the text-only version
        messages[messages.length - 1] = { role: 'user', content: contentOverride };
    }

    console.log(`[OpenAI] Sending to ${cfg.model}:`, userText.substring(0, 100) + '...');

    try {
        const headers = { 'Content-Type': 'application/json' };
        if (cfg.apiKey) headers['Authorization'] = `Bearer ${cfg.apiKey}`;

        const response = await fetchChatWithRetry(`${cfg.baseUrl}/chat/completions`, {
            method: 'POST',
            headers,
            body: JSON.stringify({
                model: cfg.model,
                messages,
                stream: true,
                temperature: 0.7,
                // Reasoning models spend completion tokens on thinking before any
                // visible content, so the budget must be generous
                max_tokens: 4096,
            }),
            signal: controller.signal,
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API error ${response.status}: ${errorText.substring(0, 200)}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let isFirst = true;
        let lineRemainder = '';
        let sawDeltas = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // Servers chunk arbitrarily — a "data:" line can be split across reads
            const lines = (lineRemainder + decoder.decode(value, { stream: true })).split('\n');
            lineRemainder = lines.pop();

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed.startsWith('data: ')) continue;

                const data = trimmed.slice(6);
                if (data === '[DONE]') continue;

                try {
                    const json = JSON.parse(data);
                    if (json.choices?.[0]?.delta) sawDeltas = true;
                    const token = json.choices?.[0]?.delta?.content || '';
                    if (token) {
                        fullText += token;
                        const displayText = stripThinkingTags(fullText);
                        if (displayText) {
                            sendToRenderer(isFirst ? 'new-response' : 'update-response', displayText);
                            isFirst = false;
                        }
                    }
                } catch (parseError) {
                    // Skip invalid JSON chunks
                }
            }
        }

        const cleanedResponse = stripThinkingTags(fullText);
        if (cleanedResponse) {
            conversationHistory.push({ role: 'assistant', content: cleanedResponse });
            saveConversationTurn(userText, cleanedResponse);
            console.log('[OpenAI] Response completed');
            sendToRenderer('update-status', 'Listening...');
        } else if (sawDeltas) {
            // Reasoning models can burn the whole token budget on thinking
            console.warn('[OpenAI] Stream finished with no visible content (reasoning-only?)');
            sendToRenderer('update-status', 'Model produced no answer (thinking used up the token budget) - Listening...');
        } else {
            sendToRenderer('update-status', 'Listening...');
        }
        return cleanedResponse;
    } catch (error) {
        // Drop the failed/superseded user message so history matches what was actually answered
        const idx = conversationHistory.indexOf(userMessage);
        if (idx !== -1) conversationHistory.splice(idx, 1);

        if (error.name === 'AbortError') {
            console.log('[OpenAI] Request superseded');
            return null;
        }

        console.error('[OpenAI] Chat error:', error);
        sendToRenderer('update-status', 'OpenAI error: ' + error.message);
        return null;
    } finally {
        if (activeAbort === controller) activeAbort = null;
    }
}

// ── Public API ──

async function initializeOpenAISession(config, profile = 'interview', customPrompt = '') {
    console.log('[OpenAI] Initializing session:', {
        baseUrl: config.baseUrl,
        model: config.model,
        sttMode: config.sttMode,
    });

    sendToRenderer('session-initializing', true);

    try {
        const baseUrl = (config.baseUrl || '').trim().replace(/\/+$/, '');
        const sttBaseUrl = (config.sttBaseUrl || '').trim().replace(/\/+$/, '') || baseUrl;
        const language = config.language || 'ru-RU';

        cfg = {
            baseUrl,
            apiKey: (config.apiKey || '').trim(),
            model: config.model,
            language,
            sttLang: language.split('-')[0], // ISO 639-1 code for Whisper (e.g. 'ru')
            sttMode: config.sttMode === 'remote' ? 'remote' : 'local',
            sttBaseUrl,
            sttApiKey: (config.sttApiKey || '').trim(),
            sttModel: config.sttModel || 'whisper-1',
        };

        currentSystemPrompt = getSystemPrompt(profile, customPrompt, false, language);

        // Probe the endpoint; some proxies don't implement /models, so non-2xx is only a warning
        try {
            const headers = cfg.apiKey ? { Authorization: `Bearer ${cfg.apiKey}` } : {};
            const probe = await fetch(`${cfg.baseUrl}/models`, { headers });
            if (probe.ok) {
                console.log('[OpenAI] Endpoint connection verified');
            } else {
                console.warn('[OpenAI] /models probe returned', probe.status, '- continuing anyway');
            }
        } catch (error) {
            console.error('[OpenAI] Cannot connect to', cfg.baseUrl, ':', error.message);
            sendToRenderer('session-initializing', false);
            sendToRenderer('update-status', 'Cannot connect to ' + cfg.baseUrl);
            return false;
        }

        if (cfg.sttMode === 'local') {
            const pipeline = await loadWhisperPipeline(config.whisperModel || 'Xenova/whisper-small');
            if (!pipeline) {
                sendToRenderer('session-initializing', false);
                return false;
            }
        }

        systemSegmenter = new SpeechSegmenter({
            label: 'OpenAI:system',
            vadConfig: VAD_MODES.VERY_AGGRESSIVE,
            onSpeechStart: () => sendToRenderer('update-status', 'Listening... (speech detected)'),
            onSpeechEnd: audio => handleSpeechEnd(audio, 'system'),
            onSpeechDiscarded: () => sendToRenderer('update-status', 'Listening...'),
        });
        micSegmenter = new SpeechSegmenter({
            label: 'OpenAI:mic',
            vadConfig: VAD_MODES.NORMAL, // mic input is quieter than loopback audio
            onSpeechStart: () => sendToRenderer('update-status', 'Listening... (speech detected)'),
            onSpeechEnd: audio => handleSpeechEnd(audio, 'mic'),
            onSpeechDiscarded: () => sendToRenderer('update-status', 'Listening...'),
        });

        conversationHistory = [];
        turnBuffer = [];

        initializeNewSession(profile, customPrompt);

        isOpenAIActive = true;
        sendToRenderer('session-initializing', false);
        sendToRenderer('update-status', 'OpenAI-compatible AI ready - Listening...');

        console.log('[OpenAI] Session initialized successfully');
        return true;
    } catch (error) {
        console.error('[OpenAI] Initialization error:', error);
        sendToRenderer('session-initializing', false);
        sendToRenderer('update-status', 'OpenAI error: ' + error.message);
        return false;
    }
}

function processOpenAIAudio(monoChunk24k, source = 'system') {
    if (!isOpenAIActive) return;

    const segmenter = source === 'mic' ? micSegmenter : systemSegmenter;
    if (segmenter) segmenter.push24k(monoChunk24k);
}

async function sendOpenAIText(text) {
    if (!isOpenAIActive || !cfg) {
        return { success: false, error: 'No active OpenAI session' };
    }

    try {
        sendToRenderer('update-status', 'Generating response...');
        await sendToOpenAIChat(text);
        return { success: true };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

async function sendOpenAIImage(base64Data, prompt) {
    if (!isOpenAIActive || !cfg) {
        return { success: false, error: 'No active OpenAI session' };
    }

    try {
        console.log('[OpenAI] Sending image');
        sendToRenderer('update-status', 'Analyzing image...');

        const textPrompt = (prompt && prompt.trim()) || DEFAULT_IMAGE_PROMPT;
        const contentOverride = [
            { type: 'text', text: textPrompt },
            { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${base64Data}` } },
        ];

        const text = await sendToOpenAIChat(textPrompt, { contentOverride });
        return { success: true, text: text || '', model: cfg.model };
    } catch (error) {
        console.error('[OpenAI] Image error:', error);
        sendToRenderer('update-status', 'OpenAI error: ' + error.message);
        return { success: false, error: error.message };
    }
}

function closeOpenAISession() {
    console.log('[OpenAI] Closing session');
    isOpenAIActive = false;
    if (activeAbort) {
        activeAbort.abort();
        activeAbort = null;
    }
    if (systemSegmenter) systemSegmenter.reset();
    if (micSegmenter) micSegmenter.reset();
    systemSegmenter = null;
    micSegmenter = null;
    conversationHistory = [];
    turnBuffer = [];
    currentSystemPrompt = null;
    cfg = null;
    // Note: the shared Whisper pipeline (stt.js) is kept loaded to avoid reloading on next session
}

function isOpenAISessionActive() {
    return isOpenAIActive;
}

module.exports = {
    initializeOpenAISession,
    processOpenAIAudio,
    sendOpenAIText,
    sendOpenAIImage,
    closeOpenAISession,
    isOpenAISessionActive,
};
