/**
 * Audio filtering utilities for VAD (Voice Activity Detection) and noise gating.
 * These filters help reduce network traffic and prevent hallucinations from silence.
 */

/**
 * Calculate RMS (Root Mean Square) energy of an audio buffer.
 * This is a measure of the overall volume/energy in the signal.
 */
const calculateRMS = (samples: Float32Array): number => {
    if (samples.length === 0) return 0;
    
    let sum = 0;
    for (let i = 0; i < samples.length; i++) {
        sum += samples[i] * samples[i];
    }
    return Math.sqrt(sum / samples.length);
};

/**
 * Calculate Zero Crossing Rate (ZCR) - how often the signal crosses zero.
 * Speech typically has lower ZCR than noise.
 */
const calculateZCR = (samples: Float32Array): number => {
    if (samples.length < 2) return 0;
    
    let crossings = 0;
    for (let i = 1; i < samples.length; i++) {
        if ((samples[i - 1] >= 0 && samples[i] < 0) || 
            (samples[i - 1] < 0 && samples[i] >= 0)) {
            crossings++;
        }
    }
    return crossings / samples.length;
};

/**
 * Voice Activity Detection configuration.
 */
export interface VADConfig {
    /** Minimum RMS energy threshold for voice detection (0.0 - 1.0). Default: 0.01 */
    energyThreshold?: number;
    /** Maximum Zero Crossing Rate for voice (higher = more noise-like). Default: 0.3 */
    maxZCR?: number;
    /** Minimum duration of speech in samples to consider it valid. Default: 4800 (~0.1s at 48kHz) */
    minSpeechSamples?: number;
}

/**
 * Detect if the audio buffer contains speech or just silence/noise.
 * Returns true if speech is detected, false otherwise.
 */
export const detectVoiceActivity = (
    samples: Float32Array,
    config: VADConfig = {}
): boolean => {
    const {
        energyThreshold = 0.01,
        maxZCR = 0.3,
        minSpeechSamples = 4800,
    } = config;

    if (samples.length < minSpeechSamples) {
        return false;
    }

    const rms = calculateRMS(samples);
    
    // If energy is too low, it's silence
    if (rms < energyThreshold) {
        return false;
    }

    const zcr = calculateZCR(samples);
    
    // If ZCR is too high, it's likely noise rather than speech
    if (zcr > maxZCR) {
        return false;
    }

    return true;
};

/**
 * Noise gate configuration.
 */
export interface NoiseGateConfig {
    /** Threshold below which audio is suppressed (0.0 - 1.0). Default: 0.01 */
    threshold?: number;
    /** Ratio of suppression (higher = more aggressive). Default: 10 (= -20dB reduction) */
    ratio?: number;
    /** Attack time in samples for gate to open. Default: 480 (~10ms at 48kHz) */
    attackSamples?: number;
    /** Release time in samples for gate to close. Default: 4800 (~100ms at 48kHz) */
    releaseSamples?: number;
}

/**
 * Apply a noise gate to suppress low-level audio.
 * Returns a new Float32Array with the gate applied.
 */
export const applyNoiseGate = (
    samples: Float32Array,
    config: NoiseGateConfig = {}
): Float32Array => {
    const {
        threshold = 0.01,
        ratio = 10,
        attackSamples = 480,
        releaseSamples = 4800,
    } = config;

    const output = new Float32Array(samples.length);
    let envelope = 0;

    for (let i = 0; i < samples.length; i++) {
        const abs = Math.abs(samples[i]);
        
        // Update envelope with attack/release
        if (abs > envelope) {
            // Attack
            envelope += (abs - envelope) / attackSamples;
        } else {
            // Release
            envelope -= envelope / releaseSamples;
        }

        // Calculate gain reduction
        let gain = 1;
        if (envelope < threshold) {
            gain = 1 / ratio;
        }

        output[i] = samples[i] * gain;
    }

    return output;
};

/**
 * Combined VAD and noise gate filter.
 * Returns null if no voice activity is detected, otherwise returns the filtered audio.
 */
export const filterAudioChunk = (
    samples: Float32Array,
    vadConfig: VADConfig = {},
    gateConfig: NoiseGateConfig = {}
): Float32Array | null => {
    // First check if there's any voice activity
    if (!detectVoiceActivity(samples, vadConfig)) {
        return null;
    }

    // Apply noise gate to clean up the audio
    return applyNoiseGate(samples, gateConfig);
};

/**
 * Default configuration for typical speech at standard sample rates.
 */
export const getDefaultFilterConfig = (sampleRate: number) => {
    // Scale thresholds based on sample rate (reference: 48kHz)
    const scale = sampleRate / 48000;

    return {
        vad: {
            energyThreshold: 0.01,
            maxZCR: 0.3,
            minSpeechSamples: Math.floor(4800 * scale),
        } as VADConfig,
        gate: {
            threshold: 0.01,
            ratio: 10,
            attackSamples: Math.floor(480 * scale),
            releaseSamples: Math.floor(4800 * scale),
        } as NoiseGateConfig,
    };
};
