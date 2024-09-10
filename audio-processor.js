// audio-processor.js
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.port.onmessage = (event) => {
            console.log()
        };
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];  // Input from the microphone
        if (input.length > 0) {
            const audioData = input[0];  // Get data from the first channel

            // Convert Float32Array to Int16Array
            const audioInt16 = new Int16Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                audioInt16[i] = audioData[i] * 32767;  // Convert float (-1.0 to 1.0) to int (-32767 to 32767)
            }

            // Send processed data to the main thread
            this.port.postMessage(audioInt16.buffer);
        }

        return true;  // Keep processing
    }
}

registerProcessor('audio-processor', AudioProcessor);
