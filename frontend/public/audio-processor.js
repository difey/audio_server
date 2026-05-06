// AudioWorkletProcessor for PCM capture
// This runs in a separate audio thread (AudioWorklet scope)
// Registered by AudioRecorder.vue via addModule()

class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    this.targetSampleRate = options.processorOptions?.targetSampleRate || 16000
    this.targetChunkSamples = options.processorOptions?.targetChunkSamples || 480
    this.inputSampleRate = sampleRate // from AudioWorkletGlobalScope
    this.resampleBuffer = []
    this.resampleRatio = this.targetSampleRate / this.inputSampleRate
    this.accumulator = 0.0
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0]
    if (!input || !input[0] || input[0].length === 0) return true

    const channelData = input[0]

    for (let i = 0; i < channelData.length; i++) {
      this.accumulator += this.resampleRatio
      while (this.accumulator >= 1.0) {
        this.accumulator -= 1.0
        this.resampleBuffer.push(channelData[i])
        if (this.resampleBuffer.length >= this.targetChunkSamples) {
          this._sendChunk()
        }
      }
    }

    return true
  }

  _sendChunk() {
    const chunk = this.resampleBuffer.splice(0, this.targetChunkSamples)
    const int16 = new Int16Array(chunk.length)
    for (let i = 0; i < chunk.length; i++) {
      const s = Math.max(-1, Math.min(1, chunk[i]))
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
    }
    this.port.postMessage(int16.buffer, [int16.buffer])
  }
}

registerProcessor('pcm-capture-processor', PCMCaptureProcessor)
