<template>
  <div class="recorder-widget">
    <canvas ref="vumeter" class="vumeter" width="240" height="24"></canvas>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const emit = defineEmits(['audio-data', 'recording-state'])
const props = defineProps({
  disabled: Boolean,
})

const vumeter = ref(null)

let audioContext = null
let mediaStream = null
let workletNode = null
let sourceNode = null
let isRecording = false
let animationId = null

const TARGET_SAMPLE_RATE = 16000
const TARGET_CHUNK_SAMPLES = 480 // 30ms at 16kHz

// ── AudioWorklet registration ───────────────────────────────────────
const WORKLET_CODE = `
class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    this.targetSampleRate = options.processorOptions?.targetSampleRate || 16000
    this.targetChunkSamples = options.processorOptions?.targetChunkSamples || 480
    this.inputSampleRate = sampleRate // from AudioWorkletGlobalScope
    this.resampleBuffer = []
    this.resampleRatio = this.targetSampleRate / this.inputSampleRate
    this.accumulator = 0.0
    this.lastSample = 0.0
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0]
    if (!input || !input[0] || input[0].length === 0) return true

    const channelData = input[0]

    // Downsample to target sample rate using linear interpolation
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
    // Convert float32 [-1,1] to int16 and send
    const int16 = new Int16Array(chunk.length)
    for (let i = 0; i < chunk.length; i++) {
      const s = Math.max(-1, Math.min(1, chunk[i]))
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
    }
    this.port.postMessage(int16.buffer, [int16.buffer])
  }
}

registerProcessor('pcm-capture-processor', PCMCaptureProcessor)
`

async function startRecording() {
  if (isRecording || props.disabled) return

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    })

    audioContext = new AudioContext()
    sourceNode = audioContext.createMediaStreamSource(mediaStream)

    // Register AudioWorklet
    const blob = new Blob([WORKLET_CODE], { type: 'application/javascript' })
    const url = URL.createObjectURL(blob)
    await audioContext.audioWorklet.addModule(url)
    URL.revokeObjectURL(url)

    // Create worklet node
    workletNode = new AudioWorkletNode(audioContext, 'pcm-capture-processor', {
      processorOptions: {
        targetSampleRate: TARGET_SAMPLE_RATE,
        targetChunkSamples: TARGET_CHUNK_SAMPLES,
      },
    })

    workletNode.port.onmessage = (event) => {
      emit('audio-data', event.data)
      drawVU()
    }

    sourceNode.connect(workletNode)
    workletNode.connect(audioContext.destination)

    isRecording = true
    emit('recording-state', true)
  } catch (err) {
    console.error('Recording error:', err)
    alert('麦克风访问失败: ' + err.message)
  }
}

function stopRecording() {
  if (workletNode) {
    workletNode.disconnect()
    workletNode = null
  }
  if (sourceNode) {
    sourceNode.disconnect()
    sourceNode = null
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop())
    mediaStream = null
  }
  if (audioContext) {
    audioContext.close()
    audioContext = null
  }
  if (animationId) {
    cancelAnimationFrame(animationId)
    animationId = null
  }
  isRecording = false
  emit('recording-state', false)
}

// ── VU meter ────────────────────────────────────────────────────────
let vuPhase = 0

function drawVU() {
  const canvas = vumeter.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')

  vuPhase = (vuPhase + 0.05) % 1
  const width = Math.sin(vuPhase * Math.PI) * 0.3 + 0.1

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // Background
  ctx.fillStyle = '#1e293b'
  ctx.roundRect(0, 0, canvas.width, canvas.height, 4)
  ctx.fill()

  // Level bar
  const barWidth = canvas.width * width
  ctx.fillStyle = '#22c55e'
  ctx.roundRect(2, 2, Math.max(barWidth, 4), canvas.height - 4, 3)
  ctx.fill()
}

// ── Lifecycle ────────────────────────────────────────────────────────
onMounted(() => {
  window.addEventListener('start-recording', startRecording)
  window.addEventListener('stop-recording', stopRecording)
})

onUnmounted(() => {
  window.removeEventListener('start-recording', startRecording)
  window.removeEventListener('stop-recording', stopRecording)
  stopRecording()
})
</script>

<style scoped>
.recorder-widget {
  display: flex;
  justify-content: center;
  padding: 8px 0;
}

.vumeter {
  width: 100%;
  max-width: 320px;
  height: 20px;
  border-radius: 4px;
}
</style>
