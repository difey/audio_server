<template>
  <div class="app">
    <header class="header">
      <h1 class="title">ASR Server</h1>
      <p class="subtitle">流式语音转写 · Streaming ASR</p>
    </header>

    <main class="main">
      <AudioRecorder
        :disabled="status !== 'connected'"
        @audio-data="onAudioData"
        @recording-state="onRecordingState"
      />

      <div class="status-bar">
        <span class="status-dot" :class="status"></span>
        <span class="status-text">{{ statusLabel }}</span>
        <span v-if="wsError" class="error-msg">{{ wsError }}</span>
      </div>

      <TranscriptionDisplay
        :results="results"
        :interim-text="currentInterim"
        :interim-metrics="currentMetrics"
        :is-recording="isRecording"
        :latency-stats="latencyStats"
      />

      <div class="controls">
        <button
          class="btn"
          :class="{ primary: !isRecording, danger: isRecording }"
          :disabled="status !== 'connected'"
          @click="toggleRecording"
        >
          {{ isRecording ? '⏹ 停止录音' : '🎤 开始录音' }}
        </button>
        <button
          class="btn secondary"
          @click="clearResults"
          :disabled="results.length === 0"
        >
          🗑 清空记录
        </button>
        <button
          class="btn secondary"
          @click="reconnect"
        >
          🔄 重连
        </button>
      </div>
    </main>

    <footer class="footer">
      <small>faster-whisper tiny · WebSocket · AudioWorklet</small>
    </footer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import AudioRecorder from './components/AudioRecorder.vue'
import TranscriptionDisplay from './components/TranscriptionDisplay.vue'

const WS_URL = `ws://${location.hostname}:8000/ws/transcribe`

const status = ref('disconnected')
const wsError = ref('')
const isRecording = ref(false)
const results = ref([])
const currentInterim = ref('')
const currentMetrics = ref(null)

// Running latency stats
const latencyStats = ref({
  count: 0,
  avg: 0,
  max: 0,
  lastInferenceMs: 0,
})

let ws = null

const statusLabel = computed(() => {
  const map = {
    disconnected: '未连接',
    connecting: '连接中...',
    connected: '已连接',
  }
  return map[status.value] || status.value
})

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return
  }

  status.value = 'connecting'
  wsError.value = ''
  ws = new WebSocket(WS_URL)
  ws.binaryType = 'arraybuffer'

  ws.onopen = () => {
    status.value = 'connected'
    wsError.value = ''
  }

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) return
    try {
      const msg = JSON.parse(event.data)
      handleMessage(msg)
    } catch (e) {
      // ignore
    }
  }

  ws.onerror = () => {
    wsError.value = '连接错误'
  }

  ws.onclose = () => {
    status.value = 'disconnected'
    if (isRecording.value) {
      isRecording.value = false
    }
  }
}

function handleMessage(msg) {
  switch (msg.type) {
    case 'speech_start':
      currentInterim.value = ''
      currentMetrics.value = null
      break
    case 'interim':
      currentInterim.value = msg.text
      currentMetrics.value = msg.metrics || null
      if (msg.metrics) {
        latencyStats.value.lastInferenceMs = msg.metrics.inference_ms
        updateLatencyStats(msg.metrics)
      }
      break
    case 'final':
      results.value.push({
        id: Date.now(),
        text: msg.text,
        timestamp: new Date().toLocaleTimeString(),
        type: 'final',
        metrics: msg.metrics || null,
      })
      currentInterim.value = ''
      currentMetrics.value = null
      if (msg.metrics) {
        latencyStats.value.lastInferenceMs = msg.metrics.inference_ms
        updateLatencyStats(msg.metrics)
      }
      scrollToBottom()
      break
    case 'error':
      wsError.value = msg.message
      break
  }
}

function updateLatencyStats(m) {
  const s = latencyStats.value
  s.count++
  const total = s.avg * (s.count - 1) + (m.inference_ms || 0)
  s.avg = Math.round(total / s.count)
  s.max = Math.max(s.max, m.inference_ms || 0)
}

function scrollToBottom() {
  setTimeout(() => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
  }, 50)
}

function onAudioData(pcmBuffer) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(pcmBuffer)
  }
}

function onRecordingState(recording) {
  isRecording.value = recording
}

function toggleRecording() {
  if (isRecording.value) {
    window.dispatchEvent(new CustomEvent('stop-recording'))
  } else {
    window.dispatchEvent(new CustomEvent('start-recording'))
  }
}

function clearResults() {
  results.value = []
  currentInterim.value = ''
}

function reconnect() {
  if (ws) {
    ws.close()
  }
  connect()
}

onMounted(() => {
  connect()
})

onUnmounted(() => {
  if (ws) {
    ws.close()
  }
})
</script>

<style scoped>
.app {
  max-width: 720px;
  margin: 0 auto;
  padding: 20px 16px;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  text-align: center;
  padding: 24px 0 16px;
}

.title {
  font-size: 1.8rem;
  font-weight: 700;
  background: linear-gradient(135deg, #60a5fa, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.subtitle {
  color: #64748b;
  font-size: 0.9rem;
  margin-top: 4px;
}

.main {
  flex: 1;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #1e293b;
  border-radius: 8px;
  margin: 12px 0;
  font-size: 0.85rem;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.status-dot.connected { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
.status-dot.connecting { background: #eab308; animation: pulse 1s infinite; }
.status-dot.disconnected { background: #ef4444; }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.status-text { flex: 1; color: #94a3b8; }
.error-msg { color: #f87171; font-size: 0.8rem; }

.controls {
  display: flex;
  gap: 8px;
  padding: 16px 0;
  flex-wrap: wrap;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.2s;
  font-weight: 500;
}
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn.primary { background: #3b82f6; color: white; }
.btn.primary:hover:not(:disabled) { background: #2563eb; }
.btn.danger { background: #ef4444; color: white; }
.btn.danger:hover:not(:disabled) { background: #dc2626; }
.btn.secondary { background: #334155; color: #cbd5e1; }
.btn.secondary:hover:not(:disabled) { background: #475569; }

.footer {
  text-align: center;
  padding: 20px 0;
  color: #475569;
  font-size: 0.8rem;
}

@media (max-width: 480px) {
  .app { padding: 12px 10px; }
  .title { font-size: 1.4rem; }
  .btn { padding: 8px 14px; font-size: 0.9rem; }
  .controls { gap: 6px; }
}
</style>
