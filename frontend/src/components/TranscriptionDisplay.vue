<template>
  <div class="transcript">
    <!-- Empty state -->
    <div v-if="results.length === 0 && !interimText && !isRecording" class="empty">
      <p class="empty-icon">🎙️</p>
      <p>点击「开始录音」并说话</p>
      <p class="empty-hint">语音转写结果将显示在这里</p>
    </div>

    <!-- Listening indicator -->
    <div v-if="isRecording && !interimText" class="result-item listening">
      <span class="text">👂 正在聆听...</span>
    </div>

    <!-- Interim result -->
    <div v-if="interimText" class="result-item interim">
      <div class="result-main">
        <span class="badge interim-badge">实时</span>
        <span class="text typing">{{ interimText }}</span>
      </div>
      <div v-if="interimMetrics" class="metrics-row live-metrics">
        <span class="metric" :class="latencyClass(interimMetrics.inference_ms)">
          ⏱ 推理 {{ interimMetrics.inference_ms }}ms
        </span>
        <span class="metric">
          🎵 音频 {{ interimMetrics.audio_duration_ms }}ms
        </span>
      </div>
    </div>

    <!-- Final results -->
    <div v-for="r in results" :key="r.id" class="result-item" :class="r.type">
      <div class="result-main">
        <span class="time">{{ r.timestamp }}</span>
        <span class="text">{{ r.text }}</span>
      </div>
      <div v-if="r.metrics" class="metrics-row">
        <span class="metric" :class="latencyClass(r.metrics.inference_ms)">
          ⏱ 推理 {{ r.metrics.inference_ms }}ms
        </span>
        <span class="metric">
          🎵 音频 {{ r.metrics.audio_duration_ms }}ms
        </span>
        <span class="metric">
          📦 端到端 {{ r.metrics.e2e_ms }}ms
        </span>
      </div>
    </div>

    <!-- Latency stats footer (visible during recording) -->
    <div v-if="isRecording && latencyStats.count > 0" class="latency-panel">
      <div class="panel-title">📊 延迟统计</div>
      <div class="panel-grid">
        <div class="stat-item">
          <span class="stat-label">推理次数</span>
          <span class="stat-value">{{ latencyStats.count }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">平均推理</span>
          <span class="stat-value" :class="latencyClass(latencyStats.avg)">{{ latencyStats.avg }}ms</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">最大推理</span>
          <span class="stat-value" :class="latencyClass(latencyStats.max)">{{ latencyStats.max }}ms</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">末次推理</span>
          <span class="stat-value" :class="latencyClass(latencyStats.lastInferenceMs)">{{ latencyStats.lastInferenceMs }}ms</span>
        </div>
      </div>
    </div>

    <div ref="bottom"></div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps({
  results: { type: Array, default: () => [] },
  interimText: { type: String, default: '' },
  interimMetrics: { type: Object, default: null },
  isRecording: Boolean,
  latencyStats: {
    type: Object,
    default: () => ({ count: 0, avg: 0, max: 0, lastInferenceMs: 0 }),
  },
})

const bottom = ref(null)

function latencyClass(ms) {
  if (!ms && ms !== 0) return ''
  if (ms < 200) return 'latency-fast'
  if (ms < 500) return 'latency-ok'
  return 'latency-slow'
}

watch(() => props.results.length, async () => {
  await nextTick()
  bottom.value?.scrollIntoView({ behavior: 'smooth' })
})
</script>

<style scoped>
.transcript {
  min-height: 200px;
  max-height: 60vh;
  overflow-y: auto;
  padding: 8px 0;
  scroll-behavior: smooth;
}

.empty {
  text-align: center;
  padding: 48px 16px;
  color: #64748b;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 8px; }
.empty-hint { font-size: 0.85rem; margin-top: 4px; color: #475569; }

.result-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 10px 12px;
  margin: 4px 0;
  border-radius: 8px;
  background: #1e293b;
  animation: fadeIn 0.3s ease;
}
.result-item.final { border-left: 3px solid #22c55e; }
.result-item.interim { border-left: 3px solid #eab308; background: #1e293b88; }
.result-item.listening { border-left: 3px solid #3b82f6; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

.result-main {
  display: flex;
  gap: 10px;
  align-items: baseline;
}

.badge {
  font-size: 0.7rem;
  padding: 1px 6px;
  border-radius: 4px;
  white-space: nowrap;
  font-weight: 600;
}
.interim-badge {
  background: #eab30833;
  color: #eab308;
}

.time {
  color: #64748b;
  font-size: 0.75rem;
  white-space: nowrap;
  padding-top: 2px;
  min-width: 48px;
}

.text {
  flex: 1;
  line-height: 1.5;
  word-break: break-all;
}

.typing::after {
  content: '▌';
  animation: blink 0.8s infinite;
  color: #eab308;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

/* ── Metrics row ──────────────────────────────────── */
.metrics-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 4px;
  padding-top: 6px;
  border-top: 1px solid #334155;
}

.live-metrics {
  border-top-color: #eab30844;
}

.metric {
  font-size: 0.72rem;
  color: #64748b;
  padding: 1px 6px;
  border-radius: 4px;
  background: #0f172a;
}

.metric.latency-fast { color: #22c55e; }
.metric.latency-ok { color: #eab308; }
.metric.latency-slow { color: #ef4444; font-weight: 600; }

/* ── Latency stats panel ──────────────────────────── */
.latency-panel {
  margin: 12px 0;
  padding: 12px;
  background: #1e293b;
  border-radius: 8px;
  border: 1px solid #334155;
}

.panel-title {
  font-size: 0.8rem;
  color: #94a3b8;
  margin-bottom: 8px;
}

.panel-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.stat-label {
  font-size: 0.7rem;
  color: #64748b;
}

.stat-value {
  font-size: 1.1rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: #e2e8f0;
}
.stat-value.latency-fast { color: #22c55e; }
.stat-value.latency-ok { color: #eab308; }
.stat-value.latency-slow { color: #ef4444; }

@media (max-width: 480px) {
  .transcript { max-height: 50vh; }
  .result-item { padding: 8px 10px; }
  .panel-grid { grid-template-columns: 1fr 1fr; }
}
</style>
