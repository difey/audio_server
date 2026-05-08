# Audio Server

流式 ASR / TTS 服务，面向嵌入式语音助手场景，支持 sherpa-onnx 引擎。

> 🎯 目标硬件：4核 A72，4GB RAM，ARM64 开发板

## 架构

```
🎤 麦克风 (浏览器/设备)
   → AudioWorklet / raw PCM
   → WebSocket Binary (Int16, 16kHz, 30ms/帧)
   → FastAPI 服务端
     → webrtcvad 语音检测
     → ASR 引擎 (sherpa-onnx)
     → 每 400ms 发 interim 中间结果
   → WebSocket JSON ← 转写结果 + 延迟指标
```

## 快速开始

### 前置要求

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/)（包管理）

### 安装 & 启动

```bash
# 1. 安装依赖
uv sync

# 2. 启动服务
uv run python -m audio_server.main

# 3. 连接 WebSocket ws://<host>:8000/ws/transcribe
```

首次启动会根据配置自动下载模型（详见下方后端说明）。WebSocket 端点即可用，**前端纯属可选**，`static/` 目录不存在时 HTTP 根路径返回 JSON 状态信息。

### 可选：Web 测试页面

```bash
# 构建前端（需要 Node.js ≥ 18）
cd frontend && npm install && npm run build && cd ..

# 浏览器打开 http://localhost:8000
```

### 命令行测试

```bash
# 转写本地 WAV 文件（16kHz 单声道）
uv run python scripts/test_client.py --file test.wav

# 使用麦克风实时测试
uv run python scripts/test_client.py --mic
```

---

## ASR 后端（sherpa-onnx）

基于 ONNX Runtime 的轻量引擎，速度最快，适合开发板。

| 模型类型 | 模型文件 | 语言 | 大小 | 设置 |
|---------|---------|------|------|------|
| **SenseVoice** | `model.int8.onnx` + `tokens.txt` | zh/en/ja/ko/yue | ~228MB | `SHERPA_ONNX_MODEL_TYPE=sense_voice` |
| **FunASR Nano** | `encoder_adaptor.int8.onnx` + llm + embedding + tokenizer | zh/en/ja | ~150MB | `SHERPA_ONNX_MODEL_TYPE=funasr_nano` |
| **Qwen3-ASR 0.6B** | `conv_frontend.onnx` + encoder + decoder + tokenizer | 多语言 | ~600MB | `SHERPA_ONNX_MODEL_TYPE=qwen3_asr` |
| **Moonshine V2** | `encoder_model.ort` + `decoder_model_merged.ort` + `tokens.txt` | zh/en/es | ~50MB | `SHERPA_ONNX_MODEL_TYPE=moonshine_v2` |

模型自动下载到 `~/.cache/audio-server/sherpa-onnx/`。

---

## 配置

通过 `.env` 文件或环境变量配置。所有可选项详见 `.env.example`：

```bash
cp .env.example .env
```

### 快速一览

| 分类 | 变量 | 默认值 | 说明 |
|------|------|--------|------|
| **功能开关** | `ASR_ENABLED` | `true` | 启用 ASR（设为 `false` 可只跑 TTS） |
| | `TTS_ENABLED` | `false` | 启用 TTS |
| **ASR 通用** | `SHERPA_ONNX_MODEL_TYPE` | 自动检测 | `sense_voice` / `funasr_nano` / `funasr_mlt_nano` / `qwen3_asr` / `moonshine_v2` |
| | `SHERPA_ONNX_MODEL` | `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` | 模型名称，首次自动下载 |
| | `SHERPA_ONNX_MODEL_DIR` | `""` | 本地模型目录（留空自动下载）|
| | `SHERPA_ONNX_NUM_THREADS` | `2` | 推理线程数 |
| | `SHERPA_ONNX_LANGUAGE` | `""` | 语种（空=自动，如 `zh`/`en`/`ja`） |
| | `SHERPA_ONNX_ITN` | `true` | 逆文本正则化 |
| **ASR VAD** | `VAD_MODE` | `0` | 灵敏度 0-3（0=最不敏感）|
| | `SILENCE_DURATION_MS` | `600` | 静音判定时长 (ms) |
| **TTS (sherpa-onnx)** | `TTS_MODEL` | `matcha-icefall-zh-en` | 模型名称，首次自动下载 |
| | `TTS_PROVIDER` | `cpu` | 推理后端 (`cpu` / `cuda`) |
| | `TTS_NUM_THREADS` | `4` | 推理线程数 |
| | `TTS_SPEED` | `1.0` | 语速 (0.5-2.0) |
| **TTS (Qwen3-TTS)** | `TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | HF 模型 ID（以 `Qwen/Qwen3-TTS-` 开头自动启用 Qwen3 后端）|
| | `TTS_QWEN3_DEVICE` | `cuda:0` | 推理设备 |
| | `TTS_QWEN3_DTYPE` | `bfloat16` | 数据类型 (`float16` / `bfloat16` / `float32`) |
| | `TTS_QWEN3_SPEAKER` | `Vivian` | 默认发音人 |
| | `TTS_QWEN3_LANGUAGE` | `auto` | 默认语种 (如 `Chinese` / `English` / `Auto`) |
| **服务器** | `HOST` | `0.0.0.0` | 监听地址 |
| | `PORT` | `8000` | 监听端口 |
| | `MAX_CONNECTIONS` | `4` | 最大并发连接数 |
| **缓存** | `MODEL_CACHE_DIR` | `~/.cache/audio-server` | 模型下载目录 |
---

## TTS (Text-to-Speech)

支持两种后端：**sherpa-onnx**（轻量，`matcha-icefall-zh-en` 等）和 **Qwen3-TTS**（高质量多语言，0.6B/1.7B）。

通过 `TTS_MODEL` 环境变量自动选择后端：
- 以 `Qwen/Qwen3-TTS-12Hz-` 开头 → Qwen3-TTS 后端
- 其他值（如 `matcha-icefall-zh-en`）→ sherpa-onnx 后端

### 启用 sherpa-onnx

```bash
export TTS_ENABLED=true
export TTS_PROVIDER=cpu   # x86 CUDA: export TTS_PROVIDER=cuda
```

### 启用 Qwen3-TTS（0.6B / 1.7B CustomVoice）

```bash
export TTS_ENABLED=true
export TTS_MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
export TTS_QWEN3_DEVICE="cuda:0"
export TTS_QWEN3_DTYPE="bfloat16"

# 可选依赖
pip install qwen-tts torch transformers
```

### 接口

#### Sherpa-onnx 请求

```
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "matcha-icefall-zh-en",
  "input": "你好，世界！今天天气不错。",
  "voice": "0",
  "response_format": "wav",
  "speed": 1.0
}
```

#### Qwen3-TTS 请求

```
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "input": "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
  "voice": "Vivian",
  "language": "Chinese",
  "response_format": "wav"
}
```

支持 9 种预置发音人：

| 发音人 | 描述 | 母语 |
|--------|------|------|
| Vivian | 明亮略带锋芒的年轻女声 | 中文 |
| Serena | 温婉轻柔的年轻女声 | 中文 |
| Uncle_Fu | 低沉圆润的成熟男声 | 中文 |
| Dylan | 京腔青年男声，清澈自然 | 中文 (北京) |
| Eric | 活泼的成都男声 | 中文 (四川) |
| Ryan | 律动感强的男声 | 英文 |
| Aiden | 阳光美式男声 | 英文 |
| Ono_Anna | 俏皮的日系女声 | 日文 |
| Sohee | 温暖的韩系女声 | 韩文 |

> 0.6B 模型不支持 `instruct` 参数。1.7B 模型支持通过 `instruct` 字段控制语气、情感：
> ```json
> { "instruct": "用特别愤怒的语气说" }
> ```

### curl 测试

```bash
# sherpa-onnx
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好，世界！"}' \
  --output speech.wav

# Qwen3-TTS 0.6B
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好，世界！", "voice": "Vivian", "language": "Chinese"}' \
  --output speech.wav
```

### 中英混合

```bash
# sherpa-onnx
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "我最近在学习machine learning，希望能有所建树。"}' \
  --output speech.wav
```

### 性能测试 (RTF)

```bash
# 安装依赖
pip install qwen-tts torch numpy

# 0.6B 模型 RTF 测试
uv run python scripts/benchmark_rtf.py

# 1.7B 模型 RTF 测试
uv run python scripts/benchmark_rtf.py \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

# CPU 测试
uv run python scripts/benchmark_rtf.py --device cpu --dtype float32
```

### 注意事项

- sherpa-onnx: `matcha-icefall-zh-en` 输出 **16kHz 单声道** WAV
- Qwen3-TTS: 输出 **24kHz 单声道** WAV
- Qwen3-TTS: 首次运行自动从 HuggingFace 下载模型（~600MB for 0.6B, ~3GB for 1.7B）
- Qwen3-TTS: 需要 GPU（至少 4GB VRAM for 0.6B, 8GB for 1.7B）
- `voice` 参数：sherpa-onnx 用数字 ID，Qwen3-TTS 用发音人名称
- `speed` 参数仅对 sherpa-onnx 生效

---

## WebSocket 协议

### 上行（Client → Server）

`Binary` — 裸 Int16 PCM，16kHz 单声道，**每帧 30ms（480 samples = 960 bytes）**。

### 下行（Server → Client）

`JSON Text`：

| type | 说明 | 关键字段 |
|------|------|----------|
| `speech_start` | VAD 检测到语音开始 | `timestamp` |
| `interim` | **中间结果**（每 ~400ms 推送） | `text`, `metrics` |
| `final` | 最终结果（语音段结束） | `text`, `vad`, `metrics`, `start`, `end` |
| `error` | 错误信息 | `message` |

### `vad` 字段（语音助手用）

| 值 | 含义 | 建议 |
|----|------|------|
| `true` | VAD 自然检测到静音 | **提交给 LLM** |
| `false` | 客户端断开强制 flush | **忽略**（可能没说完） |

```python
def on_final(msg):
    if msg["vad"]:
        send_to_llm(msg["text"])
```

> ⚠️ VAD 无法区分"说完了"和"思考停顿"。用户停顿 ≥ `SILENCE_DURATION_MS` 会触发 `final`（`vad: true`）。建议客户端收到后延迟 ~500ms 再提交 LLM，用户继续说则取消。

### `metrics` 延迟指标

每条 `interim` / `final` 都携带：

```json
{
  "type": "interim",
  "text": "今天天气",
  "metrics": {
    "inference_ms": 45.2,
    "audio_duration_ms": 800.0,
    "e2e_ms": 1250.5
  }
}
```

| 字段 | 说明 |
|------|------|
| `inference_ms` | 推理耗时 |
| `audio_duration_ms` | 送入 ASR 的音频总时长（含 VAD 静音填充） |
| `e2e_ms` | 从说话开始到收到结果的端到端延迟 |

---

## 项目结构

```
audio-server/
├── pyproject.toml                  # Python 项目配置 (uv)
├── .env                            # 环境变量配置
├── src/audio_server/
│   ├── main.py                     # FastAPI 入口 + WebSocket 端点
│   ├── config.py                   # 配置管理
│   ├── asr_engine.py               # ASR 引擎 (sherpa-onnx)
│   ├── vad.py                      # 语音活动检测 (webrtcvad)
│   ├── audio_buffer.py             # 环形音频缓冲区
│   ├── session.py                  # WebSocket 会话管理 + VAD 状态机
│   ├── tts_engine.py               # TTS 引擎 (sherpa-onnx / Qwen3-TTS)
│   └── static/                     # 构建好的前端文件 (可选)
├── frontend/                       # Vue 3 前端源码 (可选)
│   ├── src/components/
│   │   ├── AudioRecorder.vue       # 麦克风采集 (AudioWorklet)
│   │   └── TranscriptionDisplay.vue # 转写展示 + 延迟指标
│   └── public/audio-processor.js   # AudioWorkletProcessor
└── scripts/
    ├── test_client.py              # Python 命令行测试客户端
    ├── fix_vad.py                  # webrtcvad 兼容性修复
    └── benchmark_rtf.py            # Qwen3-TTS RTF 性能测试
```

---

## 开发板部署

开发板只需启动 WebSocket 服务，**不需要前端**。

```bash
# 1. 进入项目目录
cd audio-server

# 2. 安装依赖
uv sync

# 3. 配置 .env
SHERPA_ONNX_MODEL_TYPE=funasr_nano  # 或 sense_voice / qwen3_asr

# 4. 启动（首次自动下载模型）
uv run python -m audio_server.main
```

> 💡 无外网时，在有网机器上先下载模型到 `~/.cache/audio-server/sherpa-onnx/`，然后复制到开发板。

### macOS 注意

sherpa-onnx 的 PyPI 包缺少 onnxruntime 动态库，需手动修复：

```bash
ln -sf .venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.*.dylib \
       .venv/lib/python3.11/site-packages/sherpa_onnx/lib/libonnxruntime.1.24.4.dylib
```

**Linux / ARM64 上不需要此步骤。**

---

## 性能参考

### 推理耗时（Mac M 系列，2s 音频）

| 后端 | 模型 | 耗时 | 模型大小 |
|------|------|------|---------|
| sherpa-onnx | SenseVoice int8 | ~30ms | 228MB |
| sherpa-onnx | FunASR Nano int8 | ~50ms | 150MB |
| sherpa-onnx | Qwen3-ASR 0.6B int8 | ~200ms | 600MB |

### 开发板预估（4核 A53）

| 指标 | 参考值 |
|------|--------|
| 模型加载内存 | ~150-800MB（取决于模型） |
| 推理峰值内存 | ~300MB-1.2GB（取决于模型） |
| 端到端延迟 | ~500ms-3s（含 VAD 缓冲） |
| 最大并发 | 1-3 会话 |

---

## 延迟指标颜色参考

| 颜色 | `inference_ms` | 说明 |
|------|---------------|------|
| 🟢 | `< 200ms` | 正常 |
| 🟡 | `< 500ms` | 可接受 |
| 🔴 | `≥ 500ms` | 偏慢，考虑换小模型 |
