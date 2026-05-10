# Audio Server

流式 ASR / TTS 服务，支持 **sherpa-onnx**（轻量）和 **Qwen3-TTS**（高质量）两种 TTS 后端。

## 前置要求

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/)（包管理）

## 安装

```bash
git clone <repo> && cd audio-server
uv sync
```

---

## ASR 模式 — 语音识别服务

以 ASR 模式启动，通过 WebSocket 接收麦克风流式音频，实时返回识别结果。

### 启动服务

```bash
# 使用 .env 配置
cp .env.asr .env
uv run python -m audio_server.main

# 或直接通过环境变量
ASR_ENABLED=true TTS_ENABLED=false \
  SHERPA_ONNX_MODEL_TYPE=qwen3_asr \
  uv run python -m audio_server.main
```

首次启动自动下载模型到 `~/.cache/audio-server/sherpa-onnx/`。

### 支持的 ASR 模型

| 模型类型 | 语言 | 大小 | 设置 |
|---------|------|------|------|
| **SenseVoice int8** | zh/en/ja/ko/yue | ~228MB | `SHERPA_ONNX_MODEL_TYPE=sense_voice` |
| **FunASR Nano int8** | zh/en/ja | ~150MB | `SHERPA_ONNX_MODEL_TYPE=funasr_nano` |
| **Qwen3-ASR 0.6B int8** | 多语言 | ~600MB | `SHERPA_ONNX_MODEL_TYPE=qwen3_asr` |
| **Moonshine V2** | zh/en/es | ~50MB | `SHERPA_ONNX_MODEL_TYPE=moonshine_v2` |

### 调用服务

#### WebSocket 流式识别（浏览器/麦克风）

连接 `ws://<host>:8000/ws/transcribe`，发送 Int16 PCM（16kHz 单声道，每帧 30ms = 960 bytes）。

```bash
# 命令行测试客户端
uv run python scripts/test_client.py --mic

# 或使用录好的 WAV 文件
uv run python scripts/test_client.py --file test.wav
```

服务端返回 JSON 消息：

| type | 说明 |
|------|------|
| `speech_start` | VAD 检测到语音开始 |
| `interim` | 中间结果（每 ~400ms） |
| `final` | 最终结果（语音结束） |

#### HTTP 文件转录

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@speech.wav" \
  -F "response_format=json"
```

返回：

```json
{"text": "今天天气不错", "duration_sec": 2.1, "inference_ms": 85}
```

---

## TTS 模式 — 语音合成服务

以 TTS 模式启动，通过 HTTP API 合成语音。支持 **sherpa-onnx** 和 **Qwen3-TTS** 两种后端，通过 `TTS_MODEL` 自动选择。

### 模式选择

| 后端 | TTS_MODEL 值 | 特点 |
|------|-------------|------|
| **sherpa-onnx** | `matcha-icefall-zh-en`（默认） | 轻量、CPU 可跑、首次自动下载 |
| **Qwen3-TTS** | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 高质量多语言、需 GPU、需额外安装 |

### Sherpa-onnx 模式

轻量级 TTS，适合 CPU 和开发板。

#### 安装

```bash
uv sync
```

#### 启动服务

```bash
TTS_ENABLED=true ASR_ENABLED=false \
  TTS_MODEL=matcha-icefall-zh-en \
  TTS_PROVIDER=cpu \
  uv run python -m audio_server.main
```

首次启动自动从 GitHub 下载模型到 `~/.cache/audio-server/sherpa-onnx/`。

#### 调用服务

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，世界！今天天气不错。",
    "voice": "0",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

参数说明：

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | string | 合成文本（1-2000 字符） |
| `voice` | string | 说话人 ID（默认 `"0"`） |
| `response_format` | string | `wav` / `pcm` / `mp3` |
| `speed` | float | 语速 0.5-2.0 |

输出：**16kHz 单声道** WAV。

---

### Qwen3-TTS 模式

高质量多语言 TTS（中/英/日/韩/德/法/俄/葡/西/意），支持 9 种预置发音人。

#### 环境要求

| 组件 | 要求 |
|------|------|
| GPU | 0.6B: ≥4GB VRAM, 1.7B: ≥8GB VRAM |
| CUDA | ≥ 11.8 |

#### 安装

```bash
# 1. 安装基础依赖
uv sync

# 2. 安装 Qwen3-TTS 可选依赖
uv sync --extra qwen-tts

# 3. 安装 FlashAttention 2（推荐，加速推理）
uv pip install ninja
uv pip install -U flash-attn --no-build-isolation

# 4. 验证
uv run python -c "from qwen_tts import Qwen3TTSModel; print('ok')"
```

> 内存不足时 flash-attn 编译可能 OOM：`MAX_JOBS=1 uv pip install -U flash-attn --no-build-isolation` 按照8G内存/job计算MAX_JOBS

#### 启动服务

```bash
TTS_ENABLED=true ASR_ENABLED=false \
  TTS_MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" \
  TTS_QWEN3_DEVICE="cuda:0" \
  TTS_QWEN3_DTYPE="bfloat16" \
  uv run python -m audio_server.main
```

首次启动自动从 HuggingFace 下载模型（0.6B ~600MB，1.7B ~3GB）到 `~/.cache/huggingface/hub/`。

#### 调用服务

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    "voice": "Vivian",
    "language": "Chinese",
    "response_format": "wav"
  }' \
  --output speech.wav
```

参数说明：

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | string | 合成文本 |
| `voice` | string | 发音人名称（见下表） |
| `language` | string | `Chinese` / `English` / `Japanese` / `Korean` / `Auto` |
| `instruct` | string | 语气指令（仅 1.7B 模型支持） |
| `response_format` | string | `wav` / `pcm` / `mp3` |

输出：**24kHz 单声道** WAV。

#### 预置发音人

| 发音人 | 描述 | 母语 |
|--------|------|------|
| Vivian | 明亮略带锋芒的年轻女声 | 中文 |
| Serena | 温婉轻柔的年轻女声 | 中文 |
| Uncle_Fu | 低沉圆润的成熟男声 | 中文 |
| Dylan | 京腔青年男声，清澈自然 | 中文（北京） |
| Eric | 活泼的成都男声，略带沙哑明亮 | 中文（四川） |
| Ryan | 律动感强的男声 | 英文 |
| Aiden | 阳光美式男声，清晰中音 | 英文 |
| Ono_Anna | 俏皮的日系女声 | 日文 |
| Sohee | 温暖的韩系女声 | 韩文 |

#### 1.7B 指令控制（可选）

仅 `Qwen3-TTS-12Hz-1.7B-CustomVoice` 支持 `instruct` 参数：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "今天天气真不错。",
    "voice": "Vivian",
    "language": "Chinese",
    "instruct": "用特别愤怒的语气说",
    "response_format": "wav"
  }' \
  --output speech.wav
```

#### 手动下载模型

```bash
# HuggingFace CLI
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --local-dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice

# ModelScope（中国大陆推荐）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --local_dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
```

#### 性能测试 (RTF)

```bash
# 0.6B
uv run python scripts/benchmark_rtf.py

# 1.7B
uv run python scripts/benchmark_rtf.py \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

# CPU
uv run python scripts/benchmark_rtf.py --device cpu --dtype float32
```

---

## 配置参考

通过 `.env` 文件或环境变量配置：

```bash
cp .env.example .env
```

### 核心配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASR_ENABLED` | `true` | 启用 ASR |
| `TTS_ENABLED` | `false` | 启用 TTS |
| **ASR 模型** | | |
| `SHERPA_ONNX_MODEL` | `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` | ASR 模型名称 |
| `SHERPA_ONNX_MODEL_TYPE` | 自动检测 | `sense_voice` / `funasr_nano` / `qwen3_asr` / `moonshine_v2` |
| `SHERPA_ONNX_NUM_THREADS` | `2` | 推理线程数 |
| `SHERPA_ONNX_PROVIDER` | `cpu` | `cpu` / `cuda` |
| **TTS sherpa-onnx** | | |
| `TTS_MODEL` | `matcha-icefall-zh-en` | 以 `Qwen/Qwen3-TTS-12Hz-` 开头自动切换 Qwen3 后端 |
| `TTS_PROVIDER` | `cpu` | 推理后端 |
| `TTS_NUM_THREADS` | `4` | 推理线程数 |
| `TTS_SPEED` | `1.0` | 语速 0.5-2.0 |
| **TTS Qwen3-TTS** | | |
| `TTS_QWEN3_DEVICE` | `cuda:0` | 推理设备 |
| `TTS_QWEN3_DTYPE` | `bfloat16` | `float16` / `bfloat16` / `float32` |
| `TTS_QWEN3_SPEAKER` | `Vivian` | 默认发音人 |
| `TTS_QWEN3_LANGUAGE` | `auto` | 默认语种 |
| **服务器** | | |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `MAX_CONNECTIONS` | `4` | 最大并发连接数 |

---

## 项目结构

```
src/audio_server/
├── main.py          # FastAPI 入口 + ASR WebSocket + TTS HTTP
├── config.py        # 配置管理
├── asr_engine.py    # ASR 引擎 (sherpa-onnx)
├── tts_engine.py    # TTS 引擎 (sherpa-onnx / Qwen3-TTS)
├── vad.py           # 语音活动检测
├── audio_buffer.py  # 环形缓冲区
├── session.py       # WebSocket 会话管理
└── static/          # 前端文件（可选）
scripts/
├── test_client.py   # ASR 命令行测试客户端
└── benchmark_rtf.py # Qwen3-TTS RTF 性能测试
```

---

## macOS 注意

sherpa-onnx 缺少 onnxruntime 动态库，需手动修复：

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

### 延迟指标

| 颜色 | `inference_ms` |
|------|---------------|
| 🟢 | `< 200ms` |
| 🟡 | `< 500ms` |
| 🔴 | `≥ 500ms` |
