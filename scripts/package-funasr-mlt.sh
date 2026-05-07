#!/usr/bin/env bash
# Package exported FunASR-MLT-Nano ONNX files into multi-part tarballs
# suitable for uploading to a GitHub Release.
#
# Usage:
#   cd /path/to/FunASR-nano-onnx
#   bash /path/to/asr-server/scripts/package-funasr-mlt.sh
#
# Output files (each < 2GB for GitHub Release limits):
#   sherpa-onnx-funasr-mlt-nano-int8-<date>/
#   ├── encoder_adaptor.int8.onnx          # ~40MB, direct file
#   ├── embedding.int8.onnx                # ~160MB, direct file
#   ├── llm_int8.tar.bz2                   # ~600MB, archive
#   └── Qwen3-0.6B.tar.bz2                 # ~1MB, archive
#
# These are uploaded individually to GitHub Release.
# The model name prefix is used by the downloader to reconstruct
# the full URL: {base_url}/{model_name}_{part_name}

set -euo pipefail

MODELS_DIR="${1:-models}"
DATE="${2:-$(date -u +%Y-%m-%d)}"
MODEL_NAME="sherpa-onnx-funasr-mlt-nano-int8-${DATE}"
OUTDIR="/tmp/${MODEL_NAME}"

echo "=== Packaging FunASR-MLT-Nano ==="
echo "Models dir:  ${MODELS_DIR}"
echo "Output dir:  ${OUTDIR}"
echo "Model name:  ${MODEL_NAME}"
echo ""

mkdir -p "${OUTDIR}"

# 1. Single files (direct copy)
for f in encoder_adaptor.int8.onnx embedding.int8.onnx; do
  src="${MODELS_DIR}/${f}"
  if [ -f "${src}" ]; then
    cp -v "${src}" "${OUTDIR}/"
  else
    echo "WARNING: ${src} not found, skipping"
  fi
done

# 2. LLM — archive the directory (or single file)
if [ -d "${MODELS_DIR}/llm_int8" ]; then
  tar -cjf "${OUTDIR}/llm_int8.tar.bz2" -C "${MODELS_DIR}" llm_int8/
  echo "Created ${OUTDIR}/llm_int8.tar.bz2"
elif [ -f "${MODELS_DIR}/llm.onnx" ]; then
  tar -cjf "${OUTDIR}/llm_int8.tar.bz2" -C "${MODELS_DIR}" llm.onnx
  echo "Created ${OUTDIR}/llm_int8.tar.bz2 (from llm.onnx)"
elif [ -f "${MODELS_DIR}/llm.int8.onnx" ]; then
  tar -cjf "${OUTDIR}/llm_int8.tar.bz2" -C "${MODELS_DIR}" llm.int8.onnx
  echo "Created ${OUTDIR}/llm_int8.tar.bz2 (from llm.int8.onnx)"
else
  echo "WARNING: no LLM model found"
fi

# 3. Tokenizer directory
TOK_DIR="Qwen3-0.6B"
if [ -d "${MODELS_DIR}/${TOK_DIR}" ]; then
  tar -cjf "${OUTDIR}/${TOK_DIR}.tar.bz2" -C "${MODELS_DIR}" "${TOK_DIR}/"
  echo "Created ${OUTDIR}/${TOK_DIR}.tar.bz2"
else
  echo "WARNING: ${TOK_DIR} not found"
fi

# 4. README
cat > "${OUTDIR}/README.md" <<EOF
Exported from FunAudioLLM/Fun-ASR-MLT-Nano-2512
Date: ${DATE}
EOF

echo ""
echo "=== Output files ==="
ls -lh "${OUTDIR}/"

echo ""
echo "=== File sizes (for GitHub Release) ==="
for f in "${OUTDIR}"/*; do
  size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)
  echo "  $(basename "$f"): $(numfmt --to=iec $size 2>/dev/null || echo ${size} bytes)"
  if [ "${size:-0}" -gt 2000000000 ]; then
    echo "    ⚠️  EXCEEDS 2GB — split further!"
  fi
done

echo ""
echo "=== Upload commands ==="
echo "gh release upload asr-models ${OUTDIR}/*"
echo ""
echo "=== Done ==="
