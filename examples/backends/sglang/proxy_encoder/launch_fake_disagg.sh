#!/bin/bash
# Based on launch/multimodal_disagg.sh: E/P/D on three GPUs, encode worker replaced by the fake encoder.
# Activate the env first. Weights are generated on the first encode start if missing.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    echo "Missing $SCRIPT_DIR/.env"
    exit 1
fi
set -a; source "$SCRIPT_DIR/.env"; set +a
source "$SCRIPT_DIR/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL_NAME="$QWEN_MODEL_PATH"  # SERVED_MODEL_NAME also comes from .env; both can be overridden below
CHAT_TEMPLATE="qwen2-vl"
FRONTEND_DECODING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --served-model-name)
            SERVED_MODEL_NAME=$2
            shift 2
            ;;
        --chat-template)
            CHAT_TEMPLATE=$2
            shift 2
            ;;
        --frontend-decoding)
            FRONTEND_DECODING=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <path>                 (default: $MODEL_NAME, QWEN_MODEL_PATH in .env)"
            echo "  --served-model-name <name>     (default: $SERVED_MODEL_NAME, SERVED_MODEL_NAME in .env)"
            echo "  --chat-template <template>     (default: $CHAT_TEMPLATE)"
            echo "  --frontend-decoding            Decode images in the Rust frontend"
            echo "Env: settings are read from $SCRIPT_DIR/.env: QWEN_MODEL_PATH, SERVED_MODEL_NAME,"
            echo "     DYN_DISCOVERY_BACKEND, DYN_MM_ALLOW_INTERNAL,"
            echo "     SGLANG_HOST_IP, UCX_TLS, *_WORKER_GPU, ports (DYN_HTTP_PORT, DYN_SYSTEM_PORT1/2, DYN_DISAGG_BOOTSTRAP_PORT,"
            echo "     DYN_PREFILL_NCCL_PORT, DYN_DECODE_NCCL_PORT, IMG_PORT),"
            echo "     FAKE_ENABLE_TILING, FAKE_ENABLE_TILING_MULTI_IMAGE, FAKE_MAX_TILES, FAKE_MAX_TILES_MULTI_IMAGE"
            echo "     DYN_FILE_KV is set to $SCRIPT_DIR/tmp/dynamo_kv by this script; the directory is wiped on every launch"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

ENCODE_EXTRA_ARGS=""
if [[ "$FRONTEND_DECODING" == "true" ]]; then
    ENCODE_EXTRA_ARGS="--frontend-decoding"
fi

unset DYN_SYSTEM_PORT

# File discovery dir; wiped on every launch (only this directory).
export DYN_FILE_KV="$SCRIPT_DIR/tmp/dynamo_kv"
rm -rf "$SCRIPT_DIR/tmp/dynamo_kv" && mkdir -p "$SCRIPT_DIR/tmp/dynamo_kv"

print_launch_banner --multimodal "Launching Disaggregated Multimodal E/P/D (fake encoder)" "$MODEL_NAME" "$DYN_HTTP_PORT"

python3 -m dynamo.frontend &

echo "Starting fake encode worker on GPU $DYN_ENCODE_WORKER_GPU..."
DYN_SYSTEM_PORT=$DYN_SYSTEM_PORT1 \
env CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU \
  DYN_FAKE_ENCODER=1 \
  python3 "$SCRIPT_DIR/encode_entry.py" \
  --enable-multimodal \
  --disaggregation-mode encode \
  --model-path "$MODEL_NAME" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --chat-template "$CHAT_TEMPLATE" \
  --skip-tokenizer-init \
  $ENCODE_EXTRA_ARGS &

echo "Starting prefill worker on GPU $DYN_PREFILL_WORKER_GPU..."
DYN_SYSTEM_PORT=$DYN_SYSTEM_PORT2 \
env CUDA_VISIBLE_DEVICES=$DYN_PREFILL_WORKER_GPU python3 -m dynamo.sglang \
  --enable-multimodal \
  --dedicated-mm-encoder \
  --model-path "$MODEL_NAME" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode prefill \
  --nccl-port "$DYN_PREFILL_NCCL_PORT" \
  --disaggregation-bootstrap-port "$DYN_DISAGG_BOOTSTRAP_PORT" \
  --host 0.0.0.0 \
  --disable-radix-cache \
  --disaggregation-transfer-backend nixl &

echo "Starting decode worker on GPU $DYN_DECODE_WORKER_GPU..."
env CUDA_VISIBLE_DEVICES=$DYN_DECODE_WORKER_GPU python3 -m dynamo.sglang \
  --enable-multimodal \
  --dedicated-mm-encoder \
  --model-path "$MODEL_NAME" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode decode \
  --nccl-port "$DYN_DECODE_NCCL_PORT" \
  --disaggregation-bootstrap-port "$DYN_DISAGG_BOOTSTRAP_PORT" \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl &

wait_any_exit
