#!/bin/bash
# Send two image requests to the frontend with source/1.jpeg (672x588); the client sends the raw image.
# 672x588 > 512*512, so it is tiled into 2 tiles + the whole image = 3 items (256 placeholders each):
#   1. single image: 768 placeholders, grid [1,32,96]
#   2. same image twice: 6 items, 1,536 placeholders
# Check the encode log line `[fake_model] grid=... items=... total=...`.
# The embedding cache is off by default, so reusing one image is fine.
set -e
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    echo "Missing $SCRIPT_DIR/.env"
    exit 1
fi
set -a; source "$SCRIPT_DIR/.env"; set +a
HOST=${HOST:-127.0.0.1:$DYN_HTTP_PORT}
MODEL=${MODEL:-$SERVED_MODEL_NAME}
MAX_TOKENS=${MAX_TOKENS:-50}
IMG_DIR="$SCRIPT_DIR/source"

SERVER_PID=""
trap '[[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null' EXIT

python3 -m http.server "$IMG_PORT" --bind 127.0.0.1 --directory "$IMG_DIR" >/dev/null 2>&1 &
SERVER_PID=$!
sleep 1

IMG="http://127.0.0.1:$IMG_PORT/1.jpeg"

request() {  # $1: label; rest: image URLs
    local desc=$1
    shift
    local images=""
    for u in "$@"; do
        images="$images,{\"type\": \"image_url\", \"image_url\": {\"url\": \"$u\"}}"
    done
    echo "== $desc"
    curl -s --max-time "${CURL_TIMEOUT:-300}" "http://$HOST/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"描述这张图片\"}$images]}],
            \"max_tokens\": $MAX_TOKENS
        }" -w "\nHTTP %{http_code}\n" || echo "curl exit code $?"
}

request "single image" "$IMG"
request "same image twice" "$IMG" "$IMG"
