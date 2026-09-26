#!/usr/bin/env bash
set -e

GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:8000}"

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  list                  List all models currently installed and their benchmarks"
    echo "  pull <model_name>     Download and install a new model from the registry"
    echo "  rm <model_name>       Remove a model to free disk space"
    echo "  test [model_name]     Run a test decision against a model (default: decider)"
    echo ""
    echo "Examples:"
    echo "  $0 list"
    echo "  $0 pull qwen3guard"
    echo "  $0 pull gliclass"
    echo "  $0 test decider"
    exit 1
}

CMD="$1"
shift || true

case "$CMD" in
    list|ls)
        echo "Fetching installed models from $GATEWAY_URL..."
        curl -s "$GATEWAY_URL/v1/models" | jq -r '
          ["MODEL ID", "FPS", "LATENCY", "PARAM SIZE", "REASONING TIER"],
          ["--------", "---", "-------", "----------", "--------------"],
          (.data[] | [.id, (.throughput_fps | tostring), ((.latency_ms | tostring) + "ms"), .parameter_size, .reasoning_tier])
          | @tsv' | column -t -s $'\t'
        ;;
    pull)
        MODEL="$1"
        if [ -z "$MODEL" ]; then
            echo "Error: Model name required (e.g. $0 pull qwen3guard)"
            exit 1
        fi
        echo "Requesting download of model '$MODEL' via Gateway..."
        curl -s -X POST "$GATEWAY_URL/v1/models/pull" \
          -H "Content-Type: application/json" \
          -d "{\"model\": \"$MODEL\", \"stream\": true}"
        echo ""
        echo "Done!"
        ;;
    rm|delete)
        MODEL="$1"
        if [ -z "$MODEL" ]; then
            echo "Error: Model name required (e.g. $0 rm <model>)"
            exit 1
        fi
        echo "Removing model '$MODEL'..."
        curl -s -X DELETE "$GATEWAY_URL/v1/models/$MODEL" | jq .
        ;;
    test)
        MODEL="${1:-decider}"
        echo "Running test on '$MODEL'..."
        curl -s -X POST "$GATEWAY_URL/v1/systemone" \
          -H "Content-Type: application/json" \
          -d "{
            \"model\": \"$MODEL\",
            \"state\": \"The primary database connection pool is exhausted!\",
            \"questions\": {
              \"is_critical\": {\"type\": \"noul\", \"criteria\": {\"yes\": \"Critical production issue\", \"no\": \"Minor warning\"}}
            }
          }" | jq .
        ;;
    *)
        usage
        ;;
esac
