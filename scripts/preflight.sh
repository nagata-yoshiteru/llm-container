#!/usr/bin/env bash
# Read-only DeepSeek TP=3 checks; never invokes Docker or sudo.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/preflight-deepseek41.py "$@"
