FROM eugr/spark-vllm:nightly-20260710

RUN python3 -m pip install --no-cache-dir -U vllm-gguf-plugin

