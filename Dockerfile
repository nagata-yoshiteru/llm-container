FROM eugr/spark-vllm:latest

RUN python3 -m pip install --no-cache-dir -U vllm-gguf-plugin

