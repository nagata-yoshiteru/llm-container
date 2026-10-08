# patches/moe-configs

vLLM の Triton fused_moe 用 tile 設定。compose がこのディレクトリを
`/opt/moe-configs` にマウントし、`VLLM_TUNED_CONFIG_FOLDER` で vLLM に読ませる
(同梱の `fused_moe/configs/` より先に探される。同梱分は上書きしない)。

| ファイル | 対象 |
| --- | --- |
| `E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json` | GLM-5.3-Flash FP8 (routed experts 288 / moe_intermediate 2048 を TP4 で割った N=512) on GB10 |

vLLM 同梱の GB10 用は E=256 / E=512 だけなので、これが無いと
`Using default MoE config. Performance might be sub-optimal!` で既定の tile になる。
効いているかは起動ログでこの警告が消えていることで確認する。

TP を変えると N が変わる (TP2 なら N=1024) ので、このファイルは使われなくなる。

出典: [knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)
`overlay/` (commit `5c918a7`、公式 FP8 チェックポイントを 4x DGX Spark TP4 で動かしたときの設定)。
Apache-2.0。
