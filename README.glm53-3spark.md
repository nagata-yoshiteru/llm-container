# dealignai/GLM-5.3-Flash-UNCENSORED-FP8 を 3 台の DGX Spark で動かす

対象: `https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-FP8`
(native block-FP8 128x128 / 62 shard / 約 328GB = 306GiB / 320B params / MTP head 内蔵)

ハード: DGX Spark (GB10, SM121) x3、**光スイッチ経由のフルメッシュ (同一 L2)**。RING ではない。
rootful Docker 必須 (`sudo docker compose ...`)。

---

## 1. 結論: TP=1 + DP=3 + EP=3

3 台で attention を割る (`--tensor-parallel-size 3`) は**素の vllm では起動しない**。
glm5_next の形状が 3 で割れないため:

| 場所 | 内容 |
| --- | --- |
| `vllm/models/glm5next/common/attention.py:452-453` | `assert num_heads % tp_size == 0` (num_attention_heads=64) |
| `vllm/models/glm5next/common/kda.py:219` | `assert self.num_heads % self.tp_size == 0` (KDA heads=64) |
| `vocab_parallel_embedding` | `vocab_size=154880` も 3 で割れない |

通すには「head をゼロパディングしたチェックポイント + ロード時 pad overlay +
sparse-MLA カーネルの power-of-2 対応」が要る (kindling の `experimental/tp3` 一式)。
この repo は**設定だけで完結**させるため、その路線は採らない。

PP=3 も塞がっている。glm5_next は `make_empty_intermediate_tensors` を実装しておらず、
vllm 側で PP が明示的にゲートされている:

- `vllm/models/glm5next/common/model.py:788` … `# PP is gated off for GLM-5.3-Flash`
- `vllm/models/glm5next/common/model.py:1169` … `does not implement make_empty_intermediate_tensors`

残るのが **DP=3 + EP=3**。`vllm/model_executor/layers/fused_moe/config.py:1189-1233` のとおり
`--enable-expert-parallel` を付けると `ep_size = DP x TP = 3` になり、**MoE 側の `tp_size` は 1 に落ちる**。
結果:

- routed experts: `288 / 3 = 96` ずつ (EP の `num_experts % ep_size == 0` を満たす)
- `moe_intermediate_size=2048` は分割されない → **パディング不要**
- attention / embedding は TP=1 なので割り切れなくても assert に当たらない

代償は attention / dense / embed / vision が 3 ノードに複製されること (~+9GiB/rank)。

---

## 2. メモリ (★ この構成の最大リスク)

| 項目 | 1 ノードあたり |
| --- | --- |
| routed experts (304.4GB / 3) | ~94.5 GiB |
| attention / dense / embed / vision (TP=1 で複製) | ~10 GiB |
| MTP layer | ~2.4 GiB |
| **重み 小計** | **~107 GiB** |
| CUDA ctx / NCCL / vLLM プロセス | ~7 GiB |
| 合計 | **~114 GiB** |

GB10 の unified memory は 121.7GiB。残りは 6〜7GiB しかない。
`GPU_MEMORY_UTILIZATION` は **0.92 を起点に 0.90〜0.95 で調整**する。
低すぎると KV が足りず起動できず、高すぎると NVRM OOM で warmup 中に死ぬ
(GB10 ドライバは MemAvailable ではなく MemFree を見る)。

`MAX_MODEL_LEN` は OOM したら 131072 → 65536 と下げる。MLA の KV は
1トークンあたり約 6KB (fp8) なので KV 自体は軽い。**重いのは重み**。

`LANGUAGE_MODEL_ONLY=1` (既定) で vision tower を落とすと ~1.2GiB/rank 浮く。
SM12x の sparse-MLA prefill には画像幅のカーネルが無いので、text-only が安全。

---

## 3. 手順

### 3.1 重みを 3 台とも同じパスに置く

```bash
./scripts/fetch-model.sh                 # 62 shard / 約 328GB
# 3 台とも同じパス (./models/GLM-5.3-Flash-UNCENSORED-FP8) に置くこと。
# head で落として rsync で配ってもよい。
```

### 3.2 3 台とも preflight

```bash
./scripts/preflight.sh
```

特に見るのは `== RoCE リンク ==` (3 peer 全部に DF ping が通るか)、
`== RDMA / GID ==`、`== メモリ ==`。赤があるまま起動すると数分後に NCCL か OOM で死ぬ。

### 3.3 head の IP を .env に合わせる

`.env` の `HEAD_ROCE_IP` / `WORKER1_ROCE_IP` / `WORKER2_ROCE_IP` を
この機体の netplan に合わせて書き換える。3 台とも同じ `.env` を置く。
**head が DP coordinator なので `HEAD_ROCE_IP` が最重要。**
(worker の rpc はすべて head に繋ぎに行く)

### 3.4 起動 (worker2 → worker1 → head の順)

```bash
# worker2
ssh -t <worker2> 'cd ~/repos/llm-container && sudo docker compose --profile worker2 up -d'
# worker1
ssh -t <worker1> 'cd ~/repos/llm-container && sudo docker compose --profile worker1 up -d'
# head (DP coordinator / API サーバ)
sudo docker compose --profile head up -d
```

### 3.5 smoke test

```bash
# ロードは 328GB + JIT で 10〜40 分かかる。ログを追う:
sudo docker compose --profile head logs -f head      # "Application startup complete"
sudo docker compose --profile worker1 logs -f worker1

# 3 rank が揃ったか (head 側で)
ss -ltnp | grep 13345                                # DP coordinator が listen している

# 疎通 (head の HOST_PORT)
curl -s http://192.168.0.1:8910/v1/models
curl -s http://192.168.0.1:8910/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash-uncensored","messages":[{"role":"user","content":"1+1="}],"max_tokens":32}'
```

### 3.6 停止

```bash
sudo docker compose --profile head down       # 各ノードで自分の profile を down
```

---

## 4. トラブルシューティング

| 症状 | 見どころ |
| --- | --- |
| `Application startup` が出ないまま無言 | `VLLM_ENGINE_READY_TIMEOUT_S=3600` を確認。328GB のロードは長い。`py-spy dump --native <EngineCore pid>` |
| worker が rendezvous で止まる | head の `ss -ltnp \| grep 13345`。`--data-parallel-address` に head の rail0 IP が入っているか |
| NCCL `unhandled system error` | `NCCL_DEBUG=INFO`。3 台の MTU 1500 と arp_ignore/arp_announce を確認 |
| warmup で NVRM OOM | `GPU_MEMORY_UTILIZATION` を 0.90 に、`MAX_MODEL_LEN` を 131072 に。事前に `sync && echo 3 > /proc/sys/vm/drop_caches` |
| `num_heads ... not divisible` | TP_SIZE が 1 以外になっている。compose の profile と .env を確認 |
| MoE backend の選択で落ちる | `VLLM_EXTRA_ARGS` に `--moe-backend marlin` か `triton` を足す。起動ログの `Using ... MoE backend` を必ず確認 |

---

## 5. まだ確認できていないこと (実機で最初に見るべき点)

1. **SM121 で block-FP8 の MoE カーネルが選べるか。**
   このイメージ (v8) は NVFP4 レシピ向けにビルドされており、NVFP4 は `--moe-backend marlin`
   が known-good。FP8 blockwise (128x128) の MoE GEMM が SM121 で動くかは未確認。
   起動ログの `Using ... MoE backend` と、最初の decode が NaN にならないかを必ず見ること。
   動かない場合は 4bit 量子 (NVFP4) に逃げるのが現実的。
2. **DP=3 + EP=3 の multi-node internal DP がこの vllm で通るか。**(公式ドキュメントの
   `--data-parallel-size-local` / `--data-parallel-start-rank` / `--data-parallel-address`
   の形。`--nnodes` を使う形に切り替える余地もある)
3. **メモリが本当に収まるか。** 重み ~107GiB/rank は 121.7GiB に対して限界値。
   収まらなければ 4 台 `--tensor-parallel-size 4` (76.5GiB/rank) が本線。
   TP=4 なら num_heads=64/4=16、vocab=154880/4=38720、moe_i=2048/4=512 がすべて割り切れ、
   **パディングも overlay も不要**。
