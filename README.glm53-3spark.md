# dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 を 3 台の DGX Spark で動かす

対象: `https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4`
(modelopt NVFP4 / 121 shard / 約 181GiB / MTP head 内蔵・uncensored 済み)


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

代償は attention / KDA / dense / shared / embed が 3 ノードに複製されること (~17GiB/rank)。

---

## 2. メモリ

shard ヘッダを集計した実測値 (GiB)。vision は `--language-model-only` で載せない。

| 区分 | FP8 版 合計 | FP8 / rank | NVFP4 版 合計 | NVFP4 / rank |
| --- | --- | --- | --- | --- |
| routed experts (EP=3 で 1/3) | 283.57 | 94.52 | 159.47 | 53.16 |
| MTP experts (EP=3 で 1/3) | 6.75 | 2.25 | 3.80 | 1.27 |
| attention / KDA / dense / shared (複製) | 11.82 | 11.82 | 14.25 (BF16) | 14.25 |
| embed / lm_head (複製) | 2.36 | 2.36 | 2.36 | 2.36 |
| MTP その他 | 0.23 | 0.23 | 0.34 | 0.34 |
| **重み 小計** | | **~111** | | **~71.4** |

FP8 版は 2026-10-06 の head で以下のとおり落ちた。起動時の空きが 109.32GiB しかなく、
重み 111GiB/rank は GMU をどう調整しても入らない:

```
ValueError: Free memory on device cuda:0 (109.32/119.63 GiB) on startup is less than
desired GPU memory utilization (0.92, 110.06 GiB).
```

GB10 では OS 等で常時 ~10GiB 使われているので、`GPU_MEMORY_UTILIZATION` の上限は
実質 0.91。既定は 0.88 (= 105.3GiB。NVFP4 2-Spark レシピの実測値)。
NVFP4 なら 105.3 − 71.4 − CUDA ctx/NCCL ~7 ≈ **~27GiB が KV / アクティベーション**に回る。
`./scripts/preflight.sh` が「MemAvailable ≥ GMU × 119.63」を起動前に確認する。

`LANGUAGE_MODEL_ONLY=1` (既定) で vision tower を落とす。
SM12x の sparse-MLA prefill には画像幅のカーネルが無いので、text-only が安全。

---

## 3. 手順

### 3.1 重みを 3 台とも同じパスに置く

```bash
./scripts/fetch-model.sh                 # 121 shard / 約 181GiB
# 3 台とも同じパス (./models/GLM-5.3-Flash-UNCENSORED-NVFP4) に置くこと。
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
# ロードは 181GiB + JIT で 10〜40 分かかる。ログを追う:
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
| `Application startup` が出ないまま無言 | `VLLM_ENGINE_READY_TIMEOUT_S=3600` を確認。181GiB のロードは長い。`py-spy dump --native <EngineCore pid>` |
| worker が rendezvous で止まる | head の `ss -ltnp \| grep 13345`。`--data-parallel-address` に head の rail0 IP が入っているか |
| NCCL `unhandled system error` | `NCCL_DEBUG=INFO`。3 台の MTU 1500 と arp_ignore/arp_announce を確認 |
| warmup で NVRM OOM | `GPU_MEMORY_UTILIZATION` を 0.85 に、`MAX_MODEL_LEN` を 131072 に。事前に `sync && echo 3 > /proc/sys/vm/drop_caches` |
| `num_heads ... not divisible` | TP_SIZE が 1 以外になっている。compose の profile と .env を確認 |
| `Free memory ... is less than desired GPU memory utilization` | 起動時の空き不足。`GPU_MEMORY_UTILIZATION` を (空き / 119.63) 未満に下げる。3 台で一番空きの少ないノードに合わせる |
| MoE backend の選択で落ちる | `--moe-backend marlin` を外して vllm に選ばせる (`triton` 等)。起動ログの `Using ... MoE backend` を必ず確認 |

---

## 5. まだ確認できていないこと (実機で最初に見るべき点)

1. **modelopt 形式の NVFP4 + marlin + EP が SM121 で動くか。**
   このイメージ (v8) で実績があるのは RedHat の compressed-tensors 形式 NVFP4 を TP2 で
   動かした構成。dealignai 版は modelopt 形式 (`model-input-scales.safetensors` 付き)。
   起動ログの `Using ... MoE backend` と、最初の decode が NaN / 文字化けにならないかを必ず見ること。
2. **DP=3 + EP=3 の multi-node internal DP がこの vllm で通るか。**
   FP8 版の起動ログで DP coordinator 起動・`world_size=3` の NCCL 初期化・
   `Using AgRsAll2AllManager` までは確認済み (メモリで落ちたのはその後)。
3. **4 台にできるなら TP4 が本線。** num_heads=64/4=16、vocab=154880/4=38720、
   moe_i=2048/4=512 がすべて割り切れ、パディングも overlay も不要。

---

## 6. MTP の速度比較 (1〜2 リクエスト優先)

現在の uncensored NVFP4、TP=1 / DP=3 / EP=3、1M の上限はそのまま、
MTP の深さを比較できる。`.env` / `.env.example` は次の試験用に **k=3**。
FP8 化の効果を切り分けるときは MTP の k を同じ値に固定する。

```dotenv
MTP_NUM_SPECULATIVE_TOKENS=3
MAX_NUM_SEQS=2
ENFORCE_EAGER=0
VLLM_EXTRA_ARGS=--moe-backend marlin
```

`entrypoint.sh` が MTP と `FULL_DECODE_ONLY` の CUDA Graph 引数を生成する。
`mode=0` (torch.compile 無効) は従来どおり。capture size は各 DP rank の
`MAX_NUM_SEQS` と `k+1` (target 1 トークン + draft k トークン) から計算する。

| `MTP_NUM_SPECULATIVE_TOKENS` | `MAX_NUM_SEQS=2` の capture sizes | 用途 |
| --- | --- | --- |
| `1` | `[2,4]` | 従来の基準へ戻す |
| `2` | `[3,6]` | 比較用 |
| `3` | `[4,8]` | k=2 の測定後に比較 |

この変数を使う場合、`VLLM_*_ARGS` に `--speculative-config` や
`--compilation-config` を入れると起動前にエラーにする。`ENFORCE_EAGER=1` では
MTP だけを設定し、graph の引数は生成しない。変数を空にした場合は従来の
`VLLM_EXTRA_ARGS` 手動設定を使える (MTP も手動指定が必要)。

vLLM の一部の版では投機の深さが compile cache hash に含まれない
([vLLM #53366](https://github.com/vllm-project/vllm/issues/53366)) ため、
vLLM / TorchInductor キャッシュは自動で `glm53-mtp-k1` / `k2` / `k3` に分ける。
従来のキャッシュは削除しない。各 k の初回は再ウォームアップが必要。

### 反映と比較

全ノードでコードを pull し、同じ `.env` を置いてから全コンテナを再作成する。
稼働中のコンテナは `.env` を編集しただけでは変わらない。

1. head → worker1 → worker2 の順で、それぞれ `sudo docker compose --profile <役割> down`。
2. worker2 → worker1 → head の順で、それぞれ `sudo docker compose --profile <役割> up -d --force-recreate`。
3. 起動ログの `MTP k=3 decode graph sizes=[4,8]` とキャッシュの末尾 `glm53-mtp-k3` を確認。
4. KV 容量を確認する。k を増やすと MTP 用の状態等も増えるので、k=1 時の
   「1M が rank ごとに2本載る」という結果をそのまま流用しない。
5. ウォームアップ後、同じプロンプト・出力上限・sampling・reasoning effort で
   **1並列 / 2並列**を各3回以上測る。固定長の生成速度と、元の数学・コード問題の
   正答率／完了時間を別に比較する。MTP の採択率だけで採用を決めない。

`reasoning_effort` は今回変更しない。現在のテンプレートは未指定で `max`、
`enable_thinking=false` は反映されないので、thinking off の測定として記録しない。
k=3 を試す／k=1 に戻す場合は **全3ノード**の `MTP_NUM_SPECULATIVE_TOKENS` を
同じ値に変えて、上と同じ順で再作成する。

## 7. BF16 dense のロード後 FP8 化

次の試験設定は `MTP_NUM_SPECULATIVE_TOKENS=3`、`GLM53_DENSE_FP8=1`。
元の `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` を通常どおりロードし、
vLLM の weight 後処理が終わった時点で対象層だけを FP8 にする。
モデルファイル、モデル名、イメージの変更や checkpoint の再配布は不要。

| 対象 | ロード後の精度 |
| --- | --- |
| KDA の fused input / output projection | FP8 |
| MLA の fused qkv-a / q-b / output projection | FP8 |
| 最初の3層の dense MLP、shared experts の gate/up/down | FP8 |
| routed experts | 元の NVFP4 |
| router、indexer、MLA kv-b、KDA f-b/g-b、norm、embedding、LM head、MTP | 元の精度 |

現在の checkpoint では対象は **191 Linear、BF16 13.458 GiB/rank**。
FP8 化すると重みの保存量と対象 GEMM の重み読み出し量が約 **6.7 GiB/rank** 減る。
これは重みのバイト数による見積もりで、推論速度が2倍になるという意味ではない。
余ったメモリは起動時に KV へ再配分されうるため、KV 容量は再起動後に確認する。

方式は **W8A8**: weight は出力チャネル単位、activation はトークン単位の
動的 scale を持つ FP8 E4M3 で、CUTLASS scaled GEMM を使う。
静的 activation scale の校正データは不要。
[kindling の dense_fp8.py](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/c748079d45e6e070b2acb108a91edfe52f4a7747/experimental/snapshot/dense_fp8.py)
の FP8 経路を参考に、現在の TP=1 とモデルの対象層に限定した。
量子化誤差があるので、これは `lossless8` checkpoint の移植ではなく、
元の7問の正答率、長文の検索結果、ツール呼び出しを再確認する必要がある。

起動時に loader の SHA256 を確認し、現在の pinned `sm121-v8` 以外には適用しない。
小さな実 GPU テストで FP8 GEMM の数値誤差、ゼロ入力、prefill、CUDA Graph replay を
確認してから変換する。対象層の不足や非対応の shape / dtype も起動エラーにする。
これらはカーネルの動作確認であり、モデル全体の品質確認とは別。
起動ログでは次を確認する:

```text
[dense-fp8] verified v8 post-load hook installed
GLM53 dense FP8 kernel/graph check passed
GLM53 dense FP8: converted 191 target linears, saved ... GiB/rank
GLM53 dense FP8: preserving the MTP draft in its original precision
```

コンテナ内の loader に小さな hook を追加する。元 checkpoint は読み取り専用のまま。
キャッシュは `glm53-dense-fp8-v1/glm53-mtp-k3` 以下へ分離する。
BF16 に戻す場合は **全3ノード**の `GLM53_DENSE_FP8=0` にし、6節の順序で
コンテナを再作成する。再ロードで元の BF16 重みへ戻る。

### 比較ベンチ

`scripts/bench-glm53.py` は、同じ2種類のプロンプトを 256 tokens 固定で生成し、
1並列・2並列を各3回測る。ウォームアップは集計から除く。
`temperature=0`、`seed=42`、`reasoning_effort=max` を固定し、思考を含む
サーバの `completion_tokens` を使う。SSE イベント数を token 数として数えない。
EOS を無視して指定 token 数まで出力するため、これは速度テストであり品質テストではない。
他のリクエストがない状態で実行する。`--label` は手入力なので実際の起動設定と合わせる。

```bash
# BF16 / k=3 で起動した後
python3 scripts/bench-glm53.py --label bf16-mtp-k3 --output results/glm53-bf16-mtp-k3.json
# FP8 / k=3 で起動した後 (今回の .env)
python3 scripts/bench-glm53.py --label dense-fp8-mtp-k3 --output results/glm53-dense-fp8-mtp-k3.json
```
