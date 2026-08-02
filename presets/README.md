# presets/

コンテキスト長ごとの **差分だけ**を持つ `.env` オーバーレイ。

`.env` (IP・パス・機密を含む / git 管理外) はそのままに、上書きしたいキーだけを
プリセット側に置く。`docker compose` は `--env-file` を複数受け取り、**後から
指定したファイルが勝つ**ので、こう重ねる:

```bash
sudo docker compose --env-file .env --env-file presets/256k.env --profile head up -d
```

worker 側も **同じ組み合わせ**で起動すること。片方だけ違うと TP のシャード形状が
合わずに rendezvous で死ぬ。

## 一覧

| プリセット | コンテキスト | KV | 確保スロット | 状態 |
|---|---|---|---|---|
| `128k.env` | 131,072 | 18GiB | 286,458 (2.19x) | **実績あり** |
| `256k.env` | 262,144 | 18GiB | 539,285 (2.06x) | **実績あり** (250K 入力で検証) |
| `1m.env` | 1,048,576 | 18GiB | 約 206 万 (外挿) | 未検証。計算上は入る |
| `1m-ds-mla.env` | 1,048,576 | 24GiB | ? | `1m.env` が KV 不足で落ちたとき用 |

**全プリセットで KV の確保量は 18GiB (1m-ds-mla を除く) = メモリ使用量は同一。**
`.env.example` の既定は `128k.env` と同じ内容なので、128K で使うならプリセット
指定は不要。

## 実測スループット (256K 構成)

DSpark k=7 / `MAX_NUM_SEQS=1` / prefix caching なし。

| 入力 | TTFT | prefill | decode | 合計 |
|---:|---:|---:|---:|---:|
| 3,922 | 2.07s | 1,891 t/s | 48.7 t/s | 4.0s |
| 15,958 | 7.55s | 2,114 t/s | 50.0 t/s | 9.6s |
| 63,984 | 31.4s | 2,040 t/s | 49.6 t/s | 34.0s |
| 131,008 | 68.1s | 1,924 t/s | 57.1 t/s | 70.3s |
| 199,920 | 110.9s | 1,802 t/s | 44.5 t/s | 113.5s |
| 249,952 | 145.1s | 1,723 t/s | 59.1 t/s | 147.1s |

**decode がコンテキスト長でほとんど劣化しない** (4K で 48.7、250K でも 59.1 t/s)。
prefill も 250K まで 15% しか落ちない。DeepSeek Sparse Attention が効いている。

`MemAvailable` は 250K 入力時で最小 6.7GiB。コールドスタート直後の 1 発目だけ
JIT ウォームアップで TTFT が 8 秒ほど余計にかかる。

## 差分

各プリセットが持つキーは **3 つだけ**。太字が `128k.env` からの変更点:

| キー | 128k | 256k | 1m | 1m-ds-mla |
|---|---|---|---|---|
| `MAX_MODEL_LEN` | 131072 | **262144** | **1048576** | **1048576** |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | 8192 | **16384** | **16384** |
| `VLLM_KV_ARGS` → `--kv-cache-dtype` | fp8 | fp8 | fp8 | **fp8_ds_mla** |
| `VLLM_KV_ARGS` → `--kv-cache-memory-bytes` | 18GiB | 18GiB | 18GiB | **24GiB** |
| `VLLM_KV_ARGS` → `--block-size` | 256 | 256 | 256 | 256 |

`128k.env` と `256k.env` の差は **`MAX_MODEL_LEN` の 1 行だけ**。
`1m.env` はさらに `MAX_NUM_BATCHED_TOKENS` が変わるだけで、KV の指定は同じ。

構成ごとに変わるのは KV 関連のフラグだけなので、`.env` 側は vLLM の引数を
3 つに分けてある。entrypoint はこの順で `vllm serve` に渡す:

| 変数 | 中身 | プリセットで上書き |
|---|---|---|
| `VLLM_PARSER_ARGS` | reasoning / tool-call パーサ | しない |
| `VLLM_KV_ARGS` | KV dtype / サイズ / block-size | **する** |
| `VLLM_EXTRA_ARGS` | speculative-config, moe-backend, compilation-config など | しない |

こうしておかないと、プリセットごとに長い `VLLM_EXTRA_ARGS` を丸ごとコピーする
ことになり、共通部分を直したときに片方だけ古いまま、という事故が起きる。

`MAX_NUM_SEQS` / `GPU_MEMORY_UTILIZATION` / NCCL 系も全プリセット共通で、`.env` 側にある。

## KV サイズの見積もり方 — 「B/token 一定」ではない

起動中のサーバから実測できる:

```bash
curl -s http://127.0.0.1:8910/metrics | grep 'cache_config_info{' | tr ',' '\n' \
  | grep -E 'kv_cache_(memory_bytes|size_tokens)|block_size'
```

**重要**: 確保できるスロット数は `MAX_MODEL_LEN` によって変わる。
同じ 18GiB でも:

| `MAX_MODEL_LEN` | `kv_cache_size_tokens` | 倍率 | 実効 B/token |
|---|---|---|---|
| 131,072 | 286,458 | 2.19x | 67,470 |
| 262,144 | 539,285 | 2.06x | 35,838 |

DeepSeek V4 の KV は c4a (1/4) / c128a (1/128) 圧縮 + 128 token sliding window で
構成されているため、**コンテキストが長いほど 1 トークンあたりが安くなる**。
「B/token 一定」で線形に外挿すると大きく外す (実際に一度外した)。

上の 2 点から線形回帰すると:

```
slots ~= 33,620 + 1.93 * MAX_MODEL_LEN     (KV 18GiB のとき)
```

L=1,048,576 を入れると約 206 万スロット。必要量の約 2 倍あるので、
**1M も 18GiB のまま入る計算**になる (`1m.env`)。

理論的な裏付けは vLLM 公式ブログ
([DeepSeek V4 in vLLM](https://vllm.ai/blog/2026-04-24-deepseek-v4)) にあり、
1M コンテキストでも bf16 で 9.62GiB / sequence、fp8 attention + fp4 indexer なら
約 4.8GiB で足りるとされている。

### `--block-size` でレートが倍変わる

`--block-size` 未指定 (block_size=4) だと `MAX_MODEL_LEN=131072` で
130,737 B/token だったものが、`--block-size 256` を渡すと 67,470 B/token に
半減した。**設定を変えたら必ず測り直すこと。**

### 1M が KV 不足で落ちたら: `fp8_ds_mla`

vLLM がサポートする KV dtype には DeepSeek MLA 専用のパック済みレイアウト
**`fp8_ds_mla`** があり、DeepSeek-V4-Flash ではこれが推奨されている
(`nvfp4_ds_mla` は無効な値なので注意)。`1m-ds-mla.env` がそれ。

**失敗は早い。** dtype 名が無効なら起動時に即
`Invalid value for kv-cache-dtype`、KV が `MAX_MODEL_LEN` に足りなければ
「max seq len is larger than the maximum number of tokens that can be stored in
KV cache」で即死する。どちらもエラーに実際のスロット数が出るので、
そこから上限を割り出せる。

イメージが受け付ける dtype 一覧はコンテナ内で確認できる:

```bash
sudo docker exec dsv4-head vllm serve --help 2>&1 | grep -A12 'kv-cache-dtype'
```

### 実際に効いてくる制約は KV よりも prefill の活性化メモリ

250K 入力のとき `MemAvailable` の最小は 6.7GiB だった。KV は固定サイズなので
増えないが、prefill 中の活性化メモリはコンテキスト長に応じて増える。
1M で OOM-kill (exit 137) される場合はこれが原因で、KV を減らしても効かない。
その場合は 512K (`MAX_MODEL_LEN=524288`) あたりで妥協する。
