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

## 実測スループット

`1m.env` 構成、DSpark k=7 / `MAX_NUM_SEQS=1` / prefix caching なし。
`256k.env` でも重複する長さではほぼ同じ数字が出た (構成による差はない)。

| 入力 | TTFT | prefill | decode | 合計 |
|---:|---:|---:|---:|---:|
| 3,922 | 2.0s | 1,938 t/s | 40.5 t/s | 4.4s |
| 15,958 | 9.7s | 1,646 t/s | 62.1 t/s | 11.6s |
| 63,984 | 35.4s | 1,808 t/s | 47.8 t/s | 38.0s |
| 131,008 | 72.7s | 1,802 t/s | 60.4 t/s | 74.5s |
| 249,952 | 149.6s | 1,670 t/s | 49.3 t/s | 151.9s |
| 499,994 | 372.0s | 1,344 t/s | 65.5 t/s | 373.6s |
| **900,014** | **874.1s** | **1,030 t/s** | **55.5 t/s** | **876.4s** |

**decode はコンテキスト長でほぼ劣化しない** — 4K で 40.5 t/s、900K でも 55.5 t/s。
DeepSeek Sparse Attention が効いている。一方 **prefill は 900K で半減** する
(2,000 → 1,030 t/s) ので、**支配的なのは TTFT**。

| 入力 | TTFT の目安 |
|---|---|
| 〜16K | 10 秒以内 |
| 128K | 約 1 分 |
| 250K | 約 2 分半 |
| 500K | 約 6 分 |
| 900K | **約 15 分** |

長尺を投げるならクライアント側のタイムアウトを必ず伸ばすこと。
コールドスタート直後の 1 発目だけ JIT ウォームアップで +8 秒ほど余計にかかる。

## メモリが本当の上限 (KV ではない)

`MemAvailable` の最小値:

| 構成 | 投げた入力 | 最小 MemAvailable |
|---|---|---|
| `256k.env` | 249,952 | 6.7GiB |
| `1m.env` | 900,014 | **1.9GiB** |

900K の prefill で 1.9GiB まで落ちる。**ここが実質的な天井**で、これ以上
`MAX_MODEL_LEN` を伸ばしても、あるいは KV を増やしても OOM-kill (exit 137) に
なる。KV は固定サイズなので増えないが、prefill 中の活性化メモリは
コンテキスト長に比例して増えるため。

なお GB10 の UVM は解放が遅く、900K を 1 発投げたあと `MemAvailable` は
2GiB 程度までしか戻らなかった。長尺を連投する運用は未検証。

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
KV 18GiB 固定での実測 3 点:

| `MAX_MODEL_LEN` | `kv_cache_size_tokens` | `max_concurrency` | 実効 B/token | 1 本分の KV |
|---|---|---|---|---|
| 131,072 | 286,458 | 2.19 | 67,470 | 8.24GiB |
| 262,144 | 539,285 | 2.06 | 35,838 | 8.75GiB |
| 1,048,576 | 970,424 | **0.93** | 19,916 | 19.45GiB |

DeepSeek V4 の KV は c4a (1/4) / c128a (1/128) 圧縮 + 128 token sliding window で
構成されているため、**コンテキストが長いほど 1 トークンあたりが安くなる**。
ただし圧縮の効きには頭打ちがあり、128K→256K は 1 本分が +0.5GiB しか増えないのに、
256K→1M では +10.7GiB 増える。

**線形でも単純なべき乗でもないので、外挿してはいけない。**
実際に 2 点 (128K/256K) から線形回帰して 1M を「約 206 万スロット」と見積もったが、
実測は 970,424 で 2 倍以上外した。

### 唯一の判断基準: `kv_cache_max_concurrency >= 1.0`

これが 1.0 未満だと、**`MAX_MODEL_LEN` いっぱいのリクエストは KV に入らない**。
上の表では 1M 指定が 0.93 なので、実効上限は約 970K トークン。真の 1M が要るなら
KV を `18GiB / 0.93 = 19.5GiB` 以上、余裕を見て 21GiB にする必要がある
(ただしメモリ収支と要相談)。

起動したら必ずこれを確認すること:

```bash
curl -s http://127.0.0.1:8910/metrics | grep 'cache_config_info{' | tr ',' '\n' \
  | grep -E 'kv_cache_max_concurrency|kv_cache_size_tokens'
```

理論的な下限は vLLM 公式ブログ
([DeepSeek V4 in vLLM](https://vllm.ai/blog/2026-04-24-deepseek-v4)) にあり、
1M コンテキストでも bf16 で 9.62GiB / sequence、fp8 attention + fp4 indexer なら
約 4.8GiB とされている。実測の 19.45GiB はその 4 倍で、まだ削る余地はありそう
(→ `fp8_ds_mla`)。

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
