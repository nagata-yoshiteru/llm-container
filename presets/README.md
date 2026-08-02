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

| プリセット | コンテキスト | KV | 合計メモリ | 状態 |
|---|---|---|---|---|
| `128k.env` | 131,072 | 18GiB | 111GiB | **実績あり** (42 tok/s 実測) |
| `256k.env` | 262,144 | 18GiB | 111GiB | 実績構成と同メモリ。通る見込み |
| `1m-ds-mla.env` | 1,048,576 | 24GiB | 117GiB | **実験**。`fp8_ds_mla` が要る |

`128k.env` と `256k.env` は **KV の確保量が同じ** = メモリ使用量が完全に同一。
`.env.example` の既定は `128k.env` と同じ内容なので、128K で使うならプリセット
指定は不要。

## 差分

各プリセットが持つキーは **3 つだけ**。太字が `128k.env` からの変更点:

| キー | 128k | 256k | 1m-ds-mla |
|---|---|---|---|
| `MAX_MODEL_LEN` | 131072 | **262144** | **1048576** |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | 8192 | **16384** |
| `VLLM_KV_ARGS` → `--kv-cache-dtype` | fp8 | fp8 | **fp8_ds_mla** |
| `VLLM_KV_ARGS` → `--kv-cache-memory-bytes` | 18GiB | 18GiB | **24GiB** |
| `VLLM_KV_ARGS` → `--block-size` | 256 | 256 | 256 |

`128k.env` と `256k.env` の差は **`MAX_MODEL_LEN` の 1 行だけ**。

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

## 根拠になっている実測値

起動中のサーバから読める:

```bash
curl -s http://127.0.0.1:8910/metrics | grep 'cache_config_info{' | tr ',' '\n' \
  | grep -E 'kv_cache_(memory_bytes|size_tokens)|block_size'
```

この構成 (`--kv-cache-dtype fp8 --block-size 256`) での実測:

```
kv_cache_memory_bytes = 19,327,352,832  (18GiB)
kv_cache_size_tokens  = 286,458
-> 67,470 B/token = 65.9 KiB/token
```

メモリ収支は「KV 18GiB のとき host `used` が 111GiB」→ **KV 以外が 93GiB**
(重み 78GiB + ランタイム 15GiB)。unified memory は 121GiB なので、
**KV に回せるのは実質 18〜20GiB** が上限。

| コンテキスト | 必要 KV (実測レート) | 判定 |
|---|---|---|
| 128K | 8.2GiB | 余裕 |
| 256K | 16.5GiB | 入る |
| 512K | 32.9GiB | **入らない** (合計 126GiB) |
| 1M | 65.9GiB | **論外** |

つまり **plain fp8 のままでは 256K が実用上の上限**。

## 512K / 1M を狙うなら: `fp8_ds_mla`

vLLM 公式ブログ ([DeepSeek V4 in vLLM](https://vllm.ai/blog/2026-04-24-deepseek-v4))
によると、DeepSeek V4 の KV は c4a / c128a 圧縮 + 128 token sliding window のおかげで
理論上かなり小さく、**1M コンテキストでも bf16 で 9.62GiB / 1 sequence**、
fp8 attention + fp4 indexer なら **約 4.8GiB** で足りるはず。

ところが上の実測は 1M 換算で 65.9GiB。**理論値の約 7 倍**ある。これは
`--kv-cache-dtype fp8` では DS-MLA 専用のパック済みレイアウトに乗らず、
実質 bf16 相当で確保されているためと考えられる。

vLLM がサポートする KV dtype には **`fp8_ds_mla`** があり、DeepSeek-V4-Flash では
これが推奨されている。`1m-ds-mla.env` はこれを試すためのプリセット。

**失敗は早い。** dtype 名が無効なら起動時に即
`Invalid value for kv-cache-dtype`、KV が `MAX_MODEL_LEN` に足りなければ
「max seq len is larger than the maximum number of tokens that can be stored in
KV cache」で即死する。どちらもエラーメッセージに実際のトークン数が出るので、
そこから本当のレートが分かる。

イメージが受け付ける dtype 一覧はコンテナ内で確認できる:

```bash
sudo docker exec dsv4-head vllm serve --help 2>&1 | grep -A12 'kv-cache-dtype'
```
