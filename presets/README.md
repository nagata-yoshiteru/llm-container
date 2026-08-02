# presets/

コンテキスト長ごとの **差分だけ**を持つ `.env` オーバーレイ。
`.env` (IP・パス入り / git 管理外) はそのままに、上書きしたいキーだけを重ねる。

```bash
sudo docker compose --env-file .env --env-file presets/256k.env --profile head up -d
```

`docker compose` は `--env-file` を複数受け取り **後勝ち**。
**worker 側も同じ組み合わせで起動すること** (片方だけ違うと rendezvous で死ぬ)。

`.env.example` の既定は `1m.env` と同じ内容なので、1M で使うなら指定不要。

## 一覧

| プリセット | コンテキスト | KV | slots | max_concurrency | 特性 |
|---|---|---|---|---|---|
| `128k.env` | 131,072 | 18GiB | 286,458 | 2.19 | prefill の余白が最大 |
| `256k.env` | 262,144 | 18GiB | 539,285 | 2.06 | **常用向け**。250K 入力でも残 6.7GiB |
| `1m.env` | 1,048,576 | 20GiB | 1,772,550 | 1.69 | 既定。900K 入力で残 1.9GiB |

差分は 3 キーのみ (`MAX_MODEL_LEN` / `MAX_NUM_BATCHED_TOKENS` / `VLLM_KV_ARGS`)。
`128k` と `256k` は `MAX_MODEL_LEN` の 1 行しか違わない。

## ベンチ結果

TP=2 / DSpark k=7 / `MAX_NUM_SEQS=1` / prefix caching なし。

### 入力長 vs 速度

| 入力 | TTFT | prefill | decode |
|---:|---:|---:|---:|
| 3,922 | 2.0s | 1,938 t/s | 40.5 t/s |
| 15,958 | 9.7s | 1,646 t/s | 62.1 t/s |
| 63,984 | 35.4s | 1,808 t/s | 47.8 t/s |
| 131,008 | 72.7s | 1,802 t/s | 60.4 t/s |
| 249,952 | 149.6s | 1,670 t/s | 49.3 t/s |
| 499,994 | 372.0s | 1,344 t/s | 65.5 t/s |
| 900,014 | 874.1s | 1,030 t/s | 55.5 t/s |

### 出力長 vs 速度

| 出力 | 所要 | decode |
|---:|---:|---:|
| 13,367 | 237s | 56.4 t/s |
| 16,384 | 382s | 43.5 t/s |
| 24,576 | 419s | 59.4 t/s |

**decode は 37.7〜65.5 t/s (18 計測の平均 51.4)。入力・出力の長さでほぼ変わらない。**
6 分半連続生成しても劣化なし。

**支配的なのは TTFT。** prefill は 900K で半減する (2,000 → 1,030 t/s)。

| 入力 | TTFT の目安 |
|---|---|
| 〜16K | 10 秒以内 |
| 128K | 約 1 分 |
| 250K | 約 2 分半 |
| 500K | 約 6 分 |
| 900K | 約 15 分 |

長尺を投げるならクライアントのタイムアウトを伸ばすこと。
コールドスタート直後の 1 発目だけ JIT で +8 秒ほど余計にかかる。

## 設定を変えるときの注意

### KV サイズは計算で予測できない

`kv_cache_size_tokens` は `MAX_MODEL_LEN` にも KV バイト数にも比例しない。
c4a (1/4) / c128a (1/128) 圧縮と 128 token sliding window の効き方が非線形なため。
実際に 2 回とも外した:

- 128K/256K の 2 点から 1M を線形外挿 → 206 万と予測、実測 97 万 (2 倍過大)
- 18GiB の実測から 20GiB を線形換算 → 108 万と予測、実測 177 万 (1.6 倍過少)

**起動して `/metrics` を読むのが唯一確実。** 判断基準は `max_concurrency >= 1.0`
(1.0 未満だと `MAX_MODEL_LEN` いっぱいのリクエストが KV に入らない)。

```bash
curl -s http://127.0.0.1:8910/metrics | grep 'cache_config_info{' | tr ',' '\n' \
  | grep -E 'kv_cache_max_concurrency|kv_cache_size_tokens'
```

KV が足りなければ起動時に即エラーで落ちるので、試すコストは低い。

### 効いてくる制約は KV ではなく prefill の活性化メモリ

`MemAvailable` の最小値は 250K 入力で 6.7GiB、900K 入力で 1.9GiB。
KV は固定サイズなので増えないが、prefill 中の活性化メモリは入力長に比例する。
**ここが天井。** OOM-kill (exit 137) されたら `MAX_NUM_BATCHED_TOKENS` を
下げるか `256k.env` に落とす。

GB10 の UVM は解放が遅く、900K を投げたあと `MemAvailable` は 2GiB 程度までしか
戻らない。長尺の連投は未検証。

### `--block-size 256` は外さない

未指定 (block_size=4) だと KV の効率が半分になる (`MAX_MODEL_LEN=131072` で
67,470 → 130,737 B/token)。vLLM 公式 recipe も DeepSeek-V4 に 256 を指定している。
