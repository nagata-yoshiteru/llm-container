# presets/

コンテキスト長ごとの **差分だけ**を持つ `.env` オーバーレイ。
`.env` (IP・パス入り / git 管理外) はそのままに、上書きしたいキーだけを重ねる。

```bash
sudo docker compose --env-file .env --env-file presets/128k.env --profile head up -d
```

`docker compose` は `--env-file` を複数受け取り **後勝ち**。
**3 台とも同じ組み合わせで起動すること** (1 台でも違うと rendezvous で死ぬ)。

既定は 256K なので、256K で使うならプリセットの指定は不要。

| プリセット | コンテキスト | 状態 |
|---|---|---|
| `128k.env` | 131,072 | prefill の余白が最大 |
| (既定) | 262,144 | **常用** |
| `512k.env` | 524,288 | 未実測。`GPU_MEMORY_UTILIZATION` の引き上げが必須 |

1M は載らない。KV は各ノードの空きメモリで決まり、全体の block 数はその min。
head が API サーバと Ray head を抱えるぶん律速で、上限は約 612K トークン。
1M をやるなら 4 台目を足して PP=4 にするか、REAP で枝刈りした軽い重みが要る。

## 変えるときの注意

- **KV サイズは計算で予測できない。** M3 は MSA (sparse attention) なので素の GQA
  計算どおりにならない。起動して `/metrics` の `kv_cache_max_concurrency` が
  1.0 以上あることを確認するのが唯一確実。
- **`--block-size 128` は変えない。** M3 の `sparse_block_size` が 128。
- **効いてくるのは KV より prefill の活性化メモリ。** OOM-kill (exit 137) されたら
  まず `MAX_NUM_BATCHED_TOKENS` を下げる。
