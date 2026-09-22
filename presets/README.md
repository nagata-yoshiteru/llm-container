# presets/

`.env` への**差分だけ**を持つ `.env` オーバーレイ。
`.env` (IP・パス入り / git 管理外) はそのままに、上書きしたいキーだけを重ねる。

```bash
sudo docker compose --env-file .env --env-file presets/roce.env --profile head up -d
```

`docker compose` は `--env-file` を複数受け取り **後勝ち**。
**worker 側も同じ組み合わせで起動すること** (片方だけ違うと rendezvous で死ぬ)。

## 一覧

| プリセット | 内容 | イメージ差し替え | 追加要件 |
|---|---|---|---|
| (無 / `.env` 既定) | v11-dflash2 + DFlash2 k=7 + fp8 KV + KV 8 GiB 固定 | — | drafter (`fetch-model.sh draft`) |
| `mtp4.env` | v8 + MTP-4 に戻す (約 21.8 t/s) | **あり** | なし (MTP head は checkpoint 同梱) |
| `thinking-off.env` | thinking を既定 off (acceptance +8%) | なし | なし。**reasoning が content に混ざる** |
| `deep-concurrency.env` | k を C4 で 7 -> 5 に切り替える schedule | なし | なし |
| `roce.env` | b12x RoCE all-reduce (aggregate +5〜18%) | なし | `build-roce-bundle.sh` |

**イメージを差し替える preset を出入りするときは JIT キャッシュを 2 台で消すこと**:

```bash
sudo rm -rf vllm-cache/{vllm,triton,torchinductor,flashinfer,tilelang}/*
```

`thinking-off.env` / `deep-concurrency.env` / `roce.env` はイメージを変えないので
キャッシュを消す必要はない。

## 速度の実測値 (上流 / 同一の 2x Spark ハードウェア)

`.env` 既定 (DFlash2 k=7 / fp8 KV / 262K ctx / temperature 0):

| 構成 | decode (C1) | 備考 |
|---|---:|---|
| bf16 KV / 非 speculative | 14.3 t/s | 参考値 |
| fp8 KV + MTP-4 (`mtp4.env`) | 21.8 t/s | acceptance 約 0.5 |
| **fp8 KV + DFlash2 (既定)** | **46.9 t/s** | acceptance 74.1% / 2.15x |

2026-09-18 の healthy-fleet 実測 (RoCE + prefix 修正 + KV 8 GiB 固定を入れる前後):

| | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| before (aggregate t/s) | 43.4 | 29.0 | 30.2 | 50.3 | 44.4 | 47.8 |
| after | 42.7 | 33.8 | 35.9 | 59.0 | 40.9 | 51.6 |

**prefill +26〜36% / aggregate +8〜19% (C2-C4, C6) / 単発 decode はほぼ横ばい。**
この夜に触ったのは step の外側なので、単発 decode が伸びないのは想定どおり。

DFlash2 は**出力が予測しやすいほど速い**。構造化出力 / ツール引数は
acceptance ≈ 0.9、自由記述は ≈ 0.33 前後。agentic 用途は高速帯域に入る。

- `temperature: 0` が +13〜21% 高速 (greedy draft と相性がいい)
- DFlash2 の drafter はテキスト専用。vision リクエストは動くが speculation されない

出典: [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
(`CURRENT.md` / `docs/SPEED-NIGHT-2026-09-18.md` / `docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md`)

## 設定を変えるときの注意

### KV は 8 GiB に固定する (2026-09-18 に方針が逆転した)

**旧 README の「`--kv-cache-memory` を固定するな」は撤回された。**
上流も `docs/GB10-KV-MEMORY-LADDER.md` ごと superseded 扱いにしている。

| | KV トークン |
|---|---:|
| プロファイラ任せ | 581,040 |
| 6 GiB 固定 | 678,661 |
| **8 GiB 固定 (既定)** | **714,240** |

逆転が成立した条件は **boot hardening を先に入れたこと** (`MAX_JOBS=2` /
永続 JIT キャッシュ / `MEM_LIMIT=112g` / `scripts/flusher.sh`)。
hardening 抜きで固定すると旧ラダー通り warmup で NVRM OOM する。
詳細は README「KV は 8 GiB に固定する」。

固定すると **vLLM はメモリプロファイリングを丸ごとスキップする**ので、
`MAX_NUM_BATCHED_TOKENS` の活性化ピークを誰も検証しない。
**8192 を超えないこと** (16384 は両ノードで NVRM OOM / entrypoint が弾く)。

### `--block-size 2304` は変更しない

kpool(4)×64=256 と MLA の 128 アラインの両方の倍数でなければならない
(2304 = 256×9 = 128×18)。それ以外だと DeepGEMM assert で起動不能。

prefix cache の粒度もこのブロックサイズで決まる。**2,304 トークン未満の
プロンプトは原理的に hit しない**ので、検証には長いプロンプトを使うこと。

### `--enforce-eager` は外さない

CUDA graph capture は **TP4 側のレーン**。上流の実測で TP2 では
`cudagraph_mode: FULL_AND_PIECEWISE` が C1 +4.1% / C2 -4.6% / C4 +1.1% /
C6 -1.7% (平均 ≈ -0.3%) と横ばい。TP4 では 503 -> 530 tok/s と明確に効く。
TP2 は eager のまま、速度は speculation の acceptance で稼ぐ。

### `MAX_NUM_SEQS` は 6 から上げない

上流が 32 を試して C12 で aggregate +10% だけ、C8〜C16 で TTFT p90 が
60〜179 秒に悪化した。mnbt 16384 と併せると両ノードが NVRM OOM。

### `num_speculative_tokens` は 7 のまま

drafter の `block_size 8 - 1` で固定。8 にするとモデルが学習していない位置を
draft する。並行度優先なら `deep-concurrency.env` の schedule を使う。

### DFlash2 の健全性

- drafter のライセンスは **CC-BY-NC-ND-4.0** (非商用・改変禁止)。
- 初回リクエストで drafter 用カーネルの JIT が走る (cold C1 は約 10 t/s 低め)。
- acceptance が 0.15 前後に落ちたら aux hidden state の捕捉が壊れている
  可能性 (crash せずに黙って悪化する)。`/metrics` の
  `spec_decode_num_accepted_tokens_total ÷ ..._num_draft_tokens_total` を見る。
