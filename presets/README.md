# presets/

`.env` への**差分だけ**を持つ `.env` オーバーレイ。
`.env` (IP・パス入り / git 管理外) はそのままに、上書きしたいキーだけを重ねる。

```bash
sudo docker compose --env-file .env --env-file presets/dflash2.env --profile head up -d
```

`docker compose` は `--env-file` を複数受け取り **後勝ち**。
**worker 側も同じ組み合わせで起動すること** (片方だけ違うと rendezvous で死ぬ)。

## 一覧

| プリセット | 内容 | decode | 追加要件 |
|---|---|---|---|
| (無 / `.env` 既定) | v8 イメージ + MTP-4 + fp8 KV | 約 21.8 t/s | なし (MTP head は checkpoint 同梱) |
| `dflash2.env` | v11-dflash2 イメージ + DFlash2 drafter + fp8 KV | 約 46.9 t/s (2.15x) | drafter の取得 (`fetch-model.sh draft`) / drafter は非商用ライセンス |

`dflash2.env` だけが**イメージごと差し替える**プロファイル。
MTP と DFlash2 は speculative-config の method が違うだけで、他は共通
(fp8 KV / block-size 2304 / enforce-eager / gmu 0.85)。

## 速度の実測値 (上流 / 同一の 2x Spark ハードウェア)

| 構成 | decode (C1) | 備考 |
|---|---:|---|
| bf16 KV / 非 speculative | 14.3 t/s | 参考値 |
| fp8 KV + MTP-4 | 21.8 t/s | 既定構成 |
| **fp8 KV + DFlash2** | **46.9 t/s** | acceptance 74.1% |

DFlash2 は**出力が予測しやすいほど速い**。構造化出力 / ツール引数は
acceptance ≈ 0.9、自由記述は ≈ 0.33 前後。agentic 用途は高速帯域に入る。

- `temperature: 0` が +13〜21% 高速 (greedy draft と相性がいい)
- `chat_template_kwargs: {"enable_thinking": false}` が acceptance +8% だが、
  その場合 thinking がタグなしで content に出るので**エージェント用途は
  thinking on + `--reasoning-parser glm45` の組み合わせを推奨**
- DFlash2 の drafter はテキスト専用。vision リクエストは動くが speculation されない

出典: [tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark)
(BENCH-C1-C6-DFLASH2.md / DFLASH2-SPECULATIVE-DECODING.md)

## 設定を変えるときの注意

### KV サイズは固定しないこと

このモデル + GB10 では `--kv-cache-memory` を**大きな値に固定すると
warmup で NVRM OOM** になる (上流が 6 回の boot で実証 / 4.14 GiB 固定は
安定、5.5 GiB 以上は全滅)。理由は GB10 ドライバーが MemAvailable ではなく
MemFree で割当を判定するため。

既定構成は固定値を渡さず、**vLLM プロファイラの建議サイズをそのまま使う**。
起動ログの `Available KV cache memory` を両 rank で確認し、
`/metrics` の `kv_cache_max_concurrency >= 1.0` を満たすこと。

```bash
sudo docker logs glm53-head 2>&1 | grep -E 'Available KV cache memory|GPU KV cache size'
curl -s http://127.0.0.1:8910/metrics | grep 'cache_config_info{' | tr ',' '\n' \
  | grep -E 'kv_cache_max_concurrency|kv_cache_size_tokens'
```

### `--block-size 2304` は変更しない

kpool(4)×64=256 と MLA の 128 アラインの両方の倍数でなければならない
(2304 = 256×9 = 128×18)。それ以外だと DeepGEMM assert で起動不能。

### `--enforce-eager` は外さない

CUDA graph capture はこのパスでは不可 (上流実測)。速度は speculation の
acceptance で稼ぐ構成。

### DFlash2 を使う場合

- drafter のライセンスは **CC-BY-NC-ND-4.0** (非商用・改変禁止)。
- 初回リクエストで drafter 用カーネルの JIT が走る (cold C1 は約 10 t/s 低め)。
- acceptance が 0.15 前後に落ちたら aux hidden state の捕捉が壊れている
  可能性 (crash せずに黙って悪化する)。`/metrics` の
  `spec_decode_num_accepted_tokens_total ÷ ..._num_draft_tokens_total` を見る。
