# GLM-5.3-Flash UNCENSORED / MLX 4bit / Mac Studio (M5 Ultra 256GB)

dealignai の uncensored 版 GLM-5.3-Flash を Mac Studio 1 台で動かす構成。
[Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) を使い、OpenAI 互換と Anthropic 互換の API を `:8910` で提供する。
モデル名は `glm-5.3-flash-uncensored`。

DGX Spark x3 RING 版 (`DGX-Spark-3-RING/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` ブランチ) を
この Mac 用に置き換えたもの。DGX 版との違いは次の表のとおり。

| | DGX Spark x3 RING (前ブランチ) | Mac Studio (ここ) |
| --- | --- | --- |
| 重み | `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` (W4A16 NVFP4) | `Solstice-AI/GLM-5.3-Flash-UNCENSORED-mlx-oQ4e-DFlash2` (MLX 4bit) |
| ランタイム | kindling (vLLM nightly) + mentat、TP=3 | Rapid-MLX 1.0.2 (MLX / mlx-vlm)、1 プロセス |
| 起動 | `sudo docker compose --profile ...` | `./scripts/serve.sh up` (Docker は使わない) |
| MTP | k=3 | **無効** (下記) |
| コンテキスト | 524288 (1M は未検証) | **262144** (1M は未検証) |
| 同時実行 | 8 | 実質 1 (vision レーンは直列、下記) |

2026-11-07 に M5 Ultra 256GB / macOS 27.2 で、取得・準備・起動・応答まで確認した。
OpenAI 互換・Anthropic 互換・tool call・思考の分離が動くことも確認した。
ベンチ (1 本ずつ) の結果は次のとおり (下記「速度と正答率」)。

- decode 約 38.5 tok/s
- prefill 545〜860 tok/s (128K〜1K)
- 数学・コード 7/7 正解

**uncensored の度合いと、2 本以上の並列は未検証。**

**取得した重みは、そのままでは正しく動かない。**
`./scripts/prepare-model.sh` で直した serving ディレクトリを使う (下記「そのままでは動かない 3 点」)。

## 起動・停止

```bash
./scripts/serve.sh up       # 起動。/health が返るまで待つ (ロードに数分)
./scripts/serve.sh logs     # ログ (.cache/rapid-mlx/logs/serve.log)
./scripts/serve.sh status   # /health と /v1/models
./scripts/serve.sh down     # 停止
```

`up` の前に、再起動のたびに wired メモリの上限を引き上げる (後述)。

```bash
sudo sysctl iogpu.wired_limit_mb=245760
```

```bash
curl -fsS http://127.0.0.1:8910/v1/models
curl -s http://127.0.0.1:8910/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash-uncensored",
  "messages": [{"role": "user", "content": "What is 17*19? Reply with the integer only."}],
  "max_tokens": 8000}'
```

デバッグ時は `./scripts/serve.sh fg` でフォアグラウンド起動する。

## 新規ホストでの準備

```bash
cp .env.example .env        # 既定値のままでよい。.env が無くても既定値で動く
# uv が無ければ: curl -LsSf https://astral.sh/uv/install.sh | sh
./scripts/setup.sh          # ./venv に Python 3.12 + rapid-mlx==1.0.2
./scripts/fetch-model.sh    # ./models/ に約 186GB (100MB/s 程度で約 30 分)
./scripts/prepare-model.sh  # mlx-vlm 用の serving ディレクトリを作る (数秒、追加 70MB)
sudo sysctl iogpu.wired_limit_mb=245760
./scripts/preflight.sh
```

- `prepare-model.sh` は `MODEL_PATH` (取得したもの) から `SERVE_MODEL_PATH` (既定 `MODEL_PATH-mlxvlm`) を作る。
  shard は hardlink で、元のファイルは変更しない。同じディレクトリへの再実行は検証だけ行う。

- macOS 同梱の `python3` は 3.9 で、Rapid-MLX (3.12 以上が必要) は動かない。venv は uv で作る。
- **Docker は使わない。** macOS の Docker コンテナから Metal GPU は使えないため。
- setup と取得に sudo は不要。sudo が要るのは `sysctl` だけ。
- `.env` は compose 時代と同じ `KEY=VALUE` (クォートなし) の書式。
  スクリプトは `source` せず、必要なキーだけ読む (`scripts/env.sh`)。

## 構成と重み

### なぜ NVFP4 版ではないのか

DGX 版で使っていた `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` は modelopt の NVFP4 (CUDA 専用) なので、MLX では読めない。
dealignai は MLX 版を出していない。
そこで、同じ dealignai の uncensored 重み (FP8 版、328GB) を Solstice-AI が MLX に変換したものを使う。

| 項目 | 値 |
| --- | --- |
| repo | `Solstice-AI/GLM-5.3-Flash-UNCENSORED-mlx-oQ4e-DFlash2` |
| revision | `bff40d24efb689d33e094083ce4a5d6835601730` で固定 (`.env` の `MODEL_REVISION`) |
| 量子化 | 4-bit affine / group 64 (~4.6 bpw)。lm_head・router・gate は 8bit、attention の一部は 6/8bit (543 箇所を個別指定) |
| 形式 | `model_type: glm5_next` (vision tower 込み)、safetensors 38 shard / 185.7GB |
| 取得しないもの | `mmproj-BF16.gguf`、`speculative/*.gguf` (llama.cpp 用、計 3.5GB) |

8bit 版 (oQ8e) は 342GB あり、256GB 機には入らない。

注意点が 2 つある。

- dealignai の FP8 版は 2026-08-28 に「files fixed · please re-download」として再公開されている (まれに繰り返しループに入る問題の修正)。
  Solstice 版がどちらの重みから変換したかは明記されていない。
- uncensored としての性能 (HarmBench の拒否率 0%) は dealignai と Solstice の自己申告。
  MMLU も dealignai は 87.33%、Solstice は 85.28% と食い違っている。品質は起動後に自分で確認すること。

重みの値そのものは正しい。dealignai FP8 版の shard 1・2・31・32 と全テンソルを突き合わせた結果は次のとおり。

- BF16 / FP8 由来の 8bit・6bit: cos 0.9997〜1.0000
- 4bit の expert: cos 0.9957 (4bit なら妥当)

### そのままでは動かない 3 点 (`prepare-model.sh` が直す)

Solstice のモデルカードの手順 (mlx-lm で読む) は、mlx-lm に `glm5_next` が無いのでそもそも動かない。
mlx-vlm で読むと、次の 3 点で壊れる。いずれも 2026-11-07 に実機で確認した。

| # | 症状 | 原因 | 直し方 |
| --- | --- | --- | --- |
| 1 | ロード時に `[concatenate] ... (8192,768), (8192,1024), (8192,768)` | config の個別量子化キーが `model.language_model.layers.N.*` で、mlx-vlm が引く `language_model.model.layers.N.*` と一致しない。加えて mlx-vlm は q/k/v などを連結するが、同じ層の中でビット数が混在している (例: layer 20 は q=6bit、k=8bit) | モジュールパス名のキーと、連結後の名前 (`qkv_proj`、`fbg_a_proj`、`qkv_a_proj`、`gate_up_proj`、`embed_q`、`unembed_out`) のキーを config に追加する。混在している 2 個は 6bit から 8bit に再量子化する (相対誤差 0.7%) |
| 2 | 起動するが、出力が完全に文字化けする (GPU だけ。CPU では正しく答える) | 元版では float32 の `hc_*_base` / `hc_*_scale` が BF16 で保存されている。mlx-vlm の HyperConnection の Metal カーネルは base を float4 として読むため、Sinkhorn の混合行列 (comb) が壊れる (単体テストで cos 0.95)。それが 45 層×2 回積み重なる | 該当するテンソルを float32 に戻す。同じく本来 float32 の `A_log`、`dt_bias`、`e_score_correction_bias` も戻す (計 291 個)。BF16 からの upcast なので、値は BF16 精度のまま |
| 3 | 毎回、独自の長いシステムプロンプトが注入される | chat template が Solstice 独自の「Protium」テンプレートに差し替えられている | dealignai 版の `chat_template.jinja` (revision 固定) を使い、`tokenizer_config.json` に埋め込まれたテンプレートを削除する |

直したテンソルは追加の shard (`model-zz-mlxvlm-fixup.safetensors`、69MB) に入れる。
mlx-vlm は index にある shard を名前順に読み、同じキーは後から読んだもので上書きする。
この仕組みを使うので、元の 38 shard には手を入れない。
処理内容と、使った mlx-vlm / rapid-mlx のバージョンは `mlxvlm-preparation.json` に残る。
mlx-vlm のバージョンを上げたら、serving ディレクトリを消して作り直すこと。

### なぜ Rapid-MLX なのか

`glm5_next` のローダーは、2026-11 時点で mlx-lm には無く、mlx-vlm にだけある。
Rapid-MLX は mlx-vlm の上に、GLM-5.3 用に次のものを載せている。

- ランタイム修正: `patches/glm5_next_runtime.py`、`glm5_next_forget_gate_quant.py`
- tool parser `glm47` と reasoning parser `glm5`
- Anthropic Messages API (`/v1/messages`)

Claude Code からそのまま使えるのが、このランタイムを選んだ一番の理由。

- 起動フラグはすべて `scripts/entrypoint.sh` で組み立てる。
  DGX 版の entrypoint と同じく、まずモデルを検証してから `exec` する。
- ローカルパスでは parser を名前から推定できないことがある。そのため `--tool-call-parser glm47 --reasoning-parser glm5` を明示している。
- `--disable-model-downloads` と `HF_HUB_OFFLINE=1` を付けて、起動時に HF へ取りに行かせない。
  あわせて `DO_NOT_TRACK=1` で利用統計の送信を止め、更新チェックも切る。
- `--text-only` は付けない。テキスト専用の経路は mlx-lm を使うため、`glm5_next` を読めない。

### MTP / 投機的デコードは既定で無効

Rapid-MLX の公式 alias (`glm5.3-flash-4bit`) は MTP k=1 を使う。
ただし対象は別の重み (Vontra/TensorFold の 4bit、検閲あり) で、MTP には専用の sidecar を使う。
uncensored 版にその sidecar を組み合わせた例は見つからなかった。
そのため `SPEC_DECODE=0` (`--no-spec-decode`) とした。
Solstice の DFlash2 drafter は GGUF なので、Rapid-MLX では使えない。

## メモリ (256GB unified)

MLX が GPU 用に wired にできる上限は `iogpu.wired_limit_mb` で決まる。
既定値 (0) では 256GB 機で約 75% (192GiB 前後) にしかならない。
重みだけで 173GiB あるので、KV と活性化まで入れるには足りない。

| 内訳 | 量 |
| --- | --- |
| 重み (oQ4e) | 185.7GB = 173GiB |
| wired 上限 (`WIRED_LIMIT_MB`) | 245760MiB = 240GiB (OS に 16GiB を残す) |
| 参考: Rapid-MLX 公式 alias の実測 (別の 4bit 重み、M3 Ultra) | 重み 181.7GB、32K でピーク 195.6GB |
| 実測: 起動直後・短いリクエストのあと (M5 Ultra) | 空き (free + inactive + purgeable) 約 74GiB、swap 0.6GB |

- 起動時に Rapid-MLX が「Memory pressure warning: ... 111% projected utilization」と警告する。
  これは「ディスク上のサイズ × 1.5」という機械的な見積もりによるもので、実測とは合わない。
  長いコンテキストでの実測は、まだしていない。
- 240GiB という値は経験則 (OS 分を 8〜16GiB 残す) から決めたもので、上流の推奨値ではない。
  macOS が固まる・swap が増えるなら下げる。
- `sysctl` は再起動すると元に戻る。`entrypoint.sh` と `preflight.sh` は、値が足りなければ起動を止める。
- KV の 1 トークンあたりの量は実測していない。
  config から見積もると、DSA 層 11 層 × (latent 512 + indexer 128) × bf16 で約 14KB/token になる。
  256K で約 3.5GiB、1M で約 13GiB。残りの 34 層は KDA (線形 attention) で、状態のサイズはトークン数によらず一定。
  **DGX 版の README と同じく、見積もりは外れる前提で、起動後に確認すること。**
- `--kv-cache-dtype int8/int4` は、Rapid-MLX が MLA 系のモデルには拒否する仕様になっている。
  GLM-5.3 が該当するかは未確認なので、`KV_CACHE_DTYPE` は空 (bf16) のままにしてある。
- prefix cache は Rapid-MLX の既定で有効。上限 `CACHE_MEMORY_MB` は空なら自動 (空き RAM の約 20%)。

## 調整ノブ (`.env`)

| キー | 既定 | メモ |
| --- | --- | --- |
| `MAX_MODEL_LEN` | 262144 | `--context-length`。モデルの上限は 1048576。上げるならメモリと TTFT を実測してから |
| `MAX_NUM_SEQS` | 4 | 受け付ける同時実行数。ただし vision 込みのモデルは Rapid-MLX が直列に処理する (`/v1/models` の `concurrent_text_requests: false`) |
| `PREFILL_STEP_SIZE` | 1024 | 公式 alias の推奨値。大きくすると速いが、活性化メモリが増える |
| `DEFAULT_REASONING_EFFORT` | 空 | 空ならテンプレートの既定 (Max)。`none` / `minimal` / `low` / `medium` / `high` / `xhigh` |
| `SPEC_DECODE` | 0 | 1 にすると Rapid-MLX の自動判定に任せる (未検証) |
| `API_HOST` / `API_KEY` | 127.0.0.1 / 空 | LAN に出すなら `API_HOST=0.0.0.0` と `API_KEY` を設定する。キーなしの公開バインドは Rapid-MLX が拒否する |
| `WIRED_LIMIT_MB` | 245760 | 上記 |
| `RAPID_MLX_EXTRA_ARGS` | 空 | 空白区切りで追加する。JSON に空白を入れないこと |

reasoning effort について、dealignai のモデルカードには次の注意がある。

- 拒否がほぼ 0 になるのは reasoning off と max のとき。low では拒否が一部残るよう、意図的に作られている。
- max で `max_tokens` が小さい (2000 程度) と、思考だけで予算を使い切り、content が空になる。
  6000 以上を推奨。
- テンプレートには `enable_thinking` という変数が無い。

## 使う

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8910/v1", api_key="local")
r = client.chat.completions.create(
    model="glm-5.3-flash-uncensored",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=8000,
)
```

Claude Code から使うときは `scripts/local-llm-on.sh` / `local-llm-off.sh` で設定ファイルを切り替える。
Anthropic 互換のベース URL は `/v1` を付けない `http://localhost:8910` にする。

```jsonc
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8910",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_DEFAULT_OPUS_MODEL":   "glm-5.3-flash-uncensored",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.3-flash-uncensored",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL":  "glm-5.3-flash-uncensored"
  }
}
```

## Rapid-MLX の hotfix (長い生成で落ちる問題)

rapid-mlx 1.0.2 の MLLM (vision) lane では、約 11,200 トークンを生成したところで
`[metal::malloc] Resource limit (499000) exceeded` で落ちる (API は `insufficient_memory` を返す)。
GLM-5.3-Flash は vision 込みなので、このモデルは必ず MLLM lane で動く。

- 原因: text lane は、hybrid (再帰) キャッシュの関数的な更新が遅延グラフを抱え続けるのを、
  数ステップごとの `mx.eval([c.state for c in cache])` で抑えている (上流 #1834 / #2834)。
  MLLM lane の decode にはこの barrier が無い。
  そのため 45 層分の更新が、1 トークンあたり約 45 個の Metal handle として残り続ける。上流 #3270 と同じ系統。
- 対処: `scripts/rapid_mlx_launch.py` が `MLLMBatchGenerator._step` に、text lane と同じ予算の barrier を足す
  (同時実行数 × 間隔 = 64)。そのうえで、同じプロセスで rapid-mlx を起動する。
  rapid-mlx のバージョンが 1.0.2 以外なら、起動を止める (fail-closed)。
- 確認 (2026-11-07): 同じ 14,000 トークンの生成を両方で試した。
  hotfix なしでは 11,234 トークンで落ち、hotfix ありでは完走した (37.6 tok/s、低下なし)。
  素の mlx-vlm では、hotfix なしでも 12,500 トークンまで落ちない。

## 速度と正答率 (1 本ずつ)

2026-11-07、M5 Ultra 256GB、hotfix ありで測った。結果は `results/glm53-mac-oq4e-c1/` (git 管理外)。

```bash
python3 scripts/bench-glm53.py --label mac-oq4e --output results/glm53-mac-oq4e.json
cd scripts && python3 bench-glm53-eval.py --phase quality --concurrency 1 --output-dir ../results/glm53-mac-oq4e-c1
cd scripts && python3 bench-glm53-eval.py --phase profile --concurrency 1 --output-dir ../results/glm53-mac-oq4e-c1
python3 scripts/summarize-glm53-eval.py results/glm53-mac-oq4e-c1
```

### decode

| 条件 | decode |
| --- | --- |
| 固定 256 tokens、1 本 (prose / code 各 3 回) | 中央値 38.5 tok/s (37.4〜39.0)、TTFT 0.19 秒 |
| quality 7 問 (1.4K〜17K tokens) | 37.7〜39.1 tok/s。17K tokens まで、生成長による低下はほぼ無い |
| 2 本同時 | 1 本あたり 28.9 tok/s、合計 38.7 tok/s。直列に処理されるので合計は伸びない |

### cold prefill (1 本、prefix cache を外して計測)

| 入力 | TTFT | prefill |
| --- | --- | --- |
| 1K | 1.2 秒 | 約 860 tok/s |
| 8K | 10.1〜10.6 秒 | 約 810 tok/s |
| 32K | 48.5 秒 | 約 680 tok/s |
| 128K | 240 秒 | 約 545 tok/s |

DGX Spark x3 よりかなり遅い。長い入力を投げるときは、クライアントのタイムアウトを伸ばすこと。

### 正答率 (reasoning_effort=max、temperature 1.0、max_tokens 32768)

`glm53_eval_tasks.py` の数学 4 問とコード 3 問で、**7/7 正解**。
コードは、生成物を読んでから repo のテストで実行した (`code-grades.json`)。

| 問題 | 結果 | 生成 tokens | 時間 |
| --- | --- | --- | --- |
| M4 2^2026 mod 1000 | 正解 (864) | 1,363 | 36 秒 |
| M1 AIME 2006 (ゼロを含まない a+b=1000) | 正解 (738) | 17,254 | 458 秒 |
| M2 AIME 1999 (n²−19n+99 が平方数) | 正解 (38) | 4,166 | 109 秒 |
| M3 桁和 30 の個数 | 正解 (50877) | 2,666 | 69 秒 |
| C1 転倒数 O(n log n) | PASS | 3,844 | 101 秒 |
| C2 3×n ドミノ (n ≤ 10^18) | PASS | 5,288 | 139 秒 |
| C3 最小窓部分文字列 | PASS | 12,069 | 320 秒 |

各問 1 回ずつの結果なので、正答率の誤差は大きい。
M1 は 17K tokens あり、hotfix なしでは途中で落ちる長さ。

参考 (別の重み):

| 出典 | 重み | 機体 | decode |
| --- | --- | --- | --- |
| Rapid-MLX 公式 alias | Vontra 4bit + MTP | M3 Ultra 256GB | 27.8 tok/s |
| omlx.ai (llmcheck 経由) | pipenetwork の mixed 4/8bit | M5 Ultra 256GB | 61.7 tok/s (1K) / 58.1 tok/s (64K)、MTP なし |

## 検証とベンチ

DGX 版のベンチスクリプトは、`:8910` と `glm-5.3-flash-uncensored` を前提にしている。
`bench-glm53-eval.py` は Rapid-MLX にも対応させた。対応した内容は次のとおり。

- `/metrics` の名前の違いを吸収する
- `/tokenize` が無いので `/v1/messages/count_tokens` を使う
- `cache_salt` が無いので、先頭に nonce を置いて prefix cache を外す

1 本ずつ測るときは `--concurrency 1` を付ける。上の「速度と正答率」の手順で実行を確認した。
`--phase retrieval` と `bench-glm53-long-context.py` は、まだ試していない。

```bash
python3 scripts/bench-glm53.py --label mac-oq4e --output results/glm53-mac-oq4e.json
python3 scripts/bench-glm53-eval.py --phase quality --output-dir results/glm53-mac-oq4e-quality
# long-context は既定が 1M 入力と DGX の IP なので、引数で上書きする
python3 scripts/bench-glm53-long-context.py --input-tokens 200000 --concurrencies 1 --memory-hosts
```

## このブランチで消したもの

`docker-compose.yml`、`Dockerfile.kindling`、`patches/kindling/`、`README.glm53-3spark.md`、
`scripts/prepare-glm53-tp3.py`、`scripts/test-glm53-*.py` は DGX/kindling 専用なので削除した
(RING ブランチに残っている)。
`README.dspark.md`、`presets/`、`patches/` は、さらに前のブランチから引き継いだものなので、この構成では使わない。

## 出典

- [dealignai/GLM-5.3-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-FP8): 元の uncensored 重みとモデルカード
- [Solstice-AI/GLM-5.3-Flash-UNCENSORED-mlx-oQ4e-DFlash2](https://huggingface.co/Solstice-AI/GLM-5.3-Flash-UNCENSORED-mlx-oQ4e-DFlash2): 使用する MLX 版
- [raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) v1.0.2: ランタイム ([GLM-5.3 alias](https://rapidmlx.com/models/glm-5.3-flash))
- [TensorFold/GLM-5.3-Flash-MLX-4bit-MTP](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-4bit-MTP): 公式 alias の重み (比較用)
- [llmcheck: GLM-5.3-Flash on M5 Ultra](https://llmcheck.net/models/glm-53-flash-on-m5-ultra/): 速度の参考値
