# DeepSeek-V4.1-Flash-1017 UNCENSORED / JANG_2K / Mac Studio (M5 Ultra 256GB)

dealignai の uncensored 版 DeepSeek-V4.1-Flash-1017 を Mac Studio 1 台で動かす構成。
[vmlx](https://github.com/jjang-ai/vmlx) と jang_tools を使い、OpenAI 互換と Anthropic 互換の API を `:8910` で提供する。
モデル名は `deepseek-v4.1-flash-uncensored`。

GLM-5.3-Flash 版 (`Mac-Studio-256G/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` ブランチ) の構成を引き継ぎ、
モデルとランタイムを入れ替えたもの。

| | GLM-5.3-Flash 版 (前ブランチ) | DeepSeek-V4.1 版 (ここ) |
| --- | --- | --- |
| 重み | Solstice-AI の MLX 4bit (oQ4e)、常駐 173GiB | dealignai の JANG_2K (routed expert 平均 2.47bit)、常駐 197GiB |
| ディスク | 186GB | 424GB (うち Engram 203GB はディスクから読む) |
| ランタイム | Rapid-MLX 1.0.2 + hotfix | vmlx 1.6.84 + jang 2.5.53 (`./venv-vmlx`) |
| 先読み | 無効 | DSpark (バンドル同梱の MTP draft、3 段) |
| 同時実行 | 実質 1 | 1 (vmlx が固定する) |
| decode (1 本) | 約 38.5 tok/s | **約 54 tok/s** (48.6〜59.3) |
| prefill 32K / 128K | 48.5 秒 / 240 秒 | **20 秒 / 80 秒** |
| 数学・コード 7 問 | 7/7 | **7/7** (生成トークンは約半分) |

2026-11-07 に M5 Ultra 256GB / macOS 27.2 で、取得・起動・応答・ベンチまで確認した (下記「速度と正答率」)。
OpenAI 互換・Anthropic 互換 (`/v1/messages`)・tool call (`deepseek_v41`)・思考の分離も動いた。

注意点が 2 つある。

- **画像入力は使えない。** config に `vision_config` はあるが、checkpoint に vision tower の重みが無い。
  vmlx は「stripped_vision_tower」と判定してテキスト専用で動かす (モデルカードは画像入力ありと書いている)。
- **uncensored の度合いと、長いコンテキストでの品質は未検証。**

## 起動・停止

```bash
sudo sysctl iogpu.wired_limit_mb=245760   # 再起動のたびに必要
./scripts/serve.sh up       # 起動。/v1/models にモデル名が出るまで待つ
./scripts/serve.sh logs     # ログ (.cache/vmlx/logs/serve.log)
./scripts/serve.sh status
./scripts/serve.sh down
```

**GLM 版のサーバとは同時に動かせない** (173 + 197GiB)。切り替えるときは、先に片方を `down` する。
両方とも `:8910` を使う。

## 新規ホストでの準備

```bash
cp .env.example .env        # 既定値のままでよい。.env が無くても既定値で動く
# uv が無ければ: curl -LsSf https://astral.sh/uv/install.sh | sh
./scripts/setup.sh          # ./venv-vmlx に Python 3.12 + vmlx + jang
./scripts/fetch-model.sh    # ./models/ に約 424GB (65MB/s で約 1 時間 50 分)
sudo sysctl iogpu.wired_limit_mb=245760
./scripts/preflight.sh
```

- vmlx は mlx-vlm 0.5 系 / transformers 5.18+ を要求する。GLM 版の Rapid-MLX (mlx-vlm 0.7.2) とぶつかるので、venv を `./venv-vmlx` に分けている。
- **mlx-lm は 0.31.3 に固定する** (`MLX_LM_VERSION`)。
  vmlx 1.6.84 は `mlx_lm.generate.SequenceStateMachine` を使うが、依存指定は `mlx-lm>=0.31.3` だけ。
  素直に入れると 0.32.2 が入り、このクラスが無いため起動時に `ImportError` になる (2026-11-07 に確認)。
- `VERIFY=1 ./scripts/fetch-model.sh` で、取得後に `SHA256SUMS` で全ファイルを検証する (424GB を読むので数十分)。
- Docker は使わない (macOS の Docker からは Metal GPU を使えない)。sudo が要るのは `sysctl` だけ。

## 構成と重み

### なぜ Rapid-MLX ではないのか

この重みは JANG 形式 (jjang-ai の混合精度 MLX 量子化) で、Rapid-MLX には読み込み経路が無い。
Rapid-MLX のカタログにある DeepSeek-V4.1 は、自前で変換した**検閲ありの版**だけ。
モデルカードは「Osaurus 0.25.27+、または DSV4.1 に対応した jang_tools を持つ `vmlx serve`」を指定している。

- vmlx の DSV4.1 経路: `loaders/load_jangtq_dsv41.py` から `jang_tools.dsv4_1.mlx_model.load_deepseek_v41` を呼ぶ。
- vmlx は DSV4.1 を検出すると、次のように固定する (`cli._apply_dsv41_runtime_policy`)。
  - 同時実行 1、continuous batching on、汎用の KV 量子化なし
  - JIT / smelt / flash-MoE / 外部 drafter は無効
- tool parser `deepseek_v41`、reasoning parser `deepseek_r1` は、vmlx のモデル登録 (`model_configs.py`) の既定値と同じものを明示している。

### 重み

| 項目 | 値 |
| --- | --- |
| repo | `dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-JANG_2K` |
| revision | `f914c4b17429864d9cfeeebabdfaf15aa400d3d6` で固定 (`.env` の `MODEL_REVISION`) |
| 元 | `deepseek-ai/DeepSeek-V4.1-Flash-1017` → `jangq/...-JANG_2K` → dealignai が uncensored 化 |
| 量子化 | JANG profile `JANG_DSV41_Q2Q3_D3_L0-7G3U3_A8_IMATRIX`。routed expert 平均 2.47bit (imatrix 校正)、attention / shared expert / embed / lm_head は 8bit |
| 本体 | 45 shard (共通 `model-0000N-of-00045` 5 本 + routed expert `model-routed-000NN` 40 本) / 211.6GB (197GiB)。起動時にメモリへ載る |
| Engram | `engram/` 2 shard / 203GB。FP8 の n-gram テーブル。**実行中に行単位でディスクから読む** (mmap、メモリには載らない) |
| DSpark | `dspark/` 8.9GB。3 段の MTP draft (routed expert 4bit)。先読みを使うと載る |
| chat | `encoding/encoding.py` (DeepSeek 純正の Python encoder)。thinking は `enable_thinking`、強さは `reasoning_effort` (`low` / `high` / `max` / 1〜100) |

モデルカードの自己申告は次のとおり。

- MMLU-5700: 78.39% (元の JANG_2K は 76.96%)
- HarmBench-320: 100% COMPLY

推奨のサンプリングは temperature 1.0 / top_p 1.0 (`DEFAULT_TEMPERATURE` / `DEFAULT_TOP_P`)。
greedy と max effort を組み合わせると、難問で同じ所をぐるぐる回ることがあると、カードに注意書きがある。

## メモリ (256GB unified)

| 内訳 | 量 |
| --- | --- |
| 本体 (常駐) | 197GiB |
| DSpark (先読みを使うとき) | 8.3GiB |
| KV | jang_tools の DSV4.1 キャッシュは 4bit 常駐で、1M tokens でも約 0.9GB (ソースのコメントより) |
| wired 上限 (`WIRED_LIMIT_MB`) | 245760MiB = 240GiB (OS に 16GiB を残す) |

| 実測: ロード後の MLX active | 220.5GB = 205GiB (本体 + DSpark) |
| 実測: ベンチ後の空き (free + inactive + purgeable) | 約 33GiB、swap 0.7GB |

- Engram はページキャッシュ経由で読む。wired の外にある残りのメモリが、ここで効いてくる。
- 起動前の空きメモリは 215GiB 以上を目安にしている (`preflight.sh`)。
- GLM 版より空きが 40GiB ほど少ない。ほかの大きなアプリは止めておく。
- **サーバの稼働中に、大きなダウンロードなど重い I/O をしない。**
  2026-11-07、GLM 版のサーバ (wired 173GiB) を動かしたまま、この 424GB を取得した。
  その途中で、macOS のカーネルがサーバを強制終了した
  (`memorystatus: killing python3.12 [...] due to low-swap`、`log show` で確認)。
  サーバのログには何も残らない。この構成は wired が 205GiB とさらに大きいので、同じことが起きやすい。
- vmlx は prompt の KV を SSD にも保存する (block disk cache)。
  場所は `~/.cache/vmlx-engine/block-cache/` で、上限は vmlx の自動設定 (このマシンでは約 186GB)。
  抑えたいときは `VMLX_EXTRA_ARGS=--block-disk-cache-max-gb 64` のように指定する。

## 調整ノブ (`.env`)

| キー | 既定 | メモ |
| --- | --- | --- |
| `MAX_TOKENS` | 32768 | リクエストが `max_tokens` を省略したときの上限 (vmlx の既定は 4096) |
| `REQUEST_TIMEOUT` | 3600 | 1 リクエストのタイムアウト秒。長い思考を切らないように伸ばしている (vmlx の DSV4 既定は 900) |
| `DSPARK` | auto | `auto` はバンドルの draft を使う。`off` で先読みなし |
| `DEFAULT_ENABLE_THINKING` | 空 | 空ならテンプレートの既定。`true` / `false` |
| `DEFAULT_TEMPERATURE` / `DEFAULT_TOP_P` | 1.0 / 1.0 | DeepSeek-V4.1 の推奨値 |
| `API_HOST` / `API_KEY` | 0.0.0.0 / 空 | 既定は LAN からも受ける (`http://<この Mac の IP>:8910`)。**キーが空だと LAN の誰でも認証なしで叩ける**。`API_KEY` を設定すると `Authorization: Bearer <key>` (Anthropic 形式なら `x-api-key`) が必要になる。手元だけにするなら `127.0.0.1` |
| `VMLX_EXTRA_ARGS` | 空 | 空白区切りで追加する |

## 速度と正答率 (1 本ずつ)

2026-11-07、M5 Ultra 256GB、既定の設定 (DSpark auto) で測った。結果は `results/dsv41-mac-c1/` (git 管理外)。

### decode

| 条件 | decode |
| --- | --- |
| 固定 256 tokens、1 本 (prose / code 各 3 回) | 中央値 54.3 tok/s (48.6〜59.3)、TTFT 0.15 秒 |
| quality 7 問 (0.8K〜8.6K tokens) | 52.5〜64.1 tok/s |
| 2 本同時 | 1 本あたり 40.6 tok/s、合計 55.1 tok/s。直列に処理されるので合計は伸びない |

DSpark (先読み) の当たり方で、速度が 1 割ほどぶれる。

### cold prefill (1 本、先頭 nonce で prefix cache を外して計測)

| 入力 | TTFT | prefill |
| --- | --- | --- |
| 1K | 1.4〜1.5 秒 | 約 750 tok/s |
| 8K | 6.0〜6.7 秒 | 約 1,360 tok/s |
| 32K | 20〜21 秒 | 約 1,600 tok/s |
| 128K | 80 秒 | 約 1,640 tok/s |

長くなるほど速い。GLM 版 (128K で 545 tok/s) の約 3 倍。

### 正答率 (reasoning_effort=max、enable_thinking、temperature 1.0、max_tokens 32768)

`glm53_eval_tasks.py` の数学 4 問とコード 3 問で、**7/7 正解**。
コードは、生成物を読んでから repo のテストで実行した (`code-grades.json`)。

| 問題 | 結果 | 生成 tokens | 時間 | GLM 版 (tokens / 時間) |
| --- | --- | --- | --- | --- |
| M4 2^2026 mod 1000 | 正解 (864) | 782 | 13 秒 | 1,363 / 36 秒 |
| M1 AIME 2006 | 正解 (738) | 8,588 | 164 秒 | 17,254 / 458 秒 |
| M2 AIME 1999 | 正解 (38) | 2,640 | 43 秒 | 4,166 / 109 秒 |
| M3 桁和 30 の個数 | 正解 (50877) | 1,340 | 22 秒 | 2,666 / 69 秒 |
| C1 転倒数 | PASS | 1,855 | 31 秒 | 3,844 / 101 秒 |
| C2 3×n ドミノ | PASS | 5,783 | 96 秒 | 5,288 / 139 秒 |
| C3 最小窓部分文字列 | PASS | 3,625 | 67 秒 | 12,069 / 320 秒 |

各問 1 回ずつの結果なので、正答率の誤差は大きい。7 問とも簡単すぎて、2 つのモデルの差はまだ見えていない。

### 難問セット (各問 3 回、2026-11-08)

`glm53_eval_tasks.py` の `HARD_MATH` / `HARD_CODE` は、過去問の暗記が効かないように新しく作った 10 問。
正解は `scripts/verify-hard-tasks.py` が、総当たりと別解の 2 通りで計算し直して照合する。
条件は reasoning_effort=max、enable_thinking、temperature 1.0、max_tokens 65536。

結果は **30/30** (数学 18/18、コード 12/12)。30 回とも打ち切りなしで `stop` した。計 2.2 時間。

| 問題 | 内容 | 正解 | 生成 tokens (3 回) | 時間 (3 回) |
| --- | --- | --- | --- | --- |
| HM1 | 4×10 をドミノと 2×2 で敷き詰める数 | 3/3 | 24.7K / 26.3K / 24.5K | 7.1〜7.4 分 |
| HM2 | {1..20} の連続しない部分集合で和が 7 の倍数 | 3/3 | 13.6K / 15.5K / 13.6K | 3.3〜3.6 分 |
| HM3 | n ≤ 10^6 で 1625 \| n²+1 | 3/3 | 4.2K / 3.5K / 2.7K | 0.7〜1.1 分 |
| HM4 | n ≤ 10^4 で n \| 2^n+1 の和 | 3/3 | 38.2K / 40.4K / 29.2K | 8.8〜12.7 分 |
| HM5 | 3×5 グリッドの全域木の数 | 3/3 | 10.4K / 51.2K / 39.5K | 2.6〜13.5 分 |
| HM6 | 合計 10 以上までのサイコロの期待回数 (m+n) | 3/3 | 11.9K / 11.2K / 11.0K | 2.7〜3.1 分 |
| HC1 | 異なる部分文字列の数 (n ≤ 2×10^5) | 3/3 | 5.2K / 3.8K / 7.5K | 1.0〜2.0 分 |
| HC2 | 区間加算と区間二乗和 (n, q ≤ 10^5) | 3/3 | 12.6K / 12.8K / 15.8K | 3.4〜4.5 分 |
| HC3 | 木で距離ちょうど k の頂点対 (n ≤ 2×10^4) | 3/3 | 11.6K / 14.1K / 15.4K | 3.6〜4.8 分 |
| HC4 | [L, R] ≤ 10^18 で隣接桁が違い K の倍数 | 3/3 | 2.6K / 3.8K / 4.3K | 0.8〜1.3 分 |

- コードは 12 本とも、適切なアルゴリズムを選んでいた。
  サフィックスオートマトン、遅延伝播付きの区間木、再帰を使わない重心分解、桁 DP。
  テストは最大でも 2 秒以内に通った。
- decode は 53〜72 tok/s。DSpark の当たり方は問題で違い、HM2 が最も速く、HC3 が最も遅い。
- HM5 は、同じ問題でも 10K〜51K tokens とばらついた。上限の 65536 に近い回があるので、これ以上難しい問題では max_tokens を上げる。
- この難易度でもまだ全問正解なので、モデル間で差を出すには、さらに難しい問題が要る。

```bash
venv-vmlx/bin/python -I scripts/verify-hard-tasks.py         # 正解とテストの検証 (数秒)
cd scripts && python3 bench-glm53-eval.py --model deepseek-v4.1-flash-uncensored --phase quality \
    --suite hard --repeats 3 --max-tokens 65536 --concurrency 1 --output-dir ../results/dsv41-mac-hard
# 生成コード (*-generated.py) を読んでから:
venv-vmlx/bin/python -I scripts/run-code-tests.py --reviewed results/dsv41-mac-hard
python3 scripts/summarize-glm53-eval.py results/dsv41-mac-hard   # summary.json の "hard"
```

## ベンチの回し方

GLM 版と同じスクリプトを使う。vmlx には `/metrics` が無いので、`bench-glm53-eval.py` はストリーミングから TTFT と decode を測る (backend `plain`)。
cold prefill は、先頭に nonce を置いて prefix cache を外している。外れたかどうかは、サーバ側では確認できない。
vmlx は `max_tokens=1` の応答テキストを空で返すので、その場合は完了までの時間を TTFT として使う。

```bash
M=deepseek-v4.1-flash-uncensored
python3 scripts/bench-glm53.py --model $M --label mac-dsv41 --output results/dsv41-mac.json
D=results/dsv41-mac-c1 && mkdir -p $D && cp results/dsv41-mac.json $D/fixed-output-c1-c2.json
cd scripts && python3 bench-glm53-eval.py --model $M --phase quality --concurrency 1 --output-dir ../$D
cd scripts && python3 bench-glm53-eval.py --model $M --phase profile --concurrency 1 --output-dir ../$D
python3 scripts/summarize-glm53-eval.py $D
```

## このブランチで消したもの

`scripts/prepare-model.sh`、`scripts/prepare-glm53-mlx.py`、`scripts/rapid_mlx_launch.py` は GLM / Rapid-MLX 専用なので削除した
(GLM 版のブランチに残っている)。
`README.dspark.md`、`presets/`、`patches/` は、さらに前のブランチから引き継いだもので、この構成では使わない。

## 出典

- [dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-JANG_2K](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-JANG_2K): 使用する重みとモデルカード
- [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): 元モデル
- [jjang-ai/vmlx](https://github.com/jjang-ai/vmlx) (PyPI `vmlx` 1.6.84): ランタイム
- [jjang-ai/jangq](https://github.com/jjang-ai/jangq) (PyPI `jang` 2.5.53、`jang_tools`): JANG 形式と DSV4.1 runtime
