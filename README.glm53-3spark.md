# GLM-5.3-Flash UNCENSORED / kindling TP=3 / 3 DGX Spark

通常の `docker-compose.yml` と全3台共通の `.env` で起動する。
役割はこれまでと同じ `--profile head` / `worker1` / `worker2`。
各 profile はモデルとローカルの mentat daemon を1つずつ起動する。
API は head の8910番、モデル名は `glm-5.3-flash-uncensored` と `glm-5.3-flash`。

コード・env・モデル準備と CPU / Compose 検証は完了している。
**新イメージの実ビルド、TP=3 の GPU 起動、速度・品質はまだ未検証**。
前回の FP8 / MTP k=3 ベンチは旧 TP=1 / DP=3 構成の結果。

## 起動・停止

今回の3ホストでは、固定ソースと TP=3 用モデルは準備済み。
新しい共通 `.env` も全3ホストへ配布する。
コードを commit して各ノードで pull したら、次の手順で切り替える。

1. **まず全3ノードの旧コンテナを停止**する。旧 DP と新 TP を混在させない。
   各ホストで `sudo docker compose --profile <役割> down --timeout 60`。
2. worker2 → worker1 → head の順で、従来どおり起動する。

```bash
# worker2 上
sudo docker compose --profile worker2 up -d
# worker1 上
sudo docker compose --profile worker1 up -d
# head 上
sudo docker compose --profile head up -d
```

初回の `up` は Dockerfile.kindling からイメージをビルドするため時間がかかる。
以降のビルドは Docker のレイヤーキャッシュを使う。
明示的にビルドだけを行う場合は `sudo docker compose --profile <役割> build`。
イメージは各ホストのローカルに作成する。

```bash
# head のログ。worker では head を worker1 / worker2 に読み替える。
sudo docker compose --profile head logs -f head
curl -fsS http://192.168.0.1:8910/health
curl -fsS http://192.168.0.1:8910/v1/models
# 停止すると、そのホストの mentat daemon も停止する。
sudo docker compose --profile head down --timeout 60
```

全ノードで TP=3 / MTP k=3、target と MTP の `tp3pad`、dense FP8、
KV 容量と CUDA Graph のログを確認してからベンチを行う。
rootful Docker を使用する。コミット・pull・ビルド・再起動はユーザー側で実施する。

## 新規ホストでの準備

```bash
cp .env.example .env
# SOURCE_MODEL_PATH へ元の checkpoint を取得する。取得済みなら不要。
./scripts/fetch-model.sh
# 固定 kindling ソースと submodule を取得し、TP=3 用 serving directory を作る。
python3 scripts/prepare-glm53-tp3.py
./scripts/preflight.sh
```

準備は sudo 不要で、Docker を起動しない。同じ生成物への再実行は検証だけ行う。
config / index が異なるモデルや、手で変更された serving directory は上書きしない。
3台のパスと `.env` を揃え、ノードの区別には profile を使う。
`.env.kindling-tp3` や専用の `run.sh` は使用しない。

## 構成と重み

[kindling](https://github.com/kindlingai/glm-5.3-flash-gx10/tree/c748079d45e6e070b2acb108a91edfe52f4a7747)
を `c748079d45e6e070b2acb108a91edfe52f4a7747` に固定する。
ビルドソースは `.cache/kindling-tp3/upstream`、vLLM nightly は
`ddd6fbca148a867aad1fcab7ec72f582b9977db4`、mentat は0.17.1。
`Dockerfile.kindling` はこの固定版の Dockerfile を基に、必要な overlay をイメージへ含める。
旧 v8 用のホスト側パッチはマウントしない。

| 項目 | 初期値 |
| --- | --- |
| 並列化 | TP=3、3ホストで1つのモデル |
| 元モデル | dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 |
| MTP | k=3、元 checkpoint の MTP head |
| MoE | Marlin、元の W4A16 NVFP4 |
| Dense | kindling のロード後 FP8。追加4-bit化と LM head FP8 は無効 |
| 最大コンテキスト | 524288 tokens（この構成の1Mは未検証） |
| KV | 12 GiB固定、fp8_e4m3、block-size=3456 |
| 最大同時実行数 | 8 |
| prefill budget | 8192 tokens / step、長文1本の閾値7168 |
| CUDA Graph | FULL_DECODE_ONLY、4/8/16/32 tokens |
| 通信 | dual rail RoCE、光スイッチ、MTU 1500を維持 |

元モデルは `SOURCE_MODEL_PATH`、serving directory は `MODEL_PATH`。
後者の safetensors は元ファイルへの**ハードリンク**で、巨大な複製は行わない。
config だけを別ファイルにする。両ディレクトリの shard は直接編集しない。

TP=3 用に MLA / KDA heads を64→66、MoE intermediate を2048→2304とし、
ロード時にゼロ埋めしてから各 rank に分割する。最初の3層の BF16 dense
intermediate=12288 は変更しない。vocab_size=154880 は維持し、
embedding の保存領域だけ154944へ拡張する。
`patches/kindling/patch-mtp.py` は MTP ローダーにも同じ処理を追加する。
ビルド時に対象ソースの SHA256 を検証し、異なる版への適用を拒否する。

この uncensored checkpoint は `input_activations=null` で activation scale がない。
kindling の ModelOpt overlay が W4A16 と判定し、Marlin 経路を選ぶ。
W4A4 前提の CUTLASS / megamoe と fused MoE prefill は使わない。
arx 通信、sequence parallel、RecoverSSM、dense FP8 は有効。
kindling の dense FP8 は以前の限定パッチより対象が広く、KDA f-b/g-b と
MTP の対応 Linear も含むため、起動後に品質を再確認する。
上流の速度は別 checkpoint / DFlash / W4A4 / 通信条件の結果で、今回の保証値ではない。

重み snapshot と JIT cache は `.cache/kindling-tp3/runtime/cache`、
ログは同ディレクトリの `logs`。旧構成の `vllm-cache` は再利用しない。
GID index は `.env` で固定せず、kindling が各ホストの実際のアドレスから導出する。
mentat は6379/6380、管理画面は8082を使用する。

## 検証とベンチ

CPU Torch のある環境で、パディングと MTP パッチを確認できる。
MTP_SOURCE は上記 vLLM 固定コミットの `vllm/models/glm5next/common/mtp.py`。

```bash
python3 scripts/test-glm53-tp3-padding.py .cache/kindling-tp3/upstream "$MTP_SOURCE"
python3 scripts/test-glm53-compose.py
```

GPU 起動後、他のリクエストがない状態で同条件を測定する。

```bash
python3 scripts/bench-glm53.py --label kindling-tp3-mtp3 --output results/glm53-kindling-tp3.json
python3 scripts/bench-glm53-eval.py --phase quality --output-dir results/glm53-kindling-tp3-quality
python3 scripts/bench-glm53-eval.py --phase profile --output-dir results/glm53-kindling-tp3-profile
python3 scripts/bench-glm53-eval.py --phase retrieval --output-dir results/glm53-kindling-tp3-retrieval
```

固定出力ベンチは temperature=0 / seed=42 / reasoning_effort=max、
256 tokens の生成を1・2並列で各3回測る。profile は6並列と cold prefill。
quality のコード問題は生成コードを保存するだけなので、内容を確認してから
テストを実行する。思考も completion_tokens に含め、SSEイベント数で数えない。
