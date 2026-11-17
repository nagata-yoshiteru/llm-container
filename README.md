# DeepSeek V4.1-Flash-1017 UNCENSORED / 3 DGX Spark RING

`dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8` を SGLang TP=3 / EP=1 で動かす構成。
既存と同じ `docker-compose.yml`、全3台共通の `.env`、`head / worker1 / worker2` profile を使う。
API は head の8910番。各ホストでモデルプロセスを1つ起動する。

**設定・CPU検証まで実施。Dockerビルド、モデル取得、Engram生成、GPU推論は未実施。**
元のGLM用vLLM / mentatからSGLangへ変更する。重み名だけを変えてGLMのTPパッチを流用することはできない。

## 採用方式

[モデル](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8)の
ネイティブFP8 dense + FP4 expertsを使用する。
[3 Spark対応実装](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/tree/64b02dea8681fb1d8e3a659ea432a8e9e9ba3995)の
TP=3パディングとNVMe Engram adapterを組み込む。
Engramを全量RAMに置くとSparkの共有メモリを使い切るため、各ノードで必要な行だけローカルNVMeから読む。
重みの追加量子化は行わないが、上流の省メモリ用 `wo_a` / draft LM head FP8 adapterは有効。
このcheckpointと今回のNCCLビルドの組み合わせは実機検証が必要。

| 項目 | 既定値 |
| --- | --- |
| モデルrevision | `c19ca5ca4eaeb79d29fdf08aea8327da95b1116a` |
| SGLang adapter revision | `64b02dea8681fb1d8e3a659ea432a8e9e9ba3995` |
| 並列化 | TP=3 / EP=1 / 1 GPU per node |
| Engram | `OFFLOAD_MODE=nvme`、RAM cache=0、resident scales=0 |
| 推論コンテキスト / 最大同時件数 | 32768 tokens / 2 |
| KVプール / prefill chunk | 131072 tokens / 512 |
| 推測デコード | DSpark block size=3、confidence cap無効 |
| メモリ割合 | 0.95（GLMのGPU_MEMORY_UTILIZATIONとは別設定） |
| API model名 | `deepseek-v4.1-flash-1017-uncensored` |
| 重み / Engram追加容量 | 約475 GiB / 約63 GiB、各ノード |

全48 shardにEngram・vision・DSparkが含まれる。元モデルの `config.json` は書き換えず、
TP=3のパディングはロード時に実施する。各ノードに約550 GiBの空きとビルド用の追加容量を用意する。
旧モデルの重みや既存Engramは削除・流用しない。

同作者の[EXL3版](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED)もあるが、
そのモデルカードの実行系はlayer splitであり、今回の3ホストTP構成への置換には使わない。

## 確認した実機構成（2026-11-16 JST）

3台ともSSHで `NVIDIA GB10` を確認。3辺それぞれの両PCIドメインで9000-byte DF pingに成功。
管理IPは旧 `.env` のDHCPアドレスから変わっていたため、ローカル `.env` を次の値へ修正済み。
リモートホストの設定・コンテナは変更していない。配布前にDHCPを再確認する。

| role / rank | hostname | 管理IP | f0 root0 / root1 | f1 root0 / root1 |
| --- | --- | --- | --- | --- |
| head / 0 | spark-a814 | 192.168.207.160 | 192.168.4.1 / 192.168.5.1 | 192.168.0.1 / 192.168.1.1 |
| worker1 / 1 | spark-aa30 | 192.168.207.158 | 192.168.0.2 / 192.168.1.2 | 192.168.2.2 / 192.168.3.2 |
| worker2 / 2 | spark-d544 | 192.168.207.159 | 192.168.2.3 / 192.168.3.3 | 192.168.4.3 / 192.168.5.3 |

root0は `enp1s0f*`、root1は `enP2p1s0f*`、各リンクは/24・MTU 9000。
管理LANの `enP7s7` をrendezvous/Gloo/NCCL bootstrapに使用する。
確認時の空きディスクは約1.5 / 1.6 / 2.1 TiB、MemAvailableは約113–115 GiBで、
重み取得に必要なディスク容量は3台とも確保できていた。

NCCLは2.30.7の固定ソースに
[dual-PCI-domainパッチ](https://github.com/FujitsuPolycom/sparkring/blob/dcfc97430ea8095dd31233b6595bdd14a4b63097/spark_transport/nccl/DUAL_PCI_DOMAIN.md)
を適用し、SGLangと同じベースイメージ内でビルドする。pip NCCLを置き換えて二重ロードを避ける。
4 HCAのGID広告、同じPCIドメインを優先するサブネットルーティング、Ring collectiveを有効にする。
3台では全ペアが直結しているので、上流の4台専用launcher / `NCCL_SWITCHLESS_RING_ONLY` は使わない。
GID indexは各ホストの有効なRoCEv2/IPv4から起動前に検出する。
ping成功はRDMA collective成功の保証ではない。初回起動時にNCCL接続ログの確認が必要。

## 準備（各ホスト、sudo不要）

コードと同じ `.env` を全3台へ配置する。既存の `.env` は更新済みなので `cp` は不要。
新規checkoutのみ `cp .env.example .env` して、3台の管理IPを入力する。
`.env` はCompose用なので、空白を含む値もある旧形式と同様 `source .env` はしない。

Python 3.12以降を使用する。ホストでの重み取得・Engram packには `huggingface_hub` と `numpy` が必要。
既存のPython環境に無い場合は、例えば次の専用venvを作る。

```bash
python3 -m venv .cache/deepseek41-tools
source .cache/deepseek41-tools/bin/activate
python -m pip install huggingface_hub numpy
# モデルがgatedの場合、Hugging Faceでアクセス承認を受けたアカウントを使用する。
hf auth login

# 公開ソースのみ取得。アーカイブ/パッチのSHA256を検証し、Dockerは起動しない。
python3 scripts/prepare-deepseek41-tp3.py
# 約475 GiB。全3台で同じコマンド、同じパスへ取得（中断後は再実行可）。
./scripts/fetch-model.sh
# 各ホストで自分のroleだけを指定。約63 GiBのローカルEngram packを生成。
python3 scripts/prepare-deepseek41-tp3.py --pack head
# worker1では --pack worker1、worker2では --pack worker2
./scripts/preflight.sh --role head
# worker1 / worker2でも同様に実行
```

packは各ノードのローカルNVMeに置く。rankとモデルrevisionを検証し、別モデルのpackを拒否する。
起動前に48 shardのサイズ・header・config/indexハッシュとpackを検査する。
この検査は重み全体のSHA256再計算ではない。重み自体の取得・再開は固定revisionの `hf download` に任せる。

ソースは `.cache/deepseek41-tp3/{upstream,nccl}`、キャッシュと起動記録は
`.cache/deepseek41-tp3/runtime/{cache,state}`。ソース変更はreceipt検証で拒否する。
以前の `.cache/kindling-tp3` / `vllm-cache` は再利用しない。

## 切り替え・起動（ユーザーがsudoで実行）

**先に全3台の旧GLMモデルとmentatを停止する。** 新Composeからは旧mentatを停止できない。
元ブランチのComposeを使える状態で、各ホストで次を実行してからコード/.envを切り替える。

```bash
# 旧GLM設定のまま、各ホストでそのroleを指定
sudo docker compose --profile head down --timeout 60
# worker1 / worker2でも同様。旧設定への切り戻しも全ノードで揃える。
```

すでに新コードに切り替えている場合は、`sudo docker ps` で旧コンテナ名を確認して停止する。
旧版の名前は `glm53-fp8-head/worker1/worker2` と `glm53-mentat-head/worker1/worker2`。
準備とpreflightが完了したら、各ホストで以下を実行する。

```bash
# worker2上 → worker1上 → head上の順
sudo docker compose --profile worker2 up -d
sudo docker compose --profile worker1 up -d
sudo docker compose --profile head up -d
```

初回は各ノードで `Dockerfile.deepseek41` をビルドする。
起動前にビルドを済ませる場合は `sudo docker compose --profile <role> build`。
各ノードのNCCL版・ソースrevisionを揃えること。

```bash
sudo docker compose --profile head logs -f head
curl -fsS http://192.168.207.160:8910/health
curl -fsS http://192.168.207.160:8910/v1/models
curl -fsS http://192.168.207.160:8910/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash-1017-uncensored","messages":[{"role":"user","content":"19 + 23 = ? Reply with only the number."}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"thinking":false}}'
# 停止: 各ノードでそのrole
sudo docker compose --profile head down --timeout 60
```

headだけにHTTP healthcheckがあり、workerにはHTTPサーバーを期待しない。
起動時にモデル検査が通っても、CPU/GPU共有メモリ、JIT、RDMA、品質は実推論で確認する必要がある。
まず算術・日本語・ツール呼び出し・同時2件を確認し、メモリ監視下で長文を段階的に増やす。
ネイティブの最大1Mを、この3台構成で保証するものではない。画像/動画も今回未検証。

## 静的検証と旧資料

```bash
python3 scripts/test-deepseek41-config.py
./scripts/preflight.sh --network-only  # モデル未取得でもリンクだけ確認可能
```

テストはCPUとComposeの設定展開だけを使い、daemon操作・sudo・重みロードはしない。
ソース固定値は `patches/deepseek41/source-lock.json`。
旧設定は `README.glm53-3spark.md` と `README.deepseek41-exl3.md` に記録を残した。
旧 `presets/`・GLM用テスト/ベンチは現行SGLang設定用ではない。
ローカルの元.envは `.env.before-deepseek41-1017.bak`。切り戻す場合は.envだけでなく元ブランチの設定一式を戻す。

Based on [dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8).
SGLang adapterはMiaAI-Lab、NCCL routing patchはFujitsuPolycom/sparkringとその上流の実装を使用する。
ソース配布時はそれぞれのLICENSEを保持する（取得ツリーとNCCLビルド内に保持）。
