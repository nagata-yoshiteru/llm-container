# DeepSeek V4.1-Flash-1017 UNCENSORED / 3 DGX Spark RING

`dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8` を SGLang TP=3 / EP=1 で動かす構成。
既存と同じ `docker-compose.yml`、全3台共通の `.env`、`head / worker1 / worker2` profile を使う。
API は head の8910番。各ホストでモデルプロセスを1つ起動する。

**CPU検証と修正NCCLの3台AllReduce検証済み。モデル推論の成功は未確認。**
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

## 重みロード後のNCCLメモリ登録エラー

2026-11-17のheadログでは、target/draftのロードとKVプール確保後、最初の
`torch.distributed.all_reduce` で別のNCCL communicatorが初期化され、
`ibv_reg_mr_iova2 failed with error Cannot allocate memory` で停止した。
その直前のGPU空き表示は約11 GB。さらにpre-warmを有効にした再試行でも、
worker1が重みロード前に同じ登録エラーを返した。Composeにはすでに
`memlock: -1` と `IPC_LOCK` があり、モデル容量を減らすだけでは解決しない。

再試行用に `--pre-warm-nccl` でPyTorchのNCCL通信を重みロード前に初期化し、
`NCCL_MAX_NCHANNELS=4` で通信バッファと登録数を減らす。
この2点だけでは解消しなかった。次のv2イメージで通信バッファの確保方法も変更する。

### v2: NCCLのNETバッファをcudaHostRegisterへ変更

ホスト上の小さなプログラムで、`cudaHostAlloc` で確保した領域を分割して
`ibv_reg_mr_iova2` へ登録すると、モデルなしでも `ENOMEM` を再現した。
通常RAMの確保後に `cudaHostRegister` する方式では比較テストが成功した。
ドライバ内部の根本原因は未確定だが、この環境で失敗する確保経路を切り分けられた。

`patches/deepseek41/nccl-net-host-register.patch` はNET通信バッファだけを
ページ境界に揃えた通常RAM + `cudaHostRegisterMapped` へ変更する。
対応する解放も `cudaHostUnregister` + `free` に揃え、CPU/GPUのアドレス一致を検査する。
モデル・KV・他のCUDA allocationやRoCEルーティングは変更しない。
`NCCL_NET_HOST_REGISTER=1` はこのローカルパッチ専用の設定で、0なら旧経路に戻る。

2026-11-18 JST、3台のホスト上で同じ修正ライブラリを使い、設定0でworker1の
登録エラーを再現、設定1で4通信グループ×3サイズ（float32の1 / 1024 / 262144要素）の
AllReduceと全rankの結果照合・解放が成功した。検証用ビルドはSM121のfloat32 Ring
AllReduceに限定しており、本番Dockerビルドは通常どおり全collectiveを含む。
検証コードは `scripts/test-deepseek41-nccl.cpp`。コンテナ内のモデル起動・推論は別途確認する。

**v1からv2への更新には全3台で再ビルドが必要。** 更新したDockerfile・Compose・patches・
scripts・`.env` を揃え、まず各ホストで自分のroleのイメージをビルドする。

```bash
# worker2上 / worker1上 / head上で、それぞれ対応する1行を実行
sudo docker compose --profile worker2 build worker2
sudo docker compose --profile worker1 build worker1
sudo docker compose --profile head build head
```

`.env` の `SGLANG_IMAGE` は `llm-deepseek41-sglang:tp3-1017-v2`。
全台のビルド完了後に下記の順で再作成する。`restart` では旧イメージのままなので不可。

更新した `docker-compose.yml` と `.env` を全3台に揃え、まず各ノードで該当roleを停止する。

```bash
# 各ホストで自分のroleを指定（worker1 / worker2でも実行）
sudo docker compose --profile head stop --timeout 60 head
./scripts/preflight.sh --role head
```

全3台の停止・preflight完了後、各ホストでworker2 → worker1 → headの順に再作成する。
v2イメージのビルドを先に完了させる。`restart` だけではイメージや環境変数が更新されない。

```bash
# worker2上
sudo docker compose --profile worker2 up -d --no-build --pull never --force-recreate worker2
# worker1上
sudo docker compose --profile worker1 up -d --no-build --pull never --force-recreate worker1
# head上
sudo docker compose --profile head up -d --no-build --pull never --force-recreate head
sudo docker compose --profile head logs -f head
```

NCCLログの `NCCL_NET_HOST_REGISTER set by environment to 1`、
重みロード前のNCCL warmup完了、上記の `/health`・推論リクエストの成功を確認する。
再発時は全3台のコンテナログに加え、失敗時刻付近の `sudo journalctl -k` を採取する。
稼働中なら `sudo docker exec deepseek41-head sh -c 'ulimit -l; cat /proc/meminfo'`
で実際のmemlock制限とホスト共有メモリを確認する（workerはコンテナ名を変更）。

### NCCL初期化直後にworkerが終了する場合

続く再試行では、worker1が `ncclCommInitRank` 中に `unhandled system error`、
head/worker2が `remote process exited or there was a network error` で終了した。
これは重みロード前かつpre-warm collectiveより前の失敗で、前回のメモリ登録エラーと
同じ原因とはまだ確認できない。3台の `.env` と起動記録のpre-warm設定は一致している。
2026-11-18 JSTのホスト上の検査では、全6リンクで1 MiB×100回の双方向
`ib_write_bw` が成功した。ただし、コンテナ内のNCCL collectiveやCUDAメモリの登録は未検証。

提供ログではNCCL出力が行の途中で途切れ、system errorの詳細が残っていなかった。
`boot.py` はengineのstdoutをpipeで受けるため、NCCLのC stdioバッファが強制終了時に
失われる可能性がある。entrypointで `NCCL_DEBUG_FILE` を指定し、NCCL側の行バッファを
使って、起動ごと・プロセスごとのログをホストの次の場所へ保存する。
この変更は診断用で、起動障害を解消したものではない。

```text
.cache/deepseek41-tp3/runtime/state/nccl/rank*-<起動時刻>-<hostname>-<pid>.log
```

更新した `scripts/entrypoint.sh` を全3台に揃えて再起動する。bind mountなので
イメージ再ビルドは不要。NCCL詳細はDockerログから上記ファイルへ移り、Pythonの
例外やモデルの進行状況は引き続きDockerログに出る。
再発後はまずworker1の最新ログを確認する（他の2台にも同じ場所に保存される）。

```bash
ls -lt .cache/deepseek41-tp3/runtime/state/nccl/
tail -n 200 .cache/deepseek41-tp3/runtime/state/nccl/<対象ログ名>
```

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
