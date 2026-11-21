# DeepSeek V4.1 Flash UNCENSORED / vLLM on 3 DGX Sparks

DeepSeek V4.1 Flash 1017 のネイティブ FP8/FP4 checkpoint を、DGX Spark 3台の vLLM（TP=3）で配信する。
このブランチで使うプロファイルは **4 KiBカーネル・1,048,576トークン・最大2件同時**。
OpenAI互換APIはheadの `:8910/v1`、モデル名は `deepseek-v4.1-flash-1017-uncensored`。
vLLMのビルド・配布・起動はheadの `bin/spark` で操作する。リポジトリ直下のDocker Composeは旧SGLangへの切り戻しに使う。

上流の `spark-ds41f-1120`（`d3d40d941f2752fea95aa770eb10d41b5ad5aa1a`）を
3台の `.cache/spark-ds41f-1120` に配置する。管理系は `enP7s7`、TP=3 の通信は
各辺2経路の RoCE triangle。各ノードの既存の dealignai FP8/FP4 重みを直接読み、
head のポートとモデル名はこのリポジトリの `.env` から引き継ぐ。
キャッシュとコンテナ名は vLLM 専用。起動とメモリ監視には上流の `bin/spark` を使う。

| プロファイル（上流 checkout の config/ 内） | ホストページ | 1リクエストの上限 | 最大同時シーケンス | rankごとのKV |
| --- | ---: | ---: | ---: | ---: |
| `llm-4k-1m-x2.local.json`（今回の切り替え用） | 4 KiB | 1,048,576 | 2 | 2.20 GiB |
| `llm-1m.local.json` | 64 KiB + Memory Saver | 1,048,576 | 8 | 4.90625 GiB |
| `llm-256k.local.json` | 4 KiB | 262,144 | 8 | 2.20 GiB |

コンテキスト長には出力も含む。今回の4 KiB設定では最大2件を保持し、prefillは同時1件・1バッチ最大4,096トークン。
1M×2件はKV容量上の見積もりで、dealignai重みでの長文prefill・同時生成は切り替え後に検証する。
[上流のリリースと各shapeの容量](https://github.com/christopherowen/spark-ds41f/tree/spark-ds41f-1120)

## 3台の準備

全台で同じユーザー、同じリポジトリパス（例 `~/repos/llm-container`）を使う。各ノードのNVMeに同じrevisionの重みを置く。headからは自分自身を含む3ホストに名前でSSH接続できること。

| rank | role | SSH名 | f0 の接続先 | f1 の接続先 |
| ---: | --- | --- | --- | --- |
| 0 | head | `spark-a814` | worker2 | worker1 |
| 1 | worker1 | `spark-aa30` | head | worker2 |
| 2 | worker2 | `spark-d544` | worker1 | head |

**新規checkout**では、[.env.example](.env.example) を `.env` にコピーし、各台の `ip -4 addr show dev enP7s7` で確認した管理IPを3台共通の値として記入する。
生成スクリプトが使うのはテンプレート冒頭のモデル、API、IP、インターフェースの項目。コンテキスト長・並列数・KV容量はプロファイルで選ぶ。
重みの初回取得には `hf` CLIと各台に約475 GiB（ビルド・キャッシュは別途）が必要。
各台で `./scripts/fetch-model.sh` を実行すると固定revisionを取得する。[モデルカード](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8/tree/c19ca5ca4eaeb79d29fdf08aea8327da95b1116a)

**既存の3台**には重み・`.env`・生成済みのプロファイルを配置済み。新規セットアップ時やDHCPで管理IPが変わったときは、各台で以下を実行する。
スクリプトは config/index のハッシュ、48 shard のヘッダーとサイズ、管理IP、
各HCAのRoCEv2 GIDとサブネットを検査する。ローカルのチェックアウトに生成するJSONには
接続情報とモデルの場所だけが入り、HFトークンは含めない。

```bash
cd ~/repos/llm-container
python3 scripts/prepare-spark-ds41f.py
```

以降の `bin/spark` コマンドはheadから実行する。

```bash
cd ~/repos/llm-container
cd .cache/spark-ds41f-1120
profile=config/llm-4k-1m-x2.local.json
bin/spark --cluster-config "$profile" doctor
bin/spark --cluster-config "$profile" render spark-a814
bin/spark --cluster-config "$profile" cluster start  # 計画の表示
```

`doctor` の既知の差分は、上流のGLP用環境変数2個と `serve_args`。
このサイトは既存のdealignaiモデルを1つのAPI名で提供し、GLPベクトルを追加しない。

## メンテナンスに入る前の確認

- 各台でSSHユーザーをrootful Dockerのグループに追加する（例 `sudo usermod -aG docker y_nagata`）。上流CLIがSSH越しに行うホスト検査・メモリ監視サービスの起動停止に非対話sudoを許可し、新しいログインセッションで確認する。headのシェルでrootlessの `DOCKER_HOST` を設定している場合は `unset DOCKER_HOST`。Dockerグループにはホストのroot相当の操作権限がある。
- TP3はディスプレイ用の予約メモリに重みの一部を置くため、各台で `sudo cat /sys/module/nvidia_drm/parameters/modeset` を確認する。2026-11-21時点で3台とも `nvidia-drm-options-modeset0` が導入済みで、起動時の設定は `modeset=0`。実行中の値はrootで読み取る。上流は7.0系でメモリ逼迫時のRDMA登録失敗を避けるため `kho=off` も使う。worker2のみ設定済み。[上流のホスト設定・復元手順](https://github.com/christopherowen/spark-ds41f/blob/spark-ds41f-1120/docs/setup.md)
- 今回の `llm-4k-1m-x2.local.json` は各台の `getconf PAGESIZE` が `4096` の状態で使う。同じカーネルで上流の256K設定に戻す場合は `config/llm-256k.local.json` を選ぶ。上流の4 KiB検証カーネルは `7.0.0-1019-nvidia`。
- 別構成の `llm-1m.local.json` を選ぶ場合のみ、上流の検証組合せは `7.0.0-1019-nvidia-64k`、driver 580.178.04、署名済みMemory Saver 0.4.1。各台の `getconf PAGESIZE` が `65536`、`sudo cat /sys/module/nvidia_uvm/parameters/uvm_pack_sysmem_leaf_tables` が `Y` であることを確認する。

```bash
# headの新しいログインセッションで、ビルド用Dockerと各SSH先の権限を確認
unset DOCKER_HOST
docker info --format '{{.ServerVersion}}'
for host in spark-a814 spark-aa30 spark-d544; do
  ssh -o BatchMode=yes "$host" 'hostname && docker info --format "{{.ServerVersion}}" && sudo -n true && sudo -n cat /sys/module/nvidia_drm/parameters/modeset' || break
done
```

## 切り替え（メンテナンス中に実行）

1. リクエスト終了後、各ホストの `~/repos/llm-container` で、対応する1行を実行してSGLangを停止する。コンテナは残るので、同じものを切り戻しに使える。

   ```bash
   sudo docker compose --profile head stop head           # spark-a814
   sudo docker compose --profile worker1 stop worker1     # spark-aa30
   sudo docker compose --profile worker2 stop worker2     # spark-d544
   ```

   ホスト設定を変える場合はこの停止中に行う。`modeset=N` の台では上流手順で
   `sudo apt-get purge nvidia-drm-options-modeset0`、`sudo update-initramfs -u -k all` を実行する。
   headとworker1で上流と同じ `kho=off` に揃える場合は `sudo apt-get install nvidia-spark-grub-kho` と `sudo update-grub` も実行する。
   変更した台を再起動し、`modeset` が `Y`、`getconf PAGESIZE` が `4096`、管理IPが各台の `.env` と一致することを確認する。
   DHCPアドレスが変わった場合は全台の `.env` を揃えて設定を再生成する。

2. 選んだプロファイルのホスト準備後、headでビルド・配布する。ビルドは空き64 GiB以上とGPUを使用する。イメージは全プロファイル共通。3台に同じコミットとサイト設定があるため、配布対象はイメージだけ。

   ```bash
   cd ~/repos/llm-container/.cache/spark-ds41f-1120
   profile=config/llm-4k-1m-x2.local.json
   bin/spark --cluster-config "$profile" build prepare &&
   bin/spark --cluster-config "$profile" build image --apply &&
   bin/spark --cluster-config "$profile" build distribute --apply
   ```

3. headで起動計画とホストの警告を確認する。選んだページサイズ、ドライバ、メモリ監視、RoCE、イメージを確認したら起動する。初回JITが完了したら、モデル名と短い生成を確認する。

   ```bash
   bin/spark --cluster-config "$profile" cluster start
   bin/spark --cluster-config "$profile" cluster start --apply
   curl -fsS http://127.0.0.1:8910/v1/models
   curl -fsS http://127.0.0.1:8910/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"deepseek-v4.1-flash-1017-uncensored","messages":[{"role":"user","content":"19+23? Reply with the number."}],"max_tokens":32,"chat_template_kwargs":{"thinking":false}}'
   ```

   1M×2の受け入れ確認では、異なる長文でまず1件を完走させ、次に1件目の生成中に2件目を投入する。各件は入力と出力を合わせて1,048,576トークン以内とし、両方の完了と各ノードの空きメモリを記録する。

## 切り戻し

headで起動に使った `profile` を指定して `bin/spark --cluster-config "$profile" cluster stop --apply`。
3台のvLLM停止を確認する。64 KiBに移行したホストも元の検証環境へ戻す場合は、上流の復元手順で元の4 KiBカーネルを選んでから再起動する。
各ホストの `~/repos/llm-container` で `sudo docker compose --profile worker2 start worker2`、
`sudo docker compose --profile worker1 start worker1`、`sudo docker compose --profile head start head`
の順に、対応する1行だけを実行する。headで `curl -fsS http://127.0.0.1:8910/health` を確認する。
