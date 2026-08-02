# DeepSeek-V4-Flash-0731 on 2x DGX Spark

[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(304B MoE / FP8+FP4 / 167GB) を DGX Spark 2 台に **TP=2** で分割して、
OpenAI 互換 API を `:8910` に生やす構成。

- 分散バックエンドは **Ray ではなく `mp`** (torch.distributed SPMD)。head/worker が
  それぞれ `vllm serve` を `--nnodes/--node-rank/--master-addr` 付きで起動する。
- ノード間の NCCL は 200GbE QSFP 直結リンクの **RoCEv2 (RDMA)**。
- **native DSpark speculative decoding (k=7)** を使う。0731 が同梱している draft
  モジュールで、蒸留なしで速度を出している本体。

| | head | worker |
|---|---|---|
| 役割 | rank 0 / API `:8910` | rank 1 / `--headless` |
| NIC | QSFP 直結の `enp1s0f0np0` | 同左 |
| HCA | ケーブルが刺さっている側 (既定 `rocep1s0f0`, GID index 3 = RoCEv2/IPv4) | 同左 |

IP・ホスト名・パスはすべて `.env` に書く。`.env` は git 管理外。
以下の手順では worker を 2 通りの経路で呼ぶ。

```bash
WORKER=<worker の RoCE IP>       # QSFP 直結側。.env の WORKER_ROCE_IP と同じ値
WORKER_MGMT=<worker の管理用 IP> # 普通の Ethernet / tailscale 側
```

**QSFP 側の設定をいじるときは必ず `$WORKER_MGMT` 経由で SSH すること。**
`$WORKER` でログインしたまま MTU やアドレスを変えると自分の足を撃つ。

---

## Docker は rootful を使う

この構成は `--device /dev/infiniband` / `memlock unlimited` / host network での RDMA が
必要なので、**rootless では動かない**。この repo の docker コマンドは全部 `sudo` を付ける。

```bash
docker  ...        # rootless (このホストのデフォルト) -- 使わない
sudo docker ...    # rootful  -- こっちを使う
```

`sudo docker compose` はカレントディレクトリの `.env` をそのまま読む。ただし
**`.env` の中で `${HOME}` などのシェル変数は使えない** (sudo で `HOME=/root` になるため)。
`MODEL_PATH` は repo 相対 (`./models/...`) か絶対パスで書くこと。

### rootful daemon に nvidia ランタイムを登録する (最初に 1 回)

rootless 側 (`~/.config/docker/daemon.json`) に nvidia ランタイムが登録してあっても、
**rootful 側 (`/etc/docker/daemon.json`) には効かない**。未登録のまま起動すると
コンテナ内で NVML が初期化できず、vLLM が起動直後に
`RuntimeError: Failed to infer device type` で死ぬ。

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
ssh -t "$WORKER_MGMT" 'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker'
```

確認 (**これが通るまで vLLM を起動しない**):

```bash
sudo docker info | grep -i runtimes          # nvidia が出ること

IMAGE=$(sed -n 's/^VLLM_IMAGE=//p' .env)
sudo docker run --rm --gpus all \
  --device /dev/nvidia0 --device /dev/nvidiactl \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  "$IMAGE" nvidia-smi                        # GB10 が出ること
```

`systemctl restart docker` は rootful 側のコンテナを止めるので、先に
`sudo docker compose --profile head down` などで片付けておくこと。

#### `--device` を明示している理由

`/etc/nvidia-container-runtime/config.toml` に **`no-cgroups = true`** が入っていると、
`--gpus all` だけでは GPU が使えない。これは rootless docker で GPU を使うための
必須設定だが、rootful では nvidia-container-cli が **device cgroup の許可リストを
更新しなくなる**ため、コンテナ内にデバイスノードは現れるのにアクセスが弾かれ、
`Failed to initialize NVML` → `Failed to infer device type` で落ちる。

`no-cgroups = false` にすると今度は rootless 側の GPU が壊れるので、設定は触らず
**デバイスノードを明示的に渡して cgroup を通す**。compose の `devices:` に入れてある。

それでもダメな場合は `docker-compose.yml` の `x-vllm-service` に
`privileged: true` を足す (device cgroup ごとバイパスされる)。上流の
tonyd2wild のレシピも `--privileged` を使っている。

---

## セットアップ

### 0. repo と .env を 2 台に配る

```bash
cp .env.example .env
$EDITOR .env      # HEAD_ROCE_IP / WORKER_ROCE_IP / MODEL_PATH を自分の環境に合わせる

# worker へ同期 (2 台とも .env の中身は同じでよい。役割は --profile で切り替える)
rsync -av --exclude .git --exclude models --exclude vllm-cache ./ "$WORKER:~/repos/llm-container/"
```

`HEAD_ROCE_IP` / `WORKER_ROCE_IP` / `IB_HCA_NAME` / `NCCL_IB_GID_INDEX` は実機で確認する:

```bash
ip -br addr show enp1s0f0np0     # QSFP 側の IPv4
show_gids                        # その IPv4 が載っている行の DEV 名 と INDEX (RoCE v2 の方)
```

NVIDIA の connect-two-sparks は link-local (169.254.x.x) を振るので、
**再起動で IP が変わることがある**。`scripts/preflight.sh` が検出する。

### 1. 重みを取得 (2 台とも / 各 167GB, 48 shard)

```bash
./scripts/fetch-model.sh                                       # head
ssh "$WORKER" 'cd ~/repos/llm-container && ./scripts/fetch-model.sh'
```

保存先は `.env` の `MODEL_PATH`。`hf` が無ければ `pip install -U 'huggingface_hub[cli,hf_transfer]'`。

### 2. イメージを取得 (2 台とも / 約 14GB) — **rootful**

digest 固定。vLLM 0.25.0 / linux-arm64 / sm_121 ネイティブビルド。

```bash
IMAGE=$(sed -n 's/^VLLM_IMAGE=//p' .env)
sudo docker pull "$IMAGE"
ssh -t "$WORKER" "cd ~/repos/llm-container && sudo docker pull \$(sed -n 's/^VLLM_IMAGE=//p' .env)"
```

> **必ず起動前に pull しておくこと。** 片方が pull 中に rendezvous が始まるとハンドシェイクごと固まる。

### 3. メモリを空ける (2 台とも)

重みが 1 台あたり約 78GiB + KV 20GiB。121GB の unified memory に対してギリギリなので、
**起動前に他のワークロードを止める**。GB10 の UVM は一度確保されると完全には返らないので、
迷ったら再起動が一番確実。

```bash
docker ps                                    # rootless 側で動いているものを確認
docker stats --no-stream                     # どれがメモリを食っているか
docker stop <他のコンテナ>
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
sudo sysctl -w vm.swappiness=10
```

`MemAvailable` が 105GB 以上あれば OK。

### 4. MTU を 9000 に上げる (任意だが推奨 / 一度やれば済む)

QSFP の Ethernet MTU が既定の 1500 だと、**RoCE の path MTU が 1024 に落ちる**。
9000 に上げると HCA の上限である 4096 まで上がり、NCCL の 1 転送あたりの
パケット数が 1/4 になる。上流も MTU を 9000 にして運用している。

```bash
ip -d link show enp1s0f0np0 | grep -o 'maxmtu [0-9]*'   # 9978 くらいあるはず
ibv_devinfo -d rocep1s0f0 | grep -E 'active_mtu|max_mtu'
#   max_mtu: 4096 / active_mtu: 1024  <- この active_mtu を 4096 にしたい
```

> **vLLM を止めてからやること。** NCCL は初期化時に path MTU を読むので、
> 動作中に変えると中途半端な状態になる。
> また **SSH は `$WORKER_MGMT` 経由で**。QSFP 側から入っていると接続が切れる。

**(a) まず一時変更で試す** (再起動で元に戻る)。片側だけ 9000 の間は大きいパケットが
落ちるので、間を空けずに両方やる。

```bash
sudo ip link set dev enp1s0f0np0 mtu 9000
ssh -t "$WORKER_MGMT" 'sudo ip link set dev enp1s0f0np0 mtu 9000'
```

**(b) 効いたか確認**

```bash
ip -br link show enp1s0f0np0
ibv_devinfo -d rocep1s0f0 | grep active_mtu     # 4096 (5) になっていること
ping -M do -s 8972 -c 3 "$WORKER"               # 8972 = 9000 - 28、フラグメント禁止で通ること
```

`active_mtu: 4096 (5)` になって ping が通れば成功。

**(c) 永続化** — QSFP は netplan + NetworkManager が管理していて、
NVIDIA の connect-two-sparks が置いた `/etc/netplan/40-cx7.yaml` が本体。

```bash
sudo netplan get                        # 現状確認
sudo $EDITOR /etc/netplan/40-cx7.yaml
```

`enp1s0f0np0:` のブロックに `mtu: 9000` を 1 行足す。

```yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp1s0f0np0:
      dhcp4: no
      link-local: [ipv4]
      mtu: 9000          # <- これを追加
```

適用は `apply` ではなく **`try`** を使う。設定をミスって疎通が切れても 120 秒で自動的に戻る。

```bash
sudo netplan try        # 問題なければ Enter で確定、放置すれば revert
```

worker 側も同じ編集をして、(b) の確認をもう一度。

**再起動後は必ず確認すること。** netplan の適用が外れると 1500 に戻る。
`scripts/preflight.sh` が MTU を WARN で出すので、そこで気付ける。

### 5. preflight

```bash
./scripts/preflight.sh                                         # head
ssh "$WORKER" 'cd ~/repos/llm-container && ./scripts/preflight.sh'
```

NG が出ている状態で起動しても、5〜10 分待たされてから NCCL エラーか OOM-kill で死ぬだけ。

---

## 起動

**worker を先に、head を後に。** head が rendezvous の master になるので、worker が先に
待ち受けている状態にしてから head を上げる。

```bash
# --- worker ---
ssh -t "$WORKER" 'cd ~/repos/llm-container && sudo docker compose --profile worker up -d'

# --- head ---
sudo docker compose --profile head up -d
sudo docker logs -f dsv4-head
```

初回は Triton / DeepGEMM の JIT が走るので **15〜20 分**かかる。2 回目以降は
`vllm-cache/` が効いて 7〜9 分程度。`Application startup complete` が出れば完了。

### 確認

```bash
curl -s http://127.0.0.1:8910/health
curl -s http://127.0.0.1:8910/v1/models | python3 -m json.tool

curl -s http://127.0.0.1:8910/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash",
       "messages":[{"role":"user","content":"クイックソートを Rust で書いて"}],
       "temperature":1.0,"top_p":0.95,"max_tokens":512}'
```

> **コールドスタート後の最初の 1 発はタイムアウトすることがある。** 新しい prompt shape に
> 対する JIT / MoE エキスパートのウォームアップで、異常ではない。そのまま投げ直せば通る。

### 停止

```bash
sudo docker compose --profile head down
ssh -t "$WORKER" 'cd ~/repos/llm-container && sudo docker compose --profile worker down'
```

コンテナを落としても UVM は完全には解放されない。別の構成に切り替えるときは
**両ノードを再起動してから**始めること。

---

## 使う

`network_mode: host` なので、head に届くアドレスならどれでも叩ける
(通常 LAN / tailscale / QSFP)。認証はしていないが、多くのクライアントが
空でないキーを要求するので適当な文字列を渡す。

### OpenAI 互換 (`/v1/chat/completions`)

```python
from openai import OpenAI
client = OpenAI(base_url="http://<head>:8910/v1", api_key="local")
r = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "..."}],
    temperature=1.0, top_p=0.95,     # agent 用途。それ以外は top_p=1.0
)
```

**thinking は `reasoning` フィールドに入る。** `reasoning_content` ではないので注意
(多くの OpenAI 互換クライアントは `reasoning_content` を見にいくため、thinking が
表示されないことがある)。`--reasoning-parser deepseek_v4` を外すと thinking が
`content` 側に混ざるので、外さないこと。

```python
r.choices[0].message.reasoning    # <- ここ
```

### Anthropic 互換 (`/v1/messages`) — Claude Code から使う

vLLM は Anthropic Messages API も生やすので、**変換プロキシなしで Claude Code を
直結できる**。

```bash
curl -s http://127.0.0.1:8910/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"deepseek-v4-flash","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
```

`~/.claude/settings.json` の `env` を向ける (この repo の
`scripts/local-llm-on.sh` / `local-llm-off.sh` が設定ファイルを差し替える):

```jsonc
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8910",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_DEFAULT_OPUS_MODEL":   "deepseek-v4-flash",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-flash",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL":  "deepseek-v4-flash"
  }
}
```

`SERVED_MODEL_NAME` を変えたら 3 つのモデル名も揃えること。

### reasoning effort

`low` / `high` / `max` の 3 段階。**未指定だと thinking は一切出ない**(即答モード)。

```python
client.chat.completions.create(..., reasoning_effort="high")
```

**thinking がトークン予算の 7〜9 割を食う**ので、`max_tokens` は成果物の 4〜5 倍を
見積もること。16K では Rust の実装が途中で切れた。また「数値検算を付けて」のように
手計算を要求すると thinking が発散して、答えに到達していたのに本文 0 文字で
24,576 トークンを使い切った例がある。検算はコードを書かせて実行する形にする。

thinking の言語は入力言語に追随しない (日本語で聞いても中国語や英語で思考する)。
生成コードのコメントも中国語になることがあるので、必要ならプロンプトで指定する。

### ストリーミング時の注意

reasoning から content へ切り替わるデルタは **両方のフィールドを同時に持つ**。
片方だけ拾うともう片方が落ちる。

```python
rc, cc = delta.get("reasoning"), delta.get("content")
if rc: reasoning.append(rc)
if cc: content.append(cc)      # elif にすると本文の先頭が消える
```

### 状態を見る

```bash
curl -s http://127.0.0.1:8910/metrics | grep -E '^vllm:(kv_cache_usage_perc|num_requests)'
sudo docker logs -f dsv4-head
```

---

## コンテキスト長とメモリ

`MAX_MODEL_LEN` がコンテキスト長。既定は **1M** (モデルの上限)。
短くしたい場合は [`presets/`](presets/) のオーバーレイを重ねる:

```bash
sudo docker compose --env-file .env --env-file presets/256k.env --profile head up -d
```

**worker 側も同じ組み合わせで起動すること。**

| プリセット | コンテキスト | KV | max_concurrency | 特性 |
|---|---|---|---|---|
| `128k.env` | 131,072 | 18GiB | 2.19 | prefill の余白が最大 |
| `256k.env` | 262,144 | 18GiB | 2.06 | **常用向け**。250K 入力でも残 6.7GiB |
| (既定 / `1m.env`) | 1,048,576 | 20GiB | 1.69 | 900K 入力で残 1.9GiB |

### 実測

| 入力 | TTFT | prefill | decode |
|---:|---:|---:|---:|
| 15,958 | 9.7s | 1,646 t/s | 62.1 t/s |
| 131,008 | 72.7s | 1,802 t/s | 60.4 t/s |
| 249,952 | 149.6s | 1,670 t/s | 49.3 t/s |
| 499,994 | 372.0s | 1,344 t/s | 65.5 t/s |
| 900,014 | 874.1s | 1,030 t/s | 55.5 t/s |

**decode は 37.7〜65.5 t/s (18 計測の平均 51.4)** で、入力・出力の長さにほぼ依存
しない。6 分半の連続生成でも劣化なし。一方 **prefill は 900K で半減する**ので、
体感を決めるのは TTFT (128K で約 1 分、500K で約 6 分、900K で約 15 分)。

### 天井は KV ではなく prefill の活性化メモリ

`MemAvailable` の最小値は 250K 入力で 6.7GiB、900K 入力で **1.9GiB**。
KV は固定サイズなので増えないが、prefill 中の活性化メモリは入力長に比例する。
OOM-kill (exit 137) されたら `MAX_NUM_BATCHED_TOKENS` を下げるか `256k.env` に落とす。

### KV サイズは計算で予測できない

`kv_cache_size_tokens` は `MAX_MODEL_LEN` にも KV バイト数にも比例しない
(圧縮の効き方が非線形)。**起動して `/metrics` を読むこと。**
判断基準は `max_concurrency >= 1.0`。

```bash
curl -s http://127.0.0.1:8910/metrics | grep 'cache_config_info{' | tr ',' '\n' \
  | grep -E 'kv_cache_max_concurrency|kv_cache_size_tokens'
```

根拠と外した予測の記録は [`presets/README.md`](presets/README.md) に。

### その他のチューニング

| 変数 | 既定 | メモ |
|---|---|---|
| `HOST_PORT` | 8910 | API のポート |
| `MAX_NUM_SEQS` | 1 | 上げると同時実行できるが、KV を分け合うので実効コンテキストが減る |
| `GPU_MEMORY_UTILIZATION` | 0.87 | unified memory なので上げすぎるとホストごと OOM |
| `num_speculative_tokens` | 7 | DSpark の draft 長。効いていない感じなら 5 も試す価値あり |
| `NCCL_DEBUG` | WARN | ハンドシェイクを追うときは `INFO` |

`VLLM_EXTRA_ARGS` / `VLLM_PARSER_ARGS` は entrypoint が空白で分割するので、
**JSON の中に空白を入れないこと**。

---

## ハマりどころ

| 症状 | 原因と対処 |
|---|---|
| `RuntimeError: Failed to infer device type` / `Can't initialize NVML` / `No CUDA runtime is found` | ① rootful daemon に nvidia ランタイムが未登録 → `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`。② それでもダメなら `no-cgroups=true` で device cgroup に弾かれている → compose の `devices:` で `/dev/nvidia*` を渡す (対応済み)、最終手段は `privileged: true` (→ 「rootful daemon に nvidia ランタイムを登録する」) |
| NCCL が `unhandled system error` / `ibv_reg_mr` で `Cannot allocate memory` | RDMA がコンテナに通っていない。`--device /dev/infiniband`・`memlock` 無制限・`network_mode: host`・`ipc: host` は compose に入っているので、まず **rootful で起動しているか**を疑う |
| QP ハンドシェイクが `local GID ::` で死ぬ / rendezvous で固まる | Spark は RoCE ポートを 2 本見せるがケーブルは 1 本。刺さっていない側の GID index 3 は空。`show_gids` で IPv4 が QSFP 側のアドレスになっている行の DEV と INDEX を `.env` の `IB_HCA_NAME` / `NCCL_IB_GID_INDEX` に入れる |
| 再起動したら疎通しなくなった | QSFP 側は link-local なので IP が変わることがある。`ip -br addr show enp1s0f0np0` を見て `.env` を更新 (preflight が検出する) |
| worker が exit 137 | UMA の OOM-kill。他のワークロードを止めて再起動してからやり直す。`MAX_MODEL_LEN` と KV を下げるのも手 |
| 速度が 5〜10 tok/s しか出ない | ネイティブ FP8 DeepGEMM 経路に乗っていない。起動ログに `scale_fmt=ue8m0` / DeepGEMM 有効の行が出ているか確認。`VLLM_USE_DEEP_GEMM_E8M0=1` は必須 |
| 出力が文字化け・意味不明 | イメージやビルドを変えた後にコンパイルキャッシュが残っている。`sudo rm -rf vllm-cache/{vllm,triton,torchinductor}/*` して再起動 |
| prefill が遅い | 2 台の driver / kernel / firmware バージョンが揃っているか。上流はここを揃えるだけで prefill +140% と報告している。あと MTU が 1500 のままなら 9000 に上げる (→ セットアップ 4.) |
| `ping -M do -s 8972` が通らない | 片側の MTU が 1500 のまま。両ノードとも 9000 になっているか確認 (→ セットアップ 4.) |
| `unknown reasoning parser` 等で起動しない | `.env` の `VLLM_PARSER_ARGS` を空にして素の起動をまず通す |
| block size 関連のエラーで起動しない | `VLLM_EXTRA_ARGS` から `--block-size 256` を外す |

ログ:

```bash
sudo docker logs -f dsv4-head
ssh -t "$WORKER" 'sudo docker logs -f dsv4-worker'
```

---

## 出典

この構成は以下の実機レポートを組み合わせたもの。

- [bjk110/spark_vllm_docker](https://github.com/bjk110/spark_vllm_docker) — 使っているイメージと、
  native DSpark k=7 / 固定 FP8 KV という検証済みの組み合わせの出所
- [tonyd2wild/deepseek-v4-flash-dgx-spark](https://github.com/tonyd2wild/deepseek-v4-flash-dgx-spark) —
  dual Spark の RDMA / NCCL GID 周りと `mp` バックエンドでの 2 ノード TP
- [DevelopersIO: DGX Spark 2 台で DeepSeek V4 Flash-DSpark を動かしてみた](https://dev.classmethod.jp/en/articles/dgx-spark-2node-deepseek-v4-flash-dspark/) —
  QSFP ではなく Wi-Fi/Ethernet 側を掴んでしまう罠と、実測スループット
- [vLLM Recipes: DeepSeek-V4-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) — 公式推奨フラグ
- [al-engr.com: DS4 dual Spark deploy](https://al-engr.com/ds4-dual-spark-deploy.html) — 128K での KV プール実測と失敗事例
- [howtospark.com: DeepSeek V4 Flash DSpark dual Spark 1M](https://howtospark.com/recipes/deepseek-v4-flash-dspark-dual-spark-1m) — 1M コンテキストの構成
- [Flowtivity: 1M context on two DGX Sparks](https://flowtivity.ai/blog/deepseek-v4-flash-1m-context-dual-dgx-spark/) — 1M での実測値
