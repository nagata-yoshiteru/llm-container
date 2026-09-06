# DeepSeek-V4-Flash-Vision-Exp on 2x DGX Spark

[deepseek-ai/DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
(304B MoE / FP8+FP4 / vision tower BF16 / 168GB) を DGX Spark 2 台に **TP=2** で
分割して、OpenAI 互換 API を `:8910` に生やす構成。画像入力に対応する。

- 分散バックエンドは **Ray ではなく `mp`** (torch.distributed SPMD)。head/worker が
  それぞれ `vllm serve` を `--nnodes/--node-rank/--master-addr` 付きで起動する。
- ノード間の NCCL は 200GbE QSFP 直結リンクの **RoCEv2 (RDMA)**。
- **native DSpark speculative decoding (k=3)** を使う。Vision-Exp が同梱している
  draft モジュールで、蒸留なしで速度を出している本体。

## 0731 から移ってきた人へ

DeepSeek-V4-Flash-0731 の構成は `DGX-Spark-2/deepseek-ai/DeepSeek-V4-Flash-0731`
ブランチに残してある。ネットワーク・RDMA・compose まわりは全部同じで、変わったのは
次の 5 点だけ。

| | 0731 | Vision-Exp |
|---|---|---|
| イメージ | `ghcr.io/bjk110/vllm-spark` (vLLM 0.25 / SM121 ネイティブ) | `vllm/vllm-openai:deepseekv4-flash-vision-arm64-cu130` (vLLM 0.29 系 pre-release) |
| 重み | `models/DeepSeek-V4-Flash-0731` (167GB) | `models/DeepSeek-V4-Flash-Vision-Exp` (168GB / 別途 DL) |
| DSpark | k=7 greedy | **k=3** probabilistic + adaptive verification |
| cudagraph | `[8]` (=1+7) | `[4]` (=1+3) |
| mm | `--skip-mm-profiling` | 外した (vision encoder の活性化を見積もらせる) |

**イメージを変えざるを得ないのがポイント。** Vision-Exp の視覚モジュールはまだ
安定版 wheel に入っておらず (上流 PR #54566)、通常の release で起動すると text-only
のモデルクラスに解決されて vision のテンソルで落ちる。0731 で使っていた bjk110 の
SM121 ネイティブビルドにも vision 実装は無い (2026-09-02 時点で vision 対応タグ無し)。

上流イメージでも SM121 (GB10) のカーネルは入っている
(`TORCH_CUDA_ARCH_LIST=8.7 8.9 9.0 10.0+PTX 12.0 12.1`) ので動くはずだが、
**bjk110 が Spark 向けに詰めていたチューニングは載っていない**。0731 で出ていた
decode 37.7〜65.5 t/s がそのまま出る保証はないので、計測し直すこと。

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

### 1. 重みを取得 (2 台とも / 各 168GB, 48 shard)

```bash
./scripts/fetch-model.sh                                       # head
ssh "$WORKER" 'cd ~/repos/llm-container && ./scripts/fetch-model.sh'
```

保存先は `.env` の `MODEL_PATH`。`hf` が無ければ `pip install -U 'huggingface_hub[cli,hf_transfer]'`。

### 2. イメージを取得 (2 台とも / 約 14GB) — **rootful**

digest 固定。上流 `vllm/vllm-openai:deepseekv4-flash-vision-arm64-cu130` /
linux-arm64 / CUDA 13.0.1 / vLLM 0.29 系 pre-release。arch list に 12.1 (GB10) を含む。

CUDA 13 で問題が出たら `.env` の `VLLM_IMAGE` を cu129 版の digest に差し替える
(`.env` のコメントに書いてある)。

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

モデル名は `deepseek-v4-flash-vision` と `deepseek-v4-flash` の**どちらでも通る**
(`.env` の `SERVED_MODEL_NAME` に両方書いてある)。0731 のときのクライアント設定を
そのまま使えるようにしてあるだけで、中身は Vision-Exp。

### 画像を投げる

Vision-Exp の本題。OpenAI 互換の `image_url` コンテンツブロックで渡す。

```python
import base64, pathlib

b64 = base64.b64encode(pathlib.Path("shot.png").read_bytes()).decode()
r = client.chat.completions.create(
    model="deepseek-v4-flash-vision",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "このスクリーンショットで何が起きてる？"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}],
    temperature=1.0, top_p=0.95,
)
```

- **前処理はチェックポイント側で固定**されていて、`--mm-processor-kwargs` は受け付けない
  (`vision_max_n_token=384` / `vision_min_pixels=147456` / `vision_max_wh_ratio=8`)。
- 1 リクエストあたりの枚数に上限は設けていない。実質の上限は `MAX_MODEL_LEN`。
  起動時の mm profiling でメモリが足りなくなったら、`VLLM_EXTRA_ARGS` に
  `--limit-mm-per-prompt {"image":1}` を足す。
- `file://` の URL を使いたいときだけ `--allowed-local-media-path <dir>` が要る。
  http(s) と base64 data URL は追加設定なしで通る。
- 画像込みでも `MAX_NUM_SEQS=1` は変えていない。UMA の余裕がないので、
  vision encoder の分は prefill の活性化メモリから持っていかれる。

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

`low` / `high` / `max` の 3 段階。既定構成では**未指定だと thinking は一切出ない**
(即答モード)。

**`anemll-1m.env` では既定が変わる。** `--default-chat-template-kwargs
{"thinking":true,"reasoning_effort":"low"}` を渡しているので、
**クライアントが何も指定しなければ thinking on / effort low** で走る。
切るならリクエスト側で `chat_template_kwargs: {"thinking": false}` (リクエスト優先)。
サーバ既定を変えるならこの引数を編集する。

**on/off ははっきり効くが、low/high/max は思考量が単調には増えない。**
effort を上げれば深く考える、という前提でトークン予算を組まないこと。

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

> ⚠ **以下の数値はすべて 0731 + bjk110 イメージ + DSpark k=7 での実測。**
> Vision-Exp はモデルもイメージも DSpark の深さも違うので、そのままは当てはまらない。
> 上の `max_concurrency` の表も含め、起動後に `/metrics` で取り直すこと。
> 特に `num_nextn_predict_layers` が 1 -> 3 に増えているので、KV の 1 token あたりの
> バイト数は変わっている可能性が高い。

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

### QSFP を 2 枚使う (dual-HCA) — 帯域がほぼ倍

**GB10 の QSFP ケージは PCIe x4 が 2 本で、2 枚の独立した NIC に見える。**
片方しか設定していないと帯域を半分捨てている。上流報告では nccl-tests の busbw が
**98 → 161 Gb/s (+64%)**、単発 decode で **+8% (135 tok/s)**。

vLLM 公式 recipe の GB10 プロファイルも、まさにこの構成を指定している
(`IB_IF="rocep1s0f0,roceP2p1s0f0"` — この機体と同じデバイス名)。

```
rocep1s0f0   / enp1s0f0np0    ACTIVE 200Gb/s  192.168.0.2/24  MTU 9000
roceP2p1s0f0 / enP2p1s0f0np0  ACTIVE 200Gb/s  192.168.1.2/24  MTU 9000
```

別サブネットに分離済み、両リンクとも worker へ jumbo frame (8972B / DF) が通ることを確認済み。
netplan は `/etc/netplan/41-cx7-second.yaml` で永続化。

PCIe はどちらも Gen5 x4 (`32.0 GT/s` / `width=4`) なので、第 2 コントローラを
Gen5 x2 で配線する古い BIOS の既知問題には該当しない。確認コマンド:

```bash
ibdev2netdev                     # ACTIVE なのに IP が無い組があるか
for n in enp1s0f0np0 enP2p1s0f0np0; do
  p=$(readlink -f /sys/class/net/$n/device)
  echo "$n $(cat $p/current_link_speed) width=$(cat $p/current_link_width)"
done
```

**手順 (両ノードとも / vLLM を止めてから / SSH は `$WORKER_MGMT` 経由で)**

1. 2 本目に **1 本目とは別サブネット**の IP と MTU 9000 を振る。netplan の
   drop-in で永続化すること。**`nmcli` のプロファイルは `/run` に置かれて再起動で
   消える**ので使わない。

   ```yaml
   # /etc/netplan/41-cx7-second.yaml
   network:
     version: 2
     renderer: NetworkManager
     ethernets:
       enP2p1s0f0np0:
         dhcp4: no
         addresses: [192.168.1.2/24]    # worker は 192.168.1.1/24
         mtu: 9000
   ```

   ```bash
   sudo netplan try        # apply ではなく try
   ```

2. 両側 MTU 9000 を確認。片側だけ 9000 だと大きい転送で
   `IBV_WC_RETRY_EXC_ERR(12)` になる。**ファームウェア更新の再起動で片側が 1500 に
   戻る**事例があるので、再起動後は必ず見ること。

3. `.env` の dual-HCA ブロックを有効化する (コメントアウトで用意してある)。
   `IB_HCA_NAME` / `ROCE_IF_NAME` をカンマ区切りにして `NCCL_IB_MERGE_NICS=1`。
   **`NCCL_IB_GID_INDEX` は必ず空にすること** (下記)。

4. 初回起動は **NCCL トポロジが変わったので JIT が再チューニングされる。約 36 分**
   かかった報告がある。20 分で切るウォッチドッグを持っていると「ハングした」と
   誤判定するので注意。

**⚠ dual-HCA では `NCCL_IB_GID_INDEX` を固定してはいけない**

GID テーブルの index はリンクイベント (リンク上下 / 2 枚目の起動 / 再起動) で
**動く**。上流では両 HCA を上げた状態で **30 分のうちに 3 → 4 にずれた**実例がある。
厄介なのは、書いた時点では正しく検証できてしまうこと。ずれると片方の HCA だけ
`ibv_modify_qp failed with 61` になり、**モデルロード後に TP ハンドシェイクが無言で
ハングする**。最近の NCCL は RoCEv2/IPv4 の GID をデバイスごとに自力で選ぶので、
**未設定が正解**。

なお「空文字は 0 と解釈されて `fe80` link-local GID を掴むのでピン留めより悪い」と
上流が警告しているが、**この repo の entrypoint は空の変数を unset する**ので
その罠は踏まない (`.env` で行を空にすれば OK)。

### イメージ選択の現状 (2026-09-06 調査)

**Vision-Exp が動く「Spark 最適化イメージ」は、まだ存在しない。**
速いイメージと vision が入っているイメージが、いまのところ別物になっている。

| イメージ | GB10 最適化 | vision | 判定 |
|---|---|---|---|
| `vllm/vllm-openai:deepseekv4-flash-vision-arm64-cu130` (**採用中**) | △ 汎用 (`12.1`) | ✅ 参照実装 | これしかない |
| `eugr/spark-vllm-b12x:latest` | ✅ **`12.1a` 専用ビルド + B12X** | ❌ 記載なし | vision 待ち |
| `ghcr.io/bjk110/vllm-spark` | ✅ SM121 ネイティブ | ❌ | 0731 用 |
| `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | ✅ | △ 要 hotfix 注入 | 移植版経由なら |

**`eugr/spark-vllm-b12x` が本命候補だが、まだ使えない。** vLLM 公式 recipe の GB10
プロファイルが指定しているイメージで、`TORCH_CUDA_ARCH_LIST=12.1a` と GB10 専用に
ビルドされている (採用中の上流イメージは `12.1` で `a` 無しの汎用)。しかし
**vLLM main ではなくフォークの dev ブランチ (`local-inference-lab/vllm`) から
ビルドされていて、リポジトリにも NVIDIA フォーラムにも Vision-Exp の動作報告が無い**。
フォーラムでも「b12x で vision は動くのか」という質問に誰も答えていない。

**ただし見通しは良い。** 上流 PR #54566 (`[New model][Multimodal] Add
DeepSeek-V4-Flash-Vision-Exp support`) は **2026-09-02 に main へマージ済み**。
最新リリースは v0.28.0 (08-26) なのでまだ stable には入っていないが、次の
リリースが出れば b12x 側もリベースで拾える可能性が高い。

定期的に見るなら:

```bash
# b12x に vision が入ったか (Vision-Exp / DeepseekV4V の記載を探す)
curl -s https://api.github.com/repos/eugr/spark-vllm-docker/commits?per_page=20 \
  | grep -io 'vision[^"]*' | head
# 上流の vision イメージが更新されたか
curl -s 'https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/?name=deepseekv4-flash-vision' \
  | python3 -c 'import sys,json;[print(r["name"],r["last_updated"][:10]) for r in json.load(sys.stdin)["results"]]'
```

### Vision-Exp をもっと速くしたい (コミュニティ移植版という選択肢)

**2 台 DGX Spark 向けに Vision-Exp を詰めた移植版が既に 2 つある。** どちらも
既定の上流イメージより速いが、**vision の品質バグが報告されている**ので、
速度と正しさのトレードオフになる。

| | 既定 (このリポジトリ) | コミュニティ移植版 |
|---|---|---|
| イメージ | 上流 `deepseekv4-flash-vision` (PR #54566 の参照実装) | anemll 0.1.1 / tonyd2wild 系に vision ファイルを注入 |
| decode | 未計測 | 62〜83 tok/s (MiaAI-Lab 報告) |
| KV @1M | 未計測 | 2.33M tokens (nvfp4_ds_mla) |
| vision の正しさ | 参照実装のまま | ⚠ 既知の欠落あり (下記) |
| 手間 | `.env` だけ | ファイル一式の bind mount + hotfix |

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  — anemll 0.1.1 に起動時 hotfix で ViT + Aligner を注入。`MAX_NUM_SEQS=6` /
  `nvfp4_ds_mla` / `flashinfer_b12x` / `LIMIT_MM_PER_PROMPT={"image":8}`。
  画像は **`user` ロールにしか置けない** (他ロールだと 400 が返り、その履歴が
  残る限り以後も失敗し続ける)。
- [tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
  — vLLM 0.21 系に vision ファイル 4 つ + パッチ 2 つを bind mount。
  KV 2.79M tokens @ gmu 0.85 を報告。

**⚠ 移植版の vision 品質バグ (2026-09-02 時点で未修正)**

tonyd2wild の README が自ら明記している欠落が 2 つ:
① 画像スパン内の双方向 attention が未実装、② 画像用の MoE routing bias
(`bias_vl`) が欠落。どちらも**テキストには影響せず画像だけ劣化する**。
[HF の discussion](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/discussions/10)
でも「画像トークンの expert 選択が 43 層すべてで平均 6 個中 5 個外れている」という
報告が出ている (未決着)。**簡単な画像テストはバグがあっても通る**ので、
スモークテストが通ったことは正しさの証明にならない。

上流イメージを使っている限りこの問題は踏まない。**画像の精度が要るなら既定のまま、
速度が要るなら移植版**、という切り分けになる。

### DSpark の k は「イメージごとに」制約が違う

k の正解が資料によって 3 / 5 / 6 とバラバラだが、**ランタイムごとに drafter の
`n_predict` が違う**ためで、矛盾ではない。

| ランタイム | 制約 | 妥当な k |
|---|---|---|
| 上流 `deepseekv4-flash-vision` (**このリポジトリ**) | モデルカード / vLLM recipe が指定 | **3** |
| tonyd2wild 系 (vLLM 0.21) | `n_predict=5` で割り切れること。k=6 は boot 拒否 | 5 |
| anemll 0.25.2 系 | `n_predict` が 1 に解決。k=7 も通る。未指定だと k=1 になる | 5〜7 |

tonyd2wild の issue #48 は「k=3 推奨」を**撤回**している。ただしその A/B は
DSpark shared-expert ローダのパッチ (Patch 4) を当てずに測っていて、パッチ無しでは
acceptance が半減する、という文脈。**上流イメージにその問題があるかは未確認**なので、
ここでは公式の k=3 のままにしてある。

**そして撤回の根拠は「数え上げプロンプト」だった。** HF discussion #11 に
2 台 Spark での workload 別実測が出ていて、k=5 の優位はほぼ合成ベンチ限定:

| workload | 採択率 @k=5 | tok/step | 採択率 @k=3 | tok/step |
|---|---|---|---|---|
| count-to-300 | 0.974 | 5.88 | 0.997 | 4.00 |
| code | 0.464 | 3.31 | 0.642 | 2.93 |
| prose | 0.180 | 1.90 | 0.286 | 1.86 |

steps/s は k=5 で 14.6〜15.6、k=3 で 16.0〜17.3。掛け合わせると **code はほぼ互角、
prose は k=3 の方がわずかに速い**。k=5 が大きく勝つのは数え上げだけ。
**実用ワークロードでは k=3 で損をしていない**ということなので、この構成はこのままでよい。

なお同スレッドでは、Vision-Exp の採択率は 0731 より構造的に低く
(code / prose で **tok/s にして 12〜17% 減**) 、これはモデル固有だろうと結論している。
**0731 と同じ速度は出ない前提**で見積もること。

`--max-cudagraph-capture-size` は共通で `max_num_seqs × (k+1)`。
このリポジトリは `1 × (3+1) = 4`。

### 計測するときの注意 (これを知らないと 4 倍間違える)

**`stream: false` で測ること。** 投機デコードは 1 ステップにつき SSE チャンクを
1 つしか出さないので、ストリーミングの delta を数えると **tok/s ではなく steps/s** を
測ってしまう。同一リクエストで 14.7 と 60.1 という差が報告されている。

**コールドスタートのペナルティは約 30%** で、アイドル後にも再発する。
短いウォームアップでは足りず、500〜700 トークン級の生成が要る。

**decode が遅いときはまず acceptance を見る。** `tok/s = steps/s × 1 step あたりの
採択トークン数`なので、acceptance が半分なら速度も半分。出力品質は完璧なままなので
モデルが遅いように見える。prose での acceptance が 25% 前後なのは
この vision 版の素の特性だと報告されている。

### もっと速くしたい (anemll ランタイムへの差し替え) — ⚠ 0731 専用

> **この節は DeepSeek-V4-Flash-0731 の話。** `presets/anemll-1m.env` 単体では
> Vision-Exp は動かない (anemll イメージに視覚モジュールが無い)。Vision-Exp で
> 速度を追う場合は上の「コミュニティ移植版」を参照。

既定構成の decode は **37.7〜65.5 t/s (平均 51.4)**。一方、同じ 2 台構成で
**71〜76 t/s** を出している報告があり、その差はチューニングではなく
**ランタイムイメージの違い**。`presets/anemll-1m.env` がその構成。

| | 既定 (`.env`) | `anemll-1m.env` |
|---|---|---|
| イメージ | `ghcr.io/bjk110/vllm-spark` (vLLM 0.25.0) | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (0.25.2) |
| KV | `fp8` 固定 20GiB | `nvfp4_ds_mla` (1 token あたりが半分以下) |
| MoE | `marlin` | `flashinfer_b12x` |
| DSpark | k=7 greedy | **k=5** probabilistic |
| cudagraph | 固定 `[8]` | `--max-cudagraph-capture-size 36` |
| scheduler | prefix caching 無効 / seqs=1 | prefix caching + chunked prefill + async / seqs=6 |

**0731 では k=7 は正しくない。** drafter は 1 パスあたり 5 トークン
(`dspark_block_size=5`) しか出さない。上流レシピでは k>5 は boot 拒否か
生成時 crash になる。既定イメージは preview チェックポイント向けに k=7 で
検証されたもので、0731 でそのまま動いてはいるが期待どおり効いている保証はない。

```bash
# 2 台ともイメージを pull してから
sudo docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1

# worker -> head の順に
ssh -t "$WORKER" 'cd ~/repos/llm-container && sudo docker compose \
  --env-file .env --env-file presets/anemll-1m.env --profile worker up -d'
sudo docker compose --env-file .env --env-file presets/anemll-1m.env --profile head up -d
```

既定構成は `.env` のまま残っているので、`--env-file` を外せば戻る。
イメージを跨いだら `vllm-cache/{vllm,triton,torchinductor}` は消すこと。

#### 速度が出ていないときに最初に見るもの

decode が遅い原因は「step が遅い」ではなく **draft acceptance が低い**ことが多い。
`tok/s = steps/s × 1 step あたりの採択トークン数` なので、acceptance が半分なら
速度も半分になる。**出力品質は完璧なまま**なので、モデルが遅いように見える。

```bash
sudo docker logs dsv4-head 2>&1 | grep -iE 'acceptance|accepted throughput|drafted'
```

`Avg Draft acceptance rate` が 60% 前後なら健全、**25% 前後なら drafter が壊れている**。
0731 + DSpark には「draft 側の shared expert (12 テンソル) を weight loader が
黙って捨てる」既知のバグがあり、これを踏むと mean decode が 32.7 -> 55.4 t/s、
acceptance が 25.7% -> 60.2% 変わる (上流実測)。ログは `logger.debug` なので
既定の INFO では何も出ず、ロード成功として扱われる。

### その他のチューニング

| 変数 | 既定 | メモ |
|---|---|---|
| `HOST_PORT` | 8910 | API のポート |
| `MAX_NUM_SEQS` | 1 | 上げると同時実行できるが、KV を分け合うので実効コンテキストが減る |
| `GPU_MEMORY_UTILIZATION` | 0.87 | unified memory なので上げすぎるとホストごと OOM |
| `num_speculative_tokens` | 3 | DSpark の draft 長。モデルカードと vLLM recipe がどちらも 3 を指定していて、実測値 (受理率 66.3% / 平均 2.99 tok/forward) が出ているのも深さ 3 だけ。drafter の 1 パスは 5 トークン (`dspark_block_size=5`) なので 5 までは上げられる余地があるが未検証。変えるなら `cudagraph_capture_sizes` も `[1+k]` に合わせること |
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
| 画像を渡すと落ちる / vision のテンソルで `KeyError` 等 | text-only のモデルクラスに解決されている。① `vllm-cache/vllm/modelinfos` を**両ノードとも**消す (アーキ判定がキャッシュされる)。② それでもダメなら `VLLM_EXTRA_ARGS` に `--hf-overrides {"architectures":["DeepseekV4VForConditionalGeneration"]}` を足す (コミュニティ移植版で使われている手) |
| CUDAGraph のところで両ノードとも固まる | **NVIDIA ドライバ 590.x は GB10 で CUDAGraph デッドロックを起こす**という報告がある。580.x を使うこと (このマシンは 580.173.02 で該当しない) |
| decode が想定の 1/4 くらいに見える | 投機デコードをストリーミングで測っている。`stream: false` で測り直す (→ 「計測するときの注意」) |
| 起動して安定していたのに、途中のリクエストで突然エンジンごと落ちる | ウォームアップが踏まなかった MoE / batch shape に当たって**推論中に JIT が走り**、`execute_model` の既定デッドライン 300 秒を超えて「worker が死んだ」と誤判定されている。`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` を設定済み (compose の既定)。なお本物のハングは GPU 使用率 96% / 消費電力 18W 程度 (collective の spin-wait) で見分けられる — JIT 中は電力がアイドル近くまで落ちる |
| `ValueError: Adaptive verification trims verification requests on device, which the DeepseekV4IndexerBackend attention backend does not support` | `--speculative-config` の `enable_adaptive_verification` を **false** にする (対応済み)。モデルカードは `true` を指定しているが、それは 4xGB300 向けのコマンド。GB10 では DeepSeek-V4 の sparse indexer を使う attention backend に解決され、こちらは device 側で verification request を削る操作に対応していない。KV 初期化の直後に出るので、ここまで来ていれば重みロード・vision 解決・NCCL は通っている |
| モデルロードは通るのに TP ハンドシェイクで無言でハングする | `NCCL_IB_GID_INDEX` のピン留めがずれた可能性。特に dual-HCA では index が動く (→ 「QSFP を 2 枚使う」)。`.env` の該当行を空にして NCCL に選ばせる |
| JIT 由来の `FileExistsError` / `runtime != nullptr` / FlashInfer の ABI 不一致 | JIT キャッシュを 2 ノードで共有すると両 rank が同じディレクトリに書いて壊れる。`vllm-cache/` は**ノードローカル**にすること (この compose は repo 直下なので既にローカル)。一度壊したら消す |
| block size 関連のエラーで起動しない | `VLLM_EXTRA_ARGS` から `--block-size 256` を外す |

ログ:

```bash
sudo docker logs -f dsv4-head
ssh -t "$WORKER" 'sudo docker logs -f dsv4-worker'
```

---

## 出典

この構成は以下の実機レポートを組み合わせたもの。

### Vision-Exp 関連 (2026-09-02 時点)

- [vLLM Recipes: DeepSeek-V4-Flash-Vision-Exp](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) —
  公式推奨フラグ。vision は PR #54566 の pre-release イメージ限定という記述もここ。
  GB200 以外の実測は無く、GB10 / DGX Spark の構成は載っていない
- [tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark) —
  2 台 Spark 向け vision 移植。k=3 撤回の経緯 (issue #48)、cudagraph サイズの公式、
  vision の既知欠落 (双方向 attention / `bias_vl`)、`stream: false` で測れという指摘
- [HF discussions #10: Waiting for 2x NVIDIA DGX Sparks supported](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/discussions/10) —
  2 台 Spark での動作報告。`--hf-overrides` によるアーキ指定と `modelinfos` 消去、
  画像トークンの MoE routing バグ報告 (未決着)
- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) —
  anemll 0.1.1 に hotfix で vision を注入する 2 台構成。decode 62〜83 tok/s
- [hazyumps/deepseek-v4-flash-gb10](https://github.com/hazyumps/deepseek-v4-flash-gb10) —
  0731 のみだが GB10 2 台のチューニングが濃い。ドライバ 590.x の CUDAGraph
  デッドロック、NCCL 2.30.4 の `shm_broadcast` デッドロック回避

### 0731 時代からのもの

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
- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) —
  `presets/anemll-1m.env` の出所。NVFP4 DS-MLA / b12x MoE / k=5 の serve 引数一式
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) — GB10 向け vLLM 0.25.2 ポート
- [tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) —
  k の上限 (`k<=5`)、DSpark shared-expert loader バグ (+69% decode)、ランタイム比較
- [DevelopersIO: DGX Spark 2 台で DeepSeek V4 Flash-0731](https://dev.classmethod.jp/articles/dgx-spark-2node-deepseek-v4-flash-0731/) —
  上記レシピ既定値での実測 (decode 71〜76 t/s / prefill 約 1,900 t/s)
