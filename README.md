# GLM-5.3-Flash-NVFP4 on 2x DGX Spark

[RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
(320B total / 18B active MoE / NVFP4 / 198GB) を DGX Spark 2 台に **TP=2** で
分割して、OpenAI 互換 API を `:8910` に生やす構成。

- 分散バックエンドは **Ray ではなく `mp`** (torch.distributed SPMD)。head/worker が
  それぞれ `vllm serve` を `--nnodes/--node-rank/--master-addr` 付きで起動する。
- ノード間の NCCL は 200GbE QSFP 直結リンクの **RoCEv2 (RDMA)**。
- **DFlash2 speculative decoding** (k=7 / decode 約 46.9 t/s / MTP-4 比 2.15x)。
  MTP-4 に戻すなら `presets/mtp4.env`。
- **KV プールは 8 GiB 固定** (fp8 KV / 262K ctx で約 71 万トークン)。
  旧版の「固定するな」から**方針が逆転している** →
  下記「[KV は 8 GiB に固定する](#kv-は-8-gib-に固定する-2026-09-18-に方針が逆転した)」。
- **prefix cache 修正 (#18)** 込み。公開イメージは未適用なので
  `scripts/build-prefix-fix.sh` で生成が要る。無いとエージェントのセッションが
  毎ターン会話全体を再 prefill する (13K トークン反復の TTFT 21.1s -> 6.3s)。
- **boot hardening** 込み (JIT storm 抑止 / 永続カーネルキャッシュ / cgroup
  メモリ上限 / page-cache flusher)。KV 固定が成立する前提条件。
- vision (画像・動画) 対応。チェックポイント同梱の chat template が
  マルチモーダルなので追加設定なし。

| | head | worker |
|---|---|---|
| 役割 | rank 0 / API `:8910` | rank 1 / `--headless` |
| NIC | QSFP 直結の `enp1s0f0np0` | `enp1s0f1np1` (リング配線のため相手を向くポートが違う) |
| HCA | ケーブルが刺さっている側 (`rocep1s0f0`, GID index 3 = RoCEv2/IPv4) | `rocep1s0f1` |

IP・ホスト名・パスはすべて `.env` に書く。`.env` は git 管理外。
以下の手順では worker を 2 通りの経路で呼ぶ。

```bash
WORKER=<worker の RoCE IP>       # QSFP 直結側。.env の WORKER_ROCE_IP と同じ値
WORKER_MGMT=<worker の管理用 IP> # 普通の Ethernet / tailscale 側
```

**QSFP 側の設定をいじるときは必ず `$WORKER_MGMT` 経由で SSH すること。**
`$WORKER` でログインしたまま MTU やアドレスを変えると自分の足を撃つ。

---

## なぜ素の vLLM では動かないか (イメージ選択の理由)

GLM-5.3 は **NoPE MLA** (`qk_rope_head_dim=0`)。vLLM の素の SM12x sparse-MLA
カーネル (`FLASHINFER_MLA_SPARSE_SM120`) は DeepSeek の `pe_dim=64` を前提に
書かれていて、GB10 では warmup で `pe_dim must be 64 for fp8_ds_mla` の assert
で死ぬ。また `glm5_next` 自体も stock vLLM には登録がない。

だからこの構成は **day-0 公式イメージ + SM121 向けパッチ** が必要:

| イメージ | 内容 |
|---|---|
| `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8` | day-0 公式 (`vllm/vllm-openai:glm53-flash`) + パッチ 8 段: SM90 NoPE-MLA バックエンドを SM121 で有効化 (FA2) / FlashInfer 0.6.18 (0.6.17 は 64〜256 行バッチで NaN) / NCCL 2.30.7 固定 (nightly が 2.29.7 に落とすと fabric で死ぬ) / cutlass-dsl 4.6.2 / PDL を SM12x で無効 / indexer 強化 / fp8 KV。`presets/mtp4.env` で使う |
| `:sm121-v11-dflash2` (**既定**) | v8 + DFlash2 drafter 対応 overlay |

`:sm121-v11-dflash2` は 2026-08-28 ビルドで、**#18 prefix-cache 修正を持っていない**。
`scripts/build-prefix-fix.sh` がイメージからパッチ済みファイルを作り、entrypoint が
起動時に site-packages へ差し込む (下記「[prefix cache](#prefix-cache-18-の修正)」)。

さらに **bind-mount で入れる top-k 修正が必須**
([patches/sparse_attn_indexer_kpool_sm121.py](patches/sparse_attn_indexer_kpool_sm121.py))。
両公開イメージにも残っている decode 時 `persistent_topk` のバグ:
GB10 (48 SM / 99KB smem) では **~24K トークン超の context で decode した瞬間に
engine が死にます** (OOM に見えるがメモリは無関係)。compose が修正版を
`vllm/.../sparse_attn_indexer_kpool.py` として上書きマウント済み。
entrypoint がゲートの存在を確認してから起動する。

出典は [tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark)
(7 つの day-0 バグの root cause と修復 / `docs/DEPLOY-REPORT.md`)。

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
cp .env.example .env    # 既に作ってあるならそのまま (実機トポロジが入っている)
$EDITOR .env            # HEAD_ROCE_IP / WORKER_ROCE_IP / MODEL_PATH を自分の環境に合わせる

# worker へ同期 (2 台とも .env の中身は同じでよい。役割は --profile で切り替える)
rsync -av --exclude .git --exclude models --exclude vllm-cache ./ "$WORKER_MGMT:~/repos/llm-container/"
```

`HEAD_ROCE_IP` / `WORKER_ROCE_IP` / `IB_HCA_NAME` / `NCCL_IB_GID_INDEX` は実機で確認する:

```bash
ip -br addr show enp1s0f0np0     # QSFP 側の IPv4
show_gids                        # その IPv4 が載っている行の DEV 名 と INDEX (RoCE v2 の方)
```

NVIDIA の connect-two-sparks は link-local (169.254.x.x) を振るので、
**再起動で IP が変わることがある**。`scripts/preflight.sh` が検出する。

### 1. 重みを取得 (2 台とも / 各 198GB / 10 shard + MTP head)

```bash
./scripts/fetch-model.sh                                          # head
ssh "$WORKER" 'cd ~/repos/llm-container && ./scripts/fetch-model.sh'
```

保存先は `.env` の `MODEL_PATH`。`hf` が無ければ
`pip install -U 'huggingface_hub[cli,hf_transfer]'`。

`model_mtp.safetensors` (7.6GB) も同じリポジトリに含まれている。これは
`presets/mtp4.env` (MTP-4 構成) の draft head で、既定の DFlash2 では使わない。

**既定構成では DFlash2 drafter が必須**なので続けて取る (2.2GB /
ライセンス **CC-BY-NC-ND-4.0 = 非商用・改変禁止**):

```bash
./scripts/fetch-model.sh draft
ssh "$WORKER" 'cd ~/repos/llm-container && ./scripts/fetch-model.sh draft'
```

### 2. イメージを取得 (2 台とも / 約 10GB) — **rootful**

digest 固定。linux-arm64。

```bash
IMAGE=$(sed -n 's/^VLLM_IMAGE=//p' .env)
sudo docker pull "$IMAGE"
ssh -t "$WORKER" "cd ~/repos/llm-container && sudo docker pull \$(sed -n 's/^VLLM_IMAGE=//p' .env)"
```

> **必ず起動前に pull しておくこと。** 片方が pull 中に rendezvous が始まると
> ハンドシェイクごと固まる。

`presets/mtp4.env` (MTP-4 に戻す構成) を使うなら v8 も両台で pull:

```bash
sudo docker pull ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:d77d375c742fc54f436dec5108b440f58f021bc6600052bf0e8fe5840357e78f
```

### 3. メモリを空ける (2 台とも)

重みが 1 台あたり約 99GB + KV 8GiB + DFlash2 drafter + ランタイム。121GB の
unified memory に対して余裕がほとんどないので、**起動前に他のワークロードを
止める**。GB10 の UVM は一度確保されると完全には返らないので、迷ったら再起動が
一番確実。

(`presets/mtp4.env` の MTP-4 構成では drafter の代わりに MTP head が
約 4GB/rank を食う。DFlash2 の drafter は MLA テンソルに slot-share するので
KV の追加コストはほぼ 0。)

```bash
docker ps                                    # rootless 側で動いているものを確認
docker stats --no-stream                     # どれがメモリを食っているか
docker stop <他のコンテナ>
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

**swap の扱い (重要)。** `vm.swappiness=0` にすること (再起動で戻るので都度設定)。

```bash
sudo sysctl -w vm.swappiness=0
```

- swap を**無効化**すると、MoE marlin の repack 中のメモリスパイクで worker が死ぬ
- 一方 swappiness 既定のままだと、ロード中にカーネルが vLLM を swap して
  **UVM ドライバの livelock** (shard ロードが同じ場所で凍る / 復帰しない) が起きる
- `vm.swappiness=0` = swap はあるが使わせない、が正解

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
ssh -t "$WORKER_MGMT" 'sudo ip link set dev enp1s0f1np1 mtu 9000'
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

`enp1s0f0np0:` (worker は `enp1s0f1np1:`) のブロックに `mtu: 9000` を 1 行足す。

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

適用は `apply` ではなく **`try`** を使う。設定をミスって疎通が切れても 120 秒で
自動的に戻る。

```bash
sudo netplan try        # 問題なければ Enter で確定、放置すれば revert
```

worker 側も同じ編集をして、(b) の確認をもう一度。

**再起動後は必ず確認すること。** netplan の適用が外れると 1500 に戻る。
`scripts/preflight.sh` が MTU を WARN で出すので、そこで気付ける。

### 5. prefix-cache 修正を生成 (2 台とも / 一度やれば済む)

公開イメージ `sm121-v11-dflash2` は #18 の修正を持っていない。パッチは
site-packages に置かれた状態で self-check するのでホストでは当てられず、
使い捨てコンテナの中で当ててパッチ済みファイルを書き出す。

```bash
./scripts/build-prefix-fix.sh
ssh "$WORKER" 'cd ~/repos/llm-container && ./scripts/build-prefix-fix.sh'
```

`patches/kv_cache_coordinator_prefix_fix.py` (git 管理外 / イメージ依存の派生物)
が出来る。entrypoint が `PREFIX_FIX=1` のときに site-packages へコピーする。
**生成しなくても起動はする** (起動ログに OFF の理由が出る) が、prefix cache が
0 hit のままになる。

### 6. GPU がクランプしていないか確認 (ベンチを取るなら必須)

```bash
./scripts/gputest.sh
ssh "$WORKER" 'cd ~/repos/llm-container && ./scripts/gputest.sh'
```

健全なら 65〜82 TFLOPS。**50 未満はクロッククランプ**で、測定値が全部 2.5 倍
遅くなる。詳細は下記「[クロッククランプ](#クロッククランプ-ベンチの前に必ず見る)」。

### 7. preflight

```bash
./scripts/preflight.sh                                         # head
ssh "$WORKER" 'cd ~/repos/llm-container && ./scripts/preflight.sh'
```

NG が出ている状態で起動しても、5〜10 分待たされてから NCCL エラーか OOM-kill
で死ぬだけ。kpool パッチ / ModelOpt ビルドの誤用 / KV 固定と
`MAX_NUM_BATCHED_TOKENS` の危険な組み合わせもここで弾かれる。

---

## 起動

**worker を先に、head を後に。** head が rendezvous の master になるので、worker が
先に待ち受けている状態にしてから head を上げる。

```bash
# --- 毎 boot のメモリ儀式 (2 台とも / 省略しないこと) ---
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
sudo sysctl -w vm.swappiness=0
ssh -t "$WORKER_MGMT" "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' && sudo sysctl -w vm.swappiness=0"

# --- worker ---
ssh -t "$WORKER" 'cd ~/repos/llm-container && sudo docker compose --profile worker up -d'

# --- head ---
sudo docker compose --profile head up -d

# --- page-cache flusher (2 台とも / 起動中ずっと回しておく) ---
./scripts/flusher.sh &
ssh "$WORKER_MGMT" 'cd ~/repos/llm-container && nohup ./scripts/flusher.sh >/dev/null 2>&1 &'

sudo docker logs -f glm53-head
```

初回は CuTeDSL / Triton / FlashInfer の JIT が走るので **15〜20 分**かかる
(fp4 CUTLASS GEMM の variant が 1 個あたり数分)。2 回目以降は `vllm-cache/` が
効いて短くなる。`Application startup complete` が出れば完了。

**flusher は「回しておく」ものではなく、回していないと落ちる。** GB10 の NVRM は
MemFree で割当を判定し page cache を強制回収しないので、回収可能なキャッシュが
数 GB 残っているだけで KV slab が `NV_ERR_NO_MEMORY` で落ちる。上流は閾値式の
フラッシャが起動 25 分後に期限切れし、その 1 分後にノードを失っている。
**無条件に 20 秒ごと**が正解 (`scripts/flusher.sh` がそれをやる)。

**イメージを跨いだら JIT キャッシュを消すこと** (DeepSeek 構成や
`presets/mtp4.env` から切り替える場合を含む)。混ざると起動不能になる。

```bash
sudo rm -rf vllm-cache/{vllm,triton,torchinductor,flashinfer,tilelang}/*    # 2 台とも
```

### 確認

**`/health` で見て、`/v1/models` を見るな。** 後者は engine が死んでいても 200
を返す (前者のみ `EngineDeadError` で 503)。

```bash
until curl -sf http://127.0.0.1:8910/health >/dev/null; do sleep 20; done

curl -s http://127.0.0.1:8910/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash",
       "messages":[{"role":"user","content":"What is 23*17? Think briefly."}],
       "max_tokens":400,"temperature":0}'
```

正解のレスポンスは reasoning が**分離**されている:

```json
{
  "role": "assistant",
  "reasoning": "23 * 17 = 23 * 20 - 23 * 3 = 460 - 69 = 399.",
  "content": "399"
}
```

`reasoning` が `null` で chain-of-thought が `content` の先頭に貼り付いているなら、
`VLLM_PARSER_ARGS` から `--reasoning-parser glm45` が落ちている。

ツール呼び出しも確認:

```bash
curl -s http://127.0.0.1:8910/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"ls /tmp"}],
       "tools":[{"type":"function","function":{"name":"run","parameters":{"type":"object","properties":{}}}}],
       "max_tokens":100,"chat_template_kwargs":{"enable_thinking":false}}'
```

`finish_reason: "tool_calls"` が返れば OK (`--tool-call-parser glm47` が効いている)。

### 停止

```bash
sudo docker compose --profile head down
ssh -t "$WORKER" 'cd ~/repos/llm-container && sudo docker compose --profile worker down'
```

**片方だけ落とさない。** 死んでいる rank との rendezvous は新しい rank を固める。
再起動するときは両方を down してから、worker -> head の順で上げ直す。

コンテナを落としても UVM は完全には解放されない。別のモデル構成に切り替えるときは
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
    model="glm-5.3-flash",
    messages=[{"role": "user", "content": "..."}],
    temperature=0,                     # speculation と相性が良い (推奨)
)
```

**thinking は `reasoning` フィールドに入る。** `reasoning_content` ではないので注意
(多くの OpenAI 互換クライアントは `reasoning_content` を見にいくため、thinking が
表示されないことがある)。`--reasoning-parser glm45` を外すと thinking が
`content` 側に混ざる (エージェントハーネスで誤解析される) ので、外さないこと。

```python
r.choices[0].message.reasoning    # <- ここ
```

**既定は thinking ON。** 切る場合はリクエストで
`chat_template_kwargs: {"enable_thinking": false}`。ただし thinking off のまま
エージェント用途に使うと、タグのない思考文章が content に出るので非推奨。

thinking がトークン予算の大半を食うので、`max_tokens` は成果物より大きめに。

### vision (画像・動画)

チェックポイント同梱の chat template がマルチモーダル (image/video の
placeholder 発行を含む) なので、**追加設定なし**で動く:

```python
r = client.chat.completions.create(
    model="glm-5.3-flash",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "この色は?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    ]}],
    max_tokens=200,
)
```

`--chat-template` は**渡していない**。checkpoint 同梱の
`chat_template.jinja` が image/video/audio の placeholder マクロと tool_call ID の
重複判定を持つ最新版で、上流 repo が同梱している `chat_template_mm.jinja` より
新しい。上流版を渡すと機能後退になるので入れていない。

枚数上限は `.env` の `VLLM_MM_ARGS` (既定 `--limit-mm-per-prompt
{"image":2,"video":1}`)。上流は起動時の最大サイズ video encoder profile が
メモリスパイクを起こした事故 (2026-09-18) を受けて `"video":0` にしているが、
この repo では動画入力を残している。boot 時に落ちるようなら 0 にする。

DFlash2 の drafter はテキスト専用なので、vision リクエストは speculation
されない (速度だけ落ちる、機能はする)。

### Anthropic 互換 (`/v1/messages`) — Claude Code から使う

vLLM は Anthropic Messages API も生やすので、**変換プロキシなしで Claude Code を
直結できる**。

```bash
curl -s http://127.0.0.1:8910/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"glm-5.3-flash","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
```

`~/.claude/settings.json` の `env` を向ける (この repo の
`scripts/local-llm-on.sh` / `local-llm-off.sh` が設定ファイルを差し替える):

```jsonc
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8910",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_DEFAULT_OPUS_MODEL":   "glm-5.3-flash",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.3-flash",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL":  "glm-5.3-flash"
  }
}
```

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
sudo docker logs -f glm53-head
ssh -t "$WORKER" 'sudo docker logs -f glm53-worker'
```

speculative decoding の健全さ:

```bash
curl -s http://127.0.0.1:8910/metrics | grep -E 'spec_decode_num_(draft|accepted)_tokens_total'
```

accepted ÷ draft が MTP で 0.4〜0.5 前後、DFlash2 で 0.6〜0.8 が健全。
**DFlash2 で 0.15 程度に落ちたら aux 捕捉が壊れている** (crash せずに黙って
悪化する既知のモード)。

---

## コンテキスト長とメモリ

`MAX_MODEL_LEN` がコンテキスト長。既定は **262,144**。

モデルのネイティブ上限は 1,048,576 (1M) だが、TP2 の KV プールでは
`max_concurrency >= 1.0` (フルコンテキスト 1 リクエストが載る) にならないため
**TP2 では 262K が実用上の上限**。1M 単一リクエストは 4 ノード TP4 が必要
(上流の姉妹リポジトリで検証済み)。

### KV は 8 GiB に固定する (2026-09-18 に方針が逆転した)

**このレシピは `--kv-cache-memory 8589934592` (8 GiB) を固定する。**
旧版の README にあった「固定するな」は撤回された。上流も根拠にしていた
`docs/GB10-KV-MEMORY-LADDER.md` / `docs/KV-HUNT-672K-TP2-RECORD.md` ごと
superseded 扱いにしている。

| | KV トークン (262K ctx / fp8 KV) |
|---|---:|
| プロファイラ任せ | 581,040 |
| 6 GiB 固定 | 678,661 |
| **8 GiB 固定 (既定)** | **714,240** |

#### 何が変わって逆転したのか

機構そのものは変わっていない。GB10 には VRAM が無く、GPU の割当はすべて NVRM
ドライバー経由の system RAM。その割当は **MemAvailable ではなく MemFree** を見て、
page cache の回収は強制しない (bounded reclaim)。198GB の重みロードで page cache
が MemFree を取り潰すと、大きな slab は「予約は成功して touch で死ぬ」
(phantom backing) になる。

**変わったのは MemFree を空け続ける仕組みが入ったこと。** 旧ラダーは以下が全部
無い状態で測られている:

- `MAX_JOBS=2` / `FLASHINFER_NVCC_THREADS=1` — 既定値だと FlashInfer が CPU 数ぶんの
  nvcc を撒き、重みが常駐した状態で `cicc` が OOM killer を呼ぶ
- 永続 JIT キャッシュ (`flashinfer` / `tilelang` / `triton`) — 再コンパイルの
  メモリスパイクが boot ごとに来ない
- `MEM_LIMIT=112g` (cgroup) — 超過が「ホストごと巻き込む page-allocator livelock」
  ではなく「コンテナの clean な OOM」になる
- `scripts/flusher.sh` — 20 秒ごとに無条件で clean cache を落とす

さらに旧ラダーは **MTP-4** での測定で、MTP head が約 4GB/rank を別に食っていた。
DFlash2 の drafter は MLA テンソルに slot-share するので **KV 追加コストがほぼ 0**。

hardening 抜きで固定すると旧ラダー通りに落ちる。**セットで入れること。**
OOM が出るなら `KV_CACHE_MEMORY=6442450944` (6 GiB) に落とす。
`0` か空でプロファイラ任せに戻る。

#### 固定したときの制約: `MAX_NUM_BATCHED_TOKENS` を上げられない

`--kv-cache-memory` を渡すと **vLLM はメモリプロファイリングを丸ごとスキップする**。
つまり `--max-num-batched-tokens` の活性化ピークを誰も検証しない。上流の実測で
**mnbt 16384 は両ノードが `NVRM: NV_ERR_NO_MEMORY` で死んだ** (KV 8 GiB 固定時)。

検証済みの上限は **8192**。entrypoint と preflight が、KV 固定時に 8192 を超える
組み合わせを起動前に弾く。

`MAX_NUM_SEQS` も 6 から上げないこと。上流が 32 を試して C12 で aggregate +10%
だけ、C8〜C16 で TTFT p90 が 60〜179 秒に悪化した。

### boot hardening (KV 固定の前提条件)

| 対策 | どこ | 効果 |
|---|---|---|
| `MAX_JOBS=2` / `FLASHINFER_NVCC_THREADS=1` | `.env` | JIT storm で `cicc` が OOM killer を呼ぶのを止める |
| 永続 `flashinfer` / `tilelang` / `triton` キャッシュ | compose の volumes | 初回 boot 41 分 -> 以降 16〜19 分 |
| `MEM_LIMIT=112g` (`mem_limit` / `memswap_limit`) | compose | 超過をコンテナ OOM に閉じ込める。UMA の GPU 割当は cgroup に課金されないのでホスト側だけを縛る |
| `scripts/flusher.sh` (20 秒ごと無条件) | ホスト | MemFree を空け続ける |
| `drop_caches` + `swappiness=0` | 毎 boot | 下記「起動前にやるメモリ儀式」 |

上流はこれを入れる前、2026-09-18 の 1 回の boot で **4 ノード全部を落として
3 台が watchdog リブート**している。しかもリブート後に GPU の電力が約 14W に
張り付き (クロッククランプ)、その夜のベンチが全部無効になった。

### prefix cache (#18 の修正)

`scripts/build-prefix-fix.sh` を 2 台で実行していない場合、**prefix cache は
1 hit もしない**。DFlash2 の draft group が target group の hit 長を潰すバグで、
ブロック境界に揃ったプロンプトを再送しても `prefix_cache_hits_total` が 0 のまま。
エージェントのセッションが毎ターン会話全体を再 prefill する。

実測: 13K トークン反復の TTFT **21.1s -> 6.3s** (-70%)、
262K トークン反復で hit 率 **0.986** / warm TTFT p95 **2.98s**。

検証のしかた (`--enable-prompt-tokens-details` は既定で入っている):

```bash
# 同じ長いプロンプトを 3 回投げる (temperature 0)
curl -s http://127.0.0.1:8910/metrics | grep prefix_cache
```

読み違えやすい点が 2 つある:

- **2,304 トークン (1 ブロック) 未満のプロンプトは原理的に hit しない。**
  `--block-size 2304` の倍数に切り下げた分だけがキャッシュされる。
- **commit は 2 回目、hit が数字に出るのは 3 回目。** 2 回目で増えないから
  壊れていると判断しないこと。

レスポンス側では `prompt_tokens_details.cached_tokens` が同じ数を返す。

### 起動前にやるメモリ儀式 (毎 boot)

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'   # 2 台とも
sudo sysctl -w vm.swappiness=0                          # 2 台とも (再起動で消える)
```

`./scripts/preflight.sh` は未設定を検出したら `sudo -n` が使えるときに
自動で実行します (使えない場合は手動コマンドを提示)。

drop_caches は重みロードで page cache が再充填されるので完全な解決ではないが、
KV slab の割当競争に勝つ確率が上がる (上流も毎 boot 実行)。

### OOM 系の症状

- **`NV_ERR_NO_MEMORY` / `_memdescAllocInternal` (dmesg)**: KV slab の割当失敗。
  まず `scripts/flusher.sh` が両ノードで回っているか確認する (回っていないのが
  一番多い原因)。次に儀式 + 他のワークロード停止。それでも出るなら
  `KV_CACHE_MEMORY` を 6442450944 (6 GiB) に落とす。
  `MAX_NUM_BATCHED_TOKENS` が 8192 を超えていないかも見る。
- **shard ロードが同じ場所で凍る / UVM kthread が 100% になる**: swap livelock。
  `vm.swappiness=0` が効いているか確認 (swap 無効化でも既定 swappiness でも起きる)。
- **worker が exit 137**: UMA の OOM-kill。両ノード再起動してからやり直す。

---

## 速度

上流の実測 (同一の 2x Spark / TP2 / 262K / fp8 KV / C1 / temperature 0):

| 構成 | decode | 備考 |
|---|---:|---|
| bf16 KV / 非 speculative | 14.3 t/s | 参考値 |
| fp8 KV + MTP-4 (`presets/mtp4.env`) | 21.8 t/s | acceptance 約 0.5 |
| **fp8 KV + DFlash2 (既定)** | **46.9 t/s** | acceptance 74.1% / 2.15x |

2026-09-18 の healthy-fleet 実測。RoCE all-reduce + prefix 修正 + KV 8 GiB 固定を
入れる前後 (aggregate t/s / median of 3):

| | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| before | 43.4 | 29.0 | 30.2 | 50.3 | 44.4 | 47.8 |
| after | 42.7 | 33.8 | 35.9 | 59.0 | 40.9 | 51.6 |

cold prefill (salted prompt / TTFT -> tok/s):

| プロンプト長 | before | after | Δ |
|---:|---|---|---:|
| 5,942 tok | 4.98s -> 1,192 | 3.95s -> 1,506 | +26% |
| 29,868 tok | 31.4s -> 952 | 24.1s -> 1,242 | +30% |
| 113,910 tok | 115.8s -> 984 | 85.3s -> 1,336 | +36% |

**要約: prefill +26〜36% / aggregate +8〜19% (C2-C4, C6) / 単発 decode は横ばい
(flat 〜 +8%)。** この夜に触ったのは step の外側なので、単発 decode が伸びないのは
想定どおり。

実プロンプトでの単発 decode (40 プロンプト / 8 カテゴリ / 3 回):

| | tok/s |
|---|---:|
| 散文 | 18.8 |
| コード | 52.2 |
| 4 並列の実プロンプト混在 | 合計 31.4 (first token 1.97s) |

数え上げプロンプト (count-to-100 系) は **draft acceptance の上限を測る指標**で、
decode 性能として引用してはいけない。drafter にとって最も予測しやすいテキスト。

### さらに速くする

| やること | 効果 | 手順 |
|---|---|---|
| RoCE all-reduce | aggregate +5〜18% | `presets/roce.env` + `patches/roce/README.md` |
| 並行度重視の k schedule | C4 +29.9% / C6 +18.3% (単発は -15〜-24%) | `presets/deep-concurrency.env` |
| thinking off | acceptance +8% | `presets/thinking-off.env` (**reasoning が content に混ざる**) |
| `temperature: 0` | +13〜21% | リクエスト側 |

### MTP-4 に戻す

```bash
# 1. v8 イメージを 2 台で pull (README「2. イメージを取得」参照)

# 2. JIT キャッシュを 2 台で消す (イメージが変わるので必須)
sudo rm -rf vllm-cache/{vllm,triton,torchinductor,flashinfer,tilelang}/*
ssh "$WORKER" 'cd ~/repos/llm-container && sudo rm -rf vllm-cache/{vllm,triton,torchinductor,flashinfer,tilelang}/*'

# 3. 2 台とも --env-file を重ねて起動 (worker -> head)
ssh -t "$WORKER" 'cd ~/repos/llm-container && sudo docker compose \
  --env-file .env --env-file presets/mtp4.env --profile worker up -d'
sudo docker compose --env-file .env --env-file presets/mtp4.env --profile head up -d
```

`mtp4.env` は KV の固定も外す (`KV_CACHE_MEMORY=0`)。8 GiB 固定は DFlash2 構成でしか
測られておらず、MTP head の約 4GB/rank を足した組み合わせの実測が無いため。

### クロッククランプ (ベンチの前に必ず見る)

```bash
./scripts/gputest.sh      # 2 台とも
```

GB10 は不正なリセット (watchdog リブート / `nvidia-smi -r` 後の CUDA 実行) のあと、
**プラットフォームの電力バジェットが約 14W のフォールバック値に張り付く**ことがある。
負荷時 611〜890MHz / bf16 で 26〜33 TFLOPS しか出ず、測定値が全部 2.5 倍遅くなる。
上流はこれで一晩のベンチを丸ごと捨てている。

- 健全: **65〜82 TFLOPS**。**50 未満はクランプ。**
- 復帰方法は **AC 電源を抜く**だけ。`nvidia-smi -r` / `-lgc` / `-ac` / `-pl` と
  通常の再起動では戻らない。
- **`nvidia-smi -r` の後に再起動せず CUDA を回すと SMMU timeout で GPU が fault する。**
  やらないこと。

### 速度を見ていないときに最初に見るもの

decode が遅い原因は「step が遅い」ではなく **draft acceptance が低い**ことが多い。
`tok/s = steps/s × 1 step あたりの採択トークン数`。`/metrics` の
`spec_decode_num_accepted_tokens_total ÷ ..._num_draft_tokens_total` を見る
(上「状態を見る」参照)。

---

## その他

### 再起動で消えるもの

- `vm.swappiness=0` (都度 `sysctl -w`)
- QSFP の IP (link-local が再割り当てされることがある / preflight が検出)
- MTU (netplan 未適用なら 1500 に戻る / preflight が WARN)

### 2 台目の repo 同期

`.env` / `docker-compose.yml` / `scripts/` / `patches/` / `presets/` は 2 台で
**完全に一致していること**。イメージや引数が片方だけだと rendezvous で死ぬか、
起動した後に黙って壊れる。

```bash
rsync -av --exclude .git --exclude models --exclude vllm-cache ./ "$WORKER_MGMT:~/repos/llm-container/"
# 2 台で確認
grep '^VLLM_IMAGE' .env "$WORKER_MGMT 経由の同等ファイル"
```

**git に入らない生成物も 2 台で揃える必要がある** (`.gitignore` 済み):

| パス | 作り方 |
|---|---|
| `patches/kv_cache_coordinator_prefix_fix.py` | `./scripts/build-prefix-fix.sh` (イメージ依存。上の rsync でコピーしてもよい) |
| `patches/roce/{b12x,b12x-1.3.0.dist-info,b12x-roce}/` | `./scripts/build-roce-bundle.sh` (`ROCE=1` を使う場合だけ) |

片方だけに prefix 修正が入っている状態でも起動はするが、rank 間で KV の扱いが
変わるので**必ず両方揃えること**。起動ログの `prefix-cache 修正 (#18) ON` を
両 rank で確認する。

### ハマりどころ

| 症状 | 原因と対処 |
|---|---|
| `Failed to infer device type` / `Can't initialize NVML` | ① rootful daemon に nvidia ランタイムが未登録 → `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`。② `no-cgroups=true` で device cgroup に弾かれている → compose の `devices:` で `/dev/nvidia*` を渡す (対応済み)、最終手段は `privileged: true` |
| warmup で `pe_dim must be 64 for fp8_ds_mla` の assert | **素の vLLM / 素の day-0 イメージを使っている**。SM121 パッチ入り (`sm121-v8` 以上) が必要 |
| `ncclCommInitRank: internal error` で rendezvous 死亡 | イメージの NCCL バージョン問題 (2.29.x は fabric で死ぬ) か、NIC/HCA 名違い。`sm121-v8` 以上なら前者は回避済み。`show_gids` で GID index を確認し直す |
| ~24K トークン超の context で decode 開始直後に engine 死亡 (dmesg クリーン / `EngineDeadError`) | kpool top-k バグ。`patches/sparse_attn_indexer_kpool_sm121.py` の bind-mount が効いていない (entrypoint が起動時にチェックする) |
| 出力が `locklock` 系の無意味な反復 / logprobs が NaN | FlashInfer 0.6.17 の FA2 MLA NaN。`sm121-v8` 以上を使う |
| KV 割当は成功するのに warmup や 1 リクエスト目で NVRM OOM | ① `scripts/flusher.sh` が両ノードで回っていない (最も多い) ② `MAX_NUM_BATCHED_TOKENS` が 8192 超 ③ MemFree 不足 → メモリ儀式。それでも出るなら `KV_CACHE_MEMORY=6442450944` |
| ベンチが上流の半分以下 / 全プロンプトが一様に遅い | GPU のクロッククランプ。`./scripts/gputest.sh` が 50 TFLOPS 未満なら AC 電源を抜いて入れ直す (`nvidia-smi -r` では戻らない) |
| 同じプロンプトを再送しても `prefix_cache_hits_total` が 0 | ① `scripts/build-prefix-fix.sh` を実行していない (起動ログの `prefix-cache 修正 (#18) OFF` を確認) ② プロンプトが 2,304 トークン未満なので原理的に hit しない ③ まだ 2 回目 (hit は 3 回目に出る) |
| `RoCEnante all-reduce is live` がログに出ない | `patches/roce/b12x` が無いので NCCL にフォールバックしている。`scripts/build-roce-bundle.sh` (`patches/roce/README.md`) |
| 起動が「RoCE パッチの対象ツリーと一致しません」で止まる | イメージの vLLM が `patches/roce/` の想定と違う。`ROCE=0` に戻すか、アダプタを新ツリー向けに作り直す |
| preflight が ModelOpt ビルドで NG | ModelOpt NVFP4 は token を壊す (vLLM #54150)。RedHatAI (compressed-tensors) を使う |
| `reasoning` が `null` で思考が content に混ざる | `--reasoning-parser glm45` が無い / 壊れている |
| ツール呼び出しが JSON 文字列で返る | `--tool-call-parser glm47 --enable-auto-tool-choice` が無い |
| 2 台で起動したが API が永遠に出ない | 片方の `VLLM_IMAGE` / 引数が違う (2 台で `grep '^VLLM_IMAGE' .env` / compose の環境を diff)。`docker logs` を**両方**見る (worker 側の死因が log に出ることが多い) |
| 文字化け・意味不明な出力 (イメージ変更直後) | 旧 JIT キャッシュ残骸。`sudo rm -rf vllm-cache/{vllm,triton,torchinductor,flashinfer,tilelang}/*` で再起動 |
| `/v1/models` は 200 なのに生成が死んでいる | engine dead。`/health` で見る (503 なら死亡) |

---

## 出典

この構成は以下の実機レポートを組み合わせたもの。

- **[tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)**
  — 現在参照している後継リポジトリ。この repo の既定はここの `CURRENT.md` に
  合わせてある。KV 8 GiB 固定 / mnbt 8192 / gmu 0.85 / DFlash2 k=7 /
  boot hardening (`docs/SPEED-NIGHT-2026-09-18.md`) / #18 prefix-cache 修正
  (`docs/PREFIX-CACHE-DFLASH2-SM121.md`) / speculative depth の実測
  (`docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md`) / b12x RoCE all-reduce
  (`speed-night-2026-09-18/roce/`)
- [tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark) — 2x Spark TP2 の world-first (上記の前身)。
  7 つの day-0 バグの root cause と修復、使っているパッチ入りイメージ
  (`sm121-v8` / `sm121-v11-dflash2`)、kpool top-k 修正、
  DFlash2 の KV slot-share 機構とベンチ。**KV/MemFree ラダーの「固定するな」は
  後継リポジトリで撤回されている** (boot hardening 導入後に 8 GiB 固定が既定)
- [barrydeen/glm53-flash-dgx-spark](https://github.com/barrydeen/glm53-flash-dgx-spark) —
  同一パッチスタックを別の fleet で検証・再構築したレシピ。`gmu 0.85` と
  `--block-size 2304` / `--enforce-eager` の根拠
- [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) — 重みと day-0 公式イメージ / parser 名 (`glm47` / `glm45`)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) — DFlash2 drafter (CC-BY-NC-ND-4.0)
- 2 ノード TP (mp バックエンド / RoCE / GID 固定) の基盤は本 repo の
  DeepSeek-V4-Flash 構成 (main branch) のもの
