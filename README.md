# GLM-5.3-Flash-NVFP4 on 2x DGX Spark

[RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
(320B total / 18B active MoE / NVFP4 / 198GB) を DGX Spark 2 台に **TP=2** で
分割して、OpenAI 互換 API を `:8910` に生やす構成。

- 分散バックエンドは **Ray ではなく `mp`** (torch.distributed SPMD)。head/worker が
  それぞれ `vllm serve` を `--nnodes/--node-rank/--master-addr` 付きで起動する。
- ノード間の NCCL は 200GbE QSFP 直結リンクの **RoCEv2 (RDMA)**。
- **MTP-4 speculative decoding** (checkpoint 同梱の MTP head / decode 約 21.8 t/s)。
  高速化 preset で **DFlash2** (約 46.9 t/s / 2.15x) に切り替え可能。
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
| `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8` (既定) | day-0 公式 (`vllm/vllm-openai:glm53-flash`) + パッチ 8 段: SM90 NoPE-MLA バックエンドを SM121 で有効化 (FA2) / FlashInfer 0.6.18 (0.6.17 は 64〜256 行バッチで NaN) / NCCL 2.30.7 固定 (nightly が 2.29.7 に落とすと fabric で死ぬ) / cutlass-dsl 4.6.2 / PDL を SM12x で無効 / indexer 強化 / fp8 KV |
| `:sm121-v11-dflash2` (preset) | v8 + DFlash2 drafter 対応 overlay |

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

`model_mtp.safetensors` (7.6GB) も同じリポジトリに含まれていて、MTP speculative
decoding の draft head として使う。別途のダウンロードは不要。

**DFlash2 preset を使う場合だけ**追加で drafter を取る (2.2GB /
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

DFlash2 preset を使うならそのイメージも両台で pull:

```bash
sudo docker pull ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
```

### 3. メモリを空ける (2 台とも)

重みが 1 台あたり約 99GB + MTP head 約 4GB + KV + ランタイム。121GB の
unified memory に対して余裕がほとんどないので、**起動前に他のワークロードを
止める**。GB10 の UVM は一度確保されると完全には返らないので、迷ったら再起動が
一番確実。

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

### 5. preflight

```bash
./scripts/preflight.sh                                         # head
ssh "$WORKER" 'cd ~/repos/llm-container && ./scripts/preflight.sh'
```

NG が出ている状態で起動しても、5〜10 分待たされてから NCCL エラーか OOM-kill
で死ぬだけ。kpool パッチの存在もここで確認される。

---

## 起動

**worker を先に、head を後に。** head が rendezvous の master になるので、worker が
先に待ち受けている状態にしてから head を上げる。

```bash
# --- worker ---
ssh -t "$WORKER" 'cd ~/repos/llm-container && sudo docker compose --profile worker up -d'

# --- head ---
sudo docker compose --profile head up -d
sudo docker logs -f glm53-head
```

初回は CuTeDSL / Triton の JIT が走るので **15〜20 分**かかる。2 回目以降は
`vllm-cache/` が効いて短くなる。`Application startup complete` が出れば完了。

**イメージを跨いだら `vllm-cache/{vllm,triton,torchinductor}/*` を消すこと**
(DeepSeek 構成から切り替える場合を含む)。JIT キャッシュが混ざると起動不能になる。

```bash
sudo rm -rf vllm-cache/{vllm,triton,torchinductor}/*    # 2 台とも
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

DFlash2 preset では drafter がテキスト専用なので、vision リクエストは
speculation されない (速度だけ落ちる、機能はする)。

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

### KV サイズは固定しない (GB10 の MemFree 壁)

**`--kv-cache-memory` を大きな値に固定すると warmup で NVRM OOM する。**
上流が 6 回の boot で実証したラダー (fp8 KV + MTP-4 / TP2):

| 固定値 | KV トークン | 結果 |
|---:|---:|---|
| 4.14 GiB (vLLM が起動ログで出す建議値) | 507,041 | **安定 (3/3)** |
| 5.5 GiB | 672,606 | worker 死亡 (NVRM OOM) |
| 6.5 GiB | 796,779 | head 死亡 |
| 7.5 GiB | 920,953 | 5 回全滅 |

機構: GB10 には VRAM が無く、GPU の割当はすべて NVRM ドライバー経由の
system RAM。その割当は **MemAvailable ではなく MemFree** を見て、page cache の
回収は強制しない (bounded reclaim)。198GB の重みロードで page cache が
MemFree を取り潰しているので、大きな slab は「予約は成功して touch で死ぬ」
(phantom backing) になる。vLLM 0.1x は integrated GPU の free を
`psutil.available` (= page cache 込み) と誤認するため、提案値自体が
実態より大きい場合がある。**起動ログの建議値より大きくしないこと。**

既定構成は固定値を一切渡さず、プロファイラの自動サイズに任せている
(fp8 KV で約 50 万トークン / 262K ctx で `max_concurrency ≈ 1.9x`)。

### 起動前にやるメモリ儀式 (毎 boot)

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'   # 2 台とも
sudo sysctl -w vm.swappiness=0                          # 2 台とも (再起動で消える)
```

drop_caches は重みロードで page cache が再充填されるので完全な解決ではないが、
KV slab の割当競争に勝つ確率が上がる (上流も毎 boot 実行)。

### OOM 系の症状

- **`NV_ERR_NO_MEMORY` / `_memdescAllocInternal` (dmesg)**: KV slab の割当失敗。
  上の儀式 + 他のワークロード停止。KV が要らないなら `--speculative-config` を
  外す (MTP head の約 4GB/rank が解放される)。
- **shard ロードが同じ場所で凍る / UVM kthread が 100% になる**: swap livelock。
  `vm.swappiness=0` が効いているか確認 (swap 無効化でも既定 swappiness でも起きる)。
- **worker が exit 137**: UMA の OOM-kill。両ノード再起動してからやり直す。

---

## 速度

上流の実測 (同一の 2x Spark / TP2 / 262K / fp8 KV / C1):

| 構成 | decode | 備考 |
|---|---:|---|
| bf16 KV / 非 speculative | 14.3 t/s | 参考値 |
| **fp8 KV + MTP-4 (既定)** | **21.8 t/s** | acceptance 約 0.5 |
| **fp8 KV + DFlash2 (preset)** | **46.9 t/s** | acceptance 74.1% / 2.15x |

DFlash2 の並行性 (400 トークン生成 / 2 ウェーブ):

| | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| 合計 t/s | 35.1 | 41.6 | 40.6 | 47.5 | **56.2** | 47.7 |
| 1 ストリーム当り | 35.1 | 23.2 | 17.3 | 15.3 | 17.5 | 13.3 |

DFlash2 に切り替える:

```bash
# 1. drafter を 2 台に取得 (2.2GB / CC-BY-NC-ND-4.0 非商用ライセンス)
./scripts/fetch-model.sh draft
ssh "$WORKER" 'cd ~/repos/llm-container && ./scripts/fetch-model.sh draft'

# 2. イメージを 2 台で pull (README「2. イメージを取得」の末尾の digest)

# 3. JIT キャッシュを 2 台で消す
sudo rm -rf vllm-cache/{vllm,triton,torchinductor}/*
ssh "$WORKER" 'cd ~/repos/llm-container && sudo rm -rf vllm-cache/{vllm,triton,torchinductor}/*'

# 4. 2 台とも --env-file を重ねて起動 (worker -> head)
ssh -t "$WORKER" 'cd ~/repos/llm-container && sudo docker compose \
  --env-file .env --env-file presets/dflash2.env --profile worker up -d'
sudo docker compose --env-file .env --env-file presets/dflash2.env --profile head up -d
```

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

### ハマりどころ

| 症状 | 原因と対処 |
|---|---|
| `Failed to infer device type` / `Can't initialize NVML` | ① rootful daemon に nvidia ランタイムが未登録 → `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`。② `no-cgroups=true` で device cgroup に弾かれている → compose の `devices:` で `/dev/nvidia*` を渡す (対応済み)、最終手段は `privileged: true` |
| warmup で `pe_dim must be 64 for fp8_ds_mla` の assert | **素の vLLM / 素の day-0 イメージを使っている**。SM121 パッチ入り (`sm121-v8` 以上) が必要 |
| `ncclCommInitRank: internal error` で rendezvous 死亡 | イメージの NCCL バージョン問題 (2.29.x は fabric で死ぬ) か、NIC/HCA 名違い。`sm121-v8` 以上なら前者は回避済み。`show_gids` で GID index を確認し直す |
| ~24K トークン超の context で decode 開始直後に engine 死亡 (dmesg クリーン / `EngineDeadError`) | kpool top-k バグ。`patches/sparse_attn_indexer_kpool_sm121.py` の bind-mount が効いていない (entrypoint が起動時にチェックする) |
| 出力が `locklock` 系の無意味な反復 / logprobs が NaN | FlashInfer 0.6.17 の FA2 MLA NaN。`sm121-v8` 以上を使う |
| KV 割当は成功するのに warmup や 1 リクエスト目で NVRM OOM | `--kv-cache-memory` を大きく固定している、または MemFree 不足。固定値を外して (既定構成は無し) メモリ儀式をやる |
| `reasoning` が `null` で思考が content に混ざる | `--reasoning-parser glm45` が無い / 壊れている |
| ツール呼び出しが JSON 文字列で返る | `--tool-call-parser glm47 --enable-auto-tool-choice` が無い |
| 2 台で起動したが API が永遠に出ない | 片方の `VLLM_IMAGE` / 引数が違う (2 台で `grep '^VLLM_IMAGE' .env` / compose の環境を diff)。`docker logs` を**両方**見る (worker 側の死因が log に出ることが多い) |
| 文字化け・意味不明な出力 (イメージ変更直後) | 旧 JIT キャッシュ残骸。`sudo rm -rf vllm-cache/{vllm,triton,torchinductor}/*` で再起動 |
| `/v1/models` は 200 なのに生成が死んでいる | engine dead。`/health` で見る (503 なら死亡) |

---

## 出典

この構成は以下の実機レポートを組み合わせたもの。

- [tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark) — 2x Spark TP2 の world-first。
  7 つの day-0 バグの root cause と修復、使っているパッチ入りイメージ
  (`sm121-v8` / `sm121-v11-dflash2`)、kpool top-k 修正、GB10 の KV/MemFree ラダー、
  DFlash2 の KV slot-share 機構とベンチ
- [barrydeen/glm53-flash-dgx-spark](https://github.com/barrydeen/glm53-flash-dgx-spark) —
  同一パッチスタックを別の fleet で検証・再構築したレシピ。`gmu 0.85` と
  `--block-size 2304` / `--enforce-eager` の根拠
- [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) — 重みと day-0 公式イメージ / parser 名 (`glm47` / `glm45`)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) — DFlash2 drafter (CC-BY-NC-ND-4.0)
- 2 ノード TP (mp バックエンド / RoCE / GID 固定) の基盤は本 repo の
  DeepSeek-V4-Flash 構成 (main branch) のもの
