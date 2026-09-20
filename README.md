# DeepSeek-V4.1-Flash (EXL3 2.9bpw) on 2x DGX Spark

[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
を **EXL3 2.9bpw 量子化** (avg 2.9 bits/weight / codebook mul1 / head_bits=6 /
mtp_bits=4) で DGX Spark 2 台に **TP=2** で分割して、OpenAI 互換 API を `:8910` に
生やす構成。

- 分散バックエンドは **Ray ではなく `mp`** (torch.distributed SPMD)。head/worker が
  それぞれ `vllm serve` を `--nnodes/--node-rank/--master-addr` 付きで起動する。
- ノード間の NCCL は 200GbE QSFP 直結リンクの **RoCEv2 (RDMA)**。
- **native DSpark speculative decoding (k=3)** — draft がチェックポイント内蔵
  (`mtp.*` / dspark_block_size=5 / draft experts 128 top-3 / target layers 37-39)。
  DFlash のような別 drafter も追加 DL も無い。
- 出典レシピ: **[MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)**。
  向こうの `start.sh` + NFS share 構成を、この repo 流の
  「2 台ローカルに重み + docker compose + entrypoint.sh」に読み替えてある。

## Vision-Exp ブランチから変わったところ

| | Vision-Exp (前ブランチ) | V4.1-Flash (ここ) |
|---|---|---|
| 重み | FP8+FP4 ネイティブ 168GB | **EXL3 2.9bpw 197GiB + Engram 190GiB** |
| イメージ | 上流 `vllm/vllm-openai` pre-release + **ランタイム hotfix 40 個** | `ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks` (**overlay 焼き込み済み**) |
| FlashInfer ブロッカ | SM120 sparse-MLA 1152 幅が無くて上流構成はブロック | overlay の `patch_sm120_block64.py` が **max_image_tokens=0 に pin** して回避 |
| KV | `--kv-cache-dtype fp8` + `--block-size 256` を手動指定 | **渡さない** (CSA2 自動選択 / `--block-size 64` が必須側) |
| DSpark | k=3 probabilistic + adaptive=false | k=3 (method=dspark のみ。k の天井 5、実測 3 が最速) |
| コンテキスト | 1M 張り付き (実際は KV が持たない) | **600K 検証済み** (614,400 が実測天井 / 2.5GiB で 774,400 tok) |
| 精度 | ネイティブ | 3bit 級量子化。**量子化誤差は気にしない運用** (本体の指示) |

Vision-Exp 時代までに `patches/` に移植していた hotfix 群は V4.1 では使わない
(イメージ側に同等の修正が入っている)。`patches/` と `presets/` は前のブランチの
遺産として残してあるだけなので、この構成では引き合いに出さないこと。

## 重み (2 種 / 両ノードに)

| 置き場 | HF repo | 中身 | サイズ |
|---|---|---|---|
| `MODEL_PATH` (`./models/DeepSeek-V4.1-Flash-EXL3-2.9bpw`) | `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw` | EXL3 39 shard + config + k_map | 約 197 GiB |
| `ENGRAM_PATH` (`./models/DeepSeek-V4.1-Flash-engram`) | `deepseek-ai/DeepSeek-V4.1-Flash` の **shard 47+48 + index + config.json のみ** | layers 1/14 の n-gram 埋込 (FP8 rows) | 約 190 GiB |

**Engram テーブルだけは量子化されない。** EXL3 ツリーに入っていないので、
ネイティブチェックポイントから `--hf-overrides {"engram_table_dir":...}` で別途
教える。ネイティブ 48 shard / 476GiB 全部は要らない (47+48 だけ / 約 95GiB×2)。

- rows は FP8 (`fp8_e4m3fn`)。uint8 として読むと Fluent な garbage が出る
  (prepare-engram-src.py が作る slim index はその対策込み)。
- 既にネイティブ checkout を持っている人は `NATIVE_SRC=<path> ./scripts/fetch-model.sh engram`
  で 190GiB の DL をスキップできる。
- 上流には「Engram を NVMe に per-rank pack すると prefill が 25-50% 速くなる
  (`./start.sh pack`)」もあるが、**この repo では未移植** (起動には不要)。

## メモリ予算 (1 台 / 121.7 GiB unified)

GB10 は host RAM が GPU メモリなので、すべての cudaMalloc が即 host 実メモリの
commit になる。上流実測 (2026-09-12/13):

| 内訳 | 量 |
|---|---|
| EXL3 重み (resident / rank) | 約 99.5 GiB |
| KV プール (2.5GiB 固定) | 2.5 GiB |
| CUDA context + NCCL + graphs | 5〜7 GiB |
| vLLM プロセス / OS / docker | 約 9 GiB |
| **稼働後の MemAvailable** | **3.7〜4.7 GiB** |

- 600K の 1 リクエストが約 2.0GiB。2.5GiB プールは 774,400 トークン (1.26 倍)。
- 601K prefill 直後の low-water は head で 2.1GiB。これを超える構成
  (MAX_NUM_SEQS を増やす / batched tokens を上げる) は上流未検証。
- 常時 3〜4GiB しか遊ばないので、**起動前に他のワークロードを止めるのが最大の対策**。
  compose 両コンテナに `oom_score_adj: 1000` を付けているのは、kernel OOM 時に
  デスクトップ側ではなく vLLM を選ばせるため (2026-09-11 に両ノード wedge 事故の実例)。
- 上流の memguard watchdog (`DSV41_MEM_GUARD`) は既定 off。常時運用では必要なかった
  とのことなので、この repo でも移植していない。

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
`MODEL_PATH` / `ENGRAM_PATH` は repo 相対 (`./models/...`) か絶対パスで書くこと。

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

#### `--device` を明示している理由

`/etc/nvidia-container-runtime/config.toml` に **`no-cgroups = true`** が入っていると、
`--gpus all` だけでは GPU が使えない。これは rootless docker で GPU を使うための
必須設定だが、rootful では nvidia-container-cli が **device cgroup の許可リストを
更新しなくなる**ため。compose の `devices:` で /dev/nvidia* を明示して通す。
`no-cgroups = false` にすると rootless 側の GPU が壊れるので設定は触らない。
それでもダメなときは `x-vllm-service` に `privileged: true` を足す。

---

## セットアップ

### 0. repo と .env を 2 台に配る

```bash
cp .env.example .env
$EDITOR .env      # HEAD_ROCE_IP / WORKER_ROCE_IP / MODEL_PATH / ENGRAM_PATH を実機に合わせる

# worker へ同期 (2 台とも .env の中身は同じでよい。役割は --profile で切り替える)
rsync -av --exclude .git --exclude models --exclude vllm-cache ./ "$WORKER:~/repos/llm-container/"
```

`HEAD_ROCE_IP` / `WORKER_ROCE_IP` / `IB_HCA_NAME` / `NCCL_IB_GID_INDEX` は実機で確認:

```bash
ip -br addr show enp1s0f0np0     # QSFP 側の IPv4
show_gids                        # その IPv4 が載っている行の DEV 名 と INDEX (RoCE v2 の方)
```

> 出典レシピの機体は **spark1 の f1 ↔ spark2 の f0** のピン配線だった。
> 直結で同じポートを向いているなら `WORKER_ROCE_IF_NAME` 等は不要。
> NCCL は link-local (169.254.x.x) の alias を通せないことがあるので、静的 IP を
> 振るか、変わっていたら preflight で検出して直しておく。

### 1. 重みを取得 (2 台とも / 各 約 387 GiB)

```bash
./scripts/fetch-model.sh                 # EXL3 (197GiB) + Engram (190GiB)
ssh "$WORKER" 'cd ~/repos/llm-container && ./scripts/fetch-model.sh'
```

- resumable。切れたら同じコマンドの再続行で OK。
- 区切り取得したいときは `./scripts/fetch-model.sh exl3` / `... engram`。
- 保存先は `.env` の `MODEL_PATH` / `ENGRAM_PATH`。
  `hf` が無ければ `pip install -U 'huggingface_hub[cli,hf_transfer]'`。

### 2. イメージを取得 (2 台とも / 約 14GB) — **rootful**

digest 固定。`ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks:2.9bpw`
(= `vllm/vllm-openai:deepseekv41-flash-0909` + ExLlamaV3 v1.4.5 overlay)。
public / ログイン不要。

```bash
IMAGE=$(sed -n 's/^VLLM_IMAGE=//p' .env)
sudo docker pull "$IMAGE"
ssh -t "$WORKER" "cd ~/repos/llm-container && sudo docker pull \$(sed -n 's/^VLLM_IMAGE=//p' .env)"
```

> **必ず起動前に pull しておくこと。** 片方が pull 中に rendezvous が始まると
> ハンドシェイクごと固まる (上流は `IMAGE_SHIP=rsync` で head から配るが、
> ここでは 2 台とも直接 pull する)。

### 3. メモリを空ける (2 台とも)

重み ~99.5GiB + KV 2.5GiB + ランタイム ~16GiB。`MemAvailable` が **110GiB 以上**
無ければ起動手順に入らない。GB10 の UVM は一度確保されると完全には返らないので、
迷ったら再起動が一番確実。

```bash
docker ps && docker stats --no-stream
docker stop <他のコンテナ>
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
sudo sysctl -w vm.swappiness=10
```

### 4. MTU を 9000 に上げる (任意だが推奨 / 一度やれば済む)

QSFP の Ethernet MTU が既定 1500 だと RoCE path MTU が 1024 に落ちる。9000 に上げると
HCA 上限の 4096 まで上がる。永続化は netplan (`/etc/netplan/40-cx7.yaml` に `mtu: 9000`)。
手順の詳細は git 履歴の Vision-Exp 版 README §セットアップ4 と同じ。

```bash
sudo ip link set dev enp1s0f0np0 mtu 9000                # まず一時変更で試す
ssh -t "$WORKER_MGMT" 'sudo ip link set dev enp1s0f0np0 mtu 9000'
ibv_devinfo -d rocep1s0f0 | grep active_mtu              # 4096 (5) になっていれば OK
ping -M do -s 8972 -c 3 "$WORKER"
```

> **SSH は `$WORKER_MGMT` 経由で。** QSFP 側から入っていると接続が切れる。
> 確定は `sudo netplan try` (ミスっても 120 秒で revert)。

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
sudo docker logs -f dsv41-head
```

EXL3 の GEMM shape は CUDA graph capture 前に全部 autotune される (130 launch / 約 3 秒)。
初回ブートは JIT 込みで 10〜20 分、2 回目以降は `vllm-cache/` が効く。
`Application startup complete` が出れば完了。

> 上流は「head のログが 420 秒黙ったら両 rank の py-spy を叩く」hang detector を
> 持っているが、この repo では未移植。怪しいときは手で:
> `sudo docker exec dsv41-head py-spy dump --native <pid>`

### 確認

```bash
curl -s http://127.0.0.1:8910/health
curl -s http://127.0.0.1:8910/v1/models | python3 -m json.tool

# smoke: thinking を落として 17*19。期待値は "323"
curl -s http://127.0.0.1:8910/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash",
       "messages":[{"role":"user","content":"What is 17*19? Reply with the integer only."}],
       "max_tokens":32,"temperature":0,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

> **コールドスタート後の最初の 1 発はタイムアウトすることがある。** 新しい prompt
> shape に対するウォームアップで異常ではない。投げ直せば通る。
> thinking を入れた通常の応答は公式既定 **temperature=1.0 / top_p=0.95**。

### 停止

```bash
sudo docker compose --profile head down
ssh -t "$WORKER" 'cd ~/repos/llm-container && sudo docker compose --profile worker down'
```

コンテナを落としても UVM は完全には解放されない。別の構成に切り替えるときは
**両ノードを再起動してから**始めること。

---

## 使う

`network_mode: host` なので、head に届くアドレスならどれでも叩ける。

```python
from openai import OpenAI
client = OpenAI(base_url="http://<head>:8910/v1", api_key="local")
r = client.chat.completions.create(
    model="deepseek-v4.1-flash-exl3",       # deepseek-v4.1-flash でも通る
    messages=[{"role": "user", "content": "..."}],
    temperature=1.0, top_p=0.95,            # 公式既定
)
r.choices[0].message.reasoning              # thinking はここ (content には混ざらない)
```

### thinking / reasoning effort

- **thinking は既定 ON。** 消すときは `chat_template_kwargs: {"enable_thinking": false}`。
- effort は `chat_template_kwargs.reasoning_effort` で渡す。**既定 "high"**。
  取りうる値: `low`(=50) / `high`(=75) / `max`(=100) / 1〜100 の整数。
- chat template はイメージ側の `files/chat_template.jinja`
  (DeepSeek 純正 `encoding.py` の移植) が読まれる。

### 画像

EXL3 チェックポイントは vision tower を持つ (CED 20+20 / CSA2 / Engram L1,L14 /
vision tower / native DSpark)。既定は ON (`LANGUAGE_MODEL_ONLY=0`) で、
`LIMIT_MM={"image":100}` / `MAX_NUM_BATCHED_TOKENS >= 1536` が条件。

ただし **SM12x では overlay が `max_image_tokens` を 0 に pin する**。
FlashInfer の SM120 sparse-MLA prefill に 1152 幅 (SWA128 + 画像 1024) のカーネルが
無く、 widened パスは guarded branch で通さない設計。つまり画像は通るが
**双方向可視性は入っていない**。上流の実測では OCR・複数画像の属性精度への影響は
nil だったが、ネイティブとの parity probe は無い。**dense な文書タスクは自分で
検証してから信用すること。** text-only で動かすなら `LANGUAGE_MODEL_ONLY=1`
(MAX_NUM_BATCHED_TOKENS も 1024 に下げてよい)。

### Anthropic 互換 (`/v1/messages`) — Claude Code から使う

vLLM は Anthropic Messages API も生やすので、変換プロキシなしで Claude Code を
直結できる。`scripts/local-llm-on.sh` / `local-llm-off.sh` が設定ファイルを差し替える。

```jsonc
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8910",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_DEFAULT_OPUS_MODEL":   "deepseek-v4.1-flash",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4.1-flash",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL":  "deepseek-v4.1-flash"
  }
}
```

### ストリーミング時の注意

reasoning から content へ切り替わるデルタは **両方のフィールドを同時に持つ**。
片方だけ拾うともう片方が落ちる (`elif` にしない)。

```python
rc, cc = delta.get("reasoning"), delta.get("content")
if rc: reasoning.append(rc)
if cc: content.append(cc)
```

### 状態を見る

```bash
curl -s http://127.0.0.1:8910/metrics | grep -E '^vllm:(kv_cache_usage_perc|num_requests)'
sudo docker logs -f dsv41-head
```

イメージ内の overlay は `[dsv41-mem]` 行でフェーズごとのメモリをログに出す。
prefill 前後の MemAvailable 減少をこの行で追うと予算管理が楽。

---

## コンテキストと調整ノブ

**`presets/` は Vision-Exp / 0731 時代のもので、この構成には適用しないこと。**
KV を手で大きくする系譜が異なり、`--kv-cache-dtype` を入れると壊れる。

| ノブ | 既定 | 動かすとき |
|---|---|---|
| `MAX_MODEL_LEN` | 600000 | 天井 614,400 (実測)。短くするのは自由。KV プール (2.5GiB=774,400 tok) に注意 |
| `MAX_NUM_SEQS` | 2 | 上流検証済みは 2 まで。**4 ストリームのバッチ運用なら DSpark を外す方が速い** (集計 54 vs 42 t/s) |
| `MAX_NUM_BATCHED_TOKENS` | 1536 | vision ON の下限 1536 (=画像 1024+1)。`LANGUAGE_MODEL_ONLY=1` なら 1024 可。上流のエージェント向けプロファイルは 1536/96 |
| `GPU_MEMORY_UTILIZATION` | 0.88 | KV を固定している今は起動時 sanity check 相当 (0.92 は 20MB 足りて失敗した実例あり) |
| DSpark k | 3 | 天井 5 だが散文実測で k=3 > k=5 > なし (28/25/23 t/s)。`.env` の `VLLM_EXTRA_ARGS` 内の `num_speculative_tokens` を編集。外すと ~3.5GiB 空く |
| `DSV41_IO_THREADS` | 96 | Engram ランダム行読みのスレッド。host CPU 圧を優先するなら 32 |
| `KV プール` | 2.5GiB | `VLLM_KV_ARGS` の `--kv-cache-memory-bytes`。1GiB (658K tok) まで落とすと 4x128K がちょうど載る |
| `DSV41_CACHE_GIB` / `DSV41_RESIDENT_SCALES` | 0 (off) | Engram 行キャッシュ。**rows の再利用はほぼ 0% という実測なので既定 off**。上げるなら MemAvailable の実測余裕があるときだけ |
| `LONG_PREFILL_TOKEN_THRESHOLD` | off | 入れるなら `MAX_NUM_BATCHED_TOKENS - THRESHOLD >= 256` |

### QSFP を 2 枚使う (dual-HCA) — 帯域がほぼ倍

GB10 の QSFP ケージは PCIe x4 が 2 本で 2 枚の独立 NIC に見える。1 枚しか設定しないと
帯域を半分捨てる。手順 (netplan / MTU / 確認) は git 履歴の Vision-Exp 版 README の
「QSFP を 2 枚使う」がそのまま使える。注意点は共通:

```
.env: IB_HCA_NAME=rocep1s0f0,roceP2p1s0f0
      ROCE_IF_NAME=enp1s0f0np0,enP2p1s0f0np0
      NCCL_IB_MERGE_NICS=1 / NCCL_CROSS_NIC=1
      ★NCCL_IB_GID_INDEX は必ず空にする (index がずれると無言ハング)
```

### cooperative MoE (上流の拡張 / ここでは未有効化)

上流には decode 向けの cooperative MoE 専用経路がある (decode +25〜35%、非 bit-exact)。
`DSV41_COOPERATIVE_MOE=1` + `EXL3_OVERLAY_HOST` + `extensions/.../cooperative_moe.so`
(binary pin あり) が必要で、既定 off のオプション。やる気になったら上流 repo を読むこと。

---

## ハマりどころ

- **`--kv-cache-dtype` を渡さない。** fp8_ds_mla/GLM 式の手指定は NG。CSA2 が自動選択。
  (`--block-size 64` 側は逆に必須。SM12x の paged indexer kernel が 32/64 のみで、
  128 は ratio-1 indexer 層の最初の decode で死ぬ)
- **Engram が無いと「動く」。** マウント抜けは entrypoint が fail-closed で弾くが、
  別モデルの engram ディレクトリを指しているとFluentな garbage が返る。
  rows が fp8_e4m3fn として解釈されていること (`prepare-engram-src.py` の slim index) を確認。
- **exllamav3 は lock buffer が 1 device 1 個。** `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`
  と `DSV41_EXL3_SERIAL_STREAMS=1` を切るとクロスストリームの EXL3 GEMM で
  デッドロックする (上流 boot 11-14)。速そうでも戻さないこと。
- **page cache が RAM を食って見える。** EXL3 の mmap ロード後は page cache を
  捨てないと MemAvailable 9GiB あっても NVRM が OOM する (GB10 ドライバは MemFree から
  取る)。`DSV41_DROP_PAGE_CACHE=1` がその対策で既定 on。
- **JSON に空白を入れると死ぬ。** `VLLM_*_ARGS` は空白で分割して `vllm serve` に渡す。
  `--hf-overrides {"engram_table_dir":"/models/..."} `は末尾空白に注意。
- **NCCL が 10.0.0.x もどきの loopback alias を使えない** という上流の注意は、
  この repo の「QSFP 直結 IP を明示」で代替している。rendezvous が無言で 60 秒後に
  死ぬのは GID index ずれが定番 (`ibv_modify_qp` errno 61)。`show_gids` で確認。
- **再起動後の link-local IP。** NVIDIA の connect-two-sparks は 169.254.x.x を振る。
  変わっていないか preflight が確認する。

## 出典

- [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
  — 本構成のレシピ (start.sh / overlay / 実測値)。イメージ公開:
  `ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks:2.9bpw`
- [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) —
  ネイティブチェックポイント (Engram の出所)
- [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) —
  EXL3 量子化ツリー (39 shard / mul1 / 2.9bpw / quantizer 1.4.2 / runtime ExLlamaV3 1.4.5)
- 前段: [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  — Vision-Exp ブランチで移植した元レシピ。`patches/` はそちらの遺産
