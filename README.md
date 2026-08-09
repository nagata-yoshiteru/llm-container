# MiniMax-M3 (NVFP4) on 3x DGX Spark

[nvidia/MiniMax-M3-NVFP4](https://huggingface.co/nvidia/MiniMax-M3-NVFP4)
(428B MoE / A23B / NVFP4 / 250GB) を DGX Spark 3 台に **PP=3** で分割して、
OpenAI 互換 / Anthropic 互換 API を `:8910` に生やす構成。

| | |
|---|---|
| コンテキスト | 256K (既定) / 最大 512K |
| decode | 約 10 tok/s |
| イメージ | vLLM 公式 nightly (linux-arm64) を digest 固定 |
| 分散 | Ray / PP=3 / TP=1 |

前構成 (DeepSeek-V4-Flash / 2 台 / 51 tok/s) は
`DGX-Spark-2/deepseek-ai/DeepSeek-V4-Flash-0731` ブランチにある。
**速度が要る仕事はそちらの方が速い。** M3 は品質と引き換え。

## 構成の要点

- **TP=3 は使えない。** M3 は `num_key_value_heads=4` で、vLLM は kv_heads と TP の
  どちらかがもう一方で割り切れることを要求する。PP=3 なら 60 layer / 3 = 20 で割れる。
- **重みはイメージと組で選ぶ。** M3 の MSA indexer には実装が 2 系統あり、重みの命名が
  割れている。upstream vLLM は fused 実装なので `self_attn.index_k_proj` 命名の
  nvidia 版が正しい。
- **NGC (`nvcr.io/nvidia/vllm`) は使えない。** 最新の 26.07-py3 でも中身が 2026-06-17 で、
  M3 の PP 対応 (2026-06-24) に届かない。26.08 が出たら乗り換え候補。
- **投機デコードは使えない。** MTP の重みが nvidia 版に無く、EAGLE3 は draft に
  target の PP がそのままコピーされるため 1 層の draft に PP=3 が課されて落ちる。

## ノードの役割

3 台とも同じ repo と `.env` を置き、compose の `--profile` だけで役割を切り替える。
`.env` の `NODE0_MGMT_IP` に指定した機体が head (rank0) になる。

| | head | worker1 | worker2 |
|---|---|---|---|
| rank | 0 (Ray head / API `:8910`) | 1 | 2 |
| profile | `head` | `worker1` | `worker2` |
| コンテナ | `m3-head` | `m3-worker1` | `m3-worker2` |
| `.env` の IP | `NODE0_MGMT_IP` | `NODE1_MGMT_IP` | `NODE2_MGMT_IP` |

以降のコマンドは worker 2 台の管理 IP を `$N1` / `$N2` で参照する。
**必ず管理 NIC 側 (10GbE) の IP を使うこと。** QSFP 側のアドレスでログインしたまま
MTU やアドレスを変えると自分の足を撃つ。

```bash
N1=<worker1 の管理 IP>    # .env の NODE1_MGMT_IP と同じ
N2=<worker2 の管理 IP>    # .env の NODE2_MGMT_IP と同じ
```

---

## Docker は rootful を使う

`--device /dev/infiniband` / `memlock unlimited` / host network での RDMA が必要なので
**rootless では動かない**。この repo の docker コマンドは全部 `sudo` を付ける。

`sudo docker compose` はカレントディレクトリの `.env` を読む。sudo で `HOME=/root` に
なるため **`.env` の中でシェル変数は使えない**。`MODEL_PATH` は repo 相対か絶対パスで。

### nvidia ランタイムの登録 (最初に 1 回 / 3 台とも)

rootless 側に登録してあっても rootful 側 (`/etc/docker/daemon.json`) には効かない。
未登録だと `RuntimeError: Failed to infer device type` で即死する。

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
ssh -t "$N1" 'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker'
ssh -t "$N2" 'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker'

sudo docker info | grep -i runtimes    # nvidia が出るまで先に進まない
```

compose が `/dev/nvidia*` を明示的に渡しているのは、`no-cgroups=true`
(rootless で GPU を使うのに必須) だと rootful で device cgroup に弾かれるため。
それでもダメなら `x-vllm-service` に `privileged: true` を足す。

---

## セットアップ

### 0. repo と .env を 3 台に配る

```bash
cp .env.example .env
$EDITOR .env      # NODE{0,1,2}_MGMT_IP / NCCL_IB_ADDR_RANGE / MODEL_PATH を実機に合わせる

for H in "$N1" "$N2"; do
  rsync -av --exclude .git --exclude models --exclude vllm-cache ./ "$H:~/repos/llm-container/"
done
```

3 台とも `.env` の中身は同じでよい。役割は `--profile` で切り替える。

### 1. ケーブル (3 本 / 三角メッシュ)

各ノードが両ポートを使い、ケーブル 3 本で三角形を作る。スイッチは要らない。
Spark は 1 ポートにつき 2 本の "twin" インターフェースを見せるので、
`ip -br addr show` では 4 本が up になっているのが正常。

**各ノードで f0 が向いている隣人が違う**ため、NCCL の既定の「同 index の NIC 同士を
ペア」では未接続のサブネットにダイヤルする。`NCCL_CROSS_NIC=1` が必須なのはこのため。

配線しなおす場合は `/etc/netplan/40-cx7.yaml` を編集して `sudo netplan try`
(`apply` ではなく `try`。疎通が切れても 120 秒で戻る)。

### 2. MTU を 9000 に上げる (4 本すべて / 3 台とも)

1500 のままだと RoCE の path MTU が 1024 に落ちる。9000 にすると 4096 まで上がる。
**電源を落とすたびに 1500 に戻る**ので再起動後は確認すること。

```bash
for I in enp1s0f0np0 enP2p1s0f0np0 enp1s0f1np1 enP2p1s0f1np1; do
  sudo ip link set dev "$I" mtu 9000
done
ibv_devinfo -d rocep1s0f0 | grep active_mtu    # 4096 (5) になること
```

### 3. 重みを取得 (3 台とも / 各 250GB, 88 shard)

```bash
./scripts/fetch-model.sh
ssh "$N1" 'cd ~/repos/llm-container && ./scripts/fetch-model.sh'
ssh "$N2" 'cd ~/repos/llm-container && ./scripts/fetch-model.sh'
```

### 4. イメージをビルド (3 台とも) — rootful

[`Dockerfile`](Dockerfile) が公式 nightly に ray と NCCL 2.30.7 を足す。
NCCL 2.30.7 は 3 ノードメッシュに要る subnet-aware routing の対応版で、
2.30.7 未満ならビルド時に止まる。

```bash
sudo docker compose --profile head build

IMAGE=$(sed -n 's/^VLLM_IMAGE=//p' .env)
sudo docker save "$IMAGE" | ssh "$N1" 'sudo docker load'
sudo docker save "$IMAGE" | ssh "$N2" 'sudo docker load'
```

ビルドログ末尾に `BAKED_NCCL_VERSION 23007 (2.30.7)` が出れば OK。

> **起動前に 3 台とも用意しておくこと。** 1 台が build 中に rendezvous が始まると固まる。

### 5. メモリを空けて preflight

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
./scripts/preflight.sh                                         # 3 台とも
```

NG が出た状態で起動しても、数分待たされた挙句 NCCL エラーか OOM-kill で死ぬだけ。

---

## 起動

**worker を先に、head を最後に。**

```bash
ssh -t "$N2" 'cd ~/repos/llm-container && sudo docker compose --profile worker2 up -d'
ssh -t "$N1" 'cd ~/repos/llm-container && sudo docker compose --profile worker1 up -d'
sudo docker compose --profile head up -d
sudo docker logs -f m3-head
```

重みロードに 8〜10 分、初期化を含めて 12〜15 分ほど。
`Application startup complete` が出れば完了。

コンテキスト長を変えるときは [`presets/`](presets/) を重ねる (3 台とも同じ組み合わせで)。

```bash
sudo docker compose --env-file .env --env-file presets/128k.env --profile head up -d
```

### 確認

```bash
curl -s http://127.0.0.1:8910/health
curl -s http://127.0.0.1:8910/v1/models | python3 -m json.tool

# KV が足りているか。1.0 を下回っていたら設定を見直す
curl -s http://127.0.0.1:8910/metrics | grep 'cache_config_info{' | tr ',' '\n' \
  | grep -E 'kv_cache_max_concurrency|kv_cache_size_tokens'
```

> **コールドスタート後の 1 発目は TTFT が 40 秒ほどかかる。** Triton の JIT と
> autotune が走るためで異常ではない。2 回目以降は 1 秒台。

### 停止

```bash
sudo docker compose --profile head down
ssh -t "$N1" 'cd ~/repos/llm-container && sudo docker compose --profile worker1 down'
ssh -t "$N2" 'cd ~/repos/llm-container && sudo docker compose --profile worker2 down'
```

UVM は完全には解放されない。構成を変えるときは 3 台とも再起動してから。

---

## 使う

`network_mode: host` なので head に届くアドレスならどれでも叩ける。認証はしていないが、
多くのクライアントが空でないキーを要求するので適当な文字列を渡す。

```python
from openai import OpenAI
client = OpenAI(base_url="http://<head>:8910/v1", api_key="local")
r = client.chat.completions.create(
    model="minimax-m3",
    messages=[{"role": "user", "content": "..."}],
    temperature=1.0, top_p=0.95, top_k=40,   # MiniMax 推奨値
)
r.choices[0].message.reasoning               # thinking はここ
```

**thinking がトークン予算を食う。** `max_tokens` は成果物の数倍を見ること。
実測で 4,096 トークンすべてを thinking に使い切って本文 0 文字になった例がある。

### Claude Code から使う

vLLM が Anthropic Messages API も生やすので、変換プロキシなしで直結できる。
`~/.claude/settings.json` の `env` を向ける
([`scripts/local-llm-on.sh`](scripts/local-llm-on.sh) が差し替える)。

```jsonc
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8910",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_DEFAULT_OPUS_MODEL":   "minimax-m3",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "minimax-m3",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL":  "minimax-m3"
  }
}
```

> **ストリーミングでは thinking が分離されない。** 非ストリームなら
> `reasoning` / `thinking` ブロックに正しく分かれるが、ストリーミングでは
> リーズニングパーサが効かず `<mm:think>...</mm:think>` が本文に混ざる。
> Claude Code は既定でストリーミングするので表示が汚れる。動作自体はする。

---

## コンテキスト長とメモリ

`GPU_MEMORY_UTILIZATION` は「予算 = util × MemTotal(119.63GiB)」を決めるだけで、
増えるのは KV プールのみ。**decode 速度には効かない**(帯域律速のため)。
上がるのは載せられるコンテキスト長・同時実行数・prefix cache の保持量。

head は API サーバと Ray head を抱えるぶん free が 104.66GiB しかなく、
**0.875 が絶対の天井**。GB10 は UMA なので超過すると CUDA OOM では済まず、
`exit 137` でノードごと落ちる。

KV サイズを固定したいなら `VLLM_KV_ARGS` に `--kv-cache-memory <bytes>` を足す
(指定すると `GPU_MEMORY_UTILIZATION` は無視される)。util 由来だと起動時の
空きメモリでブレる。

## 速度

decode は 1 token ごとに active 23B 分の重み (NVFP4 で約 11.5GB) を読む帯域律速。
GB10 の 273GB/s から**理論上限は約 24 tok/s**、実測 10 tok/s はその 4 割強。
PP のノード間通信は 1 token あたり hidden state 12KB × 2 hop しか流れないので、
**decode ではネットワークは律速ではない**。

投機デコードが使えない以上、残っている手は限られる。

- `MAX_NUM_BATCHED_TOKENS` を上げる (適用済み / 上流報告で +1.5 tok/s)
- sm_121 ネイティブビルド ([eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker))。
  現イメージは sm_120 SASS 止まり。効果は未確認
- REAP で枝刈りした重み。active が減るぶん速いが品質とのトレードオフ

---

## ハマりどころ

| 症状 | 原因と対処 |
|---|---|
| `No common block size for 128` | GB10 特有。MSA は block size 128 しか受けないが、full attention 側で既定選択される FlashInfer は 128 以上を sm_100 系でしか advertise しない。`--attention-backend TRITON_ATTN` で回避 (既定で入れてある) |
| `Shard id for QKVParallelLinear ... got shard id index_k` | 重みとイメージの組み合わせ違い。upstream vLLM には `nvidia/MiniMax-M3-NVFP4` を使う。並列度をいじっても直らない |
| `Pipeline parallelism is not supported for this model` | イメージの vLLM が古く M3 の PP 対応を含んでいない。NGC 26.07 が該当。投機デコードを有効にしたときも draft 側で同じエラーが出る |
| `ibv_modify_qp err 110` / rendezvous で固まる | `.env` に `NCCL_IB_GID_INDEX` が残っている。3 台メッシュでは消すこと (preflight が検出) |
| NCCL が 2.30.7 未満 | subnet-aware routing が無くメッシュでは必ず失敗する。起動ログの `[entrypoint] NCCL version:` を確認 |
| Ray が rank0 を kill する (実 OOM ではない) | `RAY_memory_monitor_refresh_ms=0` が効いているか確認 |
| 重みロード中に head だけ OOM | Ray の object store が既定で約 36GB/node を予約する。`RAY_OBJECT_STORE_MEMORY` を確認 |
| `Failed to infer device type` / NVML 初期化失敗 | rootful daemon に nvidia ランタイムが未登録。または `no-cgroups=true` で device cgroup に弾かれている |
| NCCL が `unhandled system error` / `ibv_reg_mr` 失敗 | RDMA がコンテナに通っていない。まず rootful で起動しているか疑う |
| decode が極端に遅い (5 tok/s 未満) | RoCE に乗らず 1GbE を通っている。あるいは帯域が 12.8Gb/s で張り付いている。後者は **3 台とも電源ブリックを 60〜90 秒抜く**コールド電源断で戻る (ウォームリブートでは直らない) |
| worker が exit 137 | UMA の OOM-kill。`MAX_NUM_BATCHED_TOKENS` を下げる |
| `<tool_call>` や `<mm:think>` が本文に漏れる | 非ストリームなら `VLLM_PARSER_ARGS` を確認。ストリーミングでは既知の未対応 |
| 出力が文字化け・意味不明 | イメージ変更後にコンパイルキャッシュが残っている。`sudo rm -rf vllm-cache/*/*` して再起動 |

```bash
sudo docker logs -f m3-head
ssh -t "$N1" 'sudo docker logs -f m3-worker1'
```

---

## 出典

- [nvidia/MiniMax-M3-NVFP4](https://huggingface.co/nvidia/MiniMax-M3-NVFP4) — 重み
- [vLLM recipes: MiniMax-M3](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3) — 公式の serve 設定
- [NVIDIA/dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) — DGX Spark 公式 playbook
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) —
  3 ノードメッシュの配線・NCCL 変数、sm_121 ネイティブビルド
- [tonyd2wild/Minimax-M3-NVFP-3x-DGX-Sparks-TP-3](https://github.com/tonyd2wild/Minimax-M3-NVFP-3x-DGX-Sparks-TP-3) —
  Ray の OOM 修正、NCCL の subnet-aware routing、コールド電源断
