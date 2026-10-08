# dealignai/GLM-5.3-Flash-UNCENSORED-FP8 を 4 台の DGX Spark で動かす

対象: `https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-FP8`
(native block-FP8 128x128 / 62 shard / 約 328GB = 306GiB / MTP head 内蔵)

ハード: DGX Spark (GB10, SM121) x4、**光スイッチ経由のフルメッシュ (同一 L2)**。
RING ではない。rootful Docker 必須 (`sudo docker compose ...`)。

| rank | profile | rail0 (enp1s0f0np0) | rail1 (enP2p1s0f0np0) |
| --- | --- | --- | --- |
| 0 | `head` | 192.168.0.1 | 192.168.1.1 |
| 1 | `worker1` | 192.168.0.2 | 192.168.1.2 |
| 2 | `worker2` | 192.168.0.3 | 192.168.1.3 |
| 3 | `worker3` (増設機) | 192.168.0.4 | 192.168.1.4 |

IP は例。実機の割り当てに合わせて `.env` を書き換えること。
以下の ssh / curl もこの rail0 の IP を使う (管理 LAN の IP でもよい)。

---

## 1. 結論: TP=4 (モデルカードの推奨構成と同じ)

glm5_next の形状は 4 ですべて割り切れる:

| 次元 | 値 | / 4 |
| --- | --- | --- |
| num_attention_heads | 64 | 16 |
| KDA num_heads | 64 | 16 |
| vocab_size | 154880 | 38720 |
| moe_intermediate_size | 2048 | 512 (FP8 block 128 の倍数) |
| intermediate_size | 12288 | 3072 |

3 台構成で詰んだ理由 (記録):

- **TP=3**: 上の次元が 3 で割れず、`attention.py:452` / `kda.py:219` の assert で落ちる。
- **PP**: glm5_next は `make_empty_intermediate_tensors` 未実装で、vllm 側がゲートしている (`model.py:788` / `:1169`)。
- **DP3/EP3**: 重みが 111GiB/rank になり、起動時の空き 109.3GiB に載らなかった (2026-10-06)。

## 2. メモリ

shard ヘッダの実測: 合計 305.8GiB、うち vision 1.05GiB (`--language-model-only` で載せない)。

| 項目 | 1 rank あたり |
| --- | --- |
| 重み (304.7 / 4) | ~76.2 GiB |
| CUDA ctx / NCCL / プロセス | ~7 GiB |
| KV / アクティベーション (GMU 0.88 = 105.3GiB の残り) | ~20 GiB |

`GPU_MEMORY_UTILIZATION` は「起動時の空き / 119.63GiB」を超えると即落ちる。
GB10 では OS 等で常時 ~10GiB 使われているので、上限は実質 0.91。`./scripts/preflight.sh` が起動前に確認する。

---

## 3. 4 台目 (worker3) のセットアップ

増設直後に head から見えた状態 (例):

- f0 系 2 ポートは光スイッチに結線済み。RDMA は 2 本とも `PORT_ACTIVE`。
- **enp1s0f0np0**: `169.254.x.x/16` (link-local) / **MTU 9000** ← 直す
- **enP2p1s0f0np0**: オフィス LAN の DHCP からアドレスを取っている ← 直す
  (QSFP がオフィス LAN と同じ L2 にいるため)
- arp_ignore=1 / arp_announce=2 は設定済み。
- kernel / driver が既存機とずれていることがある (3.2)。

### 3.1 QSFP に静的 IP と MTU 1500 を入れる (worker3 で)

まず今の設定を確認する (既存の netplan と NetworkManager のプロファイルが混在している):

```bash
ssh -t <worker3 の管理 IP>   # QSFP にまだ IP が無いので管理 LAN 側から入る
sudo cat /etc/netplan/40-cx7.yaml
nmcli -f NAME,UUID,DEVICE,TYPE con show
```

`/etc/netplan/40-cx7.yaml` を以下に置き換える (既存機と同じ方針。ゲートウェイは書かない):

```bash
sudo cp /etc/netplan/40-cx7.yaml /etc/netplan/40-cx7.yaml.bak.$(date +%F)
sudo tee /etc/netplan/40-cx7.yaml >/dev/null <<'EOF'
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: false
      dhcp6: false
      link-local: [ipv6]
      mtu: 1500
      addresses: [192.168.0.4/24]
    enP2p1s0f0np0:
      dhcp4: false
      dhcp6: false
      link-local: [ipv6]
      mtu: 1500
      addresses: [192.168.1.4/24]
EOF
sudo chmod 600 /etc/netplan/40-cx7.yaml

# enP2p1s0f0np0 で DHCP しているのは NM の「有線接続 1」。消さないと 207.140 が残り続ける。
sudo nmcli con delete "有線接続 1"
sudo netplan apply
```

確認:

```bash
ip -br addr | grep -E 'enp1s0f0np0|enP2p1s0f0np0'   # 4.4/24 と 5.4/24 だけ
cat /sys/class/net/{enp1s0f0np0,enP2p1s0f0np0}/mtu  # 1500 / 1500
show_gids | grep -E 'v2.*192\.168\.(0|1)\.4'           # RoCEv2 の GID が出ること
for ip in 192.168.0.1 192.168.0.2 192.168.0.3; do ping -M do -s 1472 -c2 -W2 $ip; done
for ip in 192.168.1.1 192.168.1.2 192.168.1.3; do ping -M do -s 1472 -c2 -W2 $ip; done
```

### 3.2 (任意) ドライバを揃える

既存機は `6.17.0-1029 + 580.173.02`、増設機は `7.0.0-1019 + 580.178.04` のように混在しうる。
NCCL はイメージ内のものを使うので混在でも動く見込みだが、NCCL init で詰まったら
古い方を `sudo apt update && sudo apt full-upgrade` → 再起動で揃える。

---

## 4. 他の Spark への展開

head でこのブランチを push し、残り 3 台で同じブランチに切り替える。
`.env` は git 管理外なので head のものを配る (4 台とも同じ内容)。

```bash
# head で
git push -u origin DGX-Spark-4/dealignai/GLM-5.3-Flash-UNCENSORED-FP8

for h in 192.168.0.2 192.168.0.3 192.168.0.4; do
  ssh $h 'cd ~/repos/llm-container && git fetch origin \
    && git checkout DGX-Spark-4/dealignai/GLM-5.3-Flash-UNCENSORED-FP8 \
    && git pull --ff-only'
  scp .env $h:~/repos/llm-container/.env
done
```

### 4.1 重み (worker3 だけ未取得の場合)

既存の 3 台に 62 shard (306G) があれば不要。増設機には head から QSFP 経由で rsync するのが一番速い
(3.1 の IP 設定のあとで。HF から落とすなら増設機で `./scripts/fetch-model.sh`):

```bash
# head で
rsync -a --info=progress2 ./models/GLM-5.3-Flash-UNCENSORED-FP8/ \
  192.168.0.4:repos/llm-container/models/GLM-5.3-Flash-UNCENSORED-FP8/
```

### 4.2 イメージ (worker3)

```bash
ssh -t 192.168.0.4 'sudo docker pull ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:d77d375c742fc54f436dec5108b440f58f021bc6600052bf0e8fe5840357e78f'
```

### 4.3 4 台とも preflight

```bash
./scripts/preflight.sh                                 # head
for h in 192.168.0.2 192.168.0.3 192.168.0.4; do
  ssh $h 'cd ~/repos/llm-container && ./scripts/preflight.sh'
done
```

`== RoCE リンク ==` で 3 peer × 2 rail に DF ping が通ること、`== モデル ==` が 62 shard、
`== メモリ ==` が OK であること。

---

## 5. 起動 / 停止

```bash
# worker を先に (head が rendezvous の master。worker は head を待つ)
ssh -t 192.168.0.4 'cd ~/repos/llm-container && sudo docker compose --profile worker3 up -d'
ssh -t 192.168.0.3 'cd ~/repos/llm-container && sudo docker compose --profile worker2 up -d'
ssh -t 192.168.0.2 'cd ~/repos/llm-container && sudo docker compose --profile worker1 up -d'
sudo docker compose --profile head up -d

# ログ (重み 76GiB/rank のロード + JIT で 10〜30 分)
sudo docker logs -f glm53-fp8-head
ssh -t 192.168.0.4 'sudo docker logs -f glm53-fp8-worker3'

# 疎通
curl -s http://192.168.0.1:8910/v1/models
curl -s http://192.168.0.1:8910/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash-uncensored","messages":[{"role":"user","content":"1+1="}],"max_tokens":32}'

# 停止 (各ノードで自分の profile を down)
sudo docker compose --profile head down
ssh -t 192.168.0.2 'cd ~/repos/llm-container && sudo docker compose --profile worker1 down'
ssh -t 192.168.0.3 'cd ~/repos/llm-container && sudo docker compose --profile worker2 down'
ssh -t 192.168.0.4 'cd ~/repos/llm-container && sudo docker compose --profile worker3 down'
```

---

## 6. トラブルシューティング

| 症状 | 見どころ |
| --- | --- |
| `Free memory ... is less than desired GPU memory utilization` | 起動時の空き不足。`GPU_MEMORY_UTILIZATION` を (空き / 119.63) 未満に下げる。4 台で一番空きの少ないノードに合わせる |
| worker が rendezvous で止まる | head で `ss -ltnp \| grep 29501`。`HEAD_ROCE_IP` が head の rail0 IP か |
| `vLLM is using nccl==...` の直後に無言ハング | MTU。4 台 × 2 rail とも 1500 か (増設機は初期状態で 9000 のことがある)。preflight の DF ping |
| NCCL `unhandled system error` | `NCCL_DEBUG=INFO`。arp_ignore/arp_announce、`show_gids`、ドライバの混在 (3.2) |
| MoE backend の選択で落ちる / 出力が壊れる | `VLLM_EXTRA_ARGS` に `--moe-backend marlin` か `triton` を足す。起動ログの `Using ... MoE backend` を確認 |
| `num_heads ... not divisible` | TP_SIZE が 4 以外。entrypoint が起動前に弾くはず |

## 7. まだ確認できていないこと

1. **SM121 で block-FP8 の MoE / linear カーネルが選べるか。** イメージ (v8) の実績は NVFP4。
   DeepGEMM / CUTLASS の block-FP8 MoE は sm90/sm100 向けなので、おそらく Triton fused_moe になる。
   起動ログの `Using ... MoE backend` と、最初の応答が文字化け / NaN でないかを必ず見る。
2. **TP=4 の all-reduce が MTU 1500・非ロスレスの光スイッチで実用速度か。** decode が遅ければ
   スイッチ側でジャンボ (l2mtu 9216) を有効にして 4 台とも MTU 9000 に戻すのが次の一手。
