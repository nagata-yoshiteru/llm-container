# patches/roce — b12x RoCEnante one-shot RoCE all-reduce (任意 / 既定 OFF)

TP2 の all-reduce を NCCL から RDMA の one-shot 実装に差し替える。
上流の実測 (2026-09-18 / healthy fleet): **aggregate +5〜18% (C1–C6)**、
単発 decode はほぼ変化なし、リグレッションなし。

効きどころは「1 step あたり約 9ms の NCCL を削る」ところなので、
step が長い構成 (クランプした GPU / 長コンテキスト) ほど相対的な効果は小さい。

## 中身と出典

ここに入っているのは **vLLM 側のアダプタ 5 本だけ**。

| ファイル | 置き換え先 (コンテナ内) |
|---|---|
| `b12x_roce_all_reduce.py` | `vllm/distributed/device_communicators/b12x_roce_all_reduce.py` (新規) |
| `cuda_communicator.py` | `vllm/distributed/device_communicators/cuda_communicator.py` |
| `parallel_state.py` | `vllm/distributed/parallel_state.py` |
| `envs.py` | `vllm/envs.py` |
| `gpu_worker.py` | `vllm/v1/worker/gpu_worker.py` |

出典:

- アダプタ: [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
  の `speed-night-2026-09-18/roce/`。元は `local-inference-lab/vllm#597` を
  `sm121-v11-dflash2` の vLLM ツリーに移植したもの。vLLM 由来なので Apache-2.0。
- ランタイム本体 (`b12x` パッケージ / `/opt/b12x-roce`): Luke Alonso, Apache-2.0
  (`comm/roce` の `b58f34ea`)。**この repo には入っていない。**

## 前提: ランタイム本体を別イメージから抜く

`b12x` は PyPI に無いので、それを既に含むイメージから取り出す。

```bash
SRC_IMAGE=vllm-dsv41:exl3b-roce ./scripts/build-roce-bundle.sh   # 2 台とも
```

これで `patches/roce/` に `b12x/` `b12x-1.3.0.dist-info/` `b12x-roce/` が生える
(いずれも `.gitignore` 済み / repo には入らない)。

ランタイムが揃わない場合は **NCCL のままで問題ない**。`ROCE=1` でも
`patches/roce/b12x` が無ければ entrypoint が理由を print して自動で OFF にする。

## 有効化

```bash
# 2 台とも同じ --env-file の組み合わせで (worker -> head の順)
ssh -t "$WORKER" 'cd ~/repos/llm-container && sudo docker compose \
  --env-file .env --env-file presets/roce.env --profile worker up -d'
sudo docker compose --env-file .env --env-file presets/roce.env --profile head up -d
```

`B12X_ROCE_HCA` / `B12X_ROCE_GID_INDEX` は compose が `IB_HCA_NAME` /
`NCCL_IB_GID_INDEX` (worker は `WORKER_*`) から渡すので、リング配線でも
ケーブルが刺さっている HCA を向く。

起動後に **両 rank のログで `RoCEnante all-reduce is live` を確認する**。
出ていなければ NCCL にフォールバックしている。

## 安全装置

vLLM のファイルを 5 本まるごと差し替えるので、**イメージの vLLM ツリーが違うと
黙って壊れる** (エラーを出さずに数値が狂う経路がある)。

entrypoint は当てる前に vLLM のバージョンを照合し、
`0.1.dev20051+g487ecf187` (= `sm121-v11-dflash2` / digest `4def0ef6…`) と
一致しなければ **起動を拒否する**。

イメージを上げたときは:

1. 上流の後継版アダプタに差し替える、または
2. `ROCE=0` に戻す (`presets/roce.env` を外すだけ)

`ROCE_EXPECT_VLLM` を書き換えて照合を通すのは、アダプタが新しいツリーで
動くと確認できてからにすること。
