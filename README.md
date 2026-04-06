# My_MoE

PyTorchで実装したMixture of Experts (MoE) の学習・推論サンプルです。
`conda`仮想環境を利用し、CUDA対応GPU上で実行します。

## 1. 環境構築（conda）

```bash
conda env create -f environment.yml
conda activate moe-gpu

# CUDA 12.8 nightly (sm_120対応) のPyTorchをインストール
pip install --pre --upgrade torch torchvision torchaudio \
	--index-url https://download.pytorch.org/whl/nightly/cu128
```

## 2. 学習 + 推論実行（GPU）

```bash
python moe_train_infer.py \
	--epochs 10 \
	--batch-size 256 \
	--num-experts 4 \
	--top-k 2 \
	--checkpoint-path checkpoints/moe.pt
```

または `conda activate` を使わずに実行する場合:

```bash
conda run -n moe-gpu python moe_train_infer.py \
	--epochs 10 --batch-size 256 --num-experts 4 --top-k 2 \
	--checkpoint-path checkpoints/moe.pt
```

上記実行で以下が行われます。
- GPU上でMoEを学習
- 学習済み重みを保存
- 重みを再読込して推論

## 3. 学習済み重みで推論のみ実行

```bash
python moe_train_infer.py --inference-only --checkpoint-path checkpoints/moe.pt
```

実行時にCUDA GPUが見つからない場合はエラー終了します。

## 4. 期待される出力例

```text
using device: cuda NVIDIA ...
epoch=01 loss=... acc=...
...
epoch=10 loss=... acc=...
inference predictions: [...]
gate probs for first sample: [...]
checkpoint saved: checkpoints/moe.pt
checkpoint loaded: checkpoints/moe.pt
```
