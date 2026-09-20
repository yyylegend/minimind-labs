# MiniMind Labs

一个用于学习和复现小型 Decoder-only Transformer 的独立实验项目。

本项目明确参考 [MiniMind](https://github.com/jingyaogong/minimind) 的模型结构、配置方式、训练流程和权重格式。它是面向学习的独立实现，不是 MiniMind 官方仓库或官方发布版本；代码、实验和训练输出均放在本仓库中维护，原始 MiniMind 仓库仅作为只读对照。

项目的目标是把一个语言模型拆成可以单独理解、测试和组装的模块，并最终跑通：

```text
RMSNorm
  ↓
RoPE
  ↓
Attention / GQA / KV Cache
  ↓
SwiGLU / FFN
  ↓
TransformerBlock
  ↓
Causal Language Model
  ↓
Pretrain → SFT → 评测 / 部署
```

## 项目特点

- 以 PyTorch 实现 Transformer 的核心组件
- 保留手写 Attention，便于理解 Q/K/V、RoPE、GQA 和 KV Cache
- 支持 PyTorch 融合 Attention 路径，用于实际训练
- 支持 next-token prediction、预训练和监督微调
- 支持断点恢复、训练进度、速度和预计剩余时间
- 支持 TensorBoard 观察 loss、学习率、吞吐量和显存
- 输出 MiniMind 风格的纯模型权重，便于后续评测或部署

## 项目结构

```text
minimind-labs/
├── README.md
├── .gitignore
├── requirements-training.txt
├── data/                         # 本地数据集，不提交到 Git
├── dataset/
│   ├── __init__.py
│   └── lm_dataset.py             # JSONL 读取、tokenize、labels 构造
├── minimind_lab/
│   ├── __init__.py
│   ├── config.py                  # 模型配置
│   ├── rmsnorm.py                 # RMSNorm
│   ├── rope.py                    # Rotary Position Embedding
│   ├── attention.py               # MHA / GQA / KV Cache
│   ├── swiglu.py                  # SwiGLU 和 FFN
│   ├── transformer_block.py       # Attention + FFN + 残差
│   └── causal_lm.py               # Embedding、Block、LM Head、生成
├── trainer/
│   ├── __init__.py
│   ├── trainer_utils.py           # 设备、精度、checkpoint、训练循环
│   ├── train_pretrain.py          # 预训练入口
│   └── train_sft.py               # SFT 入口
├── scripts/
│   ├── demo_*.py                  # 单模块演示
│   └── export_official_weights.py # 旧 checkpoint 导出工具
└── tests/                         # 核心模块的最小测试
```

## 环境准备

建议使用 Python 3.10+。先根据自己的硬件安装匹配的 PyTorch，再安装训练依赖：

```powershell
pip install -r requirements-training.txt
```

也可以使用 `uv` 管理独立环境：

```powershell
uv venv .venv-train --python 3.11
uv pip install --python .\.venv-train\Scripts\python.exe torch
uv pip install --python .\.venv-train\Scripts\python.exe -r requirements-training.txt
```

GPU 环境请按照 [PyTorch 安装页面](https://pytorch.org/get-started/locally/) 选择对应的 CUDA wheel。Windows 单卡训练时，`float16` 通常比 `bfloat16` 更适合较老的 NVIDIA GPU；CPU 训练请使用 `float32`。

检查环境：

```powershell
$trainPy = (Resolve-Path .\.venv-train\Scripts\python.exe).Path
& $trainPy -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 运行模块演示和测试

在仓库根目录执行：

```powershell
$trainPy = (Resolve-Path .\.venv-train\Scripts\python.exe).Path

& $trainPy -m unittest discover -s tests -v
& $trainPy -m scripts.demo_rmsnorm
& $trainPy -m scripts.demo_rope
& $trainPy -m scripts.demo_attention
& $trainPy -m scripts.demo_swiglu
& $trainPy -m scripts.demo_transformer_block
& $trainPy -m scripts.demo_causal_lm
```

## 数据和参考 Tokenizer

训练入口需要一个 MiniMind 风格的 tokenizer，以及对应格式的 JSONL 数据。参考 MiniMind 的 tokenizer 通常位于其仓库的 `model/` 目录。

```text
<minimind-repo>/model
```

下载的数据放在本项目的 `data/` 目录，例如：

```text
data/
├── pretrain_t2t_mini.jsonl
└── sft_t2t_mini.jsonl
```

数据集和训练产物已加入 `.gitignore`，不会随代码提交。也可以把 `--tokenizer_path` 改成自己准备的 tokenizer 目录，把 `--data_path` 改成自己的 JSONL 文件。

预训练数据使用类似下面的格式：

```json
{"text": "机器学习模型通过数据学习规律。"}
```

SFT 数据由数据集读取器按照对话格式构造输入，并只对需要学习的回答部分计算 loss。

## 训练流程

训练分为两个阶段：

```text
预训练：学习语言和代码的基本分布
    ↓
SFT：学习问答格式、指令遵循和任务行为
    ↓
固定评测集：比较训练前后的能力变化
```

### 参数选择：通用单卡基线

本项目默认使用 MiniMind 的 `pretrain_t2t_mini.jsonl` 和 `sft_t2t_mini.jsonl`。MiniMind 官方对这两份 mini 数据都建议将 `max_seq_len` 设置在约 `768` tokens，并提醒过短会截断语义、过长会增加 padding 和计算浪费。[官方数据说明](https://github.com/jingyaogong/minimind/blob/master/README.md?plain=1#L2879-L2903)

因此，正式训练可以先从下面这组与硬件无关的单卡基线开始：

```text
max_seq_len=768
batch_size=8
accumulation_steps=1
```

这里的有效 token batch 可以粗略理解为：

```text
batch_size × max_seq_len × accumulation_steps
8 × 768 × 1 = 6144 tokens/update
```

如果显存不足，优先保持 `max_seq_len=768`，只缩小单次 batch，并用梯度累积补回来：

```text
显存较紧：batch_size=4，accumulation_steps=2
显存更紧：batch_size=2，accumulation_steps=4
```

这几组配置的有效 token batch 接近，便于比较训练结果。实际可用值仍取决于模型大小、精度、Attention 实现和 GPU 显存；不要把官方训练脚本的默认 `batch_size` 直接当成所有机器的要求。

如果使用自己的数据集，应先观察样本长度分布：序列太短会丢失长样本的上下文，序列太长则会让大量短样本 padding。调整参数时一次只改变一个主要因素，并同时记录 `tok/s`、loss 和固定评测集结果。

### 1. 先做 smoke test

先用小模型和两个 step 验证 tokenizer、数据集、前向、反向和 checkpoint 链路：

```powershell
$trainPy = (Resolve-Path .\.venv-train\Scripts\python.exe).Path

& $trainPy -m trainer.train_pretrain `
  --data_path .\data\pretrain_t2t_mini.jsonl `
  --tokenizer_path ..\minimind\model `
  --output_dir .\out\smoke_pretrain `
  --tensorboard_dir .\out\runs\smoke_pretrain `
  --device cuda:0 `
  --dtype float16 `
  --hidden_size 128 `
  --num_hidden_layers 2 `
  --max_seq_len 128 `
  --batch_size 1 `
  --accumulation_steps 1 `
  --max_steps 2
```

CPU 环境请将 `--device cpu` 和 `--dtype float32` 一起使用。

### 2. 单卡预训练

下面是一组适合作为起点的通用单卡配置。它与上面的基线一致；显存不足时按上面的回退规则调整。

```powershell
& $trainPy -m trainer.train_pretrain `
  --data_path .\data\pretrain_t2t_mini.jsonl `
  --tokenizer_path ..\minimind\model `
  --output_dir .\out\pretrain `
  --tensorboard_dir .\out\runs\pretrain `
  --device cuda:0 `
  --dtype float16 `
  --hidden_size 512 `
  --num_hidden_layers 8 `
  --num_attention_heads 8 `
  --num_key_value_heads 4 `
  --max_seq_len 768 `
  --batch_size 8 `
  --accumulation_steps 1 `
  --epochs 1 `
  --learning_rate 5e-4 `
  --grad_clip 1.0 `
  --save_interval 100 `
  --max_steps 0 `
  --log_interval 10 `
  --num_workers 0
```

`--max_steps 0` 表示按 `--epochs` 完整遍历数据；设置为正数则用于限定实验步数。

### 3. 单卡 SFT

预训练完成后，使用预训练阶段的可恢复 checkpoint 初始化 SFT：

```powershell
& $trainPy -m trainer.train_sft `
  --data_path .\data\sft_t2t_mini.jsonl `
  --tokenizer_path ..\minimind\model `
  --init_checkpoint .\out\pretrain\pretrain_last.pt `
  --output_dir .\out\sft `
  --tensorboard_dir .\out\runs\sft `
  --device cuda:0 `
  --dtype float16 `
  --hidden_size 512 `
  --num_hidden_layers 8 `
  --num_attention_heads 8 `
  --num_key_value_heads 4 `
  --max_seq_len 768 `
  --batch_size 8 `
  --accumulation_steps 1 `
  --epochs 1 `
  --learning_rate 5e-5 `
  --grad_clip 1.0 `
  --save_interval 100 `
  --max_steps 0 `
  --log_interval 10 `
  --num_workers 0
```

模型结构参数需要和预训练阶段保持一致。预训练和 SFT 都可以先使用 `max_seq_len=768`；如果 SFT 样本明显更长，再单独提高 SFT 的序列长度，并重新测量吞吐量和显存。实际训练时还应根据数据规模和评测结果调整学习率与训练轮数。

## 暂停、恢复和监控

训练循环会周期性保存 checkpoint。运行中按 `Ctrl+C` 会先保存当前安全状态；也可以使用 `--save_interval` 定期保存。

恢复训练时尽量保持模型结构、`max_seq_len`、`batch_size` 和 `accumulation_steps` 不变。如果要比较另一组训练参数，建议使用新的 `output_dir`，避免把不同实验混在同一个可恢复 checkpoint 中。

```powershell
& $trainPy -m trainer.train_pretrain `
  --resume_checkpoint .\out\pretrain\pretrain_last.pt `
  --data_path .\data\pretrain_t2t_mini.jsonl `
  --tokenizer_path ..\minimind\model `
  --output_dir .\out\pretrain `
  --tensorboard_dir .\out\runs\pretrain `
  --device cuda:0 `
  --dtype float16 `
  --hidden_size 512 `
  --num_hidden_layers 8 `
  --num_attention_heads 8 `
  --num_key_value_heads 4 `
  --max_seq_len 768 `
  --batch_size 8 `
  --accumulation_steps 1 `
  --num_workers 0
```

启动 TensorBoard：

```powershell
& $trainPy -m tensorboard.main --logdir .\out\runs --port 6006
```

然后打开 <http://localhost:6006>。

## 输出文件

每个训练阶段会产生两类权重：

```text
*_last.pt
├── 用于恢复训练
├── 包含模型、优化器、Scaler 和训练进度
└── 适合继续跑实验

pretrain_<hidden_size>.pth / full_sft_<hidden_size>.pth
├── 纯模型权重
├── MiniMind 风格的 state_dict
└── 可用于后续评测、加载或部署
```

旧版本只有 `last.pt` 时，可以导出纯模型权重：

```powershell
& $trainPy -m scripts.export_official_weights `
  --checkpoint .\out\pretrain\pretrain_last.pt `
  --output .\out\pretrain\pretrain_512.pth
```

只有模型结构、词表和 tokenizer 配置一致时，权重才可以直接交给 MiniMind 的评测或部署脚本使用。

## 后续方向

基础训练链路稳定后，再逐步加入：

- LoRA / QLoRA
- DPO 或 GRPO
- 代码和数学专项数据、固定评测集
- MoE
- DDP 和更高效的数据管线
- 多模态输入与图像生成方向

这些内容属于后续实验，不作为当前最小闭环的前置依赖。

## 参考项目

- [MiniMind](https://github.com/jingyaogong/minimind)：模型结构、训练流程和官方权重格式的主要参考
- [MiniMind Notes](https://github.com/joyehuang/minimind-notes)：学习过程中的补充讲解与实验参考

本项目是学习用途的独立实现，具体功能以当前代码和命令为准。
