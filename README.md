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
Pretrain → SFT → DPO（可选） → 固定评测 / 部署
```

## 项目特点

- 以 PyTorch 实现 Transformer 的核心组件
- 保留手写 Attention，便于理解 Q/K/V、RoPE、GQA 和 KV Cache
- 支持 PyTorch 融合 Attention 路径，用于实际训练
- 支持 next-token prediction、预训练和监督微调
- 支持使用 chosen/rejected 偏好对进行 DPO 训练
- 使用 MiniMind 风格的权重初始化，控制初始 logits 的数值尺度
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
│   └── lm_dataset.py             # JSONL 读取、tokenize、训练目标构造
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
│   ├── metrics.py                # 滚动吞吐、ETA、padding 与稳定性指标
│   ├── trainer_utils.py           # 设备、精度、checkpoint、训练循环
│   ├── train_pretrain.py          # 预训练入口
│   ├── train_sft.py               # SFT 入口
│   ├── dpo_utils.py               # DPO 偏好损失
│   └── train_dpo.py               # DPO 入口
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
DPO（可选）：用 chosen/rejected 回答偏好继续对齐
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
warmup_steps=1000
min_lr_ratio=0.1
```

这里的有效 token batch 可以粗略理解为：

```text
batch_size × max_seq_len × accumulation_steps
8 × 768 × 1 = 6144 tokens/update
```

训练器会对学习率先做 warmup，再进行 cosine decay。预训练基线使用 `3e-4` 的峰值学习率，训练结束时降到峰值的 `10%`；SFT 使用更小的峰值学习率。这样可以降低 FP16 长时间训练后逐渐发散的风险。

模型的线性层和词嵌入默认使用 `initializer_range=0.02`。如果初始 loss 异常高，优先检查初始化、精度和学习率，而不是盲目增大 batch size。

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
  --output_dir .\out\archive\smoke\pretrain_smoke `
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
  --output_dir .\out\checkpoints\pretrain `
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
  --warmup_steps 1000 `
  --min_lr_ratio 0.1 `
  --epochs 1 `
  --learning_rate 3e-4 `
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
  --init_checkpoint .\out\checkpoints\pretrain\pretrain_last.pt `
  --output_dir .\out\checkpoints\sft `
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
  --warmup_steps 500 `
  --min_lr_ratio 0.1 `
  --eval_ratio 0.02 `
  --eval_interval 500 `
  --epochs 1 `
  --learning_rate 5e-5 `
  --grad_clip 1.0 `
  --save_interval 100 `
  --max_steps 0 `
  --log_interval 10 `
  --num_workers 0
```

模型结构参数需要和预训练阶段保持一致。预训练和 SFT 都可以先使用 `max_seq_len=768`；如果 SFT 样本明显更长，再单独提高 SFT 的序列长度，并重新测量吞吐量和显存。实际训练时还应根据数据规模和评测结果调整学习率与训练轮数。

SFT 默认固定划分 2% 数据作为验证集，只对 assistant target token 计算验证 loss。没有留下有效 assistant target 的截断样本会被跳过并计数，不会参与训练。验证 loss 创新低时会额外保存 `full_sft_best_<hidden_size>.pth`；`full_sft_<hidden_size>.pth` 仍表示最近一次保存的模型。

### 4. DPO 偏好对齐（可选）

DPO 用同一个问题下的较好回答（`chosen`）和较差回答（`rejected`）训练策略模型，同时冻结一份初始 SFT 模型作为 reference。训练只计算回答部分的 token 概率，不把用户问题计入偏好分数。

MiniMind 官方提供的 `dpo.jsonl` 采用 `chosen` / `rejected` 对话列表格式，数据抽样自 [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)。它适合练习通用偏好对齐，不等同于有单元测试或数学判题器的正确性奖励，因此不能预期 DPO 自动提升代码 pass@1 或数学正确率。[MiniMind 数据说明](https://github.com/jingyaogong/minimind/blob/master/README.md?plain=1)

在训练服务器的项目根目录下载数据：

```bash
python -m pip install -U huggingface_hub
hf download jingyaogong/minimind_dataset dpo.jsonl --repo-type dataset --local-dir data
```

先跑两个 step 验证数据、模型和 checkpoint 链路：

```bash
python -m trainer.train_dpo \
  --data_path data/dpo.jsonl \
  --tokenizer_path ../minimind/model \
  --init_checkpoint out/checkpoints/sft_code_math_replay10k/full_sft_best_768.pth \
  --output_dir out/smoke/dpo \
  --tensorboard_dir out/runs/dpo_smoke \
  --device cuda:0 --dtype bfloat16 \
  --max_steps 2
```

Smoke test 通过后，将输出目录改为正式目录并移除 `--max_steps 2`：

```bash
python -m trainer.train_dpo \
  --data_path data/dpo.jsonl \
  --tokenizer_path ../minimind/model \
  --init_checkpoint out/checkpoints/sft_code_math_replay10k/full_sft_best_768.pth \
  --output_dir out/checkpoints/dpo \
  --tensorboard_dir out/runs/dpo \
  --device cuda:0 --dtype bfloat16
```

默认结构参数为 `hidden_size=768`、8 层、8 个 Q heads、4 个 KV heads、`max_seq_len=768`；每批 2 组偏好对，训练 1 轮，学习率 `4e-8`，`beta=0.15`。使用 2080 Ti 时把精度改成 `--dtype float16`。程序会保存 `dpo_last.pt`（可恢复训练）和 `dpo_768.pth`（纯模型权重）；恢复时传入 `--resume_checkpoint out/checkpoints/dpo/dpo_last.pt`，并保持原来的 `--init_checkpoint` 不变。

### 原始 SFT 与专项 SFT 对照评测

`scripts/eval_sft_comparison.py` 使用同一组贪心解码设置比较两个 checkpoint：8 条固定通用问题并排展示、未参与专项训练的 OrcaMath 数值题 exact match、以及未参与专项训练的代码题测试通过率。代码和数学样本会按训练 JSONL 中的 user prompt 去重；完整结果保存为 JSON，通用回答用于人工检查。

代码通过率需要 Linux 上可用的 [Bubblewrap](https://github.com/containers/bubblewrap) 隔离环境。服务器以 root 运行时可安装：

```bash
apt-get update && apt-get install -y bubblewrap
```

```bash
python scripts/eval_sft_comparison.py \
  --base_checkpoint out/checkpoints/sft_768/full_sft_best_768.pth \
  --candidate_checkpoint out/checkpoints/sft_code_math_replay10k/full_sft_best_768.pth \
  --train_data data/sft_t2t_mini.jsonl data/sft_code_math_mix_replay10k.jsonl \
  --code_dir data/raw/code/data \
  --math_dir data/raw/math/data \
  --tokenizer_path ../minimind/model \
  --output out/evaluations/sft_code_math_comparison.json
```

代码样本在 Bubblewrap 新建的用户、进程和网络命名空间内运行，系统目录只读，仅临时工作目录可写；子进程还受 CPU 时间、内存和输出大小限制。若运行环境不允许创建隔离命名空间，可加 `--skip_code_execution`，此时仍会比较通用回答和数学准确率，但不统计代码通过率。数学 exact match 只对答案中可抽取的数值做精确比较；通用回答需人工检查。

## 暂停、恢复和监控

训练循环会周期性保存 checkpoint。运行中按 `Ctrl+C` 会先保存当前安全状态；也可以使用 `--save_interval` 定期保存。

恢复训练时尽量保持模型结构、`max_seq_len`、`batch_size`、`accumulation_steps` 和学习率调度参数不变。如果要比较另一组训练参数，建议使用新的 `output_dir`，避免把不同实验混在同一个可恢复 checkpoint 中。训练发现非有限 loss 时会主动停止，并拒绝用坏权重覆盖已有 checkpoint；FP16 的单次梯度溢出则由 GradScaler 自动跳过并降低 scale，连续多次无法恢复才会停止。

```powershell
& $trainPy -m trainer.train_pretrain `
  --resume_checkpoint .\out\checkpoints\pretrain\pretrain_last.pt `
  --data_path .\data\pretrain_t2t_mini.jsonl `
  --tokenizer_path ..\minimind\model `
  --output_dir .\out\checkpoints\pretrain `
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
  --warmup_steps 1000 `
  --min_lr_ratio 0.1 `
  --num_workers 0
```

启动 TensorBoard：

```powershell
& $trainPy -m tensorboard.main --logdir .\out\runs --port 6006
```

然后打开 <http://localhost:6006>。

### 如何读 TensorBoard 指标

训练吞吐不再只看一个笼统的 `tokens/s`，而是按最近 100 次成功 optimizer 更新计算滚动平均。恢复训练后，旧 run 的 step 不会混进新的 ETA。

```text
throughput/raw_tokens_per_second
→ 模型实际计算的张量位置，包含 padding；适合判断硬件吞吐。

throughput/valid_tokens_per_second
→ 非 padding 的真实文本 token；适合判断数据利用率。

throughput/target_tokens_per_second
→ 真正参与 next-token loss 的标签 token；SFT 最应关注这一项。

data/padding_ratio
→ 当前窗口中被 padding 占用的比例，越高说明固定长度带来的算力浪费越多。

data/target_token_ratio
→ 有效文本中实际参与监督的比例；SFT 过低时要检查 assistant 标签和对话模板。

train/grad_norm、stability/grad_scaler_scale、stability/fp16_overflow_total
→ 分别用于观察梯度大小、FP16 缩放和溢出恢复情况。
```

控制台中的 `tok/s(raw/valid/target)` 与这些曲线一致。`raw` 高但 `valid` 或 `target` 很低时，优先检查 padding、截断和标签掩码，而不是只加大 batch size。

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
  --checkpoint .\out\checkpoints\pretrain\pretrain_last.pt `
  --output .\out\checkpoints\pretrain\pretrain_512.pth
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
