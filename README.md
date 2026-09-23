# MiniMind Labs

这是一个从 Transformer 模块开始，亲手组装、训练、评测并部署小语言模型的实验项目。

模型结构和训练流程参考 [MiniMind](https://github.com/jingyaogong/minimind)，但本仓库是独立的教学实现，不是官方 MiniMind。MiniMind 原仓库只作只读对照，训练代码、数据准备脚本和实验记录维护在这里。

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
Pretrain → 通用 SFT → 代码/数学专项 SFT → DPO → 评测与 API 部署
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
│   ├── prepare_specialized_sft.py # 混合代码、数学和通用 SFT 数据
│   ├── eval_sft_comparison.py     # 两个 checkpoint 的固定集对照
│   └── export_official_weights.py # 旧训练 checkpoint 导出工具
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

## 数据集与来源

数据文件放在 `data/`，不会提交到 Git。预训练、SFT 和 DPO 的数据来自 MiniMind 官方数据仓库；代码、数学专项数据由本项目脚本转换成 SFT 格式。

| 文件 | 来源 | 在流程中的用途 |
|---|---|---|
| `data/pretrain_t2t_mini.jsonl` | MiniMind 官方轻量预训练集。官方说明其为清洗、去重后的混合文本，来源包括匠数大模型数据集、Magpie-Align 等；没有公开逐条来源比例。 | Pretrain，学习文本分布和续写。 |
| `data/sft_t2t_mini.jsonl` | MiniMind 官方轻量 SFT 集，混合公开指令/对话与合成蒸馏数据，含部分 Tool Call。官方列出的来源包括匠数、Magpie-Align、R1-Distill-SFT、COIG、Step-3.5-Flash-SFT 等。 | 通用 SFT，也作为专项 SFT 的通用回放数据。 |
| `data/raw/code/data/*.parquet` | [`open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled`](https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled)，15,068 条通过参考测试的 Python 题；source 标签包括 TACO、CodeContests、APPS。 | 代码专项 SFT 候选数据及代码评测题源。 |
| `data/raw/math/data/*.parquet` | 下载自 [`minhdang/math`](https://huggingface.co/datasets/minhdang/math)，200,035 条 question/answer。样例与 [Microsoft Orca-Math-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) 相同，但 `minhdang/math` 数据卡没有写明上游来源，因此这里按实际下载仓库署名。 | 数学专项 SFT 候选数据及数学评测题源。 |
| `data/sft_code_math_mix_replay10k.jsonl` | 本地运行 `scripts/prepare_specialized_sft.py`，由代码题、数学题和 10,000 条通用 SFT 回放样本混合生成；当前文件 24,569 行，长样本会按 768 token 限制筛除。 | 专项 SFT。 |
| `data/dpo.jsonl` | MiniMind 官方打包的偏好数据，抽样自 [`DPO-En-Zh-20k`](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)，包含 `chosen` / `rejected` 回答对。 | DPO 偏好对齐，不是代码或数学正确性奖励。 |

MiniMind tokenizer 使用原仓库 `model/` 目录。预训练 JSONL 每行形如 `{"text":"..."}`；SFT 文件使用 `conversations` 多轮消息。SFT loss 只计算 assistant 回答部分。

专项 SFT 数据可从已下载的 parquet 文件重新生成：

```bash
python scripts/prepare_specialized_sft.py \
  --code_dir data/raw/code/data \
  --math_dir data/raw/math/data \
  --general_data data/sft_t2t_mini.jsonl \
  --output data/sft_code_math_mix_replay10k.jsonl \
  --code_samples 5000 --math_samples 10000 --general_samples 10000 \
  --max_seq_len 768 --tokenizer_path ../minimind/model
```

MiniMind 数据卡：[预训练与 SFT 来源](https://huggingface.co/datasets/jingyaogong/minimind_dataset)。该仓库提供的是整理后的混合数据，无法从文件名还原每条样本的原始出处。

## 训练流程

训练的完整路径是：

```text
Pretrain：学习通用文本的续写分布
    ↓
SFT：学习指令跟随与对话格式
    ↓
专项 SFT：混入代码、数学题和通用回放数据
    ↓
DPO：用 chosen/rejected 偏好对继续训练
    ↓
评测与 API 推理：检查训练前后变化并加载模型
```

### 本次实际训练

这次跑的是 63.91M 参数模型：`hidden_size=768`、8 层、8 个 Q heads、4 个 KV heads，训练序列长度 768。实际顺序和权重如下：

| 阶段 | 初始化与数据 | 训练后权重 |
|---|---|---|
| Pretrain | 随机初始化，使用 `pretrain_t2t_mini.jsonl`，1 epoch | `out/checkpoints/pretrain_768/pretrain_768.pth` |
| 通用 SFT | 从 Pretrain 权重开始，使用 `sft_t2t_mini.jsonl`，1 epoch | `out/checkpoints/sft_768/full_sft_best_768.pth` |
| 专项 SFT | 从通用 SFT 开始，使用 `sft_code_math_mix_replay10k.jsonl`，1 epoch | `out/checkpoints/sft_code_math_replay10k/full_sft_best_768.pth` |
| DPO | 从专项 SFT 开始，使用 `dpo.jsonl`，1 epoch，`beta=0.15`，学习率 `4e-8` | `out/checkpoints/dpo/dpo_768.pth` |

下面的 512 hidden-size 命令是较省显存的通用示例，不是这次 768 模型的实际训练配置。复现本次实验时，结构参数要保持 768/8 层/8 Q heads/4 KV heads。

### 参数选择：通用单卡基线

上面的 768 配置是本次实际训练使用的模型结构。下面的 512 命令是显存更省的通用示例，两者不要混为同一次实验。

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
python -m pip install 'huggingface-hub>=0.34.0,<1.0'
hf download jingyaogong/minimind_dataset dpo.jsonl --repo-type dataset --local-dir data
```

这里给 Hub 客户端加了版本上限：当前训练依赖使用 Transformers 4.x，Hub 1.x 会导致导入版本冲突。数据下载完成后不用再次升级 `huggingface-hub`。

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

### 评测方法与实际结果

`scripts/eval_sft_comparison.py` 用固定解码设置对比两个权重：8 条通用问题供人工检查、50 道数学题按最终数值做 Exact Match、50 道代码题计算 Pass@1。数学分数只看抽取到的最终数字，不评判推理过程；50 题里答对/答错一题，就会变化 2 个百分点。

| 对比 | 数学 Exact Match | 代码 Pass@1 | 环境与备注 |
|---|---:|---|---|
| 官方 MiniMind 64M SFT vs 本项目通用 SFT | 2% vs 2%（各 1/50） | 跳过 | CPod，BF16 |
| 本项目专项 SFT vs DPO | 2% vs 2%（各 1/50） | 跳过 | CPod，BF16；这轮没有测出 DPO 数学提升 |
| 通用 SFT vs 专项 SFT | 2% vs 4%（1/50 vs 2/50） | 跳过 | 本地 2080 Ti，FP16；另一次 CPod BF16 对比为 2% vs 2%，结果不一致，不能据此宣称稳定提升 |

代码执行没有完成：CPod 的内核不允许 Bubblewrap 创建所需的隔离命名空间。因此报告里的代码样本只是生成结果，`code_pass_at_1` 状态为 `skipped`，不是 0 分。不要把未经沙箱验证的生成代码直接运行。

评测会按训练 JSONL 里的 `conversations` user prompt 去重；目前不解析 DPO 文件的 `chosen/rejected` 对话，所以 DPO 评测还没有排除 DPO 样本本身的 prompt。完整并排回答和指标保存在 `out/evaluations/*.json`；8 条通用回答需要人工检查，不作为量化分数。

例如，比较 DPO 前后的专项 SFT：

```bash
python scripts/eval_sft_comparison.py \
  --base_checkpoint out/checkpoints/sft_code_math_replay10k/full_sft_best_768.pth \
  --candidate_checkpoint out/checkpoints/dpo/dpo_768.pth \
  --train_data data/sft_t2t_mini.jsonl data/sft_code_math_mix_replay10k.jsonl \
  --code_dir data/raw/code/data \
  --math_dir data/raw/math/data \
  --tokenizer_path ../minimind/model \
  --output out/evaluations/sft_vs_dpo.json \
  --skip_code_execution
```

要统计代码 Pass@1，需在允许创建 Bubblewrap 用户/进程/网络命名空间的 Linux 环境运行；单纯安装 Bubblewrap 不一定能解除容器内核的限制。

### API 推理

`dpo_768.pth` 已经是 MiniMind 原生 PyTorch `state_dict`，不必为了官方 MiniMind API 再转成 safetensors 或 Transformers 目录。用原仓库的 `scripts/serve_openai_api.py` 加载即可；这条路径已用 `eval_llm.py` 严格加载并成功生成。API 能返回内容不代表回答质量可靠，当前小模型仍会重复或编造信息。

在两个仓库并排放置的 Linux 服务器上，从 MiniMind 原仓库的 `scripts/` 目录启动：

```bash
cd ../minimind/scripts
# 若环境缺少服务依赖，先安装：python -m pip install fastapi uvicorn
python serve_openai_api.py \
  --load_from ../model \
  --save_dir ../minimind-labs/out/checkpoints/dpo \
  --weight dpo \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 \
  --device cuda:0
```

官方加载器会把 `--weight dpo` 和 `hidden_size=768` 拼成 `dpo_768.pth`。服务监听 `8998` 端口，提供 `/v1/chat/completions`。这个示例服务没有认证，不要直接暴露到公网；仅在本机或受控的内网端口转发中使用。

Windows 本地 `.venv-train` 若由 `uv` 创建，可能没有 `pip`。可用 `uv pip install --python .\.venv-train\Scripts\python.exe fastapi uvicorn` 安装服务依赖，再在 MiniMind 的 `scripts` 目录用同一组参数启动；`--save_dir` 要指向旁边 `minimind-labs/out/checkpoints/dpo`。

## 暂停、恢复和监控

Pretrain 和 SFT 收到 `Ctrl+C` 时会保存最近一次完整 optimizer 更新后的安全状态。DPO 则按 `--save_interval` 定期保存，并在 epoch 结束时保存；中断后从最近的 `dpo_last.pt` 恢复，可能需要重跑最后一次保存后的少量 step。

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

pretrain_<hidden_size>.pth / full_sft_<hidden_size>.pth / dpo_<hidden_size>.pth
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

## 下一步计划

当前已经跑通 Pretrain → 通用 SFT → 代码/数学专项 SFT → DPO，并用 MiniMind 官方 API 加载 DPO 权重。下一步先补可靠评测，不急着加模型结构：

1. 让评测脚本也从 DPO 的 `chosen/rejected` 样本中过滤重合题目。
2. 在允许安全隔离代码的 Linux 主机上测 Code Pass@1；固定设备、精度、采样和题目后重跑数学与通用对照。
3. 若目标仍是提高代码/数学正确率，再尝试带可验证奖励的 GRPO/RLVR；MoE 和多模态暂缓。

目前的评测样本较少，数学分数没有显示稳定提升，代码通过率也尚未测出。因此项目展示重点应放在可复现的训练、权重加载和 API 闭环，不要宣称模型能力已经超过 MiniMind。

## 参考项目

- [MiniMind](https://github.com/jingyaogong/minimind)：模型结构、训练流程和官方权重格式的主要参考
- [MiniMind Notes](https://github.com/joyehuang/minimind-notes)：学习过程中的补充讲解与实验参考

本项目是学习用途的独立实现，具体功能以当前代码和命令为准。
