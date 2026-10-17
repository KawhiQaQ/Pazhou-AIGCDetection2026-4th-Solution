<div align="center">

# DualTrace
### 面向 AI 生成图像与文本的互补证据检测

**简体中文** · [English](README.en.md) · [方法介绍](docs/method.md)

</div>

DualTrace 结合预训练模型的语义表征与局部取证线索，检测 AI 生成图像和文本。图像分支融合多深度、多视图特征及条件专家；文本分支结合连续编辑监督下的语言模型与字符级证据。两个分支独立训练和推理，可分别部署。

## 方案介绍

| 分支 | 模型与设计 |
|:--|:--|
| 图像表征 | **PE Core G/14、PE Spatial G/14、CLIP ViT-L/14**：融合全局语义、中间层统计与原生分辨率裁剪 |
| 图像取证 | **NPR、RINE、SigLIP、人脸 ViT**：结合灰度、多尺度与形状条件专家，补充局部生成痕迹 |
| 图像读出 | 条件核分类器组织互补证据，**ExtraTrees** 学习教师响应，输出单图分数 |
| 文本表征 | **Qwen2.5-1.5B-Instruct + LoRA**：公共数据预适配、连续编辑回归和目标域鲁棒适配 |
| 文本读出 | **字符 TF-IDF + Ridge** 学习教师响应，与 Qwen 分数经固定经验 CDF 对齐后融合 |

模块设计、训练目标及融合公式见[方法介绍](docs/method.md)。

## 复现指南

### 1. 环境配置

使用 Python 3.11、Linux 和 NVIDIA CUDA GPU。推理建议图像分支配置 24 GB 显存、64 GB 内存，文本分支配置 8 GB 显存、16 GB 内存；训练资源取决于阶段与批大小。

```bash
git clone https://github.com/KawhiQaQ/Pazhou-AIGCDetection2026-4th-Solution.git
cd Pazhou-AIGCDetection2026-4th-Solution
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.0 torchvision==0.24.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu.txt
python -m pip install --no-deps -e .
```

### 2. 数据准备

仅推理无需训练数据。自行训练时，从以下来源获取数据，并保留仓库提供的筛选清单。

| 数据 | 用途 | 来源 |
|:--|:--|:--|
| NLPCC 编辑文本 | 人工、生成及编辑文本的连续监督 | [NLPCC Task 6](https://github.com/NLP2CT/NLPCC-2026-Task6-Detection) |
| CommunityForensics | 视觉外部预适配 | [数据集](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small) |
| AIGI-v2 | 生成来源扩展与真实/生成配对 | [数据集](https://huggingface.co/datasets/Shanmuk4622/ai-detection-dataset-v2) |
| OpenFake | 视觉来源扩展 | [数据集](https://huggingface.co/datasets/ComplexDataLab/OpenFake) |
| 目标域数据 | 监督适配与无标签参考集 | 自备授权数据；原始数据由[赛事平台](https://www.aicompetition-pz.com/topic_detail/41)提供 |

文本训练 CSV 使用 `id,text,label`（`0=人工，1=生成`），参考 CSV 使用 `id,text`。图像原始配方使用以下目录：

```text
data/competition/image/
├── image_sample_data.csv       # id,label
├── image_test.csv              # id
├── image_submission_example.csv
└── data/
    ├── image_sample_data/      # 文件名与 id 一致
    └── image_test/
```

持有原始 ZIP 时可运行 `python tools/prepare_competition_data.py --archives /path/to/zips --output data/competition --include-test`。图像原始配方要求 1,000 张标注图像和 20,000 张无标签参考图像；更换规模时需调整数据检查与配方。

**文本预处理**：下载固定版本，去重、排除参考文本重叠，并构造连续编辑目标。

```bash
python tools/fetch_text_data.py --output data/nlpcc/train_data.json
python training/text/prepare_data.py \
  --input data/nlpcc/train_data.json \
  --competition data/text/train.csv \
  --test-reference data/text/reference.csv \
  --output data/nlpcc/filtered.jsonl
```

**图像预处理**：先准备下节的视觉骨干，再恢复选样清单和训练工作目录。

```bash
python -m pip install -r training/image/requirements-data.txt
python training/image/bootstrap.py --data-root data \
  --backbones-root .local/checkpoints/image/backbones
python training/image/fetch_data.py --plan
python training/image/fetch_data.py
```

下载器按 SHA-256 校验选定图片，支持恢复已有下载。OpenFake 行号失效时可加 `--dataset openfake --openfake-mode parquet` 按哈希检索；全量扫描可能需要较大带宽与临时磁盘空间。

### 3. 模型权重

**完整检测权重的公开下载链接尚未发布。** 已有权重包的用户应保留内部文件名，放置为 `.local/checkpoints/image/` 和 `.local/checkpoints/text/`，随后运行 `python tools/verify_assets.py`。已有原始镜像归档时，可用 `python tools/restore_assets.py` 提取权重。

仅下载公开骨干不能替代训练好的检测头。自行训练使用以下预训练模型：

| 模型 | 来源 | 视觉训练工作目录中的位置 |
|:--|:--|:--|
| PE Core G/14 | [模型](https://huggingface.co/timm/vit_pe_core_gigantic_patch14_448.fb) | `.cache/pe_core_g14/model.safetensors` |
| PE Spatial G/14 | [模型](https://huggingface.co/timm/vit_pe_spatial_gigantic_patch14_448.fb) | `.cache/pe_spatial_g14/model.safetensors` |
| CLIP ViT-L/14、ViT-L/14@336px | [OpenAI CLIP](https://github.com/openai/CLIP) | `.cache/clip/` |
| NPR / RINE | [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) / [RINE](https://github.com/mever-team/rine) | `.cache/npr/model_epoch_last_3090.pth` / `.cache/rine/model_ldm_trainable.pth` |
| SigLIP–DINOv2 | [模型](https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2) | `.cache/bombek_dual/pytorch_model.pt` |
| 人脸 ViT | [模型](https://huggingface.co/dima806/deepfake_vs_real_image_detection) | `.cache/deepfake_vit/`（含配置） |

视觉训练工作目录默认为 `runs/image_training/`。`bootstrap.py` 可从已有完整权重包链接骨干；训练还需要完整的 SigLIP–DINOv2 模型。

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir .local/checkpoints/text/qwen2.5_1.5b_instruct
hf download Bombek1/ai-image-detector-siglip-dinov2 pytorch_model.pt \
  --local-dir runs/image_training/.cache/bombek_dual
```

### 4. 直接加载权重推理

```bash
dualtrace predict --track text --input examples/text.csv --output runs/text_scores.csv
dualtrace predict --track image --input data/images.csv --output runs/image_scores.csv
# 权重位于其他目录时：
dualtrace --checkpoints /path/to/checkpoints predict \
  --track text --input examples/text.csv --output runs/text_scores_custom.csv
```

文本输入为 `id,text`，图像输入为 `id,path`；图片相对路径基于 CSV 所在目录解析。输出为 `id,score`，保持输入顺序。分数越大表示生成倾向越强，是排序置信度而非校准概率。

HTTP 服务：`dualtrace serve --track text --port 8080`，图像服务将 `text` 换成 `image`。健康检查为 `GET /health/readiness`；`POST /predict` 接收 `{"req_id":"demo","data":"..."}`。文本直接传字符串，图像传不带前缀的 base64。

### 5. 自行训练

训练从公开预训练骨干开始，重新拟合适配器和检测头，不会覆盖保留的推理权重。

**文本**：公共预适配后进行目标域鲁棒适配。

| 参数 | 公共预适配 | 目标域适配 |
|:--|:--|:--|
| Epoch / 学习率 | 1 / 3e-5 | 1 / 1e-5 |
| 最大长度 | 2,048 | 1,536 |
| Batch / 梯度累积 | 2 / 6 | 2 / 6 |
| LoRA | rank 16、alpha 32、dropout 0.05 | 相同 |
| 优化器与调度 | AdamW；weight decay 0.01；3% warmup 后恒定 | 相同 |

```bash
python training/text/train.py --competition data/text/train.csv \
  --external data/nlpcc/filtered.jsonl --check
python training/text/train.py --competition data/text/train.csv \
  --external data/nlpcc/filtered.jsonl --output runs/text_retrain
python training/text/predict.py --adapter runs/text_retrain/adapter \
  --input data/text/reference.csv --output runs/text_raw.csv --view first
```

新 adapter 输出原始回归分数。用 `--save-calibration runs/text_ecdf.npz` 在参考集上生成新 CDF，再用 `--calibration runs/text_ecdf.npz` 对新输入评分；不要为新 adapter 沿用旧 CDF。默认部署读取前 2,048 tokens，训练推理工具另支持首/中/尾窗口均值。

**图像**：按依赖顺序执行 CLIP 外部适配、PE 检测头训练、多视图提取与核分类器拟合。学习率、轮数和增强配置位于 `research/image/configs/`。

```bash
python training/image/train.py --plan
python training/image/train.py --check
python training/image/train.py
# 中断后继续；源码或配置改变时不会复用旧完成标记。
python training/image/train.py --resume
```

新视觉组件与校准资产写入 `runs/image_training/assets/`。训练 ExtraTrees 部署读出时添加 `--teacher /path/to/image_teacher.csv`，教师分数须对应同一参考图像池。字符 Ridge 读出由 `training/text/distill.py` 训练，输入参考文本、教师分数及对应 Qwen 响应，参数见 `--help`。这两个读出使用教师伪标签，单独重训骨干不会自动重建原教师。

## 项目结构

```text
src/dualtrace/       推理与 HTTP 服务
training/           数据处理、训练与读出蒸馏
research/image/     视觉训练源码、配置与外部选样清单
docs/method.md      方法介绍
tools/              数据下载与权重校验
tests/              接口与发布检查
```

运行 `python -m unittest discover -s tests -v` 检查接口和路径。已完成本地资产校验与 CPU 加载测试；完整 CUDA 训练链仍需在目标环境验证。

## 许可与致谢

原创代码采用 [MIT License](LICENSE)。第三方模型、代码和数据遵循各自许可，见 [NOTICE](NOTICE.md)。感谢 Qwen、Perception Encoder、CLIP、NPR、RINE 及相关数据集作者。
