<div align="center">

# DualTrace
### Complementary Evidence for AI-Generated Image and Text Detection

[简体中文](README.md) · **English** · [Method](docs/method.md)

</div>

DualTrace combines pretrained semantic representations with local forensic cues for AI-generated image and text detection. The image branch integrates multi-depth, multi-view features and conditional experts; the text branch combines a language model trained with continuous editing targets and character-level evidence. The two branches are trained and deployed independently.

## Method overview

| Branch | Models and design |
|:--|:--|
| Image representations | **PE Core G/14, PE Spatial G/14, CLIP ViT-L/14** combine global semantics, intermediate-layer statistics and native-resolution crops |
| Image forensics | **NPR, RINE, SigLIP and face ViT** complement grayscale, multiscale and shape-conditioned experts |
| Image readout | Conditional kernel classifiers organize evidence; an **ExtraTrees** student learns teacher responses for single-image scoring |
| Text representations | **Qwen2.5-1.5B-Instruct + LoRA** with public-data preadaptation, continuous editing regression and robust target-domain adaptation |
| Text readout | **Character TF-IDF + Ridge** learns teacher responses and is fused with Qwen after fixed empirical-CDF alignment |

See the [method description](docs/method.md) for architecture, objectives and fusion equations.

## Reproduction

### 1. Environment

Use Python 3.11, Linux and an NVIDIA CUDA GPU. For inference, recommended resources are 24 GB GPU / 64 GB RAM for images and 8 GB GPU / 16 GB RAM for text. Training requirements depend on the stage and batch size.

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

### 2. Data preparation

Training datasets are not needed for inference. For training, obtain the following datasets and retain the supplied selection manifests.

| Dataset | Purpose | Source |
|:--|:--|:--|
| NLPCC edited text | Continuous supervision from human, generated and edited text | [NLPCC Task 6](https://github.com/NLP2CT/NLPCC-2026-Task6-Detection) |
| CommunityForensics | Visual preadaptation | [Dataset](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small) |
| AIGI-v2 | Generator coverage and real/generated pairing | [Dataset](https://huggingface.co/datasets/Shanmuk4622/ai-detection-dataset-v2) |
| OpenFake | Additional visual sources | [Dataset](https://huggingface.co/datasets/ComplexDataLab/OpenFake) |
| Target-domain data | Supervised adaptation and unlabeled references | Supply authorized data; original data are provided by the [challenge platform](https://www.aicompetition-pz.com/topic_detail/41) |

Text training CSVs use `id,text,label` (`0=human, 1=generated`); reference CSVs use `id,text`. The original image recipe expects:

```text
data/competition/image/
├── image_sample_data.csv       # id,label
├── image_test.csv              # id
├── image_submission_example.csv
└── data/
    ├── image_sample_data/      # filenames match id
    └── image_test/
```

For authorized original ZIPs, run `python tools/prepare_competition_data.py --archives /path/to/zips --output data/competition --include-test`. The image recipe expects 1,000 labeled images and 20,000 unlabeled reference images; adjust its data checks and configuration when changing these sizes.

**Text:** download the pinned data, deduplicate, exclude reference overlaps and construct continuous editing targets.

```bash
python tools/fetch_text_data.py --output data/nlpcc/train_data.json
python training/text/prepare_data.py \
  --input data/nlpcc/train_data.json \
  --competition data/text/train.csv \
  --test-reference data/text/reference.csv \
  --output data/nlpcc/filtered.jsonl
```

**Images:** prepare the visual backbones below, then restore selected samples and the training workspace.

```bash
python -m pip install -r training/image/requirements-data.txt
python training/image/bootstrap.py --data-root data \
  --backbones-root .local/checkpoints/image/backbones
python training/image/fetch_data.py --plan
python training/image/fetch_data.py
```

The downloader checks selected image SHA-256 hashes and resumes existing downloads. If OpenFake row indices change, add `--dataset openfake --openfake-mode parquet` to recover by hash; a full scan may require substantial bandwidth and scratch space.

### 3. Checkpoints

**Public download links for complete trained detectors are not yet available.** If you already have the checkpoint bundle, preserve its internal filenames under `.local/checkpoints/image/` and `.local/checkpoints/text/`, then run `python tools/verify_assets.py`. Existing original image archives can be extracted with `python tools/restore_assets.py`.

Public backbones alone do not include our trained detection heads. Training uses:

| Model | Source | Location within the image training workspace |
|:--|:--|:--|
| PE Core G/14 | [Model](https://huggingface.co/timm/vit_pe_core_gigantic_patch14_448.fb) | `.cache/pe_core_g14/model.safetensors` |
| PE Spatial G/14 | [Model](https://huggingface.co/timm/vit_pe_spatial_gigantic_patch14_448.fb) | `.cache/pe_spatial_g14/model.safetensors` |
| CLIP ViT-L/14, ViT-L/14@336px | [OpenAI CLIP](https://github.com/openai/CLIP) | `.cache/clip/` |
| NPR / RINE | [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) / [RINE](https://github.com/mever-team/rine) | `.cache/npr/model_epoch_last_3090.pth` / `.cache/rine/model_ldm_trainable.pth` |
| SigLIP–DINOv2 | [Model](https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2) | `.cache/bombek_dual/pytorch_model.pt` |
| Face ViT | [Model](https://huggingface.co/dima806/deepfake_vs_real_image_detection) | `.cache/deepfake_vit/` including configuration |

The default image training workspace is `runs/image_training/`. Bootstrap can link backbones from an existing complete bundle; training additionally needs the full SigLIP–DINOv2 model.

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir .local/checkpoints/text/qwen2.5_1.5b_instruct
hf download Bombek1/ai-image-detector-siglip-dinov2 pytorch_model.pt \
  --local-dir runs/image_training/.cache/bombek_dual
```

### 4. Inference with trained weights

```bash
dualtrace predict --track text --input examples/text.csv --output runs/text_scores.csv
dualtrace predict --track image --input data/images.csv --output runs/image_scores.csv
# An alternative checkpoint location:
dualtrace --checkpoints /path/to/checkpoints predict \
  --track text --input examples/text.csv --output runs/text_scores_custom.csv
```

Text inputs use `id,text`; image inputs use `id,path`, with relative paths resolved against the CSV directory. Outputs use `id,score` and preserve row order. Higher scores indicate stronger generated-content evidence; these are ranking confidences, not calibrated probabilities.

Start HTTP serving with `dualtrace serve --track text --port 8080`, or replace `text` with `image`. Readiness: `GET /health/readiness`. Inference: `POST /predict` with `{"req_id":"demo","data":"..."}`. Supply plain text or base64 image bytes without a data-URI prefix.

### 5. Training

Training starts from public pretrained backbones and fits new adapters and detection heads without overwriting preserved inference checkpoints.

**Text:** public preadaptation followed by robust target-domain adaptation.

| Parameter | Public preadaptation | Target-domain adaptation |
|:--|:--|:--|
| Epochs / learning rate | 1 / 3e-5 | 1 / 1e-5 |
| Maximum tokens | 2,048 | 1,536 |
| Batch / accumulation | 2 / 6 | 2 / 6 |
| LoRA | rank 16, alpha 32, dropout 0.05 | same |
| Optimizer / schedule | AdamW; weight decay 0.01; 3% warmup, then constant | same |

```bash
python training/text/train.py --competition data/text/train.csv \
  --external data/nlpcc/filtered.jsonl --check
python training/text/train.py --competition data/text/train.csv \
  --external data/nlpcc/filtered.jsonl --output runs/text_retrain
python training/text/predict.py --adapter runs/text_retrain/adapter \
  --input data/text/reference.csv --output runs/text_raw.csv --view first
```

A fresh adapter produces raw regression scores. Add `--save-calibration runs/text_ecdf.npz` on reference content to create a matching CDF, then use `--calibration runs/text_ecdf.npz` for new inputs. Do not reuse an old CDF with a new adapter. Deployment uses the first 2,048 tokens; the training inference utility also supports head/middle/tail averaging.

**Images:** the launcher orders CLIP external adaptation, PE readout training, multiview extraction and kernel-classifier fitting. Default learning rates, epochs and augmentations are in `research/image/configs/`.

```bash
python training/image/train.py --plan
python training/image/train.py --check
python training/image/train.py
# Resume only matching completed recipes.
python training/image/train.py --resume
```

New visual components and calibration assets are written to `runs/image_training/assets/`. Add `--teacher /path/to/image_teacher.csv` to train the ExtraTrees deployment readout; teacher responses must correspond to the same reference image pool. Train the character Ridge readout with `training/text/distill.py`, supplying reference texts, teacher scores and corresponding Qwen responses; see `--help` for input arguments. Both readouts use teacher pseudo-labels; retraining the backbone alone does not reconstruct the original teacher.

## Structure

```text
src/dualtrace/       inference and HTTP serving
training/           preprocessing, training and readout distillation
research/image/     visual training sources, configs and external selections
docs/method.md      method description
tools/              data downloads and checkpoint verification
tests/              interface and release checks
```

Run `python -m unittest discover -s tests -v` for interface and path checks. Local asset checks and CPU loading tests have passed; the full CUDA training chain still requires validation on the target environment.

## License and acknowledgments

Original code is released under the [MIT License](LICENSE). Third-party models, code and datasets retain their own terms; see [NOTICE](NOTICE.md). We thank the authors of Qwen, Perception Encoder, CLIP, NPR, RINE and the datasets used here.
