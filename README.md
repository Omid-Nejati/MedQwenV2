# SG-MoE: Conflict-Aware Spectral Routing for Medical Segmentation and Language Tasks

**Status: runnable research prototype, not a validated published method.**

This project implements a first experimental version of the proposed extension to MedQwen. It is built around **Qwen2.5-VL** and contains:

- **SVD-initialized spectral LoRA** on selected late language-layer `q_proj` and `v_proj` matrices, including a mandatory shared expert and task-conditioned Top-K private experts.
- **Text-conditioned binary segmentation** from the Qwen hidden states corresponding to merged image tokens, using a lightweight convolutional decoder (not SAM2).
- **Medical text tasks** via standard autoregressive assistant supervision: VQA, report generation, and text-based classification.
- **Per-expert conflict diagnostics** using the cosine of segmentation and language task gradients, with exponential moving averages.
- **Conflict-aware routing penalty** that discourages both tasks from choosing the same private expert when it has opposing gradients.
- **PCGrad on shared LoRA parameters** to reduce conflict when segmentation and text objectives disagree.
- Minimal JSONL training, adapter-only checkpoints, inference, synthetic example-data generator, and CPU unit tests.

## Limitations you should know before experiments

1. The **vision encoder remains frozen**; gradients from segmentation flow through the Qwen language layers' spectral adapters and the mask decoder.
2. The mask head operates on **merged Qwen visual tokens**. This is a **low-resolution binary-mask prototype**, not a clinical-quality dense segmentation model or SAM2 integration. Adding a multi-scale image decoder is a sensible next step.
3. **One still image per example and batch size 1 per task** (one segmentation and one text example per optimization step). The paired two-task training costs memory, because both computational graphs are retained for gradient comparisons.
4. VQA, report generation, and classification all use a **language generation loss** and task-specific prompts; there is no calibrated standalone classifier head or separate evaluation implementation.
5. SVD right singular vectors give an **initial** spectral partition. Updated matrices are not mathematically guaranteed to remain orthogonal during training.
6. No segmentation benchmark results, clinical accuracy, theoretical guarantees, or published baseline improvements are claimed. **Do not report the synthetic generator results as medical performance.**

## Installation

Python 3.10+; NVIDIA CUDA GPU recommended for the actual Qwen backbone.

```bash
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

`transformers` is constrained to its v4 API line here because the prototype uses `Qwen2_5_VLForConditionalGeneration` and `backbone.model`. If a later release changes model names or output layout, adapt the wrapper and rerun the tests.

## CPU unit tests (no model download)

```bash
python -m pytest -q tests/
```

## Create synthetic data (pipeline test only)

```bash
python examples/make_synthetic.py
```

Generated files: `examples/train.jsonl` and `examples/data/*.png`. They contain geometric objects, **not medical images**.

## Real dataset format

JSON Lines file, paths relative to `--image_root`:

```jsonl
{"task":"seg","image":"images/a.png","mask":"masks/a.png","prompt":"Segment the left lung."}
{"task":"vqa","image":"images/b.png","prompt":"Is there a pulmonary opacity?","answer":"Yes, an opacity is visible."}
{"task":"report","image":"images/c.png","prompt":"Write a short clinical report.","answer":"..."}
{"task":"classification","image":"images/d.png","prompt":"Classify the image. Reply with the class label.","answer":"normal"}
```

Segmentation masks must be **binary, foreground >0**. For multi-class datasets, create per-class binary masks (or change `data.py` and decoder). The image is resized internally by the Qwen processor; ground-truth masks are resized to a square `mask_size` with nearest-neighbor sampling. This is a practical prototype, but for rigorous segmentation benchmarks you should store and undo the exact geometric preprocessing transformation, preserve original aspect ratios, and compute Dice/IoU at original sizes.

**Prompt ordering is intentional:** instruction text precedes the `<image>` marker. With a causal LLM, later image-token representations can then attend to the target-region instruction.

## Train

Begin with the 3B checkpoint to debug the pipeline. The 7B is more expensive.

```bash
python train.py \
  --train_jsonl examples/train.jsonl \
  --image_root examples/data \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --output_dir runs/first \
  --max_steps 10 \
  --last_n_layers 4 \
  --shared_rank 4 --private_rank 4 \
  --num_private 3 --top_k 1 \
  --conflict_lambda 0.1 --pcgrad \
  --grad_checkpointing --init spectral
```

To test **ordinary multitask spectral LoRA without conflict-aware routing/projection**, set `--conflict_lambda 0 --no-pcgrad`. To test routing only, disable PCGrad but keep `--conflict_lambda 0.1`. For a purely random LoRA initialization control, use `--init random`.

The model downloads pretrained weights from Hugging Face on the first run (network access and model license compliance required). Depending on training length and available VRAM, this can require significant resources. The example Slurm script requests one GPU but does **not** assume your account is entitled to any specific GPU model.

## Predict masks or VQA answers

```bash
python predict.py --adapter runs/first/adapter_final.pt \
   --image examples/data/im0.png --task seg \
   --prompt "Segment the bright circular object." \
   --output runs/first/pred_mask.png

python predict.py --adapter runs/first/adapter_final.pt \
   --image examples/data/im0.png --task vqa \
   --prompt "Is there a bright object in the image?"
```

The prediction mask is resized back to the original image size. For robust medical evaluation, handle aspect-ratio-preserving preprocessing and pixel spacing explicitly.

## Architecture and training math

For a frozen linear layer `W`:

`y = Wx + (alpha/r_shared) B_s A_s x + (alpha/r_private) sum_e g_e(x,t) B_e A_e x`

`A` matrices start from distinct **right singular vectors** of `W` found using randomized low-rank SVD. `B` matrices start at zero, preserving the original pretrained output. A learned softmax router with task embeddings selects Top-K private experts; shared LoRA is always active.

During paired training, each task's gradients are computed separately. For private expert `e`:

`c_e = max(0, -cos(grad_seg^e, grad_text^e))`

An EMA of `c_e` weights a cross-task expert-overlap penalty:

`L_route = mean_layer sum_e EMA(c_e) * p_seg(e) * p_text(e)`

Training loss:

`L = w_seg * (BCE + Dice) + w_text * LM_CE + lambda * L_route`

On **shared adapters only**, the implementation projects conflicting task gradients (symmetric PCGrad) before optimizer step. Private experts still receive conventional gradients from both tasks and the routing penalty. This is a heuristic implementation that should be compared with uncertainty balancing, GradNorm, PCGrad-only, and fully task-separated adapters.

## Recommended ablation table

| Run | SVD initialization | Task routing | Conflict overlap penalty | Shared PCGrad |
|---|---|---|---|---|
| A: ordinary LoRA | No | No (requires modifying `num_private` or router) | No | No |
| B: spectral LoRA-MoE | Yes | Yes | No | No |
| C: conflict-aware routing | Yes | Yes | Yes | No |
| D: PCGrad only | Yes | Yes | No | Yes |
| E: full SG-MoE | Yes | Yes | Yes | Yes |

The ordinary-LoRA baseline is **not implemented** by a single flag; use a PEFT LoRA control at a matched trainable parameter budget.

## Concordia SPEED cluster

After transferring this folder to your SPEED scratch space, installing dependencies in your existing venv, and editing Slurm resources as appropriate:

```bash
mkdir -p logs
sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct,MAX_STEPS=100 speed_train.sbatch
```

## Architecture references

- Original MedQwen: https://arxiv.org/abs/2604.01310
- Qwen2.5-VL implementation docs: https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl
- SAM2 (not integrated here): https://github.com/facebookresearch/sam2

## Suggested next experimental upgrades

1. Compare against the published MedQwen SVD partition and original router in an identical backbone. The code here is **not an exact reproduction** of the original paper.
2. Add a multi-scale mask decoder (e.g., a properly conditioned SAM2 head) with spatial feature fusion; retain the current branch as the low-resolution baseline.
3. Benchmark 2D segmentation and VQA on aligned, medically annotated data, controlling for task sampling, total ranks, FLOPs, and datasets.
4. Probe whether gradients conflict consistently or whether the EMA is noisy; assess causal visual grounding, not only performance gains.
5. Add full dataset validation, independent eval, multi-GPU training, and dynamic batching before publication-scale experiments.
