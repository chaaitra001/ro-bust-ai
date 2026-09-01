# Ro-Bust AI: AIGC Image Detector

**TikTok TechJam 2026 — Track 5 (AI-Generated Image Detection)**
**Team: smart cookies**

A dual-branch (spatial + frequency-domain) image classifier that predicts whether an image is real or AI-generated, trained to stay accurate after realistic post-processing (JPEG/WEBP/PNG re-encoding, blur, resizing, noise, colour jitter, cropping).

**Held-out test results:** 91.6% accuracy / 0.973 AUC on 13,500 clean images, staying above 0.90 AUC across every one of the nine robustness conditions tested. Full numbers below.

## Project overview

The model combines two complementary views of an image:

- **Spatial branch** — a pretrained ResNet-18 (via `timm`) extracting texture/edge-level visual features.
- **Frequency branch** — a small custom CNN operating on the image's log-magnitude FFT spectrum, catching statistical artifacts that aren't obvious from pixels alone.

Features from both branches are concatenated and passed through a small classifier head. Full architecture: **11.3M parameters** — far under the competition's 2B-parameter limit.

Training data combines three sources — [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set), and a [Midjourney/DALL-E/Stable Diffusion set](https://huggingface.co/datasets/julienlucas/midjourney-dalle-sd-dataset) — pooled and re-split ourselves at a consistent 70% train / 30% test ratio (stratified by class), with a `WeightedRandomSampler` ensuring the much larger CIFAKE source doesn't drown out the other two during training.

Robustness is addressed at the training-data level, not just at evaluation time: every training image is always re-encoded through a randomly-chosen file format (identity/JPEG/WEBP/PNG) before any other augmentation is applied, specifically to prevent the model from learning "has JPEG compression artifacts" as a shortcut for "real" — a failure mode we found and fixed during development (see **Limitations & reflection** below).


## Setup and installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv && source venv/bin/activate   
pip install -r requirements.txt
```

A CUDA-capable GPU is optional — `predict.py` and the notebook both fall back to CPU automatically (slower, but works). Training itself was run on a Colab T4 GPU.

## Steps to reproduce our results

**Training (full pipeline, run in Google Colab):**

1. Open `notebook/roBUST_ai.ipynb` in [Google Colab](https://colab.research.google.com/).
2. `Runtime > Change runtime type > GPU (T4)`.
3. Run Section 1 (Setup).
4. Run Section 2 (Download datasets):
   - CIFAKE (2a) prompts a one-time Kaggle login on first run — 60,000 train + 20,000 test images, pooled and capped at 18,000/class in Section 4.
   - SID_Set (2b) requires a Hugging Face token stored as a Colab secret named `HF_TOKEN` (Colab sidebar → 🔑 Secrets) — streams 1,500 real + 1,500 fake examples.
   - Midjourney/DALL-E/SD (2c) downloads automatically, no login needed — 5,000 train + 1,000 test examples.
5. Run Sections 3–4 (transforms, dataset build, model setup). This builds the 70/30 stratified split; our run produced:

   | Source | Train+Val (70%) | (val only) | Test (30%) |
   |---|---|---|---|
   | CIFAKE | 25,200 | 3,780 | 10,800 |
   | SID_Set | 2,100 | 316 | 900 |
   | Midjourney+ | 4,200 | 630 | 1,800 |
   | **Combined** | **31,500** | **4,726** | **13,500** |

6. Run Section 5 (training) — trains with early stopping (patience 4 epochs) on validation AUC; the best checkpoint is saved to `model.pt` as training progresses. Our run used all 20 available epochs (validation AUC was still improving at epoch 20, so early stopping never triggered) and finished at **val AUC 0.9752**.
7. Run Section 5b (training curves) to generate `training_curve.png`.
8. Run Section 6 (evaluation) to generate `robustness_table.json`, then Section 6b for `robustness_heatmap.png`.
9. Run Section 7 to try inference on your own uploaded images and produce a `results.json`.

**Inference only (standalone script, run anywhere with the requirements installed):**

```bash
python predict.py --input_dir path/to/images --model_path model/model.pt --output results.json
```

- `--input_dir`: folder of images to score (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`, `.tif`/`.tiff`).
- `--output`: where to write the JSON (default `results.json`).
- `--recursive`: also search subfolders.
- `--device`: force `cuda` or `cpu` (defaults to GPU if available).

Output format — one entry per successfully-read image:

```json
[
  {"image_path": "path/to/images/photo1.jpg", "pred": 0.0421},
  {"image_path": "path/to/images/ai_art_2.png", "pred": 0.9788}
]
```

`pred` is the model's confidence (0–1) that the image is AI-generated. Unreadable files are skipped with a warning printed to stderr rather than stopping the run.

## Robustness Evaluation Summary

Evaluated on the full 13,500-image held-out test set (never seen during training or validation):

| Condition | Accuracy | AUC |
|---|---|---|
| Clean | 0.916 | 0.973 |
| JPEG (q=30) | 0.894 | 0.957 |
| WEBP (q=70) | 0.910 | 0.969 |
| PNG round-trip | 0.916 | 0.973 |
| Gaussian blur (σ=2.0) | 0.844 | 0.919 |
| Resize 0.25× | 0.826 | 0.907 |
| Gaussian noise (σ=0.10) | 0.861 | 0.931 |
| Colour jitter | 0.908 | 0.968 |
| Center crop (80%) | 0.879 | 0.950 |

![Robustness heatmap](results/robustness_heatmap.png)

Format re-encoding (JPEG/WEBP/PNG) and colour jitter barely move the numbers — AUC stays within ~0.02 of clean in every case. The two conditions that actually hurt are heavy blur and aggressive downscaling (0.25×), which destroy fine-grained detail outright rather than just re-compressing it; even there, AUC never drops below 0.90.

**Per-source breakdown (clean condition):** CIFAKE 92.0% acc / 0.977 AUC (n=10,800), SID_Set 80.2% acc / 0.949 AUC (n=900), Midjourney+ 94.8% acc (n=1,800; AUC undefined for this run — see Limitations). SID_Set is our weakest source despite being real photographic content, which is the clearest signal for where to prioritise additional data.

![Training curve](results/training_curve.png)

## Limitations & reflection

**What we'd improve given more time:**

- **SID_Set underperformance.** At 80.2% clean accuracy, SID_Set is noticeably weaker than CIFAKE (92.0%) or Midjourney+ (94.8%). We only streamed 1,500 examples/class from a 140GB dataset — more samples from this source specifically is a likely fix, and with the weighted sampler already balancing training exposure, additional SID_Set data should translate directly into gains rather than being drowned out by CIFAKE's larger pool.
- **A data-partition quirk we found late.** Our per-source robustness breakdown reported an undefined (NaN) AUC for Midjourney+ under the clean condition, because that particular 1,800-image test slice ended up containing only one class for that check. The combined test-set numbers above are unaffected, but we'd fix the per-source stratification to guarantee both classes are represented in every source-level breakdown, not just the combined one.
- **Generator coverage.** CIFAKE's fakes come from an older/simpler generator family; despite the weighted sampler balancing training exposure, the model has still seen far fewer examples of the newest, most photorealistic generators than we'd like — CIFAKE alone (36,000 images) still outnumbers SID_Set and Midjourney+ combined (9,000). More diverse, modern-generator data is the single biggest lever left.
- **Format-bias shortcut, only partially verified.** We identified and fixed a specific failure mode where the model could learn to associate JPEG compression artifacts with "real" (since real-photo datasets skew JPEG-sourced and generator datasets skew PNG-sourced) instead of learning genuine synthetic-vs-real cues. We addressed this by forcing every training image through a randomly-chosen file format every epoch — the resulting robustness table shows JPEG/WEBP/PNG all landing within 0.02 AUC of the clean baseline, consistent with the fix working, but we haven't run a controlled ablation (with vs. without the fix) to isolate exactly how much of that came from this change specifically.
- **Threshold calibration.** Predictions are thresholded at a fixed 0.5. We didn't have time to calibrate this against the validation set, which likely leaves some accuracy on the table, particularly for reducing false positives on real images.
- **Explainability.** The frequency branch's contribution to a given prediction isn't currently interpretable — we'd like to add a way to indicate whether a prediction leaned more on spatial or frequency-domain evidence.
- **Robustness beyond single transformations.** We test each transformation independently; real-world redistribution often stacks several (e.g. resize, then re-compress, then re-upload). We'd extend the robustness table to cover stacked/combined transformations at multiple severity levels.


## Team member contributions

| Member | Contributions |
|---|---|
| Guda Chaaitra Joseph | Model Architechture and Development |
| Kamarudeen Hana Fathima | Research & Documentation |
| Sundar Subramanian Ayngara Sudha | Research & Documentation |
| Balaji Abarnasri | Research & Documentation |
| Pandiarajan Sreenithi | Research & Documentation |

## License / acknowledgements

Built on CIFAKE, SID_Set, and the Midjourney/DALL-E/Stable Diffusion dataset (see links above); ResNet-18 weights via `timm`, pretrained on ImageNet.
