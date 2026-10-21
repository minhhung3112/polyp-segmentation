# CMUNeXt-Plus: polyp segmentation research branch (UNVALIDATED)

**Research prototype, not a proven accuracy improvement.** The original `cmunext` architecture is unchanged; `cmunext_plus` is newly registered. The new `train_plus.py` does not alter `train.py` or its legacy outputs.

## Notebook audit, 21 October 2026

Notebook `model_cmunext_ (2)(1).ipynb` trains `--model cmunext --seeds 42 --multiseed --batchsize 8`, with Structure Loss, 100 epochs, 352×352 images, Adam lr=1e-4 and multiscale (0.75, 1.0, 1.25). Its final best checkpoint is epoch 94; mDice / mIoU on the five test datasets:

| Test dataset | mDice | mIoU |
|---|---:|---:|
| CVC-300 | .7870 | .6958 |
| CVC-ClinicDB | .8582 | .7987 |
| CVC-ColonDB | .6925 | .6128 |
| ETIS | .5939 | .5256 |
| Kvasir | .8856 | .8248 |
| Unweighted mean | .7634 | .6915 |

**Evaluation caveat:** `train.py` uses `TestDataset/CVC-ClinicDB` for validation and selects `best.pth` using that set, then tests on the same set. Accordingly, those results are NOT an uncontaminated held-out cross-dataset comparison. Pretraining from that checkpoint also inherits this problem. The current notebook's setup cells contain repository clone, dependencies, and training instructions, not PDFs with paper tables. CMUNeXt's own experiments are on different medical segmentation tasks, so do not compare their metrics directly with polyp Dice.

## Papers reviewed and hypotheses

- Tang et al., **CMUNeXt** (arXiv:2308.01239): large-kernel depthwise conv, inverted bottleneck, skip fusion. Original backbone retained for checkpoint transfer.
- Oktay et al., **Attention U-Net** (arXiv:1804.03999): learn to select useful encoder skip features. We add decoder-conditioned spatial gates, initialized as an **identity residual** to keep pretrained predictions unchanged before training.
- Fan et al., **PraNet** (arXiv:2006.11392): polyp boundary challenges, multi-scale and deep supervision. We add auxiliary prediction heads and keep weighted BCE + IoU Structure Loss as the starting point.
- Dong et al., **Polyp-PVT** (arXiv:2108.06932): fuse shallow and deep features selectively. We add an economical 3-branch dilated depthwise context pyramid at the encoder bottleneck.
- MISNet (DOI 10.1016/j.engappai.2024.109467): contextual fusion and boundary attention as independent evidence supporting ablation directions.

These are *inspirations*, not faithful implementations of the cited networks. Superiority/new contribution is not established by architecture description alone.

## Improvements in `cmunext_plus`

1. Same original CMUNeXt base encoder, upsampling and Skip-Fusion weight names. Directly load most original `cmunext` checkpoint weights.
2. Dilated depthwise context pyramid at bottleneck (dilations 1,2,3), with learned residual gain initialized to zero.
3. Four decoder-conditioned spatial attention gates for skip inputs, each with zero-initialized residual gain.
4. Three auxiliary segmentation heads + the original final segmentation head. Auxiliary loss weights: 0.15, 0.20, 0.30, final 1.0.
5. Standalone experimental trainer that creates fixed 90:10 train/validation subset from `TrainDataset`, keeps test sets out of checkpoint selection, resizes binary masks via nearest neighbor, uses AdamW, cosine LR, mixed precision, and early stopping.

Potential tradeoffs: additional parameters, memory and FLOPs; no FPS or Dice gain is yet measured. T4 batch 8 may need `--batchsize 4` if OOM.

## Colab: warm-start existing best weights (quick exploratory fine-tune)

```python
!git clone -b research/cmunext-plus https://github.com/minhhung3112/polyp-segmentation.git
%cd polyp-segmentation
!pip install -r requirements.txt
# Upload your old ./snapshots/cmunext/best.pth to Colab first.
!python train_plus.py --model cmunext_plus --init_weights ./snapshots/cmunext/best.pth --batchsize 8 --epochs 60 --seeds 42
```

**IMPORTANT:** Existing base checkpoint saw all original TrainDataset (including what would become new validation images), and was selected using CVC-ClinicDB test images. Warm-starting with it is valid for experimentation and continuity, **NOT a fair paper-style validation or held-out test claim**.

New best weights: `snapshots/cmunext_plus/best.pth`. To continue a **same-architecture** interrupted run, use the *periodic* checkpoint `--resume snapshots/cmunext_plus/cmunext_plus-20.pth` (and increase `--epochs` if needed). **Never use `--resume` to switch from CMUNeXt to CMUNeXt-Plus**; use `--init_weights` instead.

## Fair research protocol (recommended to assess actual improvement)

1. Train the baseline and Plus model **from scratch**, same split seed and preprocessing, without `--init_weights`:

```bash
python train_plus.py --model cmunext      --epochs 100 --lr 0.0001 --seeds 42
python train_plus.py --model cmunext_plus --epochs 100 --lr 0.0001 --seeds 42
```

2. Use `--fast` only for quick smoke runs, applied consistently to both models.
3. Train both again on seeds `3407` and `1234`, report mean and standard deviation; select hyperparameters on validation only.
4. After freezing choices, run final evaluation exactly once per final experiment:

```bash
python train_plus.py --model cmunext_plus --epochs 100 --lr 0.0001 --seeds 42 --eval_test
```

`--eval_test` trains again before evaluating. Alternatively, use the existing `test.py` with the chosen Plus checkpoint, if its CLI supports `--model cmunext_plus`. Results from legacy notebook are not directly equivalent to this fixed data-splitting protocol.

Recommended ablations: (a) CMUNeXt baseline, (b) + multi-scale context only, (c) + skip gates only, (d) + deep supervision only, (e) full Plus. Report test Dice, IoU, parameters, MACs/FLOPs, latency on the SAME GPU and image resolution. **Use patient/video-disjoint splits when metadata is available**; image-level random split alone may still share near-duplicate video frames.

## Limitations

- New code has not been trained on the real dataset; expected improvements are hypotheses.
- Original `Evaluator.evaluate_all` applies per-image min/max normalization before thresholding; Plus validation approximates that preprocessing on 352×352 masks, while test evaluates after upsampling to original dimensions, so metrics still differ slightly.
- The published CMUNeXt paper reports experiments on different clinical images. A valid direct comparison requires rerunning an identical baseline on the same polyp dataset.
- If performance is worse, revert to `--model cmunext`; the old model is untouched.