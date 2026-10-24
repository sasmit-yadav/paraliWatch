# Burn-scar mapping from satellite images (Phase 3)

The forecast says *where fires are likely*. VIIRS fire points say *where something hot was seen
at 1:30 pm*. Neither says *which fields actually burned*. A burned field leaves a dark, charred
scar that a 10-20 m optical satellite (Sentinel-2) can see for days, including fires that VIIRS
missed because they were small, short or under smoke. This module maps those scars.

It is also the project's computer-vision and foundation-model component: we fine-tune NASA/IBM's
**Prithvi-EO-2.0** geospatial foundation model and compare it, fairly, with a U-Net and with the
same architecture trained without pre-training.

## 1. First attempt: weak labels from VIIRS + dNBR (did not work, kept for the record)

Plan: label Sentinel-2 pixels in Punjab as "burned" where the normalised burn ratio dropped
(dNBR = NBR before - NBR after) near VIIRS fire points, then train on those labels.

What we found when checking the labels before training: pixels next to VIIRS detections were
only about **1.1 times** more likely to have a large dNBR than random cropland pixels. Reasons:

- **Harvest itself lowers NBR.** Cutting green paddy to straw changes the near-infrared signal as
  much as a light burn, so dNBR flags most harvested fields, burned or not.
- **Smoke and haze** in October-November bias the "after" image.
- **VIIRS geolocation** is a 375 m pixel; a field of 1-2 ha is a fraction of it, and the hot spot
  may be a neighbour's field.

Labels that are barely better than random teach a model nothing reliable, so we stopped and used a
dataset with **human-checked** burn masks instead. dNBR is kept as the classic remote-sensing
baseline in the Punjab check (section 5).

## 2. Data: NASA HLS Burn Scars

- 804 scenes of 512 x 512 pixels at 30 m from Harmonized Landsat-Sentinel (HLS), contiguous USA
  2018-2021. Burn masks come from the Monitoring Trends in Burn Severity (MTBS) fire perimeters;
  scenes were manually filtered for cloud and missing data and burn presence manually verified.
  The published random split: **540 training, 264 validation**. Mask values: 1 burned, 0 not
  burned, -1 no data.
- 6 bands, the same ones Prithvi was pre-trained on: Blue, Green, Red, NIR (narrow), SWIR 1, SWIR 2.
- 12.0% of valid training pixels are burned.
- Source: `ibm-nasa-geospatial/hls_burn_scars` on Hugging Face (CC-BY-4.0).

It is not Punjab (US wildfires, larger and more varied scars), which is the main limitation and
the reason the Punjab results in section 5 are labelled "indicative".

## 3. Models

| Run | What it is | Parameters (trained) |
| --- | --- | --- |
| `unet` | Classic 4-level U-Net trained from scratch | 1.1 M (all) |
| `prithvi_scratch` | Prithvi-EO-2.0-tiny architecture, **random** weights | 6.2 M (all) |
| `prithvi_frozen` | Pre-trained Prithvi encoder **frozen**, only the decoder trains | 6.2 M (0.58 M) |
| `prithvi_finetune` | Pre-trained Prithvi encoder + decoder, everything trains (encoder at 1/5 of the learning rate) | 6.2 M (all) |

Why these four: `scratch` vs `finetune` isolates what **pre-training** is worth (same
architecture, same data, only the starting weights differ); `frozen` shows how good the
pre-trained features are on their own; `unet` is the strong, simple, non-foundation baseline.

**Prithvi-EO-2.0-tiny** (Apache-2.0): a Vision Transformer (12 layers, width 192, 3 heads, 16 x 16
patches) pre-trained by NASA, IBM and Jülich Supercomputing Centre as a masked autoencoder on a
large global HLS archive: hide 75% of the patches, reconstruct them. The "TL" version also learned
optional time and location embeddings; we do not use them (Punjab dates and places differ from
the US training scenes, and the model was trained to cope with them being absent). The decoder we add takes features from layers 3, 6, 9 and 12,
fuses them and upsamples 16x back to pixels (a small UPerNet-style head).

**Removing the `timm` dependency.** The released model code imports `timm`, which pulls in
`torchvision` and broke on the CPU build of PyTorch. We replaced the two imported pieces with a
40-line local `Block` (pre-norm attention + MLP) whose parameter names match the checkpoint, then
proved it is the same network two ways:
1. a unit test checks our attention against PyTorch's reference `nn.MultiheadAttention`;
2. with the official weights loaded, the masked-autoencoder reconstruction loss on real HLS
   images is **0.043**, against **0.83** for random weights. A wrong layer would destroy that.

## 4. Training setup (identical for every run)

- Random 224 x 224 crops from the 512 x 512 scenes, random 90-degree rotations and flips.
- Inputs standardised with Prithvi's pre-training mean and standard deviation per band.
- Loss = cross-entropy + soft Dice on the burned class; no-data pixels ignored by both
  (unit-tested: they get zero gradient). Dice counters the 12% / 88% imbalance.
- AdamW, weight decay 0.05, one-cycle learning rate (peak 5e-4), batch 8, **1,200 steps**, seed 0.
- CPU only (2 cores). No early stopping and no checkpoint selection on the validation split:
  each model is trained for the same fixed number of steps and evaluated **once**, so the
  validation numbers are not tuned on.
- Evaluation on the 264 full 512 x 512 validation scenes (the ViT handles the larger size by
  interpolating its position embeddings).

## 5. Results on the validation split

From `eval/burn_scars/RESULTS.md` (regenerate with `python -m parali.detect.compare <data_dir>`).
IoU = overlap between predicted and true burned pixels, over all 264 validation scenes.

| Model | Burn IoU (95% CI) | F1 | Precision | Recall | Trained params | CPU minutes |
| --- | --- | --- | --- | --- | --- | --- |
| U-Net (scratch) | 0.700 (0.646-0.749) | 0.823 | 0.799 | 0.849 | 1.09 M | 40 |
| Prithvi architecture, random init | 0.706 (0.645-0.758) | 0.828 | 0.786 | 0.875 | 6.21 M | 24 |
| Prithvi pre-trained, encoder frozen | 0.627 (0.571-0.682) | 0.771 | 0.663 | 0.920 | 0.58 M | 15 |
| **Prithvi pre-trained, fine-tuned** | **0.758 (0.706-0.802)** | **0.863** | **0.830** | **0.898** | 6.21 M | 26 |

Paired bootstrap over scenes (2,000 resamples, same resampled scenes for every model): the
fine-tuned Prithvi model beats the U-Net by **+0.059 IoU** (95% CI +0.030 to +0.089) and the same
architecture without pre-training by **+0.052** (+0.029 to +0.084); it is ahead in 100% of
resamples in both cases.

What this says:

1. **Pre-training is worth about 5 IoU points here.** Same network, same data, same steps; only
   the starting weights differ. That is the cleanest evidence in this project that a foundation
   model helps, and the comparison most write-ups skip.
2. **Frozen features are not enough.** With the encoder frozen the model finds burns (recall 0.92)
   but over-predicts (precision 0.66). Features learned by reconstructing images are general; the
   last step to "burned vs dark soil vs shadow" needs the encoder to adapt.
3. **The U-Net is a strong baseline.** With 6x fewer parameters it matches the un-pre-trained
   transformer; architecture alone did not buy anything. Pre-training did.
4. **Cost.** All four trained on a 2-core CPU in 15-40 minutes. The fine-tuned model segments a
   512 x 512 scene in 0.3 s on CPU.

Choice: **`prithvi_finetune`** (kept at `models/burn_scars/prithvi_finetune.pt`, 25 MB).

## 6. Does it work in Punjab? Not yet, and here is the evidence

We ran the chosen model on Sentinel-2 L2A images (20 m, read straight from the public cloud
copies, same six bands) of a 19 x 28 km tile in Tarn Taran / Amritsar, the densest fire cluster
of the 2025 season. Pre- and post-images 3 days apart (9 and 12 November 2025, both clear).
A field burned between them should be a **new** scar: burned in the second image, not the first.
Check: do the VIIRS fire detections between the two dates fall near new scars more often than
random points (enrichment)?

| Window | VIIRS fires | Model: new-scar area | Model enrichment | dNBR > 0.1 enrichment |
| --- | --- | --- | --- | --- |
| 9 -> 12 Nov 2025 | 24 | 15% | 1.25x | 1.07x |
| 2 -> 12 Nov 2025 | 76 | 38% | 1.15x | 1.00x |

![Burn-scar map, Punjab](images/burn_scars_punjab_20251109_20251112.png)

Reading it honestly:

- On the single image, the model labels **81% (9 Nov) to 96% (12 Nov) of the tile as burned**.
  After harvest, Punjab is a landscape of brown stubble and bare soil; to a model trained on US
  wildfire scars surrounded by green or grey vegetation, all of it looks like a scar.
- The "new scar" differences are blobby, regional patches that follow haze and lighting changes
  between the two dates, not field boundaries. Enrichment of 1.15-1.25x is weak.
- Classic dNBR does no better (1.0-1.07x).

**Conclusion: strong in-domain (0.76 IoU), not transferable to post-harvest Punjab without local
labels.** This is a domain-shift result, and the model is therefore *not* shown on the dashboard
as fact. What would fix it, in order of cost:

1. Hand-label 200-300 fields (burned / harvested-not-burned / standing crop) on clear
   Sentinel-2 or PlanetScope images in two districts, and fine-tune from the current checkpoint.
   The frozen-encoder result says the backbone alone is not enough, so the encoder must train too.
2. Give the model time: a pair or short series of images as input (Prithvi supports multiple
   frames), so it learns the *change* (fresh char is black in every band) instead of the look.
3. Use the field-level labels to measure VIIRS's miss rate, the original goal.

## 7. Reproduce

```bash
uv sync --group vision
# data: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars (extract training/ and validation/)
# weights: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL (Prithvi_EO_V2_tiny_TL.pt)
python -m parali.detect.train <hls_dir> <Prithvi_EO_V2_tiny_TL.pt>     # 4 runs, ~2 h on 2 CPU cores
python -m parali.detect.compare <hls_dir>                               # bootstrap table
python -m parali.detect.punjab prithvi_finetune 2025-11-09 2025-11-12   # Punjab check + figure
```

Code: `parali/detect/` (`models.py`, `prithvi_mae.py` vendored from the official release with the
`timm` dependency removed, `train.py`, `compare.py`, `sentinel.py`, `punjab.py`).

Licences: Prithvi-EO-2.0 weights Apache-2.0; HLS Burn Scars CC-BY-4.0; Sentinel-2 data free and
open (Copernicus).
