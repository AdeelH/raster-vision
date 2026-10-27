# Raster Vision vs. TorchGeo: Feature Comparison

*Part of the Raster Vision discontinuation review. Research doc; it does not make the decision. Concrete bugs and maintenance items are tracked in [fixable-issues.md](./fixable-issues.md). The option of replacing RV's learners with TorchGeo is analysed in [torchgeo-backend-option.md](./torchgeo-backend-option.md).*
*Prepared: 2026-10-27*

## Versions compared

| | Raster Vision | TorchGeo |
|---|---|---|
| Code inspected | `master` @ `fed86fb9` (`0.31.2-dev`) | `main` @ `35b06a1` (`0.11.0.dev1`) |
| Latest PyPI release | 0.31.1 (2024-08-30) | 0.10.x (0.11 in development) |

Other sources: the TorchGeo docs page [*Related Libraries*](https://github.com/torchgeo/torchgeo/blob/main/docs/user/alternatives.rst) and its [`features.csv`](https://github.com/torchgeo/torchgeo/blob/main/docs/user/metrics/features.csv), GitHub issues, PRs, and code search, pypistats.org, and the TerraTorch and GeoAI READMEs.

---

## 1. Summary

1. **TorchGeo covers most of RV's library building blocks.** That includes geospatial data loading, reprojection, rasterizing vector labels, ROI-restricted window sampling, segmentation, detection, classification, and regression tasks, multi-GPU training, a config-driven CLI, and ONNX export (through Lightning). It also has many things RV lacks: 151 datasets, 137 sets of pretrained weights and geospatial foundation models, self-supervised learning, change detection, instance segmentation, time-series models and samplers, and Kornia GPU augmentations.

   However, its polished paths center on its *built-in* datasets. RV's core use case is "here are some GeoTIFFs and labels; give me a training and inference pipeline." For that use case, TorchGeo's building blocks work but take noticeably more hand-wiring, and for object detection we hit two bugs. See [§4.10](#410-hands-on-test-object-detection-from-arbitrary-geotiffs-in-torchgeo).

2. **RV's distinctive value is the "last mile": going from a trained model to georeferenced output over whole scenes.** TorchGeo returns per-chip tensors and leaves the rest to the user. Specifically, TorchGeo has no built-in way to:
   - stitch overlapping window predictions for a whole scene into one GeoTIFF. Work on this is in progress; see §4.1.
   - turn segmentation output into polygons, including the building footprint regularizer.
   - merge object detections across overlapping windows and write them as GeoJSON.
   - make a chip-classification grid over a scene and export it as GeoJSON.
   - package the model, its preprocessing, and its config into one artifact (a "model bundle") with a one-line `predict` CLI.
   - compute per-scene evaluation reports.

3. **RV's second distinctive area is workflow and orchestration.** This covers:
   - the analyze → chip → train → predict → eval → bundle pipeline
   - dataset-wide channel statistics (`StatsAnalyzer`) and `StatsTransformer`
   - splitting commands for parallel runs, and runners for AWS Batch and SageMaker
   - backward-compatible config upgraders
   - a pluggable file system for S3, HTTP, and GDAL VSI

   TorchGeo relies on Lightning for orchestration and has no equivalent of the pipeline, the stats step, or the cloud runners.

4. **Several of these gaps are closing, or are covered next door.** TorchGeo merged a COG writer in [torchgeo/torchgeo#3666](https://github.com/torchgeo/torchgeo/pull/3666) as step 1 of 3 toward a tiled-inference callback ([torchgeo/torchgeo#3214](https://github.com/torchgeo/torchgeo/pull/3214)). TerraTorch, which is built on TorchGeo, already does tiled inference with blending. GeoAI covers raster-to-vector conversion and chip generation.

5. **Few people use RV now.** It gets about 70 PyPI downloads a week, compared with about 7.6k for TorchGeo and about 2.6k for TerraTorch. The newest release is from August 2024. Five issues have been opened since January 2025, and four of them came from one user. Public code that imports `rastervision` is almost all Azavea-owned and written against the pre-0.20 API.

---

## 2. Project health

| Metric | Raster Vision | TorchGeo | Source |
|---|---|---|---|
| GitHub stars / forks | 2,242 / 397 | 4,141 / 571 | GitHub API; TorchGeo `github.csv` (Aug 2026) |
| Contributors | 32 | 131 | TorchGeo `github.csv` |
| Open issues | 45 | 192 | same |
| Last release | 0.31.1 (Aug 2024) | regular; 0.11 milestone active | PyPI |
| PyPI downloads, last week | 69 | 7,640 | pypistats.org (mirrors excluded) |
| PyPI downloads, last month | 240 | 39,822 | pypistats.org |
| PyPI downloads, all time | ~298k | ~3.4M | TorchGeo `downloads.csv` |
| Test coverage | 90% | 100% | TorchGeo `github.csv` |
| License | Apache-2.0 | MIT | |
| Built-in datasets | 0 | 151 | TorchGeo `features.csv` |
| Geospatial pretrained weights | 6 (model-zoo bundles) | 137 | TorchGeo `features.csv` |

Notes:
- TerraTorch, which is built on TorchGeo, is at about 2.6k downloads a week, roughly 37 times RV.
- RV's documented model zoo URL for 0.31 (`.../model-zoo-0.31/...`, set in [docs/conf.py](../docs/conf.py)) returns 404. The 0.30 bundles are still online.

---

## 3. Design differences

These shape what "equivalent feature" means in each library.

| Aspect | Raster Vision | TorchGeo |
|---|---|---|
| Core unit | **Scene**: one raster source, an optional label source, and an optional AOI. Windows use the scene's pixel coordinates. | **GeoDataset**: a spatial and temporal index (GeoPandas) over many files, queried with world-coordinate slices `[x, y, t]`. Datasets compose with `&` (intersection) and `\|` (union). |
| Reprojection | Vectors are reprojected into the raster's pixel space. Rasters are not warped. `MultiRasterSource` aligns sources by mapping windows through map coordinates and resizing, which is **only approximately correct** (measured in [§4.11](#411-hands-on-test-multirastersource-alignment-accuracy)). | Every file is warped on the fly to a common CRS and resolution (`WarpedVRT` + `rasterio.merge`). |
| Mosaics and many tiles | One scene per raster, or per VRT or STAC stack built by the user. | Built in: any number of files is indexed and mosaicked on read. |
| Primary interface | Declarative pipeline (Pydantic configs → CLI → runners). Can also be used as a library since 0.20. | Library first: datasets, samplers, datamodules, and Lightning tasks. LightningCLI covers YAML and the command line. |
| Training engine | Custom `Learner` (plain PyTorch loop with DDP). | PyTorch Lightning. |
| Augmentation | Albumentations, on CPU and per sample. | Kornia, on GPU and per batch, plus geospatial transforms such as spectral indices and SAR filters. |
| Scope | The whole lifecycle: data prep → train → predict → evaluate → deploy. | Data, models, and training. Prediction output is left to the user. |

---

## 4. Feature comparison

Key: ✅ supported · 🚧 partial or in progress · ❌ not supported. **RV-only** marks features with no TorchGeo equivalent.

### 4.1 Inference and outputs (RV's strongest area)

| Feature | RV | TorchGeo | Notes |
|---|---|---|---|
| Sliding-window prediction over a whole scene | ✅ | 🚧 | TorchGeo's `GridGeoSampler` and `GriddedPatchSampler` produce the windows, and `predict_step` returns `probabilities`, `bounds`, and `transform`. Stitching is left to the user. |
| Stitching overlapping predictions into one georeferenced raster | ✅ | 🚧 | RV averages overlapping windows (smooth predictions) and can drop edge pixels (`crop_sz`, including `'auto'`). TorchGeo merged `GeoTIFFWriter` ([torchgeo/torchgeo#3666](https://github.com/torchgeo/torchgeo/pull/3666), v0.11). Blending is still an open PR ([torchgeo/torchgeo#3667](https://github.com/torchgeo/torchgeo/pull/3667)), and the `TiledInferenceCallback` PR is not yet opened. The umbrella PR [torchgeo/torchgeo#3214](https://github.com/torchgeo/torchgeo/pull/3214) closed while being split up. That design covers classification only, with regression listed as a TODO. |
| Writing class-score rasters alongside discrete labels | ✅ | 🚧 | RV: `smooth_output=True`. TorchGeo: the user writes them. |
| Segmentation → polygons (GeoJSON) | ✅ **RV-only** | ❌ | RV `PolygonVectorOutputConfig`: per-class `mask_to_polygons`, with optional denoising and probability threshold. |
| Building footprint extraction (rectangle-kernel regularization) | ✅ **RV-only** | ❌ | RV `BuildingVectorOutputConfig` / `mask_to_building_polygons`. GeoAI has similar regularization utilities. |
| Object detection over a scene: cross-window box merging (NMS) → GeoJSON in map coordinates | ✅ **RV-only** | ❌ | RV `ObjectDetectionPredictOptions` (`merge_thresh`, `score_thresh`). TorchGeo's `ObjectDetection.predict_step` returns per-chip pixel boxes only. |
| Chip classification over a scene → GeoJSON grid of labeled cells | ✅ **RV-only** | ❌ | |
| Writing RGB-colored prediction rasters | ✅ | ❌ | Minor convenience. |
| Self-contained **model bundle** (weights, ONNX, config, transforms, class config, channel order, stats) | ✅ **RV-only** | ❌ | TorchGeo has Lightning checkpoints and weight enums. Normalization and band order must be rebuilt by the user. |
| One-command inference: `rastervision predict <bundle> <image> <out>` and `predict_scene` | ✅ **RV-only** | ❌ | TorchGeo has `torchgeo predict` (LightningCLI), but its outputs are tensors, not files. |
| ONNX export and ONNX-runtime inference | ✅ | 🚧 | RV exports `model.onnx` into the bundle (`Learner.export_to_onnx`) and can run inference from it. TorchGeo users can call Lightning's `to_onnx`, but there is no documented workflow. |

Code references: [semantic_segmentation_label_store_config.py](../rastervision_core/rastervision/core/data/label_store/semantic_segmentation_label_store_config.py), [vectorization.py](../rastervision_core/rastervision/core/data/utils/vectorization.py), [object_detection_config.py](../rastervision_core/rastervision/core/rv_pipeline/object_detection_config.py), [semantic_segmentation_config.py](../rastervision_core/rastervision/core/rv_pipeline/semantic_segmentation_config.py), [core/cli.py](../rastervision_core/rastervision/core/cli.py), [learner.py](../rastervision_pytorch_learner/rastervision/pytorch_learner/learner.py).

### 4.2 Labels and vector data

| Feature | RV | TorchGeo | Notes |
|---|---|---|---|
| Rasterizing vector labels on the fly (semantic segmentation) | ✅ | ✅ | RV `RasterizedSource`; TorchGeo `VectorDataset`. |
| Vector → boxes (detection) and vector → instance masks | ✅ boxes | ✅ boxes and masks | TorchGeo `VectorDataset(task=...)`. |
| Inferring chip-classification labels from polygons (IoA threshold, intersection-over-cell, background class, min-class tie-break) | ✅ **RV-only** | ❌ | RV `ChipClassificationLabelSource`. |
| Vector transformers: per-class buffering of points and lines into polygons | ✅ **RV-only** | ❌ | RV `BufferTransformer`, useful for roads and point labels. TorchGeo would need a subclass or preprocessing. |
| Inferring class IDs from feature properties (filters, name mapping, default class) | ✅ | 🚧 | RV `ClassInferenceTransformer`. TorchGeo: `label_name` or overriding `get_label()`. |
| Shifting vectors to correct misregistration | ✅ **RV-only** | ❌ | RV `ShiftTransformer`. Niche. |
| `all_touched` and background class for rasterization | ✅ | 🚧 | |
| OSM labels | 🚧 (tutorial; user fetches GeoJSON) | ✅ | TorchGeo `OpenStreetMap` dataset queries Overpass directly. |
| STAC label-extension catalogs | ✅ | ❌ | RV [stac.py](../rastervision_core/rastervision/core/utils/stac.py) reads zipped STAC label catalogs. Niche. |
| GeoJSON helpers (pixel ↔ map coordinates, splitting multipolygons, simplifying, buffering, filtering to a window) | ✅ | 🚧 | RV [geojson.py](../rastervision_core/rastervision/core/data/utils/geojson.py). Users of either library can use GeoPandas directly. |

### 4.3 Raster input

| Feature | RV | TorchGeo | Notes |
|---|---|---|---|
| GeoTIFF and anything GDAL reads | ✅ | ✅ | |
| Cloud storage (S3, HTTP, GDAL VSI) | ✅ (pluggable `FileSystem`) | ✅ (`/vsi*` paths, including listing) | RV also downloads and caches files and writes outputs to S3. |
| Automatic reprojection and resampling to a common grid | 🚧 | ✅ | |
| Combining rasters of different resolutions into one input | ✅ (`MultiRasterSource`; was misaligned before the fix in §4.11) | ✅ (`IntersectionDataset`) | |
| Mosaicking many tiles | ❌ (needs a VRT or STAC) | ✅ | |
| xarray / STAC item collections (stackstac) as input | ✅ (`XarraySource`) | 🚧 | TorchGeo `XarrayDataset` (experimental) reads xarray files, and some datasets query STAC (OpenAerialMap). There is no generic STAC-to-dataset path. The TorchGeo comparison table rates RV STAC ✅ and TorchGeo 🚧. |
| Temporal stacks | 🚧 (`TemporalMultiRasterSource`, `XarraySource`) | ✅ | TorchGeo adds a time index and six temporal samplers (v0.10). |
| Band selection, channel order, nodata | ✅ | ✅ | |

### 4.4 Preprocessing and normalization

| Feature | RV | TorchGeo | Notes |
|---|---|---|---|
| Computing per-channel statistics over a dataset (sampled) | ✅ **RV-only** | ❌ | RV `StatsAnalyzer`, stored for reuse and bundled with the model. TorchGeo datamodules hard-code mean and std for built-in datasets. |
| Stats-based conversion to uint8 (`StatsTransformer`), min-max, cast, NaN replacement, reclassification, RGB→class | ✅ | 🚧 | TorchGeo normalizes through Kornia and has percentile normalization for plotting. The rest is left to the user. |
| Spectral indices (NDVI, NDWI, EVI, and others), SAR Lee filter | ❌ | ✅ | |
| GPU batch augmentation | ❌ | ✅ (Kornia) | |

### 4.5 Sampling and splits

| Feature | RV | TorchGeo | Notes |
|---|---|---|---|
| Random and sliding-window sampling | ✅ | ✅ | |
| AOI / ROI restriction | ✅ (per-scene AOI polygons, `within_aoi`, efficient AOI sampling) | ✅ (`roi` polygon, `roi_split`) | |
| Variable or random window sizes (`size_lims`, `h_lims`, `w_lims`) | ✅ | ❌ | |
| Rejecting windows by nodata fraction (`nodata_threshold`) | ✅ **RV-only** | ❌ | |
| Class-aware sampling for segmentation (`target_class_ids`, `negative_survival_prob`) and positive/negative window ratio for detection (`neg_ratio`, `ioa_thresh`) | ✅ **RV-only** | ❌ | Useful for rare-class problems. |
| Padding control (`pad_direction`) | ✅ | 🚧 | |
| Spatial train/val/test splitting | 🚧 (by scene or AOI) | ✅ | TorchGeo: `random_bbox_assignment`, `random_grid_cell_assignment`, `roi_split`, `time_series_split`. |
| Temporal sampling | ❌ | ✅ | |

Code reference: [chip_options.py](../rastervision_core/rastervision/core/rv_pipeline/chip_options.py).

### 4.6 Tasks, models, and training

| Feature | RV | TorchGeo |
|---|---|---|
| Semantic segmentation | ✅ (DeepLabV3 or a Torch Hub model) | ✅ (U-Net, DeepLabV3+, FCN, UPerNet, SegFormer, DPT with any timm or SMP encoder) |
| Object detection | ✅ (Faster R-CNN) | ✅ (Faster R-CNN, FCOS, RetinaNet) |
| Classification (chip or scene) | ✅ | ✅ (multiclass and multilabel) |
| Regression | 🚧 (learner only, not in the pipeline) | ✅ (image-level and pixel-wise) |
| Instance segmentation | ❌ | ✅ (Mask R-CNN) |
| Change detection | ❌ | ✅ (U-Net, FC-Siam, ChangeViT, BTC, and others) |
| Spatio-temporal tasks | ❌ | ✅ (U-TAE, L-TAE, ConvLSTM) |
| Self-supervised pretraining | ❌ | ✅ (BYOL, MoCo, SimCLR, MAE) |
| Geospatial foundation models | ❌ | ✅ (DOFA, Scale-MAE, CROMA, Panopticon, Copernicus-FM, Presto, OlmoEarth, Aurora, TESSERA, and more) |
| Pretrained geospatial weights | 6 bundles | 137 |
| Adapting pretrained weights to more than 3 channels | ✅ | ✅ |
| Custom models and losses without forking | ✅ (Torch Hub `external_def`) | ✅ (subclass the task, or LightningCLI `class_path`) |
| Multi-GPU | ✅ (DDP) | ✅ (Lightning: DDP, FSDP, multi-node) |
| Mixed precision, early stopping, many loggers | ❌ / limited (TensorBoard only) | ✅ (Lightning) |
| Class loss weights, ignore index | ✅ | ✅ |
| Data-sample visualization for arbitrary user data (channel display groups) | ✅ | 🚧 (`plot()` is mostly per-dataset) |

### 4.7 Evaluation

| Feature | RV | TorchGeo |
|---|---|---|
| Training and validation metrics | ✅ | ✅ (torchmetrics) |
| **Evaluating predictions against labels per scene and in aggregate** (JSON per-class precision, recall, F1, confusion matrix) | ✅ **RV-only** | ❌ (chip-level metrics only) |

### 4.8 Workflow, orchestration, and deployment

| Feature | RV | TorchGeo | Notes |
|---|---|---|---|
| End-to-end declarative pipeline (analyze → chip → train → predict → eval → bundle) | ✅ **RV-only** | ❌ | |
| CLI | ✅ (`rastervision run`, `predict`) | ✅ (`torchgeo fit/validate/test/predict`, LightningCLI YAML) | |
| Pre-chipping to disk for reuse | ✅ | ❌ (reads on the fly) | |
| Splitting commands across N parallel jobs (`--splits`) | ✅ **RV-only** | ❌ | |
| AWS Batch runner, SageMaker runner and pipelines, CloudFormation templates | ✅ **RV-only** | ❌ | Lightning can run on SLURM and other cluster environments, but nothing AWS-specific. |
| Docker images | ✅ | ❌ (pip, conda, spack) | |
| Backward-compatible config upgraders | ✅ **RV-only** | ❌ | |
| Plugin system and cookiecutter template | ✅ | 🚧 (Python subclassing) | |
| QGIS plugin | ❌ (`raster-vision-qgis` exists but targets the pre-0.20 API) | ❌ | |

### 4.9 Data and ecosystem

| Feature | RV | TorchGeo |
|---|---|---|
| Built-in benchmark datasets with download and datamodules | ❌ | ✅ (151) |
| Embedding products (Earth embeddings, TESSERA, and others) | ❌ | ✅ |
| I/O profiling and benchmarking (`IOBench`) | ❌ | ✅ |
| Downstream frameworks built on it | — | TerraTorch (IBM/NASA), used by GeoAI and others |

### 4.10 Hands-on test: object detection from arbitrary GeoTIFFs in TorchGeo

**Question.** RV's main purpose was turning a set of arbitrary GeoTIFFs plus labels into a training and inference pipeline. How easy is that in TorchGeo, using object detection as the example?

**Setup.**
- TorchGeo `main` (0.11.0.dev1), torch 2.14 on CPU, in a scratch venv.
- Data: 3 synthetic 600×600 RGB GeoTIFFs in EPSG:32617 at 0.5 m, and one GeoJSON with 75 boxes in **EPSG:4326** carrying a `class_id` property.
- Plan: train on 2 scenes, validate on 1, then predict on that held-out scene.

**What worked well**
- `RasterDataset(paths=dir)` and `VectorDataset(paths=dir, task='object_detection', label_name='class_id')` work on arbitrary files without subclassing.
- `images & labels` reprojected the labels from 4326 to the raster CRS automatically, and took the resolution from the GeoTIFFs.
- `RandomPatchSampler` and `GriddedPatchSampler`, together with `collate_fn_detection`, produced correct `image`, `bbox_xyxy`, and `label` batches.
- `ObjectDetection(model='faster-rcnn', backbone='resnet18', num_classes=3)` trained through Lightning and reported `val_map`.
- The final working script is about 35 lines of Python (shown below).

**Friction and bugs found, in the order we hit them**

| # | Issue | Severity | Workaround |
|---|---|---|---|
| 1 | **Bug:** `VectorDataset` returns `label` as `int32`, so training crashes inside torchvision: `TypeError: target labels must of int64 type`. The obvious pipeline fails out of the box. | High (blocks training) | A sample transform that casts to `.long()`. |
| 2 | `ObjectDetection` requires a validation loop: its `ReduceLROnPlateau` watches `val_map`, so `fit()` without val data crashes at the end of the first epoch. | Medium | Always pass val dataloaders. |
| 3 | **Bug:** a label box that only *touches* the chip edge becomes a **NaN box** (`clip_by_rect` returns an empty geometry, whose bounds are NaN). NaN passes torchvision's degenerate-box check, so it would silently corrupt the loss instead of crashing. This is likely with pixel-aligned labels and grid sampling. Partial boxes are also kept regardless of how much is visible; there is nothing like RV's `ioa_thresh`. | High (silent) | A transform that drops boxes that are NaN or too small. |
| 4 | No normalization by default, and no stats computation. The user has to scale or normalize by hand. | Low | Manual transform. |
| 5 | `GeoDataModule` can't be pointed at arbitrary files. It needs a `dataset_class` that accepts `split=`, so the YAML/LightningCLI route and the datamodule's built-in augmentation hooks require writing a custom dataset or datamodule class. Plain DataLoaders get no augmentation. | Medium | Write a small datamodule. |
| 6 | Sampling is uniform. There is no way to oversample windows that contain objects or control the negative ratio (RV's `neg_ratio`), so with sparse objects most chips are empty. | Medium | Custom sampler. |
| 7 | **Prediction output isn't georeferenced.** `ObjectDetection.predict_step` returns per-chip pixel boxes, and `collate_fn_detection` drops `bounds` and `transform`. There is no cross-chip NMS and no GeoJSON or vector output. Turning predictions into a map layer needs a custom collate function, a coordinate transform, merge logic, and a writer. | High for deployment | Custom code (RV does this in config). |
| 8 | There is no scene-level evaluation and no TorchGeo tutorial covering detection on user GeoTIFFs. | Medium | — |

We searched TorchGeo's issue tracker and found no existing report for #1 or #3.

**Working minimal script after workarounds** (abridged):

```python
def scale(s): s['image'] = s['image'] / 255.0; return s
def fix(s): s['label'] = s['label'].long(); return s          # bug #1

labels = VectorDataset(paths='labels/', task='object_detection',
                       label_name='class_id', transforms=fix)
train_ds = RasterDataset(paths=[...train tifs], transforms=scale) & labels
val_ds   = RasterDataset(paths=[...val tifs],   transforms=scale) & labels
train_dl = DataLoader(train_ds, sampler=RandomPatchSampler(train_ds, size=256, length=64),
                      batch_size=4, collate_fn=collate_fn_detection)
val_dl   = DataLoader(val_ds, sampler=GriddedPatchSampler(val_ds, size=256, stride=256),
                      batch_size=4, collate_fn=collate_fn_detection)
task = ObjectDetection(model='faster-rcnn', backbone='resnet18', num_classes=3)
L.Trainer(max_epochs=...).fit(task, train_dl, val_dl)   # val required (#2)
# NaN/edge-box filtering (#3), georeferenced prediction output (#7): not shown; custom code
```

**Comparison with RV.** In RV the same workflow is one `ObjectDetectionConfig` with:
- scenes made of a `RasterioSource` and a GeoJSON label source
- `ObjectDetectionChipOptions` (`neg_ratio`, `ioa_thresh`)
- `ObjectDetectionPredictOptions` (`merge_thresh`, `score_thresh`)

A single `rastervision run` then produces stats, training, scene-wide predictions as GeoJSON in map coordinates, an evaluation JSON, and a deployable bundle. See [cowc_potsdam.py](../rastervision_pytorch_backend/rastervision/pytorch_backend/examples/object_detection/cowc_potsdam.py) for a complete example.

**Takeaway.**
- **Training:** TorchGeo is a workable, if bumpy, replacement for RV on arbitrary GeoTIFFs. Expect to fix things up by hand and write your own glue code.
- **Inference to a usable output:** this is where RV is still clearly ahead, especially for detection.

The detection bugs are small and easy to fix upstream. The missing georeferenced detection output is the substantive gap, and it isn't covered by TorchGeo's in-progress tiled-inference work, which targets per-pixel classification.

### 4.11 Hands-on test: `MultiRasterSource` alignment accuracy

> **Update (2026-10-27): fixed** on the `rv-future` branch. Non-primary `RasterioSource`s are now reprojected onto the primary raster's pixel grid; the benchmark below now gives the same errors as the proper-warp baseline (0.00 px same-CRS, ≤0.07 px across CRSs). See [fixable-issues.md #1](./fixable-issues.md#1-multirastersource-misaligns-rasters-with-different-grids-or-crss).

**Question.** RV combines rasters with different grids or CRSs without reprojecting them. Is that alignment correct?

**How RV does it** ([multi_raster_source.py](../rastervision_core/rastervision/core/data/raster_source/multi_raster_source.py), `_get_sub_chips`):
1. Read the pixel window from the primary source.
2. Convert the window to a map-coordinate box (EPSG:4326 by default) by projecting **only two corners**.
3. For each secondary source, convert that box to the secondary's pixel coordinates, again using two corners. The result is an axis-aligned window that gets **snapped to whole pixels**, either by `round_pixels` or by rasterio, which rounds fractional windows when reading.
4. Read that window and **resize** it (bilinear) to the primary chip's shape.

This has two built-in sources of error:
- **Grid snapping.** The secondary window is off by up to half a *secondary* pixel at each edge. That causes both a shift and a scale error after the resize. The coarser the secondary relative to the primary, the bigger the error in primary pixels.
- **No real warping.** Rotation (grid convergence) and non-linear distortion between CRSs are ignored. The secondary is read on its own grid and stretched as an axis-aligned rectangle. The error grows with chip size and with the angle between the two grids.

**Test.**
- Primary: a 3000×3000 GeoTIFF in EPSG:32617 at 1 m, near the eastern edge of zone 17.
- Every raster stores, in each pixel, the true UTM-17N coordinates (E, N) of that pixel's centre, whatever CRS the file itself is in.
- For a chip from `MultiRasterSource`, the difference between the secondary and primary values is therefore the misalignment in metres, which equals primary pixels here.
- Each case reads 40 random 256×256 windows.
- Baseline: proper `rasterio.warp.reproject` (bilinear) onto the primary's grid.

| Secondary raster | RV error, mean / p95 / max (primary px) | Proper warp, mean / max |
|---|---|---|
| Same CRS, 1 m, grid-aligned (control) | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| Same CRS, 3 m, grid offset | 0.89 / 1.48 / 2.33 | 0.00 / 0.00 |
| Same CRS, 10 m, grid offset | 2.74 / 4.90 / 12.1 | 0.00 / 0.00 |
| EPSG:4326, ~1 m | 3.86 / 8.95 / 12.1 | 0.03 / 0.07 |
| UTM 18N (adjacent zone), 1 m | 8.11 / 18.9 / 24.9 | 0.04 / 0.09 |

Partially out-of-bounds windows were handled correctly, because reads use `boundless=True` and there is no stretching at raster edges.

**What this means**
- **Same CRS, same grid:** exact. This is the most common case, for example bands of one product split across files.
- **Same CRS, different resolution or offset grid:** misaligned by up to about one *secondary* pixel. For a 10 m layer under 1 m imagery that is 3 px on average and up to 12 px. This matters for fine-scale tasks such as building edges, and less for coarse context layers.
- **Different CRS:** clearly wrong. It is off by up to 12 px (4326) or 25 px (adjacent UTM zones) on 256 px chips, and worse for larger chips.

**Ablations**
- Turning off `round_pixels` on the secondary does **not** help. The mean error went from 2.74 to 4.21 px in the 10 m case, because rasterio still snaps fractional windows.
- Pre-aligning each secondary to the primary's exact grid with a GDAL **warped VRT** (`WarpedVRT(src, crs=, transform=, width=, height=)` saved as `.vrt`, or `gdalwarp -of VRT -t_srs … -te … -tr …`) brings the error down to ≤0.09 px in every case.

**How easy is it to fix?**
- **Workaround today, no code changes:** build an aligned warped VRT for each secondary raster (as above) and pass it to `RasterioSource`. The VRT is lazy and stores no pixel data, works with `/vsis3/` and `/vsicurl/` sources, and costs about the same as the current read-and-resize.
- **Proper fix in RV (about 1–2 days including tests and docs):**
  - Change `MultiRasterSource` so that non-primary `RasterioSource`s open their dataset as a `WarpedVRT` that matches the primary's CRS, transform, width, and height.
  - Then read the *same pixel window* from every source. This drops the map-coordinate round trip and the resize. It is essentially what TorchGeo does (`WarpedVRT` + `rasterio.merge`).
  - Expected size: about 50–100 lines, plus a `resampling` option, since categorical layers need `nearest`.
  - Caveats:
    - Non-rasterio sub-sources (`XarraySource`, nested `MultiRasterSource`) would keep the current path or need their own fix.
    - `TemporalMultiRasterSource` inherits the fix for free.
    - It changes model inputs slightly, so it should be opt-in, or versioned through a config upgrader, to keep existing bundles' behaviour reproducible.

**Bottom line.** Your recollection is correct. RV's multi-raster alignment is exact only when the grids already match. It is approximate, up to about one coarse pixel, across resolutions in the same CRS, and noticeably wrong across CRSs. The fix is simple and well understood, but someone has to make it. TorchGeo already does this correctly, which removes one more argument for keeping RV as the "arbitrary GeoTIFFs" tool.

### 4.12 Hands-on test: does RV still need a separate GDAL install?

**Question.** Depending on GDAL binaries has made RV's dependency management hard, and is one reason it hasn't fully moved to `uv`. Is a separate GDAL still needed, or can wheel-based libraries such as rasterio handle everything?

**Short answer.** No, RV no longer needs it. Every direct GDAL use in RV has a wheel-only replacement. Only one runtime code path actually breaks without system GDAL today, and it is a small fix.

**The current Docker image ships three separate GDALs:**
- `gdal-bin` 3.4.1 from apt, for the CLIs
- `gdal` 3.6.3 from conda-forge, for the `osgeo` Python bindings needed by `rastervision_gdal_vsi`
- the `libgdal` bundled inside the rasterio wheel

The `gdal` package on PyPI (latest 3.13.3) is **still sdist-only**. Installing it means compiling against a matching system `libgdal`, and that is what breaks a pure `uv` workflow.

**Every direct GDAL use in RV library code, and its wheel-only replacement:**

| Use | Where | Needed by | Wheel-only replacement (tested) |
|---|---|---|---|
| `osgeo.gdal` bindings (`VSIFOpenL`, `ReadDir`, `VSIStatL`, and so on) | [rastervision_gdal_vsi](../rastervision_gdal_vsi/rastervision/gdal_vsi/vsi_file_system.py), an optional plugin, not installed by default since #1622 | Reading and writing non-raster files on GCS, Azure, or inside zip/tar archives. Rasters on those backends don't need it, because rasterio opens `/vsi*` paths natively. | **fsspec** (`zip://`, `tar://`, plus `s3fs`, `gcsfs`, `adlfs`) for bytes, and **pyogrio** `vsi_listtree`, `vsi_rmtree`, and `vsi_unlink` for listing and deleting. TorchGeo uses the latter. ✅ |
| `gdalbuildvrt` CLI via `subprocess` | [`build_vrt`](../rastervision_core/rastervision/core/data/utils/rasterio.py) | `RasterioSource` with **more than one URI**, which mosaics tiles into one scene | **rio-vrt** (`build_vrt`, pure Python on top of rasterio), or GDAL's **GTI** tile-index driver built into rasterio's GDAL (≥3.9): write a GeoPackage index with geopandas and open `GTI:index.gpkg`. Both gave the correct mosaic. A small hand-written VRT XML writer would also work. ✅ |
| `gdal_translate` + `gdaladdo` CLIs | [`create_cog`](../rastervision_core/rastervision/core/utils/cog.py) | **Nothing in RV calls it.** It is a public utility only. | `rasterio.shutil.copy(..., driver='COG')`. The output passed `rio_cogeo.cog_validate`, with overviews built. This is the approach TorchGeo's `GeoTIFFWriter` uses. ✅ |
| `gdalwarp`, `gdal_translate`, `gdalinfo`, `gdalsrsinfo` CLIs | `examples/.../cowc_potsdam_data_prep/*.py` | One-off data-prep scripts for an example | `rasterio.warp` / `WarpedVRT`, `rasterio.open().profile`. Not library code. |

**Capabilities of the rasterio wheel** (rasterio 1.5.2, bundling GDAL 3.12.2):
- Drivers: GTiff, COG, VRT, GTI, GPKG, JP2OpenJPEG, netCDF, and HDF5 are all present.
- Remote and archive access: `/vsizip/` and `/vsicurl/` raster reads worked.
- No GDAL CLI binaries are installed.

**RV's test suite in a pure `uv` venv**
- Setup: Python 3.11, with `rastervision_pipeline`, `rastervision_core`, and `rastervision_aws_s3` installed from source, and `PATH` restricted so no system GDAL CLIs are visible.
- Result: 397 passed, 10 failed, 1 error.
- The results were **identical with the system GDAL CLIs on `PATH`**, so **none of the failures is GDAL-related**. They come from dependency version drift (stackstac/pystac, shapely ring orientation), CLI plugin tests, and the `triangle` issue below.
- `build_vrt` has **no test coverage at all**. A manual test confirmed that multi-URI `RasterioSource` fails with `FileNotFoundError: 'gdalbuildvrt'` without system GDAL, and works with it.

**Wheel-only dependency resolution.** All non-GDAL requirements of the remaining packages, including torch, torchvision, and opencv, resolve with `uv pip compile --only-binary :all:` on Python **3.11, 3.12, and 3.13**: about 197–199 packages, none needing a compiler or system libraries. Python 3.14 is blocked only by the `pyproj<=3.7.0` pin in `rastervision_core`, added in #2317.

**Related finding: `triangle`.** The latest commit (`fed86fb9`, "remove triangle requirement") dropped `triangle` from `rastervision_core/requirements.in`. However, `AoiSampler.triangulate_polygon` still does `from triangle import triangulate`, so **random window sampling inside AOIs now fails** in a fresh environment (two `test_aoi_sampler` failures). Shapely ≥2.1 (wheel, GEOS ≥3.10) provides `shapely.constrained_delaunay_triangles`. That correctly triangulated a polygon with a hole in testing (total area 84.0 = polygon area) and is a likely drop-in replacement.

**Work needed to remove GDAL entirely and go wheel-only with `uv`**

| Change | Effort |
|---|---|
| Replace `build_vrt`'s `gdalbuildvrt` subprocess with rio-vrt or GTI, and add a test | S (hours) |
| Rewrite `create_cog` with `rasterio.shutil.copy(driver='COG')`, or deprecate it | S |
| Reimplement `rastervision_gdal_vsi` on fsspec and pyogrio, or deprecate it in favour of a generic fsspec `FileSystem` plugin | S–M |
| Replace `triangle` with `shapely.constrained_delaunay_triangles` in `AoiSampler` | S |
| Drop apt `gdal-bin`/`libgdal-dev` and conda `gdal` from the Dockerfile, and the macOS GDAL note from the docs | S |
| Relax the `pyproj<=3.7.0` pin (for Python 3.14) | S, plus re-testing |

Total: about 2–4 days, including tests and docs. This is a well-trodden path, and TorchGeo already depends only on rasterio, pyogrio, and geopandas wheels with no `osgeo` and no GDAL CLIs.

**Implication for the decision.** The GDAL problem doesn't block RV from moving to `uv`, but it is another piece of maintenance someone would have to do if we keep the project.

---

## 5. RV-exclusive features that would be missed

Ranked by impact, meaning how many users would care and how hard it is to replace. Effort assumes a competent user building on TorchGeo. S = under a day, M = a few days, L = a week or more.

| # | Feature | Who would miss it | Alternatives today | Effort to replicate |
|---|---|---|---|---|
| 1 | **Whole-scene inference with overlap stitching → GeoTIFF**, including score rasters and edge cropping | Anyone deploying a model on large imagery. This is the most common need. | TerraTorch tiled inference; TorchGeo `GeoTIFFWriter` plus the coming `TiledInferenceCallback`; GeoAI; GeoDeep | M (S once torchgeo/torchgeo#3667 and the callback land) |
| 2 | **Model bundle and `rastervision predict` CLI**: a portable artifact that knows its own preprocessing | Teams handing models to non-ML engineers or putting them in production | None equivalent. Closest is Lightning checkpoint plus `hparams` plus manual normalization, or a TorchGeo weights enum with `transforms`. | M |
| 3 | **Segmentation → vector polygons**, including building footprint regularization | Mapping applications (buildings, land cover to polygons) | GeoAI (`raster_to_vector`, building regularization); `rasterio.features.shapes` | S–M |
| 4 | **Scene-level object detection**: cross-window NMS merging and GeoJSON output in map coordinates | Detection users | GeoAI and SAHI-style tools; otherwise custom code. Confirmed hands-on in §4.10: TorchGeo detection predictions are per-chip pixel boxes without georeferencing. | M |
| 5 | **Scene evaluation reports** (per-class P/R/F1 and confusion matrix over full predicted scenes, as JSON) | Anyone reporting accuracy for map products | Custom torchmetrics over stitched rasters | S–M |
| 6 | **Dataset statistics analyzer and stats-based normalization stored with the model** | Users with non-standard sensors (16-bit, multispectral) | Custom script | S |
| 7 | **Smart window sampling**: nodata rejection, class-aware and negative sampling, variable window sizes, `neg_ratio` for detection | Rare-class and noisy-data problems | Custom sampler subclass, or filtering in a transform | M |
| 8 | **Chip classification from polygons and as GeoJSON grid output** | Coarse screening workflows (for example "does this cell contain buildings?") | Custom code | M |
| 9 | **Vector label transformers** (buffer by class, class inference from properties, shift) | Road and point labels; messy GeoJSON | GeoPandas preprocessing | S |
| 10 | **Declarative end-to-end pipeline with parallel split commands and AWS Batch / SageMaker runners**, plus config upgraders | Teams running large AWS jobs reproducibly. In practice mostly Azavea/Element 84 history. | LightningCLI plus any orchestrator (SageMaker, Batch, Kubeflow, SLURM) | L (but rarely needed as a library feature) |
| 11 | Pre-chipping to disk, pluggable file systems, Docker images | Operational convenience | Standard tooling | S each |

### Patterns across these features

- Items 1–5 are **inference and post-processing**. TorchGeo has said this is in scope ([torchgeo/torchgeo#3214](https://github.com/torchgeo/torchgeo/pull/3214) closes the old issues torchgeo/torchgeo#30, torchgeo/torchgeo#35, torchgeo/torchgeo#1407, and torchgeo/torchgeo#1440), and part of it is merged. TerraTorch and GeoAI already cover parts of it.
- Items 6–9 are **small, separable utilities**. They mostly depend on RV's scene and window abstractions in name only. Each could be ported in a small amount of code.
- Item 10 is **architectural**. It is the original reason RV exists (see [why.rst](../docs/why.rst)), but it is also the part least connected to current usage, and Lightning plus generic orchestrators largely replace it.

---

## 6. What RV lacks that TorchGeo has

This is for completeness, since it affects whether RV users are already better served elsewhere:

- Mosaicking, spatial and temporal indexing, and automatic reprojection across many files.
- 151 datasets with datamodules, and 137 sets of geospatial pretrained weights, including current foundation models.
- Instance segmentation, change detection, spatio-temporal and time-series tasks, and self-supervised pretraining.
- Kornia GPU augmentation, spectral indices, and SAR transforms.
- The full Lightning feature set (AMP, FSDP, multi-node, loggers, callbacks, HPO integrations).
- An active community (131 contributors) with frequent releases, 100% test coverage, and current dependency support.

---

## 7. Who uses RV

- **Downloads:** about 70 a week and about 240 a month. That is close to background CI and mirror levels, and less than 1% of TorchGeo.
- **Issues since January 2025:** five in total (#2304, #2305, #2320, #2323, #2326), four from the same user. Topics were nodata handling, preserving the native CRS, output compression, and a dead SpaceNet link.
- **Public dependents:** GitHub code search finds about 110 files importing `rastervision` outside the main repo. Nearly all are Azavea-owned repos written against the pre-0.20 API (`raster-vision-examples`, `raster-vision-fastai-plugin`, `raster-vision-qgis`, `raster-vision-tile2vec`, and others), plus a few student or one-off projects. No active third-party framework builds on RV.
- **Model zoo:** the 0.30 bundles are still hosted. The 0.31 path in the docs returns 404.

---

## 8. Open questions for the decision step

1. If we discontinue, should we **contribute or port** the inference and post-processing pieces (§5 items 1–5) to TorchGeo or TerraTorch first, or just point users to them?
   - TorchGeo's tiled-inference work is already underway and could take our stitching and cropping know-how.
   - Polygonization and scene-level detection merging may fit TerraTorch or GeoAI better than TorchGeo's core.
   - **Note:** TorchGeo's `AGENTS.md` and AI policy restrict LLM-generated contributions. Any upstream contribution would need to be written by a human or follow that policy.
2. What should happen to the **model zoo and docs**? Options include archiving them on Read the Docs or Zenodo, or leaving them hosted with a deprecation banner.
3. Should we write a **migration guide** ("RV concept → TorchGeo/TerraTorch equivalent")? §4 is close to being one already.
4. Are there **private or enterprise users** (for example Element 84 projects) not visible in public signals? Check before announcing.
5. What would discontinuation **look like in practice**: archive the repo, publish a final release pinned to working dependencies, and add a deprecation notice on PyPI, in the README, and in the docs?
