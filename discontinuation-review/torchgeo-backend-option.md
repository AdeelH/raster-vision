# Option: Replace RV's PyTorch Learners with TorchGeo

*Part of the Raster Vision discontinuation review. See also [feature-comparison.md](./feature-comparison.md) and [fixable-issues.md](./fixable-issues.md).*
*Prepared: 2026-10-27*

## Question

Can we remove RV's own training and model code (`rastervision_pytorch_learner`'s `Learner` classes and configs) and instead let RV's GeoDatasets feed **TorchGeo** tasks? RV would then keep only what it does uniquely well:
- turning scenes into windows (smart sampling, AOIs, raster and vector transformers)
- the "last mile": stitching window predictions, vectorizing them, merging detections across windows, writing georeferenced outputs, and evaluating

## Short answer

**Yes, this is feasible, and it fits the two libraries well.**

We built a working prototype (§3) for semantic segmentation and object detection:
- RV GeoDatasets → a ~20-line adapter → TorchGeo tasks, trained with Lightning
- RV sliding-window prediction → RV stitching, vectorization, and cross-window merging → georeferenced GeoTIFF and GeoJSON

It needed **no changes to RV or TorchGeo**. The adapter also avoids the two TorchGeo detection bugs from [feature-comparison.md §4.10](./feature-comparison.md#410-hands-on-test-object-detection-from-arbitrary-geotiffs-in-torchgeo): int32 labels, and NaN boxes at chip edges.

**Using it in library mode is nearly free.** It needs an adapter, a few prediction helpers, and a tutorial: about **1 week**.

**Full migration is more work.** That means replacing the learner inside the `rastervision run` pipeline, including bundles, examples, docs, and runners: about **3–5 weeks**. It also breaks backward compatibility, because old learner configs and model bundles stop working. Roughly 5.5k lines (about 22% of RV's library code) would be removed.

---

## 1. What RV's learner provides today, and what replaces it

RV's library code (excluding examples) is about 24.8k lines:
- `rastervision_core`: 13.2k
- `rastervision_pytorch_learner`: 8.2k
- `rastervision_pipeline`: 2.5k
- `rastervision_pytorch_backend`: 0.8k

Within `rastervision_pytorch_learner`, **about 2.2k lines are the GeoDatasets, transforms, and visualizers** (`dataset/`), which we would keep. **About 6.1k lines are the `Learner`s, their configs, detection utilities, Torch Hub loading, and DDP code**, which TorchGeo and Lightning would replace. Adding back a few hundred lines for the adapter, prediction helpers, and a slim backend gives a **net removal of about 5.5k lines**.

| Learner responsibility | Where | Replacement | Status |
|---|---|---|---|
| Training loop, checkpointing, logging, multi-GPU (DDP) | `learner.py` (1.8k lines), `utils/distributed.py` | Lightning `Trainer`, which adds AMP, FSDP, multi-node, and loggers | ✅ Works (prototype) |
| Models: DeepLabV3, Faster R-CNN, ResNet classifiers, Torch Hub `external_def` | `*_learner_config.py` | TorchGeo tasks: U-Net, DeepLabV3+, UPerNet, SegFormer, and DPT over any timm encoder; Faster R-CNN, FCOS, RetinaNet; plus 137 sets of pretrained geospatial weights | ✅ Works for segmentation and detection. Classification and regression need the same adapter but are untested. |
| Losses and metrics (class weights, ignore index, COCO mAP) | learners, `object_detection_utils.py` | TorchGeo task hparams (`loss`, `class_weights`, `ignore_index`), and torchmetrics including mAP | ✅ |
| Predicting a dataset (`predict_dataset`): the **only** learner API the scene-prediction code uses, plus `cfg.data.get_data_transforms()` | `utils/prediction.py` | About 10 lines per task: loop over the DataLoader, run the forward pass, softmax, yield numpy | ✅ Works (prototype) |
| Scene stitching, vectorization, cross-window detection merging, output writing | **`rastervision_core`**, not the learner: `SemanticSegmentationLabels.from_predictions`, `ObjectDetectionLabels.prune_duplicates`, `.save()` | Unchanged: these don't depend on the model | ✅ Works with TorchGeo models (prototype) |
| Model bundle (weights, learner config, ONNX, transforms) and `rastervision predict` | `Learner.save_model_bundle` / `from_model_bundle` | A new bundle containing a Lightning checkpoint (hparams included, so `Task.load_from_checkpoint`), the RV scene and raster-transformer config, and an optional ONNX export via `LightningModule.to_onnx` | 🔨 To build |
| Pydantic configs (`LearnerConfig`, `SolverConfig`, `ModelConfig`, `DataConfig`) inside `rastervision run` pipeline configs | `learner_config.py` (1.4k lines) | A slim `TorchGeoBackendConfig(task: str, task_kwargs: dict, trainer_kwargs: dict, data: …)`. Alternatively, LightningCLI YAML for library-mode users. | 🔨 To build; **breaks old configs** |
| Training from pre-chipped images on disk (`ImageDataConfig`, the `chip` command) | backend + learner | Keep RV's `*ImageDataset`s, which are plain torch datasets and work with the same adapter, or drop pre-chipping | 🔨 Decide |
| Batch visualization | `dataset/visualizer/` | Keep: it is part of the dataset module | ✅ |

## 2. Design of the integration

```
RV Scene (RasterSource + LabelSource + AOI, transformers)
   │  RV *RandomWindowGeoDataset / *SlidingWindowGeoDataset   ← RV-only sampling (neg_ratio, ioa_thresh,
   │  (albumentations augmentation on CPU)                       nodata_threshold, size_lims, AOIs)
   ▼
TorchGeoAdapter: (x, y) → {'image', 'mask' | 'label' | 'bbox_xyxy'}
   ▼
TorchGeo task (SemanticSegmentation / ObjectDetection / Classification / Regression)  +  Lightning Trainer
   ▼  (inference)
RV *SlidingWindowGeoDataset → model → per-window numpy predictions
   ▼
RV Labels.from_predictions (smooth, crop_sz) / prune_duplicates → .save()   ← RV-only last mile
   ▼
GeoTIFF labels and scores · polygon / building GeoJSON · detection GeoJSON · eval JSON
```

The whole adapter:

```python
class TorchGeoAdapter(Dataset):
    def __init__(self, ds, task): self.ds, self.task = ds, task
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y = self.ds[i]
        if self.task == 'segmentation':   return {'image': x, 'mask': y.long()}
        if self.task == 'classification': return {'image': x, 'label': y.long()}
        if self.task == 'detection':      # torchvision: xyxy boxes, int64 labels, 0 = background
            return {'image': x, 'bbox_xyxy': y.convert_boxes('xyxy').float(),
                    'label': y.get_field('class_ids').long() + 1}
```

Things the adapter, or a small `RVDataModule`, has to handle:
- **Class-id convention for detection.** RV uses 0-based ids. torchvision reserves 0 for background, so ids are shifted +1 going in and −1 coming out. RV's current `TorchVisionODAdapter` already does the same.
- **Box format.** RV labels use yxyx; TorchGeo and torchvision use xyxy.
- **Normalization.** RV datasets scale unsigned ints to [0, 1] (`normalize=True`). TorchGeo tasks don't normalize internally; TorchGeo datamodules do. When using pretrained TorchGeo weights, the weights' expected normalization must be applied, for example through the RV base transform or a Kornia `Normalize` in a datamodule hook.
- **Validation is mandatory.** TorchGeo's LR scheduler monitors a validation metric. That's fine for the pipeline, which always has validation scenes.
- **Augmentation.** Keep RV's albumentations on the CPU side, or move to TorchGeo/Kornia GPU augmentation through a `LightningDataModule.on_after_batch_transfer`.

## 3. Prototype results

**Setup.**
- Python 3.12, CPU torch 2.14, Lightning 2.6.6, TorchGeo 0.11.0.dev1.
- RV `master` (`rastervision_pipeline`, `rastervision_core`, `rastervision_pytorch_learner`) installed from source.
- Both libraries' dependencies resolved together without conflicts (rasterio 1.5.2, albumentations 2.0.8).
- Data: 3 synthetic 600×600 GeoTIFFs (EPSG:32617, 0.5 m) with 2 object classes, and GeoJSON labels in EPSG:4326 with `class_id`. Train on 2 scenes, validate and predict on the third.

| | Semantic segmentation | Object detection |
|---|---|---|
| RV dataset | `SemanticSegmentationRandomWindowGeoDataset` (`size_lims=(96,160)`, `out_size=128`) | `ObjectDetectionRandomWindowGeoDataset` (`neg_ratio=0.2`, `ioa_thresh=0.5`, `clip=True`) |
| TorchGeo task | `SemanticSegmentation(model='unet', backbone='resnet18', num_classes=3)` | `ObjectDetection(model='faster-rcnn', backbone='resnet18', num_classes=3)` |
| Training | 8 epochs on CPU; val Jaccard 0.92 (car) / 0.89 (truck) | 6 epochs on CPU; val mAP 0.17 (undertrained, as expected) |
| Inference | RV sliding windows (stride 64, padding 32) → `SemanticSegmentationLabels.from_predictions(smooth=True, crop_sz=16)` | RV sliding windows (stride 96) → `ObjectDetectionLabels.from_predictions` → `prune_duplicates(score_thresh=0.5, merge_thresh=0.5)` |
| Scene-level result | IoU **0.90** (car), **0.80** (truck) on the held-out scene | **298** raw boxes over overlapping windows merged to **47** (30 ground-truth objects; undertrained) |
| Outputs | `labels.tif` and `scores.tif` (EPSG:32617, same transform as the input), plus `vector_output/class-0-car.json` and `class-1-truck.json` polygons | `od_out.json` GeoJSON in lon/lat |
| Workarounds needed | None | None. RV's `clip` and `ioa_thresh` avoided TorchGeo's NaN-box issue, and the adapter casts labels to int64. |

The prototype script is about 200 lines including data generation. It is saved in the session folder (`files/rv-tg-proto/`) and could become a tutorial notebook.

## 4. Costs and risks

| Item | Notes |
|---|---|
| **Backward compatibility** | Existing pipeline configs with `PyTorch*Config` / `LearnerConfig`, and existing **model bundles**, stop working. Options: a major-version break (0.40 / 1.0) with a migration guide; a legacy bundle loader kept for inference only; or a deprecation period with both backends. |
| **Python ≥3.12** | TorchGeo requires Python ≥3.12, so RV would have to drop 3.11 (the Docker image's current default, `ARG PYTHON_VERSION=3.11`). |
| **Dependency weight** | TorchGeo adds Lightning, Kornia, timm, segmentation-models-pytorch, torchmetrics, and einops. All are wheels, so this doesn't conflict with going wheel-only ([fixable-issues.md #5](./fixable-issues.md#5-dependence-on-gdal-outside-the-rasterio-wheel)), but installs get bigger. |
| **Coupling to a pre-1.0 API** | TorchGeo renames things between releases. For example, `torchgeo.trainers.SemanticSegmentationTask` became `torchgeo.tasks.SemanticSegmentation` in 0.11, though it keeps deprecated aliases. RV would need a pinned compatible range and CI against TorchGeo's latest release. |
| **Bugs on TorchGeo's side** | Its detection rough edges (int32 labels, NaN boxes) are only partly avoided by using RV datasets. The NaN-box fix should still go upstream. |
| **Runners** | The AWS Batch and SageMaker runners currently launch RV's own DDP. They'd need to launch Lightning (`torchrun` or Lightning's own launcher) instead. That is small, but needs testing on real infrastructure. |
| **Docs and examples** | The examples (`cowc_potsdam`, `isprs_potsdam`, `spacenet_*`, `xview`) and tutorials (`train`, `pred_and_eval_ss`, `onnx_inference`, `lightning_workflow`) would all need updating. This is the largest single chunk of effort. |

## 5. Effort estimate

| Phase | Scope | Effort |
|---|---|---|
| **A. Library mode** (no breaking changes) | Add `TorchGeoAdapter` and an optional `RVDataModule` (builds RV datasets from RV scene configs; enables LightningCLI YAML), plus `predict_scene_{ss,od,cc}` helpers that accept any `nn.Module` or TorchGeo task. Add tests and one tutorial. The learner stays. | **~1 week** |
| **B. Pipeline backend** | New `TorchGeoBackend` and config for `rastervision run`: train with Lightning and sync checkpoints to `train_uri`; predict through the Phase A helpers; new bundle format (checkpoint + RV configs + optional ONNX); update `rastervision predict`. | **~1–2 weeks** |
| **C. Removal** | Delete the `Learner`s and learner configs, slim `rastervision_pytorch_backend`, write config upgraders or a deliberate break, update the runners, port the examples and tutorials, and write a migration guide. | **~1–2 weeks** |

**Total for a full migration: about 3–5 weeks**, plus the [fixable-issues.md](./fixable-issues.md) items that are needed anyway for a release that installs cleanly.

## 6. Implications for the discontinuation decision

- **This change would refocus RV**, rather than simply making it smaller. RV would stop competing with TorchGeo on models and training, where TorchGeo is far ahead, and concentrate on scene handling, smart sampling, and georeferenced outputs, which TorchGeo lacks ([feature-comparison.md §5](./feature-comparison.md#5-rv-exclusive-features-that-would-be-missed)). The maintenance surface would shrink by about 22% of the code, and model and training upkeep would move to an actively maintained project.
- **Phase A is worth doing even if we discontinue RV.** It gives remaining users a documented path onto TorchGeo models, and it keeps RV's last-mile code usable for as long as RV keeps installing.
- **A third option is upstreaming RV's last mile** (stitching, polygonization, detection merging) into TorchGeo, TerraTorch, or a small standalone package, and then archiving RV. TorchGeo's tiled-inference work ([torchgeo/torchgeo#3214](https://github.com/torchgeo/torchgeo/pull/3214)) shows it wants this. Its AI-contribution policy means the work would have to be done by hand.
