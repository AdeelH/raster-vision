# Option: Simplify the Config Registry and Backward Compatibility

*Part of the Raster Vision discontinuation review. See also [fixable-issues.md](./fixable-issues.md), [plugin-consolidation.md](./plugin-consolidation.md) and [torchgeo-backend-option.md](./torchgeo-backend-option.md).*
*Prepared: 2026-10-28. Assumes we drop compatibility with **existing** model bundles. Nothing here is implemented yet.*

## Short answers

| Question | Answer |
|---|---|
| **Can we remove most of the backward-compatibility machinery?** | ✅ Yes. The versioned-upgrader layer (per-plugin version numbers, `plugin_versions`, 31 upgrader functions, plugin aliases, renamed type hints, and the tests and docs for them) can be deleted outright. It takes about ½–1 day and is low risk. The only casualty is old bundles. |
| **Can we replace it with something simpler and more idiomatic?** | ✅ Yes. (A) Subclasses register themselves, and one pydantic validator picks the right subclass when loading, so plain `model_validate` works. (B) One `format_version` per bundle, migrations defined on the class they migrate, and lenient loading for bundles. (C) Bundles hold only what inference needs. I prototyped (A) and (B) on RV's pydantic 2.10.6. |
| **Can the manifest (and pipeline config) be YAML?** | ✅ Yes. All three integration-test pipeline configs round-trip through `yaml.safe_dump`/`safe_load` and `build_config` unchanged, and the YAML is about 30% smaller than the JSON. PyYAML 6.0.2 is already in the lock (through albumentations and dask). It would become a direct dependency of `rastervision_pipeline`. See §6. |

---

## 1. What exists today

"The registry and configs" is really four layers. Only two of them are about backward compatibility.

| Layer | What it does | About backward compatibility? |
|---|---|---|
| **1. Polymorphic (de)serialization** | [`register_config`](../rastervision_pipeline/rastervision/pipeline/config.py) uses `create_model` to build a *new subclass* of the decorated class with a `type_hint: Literal[...]` field, and registers it. [`build_config`](../rastervision_pipeline/rastervision/pipeline/config.py) walks a dict from the bottom up and builds the class registered for each `type_hint`. | **No.** Runners depend on it: `run_command` reloads `pipeline-config.json` in a subprocess, Batch job or SageMaker job. Bundles do too. Some form of it has to stay. |
| **2. Versioned upgraders** | Each plugin sets an integer version (core 14, pytorch_learner 7, pytorch_backend 2, the rest 0). `plugin_versions` is injected into every saved config. On load, [`_upgrade_config`](../rastervision_pipeline/rastervision/pipeline/config.py) walks every class in each config's inheritance chain and calls its upgrader once per version step. There are also plugin aliases (`set_plugin_aliases`) and renamed type hints (`register_renamed_type_hints`). | **Yes. All of it.** |
| **3. The rest of the [`Registry`](../rastervision_pipeline/rastervision/pipeline/registry.py)** | Runners, file systems, plugin CLI commands, the `rv_config` schema, and plugin discovery by scanning the `rastervision` namespace package with `pkgutil`. | **No.** A separate cleanup (§5). |
| **4. Bundle format** | Nested zips. The outer `bundle/model-bundle.zip` has `pipeline-config.json`, which is the **entire training pipeline config** (training and validation scenes, dataset, solver and so on). [`Predictor`](../rastervision_core/rastervision/core/predictor.py) uses `validation_scenes[0]` as a template for the scene to predict on. The inner `train/model-bundle.zip` has `learner-config.json`, weights, optional ONNX, and copies of external model and transform modules. | **This is why backward compatibility is expensive** (see below). |

### Size of the backward-compatibility layer

- **Upgraders:** 31 functions across core, pytorch_learner and pytorch_backend. Together with the upgrade helpers they come to about 430 lines.
  - 10 of them only *delete* fields that were removed.
  - Several more delete fields as part of a larger change.
- **Machinery:**
  - In `config.py`: `_upgrade_config`, `upgrade_plugin_versions`, `upgrade_config`, `get_plugin`, the `plugin_versions` handling in `dict`, `to_file`, `from_dict` and `save_pipeline_config`, and `PipelineConfig.plugin_versions`.
  - In `registry.py`: about 10 methods for versions, aliases, renamed hints, inheritance chains and upgraders.
  - The bundle-loading fallback in `get_learner_config_from_bundle_dir`.
- **Tests and docs:**
  - About 29 upgrader tests and the `tests/__init__.py::test_config_upgrader` helper. It is also collected by pytest as a test (issue #11).
  - The "upgraders" section of `docs/framework/pipelines.rst`.
  - A filter in `docs/_templates/module.rst`.

### Why layer 4 makes compatibility expensive

A bundle stores the whole pipeline config, so **every schema change anywhere in the config tree breaks bundles**. That includes changes to training-only settings such as solver options, chip sampling or training scenes. Most of the 31 upgraders exist for exactly this reason. Inference never reads most of those fields.

### Inconsistencies (minor; evidence the layer is hard to maintain)

- `run_command` calls `build_config` without upgrading. A pipeline config saved by an older RV and re-run with a newer one fails, even though bundles saved by the same RV version load.
- `Config.from_dict` upgrades only when `plugin_versions` is present. `Predictor` calls `upgrade_config` and then `build_config` directly.
- The decorated class is a *different class* from the one written. `create_model` makes a copy, so the docstring has to be copied over by hand, and the module-level name points to the copy rather than the original.

---

## 2. Proposal A: simpler polymorphism (replaces layer 1's internals)

```python
class RasterioSourceConfig(RasterSourceConfig, type_hint='rasterio_source'):
    uris: list[str]
    ...
```

- **Registration happens when the class is created.** The `type_hint` class keyword is captured in `__init_subclass__`. `__pydantic_init_subclass__` then registers the class and sets the default of the `type_hint` field. No decorator and no `create_model` copy are needed: the class you write is the class you get.
- **One wrap validator on the base `Config`** checks incoming dicts for a `type_hint`, looks up the registered class, and checks that it is a subclass of the expected type. It then validates the dict as that subclass. As a result:
  - `Config.model_validate(d)` and `model_validate_json(s)` build the right subclass at any depth, including in `list[Config]` fields. `build_config` becomes a one-line alias kept for compatibility.
  - Assigning a dict to a field (`validate_assignment=True`) also builds the right subclass.
  - A mismatched type (for example a `scene` dict where a `RasterSourceConfig` is expected) fails with a normal `ValidationError`.
- **Serialization stays the same:** `model_dump(serialize_as_any=True)`. The JSON layout doesn't change, apart from dropping `plugin_versions`.
- **To keep the diff small**, `@register_config('x')` can stay as a thin wrapper that sets the hint, so the 56 files that use it don't all have to change. The class keyword can be adopted gradually.
- **The prototype** (session scratch file `config-simplify/proto.py`) covers:
  - dict and JSON round trips
  - nested lists
  - subclass dispatch on assignment
  - rejecting mismatched types
  - pickling

### Alternatives rejected

| Alternative | Why not |
|---|---|
| Pydantic discriminated unions (`Annotated[A \| B, Field(discriminator='type_hint')]`) | They need a **closed** list of types at definition time. RV's config hierarchy is open: plugins and users add `RasterSourceConfig`, `LabelSourceConfig` and model subclasses. |
| `class_path` + `init_args` (jsonargparse, LightningCLI, Hydra `_target_`) | Saved files would contain module paths, so moving or renaming code would break bundles. Short `type_hint` names are more stable. LightningCLI is the natural format if we adopt the TorchGeo backend, but only for *learner* configs (see [torchgeo-backend-option.md](./torchgeo-backend-option.md)). |
| Keep `build_config` and only delete upgraders | Fine as a first step (§7, step 1). But the `create_model` copies, the separate dict walk, and the fact that plain `model_validate` returns the base class all remain traps. |

---

## 3. Proposal B: limited compatibility for future bundles (replaces layer 2)

1. **One `format_version` integer** in the bundle manifest (§4), instead of per-plugin version numbers. A newer manifest format than the running RV supports is a clear error. Older ones go through migrations.
2. **Migrations live on the class they migrate.** A `migrate(cls, d) -> dict` classmethod decides based on the data's shape (`if 'uri' in d: ...`), not on version numbers. The dispatching validator runs it from base class to subclass *before* validation. For simple renames, pydantic's `validation_alias=AliasChoices('new', 'old')` is enough, with no code. Renamed classes go in a small `type_hint` alias dict.
3. **Lenient loading for bundles:** `model_validate(d, context={'lenient': True})` drops unknown fields with a warning instead of failing. User-written configs stay strict (`extra='forbid'`), so typos are still caught. This alone covers every "field was removed" case: 10 of today's 31 upgraders, and part of several others.
4. **Support policy:** loading bundles from the previous **N minor releases** (for example N = 2) is supported. Migrations older than that get deleted. This puts a bound on how much migration code accumulates.
5. **CI fixture instead of a manual check:** each release commits a tiny bundle (a few KB: an untrained small model or ONNX, plus the manifest) to `tests/data_files/bundles/<version>/`, and a unit test loads and predicts with every supported one. This replaces the manual "collect all model bundles and check them with `predict`" step in `docs/release.rst`.

In the prototype, a single old-style dict that used a renamed `type_hint`, a renamed field (`uri` → `uris`, handled by `migrate`) and a removed field (dropped in lenient mode) loaded correctly. Strict mode rejected it.

Gotcha found while prototyping: the lenient filter must run *after* `migrate`. Otherwise a field that a migration would rename gets dropped as unknown first. This is why migrations are a hook the dispatcher calls, rather than separate `mode='before'` validators: the base class's wrap validator always runs first.

---

## 4. Proposal C: slim bundle with a manifest (fixes layer 4)

The bundle would contain only what inference needs, in one flat zip:

```text
model-bundle.zip
├── manifest.yaml        # format_version, rv_version, task, everything below
├── model.onnx           # and/or model.pth plus an architecture spec
├── stats.json           # normalization stats, if any
└── modules/, transforms/  # only if external defs are used (as today)
```

Example `manifest.yaml`:

```yaml
format_version: 1
rv_version: 0.32.0
task: semantic_segmentation
class_config:
  names: [background, building]
  colors: [black, red]
  null_class: background
input:
  channel_order: [0, 1, 2]
  img_sz: 256
  transformers:
    - type_hint: stats_transformer
      stats_uri: stats.json
model:
  type_hint: semantic_segmentation_model
  backbone: resnet50
  weights: model.pth
  onnx: model.onnx
predict_options:
  chip_sz: 256
  stride: 128
  type_hint: semantic_segmentation_predict_options
```

- The manifest is a small, dedicated `BundleManifest` config. It reuses only the configs that inference genuinely needs: `ClassConfig`, raster transformers, model config, predict options and the label-store defaults.
- Training data, solver, dataset, analyzers and evaluators are **not** included. A future change to those can't break bundles.
- `Predictor` builds the scene to predict on from the manifest plus the user's image URI, instead of patching `validation_scenes[0]`. `Learner.from_model_bundle` reads the same manifest.
- ONNX export (already the default in `save_model_bundle`) means predicting with a bundle doesn't depend on RV's Python model-construction code. That is the most robust kind of compatibility we can offer. Weights plus the model config remain available for fine-tuning.
- This lines up with the TorchGeo-backend option: the manifest is the boundary, so whether training used RV's learners or TorchGeo's trainers doesn't matter to inference.

---

## 5. Layer 3: the rest of the registry (separate, optional)

This isn't about backward compatibility, but it belongs to the same "overwrought" feeling:

- **Plugin discovery:** replace the `pkgutil` scan of the namespace package with `importlib.metadata.entry_points(group='rastervision.plugins')`. This is the standard mechanism and fits the pyproject migration: each package declares `[project.entry-points."rastervision.plugins"]`. It also lets third-party plugins live outside the `rastervision.*` namespace.
- **File systems:** after the fsspec `FileSystem` and the AWS merge ([plugin-consolidation.md](./plugin-consolidation.md)), there are only about 3 to choose between. The silent `LocalFileSystem` catch-all should become explicit.
- **Runners, CLI commands, and the rv_config schema** can stay as small dicts in a registry. They are simple and actually used.
- Importing `rastervision.pipeline` would no longer need to import every plugin just to fill the config registry. Classes register when their module is imported. A dict that names a `type_hint` whose module hasn't been imported yet can be handled by a lazy lookup through the entry point.

---

## 6. YAML for the manifest and pipeline configs

**Feasible, and a good fit for the manifest.** Checked by running the three integration-test pipeline configs (semantic segmentation, object detection, chip classification) through:

`model_dump(mode='json', serialize_as_any=True)` → `yaml.safe_dump(sort_keys=False)` → `yaml.safe_load` → `build_config`

| Config | JSON (indent 2) | YAML | Round trip |
|---|---|---|---|
| Semantic segmentation | 12,824 B | 9,000 B | ✅ dict and config are identical |
| Object detection | 10,133 B | 7,263 B | ✅ |
| Chip classification | 12,246 B | 8,802 B | ✅ |

### Where YAML helps

- **Bundle manifest:** people open bundles to see what's in them. YAML is easier to read and diff, and allows comments, for example a header with the RV version and the command that produced the bundle.
- **User-written pipeline configs:** `rastervision run local config.yaml` could be offered alongside today's Python `get_config()` files. That is how TorchGeo and LightningCLI users already configure experiments. Python configs stay the main way to write configs, since they can compute values.
- **Machine-to-machine files** (the `pipeline-config.json` that runners hand to `run_command`) gain nothing from YAML, but it doesn't hurt either. The simplest rule: **choose by file extension on both read and write** (`.yaml`/`.yml` → YAML, otherwise JSON). Keep JSON as the default for internal files, and accept both formats when loading.

### Caveats

- **Always use `yaml.safe_load` and `safe_dump`.** `yaml.load` with the default or full loader can create arbitrary Python objects, and a bundle is untrusted input.
- **PyYAML implements YAML 1.1, which has pitfalls in hand-written files:**
  - `lr: 1e-4` loads as the *string* `'1e-4'` (pydantic converts it to a float anyway).
  - `no`, `yes`, `on` and `off` load as booleans. Pydantic then *rejects* them for `str` fields, for example a class called `no`, so the error is loud rather than silent.
  - A bare `null` is `None`. RV's default `null_class` name `'null'` must be quoted, and `safe_dump` does quote it.
  - `0.30` loads as a float (a version string written by hand needs quotes).

  `safe_dump` quotes all of these correctly, so **files written by RV are safe**. Only hand-edited files are affected, and pydantic validation catches most cases. If this becomes a real problem, `ruamel.yaml` (YAML 1.2) avoids most of it, at the cost of another dependency.
- **Dumping must use `mode='json'`**, so that enums, tuples and paths become plain YAML types. NaN and infinity become `.nan` and `.inf`, which standard JSON can't represent at all.
- **Dependency:** add `pyyaml>=6` to `rastervision_pipeline`. It is already locked at 6.0.2 through albumentations and dask, and ships wheels for every platform we target.

---

## 7. Plan and effort

| Step | Change | Effort | Risk |
|---|---|---|---|
| 1 | **Delete layer 2:** upgraders, `plugin_versions`, aliases, renamed hints, the bundle fallback, upgrader tests, the docs section and the template filter. Keep no-op shims for `upgrader=` and `set_plugin_version` for one release so third-party plugins don't crash. | ½–1 day | Low |
| 2 | **Replace layer 1's internals** (Proposal A), keeping the `@register_config` wrapper. | 1–2 days | Medium: needs care with `update()` methods that mutate configs under `validate_assignment`, the custom `__repr_args__`, and the `build_config(cfg.dict())` re-validation calls in `cli.py` and `learner.py`. |
| 3 | **Bundle manifest and slimming** (Proposals B and C), plus YAML support for configs (§6) and the bundle fixture test. | 2–4 days | Medium: `Predictor` and `Learner.from_model_bundle` need reworking, and the CLI `predict` and `predict_scene` commands need retesting. |
| 4 | *(Optional)* Entry-point discovery and registry trimming (§5). | ½–1 day | Low |

Total: about **4–7 days**. Each step can be committed on its own.

### What breaks for users

- **Every existing bundle**, including the model zoo, which would have to be re-exported.
- **Third-party plugins** that pass `upgrader=` or call `set_plugin_version`. The shims in step 1 cover this for one release.
- **Saved `pipeline-config.json` files** from earlier releases, when re-run with `run_command`. Re-running training from an old saved config was never really supported (see §1, inconsistencies).

## 8. Implications for the decision

- This removes about 600 lines of the hardest-to-maintain code in RV (about 430 lines of upgraders plus the upgrade machinery in `config.py` and `registry.py`), along with about 29 tests.
- It replaces that code with standard pydantic patterns that any contributor will recognize.
- For users, the change they'd notice is a bundle format that is self-describing, human-readable, and small enough to commit as a test fixture.
- **Combined with the plugin consolidation and the TorchGeo-backend option, it shrinks RV towards what it's actually good at:** turning arbitrary GeoTIFFs and labels into georeferenced predictions through a portable bundle.
- If RV is discontinued instead, none of this matters. But a slim, documented bundle manifest is the one artifact worth keeping either way, because a TorchGeo-side "predict on a scene" tool could adopt it.
