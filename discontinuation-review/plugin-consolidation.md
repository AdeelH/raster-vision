# Option: Consolidate RV's Plugin Packages

*Part of the Raster Vision discontinuation review. See also [fixable-issues.md](./fixable-issues.md) and [torchgeo-backend-option.md](./torchgeo-backend-option.md).*
*Prepared: 2026-10-27. Backward compatibility is deliberately ignored here.*

## Questions and short answers

| Question | Feasible? | Effort | Notes |
|---|---|---|---|
| **Get rid of the `gdal_vsi` plugin?** | ✅ Yes — **done** (2026-10-27, without the fsspec replacement so far) | **S** (1–2 days, including a replacement) | `/vsi*` **rasters already work without it**, because rasterio handles them. Only non-raster files (GeoJSON, configs, bundles, outputs) at `/vsi*` paths lose support. A ~70-line **fsspec**-based `FileSystem` replaced it in testing and covers more backends with wheel-only dependencies. |
| **Merge the three AWS plugins into one?** | ✅ Yes | **S–M** (1–2 days) | They are small (about 940 lines in total), all built on boto3, and each registers one thing. **The SageMaker SDK must become an optional extra**: boto3 alone is 7 packages, while boto3 + `sagemaker` is 132. This works because `sagemaker` is already imported lazily. |
| **Merge the two PyTorch plugins into one?** | ✅ Yes, the easiest of the three | **S–M** (1–2 days, mostly mechanical) | The backend (835 lines) imports the learner everywhere. The learner never imports the backend, and the backend adds no real dependencies of its own. |

Doing all three takes RV from **8 distributions plus the meta-package** to **5 plus the meta-package** (§5), in about **3–5 days in total**, including tests, CI scripts, the Dockerfile, and docs.

---

## 1. How plugins work (why merging is easy)

- [`Registry.load_plugins`](../rastervision_pipeline/rastervision/pipeline/registry.py) discovers plugins with `pkgutil.iter_modules` over the `rastervision` namespace package. Every top-level `rastervision.<name>` module that has a `register_plugin(registry)` function gets called.
- A plugin registers **file systems** (`add_file_system`), **runners** (`add_runner`), **rv_config schema sections** (`add_rv_config_schema`, for example `AWS_BATCH_gpu_job_queue`), and a **plugin version** for config upgraders (`set_plugin_version('rastervision.<name>', N)`).
- Nothing requires one plugin per distribution. One package's `register_plugin` can register any number of file systems and runners.
- The AWS and VSI plugins are all at version 0 (no upgraders). `pytorch_learner` is at version 7 and `pytorch_backend` at version 2. Since we're ignoring backward compatibility, merging can reset these and drop the old upgraders.

**File-system precedence matters.** `add_file_system` appends, and the built-in `HttpFileSystem` and `LocalFileSystem` are registered *after* the plugins. `LocalFileSystem.matches_uri` returns `True` for every URI, so any URI no plugin claims is silently treated as a local path. During testing, a `memory://` write "succeeded" by creating a literal `./memory:` directory. A new file-system plugin is fine as long as it registers through `register_plugin` and therefore comes before `LocalFileSystem`.

---

## 2. Removing the `gdal_vsi` plugin

### What it does today

[`VsiFileSystem`](../rastervision_gdal_vsi/rastervision/gdal_vsi/vsi_file_system.py) (239 lines, `osgeo.gdal`) claims **only URIs that start with `/vsi`**, for example `/vsigs/bucket/x.geojson` or `/vsizip//path/a.zip/b.json`. It does *not* handle `gs://` or `az://`. No other RV package imports it. Core's `build_vrt` uses its own `uri_to_vsi_path` from `rastervision.pipeline`. It is already an optional extra (`rastervision[full]`), not installed by default since #1622.

### What breaks without it

Tested in a fresh venv with pipeline, core, and aws_s3, without `gdal_vsi` or `osgeo`:

| Case | Result |
|---|---|
| Raster at `/vsizip/…`, `allow_streaming=True` | ✅ Works (rasterio) |
| Raster at `/vsicurl/https://…`, `allow_streaming=True` | ✅ Works |
| Raster at `/vsizip/…`, `allow_streaming=False` | ✅ Works |
| Raster at `https://…`, `allow_streaming=False` (via `HttpFileSystem`) | ✅ Works |
| **GeoJSON labels (or any non-raster file) at a `/vsi…` path** | ❌ `NotReadableError` |

**The only capability lost:** reading and writing non-raster files on backends that RV's built-in file systems (local, HTTP, S3) don't cover. In practice that means **GCS, Azure, and files inside zip/tar archives**.

### Replacement: a generic fsspec `FileSystem`

We implemented a prototype `FsspecFileSystem`: about 70 lines implementing RV's `FileSystem` interface, claiming `gs://`, `gcs://`, `az://`, `abfs://`, `zip://`, `tar://`, `memory://`, `ftp://`, and `sftp://`. It passed every case we tried:

| Case | Result |
|---|---|
| `file_to_str('zip://inner/labels.geojson::file:///…/archive.zip')` | ✅ |
| `GeoJSONVectorSource` from the same `zip::` URI | ✅ |
| `str_to_file` / `file_to_str` on `memory://` (precedence fixed) | ✅ |
| `sync_to_dir` + `list_paths` on `memory://` | ✅ |
| Non-streaming `RasterioSource` from a `zip::` URI (download, then open) | ✅ |

Advantages over the VSI plugin:
- **Wheel-only.** fsspec is pure Python, and backend packages such as `gcsfs` and `adlfs` are wheels. This removes the last reason to install `osgeo` ([fixable-issues.md #5](./fixable-issues.md#5-dependence-on-gdal-outside-the-rasterio-wheel)).
- **Normal URIs.** Users write `gs://bucket/x.geojson` instead of `/vsigs/bucket/x.geojson`.
- **More backends** come for free: Azure, SFTP, HF Hub, memory (useful in tests), and others.

**Caveats.**
- `local_path` must keep the inner file's basename for chained URLs. Our first version ended the cached file in `.zip`, so pyogrio treated it as an archive; fixed in the prototype.
- Each cloud backend needs its fsspec implementation installed (`gcsfs`, `adlfs`). Expose them as extras.
- fsspec's `s3fs` could also replace `S3FileSystem`, but it pins `aiobotocore`, which tightly constrains `botocore` and is a well-known source of resolver conflicts with `boto3`. **Keep boto3 for S3** and use fsspec only for the other schemes.

**Where it should live.** Put it in `rastervision_pipeline`, next to `HttpFileSystem` and `LocalFileSystem`, with fsspec as a light dependency and the backends as extras. Alternatively, make it a small `rastervision_fsspec` plugin.

---

## 3. Merging the AWS plugins

### Current state

| Package | Lines | Registers | Dependencies |
|---|---|---|---|
| `rastervision_aws_s3` | 329 | `S3FileSystem`; `AWS_S3` config section (`requester_pays`) | boto3, botocore, tqdm |
| `rastervision_aws_batch` | 197 | `AWSBatchRunner`; `AWS_BATCH` section (job queues and definitions, attempts) | boto3 |
| `rastervision_aws_sagemaker` | 415 | `AWSSageMakerRunner`; `AWS_SAGEMAKER` section (role, images, instance types, spot settings, and so on) | **sagemaker** (plus boto3, used but not declared) |

- The meta-package `rastervision` installs `aws_s3` and `aws_batch` by default. `aws_sagemaker` is only in the `[full]` extra.
- All `sagemaker` imports are inside methods or under `TYPE_CHECKING`. Registering the SageMaker runner therefore **does not import the SDK**.

### Proposed shape

```
rastervision_aws/                 # one distribution, one plugin
  rastervision/aws/
    __init__.py                   # register_plugin: S3FileSystem, AWSBatchRunner, AWSSageMakerRunner, 3 config sections
    s3_file_system.py
    batch_runner.py
    sagemaker_runner.py           # sagemaker imported lazily; clear error if the extra isn't installed
  pyproject: dependencies = [boto3, tqdm]; optional-dependencies: sagemaker = [sagemaker]
```

- **Dependency impact:** the base install stays at 7 AWS-related packages. The `[sagemaker]` extra adds about 125 more. **Making SageMaker a required dependency would be a mistake.**
- **rv_config sections are unchanged** (`AWS_S3`, `AWS_BATCH`, `AWS_SAGEMAKER`), so environment variables and `~/.rastervision` profiles keep working.
- **Runner names are unchanged** (`batch`, `sagemaker`), so `rastervision run batch …` works as before.
- **Mechanical work:**
  - About 45 files reference the three packages, across `scripts/` (`build_packages`, `pypi_publish`, `compile_requirements`, `setup_dev_env`), `Dockerfile`, the root `pyproject.toml`, `.github/dependabot.yml`, `docs/` (API reference, setup, runners), and `tests/aws_*`.
  - None of them are notebooks.

**Fix while merging:** `S3FileSystem.sync_to_dir` and `sync_from_dir` shell out to the **`aws` CLI** (`aws s3 sync`). That is an undeclared dependency, because `awscli` is only in RV's `dev` extra, and `subprocess.run` is called without `check=True`, so a failed sync is ignored. Replace it with a boto3-based sync. See [fixable-issues.md #14](./fixable-issues.md#14-s3-sync-silently-depends-on-the-aws-cli).

**Optional further step.** Fold `S3FileSystem` into `rastervision_pipeline`, since boto3 is light, and keep only the runners in `rastervision_aws`. Most users need S3 I/O; few use Batch or SageMaker.

---

## 4. Merging the PyTorch plugins

### Current state

| Package | Lines | Contents | Imports |
|---|---|---|---|
| `rastervision_pytorch_learner` | 8,246 | `Learner`s and configs, GeoDatasets, transforms, visualizers, detection utilities, Torch Hub, DDP | core, pipeline. **Never imports `pytorch_backend`.** |
| `rastervision_pytorch_backend` | 835 (+ examples) | `PyTorchLearnerBackend` and the per-task `PyTorch*Config`s that connect the learner to `rastervision run`; chip writing; predict glue | 27 imports of `pytorch_learner`, 38 of core, 18 of pipeline |

- The backend has **no dependencies of its own**. On `master` its `requirements.in` lists only `boto3`, which it doesn't use, and `rastervision_pipeline`.
- The `rastervision_core` and `rastervision_pytorch_learner` dependencies that 0.31.1 correctly declared were **lost in #2317** (the move from `setup.py` to `pyproject.toml`). See [fixable-issues.md #15](./fixable-issues.md#15-rastervision_pytorch_backend-lost-its-real-dependencies-on-master).
- Using the learner without the pipeline backend is supported in "library mode". After a merge it still works: users just install a package with the same dependencies, because the backend adds none.

### Proposed shape (two variants)

| Variant | What changes | References to update |
|---|---|---|
| **A. Move the backend into the learner package**: `rastervision.pytorch_learner.backend` | Only `rastervision.pytorch_backend` imports change | About 51 files (backend modules, examples, tests, docs). No notebooks. |
| **B. A new combined package `rastervision_pytorch`** (`rastervision.pytorch.{learner,dataset,backend}`) | All imports change | About 140 files plus 9 tutorial notebooks. Mechanical, `sed`-able. |

- Variant **A** is the least churn.
- Variant **B** gives a cleaner name, and fits better if the learners are later replaced by TorchGeo ([torchgeo-backend-option.md](./torchgeo-backend-option.md)). The merged package would then be just datasets, the TorchGeo adapter, and the backend.

Either way:
- Merge the plugin versions (learner 7, backend 2) into one, reset to 0, and drop the old upgraders.
- Move `examples/` from the backend package to the repo root or the docs.

---

## 5. End state

| Today (8 distributions + meta) | After all three changes (5 + meta) |
|---|---|
| `rastervision_pipeline` | `rastervision_pipeline` (+ `FsspecFileSystem`; extras `[gcs]`, `[azure]`) |
| `rastervision_core` | `rastervision_core` |
| `rastervision_aws_s3`, `rastervision_aws_batch`, `rastervision_aws_sagemaker` | `rastervision_aws` (extra `[sagemaker]`) |
| `rastervision_gdal_vsi` | removed |
| `rastervision_pytorch_learner`, `rastervision_pytorch_backend` | `rastervision_pytorch` (or `rastervision_pytorch_learner` holding the backend) |
| `rastervision` (meta) | `rastervision` (meta, extras `[aws]`, `[sagemaker]`, `[gcs]`, `[azure]`) |

**Going further.** Because plugin discovery uses a *namespace*, nothing stops shipping a **single `rastervision` distribution**, with the heavy parts behind extras: `[pytorch]`, `[aws]`, `[sagemaker]`. Each `register_plugin` would skip quietly, or fail with a clear message, when its extra isn't installed. That would remove the per-package build and publish loops in `scripts/`, cut the `dist/*.whl` `file://` dependencies in the root `pyproject.toml`, and make `uv` workspace and lockfile handling trivial. It is a bigger change (about 1 week) and only worth it if RV continues.

## 6. Effort summary

| Change | Effort | Also fixes |
|---|---|---|
| Remove `gdal_vsi`; add `FsspecFileSystem` with tests and docs | 1–2 days | Removes the `osgeo` dependency (part of fixable-issues #5) |
| Merge the AWS plugins; make SageMaker an extra; boto3-based S3 sync | 1–2 days | fixable-issues #14 |
| Merge the PyTorch plugins (variant A) | 1 day (variant B: 2 days) | fixable-issues #15 |
| Update Dockerfile, `scripts/`, CI, dependabot, root `pyproject.toml`, and docs for the new package set | Included above (shared) | — |
| **Total** | **About 3–5 days** | |

## 7. Implications for the decision

- **If we keep RV:** do all three. They're cheap, they remove a source of dependency pain (`osgeo`), they fix two latent bugs (#14, #15), and they shrink the release machinery. Doing them together with the GDAL and `uv` work in fixable-issues #5, #6, and #7 makes sense.
- **If we wind RV down:** removing `gdal_vsi` and fixing #14 and #15 are worth including in a final maintenance release, because they make the last release install and behave correctly. The package merges themselves aren't worth doing for a project being archived.
