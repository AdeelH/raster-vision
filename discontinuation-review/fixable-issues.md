# Raster Vision: Fixable Issues

*Part of the Raster Vision discontinuation review. Companion to [feature-comparison.md](./feature-comparison.md); structural options are in [torchgeo-backend-option.md](./torchgeo-backend-option.md) and [plugin-consolidation.md](./plugin-consolidation.md).*
*Prepared: 2026-10-27, against `master` @ `fed86fb9` (`0.31.2-dev`); last PyPI release 0.31.1.*

This document lists concrete, fixable problems in RV that we confirmed during the review. Each entry includes:
- what is wrong and who is affected
- the evidence (how we confirmed it)
- the fix and an effort estimate

It is meant to help the discontinuation decision in two ways: it shows what keeping RV healthy would cost, and what a final maintenance release could include if we wind RV down.

**Severity:**
- 🔴 **High**: wrong results, or a core workflow is broken for users today
- 🟠 **Medium**: a feature is broken or degraded in some setups, or there is a real maintenance burden
- 🟡 **Low**: tests, docs, or polish

**Effort** assumes someone familiar with the codebase:
- **XS**: under 1 hour
- **S**: hours
- **M**: 1–3 days
- **L**: about a week

---

## Summary

| # | Issue | Area | Severity | Effort | Affects released 0.31.1? |
|---|---|---|---|---|---|
| 1 | ✅ ~~[`MultiRasterSource` misaligns rasters with different grids or CRSs](#1-multirastersource-misaligns-rasters-with-different-grids-or-crss)~~ (fixed) | Correctness | 🔴 | M | Yes |
| 2 | ✅ ~~[`MultiRasterSource` returns zeros for coarse secondary rasters with rasterio ≥1.4](#2-multirastersource-returns-zeros-for-coarse-secondary-rasters-with-rasterio-14)~~ (fixed) | Correctness / deps | 🔴 | S | Yes (with current rasterio) |
| 3 | [STAC → `XarraySource` is broken with pystac ≥1.12 (stackstac is unmaintained)](#3-stac--xarraysource-is-broken-with-pystac-112) | Correctness / deps | 🔴 | S–M | Yes (with current pystac) |
| 4 | ✅ ~~[AOI window sampling fails because `triangle` was dropped from the requirements but is still imported](#4-aoi-window-sampling-fails-because-triangle-was-dropped-but-is-still-imported)~~ (fixed) | Correctness / deps | 🔴 (on `master`) | S | No (master only) |
| 5 | [Dependence on GDAL outside the rasterio wheel blocks a wheel-only `uv` setup](#5-dependence-on-gdal-outside-the-rasterio-wheel) | Packaging | 🟠 | M | Yes |
| 6 | [CI tests a stale lockfile while users install unpinned latest dependencies](#6-ci-tests-a-stale-lockfile-while-users-get-unpinned-latest-dependencies) | Packaging / CI | 🟠 | S–M | Yes |
| 7 | [`pyproj<=3.7.0` pin blocks Python 3.14](#7-pyproj370-pin-blocks-python-314) | Packaging | 🟠 | S | Master only |
| 8 | [No-data / `null_class` handling in `SemanticSegmentationLabelSource` (#2326)](#8-no-data--null_class-handling-in-semanticsegmentationlabelsource-2326) | Correctness / API | 🟠 | S | Yes |
| 9 | [Five community PRs awaiting review](#9-five-community-prs-awaiting-review) | Project health | 🟠 | S–M | — |
| 10 | [Multi-URI `RasterioSource` (`build_vrt`) has no tests](#10-multi-uri-rasteriosource-build_vrt-has-no-tests) | Tests | 🟡 | S | — |
| 11 | [Test suite only runs inside Docker and with `unittest`](#11-test-suite-only-runs-inside-docker-and-with-unittest) | Tests | 🟡 | S | — |
| 12 | ✅ ~~[Brittle geometry-equality test](#12-brittle-geometry-equality-test)~~ (fixed) | Tests | 🟡 | XS | — |
| 13 | [Broken links: model zoo 0.31 and SpaceNet quickstart data (#2320)](#13-broken-links-model-zoo-031-and-spacenet-quickstart-data-2320) | Docs / assets | 🟡 | S | Yes (docs) |
| 14 | [S3 sync silently depends on the `aws` CLI](#14-s3-sync-silently-depends-on-the-aws-cli) | Packaging / correctness | 🟠 | S | Yes |
| 15 | [`rastervision_pytorch_backend` lost its real dependencies on `master`](#15-rastervision_pytorch_backend-lost-its-real-dependencies-on-master) | Packaging | 🟠 (next release) | XS | No (master only) |
| 16 | [Streaming a multi-URI `RasterioSource` from http(s) is broken](#16-streaming-a-multi-uri-rasteriosource-from-https-is-broken) | Correctness | 🟡 | XS | Yes |
| 17 | [Docker: NodeSource setup script 404s, so an old Ubuntu nodejs is silently installed](#17-docker-nodesource-setup-script-404s) | Docker | 🟡 | S | Yes (image) |
| 18 | [Docker: `sagemaker_pytorch_training` post-install changes locked versions (protobuf)](#18-docker-sagemaker_pytorch_training-post-install-changes-locked-versions) | Docker | 🟡 | S | Yes (image) |
| 19 | [`RasterioSource`: zero-fill of out-of-bbox areas is wrong for resized chips](#19-rasteriosource-zero-fill-of-out-of-bbox-areas-is-wrong-for-resized-chips) | Correctness | 🟡 | S | Yes |

**Rough total:** about 2–3 weeks of focused work to fix everything, or about 1 week for the 🔴 items plus #5 and #6. See [Suggested order](#suggested-order) at the end.

### Progress (2026-10-27): dependency-management migration

The migration to a uv workspace (`uv.lock`, static `pyproject.toml` dependencies, hatchling builds, conda-free Docker image) has been done on the `rv-future` branch. Effect on this list:

- **#5 (GDAL): resolved.** `build_vrt` and `create_cog` now use rasterio only (with tests), `rastervision_gdal_vsi` was removed (no fsspec replacement yet; see [plugin-consolidation.md §2](./plugin-consolidation.md#2-removing-the-gdal_vsi-plugin)), the example data-prep scripts were ported, and the Docker image has no system/conda GDAL.
- **#10 (`build_vrt` tests): resolved.** New tests, plus manual parity checks against `gdalbuildvrt` (identical output in 14 cases).
- **#15 (backend dependencies): resolved.** `rastervision_aws_sagemaker` now also declares `boto3`.
- **#6 (stale lockfile): partly resolved.** `uv.lock` replaces `requirements.txt`, and CI now runs `uv lock --check`. However, the lock deliberately keeps the *same versions* as the old `requirements.txt` (so #2, #3 and #12 are still masked), and published packages still have no version bounds. The "latest dependencies" CI job is still to do. The lock covers Python 3.11–3.12 only (`tool.uv.environments`) because the pinned numpy/rasterio have no 3.13+ wheels.
- **#4 (`triangle`): resolved** (follow-up commit). `AoiSampler` now uses `shapely.constrained_delaunay_triangles`; `triangle` is gone from the lock entirely, and `rastervision_core` requires `shapely>=2.1` (locked: 2.2.0). This also fixed a latent bug: for AOIs with a hole whose centroid lies outside the hole (e.g. a C-shaped hole), the old code triangulated the wrong region and **none** of the sampled points fell inside the AOI.
- **#12 (brittle test): resolved.** The shapely upgrade surfaced it; the test now compares normalized geometries.
- **#2 (rasterio ≥1.4): now also affects Linux ARM64 dev environments and images.** rasterio has no Linux ARM64 wheels before 1.4.4, so the lock uses 1.4.4 on that platform (1.3.10 elsewhere).
- New issues found during the migration: #16, #17, #18.

---

## Correctness

### 1. `MultiRasterSource` misaligns rasters with different grids or CRSs

> **Status: fixed.** Non-primary `RasterioSource`s are now reprojected onto the primary chip's exact pixel grid (`RasterioSource.get_chip_reprojected`, via `rasterio.warp.reproject` with bilinear resampling and per-band NODATA); sources on an identical grid (whole-pixel offset) are still read directly. Re-running the benchmark below: 0.00 px error for same-CRS different-resolution cases and ≤0.07 px across CRSs (the same as a proper `rasterio.warp` baseline), with chip read times within ~10% of before. Note: this changes the chips that models see when the grids differ, so models trained on misaligned data may behave differently. Resampling is fixed to bilinear (as before, for resized reads); a per-source resampling option (e.g. nearest for categorical layers) would be a useful follow-up.

🔴 High · Effort **M** (1–2 days) · Details and measurements in [feature-comparison.md §4.11](./feature-comparison.md#411-hands-on-test-multirastersource-alignment-accuracy)

**Problem.** [`_get_sub_chips`](../rastervision_core/rastervision/core/data/raster_source/multi_raster_source.py) does not reproject secondary rasters. Instead, it:
1. converts the primary window to map coordinates using only two corners
2. converts that box into the secondary's pixel coordinates
3. snaps the result to whole secondary pixels
4. resizes the chip to the primary's shape

As a result, rotation and distortion between CRSs are ignored, and the snapping produces shift and scale errors of up to about one *secondary* pixel.

**Evidence.** We used synthetic rasters in which every pixel stores its own true coordinates, read 40 random 256×256 windows per case, and measured the error in primary pixels at 1 m resolution:

| Secondary | Mean / max error (primary px) |
|---|---|
| Same CRS, same grid | 0 / 0 |
| Same CRS, 3 m, offset grid | 0.9 / 2.3 |
| Same CRS, 10 m, offset grid | 2.7 / 12.1 |
| EPSG:4326 | 3.9 / 12.1 |
| Adjacent UTM zone | 8.1 / 24.9 |

A proper `rasterio.warp` reprojection gives ≤0.09 px in every case. Disabling `round_pixels` doesn't help, because rasterio also rounds fractional windows when reading.

**Who is affected.** Anyone combining sources whose grids differ, such as imagery plus a DEM or Sentinel-2 plus aerial imagery. This also covers `TemporalMultiRasterSource`, which inherits the same logic.

**Fix.**
- Open each non-primary `RasterioSource` dataset as a `rasterio.vrt.WarpedVRT` aligned to the primary's CRS, transform, width, and height.
- Read the *same pixel window* from all sources, and drop the map-coordinate round trip and the resize. This is what TorchGeo does.
- Add a `resampling` option, since categorical layers need `nearest`.
- Make it opt-in, or version it through a config upgrader, so existing model bundles keep producing the same results.
- Non-rasterio sub-sources (`XarraySource`, nested `MultiRasterSource`) keep the current path or get a separate fix.

**Workaround today.** Pre-align each secondary with a warped VRT (`gdalwarp -of VRT -t_srs … -te … -tr …`, or `WarpedVRT` saved as `.vrt`) and pass that to `RasterioSource`. This was tested: it brought the error to ≤0.09 px.

**Related.** PR #2331 (`keep_native_crs`, issue #2323) avoids the EPSG:4326 map-CRS round trip. In our measurements that round trip contributes **no measurable error** when grids already match (control case: 0.00 px). So #2331 is a reasonable API improvement, but it does **not** fix this issue.

### 2. `MultiRasterSource` returns zeros for coarse secondary rasters with rasterio ≥1.4

> **Status: fixed.** For `RasterioSource`s by the reprojection above (the destination grid is explicit, so there are no degenerate windows); for other raster sources (e.g. `XarraySource`), the fallback map-window path now always reads at least one pixel. `test_nonidentical_extents_and_resolutions` passes with rasterio 1.4.4 (it fails without the fix). Upgrading the locked rasterio (1.3.10 on non-ARM64) is now unblocked.

🔴 High · Effort **S**

**Problem.** When a primary window maps to *less than one pixel* of a much coarser secondary raster (for example, a 10 px window over a raster 100× coarser), the secondary chip comes back as **all zeros** with rasterio 1.4+. With rasterio 1.3 it returns the correct value. The likely cause: the window is rounded down to zero height or width, and rasterio 1.4 changed how such reads behave.

**Evidence.** `tests/core/data/raster_source/test_multi_raster_source.py::test_nonidentical_extents_and_resolutions` fails on the current stack (rasterio 1.4.4, numpy 2.4, shapely 2.2). It passes in the same venv after swapping in rasterio 1.3.11, and passes in the conda env (rasterio 1.3.10).

**Who is affected.** Anyone doing a fresh `pip install rastervision==0.31.1`, which resolves to rasterio 1.4 or later, and who combines rasters with very different resolutions. It fails silently: the result is zeros, not an error.

**Fix.** Make sure a secondary window always covers at least one pixel (use `floor` and `ceil` instead of rounding the window bounds), then crop or resample. This goes away entirely if #1 is fixed with `WarpedVRT`. Add a regression test that runs against rasterio ≥1.4.

### 3. STAC → `XarraySource` is broken with pystac ≥1.12

🔴 High · Effort **S** (pin) to **M** (replace stackstac)

**Problem.**
- pystac 1.12 started migrating items to version 2 of the STAC projection extension, which uses `proj:code` instead of `proj:epsg`.
- **stackstac**, which RV uses to build `XarraySource` from STAC items, only reads `proj:epsg`. It fails with `ValueError: Cannot pick a common CRS, since asset 'red' … does not have one`.
- stackstac's latest release is 0.5.1 (August 2024), and its `main` branch still only reads `proj:epsg`. The project appears unmaintained.

**Evidence.** Both `test_xarray_source.py::TestXarraySourceConfig` tests pass with pystac 1.11.0 and fail with pystac 1.12.2 and later, all else being equal. The test fixtures use `proj:epsg`, and pystac's migration rewrites it.

**Who is affected.** Anyone who builds a raster source from STAC items in a fresh install. This is RV's "STAC" feature, used in the `stac_plus_osm` and `temporal` tutorials. Catalogs that already publish `proj:code` (projection extension v2) presumably fail regardless of pystac version.

**Fix options.**
1. **Short term:** cap `pystac<1.12` and document the limitation. Or pass `epsg=` explicitly to `stackstac.stack()` when it can be read from the item's `proj:code`.
2. **Proper:** replace stackstac with **odc-stac**, which is actively maintained. Check its `proj:code` support before switching. Alternatively, build the xarray stack directly with `rioxarray`.

### 4. AOI window sampling fails because `triangle` was dropped but is still imported

🔴 High (on `master`) · Effort **S**

> **Status: fixed** (`AoiSampler` now uses shapely). Correction to the description below: `RandomWindowGeoDataset` catches the `ModuleNotFoundError` and silently falls back to slower rejection sampling, so only direct use of `AoiSampler` failed outright.

**Problem.** Commit `fed86fb9` ("remove triangle requirement") removed `triangle` from `rastervision_core/requirements.in`. However, [`AoiSampler.triangulate_polygon`](../rastervision_core/rastervision/core/data/utils/aoi_sampler.py) still does `from triangle import triangulate`. In a fresh environment, random window sampling inside AOIs (`efficient_aoi_sampling`) fails with `ModuleNotFoundError`.

**Evidence.** Both `test_aoi_sampler.py` tests fail in a fresh venv. CI didn't catch it, because the Docker image installs from the old `requirements.txt` lockfile, which still pins `triangle==20250106` (see #6).

**Context.** `triangle` has wheels only up to CPython 3.13, which is presumably why it was dropped.

**Fix.** Replace it with `shapely.constrained_delaunay_triangles` (shapely ≥2.1, GEOS ≥3.10; wheels available). In testing, it correctly triangulated a polygon with a hole: 8 triangles whose total area equals the polygon area. Then recompile the lockfile.

### 8. No-data / `null_class` handling in `SemanticSegmentationLabelSource` (#2326)

🟠 Medium · Effort **S**

**Problem.**
- The `SemanticSegmentationLabelSource.__init__` docstring documents a `null_class_id` argument that the signature doesn't have.
- The user who reported #2326 needed no-data pixels in label rasters to map to the null class, so that the model doesn't learn no-data as a real class. There is no straightforward way to configure that.

**Status.** Community PR #2329 ("add `null_class_id` override") has been open since July 2026 and is unreviewed.

**Fix.** Review #2329. At a minimum, fix the docstring. Ideally, also map label-raster nodata to `null_class_id`.

---

## Dependencies and packaging

### 5. Dependence on GDAL outside the rasterio wheel

🟠 Medium · Effort **M** (about 2–4 days including the related items #4 and #7) · Details in [feature-comparison.md §4.12](./feature-comparison.md#412-hands-on-test-does-rv-still-need-a-separate-gdal-install)

**Problem.** The Docker image ships **three separate GDALs**:
- `gdal-bin` 3.4.1 from apt
- `gdal` 3.6.3 from conda-forge
- the `libgdal` bundled in the rasterio wheel

The `gdal` Python package on PyPI is still sdist-only, so installing it means compiling against a matching system `libgdal`. This is a major reason RV hasn't fully moved to `uv`.

**Every direct GDAL use, with a tested wheel-only replacement:**

| Use | Where | Needed by | Replacement |
|---|---|---|---|
| `osgeo.gdal` Python bindings | [`rastervision_gdal_vsi`](../rastervision_gdal_vsi/rastervision/gdal_vsi/vsi_file_system.py) (optional plugin) | Reading and writing non-raster files on GCS, Azure, or inside zip/tar archives | `fsspec` for bytes, plus `pyogrio.vsi_listtree`, `vsi_rmtree`, and `vsi_unlink`. TorchGeo uses the latter. |
| `gdalbuildvrt` subprocess | [`build_vrt`](../rastervision_core/rastervision/core/data/utils/rasterio.py) | `RasterioSource` with more than one URI. **This is the only runtime path that breaks without system GDAL** (`FileNotFoundError: 'gdalbuildvrt'`, confirmed). | `rio-vrt`, or GDAL's GTI tile-index driver inside the rasterio wheel. Both gave correct mosaics. |
| `gdal_translate` + `gdaladdo` subprocess | [`create_cog`](../rastervision_core/rastervision/core/utils/cog.py) | Nothing in RV calls it | `rasterio.shutil.copy(driver='COG')`. The output passed `rio_cogeo.cog_validate`. |
| `gdalwarp`, `gdal_translate`, `gdalinfo`, `gdalsrsinfo` | `examples/.../cowc_potsdam_data_prep/` | Example data-prep scripts only | `rasterio.warp` / `WarpedVRT` |

**Evidence.**
- Ran RV's core, pipeline, and S3 tests in a pure-`uv` venv with `PATH` restricted so the system GDAL CLIs are hidden. The results were identical with and without them (397 passed; the remaining failures are #2, #3, #4, #11, and #12).
- The rasterio 1.5.2 wheel bundles GDAL 3.12.2 with the COG, VRT, GTI, GPKG, JP2, netCDF, and HDF5 drivers, plus `/vsizip/` and `/vsicurl/` access.

**Fix.**
1. Rewrite `build_vrt` with rio-vrt or GTI, and add a test (see #10).
2. Rewrite `create_cog` with the rasterio COG driver, or deprecate it.
3. Reimplement `rastervision_gdal_vsi` on fsspec and pyogrio, or deprecate it in favour of a generic fsspec `FileSystem`.
4. Remove apt `gdal-bin`/`libgdal-dev` and conda `gdal` from the Dockerfile, and the macOS GDAL install note from `docs/setup/index.rst`.
5. Migrate the Dockerfile and dev setup to pure `uv`.

### 6. CI tests a stale lockfile while users get unpinned latest dependencies

🟠 Medium · Effort **S–M**

**Problem.**
- CI builds the Docker image from `requirements.txt`, which was last compiled in March 2025. It pins old versions: rasterio 1.3.10, numpy 1.26.4, pystac 1.10.1, shapely 2.0.6, geopandas 1.0.1, torch 2.6.0.
- The **published** 0.31.1 packages declare their dependencies with **no version bounds** (for example, `rasterio`, `pystac`, `numpy`).
- So `pip install rastervision` today resolves to rasterio 1.4+, numpy 2, and pystac 1.15, a combination that CI never tests.
- That is how #2, #3, and #4 reached users, or `master`, while CI stayed green.

**Evidence.**
- The three failing tests (#2, #3, and #12) pass on the locked stack and fail on current releases.
- `uv pip compile` on the current requirements resolves to about 197–199 packages on Python 3.11–3.13.

**Fix.**
1. Recompile the lockfile and fix what breaks (#2, #3, #12).
2. Add a CI job that runs the unit tests in a plain `uv` venv **with the latest versions of all dependencies**, outside Docker, and optionally one with minimum versions.
3. Add lower and upper bounds where behaviour is known to depend on the version: `rasterio>=1.4` once #2 is fixed, and `pystac` until #3 is fixed.

### 7. `pyproj<=3.7.0` pin blocks Python 3.14

🟠 Medium · Effort **S** (plus re-testing)

**Problem.** `rastervision_core/requirements.in` pins `pyproj<=3.7.0` (added in #2317), and there is no pyproj 3.7.0 wheel for Python 3.14. With `--only-binary :all:`, resolution fails on 3.14 but succeeds on 3.11–3.13.

**Fix.**
1. Find out why the pin was added.
2. Relax it.
3. Re-run the CRS-transformer tests.

---

## Project health

### 9. Five community PRs awaiting review

🟠 Medium · Effort **S–M** in total

| PR | Author | Opened | Topic |
|---|---|---|---|
| #2331 | Pranava-Kumar | 2026-07 | `keep_native_crs` option for `RasterioSource` (issue #2323). See the note under #1. |
| #2329 | Pranava-Kumar | 2026-07 | `null_class_id` override (issue #2326, see #8) |
| #2328 | Pranava-Kumar | 2026-07 | `rastervision check` command to validate config URIs |
| #2330 (draft) | Pranava-Kumar | 2026-07 | Docs for the `check` command |
| #2246 | bdevoghel | 2024-09 | `normalize` flag for `WindowSamplingConfig` |

Whatever we decide, these contributors deserve a response: merge, request changes, or close with an explanation pointing to the discontinuation plan.

---

## Tests and CI

### 10. Multi-URI `RasterioSource` (`build_vrt`) has no tests

🟡 Low · Effort **S**

**Problem.** Nothing in `tests/` exercises `build_vrt` or `download_and_build_vrt`. That is why the GDAL CLI dependency (#5) was invisible to the test suite.

**Fix.** Add a test that mosaics two adjacent tiles and checks pixel values. Do this together with the #5 rewrite.

### 11. Test suite only runs inside Docker and with `unittest`

🟡 Low · Effort **S**

**Problems.**
- `tests/pipeline/test_cli.py` hardcodes `/opt/data/...` output paths, which exist only in the Docker image. The same CLI command succeeds when given `/tmp`.
- `tests/__init__.py` defines a helper named `test_config_upgrader(cfg_class, old_cfg_dict, …)`. Under pytest it gets collected as a test and errors with `fixture 'cfg_class' not found`. CI uses `unittest`, so it doesn't notice.
- The AWS-related tests pick up real credentials from `~/.aws`. On a developer machine with expired SSO they fail instead of using mocks.

**Fix.**
- Use `tempfile` in the CLI tests.
- Rename the helper (for example `check_config_upgrader`) or set `__test__ = False` on it.
- Set dummy `AWS_*` environment variables in the test setup.

### 12. Brittle geometry-equality test

🟡 Low · Effort **XS**

> **Status: fixed.**

**Problem.** `test_geojson_vector_source.py::test_crs_in_geojson` compares polygons with `assertEqual`, which compares exact coordinate order. Newer geopandas/shapely write the ring in the opposite orientation after the CRS round trip. The geometry is identical, but the vertex order differs.

**Fix.** Compare with `geom.equals(...)` or `shapely.normalize`.

---

## Docs and hosted assets

### 13. Broken links: model zoo 0.31 and SpaceNet quickstart data (#2320)

🟡 Low · Effort **S**

**Problems.**
- `docs/conf.py` builds model-zoo URLs as `…/model-zoo-{version}/…`. For 0.31 these return **404**, because the 0.31 bundles were never uploaded. The 0.30 bundles are still online.
- The quickstart links to `spacenetchallenge.github.io/datasets/datasetHomePage.html`, which returns 404 (issue #2320).

**Fix.**
- Upload the 0.31 bundles, or point the docs at 0.30 (check that 0.30 bundles still load in 0.31).
- Update the SpaceNet link to the AWS Open Data registry entry.

---

## More packaging issues

*Found while assessing plugin consolidation; see [plugin-consolidation.md](./plugin-consolidation.md).*

### 14. S3 sync silently depends on the `aws` CLI

🟠 Medium · Effort **S**

**Problem.** [`S3FileSystem.sync_to_dir` / `sync_from_dir`](../rastervision_aws_s3/rastervision/aws_s3/s3_file_system.py) run `subprocess.run(['aws', 's3', 'sync', …])`. This has two problems:
- `awscli` is an **undeclared dependency**. It appears only in RV's `dev` extra, so `pip install rastervision` doesn't install it. Without it, syncing to or from S3 (for example, learner outputs and checkpoints when `train_uri` is on S3) raises `FileNotFoundError`.
- `subprocess.run` is called **without `check=True`**. If `aws` is installed but the sync fails (permissions, network, requester-pays), the failure is **silently ignored** and the outputs are never uploaded.

**Fix.** Reimplement sync with boto3: list the objects, then upload or download only the ones that differ by size or modification time. Alternatively, declare `awscli` as a dependency and pass `check=True`. The boto3 approach also keeps the install wheel-only.

### 15. `rastervision_pytorch_backend` lost its real dependencies on `master`

🟠 Medium (would ship in the next release) · Effort **XS**

**Problem.** In #2317 (`setup.py` → `pyproject.toml`), `rastervision_pytorch_backend/requirements.in` became just `boto3` (unused by the package) and `rastervision_pipeline`. The published 0.31.1 correctly declared `rastervision-core` and `rastervision-pytorch-learner`, which the backend imports 38 and 27 times. A release built from `master` would install a backend that can't be imported on its own.

**Fix.** Restore `rastervision_core` and `rastervision_pytorch_learner` and drop `boto3`. This also goes away if the two PyTorch packages are merged ([plugin-consolidation.md §4](./plugin-consolidation.md#4-merging-the-pytorch-plugins)).

### 16. Streaming a multi-URI `RasterioSource` from http(s) is broken

🟡 Low · Effort **XS**

**Problem.** For `RasterioSource(uris=[...], allow_streaming=True)` with http(s) URIs, `uri_to_vsi_path` (in `rastervision_pipeline`) converts `https://host/path` to `/vsicurl/host/path`, dropping the scheme. GDAL rejects that path. This broke `gdalbuildvrt` the same way before the migration; it was found while testing the new rasterio-only `build_vrt`. `s3://` URIs work.

**Fix.** Map http(s)/ftp URIs to `/vsicurl/<full URL>`.

### 17. Docker: NodeSource setup script 404s

🟡 Low · Effort **S**

**Problem.** The Dockerfile runs `curl -fsSL https://deb.nodesource.com/node_16.x | bash -`, which now returns 404. Because the output is piped into `bash`, the failure is silent, and the following `apt-get install nodejs` installs Ubuntu 22.04's much older nodejs. Node is only used for JupyterLab extensions. This predates the migration.

**Fix.** Use the current NodeSource repository setup (or drop Node if no JupyterLab extensions need building), and use `set -o pipefail` so failures aren't hidden.

### 18. Docker: `sagemaker_pytorch_training` post-install changes locked versions

🟡 Low · Effort **S**

**Problem.** The image installs `sagemaker_pytorch_training==2.8.1` after the locked dependencies, as before the migration. Its dependency `sagemaker-training` needs protobuf ≤3.20.3 or ≥5.28.1, which is incompatible with the locked protobuf (4.25.6, required by e.g. the `sagemaker` SDK, which needs <5). The post-install therefore upgrades protobuf (to 7.x when tested) and cffi in the image only. It is kept outside `uv.lock` so it doesn't force these versions on everyone.

**Fix.** Find a compatible set (e.g. upgrade `sagemaker`) and move it into the `docker` dependency group in `uv.lock`, or drop SageMaker estimator support from the image.

### 19. `RasterioSource`: zero-fill of out-of-bbox areas is wrong for resized chips

🟡 Low · Effort **S**

**Problem.** `RasterioSource._get_chip` zero-fills the parts of a window that lie outside the `bbox` using `fill_overflow`, which computes the overflow in *window* pixels but applies it to the chip *after* resizing to `out_shape`. E.g. a 100x100 window overflowing the top edge by 20 px, read with `out_shape=(50, 50)`, gets 20 (instead of 10) rows zeroed. Found while fixing #1; affects resized reads near the edges of a raster (including the primary source of a `MultiRasterSource`).

**Fix.** Scale the overflow by the resize factor (or zero-fill before resizing).

---

## Suggested order

If we **keep** RV, or do a final **maintenance release** before archiving it:

1. **#6 first**: recompile the lockfile and add a "latest dependencies" CI job. This surfaces everything else on current versions.
2. **#4, #2, #3, #15**: unblock fresh installs (triangle → shapely, coarse-window fix, pystac cap or stackstac replacement, restore the backend's dependencies).
3. **#5 + #10 + #7 + #14**: drop the extra GDALs and the `aws` CLI dependency, move to pure `uv`, and support Python 3.14.
4. **#1**: fix multi-raster alignment, opt-in.
5. **#9, #8**: review the community PRs.
6. **#11, #12, #13**: tidy up tests and docs.

**Roughly 1 week** for steps 1–3, which gives a release that installs and works on current Python and dependencies. **2–3 weeks** for everything.

## Not included here

- **Feature gaps relative to TorchGeo**, such as mixed precision (AMP), instance segmentation, change detection, self-supervised learning, and foundation models. These are enhancements rather than fixes; see [feature-comparison.md §4](./feature-comparison.md#4-feature-comparison).
- **Issues we didn't verify**. These items come only from code we ran or read, plus the issue and PR tracker. A full audit (for example, running the PyTorch learner and backend tests and the integration tests on current dependencies) would likely find more in the same categories as #2, #3, and #6.
