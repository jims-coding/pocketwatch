# Project Requirements: Unified Exploration Data Pipeline & Feature Extraction Engine

## 1. Project Overview
This project delivers an automated, production-grade spatial data ingestion and feature engineering pipeline for mineral exploration targeting within Western Australia (WA). The system accepts a single geographic coordinate pair ($\text{Latitude}, \text{Longitude}$) in decimal degrees, dynamically calculates a localized $2000\text{m} \times 2000\text{m}$ bounding box, executes concurrent web service requests to extract public government vector and raster datasets, and outputs clean, aligned arrays and tables ready for Machine Learning (ML) inference.

---

## 2. Core Functional Requirements

### 2.1 Spatial Bounding Box Engine
* **Coordinate Ingestion:** Accept center-point inputs in WGS84 decimal degrees ($\text{EPSG:4326}$).
* **Metric Projection:** Dynamically reproject coordinates into a metric system to ensure dimensional accuracy. The system defaults to **GDA2020 / MGA Zone 51 ($\text{EPSG:7851}$)** for the WA Goldfields or **MGA Zone 50 ($\text{EPSG:7850}$)** for western assets.
* **Bounding Box Calculation:** Compute an exact $2000\text{m} \times 2000\text{m}$ square centered on the projected point using a hardcoded 1000-meter buffer:
    $$X_{\min} = X_{\text{center}} - 1000 \quad \text{and} \quad X_{\max} = X_{\text{center}} + 1000$$
    $$Y_{\min} = Y_{\text{center}} - 1000 \quad \text{and} \quad Y_{\max} = Y_{\text{center}} + 1000$$
* **Inverse Transformation:** Reverse-project the bounding box corners back to geographic coordinates ($\text{EPSG:4326}$) to conform with remote server request protocols.

### 2.2 Dual-Protocol Data Ingestion Layer (OGC Integration)
The system must interact with the Western Australian Department of Energy, Mines, Industry Regulation and Safety (DMIRS) servers using two distinct Open Geospatial Consortium (OGC) specifications:

#### 2.2.1 Vector Engine (WFS)
* **Protocol:** Web Feature Service (WFS) version 2.0.0.
* **Format:** Explicitly request `application/json` (GeoJSON) responses to bypass heavy XML parsing overhead.
* **Target Layers:**
    * `geonode:mineral_occurrences` (Ground truth target labels)
    * `geonode:faults_and_lineaments` (Structural conduits/traps)

#### 2.2.2 Raster Engine (WCS)
* **Protocol:** Web Coverage Service (WCS) version 1.0.0.
* **Format:** Request raw `GeoTIFF` byte streams rather than visual image formats (JPEG/PNG) to preserve pixel bit-depth.
* **Target Layer:** `aster:quartz_index` (Or alternative CSIRO pre-calculated mineral/alteration grids).
* **Resolution Tracking:** Maintain pixel sampling intervals at native or specified spatial dimensions (e.g., approximately 30m cell resolution using a spacing of $0.00027^{\circ}$).

---

## 3. Technical & System Architecture

### 3.1 Software Dependencies
The system must run natively within a containerized Python environment utilizing the following open-source geospatial stack:

| Library | Version | Core Purpose |
| :--- | :--- | :--- |
| **Python** | 3.10+ | Core runtime environment |
| **requests** | 2.31.0+ | Synchronous HTTP layer for API endpoints |
| **geopandas** | 0.14.0+ | Vector loading, coordinate alignment, and spatial tables |
| **rasterio** | 1.3.0+ | Memory-mapped I/O parsing of GeoTIFF byte arrays |
| **numpy** | 1.24.0+ | Multidimensional grid matrix transformations and statistics |

### 3.2 In-Memory Processing & Memory Bounds
* **Zero-Disk Constraint:** All payload responses from external services must be held and processed directly inside volatile memory buffers (`io.StringIO` for vector GeoJSON, `rasterio.io.MemoryFile` for raster images) to allow execution inside stateless cloud containers.
* **Timeout Enforcements:** * WFS Vector queries must time out at 30 seconds.
    * WCS Raster queries must time out at 45 seconds due to server-side clipping loads.

---

## 4. Data Normalization & Feature Alignment

### 4.1 Missing Values and NoData Mapping
* **Raster Null-Value Handling:** The pipeline must identify the specific metadata `nodata` mask property in the GeoTIFF header and explicitly replace those coordinates with IEEE 754 floating-point `NaN` values (`np.nan`). This prevents arbitrary background numbers (e.g., -9999) from corrupting downstream statistical metrics.
* **Empty Payload Fail-safes:** If a bounding box covers an unmapped or unmineralized grid area, the system must capture empty features (`"features": []`) and hand off a graceful, structurally intact empty dataframe to the model loop instead of crashing.

### 4.2 CRS Synchronization
* **Forced Projection:** All incoming vector datasets must be explicitly forced into the target projected metric system ($\text{EPSG:7851}$) upon capture.
* **Spatial Alignment Verification:** The spatial coordinates of both the final vector bounding boxes and the raster dataset coordinate boundaries must match exactly prior to feature fusion or spatial proximity math (e.g., calculating distance to the nearest fault).

---

## 5. Security & Operation Guidelines
* **Public Endpoint Reliance:** The code must execute over open public server networks without utilizing hardcoded API authorization tokens, private keys, or enterprise credential files.
* **Local Request Caching:** Downloaded vector geometries and raster matrix grids must be structurally decoupled from the live network calls so they can be written as static files (`.geojson` and local `.npy` structures) to permit rapid offline model tuning and testing.

---

## 6. Gold-Screening Pipeline (Project Spec)

### 6.1 Objective
Build a repeatable, auditable "gold-screening" pipeline that ingests public vector and raster datasets around a point-of-interest, produces spatially aligned feature matrices and auxiliary GIS products, and exports packaged artifacts for downstream ML inference and human review.

### 6.2 High-level Inputs
- Center coordinate (latitude, longitude) in WGS84 (EPSG:4326).
- Buffer distance: default 1000 m (produces a 2000 m × 2000 m tile).
- Optional configuration: alternate target EPSG (e.g., EPSG:7850/7851), WFS/WCS endpoints, list of vector layers to request.

### 6.3 Outputs
- Primary package: `output/dataset_package.npz` — compressed payload containing:
    - `vector__<layername>` GeoJSON strings (stored in EPSG:4326 for portability)
    - `raster` numeric array (float, NaN for nodata)
    - `raster_transform` array (6-element affine tuple)
    - `raster_crs` string
- GeoTIFF: `output/aster_quartz.tif` — georeferenced raster written with LZW compression
- Web map: `output/map.html` (and `output/aster_quartz_overlay.png`) for quick visual inspection
- Optional: GeoPackage or separate GeoJSON files for vector outputs (extension task)

### 6.4 Processing Steps (detailed)
1. Compute metric bbox: reproject center to the chosen metric CRS and build a +/-1000 m square; inverse-transform corners to EPSG:4326 for service queries.
2. Discover/select WFS layers (prefer `mo:MinOccView` for mineral occurrences). Request GeoJSON via `GetFeature` with `bbox`.
3. Fetch raster via WCS `GetCoverage` (we use GA Machine Learning Models WCS coverage `ml__radmap_v4_2019_filtered_ML_pctk` as a working raster source). Request `image/tiff` and read into memory with `rasterio.io.MemoryFile`.
4. Convert raster `nodata` to `np.nan` and coerce array dtype to `float`.
5. Reproject / enforce vector CRS to the chosen metric CRS for spatial calculations; store GeoJSON in EPSG:4326 in the output package to maximize portability.
6. Write outputs into `output/`:
     - `dataset_package.npz` (packaged payload)
     - `aster_quartz.tif` (geotiff)
     - `map.html` (viewer)

### 6.5 Feature Engineering & ML-ready Matrix
- Raster-derived features: use raw raster values (windowed pixel grid) as spatial features; optionally compute local statistics (mean, std, texture) per pixel neighborhood.
- Vector-derived features: counts, distances (e.g., distance to nearest occurrence, distance to nearest fault), categorical encodings (e.g., occurrence type), and spatial joins to raster pixels.
- Output shape: a single aligned raster tile array plus a table of vector-derived features keyed to pixel centroids or to the same 2D grid indices.

### 6.6 Evaluation & Diagnostics
- Unit tests: verify bounding-box math, WFS/WCS request parameter composition, and in-memory raster reading.
- Sanity checks: assert raster array contains finite values (not all NaN), check vector GeoDataFrame non-null geometries, and confirm `raster_transform` length == 6.
- Logging: record service response times and response sizes to spot flaky endpoints.

### 6.7 Runtime & Deployment Notes
- Output directory: all files written to `output/` by default.
- Environment: follow `requirements.txt` for Python package installation.
- For reproducible builds on Windows, prefer conda/conda-forge for GIS wheels if pip installs fail.

### 6.8 Next enhancements (roadmap)
- Export Cloud-Optimized GeoTIFFs (COG) for hosting and fast web access.
- Save vector outputs into a GeoPackage (single file) for downstream GIS workflows.
- Add a small CLI to produce batched packages for many target points and a progress report.
- Implement a minimal ML inference wrapper that takes `dataset_package.npz` and returns a probability / score map for gold prospectivity.

---

Appendix: Scripts
- `script.py` — pipeline: discovery, WFS pulling, WCS raster fetch, packaging to `output/`.
- `show.py` — viewer: builds `output/map.html` using `output/dataset_package.npz` and `output/aster_quartz.tif`.

If you'd like, I can now refine this spec into a short README or generate a pinned `requirements.txt` from the active `.venv` for exact reproducibility.