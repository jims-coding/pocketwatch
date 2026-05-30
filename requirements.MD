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