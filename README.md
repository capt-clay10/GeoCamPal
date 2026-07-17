# GeoCamPal

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge" alt="Python 3.10+">
</p>

<p align="center">
  <strong>A modular GUI-driven toolbox for geospatial image processing, camera pre-processing, georeferencing, Feature labeling, semi-automatic feature detection, Machine learning integration, DEM generation, time-stacking, and wave run-up analysis.</strong>
</p>

---

## Overview

**GeoCamPal** is a modular, GUI-driven Python toolbox for geospatial image-processing workflows. It brings together camera pre-processing, lens correction, image harmonisation, Ground control point-to-pixel coordinate conversion, georeferencing, feature identification, exploratory image analysis, DEM generation, timestack generation, and wave run-up analysis in a single application.

The software is designed for practical fixed-camera and coastal image-analysis workflows, while remaining useful for broader time-lapse, geospatial image processing, and image classification tasks.

<p align="center">
  <img width="593" height="722" alt="image" src="https://github.com/user-attachments/assets/13934128-30ac-4277-abc1-26c4acebbb6c" />
</p>

<p align="center">
  <em>GeoCamPal launcher interface.</em>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Main Capabilities](#main-capabilities)
- [Installation](#installation)
- [Tools](#modules)
  - [Launcher](#launcher)
  - [Pre-processing Tools](#pre-processing-tools)
  - [Data Exploration Tools](#data-exploration-tools)
  - [Georeferencing Tools](#georeferencing-tools)
  - [Feature Identification and Labelling Tools](#feature-identification-and-labelling-tools)
  - [DEM Generator Tools](#dem-generator-tools)
  - [Time-stacking Tools](#time-stacking-tools)
- [Typical Outputs](#typical-outputs)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)
- [Creator and Contact](#creator-and-contact)

---

## Main Capabilities

| Tools | Modules |
|---|---|
| **Pre-processing** | Field-of-view generation, lens calibration, bad-image filtering, brightness harmonisation, colour harmonisation |
| **Georeferencing** | Pixel-to-GCP conversion, homography matrix creation, image rectification using several supported methods |
| **Feature identification** | HSV masking, AOI/profile filtering, colour-picker classification, manual polyline/polygon editing, GeoJSON and COCO-style export |
| **Data exploration** | Multi-time-series image selection, colour-space analysis, colour statistics, outlier detection |
| **DEM generation** | Shoreline/waterline-based DEM creation from GeoJSON features and water-level data |
| **Time-stacking** | Profile extraction, Hovmöller diagrams, raw timestack generation |
| **Wave run-up** | Run-up contour extraction, time-distance export, PSD analysis, optional Stockdon comparison |

---

## Installation

### Option 1 — Download the stand-alone GUI

The recommended option for most users is to download the latest Windows release from the [Releases](https://github.com/capt-clay10/GeoCamPal/releases) page.

No Python installation is required when using the stand-alone executable.

### Option 2 — Run from source

```bash
git clone https://github.com/capt-clay10/GeoCamPal.git
cd GeoCamPal

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python main.py
```

GeoCamPal targets **Python 3.8 or newer**.

---

# Modules

## Launcher

The launcher is the central hub for GeoCamPal. It groups the available tools into the following sections:

- Pre-processing Tools
- Data Exploration Tools
- Georeferencing Tools
- Feature Identification and Labelling Tools
- DEM Generator Tools
- Time-stacking Tools

Only one tool window is opened at a time to keep console output and GUI state manageable.

---

## Pre-processing Tools

### Field-of-View Generator

The **FOV Generator** visualises single- or multi-camera field-of-view footprints on either a user-supplied GeoTIFF basemap or a clean standalone distance-ring plot. If a DEM is supplied, the tool can compute a line-of-sight viewshed so terrain obstructions are reflected in the visible footprint.

**Typical inputs**

- Camera longitude and latitude
- Camera height
- Heading and depression angle
- Sensor and focal-length parameters
- Range and overlap settings
- Optional GeoTIFF basemap
- Optional DEM GeoTIFF
- Output folder

**Typical outputs**

- `fov_map.png`
- `fov_report.txt`
- `viewshed_mask.tif`, when DEM-based viewshed export is used

<p align="center">
  <img src="https://github.com/user-attachments/assets/6039e2ce-d0bd-4494-b717-86cea6c8f156" width="90%" alt="FOV Generator">
</p>

<p align="center">
  <em>Field-of-view visualisation with camera footprints and spatial context.</em>
</p>

---

### Lens Correction

The **Lens Correction** module computes camera intrinsic parameters from checkerboard calibration images using OpenCV. It supports rectangular checkerboards where the cell width and cell height may differ.

**Typical inputs**

- Folder of checkerboard calibration images
- Number of checkerboard squares across and down
- Cell width and height
- Output folder

**Typical outputs**

- `lens_calibration.pkl`
- `calibration_report.txt`
- Detected-corner preview
- Undistorted-image preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/3da452b0-461d-4286-a0af-598bda97e463" width="90%" alt="Lens Correction">
</p>

<p align="center">
  <em>Checkerboard-based intrinsic camera calibration and undistortion preview.</em>
</p>

---

### Harmonise Images

The **Harmonise Images** module provides a multi-stage image quality and correction workflow. Original images are not overwritten.

#### Stage 1: Filter bad images

The filtering stage can flag images affected by:

- Blur
- Clipped overexposure
- Underexposure
- Darkness
- Low information content
- Rain or droplets on the lens
- Partial obstruction
- Failed, corrupt, truncated, or unreadable image files

#### Stage 2: Harmonise brightness

Brightness harmonisation supports luminance-based gain correction with soft-knee handling and L-channel histogram matching. Optional sky masking can be used when computing reference luminance.

#### Stage 3: Harmonise colour

Colour harmonisation supports:

- Reinhard colour transfer
- Per-channel LAB histogram matching
- Iterative distribution transfer

Both brightness and colour harmonisation include a preview-first workflow. A random sample is processed before committing changes to the full dataset.

**Typical inputs**

- Image folder
- Output folder
- Selected filter thresholds
- Optional reference image
- Optional lens calibration file
- Optional existing bad-image JSON file

**Typical outputs**

- `bad_images.txt`
- `bad_images.json`
- `<input_folder>_filtered_good/`
- `<input_folder>_brightness_harmonised/`
- `<input_folder>_colour_harmonised/`
- Additional workflow-specific folders such as averaged or lens-corrected outputs when those options are used

<p align="center">
  <img src="https://github.com/user-attachments/assets/ec360899-d156-4d42-a740-123a0a59e80c" width="90%" alt="Image filtering">
</p>

<p align="center">
  <em>Image quality filtering interface.</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/5db3b403-bece-41f9-93ed-801065ea1f41" width="90%" alt="Brightness harmonisation">
</p>

<p align="center">
  <em>Brightness harmonisation preview.</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/eca0749d-84e1-44ca-b266-798e16d8ecf8" width="90%" alt="Colour harmonisation">
</p>

<p align="center">
  <em>Colour harmonisation preview.</em>
</p>

---

## Data Exploration Tools

### Time Series Explorer

The **Time Series Explorer** matches images to one or more hydrodynamic or environmental time series using timestamps extracted from image filenames. Up to five time series can be combined using AND logic.

Supported criteria include:

| Criterion | Description |
|---|---|
| **Peaks / Maxima** | Select images closest to local maxima. |
| **Troughs / Minima** | Select images closest to local minima. |
| **Above Threshold** | Select images where the time-series value exceeds a threshold. |
| **Below Threshold** | Select images where the time-series value falls below a threshold. |
| **Near Target Value** | Select images near a user-defined value. |
| **Spring Tide Peaks** | Select images near spring high-water events. |
| **Neap Tide Peaks** | Select images near neap high-water events. |
| **No Filter** | Record values without filtering by that series. |

**Typical inputs**

- Image folder
- One to five time-series CSV files
- Timestamp parsing settings
- Per-series criteria
- Output folder
- Optional copy-selected-images setting

**Typical outputs**

- `matched_images_<mode_tag>.txt`
- Optional copied image folder, such as `images_<mode_tag>/`
- Spring/neap output folders when tidal classification workflows are used

---

### Colour Space Explorer

The **Colour Space Explorer** treats an image folder as a colour dataset. It can visualise channel distributions, two-dimensional scatter-density plots, colour timelines, and colour-profile outliers.

Supported colour spaces:

- RGB
- HSV
- LAB
- Normalised RGB

Available analyses include:

| Analysis Mode | Description |
|---|---|
| **Channel histograms** | Per-channel colour distributions. |
| **2-D scatter density** | Joint distribution between selected colour channels. |
| **Colour timeline** | Channel statistics through image time. |
| **Outlier detection** | Flag images with anomalous colour statistics. |

**Typical inputs**

- Image folder
- Output folder
- Colour space
- Channel selections
- Optional AOI polygon
- Optional feature class
- Outlier threshold

**Typical outputs**

- `color_stats.csv`
- `outliers.txt`, when outliers are detected
- `color_explorer_plots.png`
- Additional plots exportable from the Matplotlib toolbar

<p align="center">
  <img src="https://github.com/user-attachments/assets/19866d49-c9d1-480c-babf-233b491a0670" width="90%" alt="Colour Space Explorer">
</p>

<p align="center">
  <em>Colour distribution, scatter-density, and temporal colour analysis.</em>
</p>

---

## Georeferencing Tools

### Pixel-to-GCP Converter

The **Pixel-to-GCP Converter** allows users to select image pixel coordinates corresponding to known ground-control points. The tool supports flexible GCP CSV column names and optional conversion from latitude/longitude to UTM coordinates.

**Typical inputs**

- Folder containing GCP images
- GCP coordinate CSV file
- Optional bad-GCP exclusion list
- Output folder
- Output CSV filename

**Typical outputs**

A CSV containing pixel-to-world mappings, including fields such as:

- `Image_name`
- `Pixel_X`
- `Pixel_Y`
- `GCP_ID`
- `camera`
- `Real_X`
- `Real_Y`
- `Real_Z`
- `EPSG`

<p align="center">
  <img src="https://github.com/user-attachments/assets/39c81770-5f93-44e7-beac-b7ee3bc098b8" width="90%" alt="Pixel to GCP Converter">
</p>

<p align="center">
  <em>Interactive selection of image pixel coordinates for ground-control points.</em>
</p>

---

### Homography Generator

The **Homography Matrix Creator** computes 3 × 3 homography matrices from pixel-to-world GCP mappings.

It supports:

- RANSAC-based homography estimation
- Accuracy computation
- Manual GCP exclusion
- Advanced simulated-annealing search for GCP subset selection

**Typical inputs**

- GCP CSV file containing at least `GCP_ID`, `Pixel_X`, `Pixel_Y`, `Real_X`, and `Real_Y`
- Optional list of GCPs to exclude
- Output folder
- Output filename

**Typical outputs**

- `<output_name>.txt`
- `<output_name>_bestsubset.txt`, when using the advanced subset-optimisation workflow

<p align="center">
  <img src="https://github.com/user-attachments/assets/496d569a-39f1-47ba-bf09-69bbf9f3be83" width="90%" alt="Homography Matrix Creator">
</p>

<p align="center">
  <em>Homography matrix creation and GCP subset optimisation.</em>
</p>

---

### Georeferencing 

The **Georeferencing Tool** rectifies oblique images into a spatial reference system. It supports several georeferencing methods, not only homography.

Supported methods include:

- Homography
- Camera Projection
- Thin Plate Spline
- Polynomial Order 1
- Polynomial Order 2

The required inputs depend on the selected method. Homography uses a precomputed matrix, while camera projection and GCP-based methods require appropriate GCP and calibration information.

**Typical inputs**

- Input image or image folder
- Selected georeferencing method
- Homography matrix or GCP CSV, depending on method
- Optional lens calibration file for camera projection workflows
- EPSG/spatial reference information
- Corner preset and crop values
- Optional AOI selection
- Output folder
- Batch-processing settings

**Typical outputs**

- Georeferenced `.tif` files
- Batch outputs for single folders or subfolders
- Preview outputs inside the GUI before final processing

<p align="center">
  <img src="https://github.com/user-attachments/assets/831781be-02f3-41cd-ac7d-2aa79cd95650" width="90%" alt="Georeferencing Tool">
</p>

<p align="center">
  <em>Georeferencing workflow with preview and batch-processing support.</em>
</p>

---

## Feature Identification and Labelling Tools

The **Feature Identification and Labelling Tools** combine automatic HSV colour masking with AOI/profile filtering, colour-picker classification, and manual feature editing. It is designed for extracting visible features such as shorelines, waterlines, swash fronts, masks, polygons, and polylines from imagery.

The tool is organised into three modes:

| Mode | Purpose |
|---|---|
| **Single Image** | Tune detection settings and export features from one image. |
| **Folder Processing** | Step through an image folder, refine features, and export training-style outputs. |
| **Batch Processing** | Apply saved or predefined detection settings across a folder. |

Feature identification tools include:

- HSV colour masking
- Optional dual-HSV masking
- AOI/profile-based filtering
- ML-predicted mask support
- Multi-sample colour picker
- Boundary extraction
- Polygon extraction
- Freehand, vertex, polyline, and polygon editing
- Undo/redo during editing
- Georeferenced export where spatial information is available

**Typical inputs**

- Image or image folder
- Optional associated mask file or mask folder
- HSV settings
- AOI/profile settings
- Colour-picker samples
- Optional georeferencing information
- Export/output folder

**Typical outputs**

For single-image and folder-processing workflows, exports may include structured folders such as:

- `images/`
- `masks/`
- `overlays/`
- `geojson/`
- `coco/`
- Optional class-point CSV outputs

For batch workflows, outputs commonly include:

- GeoJSON feature files
- Overlay PNG files

<img width="1055" height="1491" alt="GUI_screenshot_feature_identifier" src="https://github.com/user-attachments/assets/07a5d38b-6aa7-4293-96e2-6ea2f9268ba2" />

---

## DEM Generator Tools

The **DEM Generator** creates digital elevation models from shoreline GeoJSON files and water-level data.

The module uses PCA-aligned cross-shore transect interpolation to reduce artefacts that can occur with direct triangulation at contour edges. It supports both daily and composite DEM workflows.

**Typical inputs**

- Water-level CSV file
- Shoreline GeoJSON folder
- Filename pattern for parsing shoreline dates/times
- DEM resolution
- Vertex spacing
- Smoothing parameter
- Beach-shape setting
- DEM mode: daily or composite
- Output folder
- Optional XYZ export

**Typical outputs**

For daily DEM mode:

- `DEM_<date>_transect.tif`
- `shoreline_xyz_<date>.csv`, when XYZ export is enabled

For composite DEM mode:

- `DEM_composite_<tag>_transect.tif`
- `shoreline_xyz_composite_<tag>.csv`, when XYZ export is enabled

<p align="center">
  <img src="https://github.com/user-attachments/assets/1b92b28c-eba2-4ed5-975e-4f8a198e3e29" width="90%" alt="DEM Generator">
</p>

<p align="center">
  <em>DEM generation from shoreline features and water-level data.</em>
</p>

---

## Time-stacking Tools

### Profile and Hovmöller 

The **Profile and Hovmöller Tool** extracts pixel profiles along a user-defined line across a folder of images. Users can draw a profile line interactively or enter pixel coordinates manually.

The tool can generate either an RGB Hovmöller diagram or an intensity Hovmöller diagram depending on the selected mode.

**Typical inputs**

- Image folder
- Optional recursive search
- Sample image
- Profile start and end coordinates, or an interactively drawn line
- Profile width
- Selected Hovmöller mode
- Output folder

**Typical outputs**

- `hovmuller_rgb.png` or `hovmuller_intensity.png`, depending on selected mode
- `profile_overlay.png`
- `profiles.txt`

---

### Burst Images Time-stacker

The **Raw Timestacker** creates calibrated timestack images from image sequences. It supports timestamp parsing from filenames, EXIF/TIFF metadata, and file modification time as a fallback.

Selection modes include:

- Bounding-box selector
- Straight-line selector
- Freehand-line selector

**Typical inputs**

- Image folder or batch subfolders
- ROI, line, or freehand selector
- Pixel resolution
- Optional gap-filling settings
- Output folder

**Typical outputs**

- `<first_timestamp>_raw_timestack.png`

The output PNG stores timestack metadata such as selector information and pixel-resolution/time-interval metadata where available.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5a207098-7601-49df-b39b-222aadbb72cb" width="90%" alt="Raw Timestacker">
</p>

<p align="center">
  <em>Generation of calibrated timestack imagery from image sequences.</em>
</p>

---

### Wave Run-up Calculator

The **Wave Run-up Calculator** extracts run-up contours from annotated timestack imagery and exports time-distance data. It supports multiple annotation formats.

Supported annotation inputs:

- PNG binary mask
- GeoJSON annotation
- COCO-style JSON annotation

The module can use embedded metadata from raw timestack PNGs when available, including pixel resolution and time interval.

**Typical inputs**

- Raw timestack PNG
- Run-up annotation file
- Pixel resolution or embedded metadata
- Time interval or embedded metadata
- Optional horizontal flip setting
- Optional Stockdon parameters
- Output folder

**Typical outputs**

- Run-up time-distance CSV
- PSD CSV
- Clean run-up plot PNG
- Optional Stockdon comparison text output
- Batch-mode per-file outputs and optional Stockdon summary

<p align="center">
  <img src="https://github.com/user-attachments/assets/dc5407f9-a180-47d9-9167-aff85f1b4238" width="90%" alt="Wave Run-up Calculator">
</p>

<p align="center">
  <em>Wave run-up extraction and spectral analysis from timestack imagery.</em>
</p>

---

## Typical Outputs

Depending on the selected module, GeoCamPal can generate:

| Output Type | Examples |
|---|---|
| **Processed images** | Filtered, brightness-harmonised, colour-harmonised, averaged, or lens-corrected image folders |
| **Geospatial rasters** | GeoTIFF DEMs, georeferenced image outputs, optional viewshed rasters |
| **Vector data** | GeoJSON feature exports |
| **Tables** | CSV, TXT, XYZ, matched-image tables, profile tables, run-up tables |
| **Calibration files** | Lens calibration `.pkl` files |
| **Reports** | Bad-image logs, calibration summaries, FOV reports, processing summaries |
| **Plots** | FOV maps, colour-space plots, DEM previews, Hovmöller plots, run-up plots, PSD plots |

---

## Dependencies

Core dependencies include:

```text
customtkinter
Pillow
opencv-python
numpy
pandas
rasterio
geopandas
shapely
GDAL
utm
matplotlib
tifffile
scipy
pyproj
```

Some workflows may require additional packages listed in `requirements.txt`.

> **Note:** GDAL can require special installation steps on some platforms. Using `conda install -c conda-forge gdal` or pre-built wheels is often the easiest approach if `pip install GDAL` fails.

---

## Contributing

Contributions, bug reports, and feature requests are welcome.

To contribute:

1. Open an issue describing the bug, feature request, or proposed improvement.
2. Fork the repository.
3. Create a new branch.
4. Submit a pull request with a clear description of your changes.

Good first issues are listed here:

[Good first issues](https://github.com/capt-clay10/GeoCamPal/labels/good%20first%20issue)

---

## License

GeoCamPal is distributed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This software is provided *as is*. Users should validate all outputs before using them in scientific, engineering, operational, or decision-critical analyses.

---

## Creator and Contact

**Creator:** Clayton Soares
**Contact:** clayton.soares@ifg.uni-kiel.de
**Institute:** Institute of Geosciences, CAU Kiel
**Source:** [github.com/capt-clay10/GeoCamPal](https://github.com/capt-clay10/GeoCamPal)
