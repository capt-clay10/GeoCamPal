# GeoCamPal


![License](https://img.shields.io/github/license/capt-clay10/GeoCamPal?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge)

> **GeoCamPal** is a **modular, GUI‑driven toolkit** for Geospatial image‑processing. From camera pre‑processing and image harmonisation, through homography‑assisted georeferencing, rich feature-identification helpers, image-analysis tools, to DEM creation and wave-run-up analysis.

---

## What's New in V2.0.0

V2.0.0 adds a full **Pre‑prep** pipeline, a **Data Exploration** submodule, and a dedicated **Profile** tool, alongside improvements to the existing modules.

| Addition | Description |
|----------|-------------|
| **FOV Generator** | Visualise single‑ or multi‑camera field‑of‑view footprints on a basemap and/or DEM with optional line‑of‑sight viewshed masking. |
| **Lens Correction** | Compute camera intrinsic parameters from checkerboard calibration images and export a `.pkl` file for the rest of the pipeline. |
| **Harmonise Images** | Filter bad images (blur, over/under‑exposure, rain, obstruction), then harmonise brightness and colour across a folder with preview‑before‑commit. |
| **Time Series Explorer** | Match images to hydrodynamic time series by timestamp — select frames at high/low water, threshold exceedances, or classify by tidal range. |
| **Colour Space Explorer** | Analyse colour distributions (RGB, HSV, LAB, normalised RGB) across image folders — histograms, scatter densities, timelines, and outlier detection. |
| **Profile & Hovmöller** | Draw a profile line on an image, extract pixel transects across a time series, and produce RGB/intensity Hovmöller diagrams. |

---

## Key Features

### Pre‑prep Tools

| Module | What it does |
|--------|--------------|
| **FOV Generator** | Single‑ or multi‑camera FOV footprints with optional DEM‑based viewshed masking. Supports user‑supplied GeoTIFF basemaps or standalone distance‑ring plots. |
| **Lens Correction** | OpenCV checkerboard detection for camera matrix + distortion coefficients. Outputs a `.pkl` file for downstream lens undistortion. |
| **Harmonise Images** | Three‑stage pipeline: (1) filter bad images, (2) harmonise brightness (luminance gain or histogram matching), (3) harmonise colour (Reinhard, LAB matching, or iterative distribution transfer). Preview system with before/after navigation. |

### Georeferencing

| Module | What it does |
|--------|--------------|
| **Pixel → GCP Converter** | Pick Ground‑Control Points in imagery and export pixel‑to‑world mappings (CSV), with optional UTM conversion. |
| **Homography Matrix Creator** | Derive 3 × 3 homography matrices with RANSAC. Advanced mode: simulated‑annealing search for optimal GCP subset selection. |
| **Georeferencing Tool** | Batch‑warp oblique images into a spatial reference system using a precomputed homography. Secondary AOI cropping and on‑the‑fly previews. |

### Feature Identifier

| Mode | What it does |
|------|--------------|
| **Single Image** | Tweak HSV sliders, define AOI / profile‑based regions, and export masks or edge polygons from a single image. |
| **Folder Processing** | Step through a folder image‑by‑image, refine masks, manually edit polygons, and export as GeoJSON (COCO‑compatible). |
| **Batch Process** | Fire‑and‑forget HSV detection across a directory with pre‑set parameters. |

### Data Exploration

| Module | What it does |
|--------|--------------|
| **Time Series Explorer** | Match images to hydrodynamic data (tide, waves, currents) by filename timestamp. Modes: high/low water, threshold, user‑defined value, tidal‑range classification (spring/neap). |
| **Colour Space Explorer** | Channel histograms, 2‑D scatter density, colour timelines, and outlier flagging across RGB, HSV, LAB, and normalised RGB. |

### DEM Generator

| Module | What it does |
|--------|--------------|
| **Create DEM** | Fuse shoreline GeoJSONs (from the Feature Identifier) and water‑level data into daily DEM rasters using PCA‑aligned cross‑shore transect interpolation. Optional XYZ export and batch mode. |

### Time‑stacking

| Module | What it does |
|--------|--------------|
| **Profile & Hovmöller** | Draw an interactive profile line, extract pixel transects across a time series, and generate RGB Hovmöller, intensity Hovmöller, and profile overlay plots. |
| **Raw Timestacker** | Build calibrated timestack images from image bursts or video frames. Includes ROI selector, resolution tagging, and batch processing. |
| **Wave Run‑Up Calculator** | Overlay a binary mask on a timestack, extract the run‑up contour, and export (time, distance) CSVs with spectral analysis. |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
  - [Launcher](#launcher)
  - [Pre‑prep Tools](#pre-prep-tools-1)
  - [Georeferencing](#georeferencing-1)
  - [Feature Identifier](#feature-identifier-1)
  - [Data Exploration](#data-exploration-1)
  - [DEM Generator](#dem-generator-1)
  - [Time‑stacking](#time-stacking-1)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Installation

### 1 · Download the Stand‑Alone GUI (Recommended)

Grab the latest release for Windows from the [**Releases**](https://github.com/capt-clay10/GeoCamPal/releases) page and run `GeoCamPal.exe`. No Python environment required.

### 2 · Run from Source

```bash
# clone the repo
$ git clone https://github.com/capt-clay10/GeoCamPal.git && cd GeoCamPal

# create (optional) venv & install deps
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# launch the GUI
$ python main.py
```

> GeoCamPal targets **Python 3.8 or newer**.

### Dependencies

Core dependencies (see `requirements.txt` for the full list):

`customtkinter` · `Pillow` · `opencv-python` · `numpy` · `pandas` · `rasterio` · `geopandas` · `shapely` · `GDAL` · `utm` · `matplotlib` · `tifffile`

Some modules also use `scipy`, `pyproj`, and `pickle` (standard library).

---

## Quick Start

```
1. Launch the application  →  the launcher window appears.
2. Select the desired tool  →  each opens in its own window.
3. Follow the left‑to‑right workflow inside each module
   (input → configure → run → export).
```

---

## Modules

### Launcher

The central hub groups every tool into five sections: **Pre‑prep Tools**, **Georeferencing**, **Feature Identifier Tool**, **Data Exploration**, **DEM Generator**, and **Time‑stacking**. Click any button to open the corresponding module in a new window.

---

### Pre‑prep Tools

#### FOV Generator

Visualise single‑ or multi‑camera field‑of‑view footprints on an optional basemap and/or DEM. When a DEM is supplied, the tool performs a line‑of‑sight viewshed so terrain obstructions mask the FOV correctly.

*Inputs*: Camera position (lat/lon), azimuth, horizontal & vertical FOV, optional GeoTIFF basemap, optional DEM.

*Outputs*: FOV footprint plot (PNG) with distance rings and coordinate grid.

#### Lens Correction

Compute camera intrinsic parameters (camera matrix + distortion coefficients) from a set of checkerboard calibration images using OpenCV.

*Inputs*: Folder of checkerboard images, number of squares (cols × rows), cell width & height (mm).

*Outputs*: `.pkl` calibration file + summary `.txt` report. Undistorted preview shown in‑tool.

#### Harmonise Images

A three‑stage image quality pipeline:

1. **Filter bad images** — detects blur, clipping‑based over/under‑exposure, darkness, low entropy, rain/droplets, and partial obstruction.
2. **Harmonise brightness** — luminance‑based gain with soft‑knee or L‑channel histogram matching, optional sky masking.
3. **Harmonise colour** — full‑colour transfer via Reinhard, per‑channel LAB histogram matching, or iterative distribution transfer (Pitié et al., 2007).

Both harmonisation stages include a **preview system**: a random 5 % sample is processed first with before/after navigation. Originals are never modified.

*Outputs*: `bad_images.txt`, `_brightness_harmonised/` subfolder, `_colour_harmonised/` subfolder.

---

### Georeferencing

#### Pixel → GCP Converter

*Input*: A folder of `GCP_XX_cam*.jpg` images and a CSV of GCP lat/longs.

*Output*: CSV mapping `Pixel_X  Pixel_Y ↔ Real_X  Real_Y` (optionally converted to UTM).

#### Homography Matrix Creator

- Compute 3 × 3 homography matrices with RANSAC.
- **Advanced mode**: simulated‑annealing search to select the optimal GCP subset.

#### Georeferencing Tool

- Batch‑warp images using a previously computed homography.
- Secondary AOI cropping and on‑the‑fly previews.

---

### Feature Identifier

Automatic HSV masking **+** manual polygon editing **+** pre‑made binary mask support. The tool now integrates AOI / profile‑based region filtering and multi‑sample colour class selection.

| Mode | Use Case |
|------|----------|
| **Single Image** | Tweak HSV sliders on a single image, define AOI regions. |
| **Folder Processing** | Step through a folder, refine masks, edit polygons, export as GeoJSON. |
| **Batch** | Fire‑and‑forget detection across a directory. |

---

### Data Exploration

#### Time Series Explorer

Match images to hydrodynamic time series (tide gauge, wave buoy, current meter) based on timestamps extracted from filenames.

| Analysis Mode | Description |
|---------------|-------------|
| High water | Images closest to local maxima. |
| Low water | Images closest to local minima. |
| Threshold‑based | Images near user‑defined extreme events. |
| User‑defined value | Images nearest a specific level. |
| Tidal‑range classification | Classify images into spring / neap cycles. |

*Outputs*: Tab‑separated `.txt` (filename, mode, time offset) and optional copy of selected images to an output folder (tidal‑range mode creates `spring/` and `neap/` sub‑folders).

#### Color Space Explorer

Exploratory colour analysis across a folder of images. Treats images as data and visualises distributions.

| Analysis Mode | Description |
|---------------|-------------|
| Channel histograms | Per‑channel distribution across all images. |
| 2‑D scatter density | Joint distribution of any two colour channels. |
| Color timeline | Channel statistics vs. image timestamp. |
| Outlier detection | Flag images with anomalous colour profiles (N σ). |

Supported colour spaces: RGB · HSV · LAB · Normalised RGB.

*Outputs*: `color_stats.csv`, optional `outliers.txt`, plots exportable as PNG.

---

### DEM Generator

Creates DEM rasters from shoreline GeoJSONs (exported from the Feature Identifier) and water‑level data using PCA‑aligned cross‑shore transect interpolation, avoiding Delaunay triangulation artefacts.

*Inputs*: Shoreline GeoJSON files, water‑level CSV.

*Outputs*: GeoTIFF DEM raster(s), optional XYZ export. Batch mode supported.

---

### Time‑stacking

#### Profile & Hovmöller

Draw a profile line interactively (click start + end on a sample image) or enter pixel coordinates manually, then extract pixel transects across a time‑series folder.

*Outputs*: `hovmuller_rgb.png`, `hovmuller_intensity.png`, `profile_overlay.png`, `profiles.txt` (tab‑separated).

#### Raw Timestacker

Generate distance‑time timestack PNGs from image bursts or video frames. Includes ROI selector, resolution tagging, and batch processing.

#### Wave Run‑Up Calculator

Overlay a binary mask (exported from the Feature Identifier) on a timestack, extract the run‑up contour, and export **(time, distance)** CSVs with spectral analysis (Welch PSD).

---

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an **issue** or **pull request**. If you're new to the project, start with the open [good first issues](https://github.com/capt-clay10/GeoCamPal/labels/good%20first%20issue).

---

## License

GeoCamPal is distributed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This software is provided *as is*. Always validate results before using them in critical analyses.

---

**Creator**: Clayton Soares
**Contact**: clayton.soares@ifg.uni-kiel.de
**Institute**: Institute of Geosciences, CAU, Kiel
**Source**: [github.com/capt-clay10/GeoCamPal](https://github.com/capt-clay10/GeoCamPal)
