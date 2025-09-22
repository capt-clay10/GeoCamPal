# GeoCamPal

![GitHub release](https://img.shields.io/github/v/release/capt-clay10/GeoCamPal?style=for-the-badge)
![License](https://img.shields.io/github/license/capt-clay10/GeoCamPal?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge)

> **GeoCamPal** is a **modular, GUI‑driven toolkit** for image‑processing and geomatics. From HSV‑based feature extraction to homography‑assisted georeferencing and DEM creation.

---

## Key Features

| Module | What it does |
|--------|--------------|
| **Feature Identifier** | HSV masking **+** manual polygon editing **+** Use pre-made binary masks for feature identification and export |
| **Pixel → GCP Converter** | Pick Ground‑Control Points (GCPs) in imagery and export pixel‑to‑world mappings. |
| **Homography Matrix Creator** | Derive homography matrices (RANSAC & optional simulated‑annealing subset selection). |
| **Georeferencing Tool** | Batch‑warp oblique images into spatial reference systems. |
| **DEM Generator** | Fuse shorelines and water‑level data into daily DEM rasters (with optional XYZ export). |
| **Raw Timestacker** | Build calibrated timestack images from image bursts. |
| **Wave Run‑Up Calculator** | Extract run‑up contours and generate distance‑time plots or CSV outputs. |

---

## 📑 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
  - [Launcher](#launcher)
  - [Pixel → GCP Converter](#pixel-gcp-converter)
  - [Homography Matrix Creator](#homography-matrix-creator)
  - [Georeferencing Tool](#georeferencing-tool)
  - [Feature Identifier](#feature-identifier)
  - [DEM Generator](#dem-generator)
  - [Raw Timestacker](#raw-timestacker)
  - [Wave Run-Up Calculator](#wave-run-up-calculator)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Installation

### 1 · Download the Stand‑Alone GUI (Recommended)

Grab the latest release for Windows from the [**Releases**](https://github.com/capt-clay10/GeoCamPal/releases) page and run `GeoCamPal.exe`. No Python environment required but available through `main.py`.

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

> GeoCamPal targets **Python 3.8 or newer**.

---

## Quick Start

```text
1. Launch the application → select the desired module from the launcher window.
2. Follow the left‑to‑right workflow inside each module (input → configure → run → export).
```

---

## Modules

### Launcher

<img width="598" height="630" alt="Image" src="https://github.com/user-attachments/assets/3e272324-d1bb-467d-9b57-3d73b5b797f9" />e)

The central hub: click any SUB-MODULE to open the corresponding tool in a new window.

---

### Pixel → GCP Converter

![Pixel → GCP](https://github.com/user-attachments/assets/f2c5d443-49a9-429f-8850-b6efcb3afeca)

*Input*: A folder of `GCP_XX_cam*.jpg` images and a CSV of GCP lat/longs.

*Output*: CSV mapping `Pixel_X Pixel_Y ↔ Real_X Real_Y` (optionally converted to UTM).

---

### Homography Matrix Creator

![Homography](https://github.com/user-attachments/assets/dbba3cd7-e109-4362-b03a-c4639818f71f)

* Compute 3 × 3 homography matrices with RANSAC.
* **Advanced mode**: simulated‑annealing search to select the optimal GCP subset.

---

### Georeferencing Tool

![Georef](https://github.com/user-attachments/assets/1b007ab0-517b-426c-99f0-009967267673)

* Batch‑warp images using the previously computed homography.
* Secondary AOI cropping & on‑the‑fly previews.

---

### Feature Identifier

Automatic HSV masking **+** manual editing **+** Use pre-made binary masks.

| Mode | Use Case |
|------|----------|
| **Single Image** | Tweak HSV sliders on a single image. |
| **Folder processing** | Step through a folder, export masks/edges as COCO. |
| **Batch** | Fire‑and‑forget detection across a directory. |

![Auto](https://github.com/user-attachments/assets/3a3abb66-9938-4d82-93c0-c66147afa488)
![Manual](https://github.com/user-attachments/assets/cffc926c-0033-437c-b4a6-17ff03f08efb)

---

### DEM Generator
![Image](https://github.com/user-attachments/assets/f710d0b6-7fd0-4dd0-98c6-516068f73fb0)
Creates DEM rasters from shoreline GeoJSONs (exported from the feature identifier tool) and water‑level data, with optional XYZ export and batch mode support.

---

### Raw Timestacker and Wave Run-Up Calculator

![Image](https://github.com/user-attachments/assets/05a0e52a-b52d-4af1-a411-cdd6b1fe94a2)

Generate distance‑time timestack PNGs from image bursts or video frames. Includes ROI selector, resolution tagging, and batch processing. Feed time stack in the Feature identifier tool to identify the wave runup line, currently limited to HSV-based or manual identification.
Overlay the binary mask (exported from the feature identifier tool), extract the run‑up contour, and export **(time, distance)** CSVs.

---

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an **issue** or **pull request**. If youʼre new to the project, start with the open [good first issues](https://github.com/capt-clay10/GeoCamPal/labels/good%20first%20issue).

---

## License

GeoCamPal is distributed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This software is provided *as is*. Always validate results before using them in critical analyses.
