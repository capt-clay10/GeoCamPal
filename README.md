
# GeoCamPal
![Image](https://github.com/user-attachments/assets/7ead2d07-194d-4979-8191-195b9fd51d59)

GeoCamPal is a modular image processing tools with an intuitive graphical user interface (GUI). This software enables users to:

- Perform feature identification in images using HSV masking technique
- Export as training dataset following COCO JSON convention
- Convert image pixel coordinates to realy world coordinates using GCPs
- Compute homography matricesfor transforming oblique images to georeferenced images
- Can create Digital Elevation Models using detected shorelines and user provided water level data
- Create time-stack images for temporal analysis
- Calculate wave run-up distances


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Launcher Window](#launcher-window)
  - [Pixel to GCP Converter](#pixel-to-gcp-converter)
  - [Homography Matrix Creator](#homography-matrix-creator)
  - [Georeferencing Tool](#georeferencing-tool)
  - [Feature Identifier](#feature-identifier)
  - [DEM Generator](#dem-generator)
  - [Raw Timestacker](#raw-timestacker)
  - [Wave Run-Up Calculator](#wave-run-up-calculator)

- [License](#license)



## Installation
- The GUI can simply be downloaded from the release section
- For running the GUI within your own python environemt just run the *main.py* file

### Prerequisites
- **Python Version:** 3.8+
- **Dependencies:**  
  Install required packages using:
  ```bash
  pip install -r requirements.txt
  ```

## Usage
The program itself is intuitive just follow the GUI. I am not a full time software engineer and this is in no means a perfect product, if it crashes just restart it, report the errors if you like and I will try my best to fix them :)

### Launcher Window
![Image](https://github.com/user-attachments/assets/11783475-2a91-4141-8101-7bb8e8cb67ca)

- Click on the tool you want to use
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Pixel to GCP converter
![Image](https://github.com/user-attachments/assets/f2c5d443-49a9-429f-8850-b6efcb3afeca)

#### Requirements
- A GCP file with latitude longitude and gcp_id columns
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Homography Matrix creator
![Image](https://github.com/user-attachments/assets/dbba3cd7-e109-4362-b03a-c4639818f71f)

- Calculate a homography matrix to georeference all your images
- If you have too many GCPs, try the simulated annealing option in the advanced setttings to identify the best subset of GCPs

#### Requirements
- Output file from the Pixel to GCP converter tool, a csv with these columns 'Pixel_X', 'Pixel_Y', 'Real_X', 'Real_Y'
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Georeferencing tool
![Image](https://github.com/user-attachments/assets/1b007ab0-517b-426c-99f0-009967267673)

- Georeference all your images
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Feature Identifier
Use HSV masking to idenify features additionally, manually edit features for best results. Three options are presented, Individual analysis that lets you experiment with single images, Machine learning that lets you cycle through a folder of images and batch processing if you want to completely rely on the automatic detection.

#### Automatic mode
![Image](https://github.com/user-attachments/assets/3a3abb66-9938-4d82-93c0-c66147afa488)
#### Manual editing
![Image](https://github.com/user-attachments/assets/cffc926c-0033-437c-b4a6-17ff03f08efb)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Wave run up analysis
![Image](https://github.com/user-attachments/assets/d2a0f3eb-0071-4e1f-bd4f-eda190cc72df)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## License
This project is licensed under the MIT License.

# Snippets of GUI

# Image Pixel to real world


