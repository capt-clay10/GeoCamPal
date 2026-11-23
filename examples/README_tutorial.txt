# GeoCamPal v1.0.4 – Example Files

This folder contains one worked example for each module of the GeoCamPal toolbox. Every example was run with GeoCamPal v1.0.4: the settings files were saved by the v1.0.4 modules, and the expected outputs and GUI screenshots come from those runs.

---

## Folder Structure

examples/
├── 01_fov_generator/
├── 02_lens_correction/
├── ...
└── 13_wave_runup/

Each module folder contains the following:

<module_folder>/
├── input/                 # Example input files for the module
├── settings/              # .json settings file(s) saved by the module
├── expected_outputs/      # Outputs of the module run with these settings
└── GUI_screenshot_*.png   # The module window after the run

---

## Getting Started

1. Launch GeoCamPal and open the module you want to try.
2. Open the `.json` file in the example's `settings/` folder in a text editor and update the paths (see below).
3. In the module, click Load Settings and select the edited `.json` file.
4. Run the module (the buttons are listed under "Example notes") and compare your results with the files in `expected_outputs/`.

---

## Important: Update File Paths

> The `.json` settings files contain full paths for the input and output files and folders. All of them begin with the placeholder
>
>     C:/your/local/path/GeoCamPal/examples
>
> Replace this placeholder with the location of the examples folder on your computer, for example with "Replace all" in a text editor:
>
>     C:/your/local/path/GeoCamPal/examples/02_lens_correction/input/checkerboard_images
>     becomes
>     D:/Data/GeoCamPal/examples/02_lens_correction/input/checkerboard_images

If you load a settings file without changing it, GeoCamPal lists the paths it cannot find, clears them and loads all other settings. You can then select the folders and files with the module's Browse buttons and save the settings again.

---

## Example notes

Load the example's settings file first; then:

01 FOV Generator          Generate FOV Map.
02 Lens Correction        Generate Lens Correction File. The report (calibration_report.txt) ends with an Interpretation section.
03 Harmonise Images       One settings file per task, each with its own output folder:
                          filter_images -> Run Filter; harmonise_brightness -> Run Brightness;
                          harmonise_colour -> Run Colour; lens_correction -> Run Lens Correction.
                          Brightness, colour and lens correction write into a sub-folder named
                          after the input folder (for example harmonise_brightness_brightness_harmonised).
                          The lens calibration (input/raw_images/lens_correction) is the output of example 02.
04 Pixel to GCP           Start Process, click the GCP in each image and press Enter to go to the next
                          image; Enter on the last image saves the CSV. The GCP file has no row for
                          GCP_6, so that image is not in the CSV.
05 Homography Generator   Compute Matrix, then Compute Accuracy (writes the _accuracy.csv).
06 Georef Images          One settings file per method (homography, thin-plate spline, camera projection):
                          Initial Georeferencing, Set (only when an AOI is filled in), Secondary
                          Georeferencing, Process All (Compute Accuracy gives leave-one-out errors).
                          The buttons follow this order, also after Load Settings: Process All becomes
                          available with Secondary Georeferencing and writes what the right panel shows.
                          The outputs are named after the input images.
                          Turn the mouse wheel over a picture to zoom, hold the left button to move it,
                          double-click for the whole image. After each of the two georeferencing steps
                          the console gives the share of the preview that holds image data.
                          Homography and thin-plate spline use the panoramas in input/oblique_images
                          and gcp_to_pixel_coordinates.csv (the 124 GCPs that example 05 keeps); the
                          homography matrix is the output of example 05.
                          The thin-plate spline settings hold an AOI (208,15,1594,2860, in pixels of the
                          initial result): after Set, the secondary result and the GeoTIFFs are that
                          part, 1386 x 2845 px.
                          Camera projection uses the uncorrected camera frame in
                          input/oblique_images/camera_projection (the same frame as example1.bmp in 03),
                          pixel_gcp_file_cam2_example1.csv and the lens calibration of example 02, with
                          GCP_107 excluded. Camera projection applies the lens calibration itself, so it
                          needs an uncorrected frame. pixel_gcp_file_cam2.csv holds the same GCPs as
                          picked on lens-corrected frames; pixel_gcp_file_cam2_example1.csv has them
                          moved into the uncorrected frame.
                          S, N, E and W are multiples of the extent that the image corners span on the
                          ground: N and W enlarge the grid to the north and west, S and E cut it from
                          the south and east. In the camera projection example only the two bottom
                          corners of the frame reach the ground (the top ones lie above the horizon),
                          so that extent is 34.8 m x 13.8 m. N 30 and W 9 give a grid of 348 m x 427 m
                          (2295 x 2815 px), and the console reports image data in 43.2 % of it.
                          The GCP heights (Real_Z) are the GPS survey heights, as in examples 04 and 05.
                          Elev (m) is the height of the plane the image is projected onto; 0.4 m is the
                          mean height of the GCPs, which Georef fills in when the GCP file is loaded.
                          In this example a plane 0.2 m higher or lower moves the ground positions at
                          the GCPs by 2 m (median).
07 Feature Identifier     Single Image: Load Image (input/georeferenced_images/example_image.tif),
                          Load Settings, draw the sand bar as a polygon (Edit Detected Feature), enter
                          the Feature ID "sand bar" and click Export feature as training data. The
                          expected polygon was drawn by hand, so a polygon you draw will differ slightly.
08 Time Series Analysis   Run Analysis.
09 Colour Space Explorer  Run Analysis once each with the colour space set to HSV, LAB and RGB. After
                          each run, color_stats.csv and color_explorer_plots.png were copied to the
                          files with the colour space in their name; the files without it are from the
                          last run (RGB), as saved in the settings file.
10 DEM Generator          Generate next DEM (one day in this example).
11 Profile & Hovmöller    Generate Hovmöller (Load Sample Image shows the profile line on an image).
12 Burst Images Timestacker  Create Raw Timestack.
                          To keep the download small, the 590 frames are cut to the area around the
                          timestack line: 711 x 1028 px (about 118 m x 171 m) of the 2784 x 4530 px
                          frames. The georeferencing is kept, so the line is the same on the ground; the
                          timestack is identical to the one made from the full frames and is the input
                          of example 13.
13 Wave Run-up            The settings load the timestack and the mask; Calculate Run-up, then Export Run-up.
                          The record is of 22 March 2026. Beach Slope (3°) is the slope along the timestack
                          line, Slope (4°) the shore-normal foreshore slope; they differ because the line
                          runs about 50° off shore-normal. The two slopes, H₀ (0.5 m, from a wave hindcast)
                          and T₀ (15 s, the period of the swash) are rough estimates for that record.

---

## Comparing your results

Most expected outputs are reproduced byte for byte. A few files differ from run to run or between computers:

- filter_summary.txt and bad_images.json (03) contain the date and time of the run and the processing speed.
- fov_map.png (01) and color_explorer_plots*.png (09) are saved at the size of the plot in the window, so their size depends on the window size.
- The distribution and scatter panels of color_explorer_plots*.png (09) use a random sample of pixels; color_stats*.csv does not.
- Files that record a folder (calibration_report.txt in 02, filter_summary.txt and bad_images.json in 03) show the placeholder path instead of your own.

---

## Support

For issues or questions, please open a GitHub Issue.
