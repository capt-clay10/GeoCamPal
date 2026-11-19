# GeoCamPal filter test: 50 example images

Example data for the filter sensitivity test of the Harmonise Images Module (GeoCamPal v1.0.4), described in Appendix A.3 of the supplementary material (Table A2, Figure A2).

## The test

- **Images:** the full test used 400 images from the Spiekeroog beach camera (CAM3, January to April 2024). They are a random sample of the benchmark run: 200 images that the filter removed and 200 that it kept.
- **This package:** 50 of those 400 images, chosen to show the test clearly. They are therefore not a random sample. The full results are in Table A2.
- **Names:** the images keep their test names (L001–L400). `image_list.csv` gives each image's original file name.
- **Filter on the full image:** the quality filter with the settings used in the paper. It checks every pixel of the image.
- **Filter inside an AOI:** the same settings, but the filter checks only a rectangular area of interest (AOI). The AOI leaves out the sky above the horizon: x 0–1542, y 148–1031, which is 85.6% of the 1544 × 1032 px image.
- **Human blind test:** one person labelled every image as *usable* or *not usable*, judging the whole image. The images were shown in random order, with no filter results.

## The 50 examples

| Example | Images | Filter on full image | Filter inside AOI | Human |
|---|---|---|---|---|
| Sky over-exposure: the bright sky counts as over-exposed | 20 | removed | kept | usable |
| Smooth sky: the plain sky lowers the sharpness below the blur limit | 5 | removed | kept | usable |
| Not usable: fog, a blurred lens, sun glare on the water, droplets | 15 | removed | removed | not usable |
| Good image | 10 | kept | kept | usable |

## Folders

| Folder / file | Content |
|---|---|
| `input/` | The 50 images. |
| `settings/` | Harmonise Images settings files, which the module can open with **Load Settings**: `harmonise_images_settings_full_image.json` (no AOI) and `harmonise_images_settings_aoi.json` (with the AOI, drawn on L001). |
| `output/full_image_filter/` | What the module writes after **Run Filter** on the full image: `bad_images.json` and `bad_images.txt` (the removed images and why) and `filter_summary.txt`. |
| `output/aoi_filter/` | The same files for the filter inside the AOI. |
| `output/human_blind_test/` | `labels.csv`: the human label of each image. |
| `image_list.csv` | One row per image: example type, original file name, the result and reasons of both filters, and the human label. |

## Notes

- Both filter runs used the filter code of GeoCamPal v1.0.4. The output files are the ones the module writes after **Run Filter**. The results for these 50 images are the same as in the full 400-image test, which for the AOI was run in the GUI.
- The folder paths in the settings and output files begin with the placeholder `C:/your/local/path/GeoCamPal/examples`, as in the other examples (see `README_tutorial.txt`). To repeat a run, replace the placeholder in a settings file with the location of the examples folder on your computer, or load the file and choose the input and output folders in the module, and click **Run Filter**. Tick **Export good filtered images** to also copy the kept images.
