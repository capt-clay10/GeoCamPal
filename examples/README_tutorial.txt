# GeoCamPal – Example Files

This folder contains example and tutorial files for each module in the GeoCamPal toolbox. Each subfolder corresponds to an individual module and follows a consistent structure to help users get started quickly.

---

## Folder Structure

examples/
├── 01_module/
├── 02_module/
├── 03_module/
└── n_module/

Each module folder contains the following subfolders:

<module_folder>/
├── input/               # Example input files for the module
├── settings/            # .json configuration file with processing settings
└── expected_outputs/    # Pre-generated outputs using the provided settings
└── GUI screenshot       # Pre-generated screenshot of the GUI

---

## Getting Started

1. Upon launching an instance of GeoCamPal, navigate to the module folder you wish to run.
2. In GeoCamPal module of choice open the `.json` file located in the `settings/` subfolder.
3. **Update the input and output directory paths** to match your local file system.
4. Load the updated `.json` file into the corresponding GeoCamPal module.
5. Run the module and compare your results against the files in `expected_outputs/`.

---

## Important: Update File Paths

> The `.json` settings files contain **hardcoded directory paths** for input and output folders. These paths **must be updated** to reflect your own local file directory before running any module. Failing to do so may cause the module to not function correctly.

Look for the following fields in the `.json` file and update them accordingly:

{
  "input_folder": "C:/your/local/path/to/input",
  "output_folder": "C:/your/local/path/to/output"
}

Or update them directly from the GUI and save the settings again.
---

## Support

For issues or questions, please open a GitHub Issue.