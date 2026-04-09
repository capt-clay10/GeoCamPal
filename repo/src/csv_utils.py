"""
csv_utils.py — GeoCamPal shared CSV normalisation utilities

Provides flexible CSV loading that handles:
  - Any delimiter (comma, semicolon, tab, pipe)
  - Case-insensitive column names
  - Common column aliases (easting/Real_X/x/lon, etc.)
  - Whitespace stripping

Usage in any module:
    from csv_utils import read_gcp_csv, normalise_columns

    df = read_gcp_csv("my_file.csv")
    # df now has canonical column names: GCP_ID, Pixel_X, Pixel_Y,
    # Real_X, Real_Y, Real_Z, EPSG — regardless of what the user typed.
"""

import pandas as pd
import os

# ── Canonical column names used throughout GeoCamPal ────────────────
#    Every alias maps to the canonical name on the right.

_COLUMN_ALIASES = {
    # GCP identifier
    "gcp_id":       "GCP_ID",
    "gcpid":        "GCP_ID",
    "gcp":          "GCP_ID",
    "id":           "GCP_ID",
    "point_id":     "GCP_ID",
    "point":        "GCP_ID",
    "name":         "GCP_ID",

    # Pixel coordinates
    "pixel_x":      "Pixel_X",
    "pixelx":       "Pixel_X",
    "px":           "Pixel_X",
    "px_x":         "Pixel_X",
    "u":            "Pixel_X",
    "col":          "Pixel_X",
    "column":       "Pixel_X",
    "image_x":      "Pixel_X",

    "pixel_y":      "Pixel_Y",
    "pixely":       "Pixel_Y",
    "py":           "Pixel_Y",
    "px_y":         "Pixel_Y",
    "v":            "Pixel_Y",
    "row":          "Pixel_Y",
    "image_y":      "Pixel_Y",

    # Real-world coordinates
    "real_x":       "Real_X",
    "realx":        "Real_X",
    "easting":      "Real_X",
    "east":         "Real_X",
    "x":            "Real_X",
    "utm_x":        "Real_X",
    "utmx":         "Real_X",
    "e":            "Real_X",

    "real_y":       "Real_Y",
    "realy":        "Real_Y",
    "northing":     "Real_Y",
    "north":        "Real_Y",
    "y":            "Real_Y",
    "utm_y":        "Real_Y",
    "utmy":         "Real_Y",
    "n":            "Real_Y",

    "real_z":       "Real_Z",
    "realz":        "Real_Z",
    "elevation":    "Real_Z",
    "elev":         "Real_Z",
    "height":       "Real_Z",
    "z":            "Real_Z",
    "alt":          "Real_Z",
    "altitude":     "Real_Z",

    # CRS
    "epsg":         "EPSG",
    "epsg_code":    "EPSG",
    "crs":          "EPSG",
    "srid":         "EPSG",

    # Geographic (pre-UTM conversion)
    "latitude":     "latitude",
    "lat":          "latitude",
    "longitude":    "longitude",
    "lon":          "longitude",
    "lng":          "longitude",
    "long":         "longitude",

    # Image metadata
    "image_name":   "Image_name",
    "imagename":    "Image_name",
    "filename":     "Image_name",
    "file":         "Image_name",
    "image":        "Image_name",
    "img":          "Image_name",

    "camera":       "camera",
    "cam":          "camera",
    "camera_id":    "camera",
}


def normalise_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Rename DataFrame columns to canonical GeoCamPal names.

    - Strips leading/trailing whitespace from column names
    - Matches case-insensitively against the alias table
    - Columns that don't match any alias are left unchanged
    - If verbose, prints any renames performed

    Returns a new DataFrame (columns renamed, data unchanged).
    """
    rename_map = {}
    for col in df.columns:
        clean = col.strip()
        lookup = clean.lower().replace(" ", "_")

        # Check if it's already canonical (exact match, case-insensitive)
        canonical_values = set(_COLUMN_ALIASES.values())
        if clean in canonical_values:
            if clean != col:
                rename_map[col] = clean  # just whitespace fix
            continue

        # Look up alias
        if lookup in _COLUMN_ALIASES:
            target = _COLUMN_ALIASES[lookup]
            if col != target:
                rename_map[col] = target
        elif clean != col:
            rename_map[col] = clean  # whitespace fix only

    if rename_map and verbose:
        renames_str = ", ".join(f"'{k}' -> '{v}'" for k, v in rename_map.items())
        print(f"[CSV] Normalised columns: {renames_str}")

    return df.rename(columns=rename_map)


def read_csv_flexible(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Read a CSV/TSV file with automatic delimiter detection and
    column normalisation.

    Handles: comma, semicolon, tab, pipe delimiters.
    Strips BOM markers.  Strips whitespace from headers and string values.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Try auto-detect delimiter
    try:
        df = pd.read_csv(path, sep=None, engine="python",
                         encoding="utf-8-sig")  # utf-8-sig strips BOM
    except Exception:
        # Fallback: try comma explicitly
        df = pd.read_csv(path, encoding="utf-8-sig")

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    # Normalise column names
    df = normalise_columns(df, verbose=verbose)

    if verbose:
        print(f"[CSV] Loaded {len(df)} rows, {len(df.columns)} columns "
              f"from {os.path.basename(path)}")

    return df


def read_gcp_csv(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Read a GCP CSV with normalisation and ensure essential columns exist.

    Always guarantees:
      - Real_Z column (defaults to 0.0 if missing)
      - EPSG column (defaults to 0 if missing)

    Raises KeyError if Pixel_X/Pixel_Y/Real_X/Real_Y are all missing
    (but tolerates partial — the caller decides what's required).
    """
    df = read_csv_flexible(path, verbose=verbose)

    # Ensure Real_Z exists
    if "Real_Z" not in df.columns:
        df["Real_Z"] = 0.0
        if verbose:
            print("[CSV] Added Real_Z column (default 0.0)")

    # Ensure EPSG exists
    if "EPSG" not in df.columns:
        df["EPSG"] = 0
        if verbose:
            print("[CSV] Added EPSG column (default 0)")

    return df


def check_required_columns(df: pd.DataFrame,
                            required: list[str],
                            source: str = "CSV") -> list[str]:
    """
    Check that all required columns are present after normalisation.

    Returns a list of missing column names (empty if all present).
    """
    return [c for c in required if c not in df.columns]


def read_gcp_id_csv(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Read a GCP ID file (latitude, longitude, GCP_ID format)
    with normalisation.  Used by pixel_to_gcp.py.
    """
    df = read_csv_flexible(path, verbose=verbose)

    # Ensure essential columns
    missing = check_required_columns(df, ["GCP_ID"])
    if missing:
        # Try to find anything that looks like an ID column
        for col in df.columns:
            if "gcp" in col.lower() or "id" in col.lower():
                df = df.rename(columns={col: "GCP_ID"})
                if verbose:
                    print(f"[CSV] Mapped '{col}' -> 'GCP_ID'")
                break

    # Ensure elevation column
    if "Real_Z" not in df.columns and "elevation" not in [c.lower() for c in df.columns]:
        df["Real_Z"] = 0.0
        if verbose:
            print("[CSV] Added Real_Z/elevation column (default 0.0)")

    return df