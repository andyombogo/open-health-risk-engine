"""
download_data.py
----------------
Downloads the required NHANES 2017-2020 (pre-pandemic) XPT files from the CDC website.
Files are saved to data/raw/. Safe to re-run — skips already-downloaded files.

NHANES files used:
  - DPQ_J  : Depression screener (PHQ-9) — our TARGET variable
  - PAQ_J  : Physical activity
  - SLQ_J  : Sleep disorders / duration
  - BMX_J  : Body measures (BMI)
  - ALQ_J  : Alcohol use
  - DEMO_J : Demographics (age, sex, income, education)

CDC base URL: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/
"""

import requests
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CDC_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

NHANES_FILES = {
    # (filename, CDC URL)
    "P_DPQ.XPT": f"{CDC_BASE_URL}/P_DPQ.xpt",
    "P_PAQ.XPT": f"{CDC_BASE_URL}/P_PAQ.xpt",
    "P_SLQ.XPT": f"{CDC_BASE_URL}/P_SLQ.xpt",
    "P_BMX.XPT": f"{CDC_BASE_URL}/P_BMX.xpt",
    "P_ALQ.XPT": f"{CDC_BASE_URL}/P_ALQ.xpt",
    "P_DEMO.XPT": f"{CDC_BASE_URL}/P_DEMO.xpt",
}


def is_valid_xpt(path: Path) -> bool:
    """Return True when the file looks like a SAS transport file."""
    if not path.exists() or path.stat().st_size < 1024:
        return False

    with open(path, "rb") as file_obj:
        header = file_obj.read(128)

    return b"HEADER RECORD" in header


def download_file(url: str, dest: Path) -> None:
    """Stream-download a file with a progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    temp_dest = dest.with_suffix(dest.suffix + ".tmp")

    with open(temp_dest, "wb") as f, tqdm(
        desc=dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    if not is_valid_xpt(temp_dest):
        temp_dest.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded content from {url} is not a valid NHANES XPT file."
        )

    temp_dest.replace(dest)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving files to: {RAW_DIR}\n")

    for filename, url in NHANES_FILES.items():
        dest = RAW_DIR / filename
        if is_valid_xpt(dest):
            print(f"  [skip] {filename} already exists.")
            continue
        if dest.exists():
            print(f"  [redo] {filename} exists but is invalid. Re-downloading.")
        print(f"  Downloading {filename} ...")
        try:
            download_file(url, dest)
            print(f"  [ok]   {filename} saved.")
        except Exception as e:
            print(f"  [ERROR] Could not download {filename}: {e}")

    print("\nAll downloads complete. Run data_cleaning.py next.")


if __name__ == "__main__":
    main()
