#!/usr/bin/env python3
"""
Extract vasprun.xml files from FigShare ZIP archives for a set of JIDs.
"""

import os
import io
import zipfile
import requests
from jarvis.db.figshare import data  # <-- the API wrapper you already use


# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------
DATASET = "dft_3d"  # FigShare dataset name
TARGET_JIDS = {"JVASP-1002", "JVASP-107"}  # JIDs to pull
TARGET_JIDS = ["JVASP-1002", "JVASP-107"]  # JIDs to pull
OUTPUT_DIR = "vasprun"  # Where extracted XMLs go


# ------------------------------------------------------------------
# 2. Helper: download + return a ZipFile *in memory*
# ------------------------------------------------------------------
def download_zip_to_memory(url: str) -> zipfile.ZipFile:
    """Download a zip file from `url` and return a ZipFile object
    backed by a BytesIO buffer (never touches disk)."""
    r = requests.get(url, stream=True)
    r.raise_for_status()  # will raise for 4xx/5xx status
    # IO: the zip data is kept in memory
    return zipfile.ZipFile(io.BytesIO(r.content))


# ------------------------------------------------------------------
# 3. Main loop: fetch metadata, filter, download, extract
# ------------------------------------------------------------------
def main() -> None:
    # Create output dir if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Grab the FigShare meta‑data for the whole dataset
    records = data(DATASET)

    for rec in records:
        jid = rec["jid"]
        if jid not in TARGET_JIDS:
            continue  # skip unwanted JIDs

        print(f"\nProcessing {jid}")

        for raw in rec["raw_files"]:
            # The raw entry looks like:
            #  "TBMBJ,JVASP-32484.zip,https://ndownloader.figshare.com/files/23638901"
            if not raw.startswith("TBMBJ"):
                continue

            # Grab the download link (third column)
            parts = raw.split(",")
            if len(parts) < 3:
                print(f"  Skipping malformed raw entry: {raw}")
                continue
            zip_url = parts[-1].strip()

            print(f"  Downloading {zip_url}")

            # Pull the archive into a ZipFile object
            try:
                z = download_zip_to_memory(zip_url)
            except Exception as exc:
                print(f"  ❌ Failed to download {zip_url}: {exc}")
                continue

            # Inspect the archive: what files are inside?
            archive_files = z.namelist()
            print(f"  Files inside zip: {', '.join(archive_files)}")

            # Find a vasprun.xml (sometimes nested in a folder)
            xml_candidates = [
                f for f in archive_files if f.lower().endswith("vasprun.xml")
            ]
            if not xml_candidates:
                print(f"  ‼️ No vasprun.xml found in this archive.")
                continue

            # There should normally be just one – but just in case we take the first
            xml_path_in_zip = xml_candidates[0]
            print(f"  Extracting {xml_path_in_zip}")

            # Build the target path:
            #   <OUTPUT_DIR>/<JID>_<relative_path_of_xml_in_zip>
            #   e.g., vasprun/JVASP-1002_vasprun.xml
            #   or vasprun/JVASP-1002_subdir/vasprun.xml
            # We'll preserve the internal folder structure to avoid clashes.
            output_path = os.path.join(
                OUTPUT_DIR, f"{jid}_{xml_path_in_zip.replace(os.sep, '_')}"
            )

            # Make sure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Write the XML out
            with open(output_path, "wb") as out_f:
                out_f.write(z.read(xml_path_in_zip))

            print(f"  ✓ Saved to {output_path}")

            # If you want to keep the in‑memory ZipFile alive for the next file
            # you can just leave `z` as is – it will be closed automatically
            # when the function exits.

    print("\nAll done!")


# ------------------------------------------------------------------
# 4. Kick it off
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
