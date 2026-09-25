"""
Converts the MP-20 benchmark dataset -- 45,231 real Materials Project
structures with DFT-computed formation energy and band gap, introduced
by Xie et al., "Crystal Diffusion Variational Autoencoder for Periodic
Material Generation" (CDVAE), ICLR 2022 -- into this repo's CIF +
targets.csv format.

Unlike scripts/download_materials_project.py, this needs **no API key**
and pulls from GitHub (github.com / codeload.github.com), not
api.materialsproject.org -- reachable from environments (like the one
this repo was built in) that can reach GitHub but not the Materials
Project API directly. Source:
https://github.com/txie-93/cdvae/tree/main/data/mp_20 (MIT licensed,
Copyright (c) 2021 Tian Xie, Xiang Fu).

MP-20 caps every structure at <=20 atoms per unit cell, which -- as a
side effect -- keeps this repo's O(atoms * max_neighbors^2) triplet
count far more uniform than the original 101-structure demo set (which
had one 136-atom outlier producing 135,000 triplets on its own).

Usage:
    # downloads ~80 MB (the whole cdvae repo, simplest reliable way to
    # get the three CSVs) and extracts up to --limit structures per split
    python scripts/download_mp20.py --limit 2000 --out data/mp20

    python train.py  # then point CFG.raw_cif_dir at data/mp20, or pass
                      # --raw-cif-dir data/mp20 if you've added that flag
"""
import argparse
import csv
import io
import os
import sys
import tarfile
import urllib.request

CDVAE_TARBALL_URL = "https://codeload.github.com/txie-93/cdvae/tar.gz/refs/heads/main"


def _download_mp20_csvs(cache_dir: str) -> dict:
    """Downloads the cdvae repo tarball (only way the raw MP-20 CSVs are
    published -- there's no standalone MP-20-only download) and extracts
    just the three data/mp_20/*.csv members, caching them locally so a
    re-run of this script doesn't re-download ~80 MB every time."""
    cached = {
        split: os.path.join(cache_dir, f"mp20_{split}.csv") for split in ("train", "val", "test")
    }
    if all(os.path.exists(p) for p in cached.values()):
        print(f"Using cached MP-20 CSVs in {cache_dir}/")
        return cached

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading {CDVAE_TARBALL_URL} (~80 MB, one-time)...")
    with urllib.request.urlopen(CDVAE_TARBALL_URL, timeout=120) as resp:
        tarball_bytes = resp.read()
    print(f"  downloaded {len(tarball_bytes) / 1e6:.1f} MB, extracting mp_20 CSVs...")

    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tf:
        for split, out_path in cached.items():
            member_path = f"cdvae-main/data/mp_20/{split}.csv"
            member = tf.extractfile(member_path)
            if member is None:
                raise RuntimeError(f"{member_path} not found in the cdvae tarball -- "
                                    f"the upstream repo layout may have changed.")
            with open(out_path, "wb") as f:
                f.write(member.read())
    return cached


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=2000,
                         help="Max structures per split (train/val/test) to extract "
                              "(default: 2000 -- roughly matches the original CDVAE "
                              "60/20/20 split proportions if you also raise val/test "
                              "accordingly; there are 27,136 available for train, "
                              "9,047 for val, 9,046 for test if you want it all)")
    parser.add_argument("--out", type=str, default="data/mp20",
                         help="Output directory for .cif files + targets.csv")
    parser.add_argument("--cache-dir", type=str, default="/tmp/mp20_cache",
                         help="Where to cache the downloaded CSVs between runs")
    parser.add_argument("--max-sites", type=int, default=20,
                         help="MP-20 already caps every structure at 20 atoms/cell "
                              "by construction, so this mostly documents that fact "
                              "rather than actively filtering anything.")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required (already in requirements.txt): pip install pandas")

    csv_paths = _download_mp20_csvs(args.cache_dir)
    os.makedirs(args.out, exist_ok=True)

    all_rows = [("filename", "bandgap", "formation_energy")]
    written = 0
    for split, csv_path in csv_paths.items():
        df = pd.read_csv(csv_path)
        n = min(args.limit, len(df))
        print(f"{split}: extracting {n} of {len(df)} available structures...")
        for _, row in df.head(n).iterrows():
            filename = f"mp20-{split}-{row['material_id']}.cif"
            with open(os.path.join(args.out, filename), "w") as f:
                f.write(row["cif"])
            all_rows.append((filename, row["band_gap"], row["formation_energy_per_atom"]))
            written += 1

    targets_path = os.path.join(args.out, "targets.csv")
    with open(targets_path, "w", newline="") as f:
        csv.writer(f).writerows(all_rows)

    print(f"\nWrote {written} real MP-20 structures + labels to {args.out}/")
    print(f"targets.csv: {targets_path}")
    print("\nCitation (please keep if you publish results using this data):")
    print("  Xie et al., \"Crystal Diffusion Variational Autoencoder for Periodic")
    print("  Material Generation\", ICLR 2022. Structures originate from the")
    print("  Materials Project (Jain et al., 2013).")


if __name__ == "__main__":
    main()
