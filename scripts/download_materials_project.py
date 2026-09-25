"""
Downloads real crystal structures + DFT-computed bandgap/formation-energy
labels from the Materials Project, and writes them in exactly the format
`NanoGNNXDataset` expects: one .cif per structure, plus a matching
targets.csv with columns `filename,bandgap,formation_energy`.

This repo's own development environment cannot reach
api.materialsproject.org (see the README's notes on network access) --
this script has been reviewed against the real mp-api client surface
(mp_api.client.routes.materials.summary.SummaryRester.search, confirmed
via its actual source/docs: `num_sites`, `band_gap`, `fields`,
`chunk_size` are real parameters) but has NOT been run end-to-end here.
Run it yourself, locally, where you have both network access and a real
API key.

Setup:
    pip install mp-api
    export MP_API_KEY="your_key_here"   # get one at
                                         # https://next-gen.materialsproject.org/dashboard

Usage:
    # a quick, cheap sanity check first -- confirms your key works and
    # the output format is right before committing to a big download
    python scripts/download_materials_project.py --limit 50 --out /tmp/mp_test

    # the real pull -- merges into the repo's existing demo dataset by
    # default (safe to re-run; already-downloaded material_ids are
    # skipped rather than re-fetched or duplicated)
    python scripts/download_materials_project.py --limit 10000 --out data/raw_cifs

Then just: python train.py
"""

import argparse
import csv
import os
import sys


def load_existing_ids(targets_path: str) -> set:
    """Material IDs already present in an existing targets.csv, so a
    re-run (or a run into the same --out as a previous one) can skip
    work instead of re-downloading and duplicating rows."""
    if not os.path.exists(targets_path):
        return set()
    existing = set()
    with open(targets_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add(row["filename"])
    return existing


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max number of NEW structures to download (default: 10000). "
        "Structures already in an existing targets.csv at --out don't "
        "count against this limit.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw_cifs",
        help="Output directory for .cif files + targets.csv "
        "(default: data/raw_cifs, i.e. merge into this repo's dataset)",
    )
    parser.add_argument(
        "--max-sites",
        type=int,
        default=50,
        help="Skip structures with more than this many atoms in the "
        "conventional cell (default: 50). This repo's three-body "
        "message passing enumerates up to "
        "atoms * max_neighbors^2 triplets per structure (see "
        "src/config.py's max_neighbors, default 32) -- large unit "
        "cells make both CIF parsing and training dramatically "
        "slower. Raise this if you specifically need bigger "
        "structures and are prepared for the slowdown.",
    )
    parser.add_argument(
        "--chemsys",
        type=str,
        default=None,
        help="Optional chemical system filter, e.g. 'Li-Fe-O' or "
        "'Si-*' -- narrows the query if you want a specific "
        "materials family rather than a broad sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start targets.csv fresh instead of merging with whatever "
        "already exists at --out (the default, safer behavior).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Materials Project API key. Defaults to the MP_API_KEY "
        "environment variable if not given.",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit(
            "No API key found. Pass --api-key or set the MP_API_KEY environment "
            "variable (see https://next-gen.materialsproject.org/dashboard)."
        )

    try:
        from mp_api.client import MPRester
    except ImportError:
        sys.exit("mp-api isn't installed. Run: pip install mp-api")

    os.makedirs(args.out, exist_ok=True)
    targets_path = os.path.join(args.out, "targets.csv")

    already_have = set() if args.overwrite else load_existing_ids(targets_path)
    if already_have:
        print(
            f"Found {len(already_have)} structures already in {targets_path} -- "
            f"will skip those and fetch up to {args.limit} new ones."
        )

    print(
        f"Querying Materials Project for up to {args.limit} new structures "
        f"(<= {args.max_sites} sites each"
        + (f", chemsys={args.chemsys}" if args.chemsys else "")
        + ")..."
    )
    print(
        "This can take a while for a large --limit -- the API returns "
        "everything matching the filter in one call, paginated internally."
    )

    try:
        with MPRester(api_key) as mpr:
            search_kwargs = dict(
                num_sites=(1, args.max_sites),
                fields=["material_id", "structure", "band_gap", "formation_energy_per_atom"],
                chunk_size=1000,
            )
            if args.chemsys:
                search_kwargs["chemsys"] = args.chemsys
            docs = mpr.materials.summary.search(**search_kwargs)
    except Exception as exc:
        sys.exit(
            f"Materials Project query failed: {exc}\n"
            f"Common causes: an invalid/expired API key, or no network access "
            f"from this machine to api.materialsproject.org."
        )

    print(f"Materials Project returned {len(docs)} candidate structures before filtering.")

    new_rows = []
    written = 0
    skipped_incomplete = 0
    skipped_write_error = 0

    for doc in docs:
        if written >= args.limit:
            break

        filename = f"{doc.material_id}.cif"
        if filename in already_have:
            continue
        if doc.band_gap is None or doc.formation_energy_per_atom is None or doc.structure is None:
            skipped_incomplete += 1
            continue

        cif_path = os.path.join(args.out, filename)
        try:
            doc.structure.to(filename=cif_path)
        except Exception as exc:
            print(f"  skipping {doc.material_id}: failed to write CIF ({exc})")
            skipped_write_error += 1
            continue

        new_rows.append((filename, doc.band_gap, doc.formation_energy_per_atom))
        written += 1
        if written % 500 == 0:
            print(f"  ...{written} new structures written")

    # merge with whatever rows already existed (unless --overwrite)
    existing_rows = []
    if not args.overwrite and os.path.exists(targets_path):
        with open(targets_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            existing_rows = list(reader)

    with open(targets_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("filename", "bandgap", "formation_energy"))
        writer.writerows(existing_rows)
        writer.writerows(new_rows)

    total = len(existing_rows) + len(new_rows)
    print(f"\nDone. Wrote {written} new structures to {args.out}/")
    print(f"  skipped (missing bandgap/formation_energy/structure): {skipped_incomplete}")
    print(f"  skipped (CIF write error): {skipped_write_error}")
    print(f"targets.csv now has {total} total rows: {targets_path}")
    print("\nNext step: python train.py")


if __name__ == "__main__":
    main()
