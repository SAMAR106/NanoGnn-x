"""
Unlike tests/test_download_script.py (which mocks the Materials Project
API, since that's genuinely unreachable from this repo's environment),
this tests scripts/download_mp20.py against the *real* downloaded CSVs
-- GitHub, unlike api.materialsproject.org, is actually reachable from
here, so this script could be tested for real rather than mocked. See
its docstring and the README's Dataset section for how it was verified.
"""
import csv
import importlib.util
import os

import pytest

pd = pytest.importorskip("pandas")

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "download_mp20.py")
CACHE_DIR = "/tmp/mp20_cache"  # populated once while building this repo; see README


def _load_script_module():
    spec = importlib.util.spec_from_file_location("download_mp20", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download_mod = _load_script_module()

# These hit the real cached CSVs (or trigger a real ~80MB download on a
# machine that hasn't cached them yet -- skipped if that's not
# available, rather than failing the whole suite over a slow network
# fetch in an unrelated test run).
_cache_available = os.path.exists(os.path.join(CACHE_DIR, "mp20_train.csv"))


@pytest.mark.skipif(
    not _cache_available, reason="MP-20 CSVs not cached locally and not re-downloading in CI"
)
def test_download_mp20_extracts_real_structures(tmp_path):
    import sys
    from unittest.mock import patch

    out_dir = str(tmp_path / "out")
    argv = ["prog", "--limit", "10", "--out", out_dir, "--cache-dir", CACHE_DIR]
    with patch.object(sys, "argv", argv):
        download_mod.main()

    targets_path = os.path.join(out_dir, "targets.csv")
    assert os.path.exists(targets_path)
    with open(targets_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30  # 10 per split x 3 splits

    for row in rows:
        cif_path = os.path.join(out_dir, row["filename"])
        assert os.path.exists(cif_path)
        assert os.path.getsize(cif_path) > 0
        float(row["bandgap"])  # real numeric values, not placeholders
        float(row["formation_energy"])


@pytest.mark.skipif(
    not _cache_available, reason="MP-20 CSVs not cached locally and not re-downloading in CI"
)
def test_extracted_cifs_parse_through_real_pipeline(tmp_path):
    """The actual point of this dataset: confirm extracted structures
    are usable by this repo's own graph-building code, not just that
    the script writes files that look plausible."""
    import sys
    from unittest.mock import patch

    from src.dataset import cif_to_graph

    out_dir = str(tmp_path / "out")
    argv = ["prog", "--limit", "5", "--out", out_dir, "--cache-dir", CACHE_DIR]
    with patch.object(sys, "argv", argv):
        download_mod.main()

    df = pd.read_csv(os.path.join(out_dir, "targets.csv"))
    for _, row in df.iterrows():
        data = cif_to_graph(os.path.join(out_dir, row["filename"]))
        assert data.num_nodes >= 1
