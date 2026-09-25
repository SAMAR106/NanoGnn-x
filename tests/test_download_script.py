"""
Tests scripts/download_materials_project.py's logic against a mocked
mp-api client -- no real network access or API key needed, but this
still genuinely exercises the CSV merge/skip/filter logic rather than
just trusting the script compiles.
"""

import csv
import importlib.util
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "download_materials_project.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("download_materials_project", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download_mod = _load_script_module()


def _write_fake_cif(filename):
    with open(filename, "w") as f:
        f.write("fake cif")


def _fake_doc(material_id, band_gap=1.0, formation_energy_per_atom=-1.0, has_structure=True):
    structure = MagicMock()
    structure.to = MagicMock(side_effect=_write_fake_cif)
    return SimpleNamespace(
        material_id=material_id,
        band_gap=band_gap,
        formation_energy_per_atom=formation_energy_per_atom,
        structure=structure if has_structure else None,
    )


class _FakeMPRester:
    def __init__(self, docs):
        self._docs = docs
        self.materials = SimpleNamespace(
            summary=SimpleNamespace(search=MagicMock(return_value=docs))
        )

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fresh_download_writes_cifs_and_targets_csv(tmp_path, monkeypatch):
    docs = [_fake_doc(f"mp-{i}") for i in range(5)]
    fake_mp_api = SimpleNamespace(client=SimpleNamespace(MPRester=lambda key: _FakeMPRester(docs)))
    monkeypatch.setitem(sys.modules, "mp_api", fake_mp_api)
    monkeypatch.setitem(sys.modules, "mp_api.client", fake_mp_api.client)

    out_dir = str(tmp_path / "out")
    argv = ["prog", "--limit", "10", "--out", out_dir, "--api-key", "fake-key"]
    with patch.object(sys, "argv", argv):
        download_mod.main()

    targets_path = os.path.join(out_dir, "targets.csv")
    assert os.path.exists(targets_path)
    with open(targets_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert {r["filename"] for r in rows} == {f"mp-{i}.cif" for i in range(5)}
    for i in range(5):
        assert os.path.exists(os.path.join(out_dir, f"mp-{i}.cif"))


def test_skips_docs_missing_labels_or_structure(tmp_path, monkeypatch):
    docs = [
        _fake_doc("mp-good", band_gap=2.0, formation_energy_per_atom=-1.5),
        _fake_doc("mp-no-gap", band_gap=None),
        _fake_doc("mp-no-formation", formation_energy_per_atom=None),
        _fake_doc("mp-no-structure", has_structure=False),
    ]
    fake_mp_api = SimpleNamespace(client=SimpleNamespace(MPRester=lambda key: _FakeMPRester(docs)))
    monkeypatch.setitem(sys.modules, "mp_api", fake_mp_api)
    monkeypatch.setitem(sys.modules, "mp_api.client", fake_mp_api.client)

    out_dir = str(tmp_path / "out")
    argv = ["prog", "--limit", "10", "--out", out_dir, "--api-key", "fake-key"]
    with patch.object(sys, "argv", argv):
        download_mod.main()

    with open(os.path.join(out_dir, "targets.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["filename"] == "mp-good.cif"


def test_rerun_merges_and_skips_already_downloaded(tmp_path, monkeypatch):
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir)
    # simulate a prior run: mp-0 already downloaded
    with open(os.path.join(out_dir, "targets.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("filename", "bandgap", "formation_energy"))
        writer.writerow(("mp-0.cif", 1.0, -1.0))
    with open(os.path.join(out_dir, "mp-0.cif"), "w") as f:
        f.write("existing cif")

    # the API returns mp-0 again (should be skipped) plus a genuinely new mp-1
    docs = [_fake_doc("mp-0"), _fake_doc("mp-1")]
    fake_mp_api = SimpleNamespace(client=SimpleNamespace(MPRester=lambda key: _FakeMPRester(docs)))
    monkeypatch.setitem(sys.modules, "mp_api", fake_mp_api)
    monkeypatch.setitem(sys.modules, "mp_api.client", fake_mp_api.client)

    argv = ["prog", "--limit", "10", "--out", out_dir, "--api-key", "fake-key"]
    with patch.object(sys, "argv", argv):
        download_mod.main()

    with open(os.path.join(out_dir, "targets.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # merged, not duplicated
    assert {r["filename"] for r in rows} == {"mp-0.cif", "mp-1.cif"}


def test_overwrite_flag_discards_existing_rows(tmp_path, monkeypatch):
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "targets.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("filename", "bandgap", "formation_energy"))
        writer.writerow(("mp-old.cif", 1.0, -1.0))

    docs = [_fake_doc("mp-new")]
    fake_mp_api = SimpleNamespace(client=SimpleNamespace(MPRester=lambda key: _FakeMPRester(docs)))
    monkeypatch.setitem(sys.modules, "mp_api", fake_mp_api)
    monkeypatch.setitem(sys.modules, "mp_api.client", fake_mp_api.client)

    argv = ["prog", "--limit", "10", "--out", out_dir, "--api-key", "fake-key", "--overwrite"]
    with patch.object(sys, "argv", argv):
        download_mod.main()

    with open(os.path.join(out_dir, "targets.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["filename"] == "mp-new.cif"


def test_respects_limit(tmp_path, monkeypatch):
    docs = [_fake_doc(f"mp-{i}") for i in range(20)]
    fake_mp_api = SimpleNamespace(client=SimpleNamespace(MPRester=lambda key: _FakeMPRester(docs)))
    monkeypatch.setitem(sys.modules, "mp_api", fake_mp_api)
    monkeypatch.setitem(sys.modules, "mp_api.client", fake_mp_api.client)

    out_dir = str(tmp_path / "out")
    argv = ["prog", "--limit", "3", "--out", out_dir, "--api-key", "fake-key"]
    with patch.object(sys, "argv", argv):
        download_mod.main()

    with open(os.path.join(out_dir, "targets.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3


def test_missing_api_key_exits_cleanly(tmp_path, monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    argv = ["prog", "--limit", "10", "--out", str(tmp_path / "out")]
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit):
        download_mod.main()
