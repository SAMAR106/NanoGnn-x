from src.splitting import composition_aware_split, get_chemical_system


def test_get_chemical_system_on_real_cif():
    assert get_chemical_system("data/raw_cifs/NaCl.cif") == ("Cl", "Na")


def test_no_chemical_system_crosses_splits():
    """The actual guarantee this module exists for: a chemical system
    with multiple structures (e.g. several polymorphs of the same
    compound) must land entirely in one split, never spread across
    train/val/test."""
    systems = (
        [("Fe", "O")] * 5  # 5 structures sharing one system
        + [("Li", "Fe", "O")] * 3
        + [("Na", "Cl")] * 1
        + [(f"El{i}",) for i in range(20)]  # 20 structures, all unique
    )
    train_idx, val_idx, test_idx = composition_aware_split(
        systems, val_frac=0.15, test_frac=0.15, seed=0
    )

    def systems_in(indices):
        return {systems[i] for i in indices}

    train_systems = systems_in(train_idx)
    val_systems = systems_in(val_idx)
    test_systems = systems_in(test_idx)
    assert not (train_systems & val_systems)
    assert not (train_systems & test_systems)
    assert not (val_systems & test_systems)

    # every index assigned exactly once
    all_idx = sorted(train_idx + val_idx + test_idx)
    assert all_idx == list(range(len(systems)))


def test_split_proportions_roughly_match_targets_when_all_unique():
    """With no real duplicates (the bundled demo dataset's actual case
    -- 100 of 101 systems unique), this should behave close to a plain
    random split."""
    systems = [(f"El{i}",) for i in range(200)]
    train_idx, val_idx, test_idx = composition_aware_split(
        systems, val_frac=0.1, test_frac=0.1, seed=1
    )
    assert abs(len(val_idx) - 20) <= 2
    assert abs(len(test_idx) - 20) <= 2
    assert len(train_idx) + len(val_idx) + len(test_idx) == 200


def test_deterministic_given_same_seed():
    systems = [("A", "B")] * 4 + [("C",)] * 6 + [(f"El{i}",) for i in range(10)]
    split1 = composition_aware_split(systems, 0.2, 0.2, seed=7)
    split2 = composition_aware_split(systems, 0.2, 0.2, seed=7)
    assert split1 == split2


def test_different_seeds_can_differ():
    systems = [(f"El{i}",) for i in range(50)]
    split_a = composition_aware_split(systems, 0.2, 0.2, seed=1)
    split_b = composition_aware_split(systems, 0.2, 0.2, seed=2)
    assert split_a != split_b


def test_single_dominant_group_all_goes_to_one_split():
    """A degenerate case: one chemical system accounts for most of the
    dataset. The whole group must still land in exactly one split, even
    though that necessarily blows past that split's proportional
    target -- keeping the no-leakage guarantee always wins over hitting
    the target ratio exactly."""
    systems = [("Fe", "O")] * 18 + [(f"El{i}",) for i in range(2)]
    train_idx, val_idx, test_idx = composition_aware_split(systems, 0.1, 0.1, seed=3)

    fe_o_indices = {i for i, s in enumerate(systems) if s == ("Fe", "O")}
    for split in (train_idx, val_idx, test_idx):
        split_set = set(split)
        # either none or all of the Fe-O indices are in this split
        assert split_set.isdisjoint(fe_o_indices) or fe_o_indices.issubset(split_set)
