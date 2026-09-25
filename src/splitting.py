"""
Composition-aware (a materials-science analogue of "scaffold splitting"
in cheminformatics) dataset splitting.

A plain random split can put two structures with the same or very
similar composition -- e.g. two DFT relaxations of the same compound,
or two polymorphs -- into different splits. A model can then partly
memorize composition-level patterns from training and get credit for
"generalizing" to a validation structure that's really a near-duplicate.
Grouping by chemical system and keeping whole groups together closes
that leak.
"""

import random
from collections import defaultdict


def get_chemical_system(cif_path: str) -> tuple[str, ...]:
    """Returns the sorted tuple of element symbols in a structure, e.g.
    ('Fe', 'O') for Fe2O3. Deliberately cheap: reads composition only,
    skipping the periodic neighbor search `cif_to_graph` does, since
    splitting only needs to know which structures share elements."""
    from pymatgen.core import Structure  # lazy: see src/dataset.py's note

    structure = Structure.from_file(cif_path)
    return tuple(sorted(str(el) for el in structure.composition.elements))


def composition_aware_split(
    chemical_systems: list[tuple[str, ...]], val_frac: float, test_frac: float, seed: int = 42
) -> tuple[list[int], list[int], list[int]]:
    """Splits indices [0, len(chemical_systems)) into (train, val, test)
    such that every index sharing a chemical system ends up in the same
    split. Whole groups are shuffled and greedily assigned to whichever
    of val/test is proportionally furthest below its target, with the
    remainder going to train.

    Degrades gracefully to (approximately) a random per-structure split
    whenever every structure has a unique chemical system -- verified:
    this repo's bundled 101-structure demo dataset has 100 unique
    systems out of 101, so this only meaningfully changes behavior once
    your dataset has real compositional duplicates (i.e., once you've
    pulled a larger set via scripts/download_materials_project.py).
    """
    groups = defaultdict(list)
    for idx, system in enumerate(chemical_systems):
        groups[system].append(idx)

    group_list = list(groups.values())
    random.Random(seed).shuffle(group_list)

    n_total = len(chemical_systems)
    val_target = n_total * val_frac
    test_target = n_total * test_frac

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for group in group_list:
        val_deficit = val_target - len(val_idx)
        test_deficit = test_target - len(test_idx)
        if val_deficit <= 0 and test_deficit <= 0:
            train_idx.extend(group)
        elif val_deficit >= test_deficit:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, val_idx, test_idx
