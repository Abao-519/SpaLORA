import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

from SpaLORA.night23b_signed_bridge import (
    carrier_preserving_signed_partition,
    positive_connectivity_audit,
    signed_edge_energy,
    tri_state_from_thresholds,
)


def equivalent(a, b):
    k = max(a.max(), b.max()) + 1
    table = np.zeros((k, k), dtype=int)
    np.add.at(table, (a, b), 1)
    left, right = linear_sum_assignment(-table)
    return int(table[left, right].sum()) == len(a)


def toy():
    rng = np.random.default_rng(23)
    teacher = np.repeat(np.arange(3), 8)
    carrier = np.column_stack([teacher + rng.normal(0, 0.25, len(teacher)), rng.normal(size=len(teacher))])
    rows, cols = [], []
    for i in range(len(teacher)):
        for j in range(i + 1, len(teacher)):
            if teacher[i] == teacher[j] or (i + j) % 7 == 0:
                rows.append(i); cols.append(j)
    rows, cols = np.asarray(rows), np.asarray(cols)
    relation = teacher[rows] == teacher[cols]
    return carrier, teacher, rows, cols, relation


def test_tri_state_is_mutually_exclusive_complete_and_unknown_erased():
    score = np.asarray([0.01, 0.20, 0.50, 0.81, 0.99])
    positive, negative, unknown = tri_state_from_thresholds(score, 0.2, 0.8)
    assert np.all(positive.astype(int) + negative.astype(int) + unknown.astype(int) == 1)
    assert not positive[2] and not negative[2] and unknown[2]


def test_positive_and_negative_energy_directions_are_correct():
    rows = np.asarray([0, 1]); cols = np.asarray([1, 2])
    positive = np.asarray([1.0, 0.0]); negative = np.asarray([0.0, 1.0])
    correct = np.asarray([0, 0, 1]); wrong = np.asarray([0, 1, 1])
    assert signed_edge_energy(correct, rows, cols, positive, negative) < signed_edge_energy(wrong, rows, cols, positive, negative)


def test_oracle_consumer_exact_k_and_fresh_repeat():
    carrier, teacher, rows, cols, relation = toy()
    first, _ = carrier_preserving_signed_partition(carrier, rows, cols, relation.astype(float), (~relation).astype(float), 3, 4.0, carrier_dimensions=2)
    second, _ = carrier_preserving_signed_partition(carrier, rows, cols, relation.astype(float), (~relation).astype(float), 3, 4.0, carrier_dimensions=2)
    assert len(np.unique(first)) == 3
    assert np.array_equal(first, second)


def test_node_permutation_equivariance_up_to_cluster_labels():
    carrier, teacher, rows, cols, relation = toy()
    base, _ = carrier_preserving_signed_partition(carrier, rows, cols, relation.astype(float), (~relation).astype(float), 3, 4.0, carrier_dimensions=2)
    perm = np.random.default_rng(5).permutation(len(carrier))
    inverse = np.empty(len(perm), dtype=int); inverse[perm] = np.arange(len(perm))
    moved_rows, moved_cols = inverse[rows], inverse[cols]
    swap = moved_rows > moved_cols
    moved_rows[swap], moved_cols[swap] = moved_cols[swap].copy(), moved_rows[swap].copy()
    order = np.lexsort((moved_cols, moved_rows))
    moved, _ = carrier_preserving_signed_partition(carrier[perm], moved_rows[order], moved_cols[order], relation[order].astype(float), (~relation[order]).astype(float), 3, 4.0, carrier_dimensions=2)
    restored = np.empty_like(moved); restored[perm] = moved
    assert equivalent(base, restored)


def test_positive_connectivity_respects_teacher_components():
    carrier, teacher, rows, cols, relation = toy()
    audit = positive_connectivity_audit(len(carrier), rows, cols, relation, teacher)
    assert not audit["cross_teacher_cluster_positive_component"]
    assert audit["positive_component_count"] >= len(np.unique(teacher))
