import torch
from torch_geometric.data import Batch

from src.dataset import CrystalData


def test_triplet_index_offsets_by_edge_count_not_node_count():
    """Regression test for a real bug: PyG's default Batch collation
    auto-increments any "*index*" attribute by the running *node* count,
    which is wrong for triplet_index (it indexes into edges). Using a
    graph where node_count != edge_count is what actually exposes the
    bug -- a graph where they happen to be equal would pass even with
    the bug present."""
    d1 = CrystalData(
        x=torch.tensor([1, 2]),
        edge_index=torch.tensor([[0, 1, 0], [1, 0, 0]]),  # 2 nodes, 3 edges
        triplet_index=torch.tensor([[0], [1]]),
        num_nodes=2,
    )
    d2 = CrystalData(
        x=torch.tensor([3, 4, 5, 6]),
        edge_index=torch.tensor([[0, 1], [1, 2]]),  # 4 nodes, 2 edges
        triplet_index=torch.tensor([[0], [1]]),
        num_nodes=4,
    )

    batch = Batch.from_data_list([d1, d2])

    # d1 has 3 edges, so d2's triplet indices should be offset by 3, not
    # by d1's node count (2).
    expected = torch.tensor([[0, 3], [1, 4]])
    assert torch.equal(batch.triplet_index, expected)


def test_batching_three_graphs():
    graphs = []
    for n_nodes, n_edges in [(2, 5), (3, 2), (4, 7)]:
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        graphs.append(
            CrystalData(
                x=torch.arange(n_nodes),
                edge_index=edge_index,
                triplet_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=n_nodes,
            )
        )
    batch = Batch.from_data_list(graphs)
    assert batch.edge_index.size(1) == 5 + 2 + 7
    assert batch.num_nodes == 2 + 3 + 4
