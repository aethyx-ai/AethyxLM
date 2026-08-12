from dataset.dataset import DistributedStridedSampler


class VirtualDataset:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


def test_distributed_strided_sampler_partitions_without_padding():
    dataset = VirtualDataset(11)
    rank_zero = DistributedStridedSampler(dataset, num_replicas=2, rank=0)
    rank_one = DistributedStridedSampler(dataset, num_replicas=2, rank=1)

    assert list(rank_zero) == [0, 2, 4, 6, 8]
    assert list(rank_one) == [1, 3, 5, 7, 9]
    assert len(rank_zero) == len(rank_one) == 5
    assert set(rank_zero).isdisjoint(rank_one)


def test_distributed_strided_sampler_handles_huge_virtual_length_lazily():
    sampler = DistributedStridedSampler(
        VirtualDataset(476_810_152), num_replicas=2, rank=1
    )
    iterator = iter(sampler)

    assert [next(iterator) for _ in range(4)] == [1, 3, 5, 7]
    assert len(sampler) == 238_405_076


def test_distributed_strided_sampler_validates_rank():
    dataset = VirtualDataset(10)

    try:
        DistributedStridedSampler(dataset, num_replicas=2, rank=2)
    except ValueError as error:
        assert "rank" in str(error)
    else:
        raise AssertionError("invalid rank should be rejected")
