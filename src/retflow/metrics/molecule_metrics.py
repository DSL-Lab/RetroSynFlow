import torch

from retflow.metrics.metric import SamplingMetric

from torch import distributed as dist


class NDistributionKL(SamplingMetric):
    def __init__(self, max_n, real_dist, device):
        self.max_n = max_n
        self.count = torch.zeros(max_n + 1, dtype=torch.float, device=device)
        self.real_dist = real_dist
        self.device = device

    def update(self, molecules, real):
        if real:
            return
        for molecule in molecules:
            atom_types, _ = molecule
            n = atom_types.shape[0]
            self.count[n] += 1

    def compute(self, ddp: bool = False):
        if ddp:
            dist.barrier()
            dist.all_reduce(self.count)
        sample_dist = self.count / torch.sum(self.count)
        kl = (
            (sample_dist * (torch.log(sample_dist) - torch.log(self.real_dist)))
            .mean()
            .item()
        )
        return {self.__class__.__name__: kl}

    def reset(self):
        self.count = torch.zeros(self.max_n + 1, dtype=torch.float, device=self.device)


class NodesDistributionKL(NDistributionKL):
    def __init__(self, num_atom_types, real_dist, device):
        self.num_atom_types = num_atom_types
        self.count = torch.zeros(num_atom_types, dtype=torch.float, device=device)
        self.real_dist = real_dist
        self.device = device

    def update(self, molecules, real):
        if real:
            return
        for molecule in molecules:
            atom_types, _ = molecule

            for atom_type in atom_types:
                assert (
                    int(atom_type) != -1
                ), "Mask error, the molecules should already be masked at the right shape"
                self.count[int(atom_type)] += 1

    def reset(self):
        self.count = torch.zeros(
            self.num_atom_types, dtype=torch.float, device=self.device
        )


class EdgeDistributionKL(NDistributionKL):
    def __init__(self, num_edge_types, real_dist, device):
        self.num_edge_types = num_edge_types
        self.count = torch.zeros(num_edge_types, dtype=torch.float, device=device)
        self.real_dist = real_dist
        self.device = device

    def update(self, molecules, real):
        if real:
            return
        for molecule in molecules:
            _, edge_types = molecule
            mask = torch.ones_like(edge_types)
            mask = torch.triu(mask, diagonal=1).bool()
            mask_edge_types = edge_types * mask
            unique_edge_types, counts = torch.unique(
                mask_edge_types, return_counts=True
            )
            for type, cnt in zip(unique_edge_types, counts):
                self.count[type] += cnt

    def reset(self):
        self.count = torch.zeros(
            self.num_edge_types, dtype=torch.float, device=self.device
        )


class ValencyDistributionKL(NDistributionKL):
    def __init__(self, max_n, real_dist, device):
        self.max_n = max_n
        self.device = device
        self.count = torch.zeros(3 * max_n - 2, dtype=torch.float, device=device)
        self.real_dist = real_dist

    def update(self, molecules, real) -> None:
        if real:
            return

        for molecule in molecules:
            _, edge_types = molecule

            val_mask = edge_types == 4
            new_edge_types = torch.masked_fill(edge_types, val_mask, 1.5)
            valencies = torch.sum(new_edge_types, dim=0)
            unique, counts = torch.unique(valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.count[valency] += count

    def reset(self):
        self.count = torch.zeros(
            3 * self.max_n - 2, dtype=torch.float, device=self.device
        )
