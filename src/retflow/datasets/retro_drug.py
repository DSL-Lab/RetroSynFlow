import os
from dataclasses import dataclass
from typing import List, Tuple

from torch.utils.data.distributed import DistributedSampler
from torchdrug.data import DataLoader

from retflow import config
from retflow.datasets.data.uspto_drug import _TorchDrugUSPTO
from retflow.datasets.dataset import Dataset
from retflow.datasets.info import RetrosynthesisInfo
from retflow.datasets.synthon import SynthonDataset
from retflow.runner import DistributedHelper


@dataclass
class TorchDrugRetroDataset(Dataset):
    product_context: bool = False

    def load(self, dist_helper: DistributedHelper | None = None):
        if self.name not in ["DrugUSPTO"]:
            raise ValueError(f"{self.name} is not a TorchDrug Dataset.")

        save_dir = config.get_dataset_directory() / self.name

        train_dataset = _TorchDrugUSPTO(
            data_split="train", path=str(save_dir), atom_feature="center_identification"
        )
        val_dataset = _TorchDrugUSPTO(
            data_split="val", path=str(save_dir), atom_feature="center_identification"
        )

        train_loader, val_loader = self._get_train_and_val_loaders(
            train_dataset, val_dataset, dist_helper
        )
        dummy_dataset = SynthonDataset(name="MultiSynthonUSPTO", batch_size=1)
        _, _, self.info = dummy_dataset.load()

        if self.product_context:
            self.info.input_dim.node_dim += 17
            self.info.input_dim.edge_dim += 5

        return train_loader, val_loader, self.info

    def download(self) -> None:
        save_dir = config.get_dataset_directory() / self.name
        for split in ["train", "val", "test"]:
            _TorchDrugUSPTO.download(split, str(save_dir))

    def load_eval(self, load_valid=False) -> Tuple[DataLoader | RetrosynthesisInfo]:
        if self.name not in ["DrugUSPTO"]:
            raise ValueError(f"{self.name} is not any of the RetroSynthesis Datasets.")

        save_dir = config.get_dataset_directory() / self.name

        self.test_dataset = _TorchDrugUSPTO(
            data_split="val" if load_valid else "test",
            path=str(save_dir),
            atom_feature="center_identification",
        )

        dummy_dataset = SynthonDataset(name="MultiSynthonUSPTO", batch_size=1)
        _, _, self.info = dummy_dataset.load()

        if self.product_context:
            self.info.input_dim.node_dim += 17
            self.info.input_dim.edge_dim += 5

        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        return test_loader, self.info


    def _get_train_and_val_loaders(
        self, train_dataset, val_dataset, dist_helper: DistributedHelper | None = None
    ):
        if dist_helper is not None:
            batch_size_per_gpu = max(
                1, self.batch_size // dist_helper.get_ddp_status()[1]["WORLD_SIZE"]
            )
            train_loader = DataLoader(
                dataset=train_dataset,
                sampler=DistributedSampler(train_dataset, shuffle=True),
                batch_size=batch_size_per_gpu,
                pin_memory=True,
                num_workers=min(6, os.cpu_count()),
            )

            val_loader = DataLoader(
                dataset=val_dataset,
                sampler=DistributedSampler(val_dataset, shuffle=False),
                batch_size=batch_size_per_gpu,
                pin_memory=True,
                num_workers=min(6, os.cpu_count()),
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=min(6, os.cpu_count()),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=min(6, os.cpu_count()),
            )

        return train_loader, val_loader
