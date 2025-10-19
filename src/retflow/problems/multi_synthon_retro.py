import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torchdrug import models, tasks, utils, layers
from torchdrug.core import Registry as R
from torch_geometric.data import Data, Batch

from retflow import config
from retflow.optimizers.optimizer import Optimizer
from retflow.problems.problem import Problem
from retflow.retro_utils import (
    ExtraFeatures,
    GraphModelWrapper,
    predicted_reactants_molecule_graph,
    build_molecule,
    to_dense,
    build_simple_molecule,
    synthons_molecule_graph,
)
from retflow.retro_utils.data import (
    compute_graph,
    compute_nodes_mapping,
    compute_graph_with_mapping,
)
from rdkit import Chem
from retflow.runner import DistributedHelper


@dataclass
class MultiSynthonRetrosynthesis(Problem):
    synthon_topk: int = 1
    samples_per_synthon: List[int] = None
    product_context: bool = False

    def setup_problem(self, dist_helper: DistributedHelper | None = None):
        self.train_loader, self.val_loader, self.info = self.dataset.load(
            dist_helper=dist_helper
        )
        if dist_helper:
            self.torch_model = self.method.setup(
                self.info, self.model, device=dist_helper.device
            )
            self.torch_model = dist_helper.dist_adapt_model(self.torch_model)
        else:
            self.torch_model = self.method.setup(
                self.info, self.model, device=config.get_device()
            ).to(config.get_device())

        self.sampling_metrics = self.dataset.get_metrics()

        self.model_wrapper = GraphModelWrapper(
            self.torch_model,
            ExtraFeatures(self.info.max_n_nodes),
        )

    def setup_problem_eval(self, model_checkpoint, on_valid=False):
        self.test_loader, self.info = self.dataset.load_eval(load_valid=on_valid)

        self.torch_model = self.method.setup(
            self.info, self.model, device=config.get_device()
        )

        self.torch_model.load_state_dict(torch.load(model_checkpoint))
        self.torch_model = self.torch_model.to(config.get_device())

        center_model = models.RGCN(
            input_dim=43,
            hidden_dims=[256, 256, 256, 256],
            num_relation=4,
            short_cut=True,
            concat_hidden=True,
        )

        synthon_pred_task = CenterIdentificationModified(
            center_model, feature=("graph", "atom", "bond")
        )
        synthon_pred_task.preprocess(None, None, self.dataset.test_dataset)

        synthon_pred_task.load_state_dict(
            torch.load(config.get_models_directory() / "g2g_center.pth")["model"]
        )
        self.synthon_pred_task = synthon_pred_task  # .to(config.get_device())

        self.model_wrapper = GraphModelWrapper(
            self.torch_model,
            ExtraFeatures(self.info.max_n_nodes),
        )

    def get_optimizer(self, optimizer: Optimizer) -> torch.optim.Optimizer:
        return optimizer.get_optim(self.torch_model)

    def one_epoch(
        self,
        optim: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler._LRScheduler,
        log_every_n_batch: int,
        dist_helper: DistributedHelper | None,
    ):
        return {}

    @torch.no_grad()
    def validation(self, dist_helper: DistributedHelper | None):
        return {}

    def save_model(
        self, checkpoint_dir: Path, ddp: bool = False, epoch: int | None = None
    ):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if epoch:
            save_path = checkpoint_dir / f"model_epoch_{epoch}.pt"
        else:
            save_path = checkpoint_dir / "final_model.pt"
        if ddp:
            torch.save(self.torch_model.module.state_dict(), save_path)
        else:
            torch.save(self.torch_model.state_dict(), save_path)

    @torch.no_grad()
    def sample_generation(
        self,
        num_samples: int,
        examples_per_sample: int,
        dist_helper: DistributedHelper | None = None,
    ):
        return {}

    @torch.no_grad()
    def sample_generation_eval(
        self,
        examples_per_sample: int,
    ):
        self.torch_model.eval()

        num_samples = 0
        product_molecules_smiles = []
        synthon_molecule_smiles = []
        true_reactant_smiles = []
        pred_reactant_smiles = []
        computed_scores = []

        if self.samples_per_synthon is None:
            per_synthon_ex_per_sample = int(examples_per_sample / self.synthon_topk)
        else:
            assert sum(self.samples_per_synthon) == examples_per_sample

        for i, data in enumerate(self.test_loader):
            config.get_logger().info(
                f"Generated reactants for {num_samples} product molecules so far."
            )
            config.get_logger().info(f"Generating samples for batch {i} of data.")

            bs = data["graph"][0].batch_size
            # if config.get_device() == "cuda":
            #     data = utils.cuda(data)

            batch_groups = []
            batch_scores = []
            batch_synthons = []

            ground_truth = []
            input_products = []

            r_mols = data["graph"][0].to_molecule()
            p_mols = data["graph"][1].to_molecule()
            for r_mol, p_mol in zip(r_mols, p_mols):
                simple_rmol = build_simple_molecule(r_mol)
                simple_pmol = build_simple_molecule(p_mol)
                ground_truth.append(Chem.MolToSmiles(simple_rmol, canonical=True))
                input_products.append(Chem.MolToSmiles(simple_pmol, canonical=True))

            result = self.synthon_pred_task.predict_synthon(data, k=self.synthon_topk)
            grouped_synthon_smis = self.group_predicted_synthon_smiles(
                result["synthon"].to_smiles(), result["num_synthon"]
            )
            topk_grouped_synthons = self.regroup_synthons_topk(
                grouped_synthon_smis, self.synthon_topk
            )
            i_samples = 0
            for n, synthons_smi_list in enumerate(topk_grouped_synthons):
                if self.product_context:
                    synthons_batch = self.create_synthon_graphs_with_products(
                        synthons_smi_list, data["graph"][1].to_molecule()
                    )
                else:
                    synthons_batch = self.create_synthon_graphs(synthons_smi_list)
                synthons_batch = synthons_batch.to(config.get_device())

                synthon, node_mask = to_dense(
                    synthons_batch.x,
                    synthons_batch.edge_index,
                    synthons_batch.edge_attr,
                    synthons_batch.batch,
                )
                synthon = synthon.mask(node_mask)

                predicted_synthon_graphs = predicted_reactants_molecule_graph(
                    torch.argmax(synthon.X, dim=-1),
                    torch.argmax(synthon.E, dim=-1),
                    synthons_batch.batch,
                    len(synthon.X),
                )

                batch_synthon = []
                for s_graph in predicted_synthon_graphs:
                    mol = build_molecule(s_graph[0], s_graph[1], self.info.atom_decoder)
                    smi = Chem.MolToSmiles(mol, canonical=True)
                    batch_synthon.append(smi)

                if self.product_context:
                    products, p_node_mask = to_dense(
                        synthons_batch.p_x,
                        synthons_batch.p_edge_index,
                        synthons_batch.p_edge_attr,
                        synthons_batch.batch,
                    )
                    products = products.mask(p_node_mask)

                synthon_bs = len(synthon.X)
                assert synthon_bs == bs

                if self.product_context:
                    context = [synthon.clone(), products.clone()]
                else:
                    context = synthon.clone()
                num_to_sample = (
                    self.samples_per_synthon[n]
                    if self.samples_per_synthon
                    else per_synthon_ex_per_sample
                )
                for j in range(num_to_sample):
                    config.get_logger().info(
                        f"Sampling reactant {i_samples + 1} out of {examples_per_sample} for each molecule in batch."
                    )

                    X, E = self.method.sample(
                        initial_graph=synthon,
                        node_mask=node_mask,
                        context=context,
                        predictor=self.model_wrapper,
                    )

                    pred_reactants_batch_list = predicted_reactants_molecule_graph(
                        X, E, synthons_batch.batch, bs
                    )
                    pred_reactant_smis = self.convert_reactant_graphs_to_smile(
                        pred_reactants_batch_list
                    )
                    scores = [0] * len(pred_reactant_smis)

                    batch_groups.append(pred_reactant_smis)
                    batch_scores.append(scores)
                    batch_synthons.append(batch_synthon)
                    i_samples += 1

            # for loop to do transpose, batch size length array each entry is another array
            # with samples per product entries
            for mol_idx_in_batch in range(bs):
                mol_samples_group = []
                mol_scores_group = []
                mol_synthons_group = []

                for batch_group, scores_group, batch_synthon in zip(
                    batch_groups, batch_scores, batch_synthons
                ):
                    mol_samples_group.append(batch_group[mol_idx_in_batch])
                    mol_scores_group.append(scores_group[mol_idx_in_batch])
                    mol_synthons_group.append(batch_synthon[mol_idx_in_batch])

                pred_reactant_smiles.append(mol_samples_group)
                computed_scores.append(mol_scores_group)
                synthon_molecule_smiles.append(mol_synthons_group)

                true_reactant_smiles.append(ground_truth[mol_idx_in_batch])
                product_molecules_smiles.append(input_products[mol_idx_in_batch])

            num_samples += bs
 
        sampled_data = {
            "reactants": true_reactant_smiles,
            "products": product_molecules_smiles,
            "synthons": synthon_molecule_smiles,
            "predicted_reactants": pred_reactant_smiles,
            "scores": computed_scores,
        }
        return sampled_data

    def group_predicted_synthon_smiles(self, synthon_smi_list, num_synthon_list):
        curr_id = 0
        grouped_synthon_smi_list = []
        for i in range(len(num_synthon_list)):
            num_synthons = num_synthon_list[i].item()
            synthons_smi = synthon_smi_list[curr_id : curr_id + num_synthons]
            full_synthons_smi = ".".join([s for s in synthons_smi])
            grouped_synthon_smi_list.append(full_synthons_smi)
            curr_id += num_synthons
        return grouped_synthon_smi_list

    def regroup_synthons_topk(self, grouped_synthon_smi_list, k):
        topk_groups = []
        final_range = int(len(grouped_synthon_smi_list) / k)
        for i in range(k):
            idx = [k * j + i for j in range(final_range)]
            topk_groups.append([grouped_synthon_smi_list[n] for n in idx])
        return topk_groups

    def create_synthon_graphs_with_products(self, synthon_smi_list, product_mol_list):
        data_list = []
        for i, (synthon_smi, product_mol) in enumerate(
            zip(synthon_smi_list, product_mol_list)
        ):
            synthons_mol = Chem.MolFromSmiles(synthon_smi, sanitize=False)
            mapping = compute_nodes_mapping(synthons_mol)
            num_nodes = synthons_mol.GetNumAtoms() + self.info.max_n_dummy_nodes

            s_x, s_edge_index, s_edge_attr = compute_graph_with_mapping(
                synthons_mol,
                mapping,
                num_nodes,
                types=self.info.atom_encoder,
                bonds=self.info.bonds,
            )
            p_x, p_edge_index, p_edge_attr = compute_graph_with_mapping(
                product_mol,
                mapping,
                num_nodes,
                types=self.info.atom_encoder,
                bonds=self.info.bonds,
            )
            y = torch.zeros(size=(1, 0), dtype=torch.float)
            synthon_mask = ~(s_x[:, -1].bool()).squeeze()
            new2old_idx = torch.randperm(num_nodes).long()
            old2new_idx = torch.empty_like(new2old_idx)
            old2new_idx[new2old_idx] = torch.arange(num_nodes)

            p_x = p_x[new2old_idx]
            p_edge_index = torch.stack(
                [old2new_idx[p_edge_index[0]], old2new_idx[p_edge_index[1]]],
                dim=0,
            )
            p_edge_index, p_edge_attr = self.sort_edges(
                p_edge_index, p_edge_attr, num_nodes
            )

            s_x = s_x[new2old_idx]
            s_edge_index = torch.stack(
                [old2new_idx[s_edge_index[0]], old2new_idx[s_edge_index[1]]],
                dim=0,
            )
            s_edge_index, s_edge_attr = self.sort_edges(
                s_edge_index, s_edge_attr, num_nodes
            )

            synthon_mask = ~(s_x[:, -1].bool()).squeeze()
            assert torch.allclose(p_x[synthon_mask], s_x[synthon_mask])

            data = Data(
                x=s_x,
                edge_index=s_edge_index,
                edge_attr=s_edge_attr,
                p_x=p_x,
                p_edge_index=p_edge_index,
                p_edge_attr=p_edge_attr,
                y=y,
                idx=i,
            )
            data_list.append(data)
        return Batch.from_data_list(data_list)

    def create_synthon_graphs(self, synthon_smi_list):
        data_list = []
        for i, synthon_smi in enumerate(synthon_smi_list):
            synthons_mol = Chem.MolFromSmiles(synthon_smi, sanitize=False)
            num_nodes = synthons_mol.GetNumAtoms() + self.info.max_n_dummy_nodes

            s_x, s_edge_index, s_edge_attr = compute_graph(
                synthons_mol,
                num_nodes,
                types=self.info.atom_encoder,
                bonds=self.info.bonds,
            )

            y = torch.zeros(size=(1, 0), dtype=torch.float)
            data = Data(
                x=s_x,
                edge_index=s_edge_index,
                edge_attr=s_edge_attr,
                y=y,
                idx=i,
            )
            data_list.append(data)
        return Batch.from_data_list(data_list)

    def convert_reactant_graphs_to_smile(self, graph_list):
        smi_list = []
        for reactant_graph in graph_list:
            r_mol = build_molecule(
                reactant_graph[0], reactant_graph[1], self.info.atom_decoder
            )
            smi_list.append(Chem.MolToSmiles(r_mol))
        return smi_list

    @staticmethod
    def sort_edges(edge_index, edge_attr, max_num_nodes):
        if len(edge_attr) != 0:
            perm = (edge_index[0] * max_num_nodes + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

        return edge_index, edge_attr


@R.register("tasks.CenterIdentificationModified")
class CenterIdentificationModified(tasks.CenterIdentification):
    def __init__(self, model, feature=("graph", "atom", "bond")):
        super().__init__(model, feature)
        self.name = "CenterIdentificationModified"

    def preprocess(self, train_set, valid_set, test_set):
        reaction_types = set()
        bond_types = set()
        for sample in test_set:
            reaction_types.add(sample["reaction"])
            for graph in sample["graph"]:
                bond_types.update(graph.edge_list[:, 2].tolist())
        self.num_reaction = len(reaction_types)
        self.num_relation = len(bond_types)
        node_feature_dim = test_set[0]["graph"][0].node_feature.shape[-1]
        edge_feature_dim = test_set[0]["graph"][0].edge_feature.shape[-1]

        node_dim = self.model.output_dim
        edge_dim = 0
        graph_dim = 0

        for _feature in sorted(self.feature):
            if _feature == "reaction":
                graph_dim += self.num_reaction
            elif _feature == "graph":
                graph_dim += self.model.output_dim
            elif _feature == "atom":
                node_dim += node_feature_dim
            elif _feature == "bond":
                edge_dim += edge_feature_dim
            else:
                raise ValueError("Unknown feature `%s`" % _feature)

        node_dim += graph_dim  # inherit graph features
        edge_dim += node_dim * 2  # inherit node features

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.edge_mlp = layers.MLP(edge_dim, hidden_dims + [1])
        self.node_mlp = layers.MLP(node_dim, hidden_dims + [1])
