import os
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path

import torch
from rdkit import Chem

from retflow import config
from retflow.metrics.topk_accuracy import top_k_accuracy
from retflow.optimizers.optimizer import Optimizer
from retflow.problems.problem import Problem
from retflow.retro_utils import (
    ExtraFeatures,
    GraphModelWrapper,
    PlaceHolder,
    reactants_molecule_graph,
    products_molecule_graph,
    predicted_reactants_molecule_graph,
    to_dense,
    build_molecule,
)
from retflow.runner import DistributedHelper


@dataclass
class Retrosynthesis(Problem):
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
        if on_valid:
            config.get_logger().info(f"Evaluating on the validation dataset.")
        else:
            config.get_logger().info(f"Evaluating on the test dataset.")
        self.torch_model = self.method.setup(
            self.info, self.model, device=config.get_device()
        )

        self.torch_model.load_state_dict(torch.load(model_checkpoint))
        self.torch_model = self.torch_model.to(config.get_device())

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
        dist_helper: DistributedHelper | None,
    ):
        self.torch_model.train()
        num_points = 0
        total_loss = 0.0
        edge_loss = 0.0
        node_loss = 0.0

        enable = dist_helper is None or (
            dist_helper is not None and dist_helper.get_rank() == 0
        )
        for _, data in enumerate(
            tqdm(self.train_loader, leave=False, disable=not enable)
        ):
            optim.zero_grad()
            data = (
                data.to(dist_helper.device, non_blocking=True)
                if dist_helper
                else data.to(config.get_device())
            )
            reactants, r_node_mask = to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            reactants = reactants.mask(
                r_node_mask
            )  # mask out fake nodes and makes sure adj matrix is upper triangular

            product, p_node_mask = to_dense(
                data.p_x, data.p_edge_index, data.p_edge_attr, data.batch
            )
            product = product.mask(p_node_mask)
            assert torch.allclose(r_node_mask, p_node_mask)
            node_mask = r_node_mask

            noisy_graph, node_mask, t_float = self.method.apply_noise(
                X=product.X,
                E=product.E,
                y=product.y,
                X_T=reactants.X,
                E_T=reactants.E,
                y_T=reactants.y,
                node_mask=node_mask,
            )

            X, E = self.model_wrapper(noisy_graph, node_mask, product.clone(), t_float)
            prediction = PlaceHolder(X, E, product.y).mask(node_mask)

            fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
            modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)  # dummy nodes
            assert torch.all(fixed_nodes | modifiable_nodes)
            prediction.X = prediction.X * modifiable_nodes + product.X * fixed_nodes
            prediction.X = prediction.X * node_mask.unsqueeze(-1)

            loss, node_loss, edge_loss = self.method.compute_loss(
                reactants,
                prediction,
                node_mask,
                noisy_graph,
                t_float,
            )

            loss.backward()

            optim.step()

            total_loss += loss * len(data)
            node_loss += node_loss * len(data)
            edge_loss += edge_loss * len(data)

            num_points += len(data)

        sched.step()

        metrics = {}
        metrics["train_loss"] = total_loss
        metrics["train_node_loss"] = node_loss
        metrics["train_edge_loss"] = edge_loss

        if dist_helper:
            metric_dict = {
                "metrics": metrics,
                "num_points": torch.tensor(num_points, device=dist_helper.device),
            }
        else:
            for metric in metrics.keys():
                metrics[metric] = metrics[metric].item() / num_points
            metric_dict = {"metrics": metrics, "num_points": num_points}
        return metric_dict

    @torch.no_grad()
    def validation(self, dist_helper: DistributedHelper | None):
        self.torch_model.eval()

        total_loss = 0.0
        edge_loss = 0.0
        node_loss = 0.0
        num_points = 0

        for _, data in enumerate(tqdm(self.val_loader, leave=False)):
            if dist_helper:
                data = data.to(dist_helper.device, non_blocking=True)
            else:
                data = data.to(config.get_device())

            reactants, r_node_mask = to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            reactants = reactants.mask(r_node_mask)

            product, p_node_mask = to_dense(
                data.p_x, data.p_edge_index, data.p_edge_attr, data.batch
            )
            product = product.mask(p_node_mask)
            assert torch.allclose(r_node_mask, p_node_mask)
            node_mask = r_node_mask

            noisy_graph, node_mask, t_float = self.method.apply_noise(
                X=product.X,
                E=product.E,
                y=product.y,
                X_T=reactants.X,
                E_T=reactants.E,
                y_T=reactants.y,
                node_mask=node_mask,
            )

            X, E = self.model_wrapper(noisy_graph, node_mask, product.clone(), t_float)
            prediction = PlaceHolder(X, E, product.y).mask(node_mask)

            fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
            modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)  # dummy nodes
            assert torch.all(fixed_nodes | modifiable_nodes)
            prediction.X = prediction.X * modifiable_nodes + product.X * fixed_nodes
            prediction.X = prediction.X * node_mask.unsqueeze(-1)

            loss, node_loss, edge_loss = self.method.compute_loss(
                reactants, prediction, node_mask, noisy_graph, t_float
            )

            total_loss += loss * len(data)
            node_loss += node_loss * len(data)
            edge_loss += edge_loss * len(data)
            num_points += len(data)

        metrics = {}
        metrics["val_loss"] = total_loss
        metrics["val_node_loss"] = node_loss
        metrics["val_edge_loss"] = edge_loss

        if dist_helper:
            return {
                "metrics": metrics,
                "num_points": torch.tensor(num_points, device=config.get_device()),
            }

        for metric in metrics.keys():
            metrics[metric] = metrics[metric].item() / num_points
        return {"metrics": metrics, "num_points": num_points}

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
        self.torch_model.eval()
        num_samples_generated = 0

        batch_id = 0

        grouped_samples = []
        grouped_scores = []
        ground_truth = []

        for _, data in enumerate(tqdm(self.val_loader, leave=False)):
            if num_samples_generated >= num_samples:
                break

            if dist_helper:
                data = data.to(dist_helper.device, non_blocking=True)
            else:
                data = data.to(config.get_device())

            to_generate = len(data)
            batch_groups = []
            batch_scores = []

            product, node_mask = to_dense(
                data.p_x, data.p_edge_index, data.p_edge_attr, data.batch
            )
            product = product.mask(node_mask)

            true_molecule_list = reactants_molecule_graph(data, to_generate)
            products_list = products_molecule_graph(data, to_generate)
            ground_truth.extend(true_molecule_list)

            for metric in self.sampling_metrics:
                metric.update(true_molecule_list, real=False)

            for _ in range(examples_per_sample):
                X, E = self.method.sample(
                    initial_graph=product,
                    node_mask=node_mask,
                    context=product.clone(),
                    predictor=self.model_wrapper,
                )

                pred_molecule_list = predicted_reactants_molecule_graph(
                    X, E, data.batch, to_generate
                )

                scores = [0] * len(pred_molecule_list)

                batch_groups.append(pred_molecule_list)
                batch_scores.append(scores)

            num_samples_generated += to_generate
            batch_id += 1

            for mol_idx_in_batch in range(to_generate):  # batch size
                mol_samples_group = []
                mol_scores_group = []
                for batch_group, scores_group in zip(
                    batch_groups, batch_scores
                ):  # K times
                    mol_samples_group.append(batch_group[mol_idx_in_batch])
                    mol_scores_group.append(scores_group[mol_idx_in_batch])

                assert len(mol_samples_group) == examples_per_sample
                grouped_samples.append(mol_samples_group)
                grouped_scores.append(mol_scores_group)

        metrics = top_k_accuracy(
            grouped_samples=grouped_samples,
            ground_truth=ground_truth,
            atom_decoder=self.info.atom_decoder,
            grouped_scores=grouped_scores,
        )

        return metrics

    @torch.no_grad()
    def sample_generation_eval(
        self,
        examples_per_sample: int,
    ):
        self.torch_model.eval()

        num_samples = 0
        products = []
        true_reactants = []
        predicted_reactants = []
        pred_reactant_scores = []

        for i, data in enumerate(self.test_loader):
            config.get_logger().info(
                f"Generated reactants for {num_samples} product molecules so far."
            )
            config.get_logger().info(f"Generating samples for batch {i} of data.")

            bs = len(data.batch.unique())

            batch_groups = []
            batch_scores = []

            data = data.to(config.get_device())
            product, node_mask = to_dense(
                data.p_x, data.p_edge_index, data.p_edge_attr, data.batch
            )
            print(product.X.shape)
            product = product.mask(node_mask)
            reactants_list = reactants_molecule_graph(data, bs)
            products_list = products_molecule_graph(data, bs)

            ground_truth = []
            input_products = []
            for reactant_graph, product_graph in zip(reactants_list, products_list):
                r_mol = build_molecule(
                    reactant_graph[0], reactant_graph[1], self.info.atom_decoder
                )
                p_mol = build_molecule(
                    product_graph[0], product_graph[1], self.info.atom_decoder
                )
                ground_truth.append(Chem.MolToSmiles(r_mol, canonical=True))
                input_products.append(Chem.MolToSmiles(p_mol, canonical=True))

            for j in range(examples_per_sample):
                config.get_logger().info(
                    f"Sampling reactant {j + 1} out of {examples_per_sample} for each molecule in batch."
                )
                X, E = self.method.sample(
                    initial_graph=product,
                    node_mask=node_mask,
                    context=product.clone(),
                    predictor=self.model_wrapper,
                )
                pred_reactants_batch_list = predicted_reactants_molecule_graph(
                    X, E, data.batch, bs
                )
                scores = [0] * len(pred_reactants_batch_list)
                pred_reactants_smis = []
                for pred_reactant_graph in pred_reactants_batch_list:
                    pred_reactant = build_molecule(
                        pred_reactant_graph[0],
                        pred_reactant_graph[1],
                        self.info.atom_decoder,
                    )
                    pred_reactants_smis.append(
                        Chem.MolToSmiles(pred_reactant, canonical=True)
                    )

                batch_groups.append(pred_reactants_smis)
                batch_scores.append(scores)

            # for loop to do transpose, batch size length array each entry is another array
            # with samples per product entries
            for mol_idx_in_batch in range(bs):
                mol_samples_group = []
                mol_scores_group = []

                for batch_group, scores_group in zip(batch_groups, batch_scores):
                    mol_samples_group.append(batch_group[mol_idx_in_batch])
                    mol_scores_group.append(scores_group[mol_idx_in_batch])

                assert len(mol_samples_group) == examples_per_sample

                predicted_reactants.append(mol_samples_group)
                pred_reactant_scores.append(mol_scores_group)

                true_reactants.append(ground_truth[mol_idx_in_batch])
                products.append(input_products[mol_idx_in_batch])

            num_samples += bs

        sampled_data = {
            "reactants": true_reactants,
            "products": products,
            "predicted_reactants": predicted_reactants,
            "scores": pred_reactant_scores,
        }
        return sampled_data
