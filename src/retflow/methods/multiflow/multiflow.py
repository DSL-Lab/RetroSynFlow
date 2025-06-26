from dataclasses import dataclass
from typing import Callable
from tqdm import tqdm

import torch
import torch.nn.functional as F

from retflow.retro_utils.place_holders import PlaceHolder
from retflow.methods.method_utils import sample_discrete_features, pad_t_like_x
from retflow.methods.discrete_fm.basic import DiscreteFM


@dataclass
class MultiFlow(DiscreteFM):

    def sample(self, product, node_mask, batch_size, predictor: Callable):
        context = product.clone()

        # Masks for fixed and modifiable nodes
        fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
        modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
        assert torch.all(fixed_nodes | modifiable_nodes)

        # z_T – starting state (product)
        X, E, y = (
            product.X,
            product.E,
            torch.empty((node_mask.shape[0], 0), device=product.X.device),
        )

        X0 = X.clone().detach()
        E0 = E.clone().detach()

        assert (E == torch.transpose(E, 1, 2)).all()

        t = 0.0
        dt = 1.0 / self.steps
        pbar = tqdm(total=0.99)
        while t < 0.99:
            tb = t * torch.ones((X.size(0),)).to(X.device)

            noisy_graph = PlaceHolder(X, E, y)
            X_p, E_p = predictor(noisy_graph, node_mask, context, tb.unsqueeze(-1))
            pred = PlaceHolder(X_p, E_p, y=product.y).mask(node_mask)

            prob_X = self._node_rate_matrix(X, pred.X, X0, node_mask, tb, dt)
            prob_E = self._edge_rate_matrix(E, pred.E, E0, node_mask, tb, dt)

            prob_X[torch.sum(prob_X, dim=-1) == 0] = 1e-5
            prob_X = prob_X / torch.sum(prob_X, dim=-1, keepdim=True)

            prob_E[torch.sum(prob_E, dim=-1) == 0] = 1e-5
            prob_E = prob_E / torch.sum(prob_E, dim=-1, keepdim=True)

            assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
            assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

            sampled_t = sample_discrete_features(
                probX=prob_X, probE=prob_E, node_mask=node_mask
            )
            # Categorical outputs tokens so we must convert to one hot
            E_raw = F.one_hot(sampled_t.E, num_classes=self.output_dim.edge_dim)
            X_raw = F.one_hot(sampled_t.X, num_classes=self.output_dim.node_dim)

            assert (E_raw == torch.transpose(E_raw, 1, 2)).all()
            assert (X_raw.shape == X.shape) and (E_raw.shape == E.shape)

            graph = PlaceHolder(X_raw, E_raw, y).mask(node_mask).type_as(product.X)

            graph.X = graph.X * modifiable_nodes + product.X * fixed_nodes
            graph = graph.mask(node_mask)
            X, E, y = graph.X, graph.E, graph.y

            t += dt
            pbar.update(dt)

        final_graph = PlaceHolder(X, E, y)
        final_graph = final_graph.mask(node_mask, collapse=True)
        X, E, y = final_graph.X, final_graph.E, final_graph.y

        return (
            X,
            E,
            None,
            None,
        )

    def _node_rate_matrix(self, X_t, X_1_logits, X_0, node_mask, t, dt):
        X_t_arg = torch.argmax(X_t, dim=-1)
        X_1_probs = F.softmax(X_1_logits, dim=-1)
        bs, n, _ = X_1_probs.shape
        X_1_probs[~node_mask] = 1 / X_1_probs.shape[-1]
        X_1_probs = X_1_probs.reshape(bs * n, -1)

        X_1 = X_1_probs.multinomial(1)
        X_1 = X_1.reshape(bs, n)
        X_1 = F.one_hot(X_1, num_classes=self.output_dim.node_dim)

        dt_p_vals = X_1 - X_0
        dt_p_vals_at_xt = dt_p_vals.gather(-1, X_t_arg[:, :, None]).squeeze(-1)

        R_t_numer = F.relu(dt_p_vals - dt_p_vals_at_xt[:, :, None])

        pt_vals = (1 - pad_t_like_x(t, X_0)) * X_0 + pad_t_like_x(t, X_1) * X_1

        Z_t = torch.count_nonzero(pt_vals, dim=-1)

        pt_vals_at_xt = pt_vals.gather(-1, X_t_arg[:, :, None]).squeeze(-1)

        R_t_denom = Z_t * pt_vals_at_xt

        R_t = R_t_numer / R_t_denom[:, :, None]
        R_t[
            (pt_vals_at_xt == 0.0)[:, :, None].repeat(1, 1, self.output_dim.node_dim)
        ] = 0.0
        R_t[pt_vals == 0.0] = 0.0

        step_probs = (R_t * dt).clamp(max=1.0)
        step_probs.scatter_(-1, X_t_arg[:, :, None], 0.0)
        step_probs.scatter_(
            -1,
            X_t_arg[:, :, None],
            (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        return step_probs

    def _edge_rate_matrix(self, E_t, E_1_logits, E_0, node_mask, t, dt):
        E_t_arg = torch.argmax(E_t, dim=-1)
        E_1_probs = F.softmax(E_1_logits, dim=-1)

        bs, n, _, _ = E_1_probs.shape
        inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
        diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

        E_1_probs[inverse_edge_mask] = 1 / E_1_probs.shape[-1]
        E_1_probs[diag_mask.bool()] = 1 / E_1_probs.shape[-1]

        E_1_probs = E_1_probs.reshape(bs * n * n, -1)  # (bs * n * n, de_out)

        # Sample E
        E_1 = E_1_probs.multinomial(1).reshape(bs, n, n)  # (bs, n, n)
        E_1 = torch.triu(E_1, diagonal=1)
        E_1 = E_1 + torch.transpose(E_1, 1, 2)
        E_1 = F.one_hot(E_1, num_classes=self.output_dim.edge_dim)

        dt_p_vals = E_1 - E_0
        dt_p_vals_at_xt = dt_p_vals.gather(-1, E_t_arg[:, :, :, None]).squeeze(-1)

        R_t_numer = F.relu(dt_p_vals - dt_p_vals_at_xt[:, :, :, None])

        pt_vals = (1 - pad_t_like_x(t, E_0)) * E_0 + pad_t_like_x(t, E_1) * E_1

        Z_t = torch.count_nonzero(pt_vals, dim=-1)

        pt_vals_at_xt = pt_vals.gather(-1, E_t_arg[:, :, :, None]).squeeze(-1)

        R_t_denom = Z_t * pt_vals_at_xt

        R_t = R_t_numer / R_t_denom[:, :, :, None]
        R_t[
            (pt_vals_at_xt == 0.0)[:, :, :, None].repeat(
                1, 1, 1, self.output_dim.edge_dim
            )
        ] = 0.0
        R_t[pt_vals == 0.0] = 0.0

        step_probs = (R_t * dt).clamp(max=1.0)
        step_probs.scatter_(-1, E_t_arg[:, :, :, None], 0.0)
        step_probs.scatter_(
            -1,
            E_t_arg[:, :, :, None],
            (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        return step_probs
