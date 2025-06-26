from typing import Any

import torch
from torch import nn

from retflow.retro_utils.graph_features import ExtraFeatures
from retflow.retro_utils.molecule_features import ExtraMolecularFeatures
from retflow.retro_utils.place_holders import PlaceHolder


class GraphModelWrapper:
    def __init__(
        self,
        torch_model: nn.Module,
        extra_graph_features: ExtraFeatures,
        extra_mol_features: ExtraMolecularFeatures = None
    ) -> None:
        self.torch_model = torch_model
        self.extra_graph_features = extra_graph_features
        self.extra_mol_features = extra_mol_features

    def __call__(self, noisy_graph: PlaceHolder, node_mask, context, t) -> Any:
        extra_data = self.compute_extra_data(noisy_graph, node_mask, context)
        extra_data.y = torch.cat((extra_data.y, t), dim=-1)

        if self.extra_mol_features is not None:
            extra_mol_data = self.extra_mol_features(noisy_graph.X, noisy_graph.E, node_mask)

            X = torch.cat((noisy_graph.X, extra_data.X, extra_mol_data.X), dim=-1).float()
            E = torch.cat((noisy_graph.E, extra_data.E), dim=-1).float()
            y = torch.hstack((noisy_graph.y, extra_data.y, extra_mol_data.y)).float()
        else:
            X = torch.cat((noisy_graph.X, extra_data.X), dim=-1).float()
            E = torch.cat((noisy_graph.E, extra_data.E), dim=-1).float()
            y = torch.hstack((noisy_graph.y, extra_data.y)).float()
        return self.torch_model(X, E, y, node_mask)

    def compute_extra_data(self, noisy_graph: PlaceHolder, node_mask, context=None):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""

        extra_features = self.extra_graph_features(
            noisy_graph.X, noisy_graph.E, node_mask
        )

        extra_X = extra_features.X
        extra_E = extra_features.E
        extra_y = extra_features.y

        if context is not None:
            if isinstance(context, list):
                for c in context:
                    extra_X = torch.cat((extra_X, c.X), dim=-1)
                    extra_E = torch.cat((extra_E, c.E), dim=-1)
            else:
                extra_X = torch.cat((extra_X, context.X), dim=-1)
                extra_E = torch.cat((extra_E, context.E), dim=-1)

        return PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
