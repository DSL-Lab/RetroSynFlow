from retflow import Experiment
from retflow.optimizers.optimizer import AdamW
from retflow.optimizers.schedulers import ConsLR
from retflow.datasets import RetroDataset
from retflow.methods import MarkovBridge
from retflow.problems import Retrosynthesis
from retflow.models import GraphTransformer
from retflow.retro_utils import GraphModelLayerInfo
from retflow.runner import slurm_config, cli_runner

model = GraphTransformer(
    n_layers=5,
    n_head=8,
    ff_dims=GraphModelLayerInfo(256, 128, 128),
    hidden_mlp_dims=GraphModelLayerInfo(256, 128, 128),
    hidden_dims=GraphModelLayerInfo(256, 64, 64),
)

dataset = RetroDataset(
    name="USPTO",
    batch_size=64,
)
method = MarkovBridge(
    steps=50,
    noise_schedule="cosine",
    lambda_train=5.0,
)

experiment = Experiment(
    problem=Retrosynthesis(model, dataset, method),
    optim=AdamW(lr=2e-4, grad_clip=None, lr_sched=ConsLR()),
    epochs=1000,
    sample_epoch=100,
    num_samples=128,
    examples_per_sample=5,
    seed=42,
    group="markov_bridge",
    name="vlb",
)


if __name__ == "__main__":
    cli_runner([experiment], slurm_config.DEFAULT_GPU_110H)
