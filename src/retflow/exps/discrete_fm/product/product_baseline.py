from retflow import Experiment
from retflow.datasets import RetroDataset
from retflow.methods import DiscreteFM, LinearTimeScheduler, UniformTimeSampler
from retflow.models import GraphTransformer
from retflow.optimizers.optimizer import AdamW
from retflow.optimizers.schedulers import ConsLR
from retflow.problems import Retrosynthesis
from retflow.runner import cli_runner, slurm_config
from retflow.utils import GraphModelLayerInfo

model = GraphTransformer(
    n_layers=5,
    n_head=8,
    ff_dims=GraphModelLayerInfo(256, 128, 128),
    hidden_mlp_dims=GraphModelLayerInfo(256, 128, 128),
    hidden_dims=GraphModelLayerInfo(256, 64, 64),
)

dataset = RetroDataset(name="USPTO", batch_size=32)
method = DiscreteFM(
    steps=50,
    edge_time_sched=LinearTimeScheduler(),
    node_time_sched=LinearTimeScheduler(),
    time_sampler=UniformTimeSampler(),
    edge_weight_loss=5.0,
)

experiment = Experiment(
    problem=Retrosynthesis(model, dataset, method),
    optim=AdamW(lr=2e-4, grad_clip=None, lr_sched=ConsLR()),
    epochs=2,
    sample_epoch=1,
    num_samples=128,
    examples_per_sample=5,
    seed=42,
    group="discrete_fm_product",
    name="baseline",
)


if __name__ == "__main__":
    cli_runner([experiment], slurm_config.DEFAULT_3_GPU_46H)
