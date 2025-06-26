from retflow import Experiment
from retflow.optimizers.optimizer import AdamW
from retflow.optimizers.schedulers import ConsLR
from retflow.datasets import SynthonDataset
from retflow.methods import DiscreteFM, LinearTimeScheduler, UniformTimeSampler
from retflow.problems import SynthonCompletion
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

dataset = SynthonDataset(name="MultiSynthonProductUSPTO", batch_size=32 * 3)
method = DiscreteFM(
    steps=50,
    edge_time_sched=LinearTimeScheduler(),
    node_time_sched=LinearTimeScheduler(),
    time_sampler=UniformTimeSampler(),
    edge_weight_loss=5.0,
)

experiment = Experiment(
    problem=SynthonCompletion(model, dataset, method, use_product_context=True),
    optim=AdamW(lr=2e-4, grad_clip=None, lr_sched=ConsLR()),
    epochs=800,
    sample_epoch=100,
    num_samples=128,
    examples_per_sample=5,
    seed=42,
    group="discrete_fm_synthon_completion",
    name="product_context",
)


if __name__ == "__main__":
    cli_runner([experiment], slurm_config.DEFAULT_3_GPU_46H)
