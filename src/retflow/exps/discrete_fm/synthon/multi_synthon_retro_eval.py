from retflow import Experiment
from retflow.datasets import TorchDrugRetroDataset
from retflow.experiment_eval import ExperimentEvaluator
from retflow.methods import DiscreteFM, LinearTimeScheduler, UniformTimeSampler
from retflow.models import GraphTransformer
from retflow.optimizers.optimizer import AdamW
from retflow.optimizers.schedulers import ConsLR
from retflow.problems import MultiSynthonRetrosynthesis
from retflow.runner import cli_runner, slurm_config
from retflow.utils import GraphModelLayerInfo

model = GraphTransformer(
    n_layers=5,
    n_head=8,
    ff_dims=GraphModelLayerInfo(256, 128, 128),
    hidden_mlp_dims=GraphModelLayerInfo(256, 128, 128),
    hidden_dims=GraphModelLayerInfo(256, 64, 64),
)

dataset = TorchDrugRetroDataset(name="DrugUSPTO", batch_size=64 * 3)
method = DiscreteFM(
    steps=100,
    edge_time_sched=LinearTimeScheduler(),
    node_time_sched=LinearTimeScheduler(),
    time_sampler=UniformTimeSampler(),
    edge_weight_loss=5.0,
)

experiments = [
    Experiment(
        problem=MultiSynthonRetrosynthesis(model, dataset, method),
        optim=AdamW(lr=2e-4, grad_clip=None, lr_sched=ConsLR()),
        epochs=1000,
        sample_epoch=150,
        num_samples=128,
        examples_per_sample=5,
        seed=seed,
        group="discrete_fm_basic_multi_synthon_fixed",
        name="linear",
    )
    for seed in [42]
]

test_method = DiscreteFM(
    steps=50,
    edge_time_sched=LinearTimeScheduler(),
    node_time_sched=LinearTimeScheduler(),
    time_sampler=UniformTimeSampler(),
    edge_weight_loss=5.0,
)

experiments_evals = [
    ExperimentEvaluator(
        experiment=exp,
        test_method=test_method,
        examples_per_sample=50,
        checkpoint_name="model_epoch_600.pt",
        output_name="default_linear_multi_synthon_retro_fixed2",
    )
    for exp in experiments
]

if __name__ == "__main__":
    cli_runner(experiments_evals, slurm_config.DEFAULT_GPU_32H)
