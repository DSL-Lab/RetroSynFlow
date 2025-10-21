from retflow import Experiment
from retflow.optimizers.optimizer import AdamW
from retflow.optimizers.schedulers import ConsLR
from retflow.datasets import TorchDrugRetroDataset
from retflow.methods import DiscreteFM, LinearTimeScheduler, UniformTimeSampler
from retflow.problems import MultiSynthonRetrosynthesis
from retflow.models import GraphTransformer
from retflow.utils import GraphModelLayerInfo
from retflow.runner import slurm_config, cli_runner
from retflow.experiment_eval import ExperimentEvaluator

model = GraphTransformer(
    n_layers=5,
    n_head=8,
    ff_dims=GraphModelLayerInfo(256, 128, 128),
    hidden_mlp_dims=GraphModelLayerInfo(256, 128, 128),
    hidden_dims=GraphModelLayerInfo(256, 64, 64),
)

dataset = TorchDrugRetroDataset(
    name="DrugUSPTO", batch_size=32 * 3, product_context=True
)
method = DiscreteFM(
    steps=50,
    edge_time_sched=LinearTimeScheduler(),
    node_time_sched=LinearTimeScheduler(),
    time_sampler=UniformTimeSampler(),
    edge_weight_loss=5.0,
)

experiments = [
    Experiment(
        problem=MultiSynthonRetrosynthesis(
            model,
            dataset,
            method,
            synthon_topk=2,
            samples_per_synthon=[70, 30],
            product_context=True,
        ),
        optim=AdamW(lr=2e-4, grad_clip=None, lr_sched=ConsLR()),
        epochs=600,
        sample_epoch=100,
        num_samples=128,
        examples_per_sample=5,
        seed=42,
        group="discrete_fm_synthon_completion",
        name="product_context",
    )
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
        examples_per_sample=100,
        checkpoint_name="model_epoch_400.pt",
        output_name=f"retro_k={exp.problem.synthon_topk}_70_30",
    )
    for exp in experiments
]

if __name__ == "__main__":
    cli_runner(experiments_evals, slurm_config.DEFAULT_GPU_32H)
