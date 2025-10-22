from retflow import Experiment
from retflow.datasets import TorchDrugRetroDataset
from retflow.experiment_eval import ExperimentEvaluator
from retflow.methods import GraphDiscreteFM
from retflow.models import GraphTransformer
from retflow.optimizers.optimizer import AdamW
from retflow.optimizers.schedulers import ConsLR
from retflow.problems import MultiSynthonRetrosynthesis
from retflow.runner import cli_runner, slurm_config

model = GraphTransformer()

dataset = TorchDrugRetroDataset(
    name="DrugUSPTO", batch_size=32 * 3, product_context=True
)
method = GraphDiscreteFM()

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
        optim=AdamW(lr=2e-4, lr_sched=ConsLR()),
        epochs=600,
        sample_epoch=100,
        num_samples=128,
        examples_per_sample=5,
        seed=42,
        group="discrete_fm_synthon_completion",
        name="product_context",
    )
]

test_method = GraphDiscreteFM(steps=100)

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
