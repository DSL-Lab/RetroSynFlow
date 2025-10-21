from retflow.methods import DiscreteFM, LinearTimeScheduler, UniformTimeSampler
from retflow.exps.discrete_fm.synthon.multisynthon_product import experiment
from retflow.runner import slurm_config, cli_runner
from retflow.experiment_eval import ExperimentEvaluator

test_method = DiscreteFM(
    steps=50,
    edge_time_sched=LinearTimeScheduler(),
    node_time_sched=LinearTimeScheduler(),
    time_sampler=UniformTimeSampler(),
    edge_weight_loss=5.0,
)

experiments_evals = [
    ExperimentEvaluator(
        experiment=experiment,
        test_method=test_method,
        examples_per_sample=100,
        checkpoint_name=f"model_epoch_400.pt",
        output_name=f"product_context_100_reactants",
    )
]

if __name__ == "__main__":
    cli_runner(experiments_evals, slurm_config.DEFAULT_GPU_32H)
