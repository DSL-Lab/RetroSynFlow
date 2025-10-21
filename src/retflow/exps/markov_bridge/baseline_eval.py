from retflow.experiment_eval import ExperimentEvaluator
from retflow.exps.markov_bridge.baseline import experiment
from retflow.runner import cli_runner, slurm_config

experiments_evals = [
    ExperimentEvaluator(
        experiment=experiment,
        test_method=experiment.problem.method,
        test_batch_size=256,
        examples_per_sample=100,
        checkpoint_name="final_model.pt",
        output_name="baseline_vlb_100_reactants",
    )
]


if __name__ == "__main__":
    cli_runner(experiments_evals, slurm_config.DEFAULT_GPU_32H)
