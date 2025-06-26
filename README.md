# retro_fm
flow matching for retrosynthesis

Install the python libraries in the requirements folder.

In the flow_retro directory, install retflow locally using 

```
pip install -e .
```

Set the following environment variables:

```
export RETRO_WORKSPACE=~/workspace
export RETRO_WANDB_ENABLE=true
export RETRO_WANDB_MODE=offline
export RETRO_WANDB_PROJECT=testing
export RETRO_WANDB_ENTITY=wandb_entity
```

If you are not using wandb then set '''RETRO_WANDB_ENABLE=False'''. On Windows OS use SETX instead of export. 

Experiment files are located in the experiment folder. To run the experiments defined in an experiment file on a machine, use the following command

```
python [experiment_file].py --local
```

For example, to train a RetroProdFlow model, run the following experiment file,

```
cd retroflow
python ./src/retflow/exps/discrete_fm/product/product_baseline.py --local
```

If you are running on a SLURM cluster with multiple GPUs then run

```
python ./src/retflow/exps/discrete_fm/product/product_baseline.py --slurm --ddp
```





