#!/bin/bash
#SBATCH --job-name vit_training
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --output /users/ogal/nmashall/VIT/logs/%j.out

srun --container-image /users/ogal/nmashall/VIT/vit_pytorch_modified.sqsh \
     --container-mounts /users/ogal/nmashall/VIT:/code \
     --no-container-entrypoint \
     /bin/bash -c "python /code/training.py"
