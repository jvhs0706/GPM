#!/bin/bash
#SBATCH --time=2-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=236511M
#SBATCH --job-name=GPM
#SBATCH --output=logs/cifar10-da-%N-%j.out
#SBATCH --error=logs/cifar10-da-%N-%j.err

# Ensure current environment is GPM_ENV
CURRENT_ENV=$(conda info --json | jq -r '.active_prefix_name')
if [ "$CURRENT_ENV" != "GPM_ENV" ]; then
    echo "Please activate the GPM_ENV conda environment before running this script."
    exit 1
fi

# mkdir logs, if not exist
mkdir -p logs

# Delete the old logs/cifar10-da-addl.csv, if exist
LOG_FILE=logs/cifar10-da-addl.csv
if [ -f "$LOG_FILE" ]; then
    rm -f "$LOG_FILE"
fi

# Create a new logs/cifar10-da-addl.csv, with certain content
echo "dataset,model,num_params,converged,acc,batch_size,clip_norm,beta,gamma,sigma,eps_comp,eps_actual,delta,success_rate" > $LOG_FILE

EXP_SCRIPT="cifar10-da.py"

REPEAT=100
DELTA=1e-10
BATCH_SIZE=128

for model in "vgg19_bn" "resnet50" "mobilenet_v2"; do 
    for eps in 0.125 0.25 0.5 1; do
        for beta in 1e-5 1e-4 1e-3; do
            for clip_norm in 4 8; do
                for gamma_to_sqrt_d in 2 20 200; do
                    # Run the python script with the parameters
                    python $EXP_SCRIPT --model $model --batch_size $BATCH_SIZE --epsilon $eps --delta $DELTA --beta $beta --clip_norm $clip_norm --gamma_to_sqrt_d $gamma_to_sqrt_d --pretrained --repeat $REPEAT >> $LOG_FILE
                    python $EXP_SCRIPT --model $model --batch_size $BATCH_SIZE --epsilon $eps --delta $DELTA --beta $beta --clip_norm $clip_norm --gamma_to_sqrt_d $gamma_to_sqrt_d --repeat $REPEAT >> $LOG_FILE
                    git add $LOG_FILE
                    git commit -m "Add $model results to $LOG_FILE, with eps=$eps, beta=$beta, gamma_to_sqrt_d=$gamma_to_sqrt_d, clip_norm=$clip_norm"
                    git push
                done 
            done
        done
    done
done