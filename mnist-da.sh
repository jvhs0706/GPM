#!/bin/bash
#SBATCH --time=2-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=236511M
#SBATCH --job-name=ShoiguGerasimovGdeSukaBojepripasy
#SBATCH --output=logs/mnist-da-%N-%j.out
#SBATCH --error=logs/mnist-da-%N-%j.err

#SBATCH --mail-user=haochen.sun@uwaterloo.ca
#SBATCH --mail-type=ALL

source activate ~/.conda/envs/GPM_ENV

# mkdir logs, if not exist
mkdir -p logs

# Delete the old logs/mnist-da.csv, if exist
LOG_FILE=logs/mnist-da.csv
if [ -f "$LOG_FILE" ]; then
    rm -f "$LOG_FILE"
fi

# Create a new logs/mnist-da.log, witt certain content
echo "dataset,model,num_params,converged,acc,batch_size,clip_norm,beta,gamma,sigma,eps_comp,eps_actual,delta,success_rate" > $LOG_FILE

EXP_SCRIPT="mnist-da.py"

REPEAT=100
DELTA=1e-10

for batch_size in 64 128 256; do
    for eps in 0.125 0.25 0.5 1; do
        for beta in 1e-5 1e-4 1e-3; do
            for gamma in 1e3 1e4; do
                for clip_norm in 4 8; do
                    # Run the python script with the parameters
                    python $EXP_SCRIPT --batch_size $batch_size --epsilon $eps --delta $DELTA --beta $beta --gamma $gamma --clip_norm $clip_norm --pretrained --repeat $REPEAT >> $LOG_FILE
                    python $EXP_SCRIPT --batch_size $batch_size --epsilon $eps --delta $DELTA --beta $beta --gamma $gamma --clip_norm $clip_norm --repeat $REPEAT >> $LOG_FILE
                done
            done
        done
    done
done