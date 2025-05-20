#!/bin/bash
#SBATCH --time=2:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=236511M
#SBATCH --job-name=ShoiguGerasimovGdeSukaBojepripasy
#SBATCH --output=logs/hist-da-%N-%j.out
#SBATCH --error=logs/hist-da-%N-%j.err

#SBATCH --mail-user=haochen.sun@uwaterloo.ca
#SBATCH --mail-type=ALL

source activate ~/.conda/envs/GPM_ENV

# mkdir logs, if not exist
mkdir -p logs

# Delete the old logs/hist-da.csv, if exist
LOG_FILE=logs/hist-da.csv
if [ -f "$LOG_FILE" ]; then
    rm -f "$LOG_FILE"
fi

# Create a new logs/hist-da.log, witt certain content
echo "num_records,num_bins,beta,gamma,sigma,eps_comp,eps_actual,delta,success_rate" > $LOG_FILE

EXP_SCRIPT="hist-da.py"

NUM_RECORDS=1000000
REPEAT=100
DELTA=1e-10

for num_bins in 16 64 256 1024 4096 16384 65536; do
    for eps in 0.125 0.25 0.5 1; do
        for beta in 1e-5 1e-4 1e-3 1e-2 1e-1; do
            # Run the python script with the parameters
            python $EXP_SCRIPT --num_records $NUM_RECORDS --num_bins $num_bins --epsilon $eps --delta $DELTA --beta $beta --repeat $REPEAT >> $LOG_FILE
        done
    done
done

git add $LOG_FILE
git commit -m "Add histogram results to $LOG_FILE"
git push