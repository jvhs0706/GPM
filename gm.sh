#!/bin/bash
#SBATCH --time=2-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=236511M
#SBATCH --job-name=GPM
#SBATCH --output=logs/gm-%N-%j.out
#SBATCH --error=logs/gm-%N-%j.err

# Ensure current environment is GPM_ENV
CURRENT_ENV=$(conda info --json | jq -r '.active_prefix_name')
if [ "$CURRENT_ENV" != "GPM_ENV" ]; then
    echo "Please activate the GPM_ENV conda environment before running this script."
    exit 1
fi

# mkdir logs, if not exist
mkdir -p logs

# Delete the old logs/cifar10-da.csv, if exist
LOG_FILE=logs/gm.csv
if [ -f "$LOG_FILE" ]; then
    rm -f "$LOG_FILE"
fi

# Create a new logs/cifar10-da.log, witt certain content
echo "dataset,model,pretrained,l2_mean,l2_median,l2_diff_mean,l2_diff_median" > $LOG_FILE

python mnist-gm.py >> $LOG_FILE
python mnist-gm.py --pretrained >> $LOG_FILE
for model in "vgg19_bn" "resnet50" "mobilenet_v2"; do 
    python cifar10-gm.py --model $model >> $LOG_FILE
    python cifar10-gm.py --model $model --pretrained >> $LOG_FILE
done

git add $LOG_FILE
git commit -m "Add results to $LOG_FILE"
git push
# End of script