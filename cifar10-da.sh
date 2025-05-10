# mkdir logs, if not exist
mkdir -p logs

# Delete the old logs/cifar10-da.csv, if exist
LOG_FILE=logs/cifar10-da.csv
if [ -f "$LOG_FILE" ]; then
    rm -f "$LOG_FILE"
fi

# Create a new logs/cifar10-da.log, witt certain content
echo "dataset,model,num_params,converged,acc,batch_size,clip_norm,beta,gamma,sucess_rate" > $LOG_FILE

EXP_SCRIPT="cifar10-da.py"

REPEAT=10
THRESHOLD=0.95


for model in "vgg19_bn" "resnet50" "mobilenet_v2"; do 
    for batch_size in 64 128 256; do
        for sigma in 0.1 0.5 1; do
            for beta in 1e-4 1e-3 1e-2; do
                for gamma in 1e2 1e3; do
                    for clip_norm in 4 8; do
                        # Run the python script with the parameters
                        python $EXP_SCRIPT --model $model --batch_size $batch_size --sigma $sigma --beta $beta --gamma $gamma --clip_norm $clip_norm --pretrained --repeat $REPEAT --success_threshold $THRESHOLD >> $LOG_FILE
                        python $EXP_SCRIPT --model $model --batch_size $batch_size --sigma $sigma --beta $beta --gamma $gamma --clip_norm $clip_norm --repeat $REPEAT --success_threshold $THRESHOLD >> $LOG_FILE
                    done
                done
            done
        done
    done
done