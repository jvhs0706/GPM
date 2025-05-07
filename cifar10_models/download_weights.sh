# If the ./weights directory does not exist, create it
if [ ! -d "./weights" ]; then
  mkdir ./weights
fi
# Download the weights from the provided URL and unzip them into the ./weights directory
wget https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip
unzip gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip ./weights