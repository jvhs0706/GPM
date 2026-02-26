if [ ! -f "./gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip" ]; then
  echo "Downloading weights..."
  wget https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip
else
  echo "Weights already downloaded."
fi

if [ -d "./state_dicts" ]; then
  rm -rf state_dicts
  mkdir state_dicts
fi

unzip gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip
rm gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip