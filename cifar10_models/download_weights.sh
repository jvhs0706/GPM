BACKUP=https://cs.uwaterloo.ca/~h299sun/files/gpm-cifar10-state-dicts.zip
ZIP_FILE_NAME=state-dicts.zip
if [ ! -f "./$ZIP_FILE_NAME" ]; then
  echo "Downloading weights from backup link..."
  wget $BACKUP -O $ZIP_FILE_NAME
else
  echo "Weights already downloaded."
fi

unzip $ZIP_FILE_NAME
rm $ZIP_FILE_NAME