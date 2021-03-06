#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=/media/data/lzh/val
DATA=/media/data/lzh/MS-Celeb-1M
TOOLS=/home/deep-machine/caffe-master/build/tools

# TRAIN_DATA_ROOT=/path/to/imagenet/train/
# VAL_DATA_ROOT=/path/to/imagenet/val/
TRAIN_DATA_ROOT=/media/data/lzh/MS-Celeb-1M/train_data
VAL_DATA_ROOT=/media/data/lzh/MS-Celeb-1M/val_data

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
# RESIZE=false
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=224
  RESIZE_WIDTH=224
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

BACKEND="lmdb"
rm -rf $EXAMPLE/train_${BACKEND}
rm -rf $EXAMPLE/val_${BACKEND}

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $EXAMPLE/val_lmdb

echo "Done."
