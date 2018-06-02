#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=.
DATA=.
TOOLS=/home/deep-machine/caffe-master/build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $DATA/mean.binaryproto

echo "Done."
