#!/bin/sh

# creates training dataset for nntool: y = sha256(random_string_x)
# make gendata3

rm -f gendata3-test.ds
rm -f gendata3-pred.ds
rm -f gendata3nn.cfg

NNTOOL="./nntool"
DSTOOL="./dstool"
GENDATA="./gendata3"
DIM=10

$GENDATA $DIM

$DSTOOL -create gendata3-test.ds
$DSTOOL -create:$DIM:input gendata3-test.ds
$DSTOOL -create:$DIM:output gendata3-test.ds
$DSTOOL -list gendata3-test.ds

# inverse!

# hash is input
$DSTOOL -import:0 gendata3-test.ds hash_train_output.csv

# message is output
$DSTOOL -import:1 gendata3-test.ds hash_train_input.csv

$DSTOOL -data:1000 gendata3-test.ds

# chars are transformed (output)
# $DSTOOL -padd:0:meanvar gendata3-test.ds
#### $DSTOOL -padd:1:meanvar gendata3-test.ds

# $DSTOOL -padd:0:pca wine-test.ds



$DSTOOL -list gendata3-test.ds

# uses nntool trying to learn from dataset

# was 10-100-100-10
ARCH="$DIM-10000-100-1000-$DIM"
ARCH="$DIM-20-20-$DIM"

#$NNTOOL -v wine-test.ds $ARCH winenn.cfg mix
# $NNTOOL -v wine-test.ds $ARCH winenn.cfg lbfgs

################## $NNTOOL -v --samples 2000 wine-test.ds $ARCH winenn.cfg grad

$NNTOOL -v --overfit --noresidual gendata3-test.ds $ARCH gendata3nn.cfg sgrad

# $NNTOOL -v --time 600 gendata-test.ds $ARCH gendatann.cfg grad

# $NNTOOL -v --time 10 wine-test.ds $ARCH winenn.cfg random

# testing

$NNTOOL -v gendata3-test.ds $ARCH gendata3nn.cfg use

# predicting [stores results to dataset]

cp -f gendata3-test.ds gendata3-pred.ds
$DSTOOL -clear:1 gendata3-pred.ds
# $DSTOOL -remove:1 wine-pred.ds

$NNTOOL -v gendata3-pred.ds $ARCH gendata3nn.cfg use

$DSTOOL -print:1:49900:49999 gendata3-pred.ds
tail hash_train_input.csv
