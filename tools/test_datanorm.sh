#!/bin/sh

# creates training dataset for nntool: y = max(5 - ||x1+x2||, 0)
make gendata3

NNTOOL="./nntool"
DSTOOL="./dstool"
GENDATA="./gendata3"

$GENDATA 3

$DSTOOL -create gendata3-test.ds
$DSTOOL -create:6:input gendata3-test.ds
$DSTOOL -create:1:output gendata3-test.ds
$DSTOOL -list gendata3-test.ds
$DSTOOL -import:0 gendata3-test.ds norm_train_input.csv
$DSTOOL -import:1 gendata3-test.ds norm_train_output.csv
$DSTOOL -padd:0:meanvar gendata3-test.ds
# $DSTOOL -padd:0:pca wine-test.ds
$DSTOOL -padd:1:meanvar gendata3-test.ds

$DSTOOL -list gendata3-test.ds

# uses nntool trying to learn from dataset

## ARCH="6-100-100-100-1"
ARCH="6-50-50-1"

#$NNTOOL -v wine-test.ds $ARCH winenn.cfg mix
# $NNTOOL -v wine-test.ds $ARCH winenn.cfg lbfgs

################## $NNTOOL -v --samples 2000 wine-test.ds $ARCH winenn.cfg grad

$NNTOOL -v --time 6000 --overfit gendata3-test.ds $ARCH gendata3nn.cfg grad

# $NNTOOL -v --time 600 gendata-test.ds $ARCH gendatann.cfg grad

# $NNTOOL -v --time 10 wine-test.ds $ARCH winenn.cfg random

# testing

$NNTOOL -v gendata3-test.ds $ARCH gendata3nn.cfg use

# predicting [stores results to dataset]

cp -f gendata3-test.ds gendata3-pred.ds
$DSTOOL -clear:1 gendata3-pred.ds

$NNTOOL -v gendata3-pred.ds $ARCH gendata3nn.cfg use


$DSTOOL -print:1:49900:50000 gendata3-pred.ds
tail sort_train_output.csv
