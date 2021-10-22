#!/bin/sh

# creates training dataset for nntool: y = max(x1,x2,x3,x4,x5)
make gendata2

NNTOOL="./nntool"
DSTOOL="./dstool"
GENDATA="./gendata2"

$GENDATA 5

$DSTOOL -create gendata2-test.ds
$DSTOOL -create:5:input gendata2-test.ds
$DSTOOL -create:5:output gendata2-test.ds
$DSTOOL -list gendata2-test.ds
$DSTOOL -import:0 gendata2-test.ds sort_train_input.csv
$DSTOOL -import:1 gendata2-test.ds sort_train_output.csv
$DSTOOL -padd:0:meanvar gendata2-test.ds
# $DSTOOL -padd:0:pca wine-test.ds
$DSTOOL -padd:1:meanvar gendata2-test.ds

$DSTOOL -list gendata2-test.ds

# uses nntool trying to learn from dataset

## ARCH="5-1000-1"
# ARCH="5-10-10-10-1"
ARCH="5-100-100-100-5"

#$NNTOOL -v wine-test.ds $ARCH winenn.cfg mix
# $NNTOOL -v wine-test.ds $ARCH winenn.cfg lbfgs

################## $NNTOOL -v --samples 2000 wine-test.ds $ARCH winenn.cfg grad

$NNTOOL -v --time 6000 --overfit gendata2-test.ds $ARCH gendata2nn.cfg grad

# $NNTOOL -v --time 600 gendata-test.ds $ARCH gendatann.cfg grad

# $NNTOOL -v --time 10 wine-test.ds $ARCH winenn.cfg random

# testing

$NNTOOL -v gendata2-test.ds $ARCH gendata2nn.cfg use

# predicting [stores results to dataset]

cp -f gendata2-test.ds gendata2-pred.ds
$DSTOOL -clear:1 gendata2-pred.ds
# $DSTOOL -remove:1 wine-pred.ds

$NNTOOL -v gendata2-pred.ds $ARCH gendata2nn.cfg use

#$DSTOOL -list wine-test.ds
#$DSTOOL -list wine-pred.ds

$DSTOOL -print:1:49900:50000 gendata2-pred.ds
tail sort_train_output.csv
