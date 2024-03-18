#!/bin/sh

rm -f wine-test.ds
rm -f winenn.cfg

# creates training dataset for nntool

NNTOOL="./nntool"
DSTOOL="./dstool"

$DSTOOL -create wine-test.ds
$DSTOOL -create:13:input wine-test.ds
$DSTOOL -create:1:output wine-test.ds
$DSTOOL -list wine-test.ds
$DSTOOL -import:0 wine-test.ds wine_measurements.in
$DSTOOL -import:1 wine-test.ds wine_measurements.out
$DSTOOL -padd:0:meanvar wine-test.ds
# $DSTOOL -padd:0:pca wine-test.ds
$DSTOOL -padd:1:meanvar wine-test.ds

$DSTOOL -list wine-test.ds

# uses nntool trying to learn from dataset

# 20 layer neural network (works)
## ARCH="13-200-200-200-200-200-200-200-200-200-200-200-200-200-200-200-200-200-200-200-1"

ARCH="13-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-1"

## ARCH="13-100-1"

# 40 layer neural network (don't work very well)
ARCH="13-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-1"


##ARCH="13-20-20-20-20-20-20-20-20-1"
## ARCH="13-20-1"

#$NNTOOL -v wine-test.ds $ARCH winenn.cfg mix
# $NNTOOL -v wine-test.ds $ARCH winenn.cfg lbfgs

################## $NNTOOL -v --samples 2000 wine-test.ds $ARCH winenn.cfg grad

$NNTOOL -v --time 600 --threads 1 wine-test.ds $ARCH winenn.cfg grad

## $NNTOOL -v --time 600 wine-test.ds $ARCH winenn.cfg grad

## $NNTOOL -v --time 60 wine-test.ds $ARCH winenn.cfg grad

# testing

$NNTOOL -v wine-test.ds $ARCH winenn.cfg use

# predicting [stores results to dataset]

cp -f wine-test.ds wine-pred.ds
$DSTOOL -clear:1 wine-pred.ds
# $DSTOOL -remove:1 wine-pred.ds

$NNTOOL -v wine-pred.ds $ARCH winenn.cfg use

$DSTOOL -list wine-test.ds
$DSTOOL -list wine-pred.ds

$DSTOOL -print:1 wine-pred.ds
tail wine_measurements.out
