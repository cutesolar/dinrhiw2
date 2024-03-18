#!/bin/sh

rm -f examples-test.ds
rm -f examplesnn.cfg

# creates training dataset for nntool

NNTOOL="./nntool"
DSTOOL="./dstool"

$DSTOOL -create examples-test.ds
$DSTOOL -create:3:input examples-test.ds
$DSTOOL -create:1:output examples-test.ds
$DSTOOL -list examples-test.ds
$DSTOOL -import:0 examples-test.ds examples.csv.in
$DSTOOL -import:1 examples-test.ds examples.csv.out
$DSTOOL -padd:0:meanvar examples-test.ds
# $DSTOOL -padd:0:pca examples-test.ds
$DSTOOL -padd:1:meanvar examples-test.ds

$DSTOOL -list examples-test.ds

# uses nntool trying to learn from dataset

# 20 layer neural network (works)
## ARCH="13-200-200-200-200-200-200-200-200-200-200-200-200-200-200-200-200-200-200-200-1"

ARCH="3-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-1"

## ARCH="13-100-1"

# 40 layer neural network (don't work very well)
# ARCH="13-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-20-1"


##ARCH="13-20-20-20-20-20-20-20-20-1"
## ARCH="13-20-1"

#$NNTOOL -v examples-test.ds $ARCH examplesnn.cfg mix
# $NNTOOL -v examples-test.ds $ARCH examplesnn.cfg lbfgs

################## $NNTOOL -v --samples 2000 examples-test.ds $ARCH examplesnn.cfg grad

$NNTOOL -v --time 600 --threads 4 examples-test.ds $ARCH examplesnn.cfg grad

## $NNTOOL -v --time 600 examples-test.ds $ARCH examplesnn.cfg grad

## $NNTOOL -v --time 60 examples-test.ds $ARCH examplesnn.cfg grad

# testing

$NNTOOL -v examples-test.ds $ARCH examplesnn.cfg use

# predicting [stores results to dataset]

cp -f examples-test.ds examples-pred.ds
$DSTOOL -clear:1 examples-pred.ds
# $DSTOOL -remove:1 examples-pred.ds

$NNTOOL -v examples-pred.ds $ARCH examplesnn.cfg use

# $DSTOOL -list examples-test.ds
# $DSTOOL -list examples-pred.ds

# $DSTOOL -print:1 examples-pred.ds
# tail examples_measurements.out
