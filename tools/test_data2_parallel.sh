#!/bin/sh

rm -f wdbc-test.ds

# creates training dataset for nntool

./dstool -create wdbc-test.ds
./dstool -create:30:input wdbc-test.ds
./dstool -create:1:output wdbc-test.ds
./dstool -list wdbc-test.ds
./dstool -import:0 wdbc-test.ds wdbc.in
./dstool -import:1 wdbc-test.ds wdbc.out
./dstool -padd:0:meanvar wdbc-test.ds
./dstool -padd:0:pca wdbc-test.ds
./dstool -padd:1:meanvar wdbc-test.ds

./dstool -list wdbc-test.ds

# uses nntool trying to learn from dataset (4 min long search)

./nntool --time 240 -v wdbc-test.ds ?-20-? wdbcnn.cfg grad

##################################################
# testing

./nntool -v wdbc-test.ds ?-20-? wdbcnn.cfg use

##################################################
# predicting [stores results to dataset]

cp -f wdbc-test.ds wdbc-pred.ds
./dstool -clear:1 wdbc-pred.ds
# ./dstool -remove:1 wine-pred.ds

./nntool -v wdbc-pred.ds 30-20-1 wdbcnn.cfg use

./dstool -list wdbc-test.ds
./dstool -list wdbc-pred.ds

./dstool -print:1 wdbc-pred.ds
tail wdbc.out


