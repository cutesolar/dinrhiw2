#!/bin/sh

autoheader
autoconf
./configure
make depend
make all




