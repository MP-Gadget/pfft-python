#!/bin/sh -e

PREFIX="$1"
PFFT_VERSION=1.0.7-a95386
FFTW_VERSION=3.3.3
TMP="tmp-pfft-$PFFT_VERSION"
LOGFILE="build.log"

# bash check if directory exists
if [ -d $TMP ]; then
        echo "Directory $TMP already exists. Delete it? (y/n)"
	answer='y'
	if [ ${answer} = "y" ]; then
		rm -rf $TMP
	else
		echo "Program aborted."
		exit 1
	fi
fi

mkdir $TMP 

gzip -dc depends/pfft-$PFFT_VERSION.tar.gz | tar xvf - -C $TMP
cd $TMP

cd pfft-$PFFT_VERSION

./configure --prefix=$PREFIX --disable-shared --enable-static  \
--disable-fortran --disable-doc \
2>&1 | tee $LOGFILE

make -j 4 2>&1 | tee $LOGFILE
make install 2>&1 | tee $LOGFILE

