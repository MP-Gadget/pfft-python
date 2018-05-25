#!/bin/sh -e

PREFIX="$1"
shift
OPTIMIZE="$*"
OPTIMIZE1=`echo "$*" | sed 's;enable-sse2;enable-sse;'`
echo "Optimization for double" ${OPTIMIZE}
echo "Optimization for single" ${OPTIMIZE1}

PFFT_VERSION=1.0.8-alpha3-fftw3-2don2d
TMP="tmp-pfft-$PFFT_VERSION"
LOGFILE="build.log"

mkdir -p $TMP 
ROOT=`dirname $0`/../
if ! [ -f $ROOT/depends/pfft-$PFFT_VERSION.tar.gz ]; then
echo curl -L -o $ROOT/depends/pfft-$PFFT_VERSION.tar.gz \
https://github.com/rainwoodman/pfft/releases/download/$PFFT_VERSION/pfft-$PFFT_VERSION.tar.gz
curl -L -o $ROOT/depends/pfft-$PFFT_VERSION.tar.gz \
https://github.com/rainwoodman/pfft/releases/download/$PFFT_VERSION/pfft-$PFFT_VERSION.tar.gz
fi

if ! [ -f $ROOT/depends/pfft-$PFFT_VERSION.tar.gz ]; then
echo wget -P $ROOT/depends/ \
https://github.com/rainwoodman/pfft/releases/download/$PFFT_VERSION/pfft-$PFFT_VERSION.tar.gz
wget -P $ROOT/depends/ \
https://github.com/rainwoodman/pfft/releases/download/$PFFT_VERSION/pfft-$PFFT_VERSION.tar.gz
fi

if ! [ -f $ROOT/depends/pfft-$PFFT_VERSION.tar.gz ]; then
echo "Failed to get https://github.com/rainwoodman/pfft/releases/download/$PFFT_VERSION/pfft-$PFFT_VERSION.tar.gz"
echo "Please check curl or wget"
echo "You can also download it manually to $ROOT/depends/"
exit 1
fi

gzip -dc $ROOT/depends/pfft-$PFFT_VERSION.tar.gz | tar xf - -C $TMP
cd $TMP

(
mkdir -p double;cd double

../pfft-${PFFT_VERSION}/configure --prefix=$PREFIX --disable-shared --enable-static  \
--disable-fortran --disable-doc --enable-mpi ${OPTIMIZE} &&
make -j 4   &&
make install && echo "PFFT_DONE"
) 2>&1 |tee ${LOGFILE}.double | awk "{printf(\".\")} NR % 40 == 0 {printf(\"\n\")} END {printf(\"\n\")}"

if ! grep PFFT_DONE ${LOGFILE}.double > /dev/null; then
    tail ${LOGFILE}.double
    exit 1
fi
(
mkdir -p single;cd single
../pfft-${PFFT_VERSION}/configure --prefix=$PREFIX --enable-single --disable-shared --enable-static  \
--disable-fortran --disable-doc --enable-mpi $2 ${OPTIMIZE1} &&
make -j 4  &&
make install && echo "PFFT_DONE"
) 2>&1 |tee ${LOGFILE}.single | awk "{printf(\".\")} NR % 40 == 0 {printf(\"\n\")} END {printf(\"\n\")}"

if ! grep PFFT_DONE ${LOGFILE}.single > /dev/null; then
    tail ${LOGFILE}.single
    exit 1
fi
