dnl Process this file with autoconf to produce a configure script.

AC_INIT([whiteice::dinrhiw],[<tomas.ukkonen@iki.fi>],[libdinrhiw])

dnl finds the basic compiling tools
AC_PROG_CC
AC_PROG_CXX
AC_PROG_CPP

dnl needed by autoconf itself
AC_PROG_FGREP

AC_CHECK_SIZEOF(char, 1)
AC_CHECK_SIZEOF(short, 2)
AC_CHECK_SIZEOF(int, 4)
AC_CHECK_SIZEOF(long, 4)
AC_CHECK_SIZEOF(long long int, 8)

dnl does source files seem to exist?
AC_CONFIG_SRCDIR([src/dinrhiw.h])

dnl ----------------------------------------------------------
dnl do all the magick stuff here


dnl checks if -I/usr/local/include is needed
AC_MSG_CHECKING([is -I/usr/local/include -I/ucrt64/include/ command line parameter needed])

EXTRA_INCLUDEPATHS="-I/usr/local/include/"

if [[ -d "/ucrt64" ]]; then
   EXTRA_INCLUDEPATHS="-I/usr/local/include -I/ucrt64/include/ -I/ucrt64/include/openblas/"
   AC_MSG_RESULT([yes])
else
   AC_MSG_RESULT([no])
fi


dnl checks if -L/usr/local/lib is needed to
dnl link with libraries in /usr/local/lib
AC_MSG_CHECKING([is -L/usr/local/lib -L/ucrt64/lib/ command line parameter needed])

EXTRA_LIBPATHS="-L/usr/local/lib"

if [[ -d "/ucrt64" ]]; then
   EXTRA_LIBPATHS="-L/usr/local/lib -L/ucrt64/lib/"
   AC_MSG_RESULT([yes])
else
   AC_MSG_RESULT([no])
fi


dnl check for Windows (WINNT)
AC_MSG_CHECKING([Windows OS])
EXTRA_DEFINES=""
SHAREDLIBSUFFIX="so"
`g++ -o windowstest.exe windowstest.cpp > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
   EXTRA_DEFINES="-DWINOS"
   SHAREDLIBSUFFIX="dll"
dnl    AC_MSG_RESULT([yes, $conf_flags])
    AC_MSG_RESULT([yes])
else
    AC_MSG_RESULT([no])
fi


dnl checks if isfinite() is defined
AC_MSG_CHECKING([for isfinite()])
has_isfinite="no"

AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
[#include <math.h>]
[char test(void){ return isfinite(0.0f); }]])],
[has_isfinite="yes"])

if [test $has_isfinite = "yes"]; then
  AC_DEFINE(HAVE_ISFINITE,[1],[Defined if isfinite() is in <math.h>])
fi

AC_MSG_RESULT([$has_isfinite])

dnl checks if gettimeofday() is either in time.h or sys/time.h
AC_MSG_CHECKING([if gettimeofday() is available])


AC_LINK_IFELSE([AC_LANG_PROGRAM([
[#include <time.h>]
[#include <sys/time.h>]],
[struct timeval tv; gettimeofday(&tv,0);])],
[has_gettimeofday="yes"],
[has_gettimeofday="no"])

if [test $has_gettimeofday = "yes" ]; then
  AC_DEFINE(HAVE_GETTIMEOFDAY,[1],[Defined if gettimeofday() can be found.])
fi

AC_MSG_RESULT([$has_gettimeofday])


AC_MSG_CHECKING(['uname' command])
cpufamily=`uname -m`

if [[ $? -eq 0 ]]; then
    AC_MSG_RESULT([ok])
else
    AC_MSG_ERROR([DOESN'T EXIST!])
fi

AC_MSG_CHECKING([CPU family])
AC_MSG_RESULT($cpufamily)

dnl enables CPU specific optimization flags

AC_ARG_ENABLE([optimize],
	AS_HELP_STRING([--disable-optimize],[Disable optimization]))

optimization_flags="-fPIC -O3 -ffast-math -mtune=native -ftree-vectorize"

AS_IF([test "x$enable_optimize" == "xno"],[
  optimization_flags="-fPIC -g"
  CFLAGS=""
  CXXFLAGS=""
])

conf_flags=""

dnl -fPIC is needed on Linux to build shared libraries

if [[ $cpufamily = "x86_64" ]]; then
    optimization_flags="-fPIC -O3 -g -DAMD64 -ffast-math -mtune=native -ftree-vectorize"

    AS_IF([test "x$enable_optimize" == "xno"],[
      optimization_flags="-fPIC -g -DAMD64"
      CFLAGS=""
      CXXFLAGS=""
    ])

    conf_flags="-DAMD64"
elif [[ "$cpufamily" = "i386" -o "$cpufamily" = "i486" -o "$cpufamily=i586" -o "$cpufamily" = "i686" -o "$cpufamily" = "i786" -o "$cpufamily" = "i886" -o "$cpufamily" = "i986" ]]; then

    optimization_flags="-fPIC -O3 -g -DIA32 -ffast-math -mtune=native -ftree-vectorize"

    AS_IF([test "x$enable_optimize" == "xno"],[
      optimization_flags="-fPIC -g -DIA32"
      CFLAGS=""
      CXXFLAGS=""
    ])

    conf_flags="-DIA32"
else
    AC_MSG_ERROR($cpufamily [, unrecognized/unsupported machine architecture.])
fi


AC_ARG_ENABLE([openmp],
	AS_HELP_STRING([--disable-openmp],[Disable OpenMP]))

AS_IF([test "x$enable_openmp" != "xno"],[
  conf_flags="$conf_flags -fopenmp"
  optimization_flags="$optimization_flags -fopenmp"
],[
  conf_flags="$conf_flags -Wno-unknown-pragmas"
  optimization_flags="$optimization_flags -Wno-unknown-pragmas"
])



AC_MSG_CHECKING([CPUID identification])
CPUINFO_FILE="cpuinfo"
`rm -rf cputest cpuinfo &> /dev/null`
`gcc cputest.c -o cputest && ./cputest > cpuinfo`

if [[ $? -eq 0 ]]; then
    AC_MSG_RESULT([ok])
else
    AC_MSG_RESULT([failure])
    `$FGREP flags /proc/cpuinfo > cpuinfo`
fi

dnl check for GMP integer library
AC_MSG_CHECKING([for GMP library])
GMP_LIBRARY="no"

`gcc -o gmp gmptest.c -I/usr/include/ -L/usr/lib/ -I/usr/local/include/ -L/usr/local/lib/ -lgmp > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  GMP_LIBRARY="yes"
  AC_MSG_RESULT([yes])
else
  AC_MSG_ERROR([no])
fi


CBLAS_HEADERS="no"


AC_ARG_ENABLE([openblas],
	AS_HELP_STRING([--disable-openblas],[Disable OpenBLAS library]))
 
AC_MSG_CHECKING([for OpenBLAS library (1)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/usr/local/include/ -L/usr/local/lib/ -lopenblas > /dev/null 2> /dev/null`
 
if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [AS_IF([test "x$OPENBLAS" != "xyes"],[
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/usr/local/include/"
     BLAS_LIBS="-lopenblas -L/usr/local/lib/"
     AC_MSG_RESULT([yes, headers in /usr/local/include/])
  ],
  [
     AC_MSG_RESULT([no, already detected])
  ])])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi

AC_MSG_CHECKING([for OpenBLAS library (2)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/include/ -L/lib/ -lopenblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [AS_IF([test "x$OPENBLAS" != "xyes"],[
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/include/"
     BLAS_LIBS="-L/lib/ -lopenblas"
     AC_MSG_RESULT([yes, headers in /include/])
  ],
  [
     AC_MSG_RESULT([no, already detected])
  ])])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi


AC_MSG_CHECKING([for OpenBLAS library (3)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/usr/include/openblas -L/lib/ -lopenblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [AS_IF([test "x$OPENBLAS" != "xyes"],[
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/usr/include/openblas/"
     BLAS_LIBS="-L/lib/ -lopenblas"
     AC_MSG_RESULT([yes, headers in /usr/include/openblas/])
  ],
  [
     AC_MSG_RESULT([no, already detected])
  ])])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi

AC_MSG_CHECKING([for OpenBLAS library (4)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/usr/local/include/openblas -L/lib/ -lopenblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [AS_IF([test "x$OPENBLAS" != "xyes"],[
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/usr/local/include/openblas/"
     BLAS_LIBS="-L/lib/ -lopenblas"
     AC_MSG_RESULT([yes, headers in /usr/local/include/openblas/])
  ],
  [
     AC_MSG_RESULT([no, already detected])
  ])])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi


AC_MSG_CHECKING([for OpenBLAS library (5)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/opt/OpenBLAS/include/ -L/opt/OpenBLAS/bin/ -lopenblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [AS_IF([test "x$OPENBLAS" != "xyes"],[
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/opt/OpenBLAS/include/"
     BLAS_LIBS="-L/opt/OpenBLAS/bin/ -lopenblas"
     AC_MSG_RESULT([yes, headers in /opt/OpenBLAS/include/])
  ],
  [
     AC_MSG_RESULT([no, already detected])
  ])])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi


AC_MSG_CHECKING([for OpenBLAS library (6)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/opt/include/ -L/opt/lib/ -lopenblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [AS_IF([test "x$OPENBLAS" != "xyes"],[
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/opt/include/"
     BLAS_LIBS="-L/opt/lib/ -lopenblas"
     AC_MSG_RESULT([yes, headers in /opt/include/])
  ],
  [
     AC_MSG_RESULT([no, already detected])
  ])])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi

AC_MSG_CHECKING([for OpenBLAS library (7)])
obcflags=`pkg-config openblas --cflags 2> /dev/null`
oblibs=`pkg-config openblas --libs 2> /dev/null`
`gcc -o openblastest openblastest.c -DOPENBLAS $obcflags $oblibs > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [AS_IF([test "x$OPENBLAS" != "xyes"],[
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS `pkg-config openblas --cflags`"
     BLAS_LIBS="`pkg-config openblas --libs`"
     AC_MSG_RESULT([yes, pkg-config conf file])
  ],
  [
     AC_MSG_RESULT([no, already detected])
  ])])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi


AC_MSG_CHECKING([for OpenBLAS library (8)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/ucrt64/include/openblas/ -L/ucrt64/lib/ -lopenblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [AS_IF([test "x$OPENBLAS" != "xyes"],[
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/ucrt64/include/openblas/"
     BLAS_LIBS="-L/ucrt64/lib/ -lopenblas"
     AC_MSG_RESULT([yes, headers in /ucrt64/include/openblas/])
  ],
  [
     AC_MSG_RESULT([no, already detected])
  ])])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi


AC_MSG_CHECKING([for OpenBLAS library (9)])
obcflags=`pkg-config openblas64 --cflags > /dev/null 2> /dev/null`
oblibs=`pkg-config openblas64 --libs > /dev/null 2> /dev/null`
`gcc -o openblastest openblastest.c -DOPENBLAS $obcflags $oblibs > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [AS_IF([test "x$OPENBLAS" != "xyes"],[
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS `pkg-config openblas64 --cflags 2> /dev/null`"
     BLAS_LIBS="`pkg-config openblas64 --libs 2> /dev/null`"
     AC_MSG_RESULT([yes, pkg-config conf file])
  ],
  [
     AC_MSG_RESULT([no, already detected])
  ])])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi



AC_ARG_ENABLE([mkl],
	AS_HELP_STRING([--disable-mkl],[Disable Intel MKL library]))

MKLROOT="/opt/intel/compilers_and_libraries/linux/mkl"
AC_MSG_CHECKING([for Intel MKL library (1) (Linux)])
`g++ -o openblastest openblastest.c -DINTELMKL -I$MKLROOT/include/ -L$MKLROOT/lib/intel64_lin -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_mkl" == "xno"],[
     INTELMKL="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     INTELMKL="yes"
     BLAS_CFLAGS="-m64 -I$MKLROOT/include -DINTELMKL"
     BLAS_LIBS="-L$MKLROOT/lib/intel64_lin -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
     AC_MSG_RESULT([yes, installed in $MKLROOT])
  ])
else
   AC_MSG_RESULT([no Intel MKL detected])
   if [[ "x$INTELMKL" != "xyes" ]]; then
      INTELMKL="no"
   fi	
fi


AC_MSG_CHECKING([for Intel MKL library (2) (Linux, pkg-config)])
PKG_MKL_CFLAGS=`pkg-config mkl-sdl-lp64 --cflags 2> /dev/null`
PKG_MKL_LIBS=`pkg-config mkl-sdl-lp64 --libs 2> /dev/null`
`g++ -o openblastest openblastest.c -DINTELMKL $PKG_MKL_CFLAGS $PKG_MKL_LIBS > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_mkl" == "xno"],[
     INTELMKL="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     INTELMKL="yes"
     BLAS_CFLAGS="$PKG_MKL_CFLAGS -DINTELMKL"
     BLAS_LIBS="$PKG_MKL_LIBS"
     AC_MSG_RESULT([yes, pkg-config mkl-sdl-lp64 FOUND.])
  ])
else
   AC_MSG_RESULT([no Intel MKL detected])
   if [[ "x$INTELMKL" != "xyes" ]]; then
      INTELMKL="no"
   fi	
fi


MKLROOT="/opt/intel/psxe_runtime/linux/mkl"
AC_MSG_CHECKING([for Intel MKL library (3) (Linux)])
`g++ -o openblastest openblastest.c -DINTELMKL -I$MKLROOT/include/ -L$MKLROOT/lib/intel64_lin -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_mkl" == "xno"],[
     INTELMKL="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     INTELMKL="yes"
     BLAS_CFLAGS="-m64 -I$MKLROOT/include -DINTELMKL"
     BLAS_LIBS="-L$MKLROOT/lib/intel64_lin -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
     AC_MSG_RESULT([yes, installed in $MKLROOT])
  ])
else
   AC_MSG_RESULT([no Intel MKL detected])
   if [[ "x$INTELMKL" != "xyes" ]]; then
      INTELMKL="no"
   fi	
fi


MKLROOT="/opt/intel/oneapi/mkl/latest"
AC_MSG_CHECKING([for Intel MKL library (4) (Linux)])
`g++ -o openblastest openblastest.c -DINTELMKL -I$MKLROOT/include/ -L$MKLROOT/lib/ -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_mkl" == "xno"],[
     INTELMKL="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     INTELMKL="yes"
     BLAS_CFLAGS="-m64 -I$MKLROOT/include -DINTELMKL"
     BLAS_LIBS="-L$MKLROOT/lib/ -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
     AC_MSG_RESULT([yes, installed in $MKLROOT])
  ])
else
   AC_MSG_RESULT([no Intel MKL detected])
   if [[ "x$INTELMKL" != "xyes" ]]; then
      INTELMKL="no"
   fi	
fi

dnl Intel MKL in Windows
MKLROOT=/c/Program\ Files\ \(x86\)/IntelSWTools/compilers_and_libraries/windows/mkl

AC_MSG_CHECKING([for Intel MKL library (Windows)])
`g++ -o openblastest openblastest.c -DINTELMKL -I$MKLROOT/include/ -L$MKLROOT/lib/intel64_win -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lgomp -lm > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_mkl" == "xno"],[
     INTELMKL="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     INTELMKL="yes"
     BLAS_CFLAGS="-m64 -I$MKLROOT/include -DINTELMKL"
     BLAS_LIBS="-L$MKLROOT/lib/intel64_win -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lgomp -lpthread -lm -ldl"
     AC_MSG_RESULT([yes, installed in $MKLROOT])
  ])
else
   AC_MSG_RESULT([no working Intel MKL detected])
   if [[ "x$INTELMKL" != "xyes" ]]; then
      INTELMKL="no"
   fi	
fi

dnl AMD BLIS library support
AC_ARG_ENABLE([amdblis],
	AS_HELP_STRING([--disable-amdblis],[Disable AMD BLIS (BLAS) library (must be installed in /opt/amd-blis/ directory)]))
 
AC_MSG_CHECKING([for AMD BLIS library])
`gcc -o openblastest openblastest.c -DAMDBLIS -I/opt/amd-blis/include/LP64/ -L/opt/amd-blis/lib/LP64/ -lblis-mt > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_amdblis" == "xno"],[
     AMDBLIS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     AMDBLIS="yes"
     BLAS_CFLAGS="-DAMDBLIS -I/opt/amd-blis/include/LP64/"
     BLAS_LIBS="-lblis-mt -L/opt/amd-blis/lib/LP64/"
     AC_MSG_RESULT([yes, headers in /opt/amd-blis/include/])
  ])
else
  AC_MSG_RESULT([no AMD BLIS detected])
  if [[ "x$AMDBLIS" != "xyes" ]]; then
     AMDBLIS="no"
  fi
fi
 
dnl    AC_MSG_RESULT([AMD BLIS currently doesn't support full CBLAS!])
dnl   if [[ "x$AMDBLIS" != "xyes" ]]; then
dnl      AMDBLIS="no"
dnl   fi



AC_ARG_ENABLE([cublas],
	AS_HELP_STRING([--disable-cublas],[Disable NVIDIA cuBLAS library]))

CUBLASROOT="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
cublasdir=`ls -1 -Art "$CUBLASROOT" 2> /dev/null | tail -n 1`
CUBLASROOT="$CUBLASROOT/$cublasdir"

AC_MSG_CHECKING([for NVIDIA cuBLAS library [SLOW] (Windows)])
`g++ -o cublastest cublastest.cpp -DCUBLAS "-I$CUBLASROOT/include/" "-L$CUBLASROOT/lib/x64/" -lcublas -lcudart > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_cublas" == "xno"],[
     NVIDIACUBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     NVIDIACUBLAS="yes"
     BLAS_CFLAGS="\"-I$CUBLASROOT/include/\" -DCUBLAS"
     BLAS_LIBS="\"-L$CUBLASROOT/lib/x64/\" -lcublas -lcudart"
     AC_MSG_RESULT([yes, installed in $CUBLASROOT])
  ])
else
   AC_MSG_RESULT([no NVIDIA cuBLAS detected])
   if [[ "x$NVIDIACUBLAS" != "xyes" ]]; then
      NVIDIACUBLAS="no"
   fi
fi


AC_MSG_CHECKING([for NVIDIA cuBLAS 11.8 library [SLOW] (Linux)])
CUBLAS_FLAGS="`pkg-config cublas-11.8 --cflags --libs 2> /dev/null` `pkg-config cudart-11.8 --cflags --libs 2> /dev/null`"
`g++ -o cublastest cublastest.cpp -DCUBLAS $CUBLAS_FLAGS > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_cublas" == "xno"],[
     NVIDIACUBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     NVIDIACUBLAS="yes"x
     BLAS_CFLAGS="-DCUBLAS `pkg-config cublas-11.8 --cflags` `pkg-config cudart-11.8 --cflags`"
     BLAS_LIBS="`pkg-config cublas-11.8 --libs` `pkg-config cudart-11.8 --libs`"
     AC_MSG_RESULT([yes, using pkg-config cublas-11.8 cudart-11.8])
  ])
else
   AC_MSG_RESULT([no NVIDIA cuBLAS detected])
   if [[ "x$NVIDIACUBLAS" != "xyes" ]]; then
      NVIDIACUBLAS="no"
   fi	
fi

AC_MSG_CHECKING([for NVIDIA cuBLAS 12.5 library [SLOW] (Linux pkg-config)])
CUBLAS_FLAGS="`pkg-config cublas-12.5 --cflags --libs 2> /dev/null` `pkg-config cudart-12.5 --cflags --libs 2> /dev/null`"
`g++ -o cublastest cublastest.cpp -DCUBLAS $CUBLAS_FLAGS > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_cublas" == "xno"],[
     NVIDIACUBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     NVIDIACUBLAS="yes"x
     BLAS_CFLAGS="-DCUBLAS `pkg-config cublas-12.5 --cflags` `pkg-config cudart-12.5 --cflags`"
     BLAS_LIBS="`pkg-config cublas-12.5 --libs` `pkg-config cudart-12.5 --libs`"
     AC_MSG_RESULT([yes, using pkg-config cublas-12.5 cudart-12.5])
  ])
else
   AC_MSG_RESULT([no NVIDIA cuBLAS detected])
   if [[ "x$NVIDIACUBLAS" != "xyes" ]]; then
      NVIDIACUBLAS="no"
   fi
fi





AC_ARG_ENABLE([nvblas],
	AS_HELP_STRING([--disable-nvblas],[Disable NVIDIA NVBLAS library]))

CUBLASROOT="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
cublasdir=`ls -1 -Art "$CUBLASROOT" 2> /dev/null | tail -n 1`
CUBLASROOT="$CUBLASROOT/$cublasdir"

AC_MSG_CHECKING([for NVIDIA NVBLAS library with CBLAS inteface (don't exist)])
`g++ -o openblastest openblastest.c -DNVBLAS "-I$CUBLASROOT/include/" "-L$CUBLASROOT/lib/x64/" -lnvblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_nvblas" == "xno"],[
     NVIDIANVBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     NVIDIANVBLAS="yes"
     BLAS_CFLAGS="\"-I$CUBLASROOT/include/\" -DNVBLAS"
     BLAS_LIBS="\"-L$CUBLASROOT/lib/x64/\" -lnvblas"
     AC_MSG_RESULT([yes, installed in $CUBLASROOT])
  ])
else
   AC_MSG_RESULT([no NVIDIA NVBLAS with CBLAS detected])
   NVIDIANVBLAS="no"
fi


if [[ $INTELMKL == "no" -a $OPENBLAS == "no" ]]; then
   if [[ $NVIDIACUBLAS == "no" -a $NVIDIANVBLAS == "no" ]]; then
      if [[ $AMDBLIS == "no" ]]; then
           AC_MSG_ERROR([NO BLAS DETECTED])
      fi
   fi
fi


dnl ZLIB IS DISABLED AND NOT USED ANYMORE
dnl 
dnl dnl checking for ZLIB header files and library
dnl AC_MSG_CHECKING([for ZLIB library and headers])
ZLIB_USABLE="no"
dnl 
dnl `gcc -o zlibtest zlibtest.c -I/usr/local/include/ -L/usr/local/lib/ -lz > /dev/null 2> /dev/null`
dnl 
dnl if [[ $? -eq 0 ]]; then
dnl    ZLIB_USABLE="yes"
dnl    AC_MSG_RESULT([yes])
dnl else
dnl    AC_MSG_ERROR([no, install ZLIB and header files])
dnl fi


dnl adds --with-atlaslibpath
dnl ATLAS support is NOT working / is broken!!!
dnl ATLASLIBS_PATH="/usr/local/lib/"
dnl 
dnl AC_ARG_WITH([atlaslpath], AS_HELP_STRING([--with-atlaslpath=PATH],[sets search path for ATLAS libraries]),
dnl			  [if [[ -n $withval ]]; then ATLASLIBS_PATH=$withval; fi], [])
dnl


optimization_flags="$optimization_flags -mtune=native"


AC_ARG_ENABLE([native],
	AS_HELP_STRING([--enable-native],[Enable native optimization]))

AS_IF([test "x$enable_native" = "xyes"],[
  optimization_flags="$optimization_flags -march=native"
])

AC_ARG_ENABLE([cppdebug],
	AS_HELP_STRING([--enable-cppdebug],[Enable C++ debugging flags (stdc++)]))

AS_IF([test "x$enable_cppdebug" = "xyes"],[
  optimization_flags="$optimization_flags -D_GLIBCXX_DEBUG -fsanitize=address -fsanitize=undefined"
dnl  optimization_flags="$optimization_flags -D_GLIBCXX_DEBUG -fsanitize=address"
  conf_flags="$conf_flags -D_GLIBCXX_DEBUG -fsanitize=address -fsanitize=undefined"
])

AC_ARG_ENABLE([threadsdebug],
	AS_HELP_STRING([--enable-threadsdebug],[Enable g++ thread sanitizer debugging]))

AS_IF([test "x$enable_threadsdebug" = "xyes"],[
  optimization_flags="$optimization_flags -fsanitize=thread"
dnl  optimization_flags="$optimization_flags -D_GLIBCXX_DEBUG -fsanitize=address"
  conf_flags="$conf_flags -fsanitize=thread"
])

dnl enable newer C++ standards
dnl CFLAGS="-std=gnu++20 $CFLAGS"
dnl conf_flags="-std=gnu++20 -Wno-strict-aliasing $conf_flags `python3-config --cflags --embed`"
CFLAGS="$CFLAGS"
conf_flags="-Wno-strict-aliasing $conf_flags"
dnl `python3-config --cflags --embed`"

dnl 
dnl AC_ARG_ENABLE([profiling],
dnl	AS_HELP_STRING([--enable-profiling],[Enable code profiling (speed) mode (-pg)]))
dnl 
dnl AS_IF([test "x$enable_profiling" = "xyes"],[
dnl   optimization_flags="$optimization_flags -pg"
dnl ])
dnl 


EXTRA_INCLUDEPATHS="-I/usr/include/ $EXTRA_INCLUDEPATHS $BLAS_CFLAGS $EXTRA_DEFINES"
EXTRA_LIBS="-L/usr/lib/ $EXTRA_LIBS $BLAS_LIBS -lstdc++"
dnl `python3-config --libs --embed`"

AC_SUBST([EXTRA_INCLUDEPATHS],[$EXTRA_INCLUDEPATHS])
AC_SUBST([EXTRA_LIBPATHS],[$EXTRA_LIBPATHS])
AC_SUBST([EXTRA_LIBS],[$EXTRA_LIBS])
AC_SUBST([optimization_flags],[$optimization_flags])
AC_SUBST([conf_flags], [$conf_flags])
AC_SUBST([conf_libs], [$conf_libs])
AC_SUBST([SHAREDLIBSUFFIX], [$SHAREDLIBSUFFIX])

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile src/Makefile src/math/Makefile])
AC_CONFIG_FILES([src/crypto/Makefile src/neuralnetwork/Makefile])
AC_CONFIG_FILES([src/datamining/Makefile src/hmm/Makefile])
AC_CONFIG_FILES([src/reinforcement/Makefile src/rnn/Makefile])
AC_CONFIG_FILES([src/decisiontree/Makefile])
AC_CONFIG_FILES([tools/Makefile])
AC_CONFIG_FILES([dinrhiw.pc])
AC_OUTPUT


