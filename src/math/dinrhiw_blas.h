
#ifndef dinrhiw_blas_h
#define dinrhiw_blas_h

extern "C" {

#ifdef OPENBLAS
  
#include <cblas.h>
#include "openblas_config.h"
  
#endif

#ifdef INTELMKL
#include "mkl.h"
#include "mkl_blas.h"
#endif

#ifdef AMDBLIS
#include <cblas.h>
#endif

}

#include "blas_primitives.h"
#include "superresolution.h"


#endif



