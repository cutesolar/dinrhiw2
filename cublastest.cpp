
#include <stdio.h>

#ifdef CUBLAS

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// loading vertex.o object file initializes NVIDIA cuBLAS
extern cublasHandle_t cublas_handle;
extern cublasStatus_t cublas_status;

// loading vertex.o object file initializes cuBLAS
cublasHandle_t cublas_handle;
cublasStatus_t cublas_status  = cublasCreate(&cublas_handle);
cublasStatus_t cublas_math = cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);


volatile bool use_gpu_sync = true;

#endif


int main(void)
{
#ifdef CUBLAS
  
  cudaError_t cudaStat;
  void* cudaptr = NULL;
  cudaStat = cudaMallocManaged(&cudaptr, 1024); // 1 KB
  
  if(cudaStat != cudaSuccess || cudaptr == NULL){
    printf("CUDA MEMORY ALLOCATION FAILED.\n");
    fflush(stdout);
    return -1;
  }

  printf("CUDA Memory allocated at address: 0x%llx\n", (unsigned long long)cudaptr);
  fflush(stdout);
  
  cudaFree(cudaptr);
  
#endif

  return 0;
}
