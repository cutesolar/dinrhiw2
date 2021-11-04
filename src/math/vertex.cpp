

#ifndef vertex_cpp
#define vertex_cpp

#include "vertex.h"
#include "dinrhiw_blas.h"
#include "ownexception.h"
#include "gcd.h"
#include "number.h"

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <exception>
#include <typeinfo>
#include <vector>
#include <new>
#include <cassert>

#include <stdlib.h>
#include <string.h>
#include "Log.h"

#ifdef CUBLAS

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"


// loading vertex.o object file initializes cuBLAS
cublasHandle_t cublas_handle;
cublasStatus_t cublas_status  = cublasCreate(&cublas_handle);
cublasStatus_t cublas_math = cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);


volatile bool use_gpu_sync = true;

#endif

// #define OPENBLAS 0

namespace whiteice
{
  namespace math
  {
    
    template <typename T>
    vertex<T>::vertex()
    {
      this->compressor = nullptr;
      this->dataSize = 0;      
      this->data = nullptr;

#ifdef CUBLAS

      cudaError_t cudaStat;
      void* cudaptr = NULL;
      cudaStat = cudaMallocManaged(&cudaptr, 1*sizeof(T));

      if(cudaStat != cudaSuccess || cudaptr == NULL){
	whiteice::logging.error("vertex ctor: cudaMallocManaged() failed.");
	throw CUDAException("CUBLAS memory allocation failure.");
      }

      // no memory initialization!!
      
      this->data = (T*)cudaptr;
      
#else
      
      this->data = (T*)malloc(sizeof(T));
      if(this->data == nullptr) throw std::bad_alloc();
      
      memset(this->data, 0, sizeof(T));
#endif
      
      this->dataSize = 1;      
    }
    
    
    // vertex ctor, i is dimension of vector
    template <typename T>
    vertex<T>::vertex(unsigned int i)
    {
      this->compressor = nullptr;
      this->dataSize = 0;
      this->data = nullptr;

#if CUBLAS

      if(i > 0){
	cudaError_t cudaStat;
	void* cudaptr = NULL;
	cudaStat = cudaMallocManaged(&cudaptr, i*sizeof(T));
	
	if(cudaStat != cudaSuccess || cudaptr == NULL){
	  whiteice::logging.error("vertex ctor: cudaMallocManaged() failed.");
	  throw CUDAException("CUBLAS memory allocation failure.");
	}

	// no memory initialization!!
	
	this->data = (T*)cudaptr;
      }
      
#else
      
      if(i > 0){
#ifdef BLAS_MEMALIGN
	// electric fence don't know about posix_memalign()
	posix_memalign((void**)&(this->data),
		       (8/whiteice::gcd<unsigned int>(8,sizeof(void*)))*sizeof(void*),
		       i*sizeof(T));
#else
	this->data = (T*)malloc(i*sizeof(T));
#endif
	
	if(this->data == 0)
	  throw std::bad_alloc();
	
	
	memset(this->data, 0, i*sizeof(T));
      }

#endif
      
      
      this->dataSize = i;
    }
    
    
    // vertex ctor - makes copy of v
    template <typename T>
    vertex<T>::vertex(const vertex<T>& v)
    {
      this->compressor = 0;
      this->dataSize = 0;
      this->data = 0;
      
      if(v.compressor != 0){
	assert(0);
	throw illegal_operation("vertex ctor: to be copied vertex is compressed");
      }

#ifdef CUBLAS

      if(v.data){
	cudaError_t cudaErr;
	cublasStatus_t cudaStat;
	void* cudaptr = NULL;
	cudaErr = cudaMallocManaged(&cudaptr, v.dataSize*sizeof(T));
	
	if(cudaErr != cudaSuccess || cudaptr == NULL){
	  whiteice::logging.error("vertex ctor: cudaMallocManaged() failed.");
	  throw CUDAException("CUBLAS memory allocation failure.");
	}
	
	// cuda copy memory
	if(typeid(T) == typeid(blas_real<float>)){
	  cudaStat = cublasScopy(cublas_handle, v.dataSize,
				 (const float*)v.data, 1, (float*)cudaptr, 1);

	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex ctor: cublasScopy() failed.");
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cublasScopy() failed.");
	  }
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  cudaStat = cublasDcopy(cublas_handle, v.dataSize,
				 (const double*)v.data, 1, (double*)cudaptr, 1);

	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex ctor: cublasDcopy() failed.");
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cublasDcopy() failed.");
	  }
	}
	else if(typeid(T) == typeid(blas_complex<float>)){
	  cudaStat = cublasCcopy(cublas_handle, v.dataSize,
				 (const cuComplex*)v.data, 1, (cuComplex*)cudaptr, 1);

	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex ctor: cublasCcopy() failed.");
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cublasCcopy() failed.");
	  }
	}
	else if(typeid(T) == typeid(blas_complex<double>)){
	  cudaStat = cublasZcopy(cublas_handle, v.dataSize,
				 (const cuDoubleComplex*)v.data, 1,
				 (cuDoubleComplex*)cudaptr, 1);
	  
	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex ctor: cublasZcopy() failed.");
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cublasZcopy() failed.");
	  }
	}
	else{
	  // generic memcopy [assumes type T does not allocated memory dynamically]
	  auto e = cudaMemcpy(cudaptr, v.data, v.dataSize*sizeof(T),
			      cudaMemcpyDeviceToDevice);

	  gpu_sync();

	  if(e != cudaSuccess){
	    whiteice::logging.error("vertex ctor: cudaMemcpy() failed.");
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cudaMemcpy() failed.");
	  }
	  
	}

	// memory initialization successful
	this->data = (T*)cudaptr;
      }

#else
      
      if(v.data){
#ifdef BLAS_MEMALIGN
	// electric fence don't know about posix_memalign()
	posix_memalign((void**)&(this->data),
		       (8/whiteice::gcd<unsigned int>(8,sizeof(void*)))*sizeof(void*),
		       v.dataSize*sizeof(T));
#else
	this->data = (T*)malloc(v.dataSize*sizeof(T));
#endif
	
	if(this->data == 0)
	  throw std::bad_alloc();
	
	
	if(typeid(T) == typeid(blas_real<float>)){
	  cblas_scopy(v.dataSize,
		      (const float*)v.data, 1,
		      (float*)(this->data), 1);
	}
	else if(typeid(T) == typeid(blas_complex<float>)){
	  cblas_ccopy(v.dataSize,
		      (const float*)v.data, 1,
		      (float*)(this->data), 1);
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  cblas_dcopy(v.dataSize,
		      (const double*)v.data, 1,
		      (double*)(this->data), 1);
	}
	else if(typeid(T) == typeid(blas_complex<double>)){
	  cblas_zcopy(v.dataSize,
		      (const double*)v.data, 1,
		      (double*)(this->data), 1);
	}
	else{ // generic memcpy
	  memcpy(this->data, v.data, v.dataSize*sizeof(T));
	}	  	
      }

#endif
      
      this->dataSize = v.dataSize;
    }
    
    
    // makes direct copy of temporal value
    template <typename T>
    vertex<T>::vertex(vertex<T>&& t)
    {
      this->data = NULL;
      this->dataSize = 0;
      this->compressor = NULL;
      
      std::swap(this->data, t.data);
      std::swap(this->dataSize, t.dataSize);
      std::swap(this->compressor, t.compressor);
      
    }
    
    
    // vertex ctor - makes copy of v
    template <typename T>
    vertex<T>::vertex(const std::vector<T>& v)
    {
      this->compressor = 0;
      this->dataSize = 0;
      this->data = 0;

#ifdef CUBLAS

      if(v.size() > 0){
	cudaError_t cudaErr;
	cublasStatus_t cudaStat;
	void* cudaptr = NULL;
	cudaErr = cudaMallocManaged(&cudaptr, v.size()*sizeof(T));
	
	if(cudaErr != cudaSuccess || cudaptr == NULL){
	  whiteice::logging.error("vertex ctor: cudaMallocManaged() failed.");
	  throw CUDAException("CUBLAS memory allocation failure.");
	}
	
	// cuda copy memory
	if(typeid(T) == typeid(blas_real<float>)){
	  cudaStat = cublasScopy(cublas_handle, v.size(),
				 (const float*)v.data(), 1, (float*)cudaptr, 1);
	  
	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex ctor: cublasScopy() failed.");
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cublasScopy() failed.");
	  }
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  cudaStat = cublasDcopy(cublas_handle, v.size(),
				 (const double*)v.data(), 1, (double*)cudaptr, 1);

	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex ctor: cublasDcopy() failed.");
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cublasDcopy() failed.");
	  }
	}
	else if(typeid(T) == typeid(blas_complex<float>)){
	  cudaStat = cublasCcopy(cublas_handle, v.size(),
				 (const cuComplex*)v.data(), 1, (cuComplex*)cudaptr, 1);

	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex ctor: cublasCcopy() failed.");
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cublasCcopy() failed.");
	  }
	}
	else if(typeid(T) == typeid(blas_complex<double>)){
	  cudaStat = cublasZcopy(cublas_handle, v.size(),
				 (const cuDoubleComplex*)v.data(), 1,
				 (cuDoubleComplex*)cudaptr, 1);

	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex ctor: cublasZcopy() failed.");
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cublasZcopy() failed.");
	  }
	}
	else{
	  auto e = cudaMemcpy(cudaptr, v.data(), v.size()*sizeof(T),
			      cudaMemcpyHostToDevice);

	  gpu_sync();

	  if(e != cudaSuccess){
	    whiteice::logging.error("vertex ctor: cudaMemcpy() failed.");
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cudaMemcpy() failed.");
	  }
	  
	}

	// memory initialization successful
	this->data = (T*)cudaptr;
      }

#else
      
      if(v.size() > 0){
#ifdef BLAS_MEMALIGN
	posix_memalign((void**)&(this->data),
		       (8/whiteice::gcd<unsigned int>(8,sizeof(void*)))*sizeof(void*),
		       v.size()*sizeof(T));
#else
	this->data = (T*)malloc(v.size()*sizeof(T));
#endif
	
	if(this->data == 0)
	  throw std::bad_alloc();
	
	
	for(unsigned int i=0;i<v.size();i++)
	  (this->data)[i] = v[i];
      }

#endif
      
      this->dataSize = v.size();
    }
    
    
    // vertex dtor
    template <typename T>
    vertex<T>::~vertex()
    {
      if(this->compressor) delete (this->compressor);

#ifdef CUBLAS

      if(this->data){
	cudaFree(this->data);
      }

#else
      
      if(this->data) free(this->data);
      
#endif
    }
    
    /***************************************************/
    
    // returns vertex dimension/size
    template <typename T>
    unsigned int vertex<T>::size() const { return dataSize; }
    
    
    // sets vertex dimension/size, fills new dimensios with zero
    template <typename T>
    unsigned int vertex<T>::resize(unsigned int d) 
    {

#ifdef CUBLAS

      if(d == 0){
	cudaFree(data);
	data = NULL;
	dataSize = 0;
	return 0;
      }
      else if(d == dataSize){
	return dataSize; // nothing to do
      }
      else{
	// there is no realloc() in CUDA ??? so I allocate 
	// new block and copy

	cudaError_t cudaErr;
	cublasStatus_t cudaStat;

	void* cudaptr = NULL;
	cudaErr = cudaMallocManaged(&cudaptr, d*sizeof(T));

	unsigned int copylen = dataSize;
	if(d < dataSize) copylen = d;
	
	if(cudaErr != cudaSuccess || cudaptr == NULL){
	  whiteice::logging.error("vertex::resize(): cudaMallocManaged() failed.");
	  throw CUDAException("CUBLAS memory allocation failure.");
	}
	
	// cuda copy memory
	if(typeid(T) == typeid(blas_real<float>)){
	  cudaStat = cublasScopy(cublas_handle, copylen,
				 (const float*)this->data, 1, (float*)cudaptr, 1);
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){	    
	    whiteice::logging.error("vertex::resize(): cublasScopy() failed.");
	    gpu_sync();
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cublasScopy() failed.");
	  }
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  cudaStat = cublasDcopy(cublas_handle, copylen,
				 (const double*)this->data, 1, (double*)cudaptr, 1);
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::resize(): cublasDcopy() failed.");
	    gpu_sync();
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cublasDcopy() failed.");
	  }
	}
	else if(typeid(T) == typeid(blas_complex<float>)){
	  cudaStat = cublasCcopy(cublas_handle, copylen,
				 (const cuComplex*)this->data, 1, (cuComplex*)cudaptr, 1);
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::resize(): cublasCcopy() failed.");
	    gpu_sync();
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cublasCcopy() failed.");
	  }
	}
	else if(typeid(T) == typeid(blas_complex<double>)){
	  cudaStat = cublasZcopy(cublas_handle, copylen,
				 (const cuDoubleComplex*)this->data, 1,
				 (cuDoubleComplex*)cudaptr, 1);
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){	    
	    whiteice::logging.error("vertex::resize(): cublasZcopy() failed.");
	    gpu_sync();
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cublasZcopy() failed.");
	  }
	}
	else{
	  // generic memcopy [assumes type T does not allocate memory dynamically]
	  auto e = cudaMemcpy(cudaptr, data, copylen*sizeof(T),
			      cudaMemcpyDeviceToDevice);

	  if(e != cudaSuccess){
	    whiteice::logging.error("vertex::resize(): cudaMemcopy() failed.");
	    gpu_sync();
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cudaMemcpy() failed.");
	  }
	  
	}

	if(copylen < d){
	  unsigned char* bytes = (unsigned char*)cudaptr;
	  
	  auto err = cudaMemset(&(bytes[copylen*sizeof(T)]), 0,
				(d - copylen)*sizeof(T));

	  gpu_sync();

	  if(err != cudaSuccess){
	    whiteice::logging.error("vertex::resize(): cudaMemset() failed.");
	    cudaFree(cudaptr);
	    throw CUDAException("cudaMemset() failed.");
	  }
	}
	else{
	  gpu_sync();
	}

	// memory initialization successful
	if(this->data) cudaFree(this->data);
	this->data = (T*)cudaptr;
	this->dataSize = d;
      }
      
#else
      
      if(d == 0){
	if(data) free(data);
	data = 0;
	dataSize = 0;
	return 0;
      }
      else if(d == dataSize){
	return dataSize; // nothing to do
      }
      else{
	T* new_area = 0;
	
	if(data != 0){
	  new_area = (T*)realloc(data, sizeof(T)*d);
	  
	  if(new_area == 0)
	    return dataSize; // mem. alloc failure
	}
	else{
	  new_area = (T*)malloc(sizeof(T)*d);

	  if(new_area == 0)
	    return dataSize; // mem. alloc failure
	}
	
	data = new_area;
    
	// fills new memory area with zeros
	if(dataSize < d)
	  for(unsigned int s = dataSize;s<d;s++)
	    data[s] = T(0.0);
	
	dataSize = d;
      }

#endif
      
      return dataSize;
    }

    
    // returns length of vertex
    template <typename T>
    T vertex<T>::norm() const 
    {

#ifdef CUBLAS

      if(typeid(T) == typeid(blas_real<float>)){
	float result;
	cublasStatus_t s = cublasSnrm2(cublas_handle,
				       (int)dataSize,
				       (const float*)data, 1,
				       &result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::norm(): cublasSnrm2() failed.");
	  throw CUDAException("CUBLAS cublasSnrm2() failed.");
	}
	
	return T(result);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double result;

	cublasStatus_t s = cublasDnrm2(cublas_handle,
				       (int)dataSize,
				       (const double*)data, 1,
				       &result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::norm(): cublasDnrm2() failed.");
	  throw CUDAException("CUBLAS cublasDnrm2() failed.");
	}
	
	return T(result);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	float result;
	cublasStatus_t s = cublasScnrm2(cublas_handle,
					(int)dataSize,
					(const cuComplex*)data, 1,
					&result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::norm(): cublasScnrm2() failed.");
	  throw CUDAException("CUBLAS cublasScnrm2() failed.");
	}

	T rv = result;
	
	return rv;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	double result;
	cublasStatus_t s = cublasDznrm2(cublas_handle,
					(int)dataSize,
					(const cuDoubleComplex*)data, 1,
					&result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::norm(): cublasDznrm2() failed.");
	  throw CUDAException("CUBLAS cublasDznrm2() failed.");
	}

	T rv = result;
	
	return rv;
      }
      else{
	T len = T(0.0f);
	
	for(unsigned int i=0;i<dataSize;i++)
	  len += data[i]*whiteice::math::conj(data[i]);
	
	len = (T)whiteice::math::sqrt(whiteice::math::abs(len));
	return len;
      }
      
#else
      
      T len; // cblas_Xnrm2 optimizated functions
      
      if(typeid(T) == typeid(blas_real<float>)){
	len = (T)cblas_snrm2(dataSize, (float*)data, 1);
	
	return len;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	len = (T)cblas_scnrm2(dataSize, (float*)data, 1);
	
	return len;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	len = (T)cblas_dnrm2(dataSize, (double*)data, 1);
	
	return len;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	len = (T)cblas_dznrm2(dataSize, (double*)data, 1);
	
	return len;
      }
      else{ // generic length calculation
	len = T(0.0f);
	
	for(unsigned int i=0;i<dataSize;i++)
	  len += data[i]*whiteice::math::conj(data[i]);
	
	len = (T)whiteice::math::sqrt(whiteice::math::abs(len));
	return len;
      }

#endif
      
    }
    
    
    // calculates partial norm for vertex(i:j)
    template <typename T>
    T vertex<T>::norm(unsigned int i, unsigned int j) const 
    {
      if(i >= j || i > dataSize || j > dataSize)
	return T(0.0f);
      
#ifdef CUBLAS

      const unsigned int L = j-i;

      if(typeid(T) == typeid(blas_real<float>)){
	float result;
	cublasStatus_t s = cublasSnrm2(cublas_handle,
				       (int)L,
				       (const float*)&(data[i]), 1,
				       &result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::norm(): cublasSnrm2() failed.");
	  throw CUDAException("CUBLAS cublasSnrm2() failed.");
	}
	
	return T(result);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double result;

	cublasStatus_t s = cublasDnrm2(cublas_handle,
				       (int)L,
				       (const double*)&(data[i]), 1,
				       &result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::norm(): cublasDnrm2() failed.");
	  throw CUDAException("CUBLAS cublasDnrm2() failed.");
	}
	
	return T(result);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	float result;
	cublasStatus_t s = cublasScnrm2(cublas_handle,
					(int)L,
					(const cuComplex*)&(data[i]), 1,
					&result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::norm(): cublasScnrm2() failed.");
	  throw CUDAException("CUBLAS cublasScnrm2() failed.");
	}

	T rv = result;
	
	return rv;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	double result;
	cublasStatus_t s = cublasDznrm2(cublas_handle,
					(int)L,
					(const cuDoubleComplex*)&(data[i]), 1,
					&result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::norm(): cublasDznrm2() failed.");
	  throw CUDAException("CUBLAS cublasDznrm2() failed.");
	}

	T rv = result;
	
	return rv;
      }
      else{
	T len = T(0.0f);
	
	for(unsigned int k=i;k<j;k++)
	  len += data[k]*whiteice::math::conj(data[k]);
	
	len = (T)whiteice::math::sqrt(whiteice::math::abs(len));
	return len;
      }
	    
      
#else
      T len = T(0.0f); // cblas_Xnrm2 optimizated functions
      
      if(typeid(T) == typeid(blas_real<float>)){
	len = (T)cblas_snrm2(j - i,(float*)(&(data[i])), 1);
	
	return len;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	len = (T)cblas_scnrm2(j - i,(float*)(&(data[i])), 1);
	
	return len;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	len = (T)cblas_dnrm2(j - i,(double*)(&(data[i])), 1);
	
	return len;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	len = (T)cblas_dznrm2(j - i,(double*)(&(data[i])), 1);
	
	return len;
      }
      else{ // generic length calculation
	
	for(unsigned int i=0;i<dataSize;i++)
	  len += (T)(data[i]*data[i]);
	
	len = (T)sqrt(len);
	return len;
      }
#endif
    }
    
    
    // sets length to one, zero length -> returns false
    template <typename T>
    bool vertex<T>::normalize() 
    {
      T len = norm();
      if(len == T(0.0f)) return false;
      len = T(1.0f) / len;
      
#ifdef CUBLAS

      if(typeid(T) == typeid(blas_real<float>)){
	float alpha;
	whiteice::math::convert(alpha, len);

	cublasStatus_t s = cublasSscal(cublas_handle,
				       dataSize,
				       (const float*)&alpha,
				       (float*)data, 1);
	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::normalize(): cublasSscal() failed.");
	  throw CUDAException("CUBLAS cublasSscal() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double alpha;
	whiteice::math::convert(alpha, len);

	cublasStatus_t s = cublasDscal(cublas_handle,
				       dataSize,
				       (const double*)&alpha,
				       (double*)data, 1);
	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::normalize(): cublasDscal() failed.");
	  throw CUDAException("CUBLAS cublasDscal() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	float alpha;
	whiteice::math::convert(alpha, len);

	cublasStatus_t s = cublasCsscal(cublas_handle,
					dataSize,
					(const float*)&alpha,
					(cuComplex*)data, 1);
	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::normalize(): cublasCsscal() failed.");
	  throw CUDAException("CUBLAS cublasCsscal() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	double alpha;
	whiteice::math::convert(alpha, len);

	cublasStatus_t s = cublasZdscal(cublas_handle,
					dataSize,
					(const double*)&alpha,
					(cuDoubleComplex*)data, 1);
	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::normalize(): cublasZdscal() failed.");
	  throw CUDAException("CUBLAS cublasZdscal() failed.");
	}

	return true;
      }
      else{

	for(unsigned int i=0;i<dataSize;i++)
	  data[i] *= len;

	return true;
      }
      
#else
      // uses optimized cblas_Xscal() routines
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sscal(dataSize, *((float*)&len), (float*)data, 1);
	return true;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_cscal(dataSize, (const float*)&len, (float*)data, 1);
	return true;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dscal(dataSize, *((double*)&len), (double*)data, 1);
	return true;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zscal(dataSize, (const double*)&len, (double*)data, 1);
	return true;
      }    
      else{
	
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] *= len;
	
	return true;
      }
#endif
      
    }
    
    
    // vertex = 0;
    template <typename T>
    void vertex<T>::zero() 
    {
      if(dataSize <= 0) return;
      
#ifdef CUBLAS
      // memset is maybe fastest way to do this??
      
      if(typeid(T) == typeid(blas_real<float>) ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>) ||
	 typeid(T) == typeid(blas_complex<double>))
	{
	  auto err = cudaMemset(data, 0, dataSize*sizeof(T));

	  gpu_sync();

	  if(err != cudaSuccess){
	    whiteice::logging.error("vertex::zero(): cudaMemset() failed.");
	    throw CUDAException("cudaMemset() failed.");
	  }
	  
      }
      else{
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] = T(0.0f);
      }
      
#else
      if(typeid(T) == typeid(blas_real<float>) ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>) ||
	 typeid(T) == typeid(blas_complex<double>))
      {
	// all bits = 0, is zero number representation
	memset(data, 0, dataSize*sizeof(T));
      }
      else{
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] = T(0.0f);
      }
#endif
    }


    template <typename T>
    void vertex<T>::hermite() 
    {
      if(dataSize <= 0) return;
      
#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<dataSize;i++)
	data[i] = whiteice::math::conj(data[i]);
      
    }
    
    
    // calculates sum of vertexes
    template <typename T>
    vertex<T> vertex<T>::operator+(const vertex<T>& v) const
      
    {
      if(v.dataSize != dataSize){
	printf("ERROR: illegal operation: vector operator+ failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);

	whiteice::logging.error("vertex::operator+(): vertex dimension mismatch.");
	assert(0);
	throw illegal_operation("vector op: vector dim. mismatch");
      }

#ifdef CUBLAS
      
      // copy of this vector
      vertex<T> r(*this);
      
      // no direct BLAS speedups (alpha = 1, alpha*x + y , faster than manual?)
      
      r += v; // operator += uses BLAS

      return r;

#else
      // copy of this vector
      vertex<T> r(*this);
      
      // no direct BLAS speedups (alpha = 1, alpha*x + y , faster than manual?)
      
      r += v; // operator += uses BLAS

      return r;
#endif
    }

    
    
    // substracts two vertexes
    template <typename T>
    vertex<T> vertex<T>::operator-(const vertex<T>& v) const
      
    {
      if(v.dataSize != dataSize){
	printf("ERROR: illegal operation: vector operator- failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);
	whiteice::logging.error("vertex::operator-(): vertex dimension mismatch.");
	assert(0);
	throw illegal_operation("vector op: vector dim. mismatch");
      }

#ifdef CUBLAS
      // copy of this vector: no direct BLAS speedups
      vertex<T> r(*this);
      
      r -= v; // uses BLAS
      
      return r;
#else
      // copy of this vector: no direct BLAS speedups
      vertex<T> r(*this);
      
      r -= v; // uses BLAS

      return r;

#if 0
      // cblas_Xaxpy() (alpha = -1) -> r = x - y      
      if(typeid(T) == typeid(blas_real<float>)){
	float alpha = -1;
	
	cblas_saxpy(dataSize, alpha, (float*)v.data, 1,
		    (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	T alpha; alpha = -1;
	
	cblas_caxpy(dataSize, (const float*)&alpha, (float*)v.data, 1,
		    (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double alpha = -1;
	
	cblas_daxpy(dataSize, alpha, (double*)v.data, 1,
		    (double*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	T alpha; alpha = -1;
	
	cblas_zaxpy(dataSize, (const double*)&alpha, (double*)v.data, 1,
		    (double*)r.data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<v.dataSize;i++){
	  r.data[i] -= v.data[i];
	}
      }
#endif
      
      return r;
      
#endif
      
    }
    
    
    // calculates dot product - returns 1-dimension vertex
    // if input either of the vertexes is dim=1 vertex this
    // is scalar product
    // calculates z = (*this)^t * v (no complex conjugate)
    template <typename T>
    vertex<T> vertex<T>::operator*(const vertex<T>& v) const
      
    {
      if(!(dataSize == v.dataSize || (dataSize == 1 || v.dataSize == 1))){
	printf("ERROR: illegal operation: vector operator* failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);
	whiteice::logging.error("vertex::operator*(): vertex dimension mismatch.");
	assert(0);
	throw illegal_operation("vector op: vector dim. mismatch");
      }

#ifdef CUBLAS

      vertex<T> r(1);

      if(dataSize != 1 && v.dataSize != 1){
	
	if(typeid(T) == typeid(blas_real<float>)){
	  T result;
	  
	  cublasStatus_t s = cublasSdot(cublas_handle, (int)dataSize,
					(const float*)(this->data), 1,
					(const float*)(v.data), 1,
					(float*)&result);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator*(): cublasSdot() failed.");
	    throw CUDAException("CUDA cublasSdot() failed.");
	  }

	  r[0] = result;
	  return r;
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  T result;
	  
	  cublasStatus_t s = cublasDdot(cublas_handle, (int)dataSize,
					(const double*)(this->data), 1,
					(const double*)(v.data), 1,
					(double*)&result);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator*(): cublasDdot() failed.");
	    throw CUDAException("CUDA cublasDdot() failed.");
	  }

	  r[0] = result;
	  return r;
	}
	else if(typeid(T) == typeid(blas_complex<float>)){
	  T result;

	  cublasStatus_t s = cublasCdotu(cublas_handle, (int)dataSize,
					 (const cuComplex*)(this->data), 1,
					 (const cuComplex*)(v.data), 1,
					 (cuComplex*)&result);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator*(): cublasCdotu() failed.");
	    throw CUDAException("CUDA cublasCdotu() failed.");
	  }

	  r[0] = result;
	  return r;
	}
	else if(typeid(T) == typeid(blas_complex<double>)){
	  T result;

	  cublasStatus_t s = cublasZdotu
	    (cublas_handle, (int)dataSize,
	     (const cuDoubleComplex*)(this->data), 1,
	     (const cuDoubleComplex*)(v.data), 1,
	     (cuDoubleComplex*)&result);
	  
	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator*(): cublasZdotu() failed.");
	    throw CUDAException("CUDA cublasZdotu() failed.");
	  }

	  r[0] = result;
	  return r;
	}
	else{
	  r[0] = T(0.0f);
	  
	  for(unsigned int i=0;i<v.dataSize;i++)
	    r.data[0] += data[i]*v.data[i];
	  
	  return r;
	}
	
      }
      else{ // dataSize == 1 || v.dataSize == 1

	// scalar product
	
	if(dataSize == 1){
	  r = data[0] * v;
	  return r;
	}
	else{ // v.dataSize == 1
	  r = v.data[0] * (*this);
	  return r;
	}
		
      }
      
      
#else      
      
      // uses BLAS
      
      if(dataSize != 1 && v.dataSize != 1){
	
	vertex<T> r(1);
	r.resize(1);
	r[0] = T(0.0f);
	
	if(typeid(T) == typeid(blas_real<float>)){
	  *((T*)&(r.data[0])) = T(cblas_sdot(dataSize, (float*)data, 1,
					     (float*)v.data, 1));
	  return r;
	}
	else if(typeid(T) == typeid(blas_complex<float>)){
#ifdef OPENBLAS
	  cblas_cdotu_sub(dataSize, (float*)data, 1,
			  //  (float*)v.data, 1, (openblas_complex_float*)&(r.data[0]));
			  (float*)v.data, 1, (openblas_complex_float*)&(r.data[0]));
#else
	  cblas_cdotu_sub(dataSize, (float*)data, 1,
			  //  (float*)v.data, 1, (openblas_complex_float*)&(r.data[0]));
			  (float*)v.data, 1, (float*)&(r.data[0]));	  
#endif
	  return r;
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  *((T*)&(r.data[0])) = T(cblas_ddot(dataSize, (double*)data, 1,
					     (double*)v.data, 1));
	  return r;
	}
	else if(typeid(T) == typeid(blas_complex<double>)){
#ifdef OPENBLAS
	  cblas_zdotu_sub(dataSize, (double*)data, 1,
			  // (double*)v.data, 1, (openblas_complex_double*)&(r.data[0]));
			  (double*)v.data, 1, (openblas_complex_double*)&(r.data[0]));
#else
	  cblas_zdotu_sub(dataSize, (double*)data, 1,
			  // (double*)v.data, 1, (openblas_complex_double*)&(r.data[0]));
			  (double*)v.data, 1, (double*)&(r.data[0]));
#endif
	  return r;
	}
	else{ // "normal implementation"
	  for(unsigned int i=0;i<v.dataSize;i++)
	    r.data[0] += data[i]*v.data[i];
	  
	  return r;
	}
	
      }
      else{ // scalar product
	
	if(dataSize == 1){
	  vertex<T> r = v * data[0];
	  return r;
	}
	else{ // v.dataSize == 1
	  vertex<T> r = (*this) * v.data[0];
	  return r;
	}
      }
      
#endif
    }
    
    // no divide operation
    template <typename T>
    vertex<T> vertex<T>::operator/(const vertex<T>& v) const {
      if(v.size() == 1){
	return this->operator/(v[0]);
      }
      else{
	whiteice::logging.error("vertex::operator/(): division not defined for vectors.");
	assert(0);
	throw illegal_operation("vertex(): '/'-operator not available");
      }
    }
    
    // no "!" operation
    template <typename T>
    vertex<T> vertex<T>::operator!() const {
      assert(0);
      throw illegal_operation("vertex(): '!'-operation not available");
    }
    
    // changes vertex sign
    template <typename T>
    vertex<T> vertex<T>::operator-() const
      
    {
#ifdef CUBLAS

      vertex<T> r(*this);

      if(typeid(T) == typeid(blas_real<float>)){
	float alpha = -1.0f;

	cublasStatus_t s = cublasSscal(cublas_handle,
				       dataSize,
				       (const float*)&alpha,
				       (float*)r.data, 1);
	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator-(): cublasSscal() failed.");
	  throw CUDAException("CUBLAS cublasSscal() failed.");
	}

	return r;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double alpha = -1.0;

	cublasStatus_t s = cublasDscal(cublas_handle,
				       dataSize,
				       (const double*)&alpha,
				       (double*)r.data, 1);
	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator-(): cublasDscal() failed.");
	  throw CUDAException("CUBLAS cublasDscal() failed.");
	}

	return r;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	float alpha = -1.0f;
	
	cublasStatus_t s = cublasCsscal(cublas_handle,
					dataSize,
					(const float*)&alpha,
					(cuComplex*)r.data, 1);
	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator-(): cublasCsscal() failed.");
	  throw CUDAException("CUBLAS cublasCsscal() failed.");
	}

	return r;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	double alpha = -1.0;
	
	cublasStatus_t s = cublasZdscal(cublas_handle,
					dataSize,
					(const double*)&alpha,
					(cuDoubleComplex*)r.data, 1);
	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator-(): cublasZdscal() failed.");
	  throw CUDAException("CUBLAS cublasZdscal() failed.");
	}

	return r;
      }
      else{
	const T alpha = -1.0f;

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<dataSize;i++)
	  r.data[i] *= alpha;

	return r;
      }
      
#else
      vertex<T> r(*this);
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sscal(dataSize, -1.0f, (float*)r.data, 1);
	return r;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_csscal(dataSize, -1.0f, (float*)r.data, 1);
	return r;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dscal(dataSize, -1.0, (double*)r.data, 1);
	return r;      
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zdscal(dataSize, -1.0, (double*)r.data, 1);
	return r;
      }    
      else{
	for(unsigned int i=0;i<dataSize;i++)
	  r.data[i] = -data[i];
	
	return r;
      }
#endif
      
    }
    
    // calculates cross product
    template <typename T>
    vertex<T> vertex<T>::operator^(const vertex<T>& v) const
      
    {
      if(v.dataSize != 3 || this->dataSize != 3){
	whiteice::logging.error("vertex::operator^(): input vector dimension != 3.");
	assert(0);
	throw illegal_operation("crossproduct: vector dimension != 3");
      }
      
      // *NO* CBLAS USED
      
      vertex<T> r(3);
      
      r[0] = data[1]*v.data[2] - data[2]*v.data[1];
      r[1] = data[2]*v.data[0] - data[0]*v.data[2];
      r[2] = data[0]*v.data[1] - data[1]*v.data[0];
      
      return r;    
    }
    
    /***************************************************/
    
    // adds vertexes
    template <typename T>
    vertex<T>& vertex<T>::operator+=(const vertex<T>& v)
      
    {
      if(v.dataSize != dataSize){
	printf("ERROR: illegal operation: vector operator+= failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);

	whiteice::logging.error("vertex::operator+=(): vector dimension mismatch.");
	assert(0);
	throw illegal_operation("vector op: vector dim. mismatch");
      }

#ifdef CUBLAS

      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = +1.0f;

	cublasStatus_t s = cublasSaxpy(cublas_handle, (int)dataSize,
				       (const float*)&alpha,
				       (const float*)v.data, 1,
				       (float*)data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator+=(): cublaSaxpy() failed.");
	  throw CUDAException("CUBLAS cublasSaxpy() failed.");
	}
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = +1.0;
	
	cublasStatus_t s = cublasDaxpy(cublas_handle, (int)dataSize,
				       (const double*)&alpha,
				       (const double*)v.data, 1,
				       (double*)data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator+=(): cublaDaxpy() failed.");
	  throw CUDAException("CUBLAS cublasDaxpy() failed.");
	}
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = 1.0f;
	
	cublasStatus_t s = cublasCaxpy(cublas_handle, (int)dataSize,
				       (cuComplex*)&alpha,
				       (const cuComplex*)v.data, 1,
				       (cuComplex*)data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator+=(): cublaCaxpy() failed.");
	  throw CUDAException("CUBLAS cublasCaxpy() failed.");
	}
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = 1.0;
	
	cublasStatus_t s = cublasZaxpy(cublas_handle, (int)dataSize,
				       (cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)v.data, 1,
				       (cuDoubleComplex*)data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator+=(): cublaZaxpy() failed.");
	  throw CUDAException("CUBLAS cublasZaxpy() failed.");
	}
      }
      else{
	// *NO* CBLAS

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] += v.data[i];
      }
      
#else
      
      if(typeid(T) == typeid(blas_real<float>)){
        float alpha = +1.0f;
	
	cblas_saxpy(dataSize, alpha, (float*)v.data, 1, (float*)(this->data), 1);
	
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double alpha = +1.0;

	cblas_daxpy(dataSize, alpha, (double*)v.data, 1, (double*)(this->data), 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	blas_complex<float> alpha;
	alpha = +1.0f;

	cblas_caxpy(dataSize, (float*)&alpha, (float*)v.data, 1, (float*)(this->data), 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	blas_complex<double> alpha;
	alpha = +1.0;

	cblas_zaxpy(dataSize, (double*)&alpha, (double*)v.data, 1, (double*)(this->data), 1);
      }
      else{
	// *NO* CBLAS
	
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] += v.data[i];
      }

#endif
      
      
      return *this;
    }
    
    // subtracts vertexes
    template <typename T>
    vertex<T>& vertex<T>::operator-=(const vertex<T>& v)
      
    {
      if(dataSize != v.dataSize){
	printf("ERROR: illegal operation: vector operator-= failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);

	whiteice::logging.error("vertex::operator-=(): vector dimension mismatch.");
	assert(0);
	throw illegal_operation("vector op: vector dim. mismatch");
      }

#ifdef CUBLAS

      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = -1.0f;

	cublasStatus_t s = cublasSaxpy(cublas_handle, (int)dataSize,
				       (const float*)&alpha,
				       (const float*)v.data, 1,
				       (float*)data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator-=(): cublasSaxpy() failed.");
	  throw CUDAException("CUBLAS cublasSaxpy() failed.");
	}
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = -1.0;
	
	cublasStatus_t s = cublasDaxpy(cublas_handle, (int)dataSize,
				       (const double*)&alpha,
				       (const double*)v.data, 1,
				       (double*)data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator-=(): cublasDaxpy() failed.");
	  throw CUDAException("CUBLAS cublasDaxpy() failed.");
	}
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = -1.0f;
	
	cublasStatus_t s = cublasCaxpy(cublas_handle, (int)dataSize,
				       (cuComplex*)&alpha,
				       (const cuComplex*)v.data, 1,
				       (cuComplex*)data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator-=(): cublasCaxpy() failed.");
	  throw CUDAException("CUBLAS cublasCaxpy() failed.");
	}
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = -1.0;
	
	cublasStatus_t s = cublasZaxpy(cublas_handle, (int)dataSize,
				       (cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)v.data, 1,
				       (cuDoubleComplex*)data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator-=(): cublasZaxpy() failed.");
	  throw CUDAException("CUBLAS cublasZaxpy() failed.");
	}
      }
      else{
	// *NO* CBLAS

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] -= v.data[i];
      }
      

#else
      
      if(typeid(T) == typeid(blas_real<float>)){
	float alpha = -1.0f;
	
	cblas_saxpy(dataSize, alpha, (float*)v.data, 1, (float*)(this->data), 1);
	
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double alpha = -1.0;

	cblas_daxpy(dataSize, alpha, (double*)v.data, 1, (double*)(this->data), 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	blas_complex<float> alpha;
	alpha = -1.0f;

	cblas_caxpy(dataSize, (float*)&alpha, (float*)v.data, 1, (float*)(this->data), 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	blas_complex<double> alpha;
	alpha = -1.0;

	cblas_zaxpy(dataSize, (double*)&alpha, (double*)v.data, 1, (double*)(this->data), 1);
      }     
      else{
	// *NO* CBLAS
	
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] -= v.data[i];
      }

#endif
      
      return *this;
    }

    
    // calculates dot product (no conjugate transpose!)
    template <typename T>
    vertex<T>& vertex<T>::operator*=(const vertex<T>& v)
      
    {
      if(v.dataSize != dataSize && v.dataSize != 1 && dataSize != 1){
	printf("ERROR: illegal operation: vector operator*= failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);
	whiteice::logging.error("vertex::operator*=(): vector dimension mismatch.");
	assert(0);
	throw illegal_operation("vector op: vector dim. mismatch");
      }

#ifdef CUBLAS
      
      vertex<T> r(1);

      if(dataSize != 1 && v.dataSize != 1){
	
	if(typeid(T) == typeid(blas_real<float>)){
	  T result;
	  
	  cublasStatus_t s = cublasSdot(cublas_handle, (int)dataSize,
					(const float*)(this->data), 1,
					(const float*)(v.data), 1,
					(float*)&result);
	  
	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator*=(): cublasSdot() failed.");
	    throw CUDAException("CUDA cublasSdot() failed.");
	  }

	  r[0] = result;

	  (*this) = r;

	  gpu_sync();
	  
	  return (*this);
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  T result;
	  
	  cublasStatus_t s = cublasDdot(cublas_handle, (int)dataSize,
					(const double*)(this->data), 1,
					(const double*)(v.data), 1,
					(double*)&result);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator*=(): cublasDdot() failed.");
	    throw CUDAException("CUDA cublasDdot() failed.");
	  }

	  r[0] = result;

	  (*this) = r;

	  gpu_sync();
	  
	  return (*this);
	}
	else if(typeid(T) == typeid(blas_complex<float>)){
	  T result;

	  cublasStatus_t s = cublasCdotu(cublas_handle, (int)dataSize,
					 (const cuComplex*)(this->data), 1,
					 (const cuComplex*)(v.data), 1,
					 (cuComplex*)&result);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator*=(): cublasCdotu() failed.");
	    throw CUDAException("CUDA cublasCdotu() failed.");
	  }

	  r[0] = result;

	  (*this) = r;

	  gpu_sync();
	  
	  return (*this);
	}
	else if(typeid(T) == typeid(blas_complex<double>)){
	  T result;

	  cublasStatus_t s = cublasZdotu
	    (cublas_handle, (int)dataSize,
	     (const cuDoubleComplex*)(this->data), 1,
	     (const cuDoubleComplex*)(v.data), 1,
	     (cuDoubleComplex*)&result);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator*=(): cublasZdotu() failed.");
	    throw CUDAException("CUDA cublasZdotu() failed.");
	  }

	  r[0] = result;

	  (*this) = r;

	  gpu_sync();
	  
	  return (*this);
	}
	else{
	  r[0] = T(0.0f);

#pragma omp parallel
	  {
	    T rvalue = T(0.0f);

#pragma omp for schedule(auto) nowait
	    for(unsigned int i=0;i<v.dataSize;i++)
	      rvalue += data[i]*v.data[i];

#pragma omp critical
	    {
	      r.data[0] += rvalue;
	    }
	  }

	  (*this) = r;
	  return (*this);
	}
	
      }
      else{ // dataSize == 1 || v.dataSize == 1

	// scalar product
	
	if(dataSize == 1){
	  r = data[0] * v;

	  (*this) = r;
	  return (*this);
	}
	else{ // v.dataSize == 1
	  r = v.data[0] * (*this);

	  (*this) = r;
	  return (*this);
	}
		
      }

      
#else
      // calculates r = a^t * v where a vector is this
      
      vertex<T> r(1);

      if(typeid(T) == typeid(blas_real<float>)){
	
	r[0] = T(cblas_sdot(dataSize, (float*)data, 1, (float*)v.data, 1));
	
	*this = r;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	r[0] = T(cblas_ddot(dataSize, (double*)data, 1, (double*)v.data, 1));
	
	*this = r;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	T value;
	// blas_complex<float> value;

#ifdef OPENBLAS
	cblas_cdotu_sub(dataSize, (float*)data, 1, (float*)v.data, 1, (openblas_complex_float*)&value);
#else
	cblas_cdotu_sub(dataSize, (float*)data, 1, (float*)v.data, 1, &value);
#endif

	//whiteice::math::convert(r[0], value);
	r[0] = value;
	*this = r;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	//blas_complex<double> value;
	T value;

#ifdef OPENBLAS
	cblas_zdotu_sub(dataSize, (double*)data, 1, (double*)v.data, 1, (openblas_complex_double*)&value);
#else
	cblas_zdotu_sub(dataSize, (double*)data, 1, (double*)v.data, 1, &value);
#endif

	//whiteice::math::convert(r[0], value);
	r[0] = value;
	*this = r;

      }
      else{
      	// *NO* CBLAS
	
	r[0] = T(0.0f);

#pragma omp parallel
	{
	  T rvalue = T(0.0f);

#pragma omp for nowait schedule(auto)
	  for(unsigned int i=0;i<v.dataSize;i++)
	    rvalue += data[i]*v.data[i];

#pragma omp critical
	  {
	    r[0] += rvalue;
	  }
	}
      }

      *this = r;

      return *this;
#endif
      
    }
    
    // dividing not available
    template <typename T>
    vertex<T>& vertex<T>::operator/=(const vertex<T>& v)
    {
      if(v.dataSize == 1){
	return ((*this) /= v[0]);
      }
      else{
	whiteice::logging.error("vertex::operator/=(): division not defined for vectors.");
	assert(0);
	throw illegal_operation("vertex(): '/='-operator not available");
      }
    }
    
    // assigns given vertex value to this vertex
    template <typename T>
    vertex<T>& vertex<T>::operator=(const vertex<T>& v)
      
    {
      if(v.compressor != 0 || this->compressor != 0){
	whiteice::logging.error("vertex::operator=(): compressed vector data.");
	assert(0);
	throw illegal_operation("vertex '='-operator: compressed vertex data");
      }
      
      if(v.data == this->data) // self-assignment
	return (*this);

      if(v.dataSize != this->dataSize)
	if(this->resize(v.dataSize) != v.dataSize){
	  whiteice::logging.error("vertex::operator=(): resize() failed.");
	  assert(0);
	  throw illegal_operation("vertex '='-operator: out of memory");
	}

#ifdef CUBLAS

      if(v.data){
	cublasStatus_t cudaStat;
	
	// cuda copy memory
	if(typeid(T) == typeid(blas_real<float>)){
	  cudaStat = cublasScopy(cublas_handle, v.dataSize,
				 (const float*)v.data, 1, (float*)data, 1);
	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator=(): cublasScopy() failed.");
	    throw CUDAException("CUBLAS cublasScopy() failed.");
	  }

	  return (*this);
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  cudaStat = cublasDcopy(cublas_handle, v.dataSize,
				 (const double*)v.data, 1, (double*)data, 1);
	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator=(): cublasDcopy() failed.");
	    throw CUDAException("CUBLAS cublasDcopy() failed.");
	  }

	  return (*this);
	}
	else if(typeid(T) == typeid(blas_complex<float>)){
	  cudaStat = cublasCcopy(cublas_handle, v.dataSize,
				 (const cuComplex*)v.data, 1,
				 (cuComplex*)data, 1);
	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator=(): cublasCcopy() failed.");
	    throw CUDAException("CUBLAS cublasCcopy() failed.");
	  }

	  return (*this);
	}
	else if(typeid(T) == typeid(blas_complex<double>)){
	  cudaStat = cublasZcopy(cublas_handle, v.dataSize,
				 (const cuDoubleComplex*)v.data, 1,
				 (cuDoubleComplex*)data, 1);
	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator=(): cublasZcopy() failed.");
	    throw CUDAException("CUBLAS cublasZcopy() failed.");
	  }

	  return (*this);
	}
	else{
	  // generic memcopy [assumes type T does not allocated memory dynamically]
	  auto e = cudaMemcpy(data, v.data, v.dataSize*sizeof(T),
			      cudaMemcpyDeviceToDevice);
	  gpu_sync();

	  if(e != cudaSuccess){
	    whiteice::logging.error("vertex::operator=(): cudaMemcpy() failed.");
	    throw CUDAException("CUBLAS cudaMemcpy() failed.");
	  }
	  
	  return (*this);
	}
      }
      
#else
      if(this != &v){ // no self-assignment
	memcpy(this->data, v.data, sizeof(T)*v.dataSize);
      }
#endif
      
      return *this;
    }    
    

    template <typename T>
    vertex<T>& vertex<T>::operator=(vertex<T>&& t) 
    {
      if(this == &t) return *this; // self-assignment

      std::swap(this->data, t.data);
      std::swap(this->dataSize, t.dataSize);
      std::swap(this->compressor, t.compressor);
      
      return *this;
    }


    template <typename T>
    vertex<T>& vertex<T>::operator=(const matrix<T>& M)
    {
      if(M.compressor != 0 || this->compressor != 0){
	whiteice::logging.error("vertex::operator=(): compressed vector/matrix data.");
	assert(0);
	throw illegal_operation("vertex '='-operator: compressed vertex/matrix data.");
      }

      if(M.numRows != 1 || M.numCols != 1){
	whiteice::logging.error("vertex::operator=(): wrong dimension matrix M data.");
	assert(0);
	throw illegal_operation("vertex '='-operator: wrong dimension matrix M data.");
      }

      const unsigned int Mlen = M.numRows*M.numCols;
      
      if(Mlen != this->dataSize)
	if(this->resize(Mlen) != Mlen){
	  whiteice::logging.error("vertex::operator=(): resize() failed.");
	  assert(0);
	  throw illegal_operation("vertex '='-operator: out of memory");
	}

#ifdef CUBLAS

      if(M.data){
	cublasStatus_t cudaStat;
	
	// cuda copy memory
	if(typeid(T) == typeid(blas_real<float>)){
	  cudaStat = cublasScopy(cublas_handle, Mlen,
				 (const float*)M.data, 1, (float*)data, 1);
	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator=(): cublasScopy() failed.");
	    throw CUDAException("CUBLAS cublasScopy() failed.");
	  }

	  return (*this);
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  cudaStat = cublasDcopy(cublas_handle, Mlen,
				 (const double*)M.data, 1, (double*)data, 1);
	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator=(): cublasDcopy() failed.");
	    throw CUDAException("CUBLAS cublasDcopy() failed.");
	  }

	  return (*this);
	}
	else if(typeid(T) == typeid(blas_complex<float>)){
	  cudaStat = cublasCcopy(cublas_handle, Mlen,
				 (const cuComplex*)M.data, 1,
				 (cuComplex*)data, 1);
	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator=(): cublasCcopy() failed.");
	    throw CUDAException("CUBLAS cublasCcopy() failed.");
	  }

	  return (*this);
	}
	else if(typeid(T) == typeid(blas_complex<double>)){
	  cudaStat = cublasZcopy(cublas_handle, Mlen,
				 (const cuDoubleComplex*)M.data, 1,
				 (cuDoubleComplex*)data, 1);
	  gpu_sync();
	  
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("vertex::operator=(): cublasZcopy() failed.");
	    throw CUDAException("CUBLAS cublasZcopy() failed.");
	  }

	  return (*this);
	}
	else{
	  // generic memcopy [assumes type T does not allocated memory dynamically]
	  auto e = cudaMemcpy(data, M.data, Mlen*sizeof(T),
			      cudaMemcpyDeviceToDevice);
	  gpu_sync();

	  if(e != cudaSuccess){
	    whiteice::logging.error("vertex::operator=(): cudaMemcpy() failed.");
	    throw CUDAException("CUBLAS cudaMemcpy() failed.");
	  }
	  
	  return (*this);
	}
      }
      
#else
      memcpy(this->data, M.data, sizeof(T)*Mlen);
    
#endif
      
      return *this;
    }

    
    /***************************************************/

    // compares two vertexes for equality
    template <typename T>
    bool vertex<T>::operator==(const vertex<T>& v) const
      
    {
      if(v.dataSize != dataSize)
	return false; // throw uncomparable("vertex compare: dimension mismatch");

#ifdef CUBLAS
      // no fast BLAS code for this one

      // TODO implement fast OpenMP code for this one which stops
      // when change is found and don't continue
      for(unsigned int i=0;i<v.dataSize;i++){
	if(data[i] != v.data[i]) return false;
      }
      
      return true;
#else
      
      if(typeid(T) == typeid(blas_real<float>)    ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>)   ||
	 typeid(T) == typeid(blas_complex<double>)){
	
	return (memcmp(v.data, data, dataSize*sizeof(T)) == 0);
      }
      else{
	for(unsigned int i=0;i<v.dataSize;i++)
	  if(data[i] != v.data[i]) return false;
      }

#endif
      
      return true;
    }
    
    // compares two vertexes for non-equality
    template <typename T>
    bool vertex<T>::operator!=(const vertex<T>& v) const
      
    {
      if(v.dataSize != dataSize)
	return true; // throw uncomparable("vertex compare: dimension mismatch");

#ifdef CUBLAS

      return (memcmp(v.data, data, dataSize*sizeof(T)) != 0);
      
#else
      if(typeid(T) == typeid(blas_real<float>)    ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>)   ||
	 typeid(T) == typeid(blas_complex<double>)){
	
	return (memcmp(v.data, data, dataSize*sizeof(T)) != 0);
      }
      else{
	for(unsigned int i=0;i<v.dataSize;i++)
	  if(data[i] != v.data[i]) return true;
      }
      
#endif
      
      return false;
    }
    
    
    template <typename T>
    bool vertex<T>::operator>=(const vertex<T>& v) const {
      if(dataSize != 1 || v.dataSize != 1){
	whiteice::logging.error("vertex::operator>=(): no comparision for vector data.");
	throw uncomparable("vertex(): '>='-operator not defined");
      }
      else{
	return (data[0] >= v.data[0]);
      }
    }
    
    template <typename T>
    bool vertex<T>::operator<=(const vertex<T>& v) const {
      if(dataSize != 1 || v.dataSize != 1){
	whiteice::logging.error("vertex::operator<=(): no comparision for vector data.");
	throw uncomparable("vertex(): '<='-operator not defined");
      }
      else{
	return (data[0] <= v.data[0]);
      }
    }
    
    template <typename T>
    bool vertex<T>::operator< (const vertex<T>& v) const {
      if(dataSize != 1 || v.dataSize != 1){
	whiteice::logging.error("vertex::operator<(): no comparision for vector data.");
	throw uncomparable("vertex(): '<'-operator not defined");
      }
      else{
	return (data[0] < v.data[0]);
      }
    }

    template <typename T>
    bool vertex<T>::operator> (const vertex<T>& v) const {
      if(dataSize != 1 || v.dataSize != 1){
	whiteice::logging.error("vertex::operator>(): no comparision for vector data.");
	throw uncomparable("vertex(): '>'-operator not defined");
      }
      else{
	return (data[0] > v.data[0]);
      }
    }
    

    // assigns quaternion to 4 dimension vertex
    template <typename T>
    vertex<T>& vertex<T>::operator=(const quaternion<T>& q)
      
    {
      if(dataSize != 4){
      
	if(this->resize(4) != 4){
	  printf("ERROR: illegal operation: vector operator= failed: vector dim is not 4 (%s:%d)\n",
		 __FILE__, __LINE__);

	  whiteice::logging.error("vertex::operator=(): resize() failed.");
	  assert(0);
	  throw std::domain_error("vertex '='-operator: cannot assign quaternion - dimension mismatch");
	}
      }
      
      for(unsigned int i=0;i<4;i++)
	data[i] = q[i];
      
      return (*this);
    }

    
    // returns vertex with absolute value of each vertex element
    template <typename T>
    vertex<T>& vertex<T>::abs() 
    {
#ifdef CUBLAS

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<dataSize;i++)
	data[i] = whiteice::math::abs(data[i]);
      
#else

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<dataSize;i++)
	data[i] = whiteice::math::abs(data[i]);

#endif
      
      return (*this);
    }

    template <typename T>
    vertex<T>& vertex<T>::real()
    {
#ifdef CUBLAS

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<dataSize;i++)
	data[i] = whiteice::math::real(data[i]);
      
#else

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<dataSize;i++)
	data[i] = whiteice::math::real(data[i]);
      
#endif
      
      return (*this);
    }

    template <typename T>
    vertex<T>& vertex<T>::imag()
    {
#ifdef CUBLAS
      
#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<dataSize;i++)
	data[i] = whiteice::math::imag(data[i]);
      
#else

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<dataSize;i++)
	data[i] = whiteice::math::imag(data[i]);

#endif
      
      return (*this);
    }
    
    
    /***************************************************/
    // scalars
    
    
    /* sets all elements of vertex = given scalar */
    template <typename T>
    vertex<T>& vertex<T>::operator=(const T& s)
      
    {
#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<dataSize;i++)
	data[i] = s;
      
      return *this;
    }
    
    
    
    // multiples vertex with scalar */
    template <typename T>
    vertex<T> vertex<T>::operator*(const T& ss) const 
    {

#ifdef CUBLAS
      vertex<T> r(dataSize);
      r.zero();
      
      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = ss;
	
	cublasStatus_t s = cublasSaxpy(cublas_handle, (int)dataSize,
				       (const float*)&alpha,
				       (const float*)data, 1,
				       (float*)r.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*(): cublasSaxpy() failed.");
	  throw CUDAException("CUBLAS cublasSaxpy() failed.");
	}
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = ss;
	
	cublasStatus_t s = cublasDaxpy(cublas_handle, (int)dataSize,
				       (const double*)&alpha,
				       (const double*)data, 1,
				       (double*)r.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*(): cublasDaxpy() failed.");
	  throw CUDAException("CUBLAS cublasDaxpy() failed.");
	}
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = ss;
	
	cublasStatus_t s = cublasCaxpy(cublas_handle, (int)dataSize,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)data, 1,
				       (cuComplex*)r.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*(): cublasCaxpy() failed.");
	  throw CUDAException("CUBLAS cublasCaxpy() failed.");
	}
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = ss;
	
	cublasStatus_t s = cublasZaxpy(cublas_handle, (int)dataSize,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)data, 1,
				       (cuDoubleComplex*)r.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*(): cublasZaxpy() failed.");
	  throw CUDAException("CUBLAS cublasZaxpy() failed.");
	}
      }
      else{
	// *NO* CBLAS

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<dataSize;i++)
	  r.data[i] = ss*data[i];
      }

      return r;
      
      
#else
      vertex<T> r(dataSize);
      r.zero();
	
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_saxpy(dataSize, *((float*)&ss), (float*)data, 1, (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_caxpy(dataSize, (const float*)&ss, (float*)data, 1, (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_daxpy(dataSize, *((double*)&ss), (double*)data, 1, (double*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zaxpy(dataSize, (const double*)&ss, (double*)data, 1, (double*)r.data, 1);
      }
      else{ // "normal implementation"

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<dataSize;i++)
	  r.data[i] = data[i]*ss;
      }
      
      return r;
#endif
    }        
    
    
    // divides vertex with scalar */
    template <typename T>
    vertex<T> vertex<T>::operator/(const T& s) const 
    {
#ifdef CUBLAS

      vertex<T> r(dataSize);
      r.zero();
      const T invs = T(1.0f)/s;

      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = invs;

	cublasStatus_t e = cublasSaxpy(cublas_handle, (int)dataSize,
				       (const float*)&alpha,
				       (const float*)data, 1,
				       (float*)r.data, 1);
	gpu_sync();

	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator/(): cublasSaxpy() failed.");
	  throw CUDAException("CUBLAS cublasSaxpy() failed.");
	}
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = invs;
	
	cublasStatus_t e = cublasDaxpy(cublas_handle, (int)dataSize,
				       (const double*)&alpha,
				       (const double*)data, 1,
				       (double*)r.data, 1);
	gpu_sync();

	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator/(): cublasDaxpy() failed.");
	  throw CUDAException("CUBLAS cublasDaxpy() failed.");
	}
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = invs;
	
	cublasStatus_t e = cublasCaxpy(cublas_handle, (int)dataSize,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)data, 1,
				       (cuComplex*)r.data, 1);
	gpu_sync();

	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator/(): cublasCaxpy() failed.");
	  throw CUDAException("CUBLAS cublasCaxpy() failed.");
	}
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = invs;
	
	cublasStatus_t e = cublasZaxpy(cublas_handle, (int)dataSize,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)data, 1,
				       (cuDoubleComplex*)r.data, 1);
	gpu_sync();

	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator/(): cublasZaxpy() failed.");
	  throw CUDAException("CUBLAS cublasZaxpy() failed.");
	}
      }
      else{
	// *NO* CBLAS

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<dataSize;i++)
	  r.data[i] = data[i]*invs;
      }

      return r;
      
#else
      vertex<T> r(dataSize);
      r.zero();
      T ss = T(1)/s;      
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_saxpy(dataSize, *((float*)&ss), (float*)data, 1, (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_caxpy(dataSize, (const float*)&ss, (float*)data, 1, (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_daxpy(dataSize, *((double*)&ss), (double*)data, 1, (double*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zaxpy(dataSize, (const double*)&ss, (double*)data, 1, (double*)r.data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<dataSize;i++)
	  r.data[i] = data[i]*ss;
      }

      return r;
#endif
      
    }
    
    
    
    // multiples vertex with scalar */
    template <typename T>
    vertex<T>& vertex<T>::operator*=(const T& s) 
    {
#ifdef CUBLAS
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cublasStatus_t e = cublasSscal(cublas_handle,
				       dataSize,
				       (const float*)&s,
				       (float*)data, 1);
	gpu_sync();
	
	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*=(): cublasSscal() failed.");
	  throw CUDAException("CUBLAS cublasSscal() failed.");
	}

	return (*this);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cublasStatus_t e = cublasDscal(cublas_handle,
				       dataSize,
				       (const double*)&s,
				       (double*)data, 1);
	gpu_sync();
	
	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*=(): cublasDscal() failed.");
	  throw CUDAException("CUBLAS cublasDscal() failed.");
	}

	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cublasStatus_t e = cublasCscal(cublas_handle,
				       dataSize,
				       (const cuComplex*)&s,
				       (cuComplex*)data, 1);
	gpu_sync();
	
	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*=(): cublasCscal() failed.");
	  throw CUDAException("CUBLAS cublasCscal() failed.");
	}

	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cublasStatus_t e = cublasZscal(cublas_handle,
				       dataSize,
				       (const cuDoubleComplex*)&s,
				       (cuDoubleComplex*)data, 1);
	gpu_sync();
	
	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*=(): cublasZscal() failed.");
	  throw CUDAException("CUBLAS cublasZscal() failed.");
	}

	return (*this);
      }
      else{

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] *= s;

	return (*this);
      }
      
#else
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sscal(dataSize, *((float*)&s), (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_cscal(dataSize, (const float*)&s, (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dscal(dataSize, *((double*)&s), (double*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zscal(dataSize, (const double*)&s, (double*)data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] *= s;
      }
      
      return *this;
#endif
    }
    
    
    // multiples vertex with scalar */
    template <typename T>
    vertex<T>& vertex<T>::operator/=(const T& s) 
    {
#ifdef CUBLAS

      const T invs = T(1.0f)/s;

      if(typeid(T) == typeid(blas_real<float>)){
	
	cublasStatus_t e = cublasSscal(cublas_handle,
				       dataSize,
				       (const float*)&invs,
				       (float*)data, 1);
	gpu_sync();
	
	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator/=(): cublasSscal() failed.");
	  throw CUDAException("CUBLAS cublasSscal() failed.");
	}

	return (*this);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cublasStatus_t e = cublasDscal(cublas_handle,
				       dataSize,
				       (const double*)&invs,
				       (double*)data, 1);
	gpu_sync();
	
	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator/=(): cublasDscal() failed.");
	  throw CUDAException("CUBLAS cublasDscal() failed.");
	}

	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cublasStatus_t e = cublasCscal(cublas_handle,
				       dataSize,
				       (const cuComplex*)&invs,
				       (cuComplex*)data, 1);
	gpu_sync();
	
	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator/=(): cublasCscal() failed.");
	  throw CUDAException("CUBLAS cublasCscal() failed.");
	}

	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){

	cublasStatus_t e = cublasZscal(cublas_handle,
				       dataSize,
				       (const cuDoubleComplex*)&invs,
				       (cuDoubleComplex*)data, 1);
	gpu_sync();

	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator/=(): cublasZscal() failed.");
	  throw CUDAException("CUBLAS cublasZscal() failed.");
	}

	return (*this);
      }
      else{

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] *= invs;

	return (*this);
      }
      
#else
      T ss = T(1.0)/s;
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sscal(dataSize, *((float*)&ss), (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_cscal(dataSize, (const float*)&ss, (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dscal(dataSize, *((double*)&ss), (double*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zscal(dataSize, (const double*)&ss, (double*)data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] *= ss;
      }
      
      return *this;
#endif
    }
    
    
    
    // scalar times vertex
    template <typename T>
    vertex<T> operator*(const T& s, const vertex<T>& v)
    {
#ifdef CUBLAS
      vertex<T> r(v.dataSize);
      r.zero();

      if(typeid(T) == typeid(blas_real<float>)){
	
	cublasStatus_t ss = cublasSaxpy(cublas_handle, (int)v.dataSize,
				       (float*)&s,
				       (const float*)v.data, 1,
				       (float*)r.data, 1);
	gpu_sync();
	
	if(ss != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*(): cublasSaxpy() failed.");
	  throw CUDAException("CUBLAS cublasSaxpy() failed.");
	}

	return r;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cublasStatus_t ss = cublasDaxpy(cublas_handle, (int)v.dataSize,
					(double*)&s,
					(const double*)v.data, 1,
					(double*)r.data, 1);
	gpu_sync();

	if(ss != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*(): cublasDaxpy() failed.");
	  throw CUDAException("CUBLAS cublasDaxpy() failed.");
	}

	return r;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cublasStatus_t ss = cublasCaxpy(cublas_handle, (int)v.dataSize,
					(cuComplex*)&s,
					(const cuComplex*)v.data, 1,
					(cuComplex*)r.data, 1);
	gpu_sync();

	if(ss != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*(): cublasCaxpy() failed.");
	  throw CUDAException("CUBLAS cublasCaxpy() failed.");
	}

	return r;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cublasStatus_t ss = cublasZaxpy(cublas_handle, (int)v.dataSize,
					(cuDoubleComplex*)&s,
					(const cuDoubleComplex*)v.data, 1,
					(cuDoubleComplex*)r.data, 1);
	gpu_sync();
	
	if(ss != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*(): cublasZaxpy() failed.");
	  throw CUDAException("CUBLAS cublasZaxpy() failed.");
	}

	return r;
      }
      else{
	// *NO* CBLAS

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<v.dataSize;i++)
	  r.data[i] = s*v.data[i];

	return r;
      }
      
#else
      
      vertex<T> r(v.dataSize);
      r.zero();
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_saxpy(v.dataSize, *((float*)&s), (float*)v.data, 1, (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	blas_complex<float> ss(*((blas_complex<float>*)&s));
	
	cblas_caxpy(v.dataSize, (const float*)&ss, (float*)v.data, 1, (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_daxpy(v.dataSize, *((double*)&s), (double*)v.data, 1, (double*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	blas_complex<double> ss(*((blas_complex<double>*)&s));
	
	cblas_zaxpy(v.dataSize, (const double*)&ss, (double*)v.data, 1, (double*)r.data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<v.dataSize;i++)
	  r.data[i] = v.data[i]*s;
      }
      
      return r;
#endif
    }
    
    
    // multiplies matrix from left: return v = (*this)*M
    template <typename T>
    vertex<T> vertex<T>::operator* (const matrix<T>& M) const
      
    {
      if(dataSize != M.numRows){
	printf("ERROR: illegal operation: vector/matrix operator* failed: dim %d != dim %dx%d (%s:%d)\n",
	       dataSize, M.numRows, M.numCols, __FILE__, __LINE__);
	whiteice::logging.error("vertex::operator*(): vertex*matrix dimensions mismatch.");
	assert(0);
	throw std::invalid_argument("multiply: vertex/matrix dim. mismatch");
      }

#ifdef CUBLAS

      vertex<T> r(M.numCols);
      r.zero();

      // BLAS level 2
      // uses optimized cblas_Xgemv() functions

      if(typeid(T) == typeid(blas_real<float>)){

	// v^t*M = M^t * v

	const T alpha = 1.0f;
	const T beta = 0.0f;

	cublasStatus_t s = cublasSgemv
	  (cublas_handle,
	   CUBLAS_OP_T,
	   M.numRows, M.numCols,
	   (const float*)&alpha,
	   (const float*)M.data,
	   M.numRows, // different value than in column major matrixes! (cblas)
	   (const float*)(this->data), 1,
	   (const float*)&beta,
	   (float*)r.data, 1);

	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*(): cublasSgemv() failed.");
	  throw CUDAException("CUBLAS cublasSgemv() call failed.");
	}
	
	/*
	cblas_sgemv(CblasRowMajor, CblasTrans,
		    M.numRows, M.numCols,
		    1.0f, (float*)M.data, M.numCols, // 1,
		    (float*)data, 1,
		    0.0f, (float*)r.data, 1);
	*/

	return r;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = 1.0f;
	const T beta = 0.0f;

	cublasStatus_t s = cublasDgemv
	  (cublas_handle,
	   CUBLAS_OP_T,
	   M.numRows, M.numCols,
	   (const double*)&alpha,
	   (const double*)M.data,
	   M.numRows, // different value than in column major matrixes! (cblas)
	   (const double*)(this->data), 1,
	   (const double*)&beta,
	   (double*)r.data, 1);

	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*(): cublasDgemv() failed.");
	  throw CUDAException("CUBLAS cublasDgemv() call failed.");
	}

	return r;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = 1.0f;
	const T beta = 0.0f;

	cublasStatus_t s = cublasCgemv
	  (cublas_handle,
	   CUBLAS_OP_T,
	   M.numRows, M.numCols,
	   (const cuComplex*)&alpha,
	   (const cuComplex*)M.data,
	   M.numRows, // different value than in column major matrixes! (cblas)
	   (const cuComplex*)(this->data), 1,
	   (const cuComplex*)&beta,
	   (cuComplex*)r.data, 1);

	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*(): cublasCgemv() failed.");
	  throw CUDAException("CUBLAS cublasCgemv() call failed.");
	}

	return r;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = 1.0;
	const T beta = 0.0;

	cublasStatus_t s = cublasZgemv
	  (cublas_handle,
	   CUBLAS_OP_T,
	   M.numRows, M.numCols,
	   (const cuDoubleComplex*)&alpha,
	   (const cuDoubleComplex*)M.data,
	   M.numRows, // different value than in column major matrixes! (cblas)
	   (const cuDoubleComplex*)(this->data), 1,
	   (const cuDoubleComplex*)&beta,
	   (cuDoubleComplex*)r.data, 1);

	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::operator*(): cublasZgemv() failed.");
	  throw CUDAException("CUBLAS cublasZgemv() call failed.");
	}

	return r;
      }
      else{

#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<M.numCols;j++){
	  for(unsigned int i=0;i<M.numRows;i++)
	    r.data[j] += data[i]*M(i,j);
	}
	
	return r;      
      }
      
      
#else
      vertex<T> r(M.numCols);
      r.zero();
      
      // BLAS level 2
      // uses optimized cblas_Xgemv() functions
      
      if(typeid(T) == typeid(blas_real<float>)){

	cblas_sgemv(CblasRowMajor, CblasTrans,
		    M.numRows, M.numCols,
		    1.0f, (float*)M.data, M.numCols, // 1,
		    (float*)data, 1,
		    0.0f, (float*)r.data, 1);
	
	return r;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	blas_complex<float> a, b;
	a = 1.0f; b = 0.0f;
	
	cblas_cgemv(CblasRowMajor, CblasTrans,
		    M.numRows, M.numCols,
		    (float*)(&a), (float*)M.data, M.numCols,
		    (float*)data, 1,
		    (float*)(&b), (float*)r.data, 1);
	
	return r;

      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dgemv(CblasRowMajor, CblasTrans,
		    M.numRows, M.numCols,
		    1.0, (double*)M.data, M.numCols,
		    (double*)data, 1,
		    0.0, (double*)r.data, 1);
	
	return r;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	blas_complex<double> a, b;
	a = 1.0f; b = 0.0f;	
	
	cblas_zgemv(CblasRowMajor, CblasTrans,
		    M.numRows, M.numCols,
		    (double*)(&a), (double*)M.data, M.numCols,
		    (double*)data, 1,
		    (double*)(&b), (double*)r.data, 1);
	
	return r;
      }
      else{ // vertex * matrix calculation
	
	for(unsigned int j=0;j<M.numCols;j++){
	  for(unsigned int i=0;i<M.numRows;i++)
	    r.data[j] += data[i]*M(i,j);
	}
	
	return r;
      }
#endif
      
    }
    
    
    /***************************************************/
    
    
    template <typename T>
    matrix<T> vertex<T>::outerproduct() const 
    {
#ifdef CUBLAS

      const unsigned int N = dataSize;
      if(N<0){
	whiteice::logging.error("vertex::outerproduct(): zero length vector");
	assert(0);
	throw illegal_operation("vertex<T>::outerproduct(): zero length vector");
      }

      if(typeid(T) == typeid(blas_real<float>)){
	const float alpha = 1.0f;
	
	matrix<T> M(N,N);
	M.zero();

	cublasStatus_t s = cublasSspr
	  (cublas_handle,
	   CUBLAS_FILL_MODE_LOWER,
	   N, (const float*)&alpha,
	   (const float*)this->data, 1,
	   (float*)M.data);	

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::outerproduct(): cublasSspr() failed.");
	  throw CUDAException("CUBLAS cublasSspr() call failed.");
	}

	
	// NOT OPTIMIZED
	{
	  // last element of the matrix
	  unsigned int non_packed_index = N*N;
	  unsigned int packed_index = N*(N+1)/2;
	  
	  for(unsigned int i=0;i<(N-1);i++){ // (N-i):th column
	    auto diag_non_packed_index = non_packed_index - (i+1);
	    packed_index -= (i+1);
	    
	    auto e = cudaMemcpy(&(M.data[diag_non_packed_index]), &(M.data[packed_index]),
				sizeof(T)*(i+1), cudaMemcpyDeviceToDevice);

	    if(e != cudaSuccess){
	      whiteice::logging.error("vertex::outerproduct(): cudaMemcpy() failed.");
	      throw CUDAException("CUBLAS cudaMemcpy() call failed.");
	    }

	    non_packed_index -= N;
	  }
	}

	gpu_sync();

	// copies lower triangular elements to upper triangular part of the M matrix
	for(unsigned int i=0;i<N;i++)
	  for(unsigned int j=(i+1);j<N;j++)
	    M(i,j) = M(j,i);

	return M;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const double alpha = 1.0;
	
	matrix<T> M(N,N);
	M.zero();

	cublasStatus_t s = cublasDspr
	  (cublas_handle,
	   CUBLAS_FILL_MODE_LOWER,
	   N, (const double*)&alpha,
	   (const double*)this->data, 1, 
	   (double*)M.data);

	

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::outerproduct(): cublasDspr() failed.");
	  throw CUDAException("CUBLAS cublasDspr() call failed.");
	}

	
	// NOT OPTIMIZED
	{
	  // last element of the matrix
	  unsigned int non_packed_index = N*N;
	  unsigned int packed_index = N*(N+1)/2;
	  
	  for(unsigned int i=0;i<(N-1);i++){ // (N-i):th column
	    auto diag_non_packed_index = non_packed_index - (i+1);
	    packed_index -= (i+1);
	    
	    auto e = cudaMemcpy(&(M.data[diag_non_packed_index]), &(M.data[packed_index]),
				sizeof(T)*(i+1), cudaMemcpyDeviceToDevice);

	    if(e != cudaSuccess){
	      whiteice::logging.error("autocorrelation(): cudaMemcpy() failed.");
	      throw CUDAException("CUBLAS cudaMemcpy() call failed.");
	    }

	    non_packed_index -= N;
	  }
	}

	gpu_sync();

	// copies lower triangular elements to upper triangular part of the M matrix
	for(unsigned int i=0;i<N;i++)
	  for(unsigned int j=(i+1);j<N;j++)
	    M(i,j) = M(j,i);

	return M;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const float alpha = 1.0f;
		
	matrix<T> M(N,N);
	M.zero();

	cublasStatus_t s = cublasChpr
	  (cublas_handle,
	   CUBLAS_FILL_MODE_LOWER,
	   N, (const float*)&alpha,
	   (const cuComplex*)this->data, 1,
	   (cuComplex*)M.data);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::outerproduct(): cublasChpr() failed.");
	  throw CUDAException("CUBLAS cublasChpr() call failed.");
	}

	
	// NOT OPTIMIZED
	{
	  // last element of the matrix
	  unsigned int non_packed_index = N*N;
	  unsigned int packed_index = N*(N+1)/2;
	  
	  for(unsigned int i=0;i<(N-1);i++){ // (N-i):th column
	    auto diag_non_packed_index = non_packed_index - (i+1);
	    packed_index -= (i+1);
	    
	    auto e = cudaMemcpy(&(M.data[diag_non_packed_index]), &(M.data[packed_index]),
				sizeof(T)*(i+1), cudaMemcpyDeviceToDevice);

	    if(e != cudaSuccess){
	      whiteice::logging.error("autocorrelation(): cudaMemcpy() failed.");
	      throw CUDAException("CUBLAS cudaMemcpy() call failed.");
	    }

	    non_packed_index -= N;
	  }
	}
	
	gpu_sync();

	// copies lower triangular elements to upper triangular part of the M matrix
	for(unsigned int i=0;i<N;i++)
	  for(unsigned int j=(i+1);j<N;j++)
	    M(i,j) = M(j,i);

	return M;	
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	const double alpha = 1.0;
		
	matrix<T> M(N,N);
	M.zero();

	cublasStatus_t s = cublasZhpr
	  (cublas_handle,
	   CUBLAS_FILL_MODE_LOWER,
	   N, (const double*)&alpha,
	   (const cuDoubleComplex*)this->data, 1,
	   (cuDoubleComplex*)M.data);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::outerproduct(): cublasZhpr() failed.");
	  throw CUDAException("CUBLAS cublasZhpr() call failed.");
	}

	// NOT OPTIMIZED
	{
	  // last element of the matrix
	  unsigned int non_packed_index = N*N;
	  unsigned int packed_index = N*(N+1)/2;
	  
	  for(unsigned int i=0;i<(N-1);i++){ // (N-i):th column
	    auto diag_non_packed_index = non_packed_index - (i+1);
	    packed_index -= (i+1);
	    
	    auto e = cudaMemcpy(&(M.data[diag_non_packed_index]), &(M.data[packed_index]),
				sizeof(T)*(i+1), cudaMemcpyDeviceToDevice);

	    if(e != cudaSuccess){
	      whiteice::logging.error("autocorrelation(): cudaMemcpy() failed.");
	      throw CUDAException("CUBLAS cudaMemcpy() call failed.");
	    }

	    non_packed_index -= N;
	  }
	}

	gpu_sync();

	// copies lower triangular elements to upper triangular part of the M matrix
	for(unsigned int i=0;i<N;i++)
	  for(unsigned int j=(i+1);j<N;j++)
	    M(i,j) = M(j,i);

	return M;
      }
      else{
	return outerproduct(*this, *this);
      }
      
#else
      const unsigned int N = dataSize;
      T s = T(1.0f);
      
      if(typeid(T) == typeid(blas_real<float>)){
	matrix<T> M(N, N);
	
	cblas_sspr(CblasRowMajor, CblasUpper, N, 
		   *((float*)&s), (float*)data, 1, (float*)M.data);

	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(M.data[r*N + r]), &(M.data[r*N - ((r - 1)*r)/2]), sizeof(T)*(i+1));
	}
	
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_scopy(N - i - 1, 
		      (float*)&(M.data[i*N + 1 + i]), 1,
		      (float*)&(M.data[(i+1)*N + i]), N);
	return M;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	matrix<T> M(N, N);
	
	cblas_chpr(CblasRowMajor, CblasUpper, N,
		   *((float*)&s), (float*)data, 1, (float*)M.data);

	
	
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(M.data[r*N + r]), &(M.data[r*N - ((r - 1)*r)/2]), sizeof(T) * (i+1));
	}
	
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_ccopy(N - i - 1, 
		      (float*)&(M.data[i*N + 1 + i]), 1,
		      (float*)&(M.data[(i+1)*N + i]), N);
	return M;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	matrix<T> M(N, N);
	
	cblas_dspr(CblasRowMajor, CblasUpper, N, 
		   *((double*)&s), (double*)data, 1, (double*)(M.data));
	
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(M.data[r*N + r]), &(M.data[r*N - ((r - 1)*r)/2]), sizeof(T) * (i+1));	
	}
	
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_dcopy(N - i - 1,
		      (double*)&(M.data[i*N + 1 + i]), 1,
		      (double*)&(M.data[(i+1)*N + i]), N);
	return M;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	matrix<T> M(N, N);
	
	cblas_zhpr(CblasRowMajor, CblasUpper, N,
		   *((double*)&s), (double*)data, 1, (double*)M.data);
	
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(M.data[r*N + r]), &(M.data[r*N - ((r - 1)*r)/2]), sizeof(T) * (i+1));
	}
	
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_zcopy(N - i - 1, 
		      (double*)&(M.data[i*N + 1 + i]), 1,
		      (double*)&(M.data[(i+1)*N + i]), N);
	return M;	  
      }
      else
	return outerproduct(*this, *this);
#endif
    }


    template <typename T>
    matrix<T> vertex<T>::outerproduct(const vertex<T>& v) const
      
    {
      return outerproduct(*this, v);
    }
    
    
    /* outerproduct of N length vertexes */
    template <typename T>
    matrix<T> vertex<T>::outerproduct(const vertex<T>& v0,
				      const vertex<T>& v1) const
      
    {
#ifdef CUBLAS

      matrix<T> m(v0.dataSize, v1.dataSize);
      
#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<v0.dataSize;i++)
	for(unsigned int j=0;j<v1.dataSize;j++)
	  m(i,j) = v0.data[i]*whiteice::math::conj(v1.data[j]);
      
      return m;
      
#else
      
      matrix<T> m(v0.dataSize, v1.dataSize);

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<v0.dataSize;i++)
	for(unsigned int j=0;j<v1.dataSize;j++)
	  m(i,j) = v0.data[i]*whiteice::math::conj(v1.data[j]);
      
      return m;
#endif
    }
    
    
    // element-wise multiplication of vector elements
    template <typename T>
    vertex<T>& vertex<T>::dotmulti(const vertex<T>& v) 
    {
      if(this->dataSize != v.dataSize){
	printf("ERROR: illegal operation: vector dotmulti() failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);

	whiteice::logging.error("vertex::dotmulti(): vector dimension mismatch.");
	assert(0);
	throw illegal_operation("vector op: vector dim. mismatch");
      }

      // should be optimized

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<dataSize;i++)
	data[i] *= v.data[i];
      
      return (*this);
    }
    
    
    template <typename T>
    bool vertex<T>::subvertex(vertex<T>& v,
			      unsigned int x0,
			      unsigned int len) const 
    {
      if(x0+len > dataSize)
	return false;

      v.resize(len);
      
#ifdef CUBLAS

      if(typeid(T) == typeid(blas_real<float>)){
	cublasStatus_t  s = cublasScopy(cublas_handle,
					(int)len,
					(const float*)&(this->data[x0]), 1,
					(float*)v.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::subvertex(): cublasScopy() failed.");
	  throw CUDAException("CUBLAS cublasScopy() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cublasStatus_t  s = cublasDcopy(cublas_handle,
					(int)len,
					(const double*)&(this->data[x0]), 1,
					(double*)v.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::subvertex(): cublasDcopy() failed.");
	  throw CUDAException("CUBLAS cublasDcopy() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cublasStatus_t  s = cublasCcopy(cublas_handle,
					(int)len,
					(const cuComplex*)&(this->data[x0]), 1,
					(cuComplex*)v.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::subvertex(): cublasCcopy() failed.");
	  throw CUDAException("CUBLAS cublasCcopy() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cublasStatus_t  s = cublasZcopy(cublas_handle,
					(int)len,
					(const cuDoubleComplex*)&(this->data[x0]), 1,
					(cuDoubleComplex*)v.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::subvertex(): cublasZcopy() failed.");
	  throw CUDAException("CUBLAS cublasZcopy() failed.");
	}

	return true;
      }
      else{
	auto s = cudaMemcpy(v.data, data + x0, len*sizeof(T),
			    cudaMemcpyDeviceToDevice);
	gpu_sync();

	if(s != cudaSuccess){
	  whiteice::logging.error("vertex::subvertex(): cudaMemcpy() failed.");
	  throw CUDAException("CUBLAS cudaMemcpy() failed.");
	}
	
	return true;
      }
      
#else
      memcpy(v.data, data + x0, len*sizeof(T));
      return true;
#endif
    }
    
    
    template <typename T>
    bool vertex<T>::write_subvertex(const vertex<T>& v, unsigned int x0) 
    {
      const unsigned int len = v.size();
      
      if(x0+len > dataSize)
	return false;
      
#ifdef CUBLAS
      
      if(typeid(T) == typeid(blas_real<float>)){
	cublasStatus_t  s = cublasScopy(cublas_handle,
					(int)len,
					(const float*)v.data, 1,
					(float*)&(this->data[x0]), 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::write_subvertex(): cublasScopy() failed.");
	  throw CUDAException("CUBLAS cublasScopy() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cublasStatus_t  s = cublasDcopy(cublas_handle,
					(int)len,
					(const double*)v.data, 1,
					(double*)&(this->data[x0]), 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::write_subvertex(): cublasDcopy() failed.");
	  throw CUDAException("CUBLAS cublasDcopy() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cublasStatus_t  s = cublasCcopy(cublas_handle,
					(int)len,
					(const cuComplex*)v.data, 1,
					(cuComplex*)&(this->data[x0]), 1);
	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::write_subvertex(): cublasCcopy() failed.");
	  throw CUDAException("CUBLAS cublasCcopy() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cublasStatus_t  s = cublasZcopy(cublas_handle,
					(int)len,
					(const cuDoubleComplex*)v.data, 1,
					(cuDoubleComplex*)&(this->data[x0]), 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::write_subvertex(): cublasZcopy() failed.");
	  throw CUDAException("CUBLAS cublasZcopy() failed.");
	}

	return true;
      }
      else{
	auto s = cudaMemcpy(data + x0, v.data, len*sizeof(T),
			    cudaMemcpyDeviceToDevice);
	gpu_sync();
	
	if(s != cudaSuccess){
	  whiteice::logging.error("vertex::write_subvertex(): cudaMemcpy() failed.");
	  throw CUDAException("CUBLAS cudaMemcpy() failed.");
	}
	  
	return true;
      }
      
#else      
      memcpy(data + x0, v.data, len*sizeof(T));
      return true;
#endif
    }
    
    
    
    template <typename T>
    bool vertex<T>::comparable() 
    {
      return false;
    }
    
    
    template <typename T>
    bool vertex<T>::saveAscii(const std::string& filename) const 
    {
      FILE* fp = fopen(filename.c_str(), "wt");
      if(fp == NULL || ferror(fp)) return false;

      if(this->dataSize > 0){
	T f = this->data[0];

	if(typeid(T) == typeid(blas_complex<float>) ||
	   typeid(T) == typeid(blas_complex<double>)){
	  
	  auto r = whiteice::math::real(f);
	  auto i = whiteice::math::imag(f);
	
	  double rd, id;
	  whiteice::math::convert(rd, r);
	  whiteice::math::convert(id, i);
	  
	  fprintf(fp, "%f+%fi", rd, id);
	}
	else{
	  auto r = whiteice::math::real(f);
	  double rd;
	  whiteice::math::convert(rd, r);
	  
	  fprintf(fp, "%f", rd);
	}
      }
      
      
      for(unsigned int k=1;k<this->dataSize;k++){
	T f = this->data[k];

	if(typeid(T) == typeid(blas_complex<float>) ||
	   typeid(T) == typeid(blas_complex<double>)){
	  auto r = whiteice::math::real(f);
	  auto i = whiteice::math::imag(f);
	
	  double rd, id;
	  whiteice::math::convert(rd, r);
	  whiteice::math::convert(id, i);
	  
	  fprintf(fp, "%f+%fi", rd, id);
	}
	else{
	  auto r = whiteice::math::real(f);
	  double rd;
	  whiteice::math::convert(rd, r);
	  
	  fprintf(fp, "%f", rd);
	}
      }
      
      fprintf(fp, "\n");
      
      if(ferror(fp)){
	fclose(fp);
	return false;
      }
      
      fclose(fp);
      return true;
    }
    

    ////////////////////////////////////////////////////////////
    // matrix data compression
    // note: compressor destroys possible memory
    // aligmentations
    
    
    template <typename T>
    bool vertex<T>::compress() 
    {
      if(compressor != 0) return false; // already compressed

#ifdef CUBLAS
      whiteice::logging.error("FIXME Memory compression is not supported with GPU memory.");
      return false;
#else
      
      compressor = new MemoryCompressor();
      
      compressor->setMemory(data, sizeof(T)*dataSize);
      // let compressor allocate the memory
      
      if(compressor->compress()){ // compression ok.
	free(data); data = 0; // free's memory
	compressor->setMemory(data, 0);
	return true;
      }
      else{
	if(compressor->getTarget() != 0)
	  free(compressor->getTarget());
	
	delete compressor;
	compressor = 0;
	return false;
      }
#endif
    }
    
    
    template <typename T>
    bool vertex<T>::decompress() 
    {
      if(compressor == 0) return false; // not compressed

#ifdef CUBLAS
      whiteice::logging.error("FIXME Memory compression is not supported with GPU memory.");
      return false;
#else
      
      if(compressor->decompress()){ // decompression ok.
	data = (T*)( compressor->getMemory() );
	
	free(compressor->getTarget());
	
	delete compressor;
	compressor = 0;
	
	return true;
      }
      else{
	return false;
      }
#endif
      
    }

    
    
    template <typename T>
    bool vertex<T>::iscompressed() const 
    {
      return (compressor != 0);
    }
    

    template <typename T>
    float vertex<T>::ratio() const 
    {
      if(compressor == 0) return 1.0f;
      return compressor->ratio();
    }
    
    
    ////////////////////////////////////////////////////////////
    
    
    // copies this->vertex[start:(start+len-1)] = data[0:(len-1)]
    template <typename T>
    bool vertex<T>::importData(const T* data_,
			       unsigned int len,
			       unsigned int start) 
    {
      if(len == 0)
	len = dataSize - start;
      else if(len+start > dataSize)
	return false;
      if(start >= dataSize)
	return false;

#ifdef CUBLAS
      const auto& x0 = start;

      if(typeid(T) == typeid(blas_real<float>)){
	cublasStatus_t  s = cublasScopy(cublas_handle,
					(int)len,
					(const float*)data_, 1,
					(float*)&(this->data[x0]), 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::importData(): cublasScopy() failed.");
	  throw CUDAException("CUBLAS cublasScopy() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cublasStatus_t  s = cublasDcopy(cublas_handle,
					(int)len,
					(const double*)data_, 1,
					(double*)&(this->data[x0]), 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::importData(): cublasDcopy() failed.");
	  throw CUDAException("CUBLAS cublasDcopy() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cublasStatus_t  s = cublasCcopy(cublas_handle,
					(int)len,
					(const cuComplex*)data_, 1,
					(cuComplex*)&(this->data[x0]), 1);
	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::importData(): cublasCcopy() failed.");
	  throw CUDAException("CUBLAS cublasCcopy() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cublasStatus_t  s = cublasZcopy(cublas_handle,
					(int)len,
					(const cuDoubleComplex*)data_, 1,
					(cuDoubleComplex*)&(this->data[x0]), 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::importData(): cublasZcopy() failed.");
	  throw CUDAException("CUBLAS cublasZcopy() failed.");
	}

	return true;
      }
      else{
	auto s = cudaMemcpy(data + x0, data_, len*sizeof(T),
			    cudaMemcpyHostToDevice);
	gpu_sync();
	
	if(s != cudaSuccess){
	  whiteice::logging.error("vertex::importData(): cudaMemcpy() failed.");
	  throw CUDAException("CUBLAS cudaMemcpy() failed.");
	}
	
	return true;
      }

#else
      
      memcpy(this->data + start, data_, len*sizeof(T));

#endif
      
      return true;
    }
    
    
    // copies data[0:(len-1)] = this->vertex[start:(start+len-1)]
    template <typename T>
    bool vertex<T>::exportData(T* data_,
			       unsigned int len,
			       unsigned int start) const 
    {
      if(len == 0)
	len = dataSize - start;
      else if(len+start > dataSize)
	return false;
      if(start >= dataSize)
	return false;

#ifdef CUBLAS
      const auto& x0 = start;

      if(typeid(T) == typeid(blas_real<float>)){
	cublasStatus_t  s = cublasScopy(cublas_handle,
					(int)len,
					(const float*)&(this->data[x0]), 1,
					(float*)data_, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::exportData(): cublasScopy() failed.");
	  throw CUDAException("CUBLAS cublasScopy() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cublasStatus_t  s = cublasDcopy(cublas_handle,
					(int)len,
					(const double*)&(this->data[x0]), 1,
					(double*)data_, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::exportData(): cublasDcopy() failed.");
	  throw CUDAException("CUBLAS cublasDcopy() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cublasStatus_t  s = cublasCcopy(cublas_handle,
					(int)len,
					(const cuComplex*)&(this->data[x0]), 1,
					(cuComplex*)data_, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("vertex::exportData(): cublasCcopy() failed.");
	  throw CUDAException("CUBLAS cublasCcopy() failed.");
	}

	return true;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cublasStatus_t  s = cublasZcopy(cublas_handle,
					(int)len,
					(const cuDoubleComplex*)&(this->data[x0]), 1,
					(cuDoubleComplex*)data_, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
 	  whiteice::logging.error("vertex::exportData(): cublasZcopy() failed.");
	  throw CUDAException("CUBLAS cublasZcopy() failed.");
	}

	return true;
      }
      else{

	auto s = cudaMemcpy(data_, data + x0, len*sizeof(T),
			    cudaMemcpyDeviceToHost);
	gpu_sync();
	
	if(s != cudaSuccess){
	  whiteice::logging.error("vertex::exportData(): cudaMemcpy() failed.");
	  throw CUDAException("CUBLAS cudaMemcpy() failed.");
	}
	
	return true;
      }

#else
      
      memcpy(data_, this->data + start, len*sizeof(T));

#endif
      
      return true;
    }


    template <typename T>
    void vertex<T>::toString(std::string& line) const 
    {
      if(this->size() == 0){ line = ""; return; }

      if(typeid(T) == typeid(blas_real<float>) ||
	 typeid(T) == typeid(blas_real<double>)){

	char buffer[30];
	
	if(this->size() == 1){
	  double temp = 0.0;
	  whiteice::math::convert(temp, (*this)[0]);
	  snprintf(buffer, 30, "%f", temp);
	  line = buffer;
	  return;
	}
	
	line = "[";
	double temp = 0.0;
	
	whiteice::math::convert(temp, (*this)[0]);
	snprintf(buffer, 30, "%f", temp);
	line += buffer;
	
	for(unsigned int i=1;i<this->size();i++){
	  whiteice::math::convert(temp, (*this)[i]);
	  snprintf(buffer, 30, " %f", temp);
	  line += buffer;
	}
	
	line += "]";
      }
      else{ // prints complex numbers
	char buffer[30];

	if(this->size() == 1){
	  auto r = whiteice::math::real((*this)[0]);
	  auto i = whiteice::math::imag((*this)[0]);
	  
	  double temp, temp2;
	  whiteice::math::convert(temp, r);
	  whiteice::math::convert(temp2, i);
	  
	  snprintf(buffer, 30, "%f+%fi", temp, temp2);
	  line = buffer;
	  return;
	}
	
	line = "[";

	auto r = whiteice::math::real((*this)[0]);
	auto i = whiteice::math::imag((*this)[0]);
	
	double temp, temp2;
	
	whiteice::math::convert(temp, r);
	whiteice::math::convert(temp2, i);
	
	snprintf(buffer, 30, "%f+%fi", temp, temp2);
	line += buffer;
	
	for(unsigned int k=1;k<this->size();k++){
	  r = whiteice::math::real((*this)[k]);
	  i = whiteice::math::imag((*this)[k]);
	  
	  whiteice::math::convert(temp, r);
	  whiteice::math::convert(temp2, i);

	  snprintf(buffer, 30, "%f+%fi", temp, temp2);
	  line += buffer;
	}
	
	line += "]";
      }
	
      return;
    }
    
    ////////////////////////////////////////////////////////////
    
    /***************************************************/
    
    template <typename T>
    std::ostream& operator<<(std::ostream& ios,
			     const whiteice::math::vertex<T>& v)
    {
      if(v.size() == 1){
	ios << v[0];
      }
      else if(v.size() > 1){
	
	ios << "[";
	
	for(unsigned int i=0;i<v.size();i++){
	  ios << " " << v[i];
	}
	
	ios << " ]";
      }
    
      return ios;
    }
    
    
    // explicit template instantations
    
    template class vertex<float>;
    template class vertex<double>;
    template class vertex<complex<float> >;
    template class vertex<complex<double> >;
    
    //template class vertex<int>;
    //template class vertex<char>;
    //template class vertex<unsigned int>;
    //template class vertex<unsigned char>;
    
    template class vertex< blas_real<float> >;
    template class vertex< blas_real<double> >;
    template class vertex< blas_complex<float> >;
    template class vertex< blas_complex<double> >;

    template class vertex< superresolution< blas_real<float>, modular<unsigned int> > >;
    template class vertex< superresolution< blas_real<double>, modular<unsigned int> > >;

    template class vertex< superresolution< blas_complex<float>, modular<unsigned int> > >;
    template class vertex< superresolution< blas_complex<double>, modular<unsigned int> > >;
    
    
    template vertex<float> operator*<float>(const float& s, const vertex<float>& v);
    template vertex<double> operator*<double>(const double& s, const vertex<double>& v);
    
    template vertex<complex<float> > operator*<complex<float> >
      (const complex<float>& s, const vertex<complex<float> >& v);
    
    template vertex<complex<double> > operator*<complex<double> >
      (const complex<double>& s, const vertex<complex<double> >& v);
    
    //template vertex<int> operator*<int>(const int& s, const vertex<int>& v);
    //template vertex<char> operator*<char>(const char& s, const vertex<char>& v);
    //template vertex<unsigned int> operator*<unsigned int>(const unsigned int& s, const vertex<unsigned int>& v);
    //template vertex<unsigned char> operator*<unsigned char>(const unsigned char& s, const vertex<unsigned char>& v);
      
    template vertex<blas_real<float> > operator*<blas_real<float> >
      (const blas_real<float>& s, const vertex<blas_real<float> >& v);
									     
    template vertex<blas_real<double> > operator*<blas_real<double> >
      (const blas_real<double>& s, const vertex<blas_real<double> >& v);
    
    template vertex<blas_complex<float> > operator*<blas_complex<float> >
      (const blas_complex<float>& s, const vertex<blas_complex<float> >& v);
    
    template vertex<blas_complex<double> > operator*<blas_complex<double> >
      (const blas_complex<double>& s, const vertex<blas_complex<double> >& v);

    template vertex<superresolution<blas_real<float>, modular<unsigned int> > > operator*<superresolution<blas_real<float>, modular<unsigned int> > >
    (const superresolution<blas_real<float>, modular<unsigned int> >& s,
     const vertex<superresolution<blas_real<float>, modular<unsigned int> > >& v);
    
    template vertex<superresolution<blas_real<double>, modular<unsigned int> > > operator*<superresolution<blas_real<double>, modular<unsigned int> > >
    (const superresolution<blas_real<double>, modular<unsigned int> >& s,
     const vertex<superresolution<blas_real<double>, modular<unsigned int> > >& v);

    template vertex<superresolution<blas_complex<float>, modular<unsigned int> > > operator*<superresolution<blas_complex<float>, modular<unsigned int> > >
    (const superresolution<blas_complex<float>, modular<unsigned int> >& s,
     const vertex<superresolution<blas_complex<float>, modular<unsigned int> > >& v);
    
    template vertex<superresolution<blas_complex<double>, modular<unsigned int> > > operator*<superresolution<blas_complex<double>, modular<unsigned int> > >
    (const superresolution<blas_complex<double>, modular<unsigned int> >& s,
     const vertex<superresolution<blas_complex<double>, modular<unsigned int> > >& v);

       
    
    template std::ostream& operator<< <float>(std::ostream& ios, const vertex<float>&);
    template std::ostream& operator<< <double>(std::ostream& ios, const vertex<double>&);
    template std::ostream& operator<< <complex<float> >(std::ostream& ios, const vertex<complex<float> >&);
    template std::ostream& operator<< <complex<double> >(std::ostream& ios, const vertex<complex<double> >&);
    
    template std::ostream& operator<< <int>(std::ostream& ios, const vertex<int>&);
    template std::ostream& operator<< <char>(std::ostream& ios, const vertex<char>&);
    template std::ostream& operator<< <unsigned int>(std::ostream& ios, const vertex<unsigned int>&);
    template std::ostream& operator<< <unsigned char>(std::ostream& ios, const vertex<unsigned char>&);
    
    template std::ostream& operator<< <blas_real<float> >(std::ostream& ios, const vertex<blas_real<float> >&);
    template std::ostream& operator<< <blas_real<double> >(std::ostream& ios, const vertex<blas_real<double> >&);
    template std::ostream& operator<< <blas_complex<float> >(std::ostream& ios, const vertex<blas_complex<float> >&);
    template std::ostream& operator<< <blas_complex<double> >(std::ostream& ios, const vertex<blas_complex<double> >&);

    template std::ostream& operator<< <superresolution<blas_real<float>, modular<unsigned int> > >(std::ostream& ios, const vertex<superresolution<blas_real<float>, modular<unsigned int> > >&);
    template std::ostream& operator<< <superresolution<blas_real<double>, modular<unsigned int> > >(std::ostream& ios, const vertex<superresolution<blas_real<double>, modular<unsigned int> > >&);


    template std::ostream& operator<< <superresolution<blas_complex<float>, modular<unsigned int> > >(std::ostream& ios, const vertex<superresolution<blas_complex<float>, modular<unsigned int> > >&);
    template std::ostream& operator<< <superresolution<blas_complex<double>, modular<unsigned int> > >(std::ostream& ios, const vertex<superresolution<blas_complex<double>, modular<unsigned int> > >&);
    
  };
};




#endif
