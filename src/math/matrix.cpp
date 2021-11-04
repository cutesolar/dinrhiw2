
#ifndef matrix_cpp
#define matrix_cpp

#include "matrix.h"
#include "gcd.h"
#include "eig.h"
#include "norms.h"
#include "blade_math.h"
#include "Log.h"

#include <list>
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <typeinfo>

#include <string.h>
#include <stdlib.h>

using namespace std;


namespace whiteice
{
  namespace math
  {
    
    template <typename T>
    matrix<T>::matrix(const unsigned int ysize,
		      const unsigned int xsize)
    {
      numRows = 0;
      numCols = 0;
      data = NULL;
      compressor = NULL;

#ifdef CUBLAS
      
      cudaError_t cudaStat;
      void* cudaptr = NULL;
      cudaStat = cudaMallocManaged(&cudaptr, ysize*xsize*sizeof(T));

      if(cudaStat != cudaSuccess || cudaptr == NULL){
	whiteice::logging.error("matrix ctor failed. cuBLAS memory allocation failure.");
	throw CUDAException("CUBLAS memory allocation failure (matrix).");
      }

      // no memory initialization!!
      
      this->data = (T*)cudaptr;

      numRows = ysize;
      numCols = xsize;
      
#else

#ifdef BLAS_MEMALIGN
      // NOTE: electric fence don't know about posix_memalign()
      posix_memalign((void**)&data,
		     (8/whiteice::gcd<unsigned int>(8,sizeof(void*)))*sizeof(void*),
		     ysize*xsize*sizeof(T));
#else
      data = (T*)malloc(ysize*xsize*sizeof(T));
#endif
      
      if(!data) throw std::bad_alloc();
      
      memset(data, 0, ysize*xsize*sizeof(T));
      
      numRows = ysize;
      numCols = xsize;
      compressor = 0;
#endif
      
    }
    
    
    template <typename T>
    matrix<T>::matrix(const matrix<T>& M)
    {
      data = NULL;
      compressor = NULL;

      if(M.compressor != 0){
	assert(0);
	throw illegal_operation("matrix ctor: to be copied matrix is compressed");
      }
      
#if CUBLAS
      
      if(M.data){
	cudaError_t cudaErr;
	cublasStatus_t cudaStat;
	void* cudaptr = NULL;
	cudaErr = cudaMallocManaged(&cudaptr, M.numRows*M.numCols*sizeof(T));
	
	if(cudaErr != cudaSuccess || cudaptr == NULL){
	  whiteice::logging.error("matrix ctor failed. cuBLAS memory allocation failure.");
	  throw CUDAException("CUBLAS memory allocation failure (matrix).");
	}
	
	// cuda copy memory
	if(typeid(T) == typeid(blas_real<float>)){
	  cudaStat = cublasScopy(cublas_handle, M.numRows*M.numCols,
				 (const float*)M.data, 1, (float*)cudaptr, 1);
	  gpu_sync();
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    cudaFree(cudaptr);
	    whiteice::logging.error("matrix ctor failed. cublasScopy() failed.");
	    throw CUDAException("CUBLAS cublasScopy() failed.");
	  }
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  cudaStat = cublasDcopy(cublas_handle, M.numRows*M.numCols,
				 (const double*)M.data, 1, (double*)cudaptr, 1);
	  gpu_sync();
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    cudaFree(cudaptr);
	    whiteice::logging.error("matrix ctor failed. cublasDcopy() failed.");
	    throw CUDAException("CUBLAS cublasDcopy() failed.");
	  }
	}
	else if(typeid(T) == typeid(blas_complex<float>)){
	  cudaStat = cublasCcopy(cublas_handle, M.numRows*M.numCols,
				 (const cuComplex*)M.data, 1, (cuComplex*)cudaptr, 1);
	  gpu_sync();
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    cudaFree(cudaptr);
	    whiteice::logging.error("matrix ctor failed. cublasCcopy() failed.");
	    throw CUDAException("CUBLAS cublasCcopy() failed.");
	  }
	}
	else if(typeid(T) == typeid(blas_complex<double>)){
	  cudaStat = cublasZcopy(cublas_handle, M.numRows*M.numCols,
				 (const cuDoubleComplex*)M.data, 1,
				 (cuDoubleComplex*)cudaptr, 1);
	  gpu_sync();
	  if(cudaStat != CUBLAS_STATUS_SUCCESS){
	    cudaFree(cudaptr);
	    whiteice::logging.error("matrix ctor failed. cublasZcopy() failed.");
	    throw CUDAException("CUBLAS cublasZcopy() failed.");
	  }
	}
	else{
	  // generic memcopy [assumes type T does not allocated memory dynamically]
	  auto s = cudaMemcpy(cudaptr, M.data, M.numRows*M.numCols*sizeof(T),
			      cudaMemcpyDeviceToDevice);
	  gpu_sync();
	  if(s != cudaSuccess){
	    whiteice::logging.error("matrix ctor failed. cudaMemcopy() failed.");
	    cudaFree(cudaptr);
	    throw CUDAException("CUBLAS cudaMemcpy() failed.");
	  }
	  
	}

	// memory initialization successful
	this->data = (T*)cudaptr;

	this->numRows = M.numRows;
	this->numCols = M.numCols;
      }
      
#else

#ifdef BLAS_MEMALIGN
      // electric fence don't know about posix_memalign()
      posix_memalign((void**)&data,
		     (8/whiteice::gcd<unsigned int>(8,sizeof(void*)))*sizeof(void*),
		     M.numRows*M.numCols*sizeof(T));
#else
      data = (T*)malloc(M.numRows*M.numCols*sizeof(T));
#endif
      
      if(!data) throw std::bad_alloc();
      
      
      if(typeid(T) == typeid(blas_real<float>)){
	cblas_scopy(M.numRows*M.numCols,
		    (const float*)M.data, 1,
		    (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cblas_ccopy(M.numRows*M.numCols,
		    (const float*)M.data, 1,
		    (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dcopy(M.numRows*M.numCols,
		    (const double*)M.data, 1,
		    (double*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cblas_zcopy(M.numRows*M.numCols,
		    (const double*)M.data, 1,
		    (double*)data, 1);
      }
      else{ // generic memcpy
	memcpy(data, M.data, M.numRows*M.numCols*sizeof(T));
      }
      
      
      numRows = M.numRows;
      numCols = M.numCols;
      
      compressor = M.compressor;
#endif
    }
    
    template <typename T>
    matrix<T>::matrix(matrix<T>&& t)
    {
      this->data = NULL;
      this->numRows = 0;
      this->numCols = 0;
      this->compressor = NULL;
      
      std::swap(this->data, t.data);
      std::swap(this->numRows, t.numRows);
      std::swap(this->numCols, t.numCols);
      std::swap(this->compressor, t.compressor);
    }
    
    template <typename T>
    matrix<T>::matrix(const vertex<T>& diagonal)
    {
      data = NULL;
      compressor = NULL;

#ifdef CUBLAS
      
      cudaError_t cudaErr;
      void* cudaptr = NULL;
      cudaErr = cudaMallocManaged(&cudaptr,
				  diagonal.dataSize*diagonal.dataSize*sizeof(T));
      
      if(cudaErr != cudaSuccess || cudaptr == NULL){
	whiteice::logging.error("matrix ctor failed. cudaMallocManaged() failed.");
	throw CUDAException("CUBLAS memory allocation failure (matrix).");
      }

      this->data = (T*)cudaptr;
      this->numCols = diagonal.dataSize;
      this->numRows = diagonal.dataSize;

      this->zero(); // cuBLAS optimized

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<diagonal.dataSize;i++){
	data[i + i*numRows] = diagonal[i];
      }
      
#else

#ifdef BLAS_MEMALIGN
      // electric fence don't know about posix_memalign()
      posix_memalign((void**)&data,
		     (8/whiteice::gcd<unsigned int>(8,sizeof(void*)))*sizeof(void*),
		     diagonal.size()*diagonal.size()*sizeof(T));
#else
      data = (T*)malloc(diagonal.size()*diagonal.size()*sizeof(T));
#endif
      
      if(!data) throw std::bad_alloc();
      
      memset(data, 0, diagonal.size()*diagonal.size());
      numCols = diagonal.size();
      numRows = diagonal.size();
      
      unsigned int index = 0;
      for(unsigned int i=0;i<diagonal.size();i++){
	data[index] = diagonal[i];
	index += numCols + 1;
      }
#endif 
      
    }
    
    
    template <typename T>
    matrix<T>::~matrix()
    {
      if(compressor) delete compressor;

#ifdef CUBLAS
      
      if(data)
	cudaFree(data);
      
#else
      if(data) free(data);
#endif
      
    }

    /**********************************************************************/
    
    
    template <typename T>
    matrix<T> matrix<T>::operator+(const matrix<T>& M) const
      
    {
      if(M.numCols != numCols || M.numRows != numRows){
	whiteice::logging.error("matrix::operator+(): matrix dimensions mismatch.");
	assert(0);
	throw illegal_operation("'+' operator: matrix size mismatch");
      }

#ifdef CUBLAS
      
      matrix<T> R(M.numRows, M.numCols);

      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(1.0f);

	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const float*)&alpha,
				       (const float*)(this->data), this->numRows,
				       (const float*)&alpha,
				       (const float*)(M.data), M.numRows,
				       (float*)(R.data), R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator+(): cublassSgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasSgeam() failed.");
	}
				       
	return R;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(1.0);

	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const double*)&alpha,
				       (const double*)(this->data), this->numRows,
				       (const double*)&alpha,
				       (const double*)(M.data), M.numRows,
				       (double*)(R.data), R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator+(): cublassDgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasDgeam() failed.");
	}
				       
	return R;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(1.0f);

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(this->data), this->numRows,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(M.data), M.numRows,
				       (cuComplex*)(R.data), R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator+(): cublassCgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasCgeam() failed.");
	}
				       
	return R;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(1.0f);
	
	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(this->data), this->numRows,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(M.data), M.numRows,
				       (cuDoubleComplex*)(R.data), R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator+(): cublassZgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasZgeam() failed.");
	}
				       
	return R;
      }
      else{
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	  R.data[i] = data[i] + M.data[i];

	return R;
      }

#else
      matrix<T> R(M.numRows, M.numCols);
      
#pragma omp parallel for schedule(auto)      
      for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	R.data[i] = data[i] + M.data[i];

      return R;
      
#endif
      
    }
    
    
    template <typename T>
    matrix<T> matrix<T>::operator-(const matrix<T>& M) const
      
    {
      if(M.numCols != numCols || M.numRows != numRows){
	whiteice::logging.error("matrix::operator-(): matrix dimensions mismatch.");
	assert(0);
	throw illegal_operation("'-' operator: matrix size mismatch");
      }

#ifdef CUBLAS
      
      matrix<T> R(M.numRows, M.numCols);

      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(1.0f);
	const T beta = T(-1.0f);

	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const float*)&alpha,
				       (const float*)(this->data), this->numRows,
				       (const float*)&beta,
				       (const float*)(M.data), M.numRows,
				       (float*)(R.data), R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator-(): cublassSgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasSgeam() failed.");
	}
				       
	return R;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(1.0);
	const T beta = T(-1.0);

	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const double*)&alpha,
				       (const double*)(this->data), this->numRows,
				       (const double*)&beta,
				       (const double*)(M.data), M.numRows,
				       (double*)(R.data), R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator-(): cublassDgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasDgeam() failed.");
	}
				       
	return R;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(1.0f);
	const T beta = T(-1.0);

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(this->data), this->numRows,
				       (const cuComplex*)&beta,
				       (const cuComplex*)(M.data), M.numRows,
				       (cuComplex*)(R.data), R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator-(): cublassCgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasCgeam() failed.");
	}
				       
	return R;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(+1.0);
	const T beta = T(-1.0);
	
	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(this->data), this->numRows,
				       (const cuDoubleComplex*)&beta,
				       (const cuDoubleComplex*)(M.data), M.numRows,
				       (cuDoubleComplex*)(R.data), R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator-(): cublassZgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasCgeam() failed.");
	}
				       
	return R;
      }
      else{
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	  R.data[i] = data[i] - M.data[i];

	return R;
      }

#else
      
      matrix<T> R(M.numRows, M.numCols);

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	R.data[i] = data[i] - M.data[i];
      
      return R;
      
#endif
    }
    
    
    template <typename T>
    matrix<T> matrix<T>::operator*(const matrix<T>& M) const
      
    {	
      if(numCols != M.numRows){
	whiteice::logging.error("matrix::operator*(): matrix dimensions mismatch.");
	assert(0);
	throw illegal_operation("'*' operator: matrix size mismatch");
      }

#ifdef CUBLAS

      matrix<T> R(numRows, M.numCols);

      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	auto s = cublasSgemm(cublas_handle,
			     CUBLAS_OP_N, CUBLAS_OP_N,
			     numRows, M.numCols, numCols,
			     (const float*)&alpha,
			     (const float*)(this->data), numRows,
			     (const float*)(M.data), M.numRows,
			     (const float*)&beta,
			     (float*)R.data, R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*(): cublasSgemm() failed.");
	  throw CUDAException("CUBLAS cublasSgemm() failed.");
	}
	
	return R;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	auto s = cublasDgemm(cublas_handle,
			     CUBLAS_OP_N, CUBLAS_OP_N,
			     numRows, M.numCols, numCols,
			     (const double*)&alpha,
			     (const double*)(this->data), numRows,
			     (const double*)(M.data), M.numRows,
			     (const double*)&beta,
			     (double*)R.data, R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*(): cublasDgemm() failed.");
	  throw CUDAException("CUBLAS cublasDgemm() failed.");
	}
	
	return R;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	auto s = cublasCgemm(cublas_handle,
			     CUBLAS_OP_N, CUBLAS_OP_N,
			     numRows, M.numCols, numCols,
			     (const cuComplex*)&alpha,
			     (const cuComplex*)(this->data), numRows,
			     (const cuComplex*)(M.data), M.numRows,
			     (const cuComplex*)&beta,
			     (cuComplex*)R.data, R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*(): cublasCgemm() failed.");
	  throw CUDAException("CUBLAS cublasCgemm() failed.");
	}
	
	return R;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);
	
	auto s = cublasZgemm(cublas_handle,
			     CUBLAS_OP_N, CUBLAS_OP_N,
			     numRows, M.numCols, numCols,
			     (const cuDoubleComplex*)&alpha,
			     (const cuDoubleComplex*)(this->data), numRows,
			     (const cuDoubleComplex*)(M.data), M.numRows,
			     (const cuDoubleComplex*)&beta,
			     (cuDoubleComplex*)R.data, R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*(): cublasZgemm() failed.");
	  throw CUDAException("CUBLAS cublasZgemm() failed.");
	}
	
	return R;
      }
      else{
	// generic matrix multiplication (SLOW)
	
	R.zero(); // zero matrix
	
	for(unsigned int i=0;i<R.numCols;i++)
	  for(unsigned int j=0;j<R.numRows;j++)
	    for(unsigned int k=0;k<numCols;k++)
	      R(j,i) += (*this)(j,k) * M(k,i);
	
	return R;
      }

#else
      // uses cblas_Xgemm optimized matrix multiplication
      
      matrix<T> R(numRows, M.numCols);
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    1.0f, (float*)data, numCols, (float*)M.data, M.numCols,
		    0.0f, (float*)R.data, R.numCols);
	return R;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	blas_complex<float> a, b;
	a = 1.0f; b = 0.0f;
	
	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    (float*)(&a), (float*)data, numCols, (float*)M.data, M.numCols,
		    (float*)(&b), (float*)R.data, R.numCols);
	return R;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    1.0, (double*)data, numCols, (double*)M.data, M.numCols,
		    0.0, (double*)R.data, R.numCols);
	return R;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	blas_complex<double> a, b;
	a = 1.0; b = 0.0;
	
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    (double*)(&a), (double*)data, numCols, (double*)M.data, M.numCols,
		    (double*)(&b), (double*)R.data, R.numCols);
	return R;
      }
      else{ // generic matrix multiplication
	
	// zero matrix	
	
	unsigned int rindex = 0;
	for(unsigned int j=0;j<numRows;j++){
	  for(unsigned int i=0;i<M.numCols;i++,rindex++){
	    for(unsigned int k=0;k<M.numRows;k++){
	      R[rindex] += 
		data[j*numCols + k] * M[k*M.numCols + i];
	    }
	  }
	}
      	
	
	return R;
      }
#endif
    }
    
    
    
    template <typename T>
    matrix<T> matrix<T>::operator/(const matrix<T>& M) const
      
    {
      matrix<T> R(*this);
      R /= M;
      return R;
    }
    
    
    template <typename T>
    matrix<T> matrix<T>::operator!() const {
      whiteice::logging.error("matrix::operator!(): illegal operation called.");
      assert(0);
      throw illegal_operation("'!'-operator");
    }
    
    
    template <typename T>
    matrix<T> matrix<T>::operator-() const
      
    {
#ifdef CUBLAS
      matrix<T> M(numRows, numCols);
      
      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(-1.0f);
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const float*)&alpha,
				       (const float*)(this->data), this->numRows,
				       (const float*)&beta,
				       (const float*)NULL, M.numRows,
				       (float*)(M.data), M.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator-(): cublassSgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasSgeam() failed.");
	}
				       
	return M;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(-1.0);
	const T beta  = T(0.0);

	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const double*)&alpha,
				       (const double*)(this->data), this->numRows,
				       (const double*)&beta,
				       (const double*)(NULL), M.numRows,
				       (double*)(M.data), M.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator-(): cublassDgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasDgeam() failed.");
	}
				       
	return M;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(-1.0f);
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(this->data), this->numRows,
				       (const cuComplex*)&beta,
				       (const cuComplex*)(NULL), M.numRows,
				       (cuComplex*)(M.data), M.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator-(): cublassCgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasCgeam() failed.");
	}
				       
	return M;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(-1.0);
	const T beta  = T(0.0);
	
	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(this->data), this->numRows,
				       (const cuDoubleComplex*)&beta,
				       (const cuDoubleComplex*)(NULL), M.numRows,
				       (cuDoubleComplex*)(M.data), M.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator-(): cublassZgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasZgeam() failed.");
	}
				       
	return M;
      }
      else{
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	  M.data[i] = -data[i];

	return M;
      }
      
#else
      matrix<T> M(numRows, numCols);

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<numRows*numCols;i++)
	M.data[i] = -data[i];
      
      return M;
#endif
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::operator+=(const matrix<T>& M)      
    {
      if(M.numCols != numCols || M.numRows != numRows){
	whiteice::logging.error("matrix::operator+=(): matrix dimensions mismatch.");
	assert(0);
	throw illegal_operation("'+=' operator: matrix size mismatch");
      }

#ifdef CUBLAS

      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(1.0f);

	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const float*)&alpha,
				       (const float*)(this->data), this->numRows,
				       (const float*)&beta,
				       (const float*)(M.data), M.numRows,
				       (float*)(this->data), this->numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator+=(): cublasSgeam() failed.");
	  throw CUDAException("CUBLAS cublasSgeam() failed.");
	}
				       
	return (*this);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(1.0f);
	const T beta  = T(1.0f);

	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const double*)&alpha,
				       (const double*)(this->data), this->numRows,
				       (const double*)&beta,
				       (const double*)(M.data), M.numRows,
				       (double*)(this->data), this->numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator+=(): cublasDgeam() failed.");
	  throw CUDAException("CUBLAS cublasDgeam() failed.");
	}
				       
	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(1.0f);

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(this->data), this->numRows,
				       (const cuComplex*)&beta,
				       (const cuComplex*)(M.data), M.numRows,
				       (cuComplex*)(this->data), this->numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator+=(): cublasCgeam() failed.");
	  throw CUDAException("CUBLAS cublasCgeam() failed.");
	}
				       
	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(1.0f);
	const T beta  = T(1.0f);

	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(this->data), this->numRows,
				       (const cuDoubleComplex*)&beta,
				       (const cuDoubleComplex*)(M.data), M.numRows,
				       (cuDoubleComplex*)(this->data), this->numRows);
	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator+=(): cublasZgeam() failed.");
	  throw CUDAException("CUBLAS cublasZgeam() failed.");
	}
	
	return (*this);
      }
      else{
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	  data[i] += M.data[i];

	return (*this);
      }
      
#else

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	data[i] += M.data[i];
      
      return (*this);
#endif
    }
  
    
    template <typename T>
    matrix<T>& matrix<T>::operator-=(const matrix<T>& M)
      
    {
      if(M.numCols != numCols || M.numRows != numRows){
	whiteice::logging.error("matrix::operator-=(): matrix dimensions mismatch.");
	assert(0);
	throw illegal_operation("'-=' operator: matrix size mismatch");
      }

#ifdef CUBLAS
      
      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(-1.0f);

	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const float*)&alpha,
				       (const float*)(this->data), this->numRows,
				       (const float*)&beta,
				       (const float*)(M.data), M.numRows,
				       (float*)(this->data), this->numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator-=(): cublasSgeam() failed.");
	  throw CUDAException("CUBLAS cublasSgeam() failed.");
	}
				       
	return (*this);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(1.0);
	const T beta  = T(-1.0);

	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const double*)&alpha,
				       (const double*)(this->data), this->numRows,
				       (const double*)&beta,
				       (const double*)(M.data), M.numRows,
				       (double*)(this->data), this->numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator-=(): cublasDgeam() failed.");
	  throw CUDAException("CUBLAS cublasDgeam() failed.");
	}
				       
	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(-1.0f);

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(this->data), this->numRows,
				       (const cuComplex*)&beta,
				       (const cuComplex*)(M.data), M.numRows,
				       (cuComplex*)(this->data), this->numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator-=(): cublasCgeam() failed.");
	  throw CUDAException("CUBLAS cublasCgeam() failed.");
	}
				       
	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(1.0);
	const T beta  = T(-1.0);

	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(this->data), this->numRows,
				       (const cuDoubleComplex*)&beta,
				       (const cuDoubleComplex*)(M.data), M.numRows,
				       (cuDoubleComplex*)(this->data), this->numRows);
	gpu_sync();
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator-=(): cublasZgeam() failed.");
	  throw CUDAException("CUBLAS cublasZgeam() failed.");
	}
	
	return (*this);
      }
      else{
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	  data[i] -= M.data[i];

	return (*this);
      }
      
#else
#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	data[i] -= M.data[i];
      
      return (*this);
#endif
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::operator*=(const matrix<T>& M)
      
    {
      if(numCols != M.numRows){
	whiteice::logging.error("matrix::operator*=(): matrix dimensions mismatch.");
	assert(0);
	throw illegal_operation("'*=' operator: matrix size mismatch");
      }

#if CUBLAS

      // R = alpha*(*this) + beta*M
      matrix<T> R(numRows, M.numCols);

      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(1.0f);

	auto s = cublasSgemm(cublas_handle,
			     CUBLAS_OP_N, CUBLAS_OP_N,
			     numRows, M.numCols, numCols,
			     (const float*)&alpha,
			     (const float*)(this->data), numRows,
			     (const float*)(M.data), M.numRows,
			     (const float*)&beta,
			     (float*)R.data, R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*=(): cublasSgemm() failed.");
	  throw CUDAException("CUBLAS cublasSgemm() failed.");
	}

	(*this) = R;
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(1.0f);
	const T beta  = T(1.0f);

	auto s = cublasDgemm(cublas_handle,
			     CUBLAS_OP_N, CUBLAS_OP_N,
			     numRows, M.numCols, numCols,
			     (const double*)&alpha,
			     (const double*)(this->data), numRows,
			     (const double*)(M.data), M.numRows,
			     (const double*)&beta,
			     (double*)R.data, R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*=(): cublasDgemm() failed.");
	  throw CUDAException("CUBLAS cublasDgemm() failed.");
	}

	(*this) = R;
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(1.0f);

	auto s = cublasCgemm(cublas_handle,
			     CUBLAS_OP_N, CUBLAS_OP_N,
			     numRows, M.numCols, numCols,
			     (const cuComplex*)&alpha,
			     (const cuComplex*)(this->data), numRows,
			     (const cuComplex*)(M.data), M.numRows,
			     (const cuComplex*)&beta,
			     (cuComplex*)R.data, R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*=(): cublasCgemm() failed.");
	  throw CUDAException("CUBLAS cublasCgemm() failed.");
	}

	(*this) = R;
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(1.0);
	const T beta  = T(1.0);
	
	auto s = cublasZgemm(cublas_handle,
			     CUBLAS_OP_N, CUBLAS_OP_N,
			     numRows, M.numCols, numCols,
			     (const cuDoubleComplex*)&alpha,
			     (const cuDoubleComplex*)(this->data), numRows,
			     (const cuDoubleComplex*)(M.data), M.numRows,
			     (const cuDoubleComplex*)&beta,
			     (cuDoubleComplex*)R.data, R.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*=(): cublasZgemm() failed.");
	  throw CUDAException("CUBLAS cublasZgemm() failed.");
	}

	(*this) = R;
	
	return (*this);
      }
      else{
	// generic matrix multiplication (SLOW)
	
	R.zero(); // zero matrix
	
	for(unsigned int i=0;i<R.numCols;i++)
	  for(unsigned int j=0;j<R.numRows;j++)
	    for(unsigned int k=0;k<numCols;k++)
	      R(j,i) += (*this)(j,k) * M(k,i);

	(*this) = R;

	return (*this);
      }
      
#else
      // uses cblas_Xgemm optimized matrix multiplication
      
      matrix<T> R(numRows, M.numCols);
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    1.0f, (float*)data, numCols,
		    (float*)M.data, M.numCols,
		    0.0f, (float*)R.data, R.numCols);
	
	// if(!resize_x(R.numCols)) throw bad_alloc();
	// memcpy(data, R.data, numRows*numCols*sizeof(T));
	
	this->numCols = M.numCols;
	free(data);
	data = R.data;
	R.data = nullptr;
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	blas_complex<float> a, b;
	a = 1.0f; b = 0.0f;
	
	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    (float*)(&a), (float*)data, numCols, (float*)M.data, M.numCols,
		    (float*)(&b), (float*)R.data, R.numCols);
	
	if(!resize_x(R.numCols)) throw bad_alloc();
	memcpy(data, R.data, numRows*numCols*sizeof(T));
	
	return (*this);	
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    1.0, (double*)data, numCols, (double*)M.data, M.numCols,
		    0.0, (double*)R.data, R.numCols);
	
	
	if(!resize_x(R.numCols)) throw bad_alloc();
	memcpy(data, R.data, numRows*numCols*sizeof(T));
	
	return (*this);	
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	blas_complex<double> a, b;
	a = 1.0; b = 0.0;	
	
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    (double*)(&a), (double*)data, numCols, (double*)M.data, M.numCols,
		    (double*)(&b), (double*)R.data, R.numCols);
	
	if(!resize_x(R.numCols)) throw bad_alloc();
	memcpy(data, R.data, numRows*numCols*sizeof(T));
	
	return (*this);	
      }
      else{ // generic matrix multiplication
	
	// zero matrix	
	
	unsigned int rindex = 0;
	for(unsigned int j=0;j<numRows;j++){
	  for(unsigned int i=0;i<M.numCols;i++,rindex++){
	    for(unsigned int k=0;k<M.numRows;k++){
	      R[rindex] += 
		data[j*numCols + k] * M[k*M.numCols + i];
	    }
	  }
	}
      	
	
	if(!resize_x(R.numCols)) throw bad_alloc();
	memcpy(data, R.data, numRows*numCols*sizeof(T));
	
	return (*this);
      }
#endif
    }
    
    
    
    template <typename T>
    matrix<T>& matrix<T>::operator/=(const matrix<T>& M) 
    {
      // C BLAS OPTIMIZE ("/" operator, too)
      matrix<T> N(M);

      if(!N.inv()){
	whiteice::logging.error("matrix::operator/=() failed. singular matrix.");
	assert(0);
	throw illegal_operation("Cannot 'divide' with singular matrix");
      }
      
      (*this) *= N;
      
      return (*this);
    }
    

    template <typename T>
    matrix<T>& matrix<T>::operator=(const matrix<T>& M) 
    {
      if(this == &M) return (*this); // self-assignment
      
      if(M.compressor != 0){
	assert(0);
	throw illegal_operation("matrix ctor: to be copied matrix is compressed");
      }
      
      if(M.numCols != numCols) resize_x(M.numCols);
      if(M.numRows != numRows) resize_y(M.numRows);
      
#ifdef CUBLAS
      // copies matrix using cublas*geam() calls as recommended
      // in NVIDIA cuBLAS documentation (alpha=1, beta=0)
      
      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const float*)&alpha,
				       (const float*)(M.data), numRows,
				       (const float*)&beta,
				       (const float*)NULL, numRows,
				       (float*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator=() failed. cuBlasSgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasSgeam() failed.");
	}
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const double*)&alpha,
				       (const double*)(M.data), numRows,
				       (const double*)&beta,
				       (const double*)NULL, numRows,
				       (double*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator=() failed. cuBlasDgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasDgeam() failed.");
	}
	
	return (*this);
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(M.data), numRows,
				       (const cuComplex*)&beta,
				       (const cuComplex*)NULL, numRows,
				       (cuComplex*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator=() failed. cuBlasCgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasCgeam() failed.");
	}

	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(1.0);
	const T beta  = T(0.0);
	
	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(M.data), numRows,
				       (const cuDoubleComplex*)&beta,
				       (const cuDoubleComplex*)NULL, numRows,
				       (cuDoubleComplex*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator=() failed. cuBlasZgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasZgeam() failed.");
	}
	
	return (*this);
      }
      else{

	auto e = cudaMemcpy(data, M.data, M.numCols*M.numRows*sizeof(T),
			    cudaMemcpyDeviceToDevice);
	
	gpu_sync();

	if(e != cudaSuccess){
	  whiteice::logging.error("matrix::operator=() failed. cudaMemcpy() failed.");
	  throw CUDAException("CUBLS cudaMemcpy() failed.");
	}
	
	return (*this);
      }
      
#else
      
      if(this != &M){ // no self-assignment
	if(M.numCols != numCols) resize_x(M.numCols);
	if(M.numRows != numRows) resize_y(M.numRows);
	
	memcpy(data, M.data, sizeof(T)*numCols*numRows);
      }
      
      return (*this);
#endif
    }
    

    template <typename T>
    matrix<T>& matrix<T>::operator=(matrix<T>&& t) 
    {
      if(this == &t) return *this; // self-assignment

      std::swap(this->data, t.data);
      std::swap(this->numRows, t.numRows);
      std::swap(this->numCols, t.numCols);
      std::swap(this->compressor, t.compressor);
      
      return *this;
    }


    template <typename T>
    matrix<T>& matrix<T>::operator=(const vertex<T>& v)
    {
      
      if(v.compressor != NULL || this->compressor != NULL){
	assert(0);
	throw illegal_operation("matrix::operator=(): to be copied matrix/vertex is compressed");
      }

      if(this->resize_x(1) == false){
	whiteice::logging.error("matrix::operator=(): matrix resize_x() failed.");
	assert(0);
	throw illegal_operation("matrix::operator=(): matrix resize_x() failed.");
      }
      
      if(this->resize_y(v.dataSize) == false){
	whiteice::logging.error("matrix::operator=(): matrix resize_y() failed.");
	assert(0);
	throw illegal_operation("matrix::operator=(): matrix resize_y() failed.");
      }
      
#ifdef CUBLAS
      // copies matrix using cublas*geam() calls as recommended
      // in NVIDIA cuBLAS documentation (alpha=1, beta=0)
      
      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const float*)&alpha,
				       (const float*)(v.data), numRows,
				       (const float*)&beta,
				       (const float*)NULL, numRows,
				       (float*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator=() failed. cuBlasSgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasSgeam() failed.");
	}
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const double*)&alpha,
				       (const double*)(v.data), numRows,
				       (const double*)&beta,
				       (const double*)NULL, numRows,
				       (double*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator=() failed. cuBlasDgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasDgeam() failed.");
	}
	
	return (*this);
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(v.data), numRows,
				       (const cuComplex*)&beta,
				       (const cuComplex*)NULL, numRows,
				       (cuComplex*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator=() failed. cuBlasCgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasCgeam() failed.");
	}

	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(1.0);
	const T beta  = T(0.0);
	
	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(v.data), numRows,
				       (const cuDoubleComplex*)&beta,
				       (const cuDoubleComplex*)NULL, numRows,
				       (cuDoubleComplex*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator=() failed. cuBlasZgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasZgeam() failed.");
	}
	
	return (*this);
      }
      else{

	auto e = cudaMemcpy(data, v.data, (this->numCols)*(this->numRows)*sizeof(T),
			    cudaMemcpyDeviceToDevice);
	
	gpu_sync();

	if(e != cudaSuccess){
	  whiteice::logging.error("matrix::operator=() failed. cudaMemcpy() failed.");
	  throw CUDAException("CUBLS cudaMemcpy() failed.");
	}
	
	return (*this);
      }
      
#else
      
      memcpy(data, v.data, sizeof(T)*numCols*numRows);
      
      return (*this);
#endif
    }
    
    
    template <typename T>
    bool matrix<T>::operator==(const matrix<T>& M) const
      
    {
      if(M.numCols != numCols || M.numRows != numRows)
	return false; // throw illegal_operation("'==' operator: matrix size mismatch");
      
      if(typeid(T) == typeid(blas_real<float>)    ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>)   ||
	 typeid(T) == typeid(blas_complex<double>)){
	
	return (memcmp(data, M.data, numCols*numRows*sizeof(T)) == 0);
      }
      else{
	for(unsigned int i=0;i<numCols*numRows;i++)
	  if(data[i] != M.data[i]) return false;
      }
      
      return true;
    }
    
    
    template <typename T>
    bool matrix<T>::operator!=(const matrix<T>& M) const
      
    {
      if(M.numCols != numCols || M.numRows != numRows)
	return true; // throw illegal_operation("'!=' operator: matrix size mismatch");
      
      if(numCols*numRows <= 0) return false;
      
      if(typeid(T) == typeid(blas_real<float>)    ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>)   ||
	 typeid(T) == typeid(blas_complex<double>)){
	
	return (memcmp(data, M.data, numCols*numRows*sizeof(T)) != 0);
      }
      else{
	for(unsigned int i=0;i<numCols*numRows;i++)
	  if(data[i] == M.data[i]) return false;
      }
      
      return true;
    }
    
    
    template <typename T>
    bool matrix<T>::operator>=(const matrix<T>& M) const
      
    {
      if(M.numCols != 1 || numCols != 1 || M.numRows != 1 || numRows != 1){
	assert(0);
	throw illegal_operation("matrix '>=': not a 1x1 matrix ");
      }
      
      return (data[0] < M.data[0]);
    }
    
    
    template <typename T>
    bool matrix<T>::operator<=(const matrix<T>& M) const
      
    {
      if(M.numCols != 1 || numCols != 1 || M.numRows != 1 || numRows != 1){
	assert(0);
	throw illegal_operation("matrix '<=': not a 1x1 matrix");
      }
      
      return (data[0] <= M.data[0]);
    }
    
    
    template <typename T>
    bool matrix<T>::operator< (const matrix<T>& M) const
      
    {
      if(M.numCols != 1 || numCols != 1 || M.numRows != 1 || numRows != 1){
	assert(0);
	throw illegal_operation("matrix  '<': not a 1x1 matrix");
      }
      
      return (data[0] < M.data[0]);
    }
    
  
    template <typename T>
    bool matrix<T>::operator> (const matrix<T>& M) const
      
    {
      if(M.numCols != 1 || numCols != 1 || M.numRows != 1 || numRows != 1){
	assert(0);
	throw illegal_operation("matrix  '>': not a 1x1 matrix");
      }
      
      return (data[0] > M.data[0]);
    }

    
    /***************************************************/
    
    
    template <typename T>
    matrix<T>& matrix<T>::operator=(const T& s) 
    {
      const unsigned int N = numRows*numCols;

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<N;i++)
	data[i] = s;
      
      return (*this);
    }
    
    
    
    template <typename T>
    matrix<T>  matrix<T>::operator* (const T& a) const 
    {
#ifdef CUBLAS
      // multiplies matrix using cublas*geam() calls as recommended
      // in NVIDIA cuBLAS documentation (alpha=a, beta=0)

      matrix<T> M(numRows, numCols);

      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = a;
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const float*)&alpha,
				       (const float*)(this->data), numRows,
				       (const float*)&beta,
				       (const float*)NULL, numRows,
				       (float*)(M.data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*() failed. cublasSgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasSgeam() failed.");
	}
	
	return M;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = a;
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const double*)&alpha,
				       (const double*)(this->data), numRows,
				       (const double*)&beta,
				       (const double*)NULL, numRows,
				       (double*)(M.data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*() failed. cublasDgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasDgeam() failed.");
	}
	
	return M;
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = a;
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(this->data), numRows,
				       (const cuComplex*)&beta,
				       (const cuComplex*)NULL, numRows,
				       (cuComplex*)(M.data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*() failed. cublasCgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasCgeam() failed.");
	}

	return M;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = a;
	const T beta  = T(0.0);
	
	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(this->data), numRows,
				       (const cuDoubleComplex*)&beta,
				       (const cuDoubleComplex*)NULL, numRows,
				       (cuDoubleComplex*)(M.data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*() failed. cublasZgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasZgeam() failed.");
	}
	
	return M;
      }
      else{
	// SLOW IMPLEMENTATION
	
#pragma omp parallel for schedule(auto)
	for(unsigned int index=0;index<(M.numCols*M.numRows);index++)
	  M[index] = a*(*this)[index];
	
	return M;
      }
      
#else
      matrix<T> M(numRows, numCols);
      const unsigned int MSIZE = numRows*numCols;
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_saxpy(MSIZE, *((float*)&a), (float*)data, 1, (float*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_caxpy(MSIZE, (const float*)&a, (float*)data, 1, (float*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_daxpy(MSIZE, *((double*)&a), (double*)data, 1, (double*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zaxpy(MSIZE, (const double*)&a, (double*)data, 1, (double*)M.data, 1);
      }
      else{ // "normal implementation"

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<MSIZE;i++)
	  M.data[i] = data[i]*a;
      }

      return M;
      
#endif
    }
    
    
    template <typename T>
    matrix<T> operator*(const T& a, const matrix<T>& M)
      
    {
#ifdef CUBLAS

      matrix<T> R(M.numRows, M.numCols);

      // multiplies matrix using cublas*geam() calls as recommended
      // in NVIDIA cuBLAS documentation (alpha=a, beta=0)

      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = a;
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const float*)&alpha,
				       (const float*)(M.data), M.numRows,
				       (const float*)&beta,
				       (const float*)NULL, M.numRows,
				       (float*)(R.data), M.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*() failed. cublasSgeam() failed.");
	  throw CUDAException("CUBLAS cublasSgeam() failed.");
	}
	
	return R;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = a;
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const double*)&alpha,
				       (const double*)(M.data), M.numRows,
				       (const double*)&beta,
				       (const double*)NULL, M.numRows,
				       (double*)(R.data), M.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*() failed. cublasDgeam() failed.");
	  throw CUDAException("CUBLAS cublasDgeam() failed.");
	}
	
	return R;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = a;
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(M.data), M.numRows,
				       (const cuComplex*)&beta,
				       (const cuComplex*)NULL, M.numRows,
				       (cuComplex*)(R.data), M.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*() failed. cublasCgeam() failed.");
	  throw CUDAException("CUBLAS cublasCgeam() failed.");
	}

	return R;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = a;
	const T beta  = T(0.0);
	
	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       M.numRows, M.numCols,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(M.data), M.numRows,
				       (const cuDoubleComplex*)&beta,
				       (const cuDoubleComplex*)NULL, M.numRows,
				       (cuDoubleComplex*)(R.data), M.numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*() failed. cublasZgeam() failed.");
	  throw CUDAException("CUBLAS cublasZgeam() failed.");
	}
	
	return R;
      }
      else{
	// SLOW IMPLEMENTATION
	
#pragma omp parallel for schedule(auto)
	for(unsigned int index=0;index<(M.numCols*M.numRows);index++)
	  R[index] = a*M[index];
	
	return R;
      }

#else
      matrix<T> R(M.numRows, M.numCols);
      const unsigned int MSIZE = M.numRows*M.numCols;
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_saxpy(MSIZE, *((float*)&a), (float*)M.data, 1, (float*)R.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_caxpy(MSIZE, (const float*)&a, (float*)M.data, 1, (float*)R.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_daxpy(MSIZE, *((double*)&a), (double*)M.data, 1, (double*)R.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zaxpy(MSIZE, (const double*)&a, (double*)M.data, 1, (double*)R.data, 1);
      }
      else{ // "normal implementation"

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<MSIZE;i++)
	  R.data[i] = M.data[i]*a;
      }
            
      return R;
#endif
    }
    
    
    template <typename T>
    matrix<T>  matrix<T>::operator/ (const T& s) const 
    {
#ifdef CUBLAS

      const T invs = T(1.0f)/s;
      return ((*this) * invs); // should call CUBLAS optimized code
      
#else      
      matrix<T> M(numRows, numCols);
      const unsigned int MSIZE = numRows*numCols;
      T ss = T(1.0)/s;
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_saxpy(MSIZE, *((float*)&ss), (float*)data, 1, (float*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_caxpy(MSIZE, (const float*)&ss, (float*)data, 1, (float*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_daxpy(MSIZE, *((double*)&ss), (double*)data, 1, (double*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zaxpy(MSIZE, (const double*)&ss, (double*)data, 1, (double*)M.data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<MSIZE;i++)
	  M.data[i] = data[i]*ss;
      }      
      
      return M;
#endif
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::operator*=(const T& a) 
    {

#ifdef CUBLAS

      // multiplies matrix using cublas*geam() calls as recommended
      // in NVIDIA cuBLAS documentation (alpha=a, beta=0)

      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = a;
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const float*)&alpha,
				       (const float*)(this->data), numRows,
				       (const float*)&beta,
				       (const float*)NULL, numRows,
				       (float*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*=() failed. cublasSgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasSgeam() failed.");
	}
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = a;
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const double*)&alpha,
				       (const double*)(this->data), numRows,
				       (const double*)&beta,
				       (const double*)NULL, numRows,
				       (double*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*=() failed. cublasDgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasDgeam() failed.");
	}
	
	return (*this);
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = a;
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(this->data), numRows,
				       (const cuComplex*)&beta,
				       (const cuComplex*)NULL, numRows,
				       (cuComplex*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*=() failed. cublasCgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasCgeam() failed.");
	}

	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = a;
	const T beta  = T(0.0);
	
	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(this->data), numRows,
				       (const cuDoubleComplex*)&beta,
				       (const cuDoubleComplex*)NULL, numRows,
				       (cuDoubleComplex*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*=() failed. cublasZgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasZgeam() failed.");
	}
	
	return (*this);
      }
      else{
	// SLOW IMPLEMENTATION
	
#pragma omp parallel for schedule(auto)
	for(unsigned int index=0;index<(numCols*numRows);index++)
	  (*this)[index] *= a;
	
	return (*this);
      }

#else
      const unsigned int MSIZE = numRows*numCols;
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sscal(MSIZE, *((float*)&a), (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_cscal(MSIZE, (const float*)&a, (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dscal(MSIZE, *((double*)&a), (double*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zscal(MSIZE, (const double*)&a, (double*)data, 1);
      }
      else{ // "normal implementation"
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<MSIZE;i++)
	  data[i] *= a;
      }
      
      return (*this);
#endif
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::operator/=(const T& s) 
    {
#ifdef CUBLAS
      const T invs = T(1.0f)/s;
      
      (*this) *= invs; // makes CUBLAS optimized call

      return (*this);
      
#else
      const unsigned int MSIZE = numRows*numCols;
      T ss = T(1.0)/s;
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sscal(MSIZE, *((float*)&ss), (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_cscal(MSIZE, (const float*)&ss, (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dscal(MSIZE, *((double*)&ss), (double*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zscal(MSIZE, (const double*)&ss, (double*)data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<MSIZE;i++)
	  data[i] *= ss;
      }      
      
      return (*this);
#endif
    }
    
    
    /***************************************************/
    
    template <typename T>
    vertex<T> matrix<T>::operator*(const vertex<T>& v) const
      
    {
      if(v.size() == 0)
	throw std::invalid_argument("multiply: incompatible vertex/matrix sizes");
      if(numCols != v.size())
	throw std::invalid_argument("multiply: incompatible vertex/matrix sizes");

#ifdef CUBLAS

      // BLAS v2 optimized cublas*gemv() functions
      // NOTE: cublas keeps matrices in column order so multiplication
      //       from right r = A*v is slow(??). For faster code
      //       try to later keep matrices in transposed form in memory
      //       so vector multiplication works fast (give transposed flag to gemv)
      
      vertex<T> r(numRows);
      
      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);
	
	cublasStatus_t s = cublasSgemv(cublas_handle,
				       CUBLAS_OP_N,
				       numRows, numCols,
				       (const float*)&alpha,
				       (const float*)(data), (int)numRows,
				       (const float*)(v.data), 1,
				       (const float*)&beta,
				       (float*)r.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*() failed. cublasSgemv() failed.");
	  throw CUDAException("CUBLAS cublasSgemv() failed.");
	}

	return r;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(1.0);
	const T beta  = T(0.0);
	
	cublasStatus_t s = cublasDgemv(cublas_handle,
				       CUBLAS_OP_N,
				       numRows, numCols,
				       (const double*)&alpha,
				       (const double*)(data), (int)numRows,
				       (const double*)(v.data), 1,
				       (const double*)&beta,
				       (double*)r.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*() failed. cublasDgemv() failed.");
	  throw CUDAException("CUBLAS cublasDgemv() failed.");
	}

	return r;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);
	
	cublasStatus_t s = cublasCgemv(cublas_handle,
				       CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(data), (int)numRows,
				       (const cuComplex*)(v.data), 1,
				       (const cuComplex*)&beta,
				       (cuComplex*)r.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*() failed. cublasCgemv() failed.");
	  throw CUDAException("CUBLAS cublasCgemv() failed.");
	}

	return r;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(1.0);
	const T beta  = T(0.0);
	
	cublasStatus_t s = cublasZgemv(cublas_handle,
				       CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(data), (int)numRows,
				       (const cuDoubleComplex*)(v.data), 1,
				       (const cuDoubleComplex*)&beta,
				       (cuDoubleComplex*)r.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::operator*() failed. cublasZgemv() failed.");
	  throw CUDAException("CUBLAS cublasZgemv() failed.");
	}

	return r;
      }
      else{

#pragma omp parallel for schedule(auto)
	for(unsigned int k=0;k<r.size();k++){
	  r[k] = T(0.0f);

	  for(unsigned int i=0;i<numCols;i++){
	    r[k] += (*this)(k, i) * v.data[i];
	  }
	}

	return r;
      }
      
#else      
      vertex<T> r(numRows);
      
      // BLAS level 2
      // uses optimized cblas_Xgemv() functions
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sgemv(CblasRowMajor, CblasNoTrans,
		    numRows, numCols,
		    1.0f, (float*)data, numCols, (float*)v.data, 1,
		    0.0f, (float*)r.data, 1);
	
	return r;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	blas_complex<float> a, b;
	a = 1.0f; b = 0.0f;
	
	cblas_cgemv(CblasRowMajor, CblasNoTrans,
		    numRows, numCols,
		    (float*)(&a), (float*)data, numCols, (float*)v.data, 1,
		    (float*)(&b), (float*)r.data, 1);
	
	return r;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dgemv(CblasRowMajor, CblasNoTrans,
		    numRows, numCols,
		    1.0, (double*)data, numCols, (double*)v.data, 1,
		    0.0, (double*)r.data, 1);
	
	return r;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	blas_complex<double> a, b;
	a = 1.0f; b = 0.0f;
	
	cblas_zgemv(CblasRowMajor, CblasNoTrans,
		    numRows, numCols,
		    (double*)(&a), (double*)data, numCols, (double*)v.data, 1,
		    (double*)(&b), (double*)r.data, 1);
	
	return r;
      }
      else{ // generic matrix * vertex code r = A*v
	
	unsigned int k = 0;
	for(unsigned int j=0;j<r.size();j++){
	  for(unsigned int i=0;i<numCols;i++,k++)
	    r[j] += data[k]*v.data[i];
	}
	
	return r;
      }
#endif
    }
    
    
    
    /***************************************************/

    // crossproduct matrix M(z): M(z) * y = z x y
    template <typename T>
    matrix<T>& matrix<T>::crossproduct(const vertex<T>& v) 
    {
      if(v.size() != 3){
	whiteice::logging.error("matrix::crossproduct() invalid input parameters");
	throw std::out_of_range("crossproduct() requires 3 dimensions");
      }
      
      if(numRows != 3 || numCols != 3)
	if(!resize(3,3)) throw std::bad_alloc();
      
      
      (*this)(0,0) = T(0);
      (*this)(0,1) = T(-v[2]);
      (*this)(0,2) = T( v[1]);
      (*this)(1,0) = T( v[2]);
      (*this)(1,1) = T(0);
      (*this)(1,2) = T(-v[0]);
      (*this)(2,0) = T(-v[1]);
      (*this)(2,1) = T( v[0]);
      (*this)(2,2) = T(0);
  
      return *this;
    }
    
    
    // euclidean rotation XYZ matrix
    template <typename T>
    matrix<T>& matrix<T>::rotation(const T& xr, const T& yr, const T& zr) 
    {
      if( (xsize() != 3 && ysize() != 3) ||
	  (xsize() != 4 && ysize() != 4) ){
	
	if(!resize_y(4)){
	  whiteice::logging.error("matrix::rotation(): resize failed.");
	  throw std::bad_alloc();
	}
	
	if(!resize_x(4)){
	  whiteice::logging.error("matrix::rotation(): resize failed.");
	  throw std::bad_alloc();
	}
      }
      
      T a(cos(xr));
      T b(sin(xr));
      T c(cos(yr));
      T d(sin(yr));
      T e(cos(zr));
      T f(sin(zr));
      
      
      (*this)(0,0) = T( c*e);
      (*this)(0,1) = T(-c*f);
      (*this)(0,2) = T( d);
      (*this)(1,0) = T( b*d*e + a*f);
      (*this)(1,1) = T(-b*d*f + a*e);
      (*this)(1,2) = T(-b*c);
      (*this)(2,0) = T(-a*d*e + b*f);
      (*this)(2,1) = T( a*d*f + b*e);
      (*this)(2,2) = T( a*c);
      
      if(ysize() == 4){
	(*this)(0,3) = T(0);
	(*this)(1,3) = T(0);
	(*this)(2,3) = T(0);
	(*this)(3,0) = T(0);
	(*this)(3,1) = T(0);
	(*this)(3,2) = T(0);
	(*this)(3,3) = T(1);      
      }
      
      return *this;
    }
    
    
    // 4x4 translation matrix
    template <typename T>
    matrix<T>& matrix<T>::translation(const T& dx, const T& dy, const T& dz) 
    {
      if(ysize() != 4){
	if(!resize_y(4)) throw std::bad_alloc();
	if(!resize_x(4)) throw std::bad_alloc();
      }
      
      for(unsigned int j=0;j<3;j++)
	for(unsigned int i=0;i<4;i++){
	  if(j == i) (*this)(j,i) = T(1);
	  else (*this)(j,i) = T(0);
	}
      
      (*this)(3,0) = T(dx);
      (*this)(3,1) = T(dy);
      (*this)(3,2) = T(dz);
      (*this)(3,3) = T(1);
      
      return (*this);
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::abs() 
    {
      const unsigned int N = numRows*numCols;

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<N;i++)
	data[i] = whiteice::math::abs(data[i]);
	  
      return (*this);
    }

    template <typename T>
    matrix<T>& matrix<T>::real()
    {
      const unsigned int N = numRows*numCols;

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<N;i++)
	data[i] = whiteice::math::real(data[i]);
      
      return (*this);
    }

    template <typename T>
    matrix<T>& matrix<T>::imag()
    {
      const unsigned int N = numRows*numCols;

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<N;i++)
	data[i] = whiteice::math::imag(data[i]);
      
      return (*this);
    }
    
  
    template <typename T>
    matrix<T>& matrix<T>::transpose() 
    {

#ifdef CUBLAS
      
      // transposes matrix using cublas*geam() calls as recommended
      // in NVIDIA cuBLAS documentation (alpha=1, beta=0)

      // FIXME should do in memory transposition, just keep swapping (i,j)<->(j,i) vars
      
      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	// don't do in-place transposition [SLOW(?)]
	math::matrix<T> C(numCols, numRows);
	C.zero();

	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_T, CUBLAS_OP_N,
				       numCols, numRows,
				       (const float*)&alpha,
				       (const float*)(this->data), numRows,
				       (const float*)&beta,
				       (const float*)C.data, numCols,
				       (float*)(C.data), numCols);
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::transpose(): cuBlasSgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasSgeam() failed.");
	}

	(*this) = C;

	gpu_sync();
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(1.0);
	const T beta  = T(0.0);

	// don't do in-place transposition [SLOW(?)]
	math::matrix<T> C(numCols, numRows);
	C.zero();

	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_T, CUBLAS_OP_N,
				       numCols, numRows,
				       (const double*)&alpha,
				       (const double*)(this->data), numRows,
				       (const double*)&beta,
				       (const double*)(C.data), numCols,
				       (double*)(C.data), numCols);
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::transpose(): cuBlasDgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasDgeam() failed.");
	}

	(*this) = C;

	gpu_sync();
	
	return (*this);
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	// don't do in-place transposition [SLOW(?)]
	math::matrix<T> C(numCols, numRows);
	C.zero();

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_T, CUBLAS_OP_N,
				       numCols, numRows,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(this->data), numRows,
				       (const cuComplex*)&beta,
				       (const cuComplex*)(C.data), numCols,
				       (cuComplex*)(C.data), numCols);
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::transpose(): cuBlasCgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasCgeam() failed.");
	}

	(*this) = C;

	gpu_sync();
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(1.0);
	const T beta  = T(0.0);

	// don't do in-place transposition [SLOW(?)]
	math::matrix<T> C(numCols, numRows);
	C.zero();
	
	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_T, CUBLAS_OP_N,
				       numCols, numRows,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(this->data), numRows,
				       (const cuDoubleComplex*)&beta,
				       (const cuDoubleComplex*)(C.data), numCols,
				       (cuDoubleComplex*)(C.data), numCols);
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::transpose(): cuBlasZgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasZgeam() failed.");
	}

	(*this) = C;

	gpu_sync();
	
	return (*this);
      }
      else{
	// SLOW IMPLEMENTATION
	matrix<T> R(numCols, numRows); // transposed matrix
	
#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<numRows;j++)
	  for(unsigned int i=0;i<numCols;i++)
	    R(i,j) = (*this)(j,i);

	(*this) = R;
	
	return (*this);
      }
      
#else
      
      const matrix<T> A(*this);
      this->resize(A.xsize(), A.ysize());
      
      // divide matrix into 16*16 blocks => 2*16 = 32 cache lines are in use
      // (small enough for most platforms)
      
      const unsigned int xleft = A.xsize() & 0x0F;
      const unsigned int yleft = A.ysize() & 0x0F;
      const unsigned int ys = A.ysize() & 0xF0;
      const unsigned int xs = A.xsize() & 0xF0;
      
      // transposes block by block
      // (so we hopefully are all the time within cache limits)
      unsigned int y, x;
      
      for(y=0;y<ys;y+=16){
	
	for(x=0;x<xs;x+=16){
	  
	  for(unsigned int j=0;j<16;j++){
	    for(unsigned int i=0;i<16;i++){
	      
	      (*this)(x + i, y + j) = A(y + j, x + i);
	    }
	  }
	}
	
	if(xleft){
	  
	  for(unsigned int j=0;j<16;j++){
	    for(unsigned int i=0;i<xleft;i++){
	      
	      (*this)(x + i, y + j) = A(y + j, x + i);
	      
	    }
	  }
	}
      }
      
      
      if(yleft){
	
	for(x=0;x<xs;x+=16){
	  
	  for(unsigned int j=0;j<yleft;j++){
	    for(unsigned int i=0;i<16;i++){
	      
	      (*this)(x + i, y + j) = A(y + j, x + i);
	    }
	  }
	}
	
	if(xleft){
	  
	  for(unsigned int j=0;j<yleft;j++){
	    for(unsigned int i=0;i<xleft;i++){
	      
	      (*this)(x + i, y + j) = A(y + j, x + i);
	      
	    }
	  }
	}
      }
            
      return *this;
#endif
    }
    
    
    // calculates hermitian matrix (conjugate transpose matrix)
    template <typename T>
    matrix<T>& matrix<T>::hermite() 
    {

#ifdef CUBLAS

      // calculates hermitian matrix using cublas*geam() calls as recommended
      // in NVIDIA cuBLAS documentation (alpha=1, beta=0)

      
      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	// don't do in-place transposition [SLOW(?)]
	math::matrix<T> C(numCols, numRows);
	C.zero();
	
	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_T, CUBLAS_OP_N,
				       numCols, numRows,
				       (const float*)&alpha,
				       (const float*)(this->data), numRows,
				       (const float*)&beta,
				       (const float*)(C.data), numCols,
				       (float*)(C.data), numCols);
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::hermite(): cuBlasSgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasSgeam() failed.");
	}

	(*this) = C;

	gpu_sync();
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);

	// don't do in-place transposition [SLOW(?)]
	math::matrix<T> C(numCols, numRows);
	C.zero();
	
	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_T, CUBLAS_OP_N,
				       numCols, numRows,
				       (const double*)&alpha,
				       (const double*)(this->data), numRows,
				       (const double*)&beta,
				       (const double*)(C.data), numCols,
				       (double*)(C.data), numCols);
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::hermite(): cuBlasDgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasDgeam() failed.");
	}

	(*this) = C;

	gpu_sync();
	
	return (*this);
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(1.0f);
	const T beta  = T(0.0f);
	
	// don't do in-place transposition [SLOW(?)]
	math::matrix<T> C(numCols, numRows);
	C.zero();

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_C, CUBLAS_OP_N,
				       numCols, numRows,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)(this->data), numRows,
				       (const cuComplex*)&beta,
				       (const cuComplex*)(C.data), numCols,
				       (cuComplex*)(C.data), numCols);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::hermite(): cuBlasCgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasCgeam() failed.");
	}

	(*this) = C;

	gpu_sync();
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(1.0);
	const T beta  = T(0.0);

	// don't do in-place transposition [SLOW(?)]
	math::matrix<T> C(numCols, numRows);
	C.zero();
	
	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_C, CUBLAS_OP_N,
				       numCols, numRows,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)(this->data), numRows,
				       (const cuDoubleComplex*)&beta,
				       (const cuDoubleComplex*)(C.data), numCols,
				       (cuDoubleComplex*)(C.data), numCols);
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::hermite(): cuBlasZgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasZgeam() failed.");
	}

	(*this) = C;

	gpu_sync();
	
	return (*this);
      }
      else{
	// SLOW IMPLEMENTATION
	matrix<T> R(numCols, numRows); // transposed matrix
	
#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<numRows;j++)
	  for(unsigned int i=0;i<numCols;i++)
	    R(i,j) = whiteice::math::conj( (*this)(j,i) );

	(*this) = R;
	
	return (*this);
      }
      
#else      
      this->transpose();

      // conjugates complex matrices
      if(typeid(T) == typeid(complex<float>) ||
	 typeid(T) == typeid(complex<double>) ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_complex<double>))
	{
	  auto& M = (*this);
	  
	  for(unsigned int j=0;j<M.numRows;j++)
	    for(unsigned int i=0;i<M.numCols;i++)
	      M(j,i) = whiteice::math::conj(M(j,i));
	}

      return (*this);
#endif
    }


    // just calculates complex conjugate of matrix values
    template <typename T>
    matrix<T>& matrix<T>::conj()
    {
      // conjugates complex matrices

      if(typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_complex<double>) ||
	 typeid(T) == typeid(complex<float>) ||
	 typeid(T) == typeid(complex<double>)){
	
	auto& M = (*this);
	
#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<M.numRows;j++)
	  for(unsigned int i=0;i<M.numCols;i++)
	    M(j,i) = whiteice::math::conj(M(j,i));

      }
      
      return (*this);
    }
    
    
    template <typename T>
    T matrix<T>::det() const 
    {
      if(ysize() != xsize()){
	whiteice::logging.error("matrix::det(): non square matrix");
	throw std::logic_error("matrix::determinate() - non square matrix");
      }
                  
      const unsigned int N = numRows;
      
      // TODO: calculate faster with better algorithms and
      // fast matrix multiplication
      
      // calculates determinant using
      // gauss-jordan elimination
      // (upper triangle matrix 
      // -> multiplication of diagonal is determinate)
      
      matrix<T> copy(*this); // copy of matrix
      T det = T(1);
      
      for(unsigned int i=0;i<N;i++)
      {
	if(copy(i,i) == T(0)) // resort
	{
	  // (tries to) finds non-zero entry
	  for(unsigned int j=i+1;j<N;j++)
	  {
	    
	    if(copy(j,i) != T(0)) // swaps values
	    {
	      for(unsigned int k=i;k<N;k++) // starts from i: values before it are zero
		std::swap<T>(copy(j,k), copy(i,k));
	      
	      det *= T(-1);
	    }
	  }
	}
	
	
	// diagonal entry is zero -> det = 0
	if(copy(i,i) == T(0)) return T(0);
	
	det *= copy(i,i);
	
	// sets a_ii element to 1
	// starts from i: values before it are zero
	for(unsigned int k=i+1;k<N;k++)
	  copy(i,k) /= copy(i,i);
	
	copy(i,i) = T(1);
	
	// eliminates lower rows
	for(unsigned int j=i+1;j<N;j++)
	{
	  // i:th element will become zero in a process
	  T l = copy(j,i);
	  
	  // stars from i: values before it are already zero
	  for(unsigned int k=i;k<numCols;k++){
	    copy(j,k) -= l*copy(i,k);
	  }
	}
	
	
      }
      
      return det;
    }
    
    
    
    template <typename T>
    bool  matrix<T>::inv() 
    {
      // simple and slow: gaussian elimination - works for small matrixes
      
      if(numRows != numCols){ // only square matrices has well-defined inverses
	whiteice::logging.warn("matrix::inv(): non square matrix");
	return false;
      }

#ifdef CUBLAS
      // cuBLAS code for matrix inverses

      if(typeid(T) == typeid(blas_real<float>)){
	matrix<T> A(*this);

	T** Aarray = NULL;
	int* PivotArray = NULL;
	int* infoArray = NULL;
	int infovalue = -1;

	auto e1 = cudaMallocManaged(&Aarray, sizeof(T*));
	auto e2 = cudaMallocManaged(&PivotArray, numRows*sizeof(int));
	auto e3 = cudaMallocManaged(&infoArray, sizeof(int));

	if(e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess){
	  if(Aarray) cudaFree(Aarray);
	  if(PivotArray) cudaFree(PivotArray);
	  if(infoArray) cudaFree(infoArray);
	  gpu_sync();
	  return false;
	}

	Aarray[0] = A.data;

	
	// step 1: perform in-place LU decomposition, P*A = L*U.
	//      Aarray[i] is n*n matrix A[i]
	auto s = cublasSgetrfBatched(cublas_handle,
				     numRows,
				     (float**)Aarray, numRows,
				     PivotArray, infoArray, 1);
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::inv(): cublasSgetrfBatched() failed.");

	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  
	  throw CUDAException("CUBLAS cublasSgetrfBatched() failed.");
	}
	
	//      check infoArray[i] to see if factorization of A[i] is successful or not.
	//      Array[i] contains LU factorization of A[i]

	cudaMemcpy(&infovalue, infoArray, sizeof(int), cudaMemcpyDeviceToHost);

	if(infovalue != 0){
	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);

	  gpu_sync();
	  return false; // fail with singular matrix
	}

	matrix<T> C(numRows, numCols);

	T** Carray = NULL;
	auto e4 = cudaMallocManaged(&Carray, sizeof(T*));

	if(e4 != cudaSuccess){
	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);

	  gpu_sync();
	  return false; // fail with singular matrix
	}

	Carray[0] = C.data;

	
	// step 2: perform out-of-place inversion, Carray[i] = inv(A[i])
	s = cublasSgetriBatched(cublas_handle,
				numRows,
				(float**)Aarray,
				numRows,
				PivotArray,
				(float**)Carray,
				numRows,
				infoArray, 1);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::inv(): cublasSgetriBatched() failed.");

	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  cudaFree(Carray);
	  
	  throw CUDAException("CUBLAS cublasSgetriBatched() failed.");
	}
	
	//      check infoArray[i] to see if inversion of A[i] is successful or not
	infovalue = -1;
	cudaMemcpy(&infovalue, infoArray, sizeof(int), cudaMemcpyDeviceToHost);
	
	if(infovalue != 0){

	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  cudaFree(Carray);
	  
	  gpu_sync();
	  return false; // fail with singular matrix
	}

	(*this) = C;

	cudaFree(Aarray);
	cudaFree(PivotArray);
	cudaFree(infoArray);
	cudaFree(Carray);

	gpu_sync();

	return true;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	matrix<T> A(*this);

	T** Aarray = NULL;
	int* PivotArray = NULL;
	int* infoArray = NULL;
	int infovalue = -1;

	auto e1 = cudaMallocManaged(&Aarray, sizeof(T*));
	auto e2 = cudaMallocManaged(&PivotArray, numRows*sizeof(int));
	auto e3 = cudaMallocManaged(&infoArray, sizeof(int));

	if(e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess){
	  if(Aarray) cudaFree(Aarray);
	  if(PivotArray) cudaFree(PivotArray);
	  if(infoArray) cudaFree(infoArray);
	  gpu_sync();
	  return false;
	}

	Aarray[0] = A.data;
	
	// step 1: perform in-place LU decomposition, P*A = L*U.
	//      Aarray[i] is n*n matrix A[i]
	cublasStatus_t s = cublasDgetrfBatched(cublas_handle,
					       numRows,
					       (double**)Aarray, numRows,
					       PivotArray, infoArray, 1);
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::inv(): cublasDgetrfBatched() failed.");

	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  
	  throw CUDAException("CUBLAS cublasDgetrfBatched() failed.");
	}
	
	//      check infoArray[i] to see if factorization of A[i] is successful or not.
	//      Array[i] contains LU factorization of A[i]
	cudaMemcpy(&infovalue, infoArray, sizeof(int), cudaMemcpyDeviceToHost);

	if(infovalue != 0){
	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  
	  gpu_sync();
	  return false; // fail with singular matrix
	}

	matrix<T> C(numRows, numCols);

	T** Carray = NULL;
	auto e4 = cudaMallocManaged(&Carray, sizeof(T*));

	if(e4 != cudaSuccess){
	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);

	  gpu_sync();
	  return false; // fail with singular matrix
	}

	Carray[0] = C.data;

	
	// step 2: perform out-of-place inversion, Carray[i] = inv(A[i])
	s = cublasDgetriBatched(cublas_handle,
				numRows,
				(double**)Aarray,
				numRows,
				PivotArray,
				(double**)Carray,
				numRows,
				infoArray, 1);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::inv(): cublasDgetriBatched() failed.");

	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  cudaFree(Carray);
	  
	  throw CUDAException("CUBLAS cublasDgetriBatched() failed.");
	}
	
	//      check infoArray[i] to see if inversion of A[i] is successful or not
	infovalue = -1;
	cudaMemcpy(&infovalue, infoArray, sizeof(int), cudaMemcpyDeviceToHost);
	
	if(infovalue != 0){

	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  cudaFree(Carray);
	  
	  gpu_sync();
	  return false; // fail with singular matrix
	}

	(*this) = C;

	cudaFree(Aarray);
	cudaFree(PivotArray);
	cudaFree(infoArray);
	cudaFree(Carray);

	gpu_sync();

	return true;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){

	matrix<T> A(*this);
	
	T** Aarray = NULL;
	int* PivotArray = NULL;
	int* infoArray = NULL;
	int infovalue = -1;

	auto e1 = cudaMallocManaged(&Aarray, sizeof(T*));
	auto e2 = cudaMallocManaged(&PivotArray, numRows*sizeof(int));
	auto e3 = cudaMallocManaged(&infoArray, sizeof(int));

	if(e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess){
	  if(Aarray) cudaFree(Aarray);
	  if(PivotArray) cudaFree(PivotArray);
	  if(infoArray) cudaFree(infoArray);
	  gpu_sync();
	  return false;
	}

	Aarray[0] = A.data;
	
	// step 1: perform in-place LU decomposition, P*A = L*U.
	//      Aarray[i] is n*n matrix A[i]
	cublasStatus_t s = cublasCgetrfBatched(cublas_handle,
					       numRows,
					       (cuComplex**)Aarray, numRows,
					       PivotArray, infoArray, 1);
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::inv(): cublasCgetrfBatched() failed.");
	  
	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  
	  throw CUDAException("CUBLAS cublasDgetrfBatched() failed.");
	}
	
	//      check infoArray[i] to see if factorization of A[i] is successful or not.
	//      Array[i] contains LU factorization of A[i]
	cudaMemcpy(&infovalue, infoArray, sizeof(int), cudaMemcpyDeviceToHost);

	if(infovalue != 0){
	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  
	  gpu_sync();
	  return false; // fail with singular matrix
	}

	matrix<T> C(numRows, numCols);
	
	T** Carray = NULL;
	auto e4 = cudaMallocManaged(&Carray, sizeof(T*));

	if(e4 != cudaSuccess){
	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);

	  gpu_sync();
	  return false; // fail with singular matrix
	}

	Carray[0] = C.data;

	
	// step 2: perform out-of-place inversion, Carray[i] = inv(A[i])
	s = cublasCgetriBatched(cublas_handle,
				numRows,
				(cuComplex**)Aarray,
				numRows,
				PivotArray,
				(cuComplex**)Carray,
				numRows,
				infoArray, 1);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::inv(): cublasCgetriBatched() failed.");

	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  cudaFree(Carray);
	  
	  throw CUDAException("CUBLAS cublasDgetriBatched() failed.");
	}
	
	//      check infoArray[i] to see if inversion of A[i] is successful or not
	infovalue = -1;
	cudaMemcpy(&infovalue, infoArray, sizeof(int), cudaMemcpyDeviceToHost);
	
	if(infovalue != 0){
	  
	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  cudaFree(Carray);
	  
	  gpu_sync();
	  return false; // fail with singular matrix
	}

	(*this) = C;

	cudaFree(Aarray);
	cudaFree(PivotArray);
	cudaFree(infoArray);
	cudaFree(Carray);

	gpu_sync();

	return true;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){

	matrix<T> A(*this);

	T** Aarray = NULL;
	int* PivotArray = NULL;
	int* infoArray = NULL;
	int infovalue = -1;

	auto e1 = cudaMallocManaged(&Aarray, sizeof(T*));
	auto e2 = cudaMallocManaged(&PivotArray, numRows*sizeof(int));
	auto e3 = cudaMallocManaged(&infoArray, sizeof(int));

	if(e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess){
	  if(Aarray) cudaFree(Aarray);
	  if(PivotArray) cudaFree(PivotArray);
	  if(infoArray) cudaFree(infoArray);
	  gpu_sync();
	  return false;
	}

	Aarray[0] = A.data;

	
	// step 1: perform in-place LU decomposition, P*A = L*U.
	//      Aarray[i] is n*n matrix A[i]
	cublasStatus_t s = cublasZgetrfBatched(cublas_handle,
					       numRows,
					       (cuDoubleComplex**)Aarray, numRows,
					       PivotArray, infoArray, 1);
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::inv(): cublasZgetrfBatched() failed.");

	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  
	  throw CUDAException("CUBLAS cublasZgetrfBatched() failed.");
	}
	
	//      check infoArray[i] to see if factorization of A[i] is successful or not.
	//      Array[i] contains LU factorization of A[i]

	cudaMemcpy(&infovalue, infoArray, sizeof(int), cudaMemcpyDeviceToHost);

	if(infovalue != 0){
	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  
	  gpu_sync();
	  return false; // fail with singular matrix
	}

	matrix<T> C(numRows, numCols);

	T** Carray = NULL;
	auto e4 = cudaMallocManaged(&Carray, sizeof(T*));
	
	if(e4 != cudaSuccess){
	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);

	  gpu_sync();
	  return false; // fail with singular matrix
	}

	Carray[0] = C.data;
	
	// step 2: perform out-of-place inversion, Carray[i] = inv(A[i])
	s = cublasZgetriBatched(cublas_handle,
				numRows,
				(cuDoubleComplex**)Aarray,
				numRows,
				PivotArray,
				(cuDoubleComplex**)Carray,
				numRows,
				infoArray, 1);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::inv(): cublasZgetriBatched() failed.");

	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  cudaFree(Carray);
	  
	  throw CUDAException("CUBLAS cublasZgetriBatched() failed.");
	}
	
	//      check infoArray[i] to see if inversion of A[i] is successful or not
	infovalue = -1;
	cudaMemcpy(&infovalue, infoArray, sizeof(int), cudaMemcpyDeviceToHost);
	
	if(infovalue != 0){
	  cudaFree(Aarray);
	  cudaFree(PivotArray);
	  cudaFree(infoArray);
	  cudaFree(Carray);
	  
	  gpu_sync();
	  return false; // fail with singular matrix
	}

	(*this) = C;

	cudaFree(Aarray);
	cudaFree(PivotArray);
	cudaFree(infoArray);
	cudaFree(Carray);

	gpu_sync();

	return true;
      }
      else{
	// general code for solving matrix inverse [long]
	// copy from cblas section
	
	const unsigned int N = numRows;
	
	// Gauss-Jordan elimination
	
	matrix<T> copy(*this);
	this->identity();
	
	for(unsigned int i=0;i<N;i++){
	  if(copy(i,i) == T(0.0)){ // resort
	    // (tries to) finds non-zero entry
	    bool singular = true;
	    
	    for(unsigned int j=i+1;j<N;j++){
	      if(copy(j,i) != T(0.0)){ // swaps values
		
		for(unsigned int k=0;k<N;k++){
		  std::swap<T>(copy(j,k), copy(i,k));
		  std::swap<T>((*this)(j,k), (*this)(i,k));
		}
		
		// could find a way to solve apparent
		// singualirity problem.
		singular = false;
		break;
	      }
	    }
	    
	    if(singular)
	      return false;
	  }
	  
	  
	  // sets a_ii = 1
	  {
	    T t = copy(i,i);
	    
	    for(unsigned int j=0;j<N;j++){
	      copy(i,j) /= t;
	      (*this)(i,j) /= t;
	    }
	  }
	  
	  
	  if(i >= 1){
	    // eliminates upper row columns to zero	  
	    for(unsigned int j=0;j<i;j++){
	      T k = copy(j,i);
	      
	      for(unsigned int r=0;r<N;r++){
		copy(j,r) -= k * copy(i,r);
		(*this)(j,r) -= k * (*this)(i,r);
	      }
	    }
	  }
	  
	  
	  if(i < N-1){
	    // eliminates lower row columns to zero
	    for(unsigned int j=i+1;j<N;j++){
	      T k = copy(j,i);
	      
	      for(unsigned int r=0;r<N;r++){
		copy(j,r) -= k * copy(i,r);
		(*this)(j,r) -= k * (*this)(i,r);
	      }
	    }
	  }
	  
	}
	
	return true;
	
      }
      
#else
      const unsigned int N = numRows;
      
      // gauss-jordan elimination
      
      matrix<T> copy(*this);
      this->identity();
      
      for(unsigned int i=0;i<N;i++){
	if(copy(i,i) == T(0.0)){ // resort
	  // (tries to) finds non-zero entry
	  bool singular = true;
	  
	  for(unsigned int j=i+1;j<N;j++){
	    if(copy(j,i) != T(0.0)){ // swaps values
	      
	      for(unsigned int k=0;k<N;k++){
		std::swap<T>(copy(j,k), copy(i,k));
		std::swap<T>((*this)(j,k), (*this)(i,k));
	      }
	      
	      // could find a way to solve apparent
	      // singualirity problem.
	      singular = false;
	      break;
	    }
	  }
	  
	  if(singular)
	    return false;
	}
	
	
	// sets a_ii = 1
	{
	  T t = copy(i,i);
	  
	  for(unsigned int j=0;j<N;j++){
	    copy(i,j) /= t;
	    (*this)(i,j) /= t;
	  }
	}
	
	
	if(i >= 1){
	  // eliminates upper row columns to zero	  
	  for(unsigned int j=0;j<i;j++){
	    T k = copy(j,i);
	    
	    for(unsigned int r=0;r<N;r++){
	      copy(j,r) -= k * copy(i,r);
	      (*this)(j,r) -= k * (*this)(i,r);
	    }
	  }
	}
	
	
	if(i < N-1){
	  // eliminates lower row columns to zero
	  for(unsigned int j=i+1;j<N;j++){
	    T k = copy(j,i);
	    
	    for(unsigned int r=0;r<N;r++){
	      copy(j,r) -= k * copy(i,r);
	      (*this)(j,r) -= k * (*this)(i,r);
	    }
	  }
	}
	
      }
      
      return true;
#endif
    }
    
    
    template <typename T>
    bool matrix<T>::pseudoinverse(const T machine_epsilon) 
    {

#ifdef CUBLAS
      // slow code for pseudoinverse using cuBLAS matrix inverse code
      
      if(numCols <= numRows){ // pinv(A) = inv(A^h*A) * A^h

	auto& A = (*this);
	auto Ah = (*this);
	Ah.hermite();
	
	auto AhA = Ah*A;
	
	if(AhA.inv() == false){
	  whiteice::logging.error("matrix::pseudoinverse() failed. singular matrix.");
	  return false;
	}
	
	(*this) = Ah*A * Ah;
      
	return true;
      }
      else{ // pinv(A) = A^h * pinv(A A^h)
	
	auto& A = *this;
	auto Ah = *this;
	Ah.hermite();

	auto AAh = A*Ah;

	if(AAh.inv() == false){
	  whiteice::logging.error("matrix::pseudoinverse() failed. singular matrix.");
	  return false;
	}
	
	(*this) = Ah * AAh;

	return true;
	
      }
#else
      
      
#if 1
      // calculates pseudoinverse using symmetric_pseudoinverse

      if(numCols <= numRows){
	// pinv(A) = pinv(A^h A) * A^h

	auto&A = *this;
	auto Ah = *this;
	Ah.hermite();

	auto AhA = Ah*A; // (numCols x numRows) (numRows x numCols)

	if(AhA.symmetric_pseudoinverse(machine_epsilon) == false){
	  whiteice::logging.error("matrix::pseudoinverse(): symmetric_pseudoinverse() FAILED");
	  return false;
	}
	
	*this = AhA * Ah;

	return true;
      }
      else{
	// pinv(A) = A^h * pinv(A A^h)

	auto&A = *this;
	auto Ah = *this;
	Ah.hermite();

	auto AAh = A*Ah;

	if(AAh.symmetric_pseudoinverse(machine_epsilon) == false){
	  whiteice::logging.error("matrix::pseudoinverse(): symmetric_pseudoinverse() FAILED");
	  return false;
	}
	
	*this = Ah * AAh;

	return true;
      }
#else
      // this currently makes copy of itself which is SLOW but
      // allows us to always give some kind of result even when
      // SVD fails (should never happen)

      // !!!!!!!!!!!!!!!!!!!! SVD is buggy so this doesn't work..
      
      matrix<T> U, V;
      matrix<T> S(*this);

      // machine epsilon [numerical accuracy] (see wikipedia)
      T epsilon = T(10e-10);
      
      if(machine_epsilon <= T(0.0)){

	if(typeid(T) == typeid(float)){
	  epsilon = T(1.19e-07);
	}
	else if(typeid(T) == typeid(double)){
	  epsilon = T(2.22e-16);
	}
	else if(typeid(T) == typeid(blas_real<float>)){
	  epsilon = T(1.19e-07);
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  epsilon = T(2.22e-16);
	}
      }
      else{
	epsilon = machine_epsilon; // user specified machine epsilon
      }
      

      const T tolerance = whiteice::math::abs(epsilon * T(max(this->numRows, this->numCols)) * norm_inf(*this));
      double k = 0.0;

      // fail safe to work even if SVD fails for mysterious reasons (should not never happen)..
      while(svd(S, U, V) == false){ // this = U*S*V^h
	whiteice::logging.warn("matrix::pseudoinverse(): computation of svd failed. applying rescue heuristics");
	
	S = *this;

	// hack to remove singular/bad data (this should never happen)
	for(unsigned int i=0;i<min(S.numRows,S.numCols);i++)
	  S(i,i) += T(pow(2.0,k))*tolerance;

	k++;
      }

      // calculates pseudoinverse U*S*V^h => V*inv(S)*U^h
      {
	for(unsigned int i=0;i<min(S.numRows,S.numCols);i++){
	  if(whiteice::math::abs(S(i,i)) <= whiteice::math::abs(tolerance))
	    S(i,i) = T(0.0); // zero elements are kept zero
	  else
	    S(i,i) = T(1.0)/S(i,i); // calculates inverse
	}

	S.hermite();

	(*this) = V*S*U.hermite();
      }

      return *this;
#endif
#endif
    }


    template <typename T>
    bool matrix<T>::symmetric_pseudoinverse(const T machine_epsilon) 
    {

#ifdef CUBLAS

      return this->pseudoinverse(machine_epsilon); // "optimized" cuBLAS code

#else
      
      // TODO: fix symmetric_eig to work with complex numbers!!! so the compilation would work
      assert(1); 
      
#if 1
      if(numCols != numRows)
	return false;

      // this currently makes copy of itself which is SLOW but
      // allows us to always give some kind of result even when
      // EIG fails (should never happen)
      
      matrix<T> X;
      matrix<T>&D = *this;

      // machine epsilon [numerical accuracy] (see wikipedia)
      T epsilon = T(10e-10);
      
      if(machine_epsilon <= T(0.0)){

	if(typeid(T) == typeid(float)){
	  epsilon = T(1.19e-07);
	}
	else if(typeid(T) == typeid(double)){
	  epsilon = T(2.22e-16);
	}
	else if(typeid(T) == typeid(blas_real<float>)){
	  epsilon = T(1.19e-07);
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  epsilon = T(2.22e-16);
	}
      }
      else{
	epsilon = machine_epsilon; // user specified machine epsilon
      }
      

      const T tolerance = whiteice::math::abs(epsilon * T(max(this->numRows, this->numCols)) * norm_inf(*this));

      {
	// KNOWN BUG/HACK to make compilation go through. Always converts to real valued data
	whiteice::math::matrix< whiteice::math::blas_real<float> > DD, XX;

	convert(DD, D); // converts matrixes to blas_real format
	convert(XX, X);

	int counter = 0;
	const int MAXCOUNT = 20;
	
	while(symmetric_eig(DD, XX) == false){ // this = X*D*X^h
	  if(counter > MAXCOUNT){
	    whiteice::logging.error("matrix::symmetric_pseudoinverse(): computation of symmetric evd failed.");
	    return false;
	  }

	  // assumes failure is because of diagonal zeros

	  convert(DD, D); // converts matrixes to blas_real format
	  convert(XX, X);

	  blas_real<float> tol = 0.0f;
	  convert(tol, tolerance);
	  tol = whiteice::math::abs(tol);

	  // adds regularizer constant (HACK to diagonalize matrix)
	  const blas_real<float> e =
	    whiteice::math::pow(2.0f, (float)counter);
	  tol = tol*e;

	  for(unsigned int i=0;i<(this->numRows);i++){
	    DD(i,i) += tol; // regularizes eigenvalue decomposition
	  }

	  counter++;
	}

	convert(D, DD);
	convert(X, XX);
      }


      // calculates pseudoinverse X*D*X^h => X*inv(D)*X^h
      {
	for(unsigned int i=0;i<D.numRows;i++){
	  if(whiteice::math::abs(D(i,i)) <= whiteice::math::abs(tolerance))
	    D(i,i) = T(0.0); // zero elements are kept zero
	  else
	    D(i,i) = T(1.0)/D(i,i); // calculates inverse
	}

	(*this) = X*D;
	(*this) *= X.hermite();
      }

      return true;
#endif

#endif
    }
    
    
    template <typename T>
    T matrix<T>::trace() const 
    {
      if(numCols != numRows){
	whiteice::logging.error("matrix::trace(): non square matrix");
	throw std::logic_error("matrix::trace() non square matrix");
      }

#ifdef CUBLAS
      T tr = T(0.0f);

#pragma omp parallel
      {
	T t = T(0.0f);

#pragma omp for schedule(auto) nowait
	for(unsigned int i=0;i<numCols;i++)
	  t += data[i*(numRows+1)];

#pragma omp critical
	{
	  tr += t;
	}
      }
	
      return tr;

#else
      T tr = T(0.0f);

#pragma omp parallel
      {
	T t = T(0.0f);

#pragma omp for schedule(auto) nowait
	for(unsigned int i=0;i<numRows;i++){
	  t += data[i*(numCols+1)];
	}

#pragma omp critical
	{
	  tr += t;
	}
      }
      
      return tr;
#endif
    }

    
    template <typename T>
    void matrix<T>::diag(vertex<T>& diagonal) const 
    {
      if(numCols < numRows)
	diagonal.resize(numCols);
      else
	diagonal.resize(numRows);

#ifdef CUBLAS

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<diagonal.size();i++){
	diagonal[i] = data[i*(numRows+1)];
      }
      
#else
      unsigned int index = 0;
      for(unsigned int i=0;i<diagonal.size();i++){
	diagonal[i] = data[index];
	index += numCols+1;
      }
#endif
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::identity()
    {
#ifdef CUBLAS
      zero();

      const unsigned int N = (numRows<numCols) ? numRows : numCols;

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<N;i++)
	data[i*(numRows+1)] = T(1.0f);

      return (*this);
      
#else
      unsigned int index = 0;
      for(unsigned int j=0;j<numRows;j++){
	for(unsigned int i=0;i<numCols;i++, index++)
	{
	  if(i == j) data[index] = T(1.0f);
	  else data[index] = T(0.0f);
	}
      }
      
      return (*this);
#endif
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::zero()
    {

#ifdef CUBLAS

      // zeros matrix using cublas*geam() calls as recommended
      // in NVIDIA cuBLAS documentation (alpha=0, beta=0 should clear matrix to zero)
      // 
      // NOTE: cudaMemset() is maybe a little slower if cublas*geam() is optimized
      // and writes whole floats instead of bytes?????
      //
      // TODO: compare cudaMemset and cublas*geam() speed with special code
      //       to optimize zeroing matrix.
      
      if(typeid(T) == typeid(blas_real<float>)){
	const T alpha = T(0.0f);
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasSgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const float*)&alpha,
				       (const float*)NULL, numRows,
				       (const float*)&beta,
				       (const float*)NULL, numRows,
				       (float*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::zero(): cublasSgeam() failed.");
	  throw CUDAException("CUBLAS cublasSgeam() failed.");
	}
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	const T alpha = T(0.0f);
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasDgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const double*)&alpha,
				       (const double*)NULL, numRows,
				       (const double*)&beta,
				       (const double*)NULL, numRows,
				       (double*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::zero(): cublasDgeam() failed.");
	  throw CUDAException("CUBLAS cublasDgeam() failed.");
	}
	
	return (*this);
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	const T alpha = T(0.0f);
	const T beta  = T(0.0f);

	cublasStatus_t s = cublasCgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuComplex*)&alpha,
				       (const cuComplex*)NULL, numRows,
				       (const cuComplex*)&beta,
				       (const cuComplex*)NULL, numRows,
				       (cuComplex*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::zero(): cublasCgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasCgeam() failed.");
	}
				       
	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	const T alpha = T(0.0);
	const T beta  = T(0.0);
	
	cublasStatus_t s = cublasZgeam(cublas_handle,
				       CUBLAS_OP_N, CUBLAS_OP_N,
				       numRows, numCols,
				       (const cuDoubleComplex*)&alpha,
				       (const cuDoubleComplex*)NULL, numRows,
				       (const cuDoubleComplex*)&beta,
				       (const cuDoubleComplex*)NULL, numRows,
				       (cuDoubleComplex*)(this->data), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::zero(): cublasZgeam() failed.");
	  throw CUDAException("CUBLAS cuBlasZgeam() failed.");
	}
				       
	return (*this);
      }
      else{
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<numCols*numRows;i++)
	  data[i] = T(0.0f);
	
	return (*this);
      }
      
      
#else
      
      if(typeid(T) == typeid(blas_real<float>) ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>) ||
	 typeid(T) == typeid(blas_complex<double>) ||
	 typeid(T) == typeid(float) ||
	 typeid(T) == typeid(double)){
	
	memset(data, 0, numRows*numCols*sizeof(T));
      }
      else{
	for(unsigned int index=0;index<numRows*numCols;index++)
	  data[index] = T(0.0);
      }
      
      return (*this);
#endif
    }
    
    
    template <typename T>
    unsigned int matrix<T>::xsize() const 
    {
      if(numRows <= 0) return 0;
      
      return numCols;
    }
    
    
    template <typename T>
    unsigned int matrix<T>::size() const 
    {
      return numRows*numCols;
    }
    
    template <typename T>
    unsigned int matrix<T>::ysize() const 
    {
      return numRows;
    }
    
    
    template <typename T>
    bool matrix<T>::resize(unsigned int y, unsigned int x) 
    {
      // whiteice::logging.warn("matrix::resize(): is slow when not done as elementary operation.");

#ifdef CUBLAS

      if(y == 0 || x == 0){
	if(data) cudaFree(data);
	if(compressor) delete compressor;
	data = NULL;
	compressor = NULL;
	numRows = y;
	numCols = x;
	
	return true;
      }
      else if(y*x = numCols*numRows){
	numRows = y;
	numCols = x;
	
	err = cudaMemset(data, 0, numRows*numCols*sizeof(T));
	
	if(err != cudaSuccess){
	  whiteice::logging.error("matrix::resize(): cudaMemset() failed.");
	  throw CUDAException("CUBLAS cudaMemset() failed");
	}

	if(compressor) delete compressor;
	compressor = NULL;

	return true;
      }
      else{
	cudaError_t err;
	void* cudaptr = NULL;
	err = cudaMallocManaged(&cudaptr, y*x*sizeof(T));

	if(err != cudaSuccess || cudaptr == NULL){
	  gpu_sync();
	  whiteice::logging.error("matrix::resize(): cudaMallocManaged() failed.");
	  throw CUDAException("CUBLAS memory allocation failure.");
	}
	
	err = cudaMemset(cudaptr, 0, y*x*sizeof(T));
	
	if(err != cudaSuccess){
	  whiteice::logging.error("matrix::resize(): cudaMemset() failed.");
	  throw CUDAException("CUBLAS cudaMemset() failed");
	}

	if(data) cudaFree(data);

	numRows = y;
	numCols = x;
	data = (T*)cudaptr;
	if(compressor) delete compressor;
	compressor = NULL;

	return true;
      }
      
#else
      if(y == 0 || x == 0){
	if(data) free(data);
	if(compressor) delete compressor;
	data = NULL;
	compressor = NULL;
	numRows = y;
	numCols = x;
	return true;
      }
      else if(y*x == numCols*numRows){
	numRows = y;
	numCols = x;
	memset(data, 0, numRows*numCols*sizeof(T)); // resets values to zero [remove]
	if(compressor) delete compressor;
	compressor = NULL;
	return true;
      }
      else{
	T* new_area = NULL;

	new_area = (T*)realloc(data, sizeof(T)*x*y);
	if(new_area == NULL) return false;
	data = new_area;

	if(compressor) delete compressor;
	compressor = NULL;
	
	numRows = y;
	numCols = x;
	
	memset(data, 0, numRows*numCols*sizeof(T));
	return true;
      }
      
#endif
      
    }
    
    
    template <typename T>
    bool matrix<T>::resize_x(unsigned int d) 
    {
#ifdef CUBLAS
      if(d == numCols) return true;
      else if(d == 0){
	if(data) cudaFree(data);
	if(compressor) delete compressor;
	data = NULL;
	compressor = NULL;
	numCols = d;
	
	return true;
      }
      else{

	cudaError_t err;
	void* cudaptr = NULL;
	err = cudaMallocManaged(&cudaptr, numRows*d*sizeof(T));

	if(err != cudaSuccess || cudaptr == NULL){
	  gpu_sync();
	  whiteice::logging.error("matrix::resize_x(): cudaMallocManaged() failed.");
	  throw CUDAException("CUBLAS memory allocation failure.");
	}
	
	err = cudaMemset(cudaptr, 0, numRows*d*sizeof(T));
	
	if(err != cudaSuccess){
	  whiteice::logging.error("matrix::resize_x(): cudaMemset() failed.");
	  throw CUDAException("CUBLAS cudaMemset() failed");
	}

	if(data) cudaFree(data);

	numCols = d;
	data = (T*)cudaptr;
	if(compressor) delete compressor;
	compressor = NULL;

	return true;
	
      }
#else
      if(d == numCols){
	return true;
      }
      else if(d == 0){
	if(data) free(data);
	if(compressor) delete compressor;
	
	data = NULL;
	compressor = NULL;
	
	numCols = d; 
	
	return true;
      }
      else{
	T* new_area = NULL;

	new_area = (T*)malloc(sizeof(T)*d*numRows);
	if(new_area == NULL) return false;
	
	if(data) free(data);
	data = new_area;
	
	if(compressor) delete compressor;
	compressor = NULL;
	
	numCols = d;
	
	memset(data, 0, numRows*numCols*sizeof(T));
	return true;
	
      }
#endif
    }
    
    
    template <typename T>
    bool matrix<T>::resize_y(unsigned int d) 
    {
#ifdef CUBLAS

      if(d == numRows) return true; // nothing to do
      else if(d == 0){
	if(data) cudaFree(data);
	if(compressor) delete compressor;
	data = NULL;
	compressor = NULL;
	numRows = d;
	
	return true;
      }
      else{

	cudaError_t err;
	void* cudaptr = NULL;
	err = cudaMallocManaged(&cudaptr, numCols*d*sizeof(T));

	if(err != cudaSuccess || cudaptr == NULL){
	  gpu_sync();
	  whiteice::logging.error("matrix::resize_y(): cudaMallocManaged() failed.");
	  throw CUDAException("CUBLAS memory allocation failure.");
	}
	
	err = cudaMemset(cudaptr, 0, numCols*d*sizeof(T));
	
	if(err != cudaSuccess){
	  whiteice::logging.error("matrix::resize_y(): cudaMemset() failed.");
	  throw CUDAException("CUBLAS cudaMemset() failed");
	}

	if(data) cudaFree(data);

	numRows = d;
	data = (T*)cudaptr;
	if(compressor) delete compressor;
	compressor = NULL;

	return true;
	
      }
#else
      if(d == numRows){
	return true;
      }
      else if(d == 0){
	if(data) free(data);
	if(compressor) delete compressor;
	
	data = NULL;
	compressor = NULL;
	
	numRows = d;
	
	return true;
      }
      else{
	T* new_area = 0;

	new_area = (T*)malloc(sizeof(T)*d*numCols);
	  
	if(new_area == NULL) return false;
	if(data) free(data);
	data = new_area;

	if(compressor) delete compressor;
	compressor = NULL;
	
	numRows = d;
	
	memset(data, 0, numRows*numCols*sizeof(T));
	return true;
	
      }

#endif
    }
    
    
    
    template <typename T>
    T matrix<T>::rownorm(unsigned int y, unsigned int x1, unsigned int x2) const
      
    {
      if(x2 >= numCols) x2 = numCols - 1;
      
      if(x2 < x1 || x1 >= numCols || y >= numRows){
	whiteice::logging.error("matrix::rownorm(): bad index parameter to matrix.");
	throw std::out_of_range("rownorm(): bad indeces to matrix");
      }
      
#ifdef CUBLAS
      x2++;
      
      if(typeid(T) == typeid(blas_real<float>)){
	float result;

	auto s = cublasSnrm2(cublas_handle, (int)(x2-x1),
			     (const float*)&(data[y + x1*numRows]),
			     numRows, (float*)&result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::rownorm(): cublasSnrm2() failed.");
	  throw CUDAException("CUBLAS cublasSnrm2() failed.");
	}

	return T(result);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double result;

	auto s = cublasDnrm2(cublas_handle, (int)(x2-x1),
			     (const double*)&(data[y + x1*numRows]),
			     numRows, (double*)&result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::rownorm(): cublasDnrm2() failed.");
	  throw CUDAException("CUBLAS cublasDnrm2() failed.");
	}

	return T(result);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	float result;
	
	auto s = cublasScnrm2(cublas_handle, (int)(x2-x1),
			      (const cuComplex*)&(data[y + x1*numRows]),
			      numRows, (float*)&result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::rownorm(): cublasScnrm2() failed.");
	  throw CUDAException("CUBLAS cublasScnrm2() failed.");
	}

	return T(result);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	double result;
	
	auto s = cublasDznrm2(cublas_handle, (int)(x2-x1),
			      (const cuDoubleComplex*)&(data[y + x1*numRows]),
			      numRows, (double*)&result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::rownorm(): cublasDznrm2() failed.");
	  throw CUDAException("CUBLAS cublasDznrm2() failed.");
	}

	return T(result);
      }
      else{
	T len = T(0.0f);

#pragma omp parallel
	{
	  T l = T(0.0f);

#pragma omp for schedule(auto) nowait
	  for(unsigned int i=x1;i<=x2;i++)
	    l += data[y*numCols + i]*whiteice::math::conj(data[y*numCols + i]);

#pragma omp critical
	  {
	    len += l;
	  }
	}
	
	return whiteice::math::sqrt(whiteice::math::abs(len));
      }
      
#else      
      
      if(typeid(T) == typeid(blas_real<float>)){
	return T( cblas_snrm2(x2 - x1 + 1, (const float*)&(data[y*numCols + x1]), 1) );
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	return T( cblas_scnrm2(x2 - x1 + 1, (const float*)&(data[y*numCols + x1]), 1) );
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	return T( cblas_dnrm2(x2 - x1 + 1, (const double*)&(data[y*numCols + x1]), 1) );
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	return T( cblas_dznrm2(x2 - x1 + 1, (const double*)&(data[y*numCols + x1]), 1) );
      }
      else{ // generic length calculation
	T len = T(0.0f);

#pragma omp parallel
	{
	  T l = T(0.0f);

#pragma omp for schedule(auto) nowait
	  for(unsigned int i=x1;i<x2;i++)
	    l += data[y + i*numRows]*whiteice::math::conj(data[y + i*numRows]);

#pragma omp critical
	  {
	    len += l;
	  }
	}
	
	return whiteice::math::sqrt(whiteice::math::abs(len));
      }
#endif
    }
    
    
    template <typename T>
    T matrix<T>::colnorm(unsigned int x, unsigned int y1, unsigned int y2) const
    {
      if(y2 > numRows) y2 = numRows - 1;
      
      if(y2 < y1 || y1 >= numRows || x >= numCols){
	printf("colnorm failed: %d %d %d %d %d\n", x, y1, y2, numRows, numCols);
	fflush(stdout);
	
	whiteice::logging.error("matrix::colnorm(): bad index parameter to matrix.");

	assert(0);
	
	throw std::out_of_range("colnorm(): bad indeces to matrix");
      }
      
#ifdef CUBLAS
      y2++;

      if(typeid(T) == typeid(blas_real<float>)){
	float result;

	auto s = cublasSnrm2(cublas_handle, (int)(y2-y1),
			     (const float*)&(data[y1 + x*numRows]),
			     1, (float*)&result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::colnorm(): cublasSnrm2() failed.");
	  throw CUDAException("CUBLAS cublasSnrm2() failed.");
	}

	return T(result);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double result;

	auto s = cublasDnrm2(cublas_handle, (int)(y2-y1),
			     (const double*)&(data[y1 + x*numRows]),
			     1, (double*)&result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::colnorm(): cublasDnrm2() failed.");
	  throw CUDAException("CUBLAS cublasDnrm2() failed.");
	}

	return T(result);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	float result;
	
	auto s = cublasScnrm2(cublas_handle, (int)(y2-y1),
			      (const cuComplex*)&(data[y1 + x*numRows]),
			      1, (float*)&result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::colnorm(): cublasScnrm2() failed.");
	  throw CUDAException("CUBLAS cublasScnrm2() failed.");
	}

	return T(result);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	double result;
	
	auto s = cublasDznrm2(cublas_handle, (int)(y2-y1),
			      (const cuDoubleComplex*)&(data[y1 + x*numRows]),
			      1, (double*)&result);

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::colnorm(): cublasDznrm2() failed.");
	  throw CUDAException("CUBLAS cublasDznrm2() failed.");
	}

	return T(result);
      }
      else{
	T len = T(0.0f);

#pragma omp parallel
	{
	  T l = T(0.0f);

#pragma omp for schedule(auto) nowait
	  for(unsigned int i=y1;i<y2;i++)
	    l += data[i + x*numRows]*whiteice::math::conj(data[i + x*numRows]);

#pragma omp critical
	  {
	    len += l;
	  }
	}
	
	return whiteice::math::sqrt(whiteice::math::abs(len));
      }

#else      
      
      if(typeid(T) == typeid(blas_real<float>)){
	return T( cblas_snrm2(y2 - y1 + 1, (const float*)&(data[y1*numCols + x]), numCols) );
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	return T( cblas_scnrm2(y2 - y1 + 1, (const float*)&(data[y1*numCols + x]), numCols) );
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	return T( cblas_dnrm2(y2 - y1 + 1, (const double*)&(data[y1*numCols + x]), numCols) );
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	return T( cblas_dznrm2(y2 - y1 + 1, (const double*)&(data[y1*numCols + x]), numCols) );
      }
      else{ // generic length calculation
	T len = T(0.0f);

#pragma omp parallel
	{
	  T l = T(0.0f);
	  
#pragma omp for schedule(auto) nowait
	  for(unsigned int i=y1;i<=y2;i++)
	    l += data[x + i*numCols] * data[x + i*numCols];

#pragma omp critical
	  {
	    len += l;
	  }
	}
	
	return whiteice::math::sqrt(len);
      }
#endif
    }
    
    
    
    // copies row data to a given vector, M(y,x1:x2) -> v
    template <typename T>
    void matrix<T>::rowcopyto(vertex<T>& v,
			      unsigned int y,
			      unsigned int x1, unsigned int x2) const
    {
      if(x2 >= numCols) x2 = numCols - 1;
      
      if(x2 < x1 || x1 >= numCols || y >= numRows){
	whiteice::logging.error("matrix::rowcopyto(): bad index parameter to matrix.");
	throw std::out_of_range("rowcopyto(): bad indeces to matrix");
      }
      
#ifdef CUBLAS
      x2++;

      if(v.resize(x2-x1) != (x2-x1)){
	whiteice::logging.error("matrix::rowcopyto(): v.resize() failed.");
	throw std::bad_alloc();
      }

      if(typeid(T) == typeid(blas_real<float>)){
	
	auto s = cublasScopy(cublas_handle, (int)(x2-x1),
			     (const float*)&(data[y + x1*numRows]), numRows,
			     (float*)v.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::rowcopyto(): cublasScopy() failed.");
	  throw CUDAException("CUBLAS cublasScopy() failed.");
	}
	
      }
      else if(typeid(T) == typeid(blas_real<double>)){

	auto s = cublasDcopy(cublas_handle, (int)(x2-x1),
			     (const double*)&(data[y + x1*numRows]), numRows,
			     (double*)v.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::rowcopyto(): cublasDcopy() failed.");
	  throw CUDAException("CUBLAS cublasDcopy() failed.");
	}
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){

	auto s = cublasCcopy(cublas_handle, (int)(x2-x1),
			     (const cuComplex*)&(data[y + x1*numRows]), numRows,
			     (cuComplex*)v.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::rowcopyto(): cublasCcopy() failed.");
	  throw CUDAException("CUBLAS cublasCcopy() failed.");
	}
	
      }
      else if(typeid(T) == typeid(blas_complex<double>)){

	auto s = cublasZcopy(cublas_handle, (int)(x2-x1),
			     (const cuDoubleComplex*)&(data[y + x1*numRows]), numRows,
			     (cuDoubleComplex*)v.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::rowcopyto(): cublasZcopy() failed.");
	  throw CUDAException("CUBLAS cublasZcopy() failed.");
	}
	
      }
      else{
#pragma omp parallel for schedule(auto)
	for(unsigned int i=x1;i<x2;i++)
	  v[i - x1] = data[y + i*numRows];
      }
      
	
#else      
      
      if(v.resize(x2 - x1 + 1) != x2 - x1 + 1){
	whiteice::logging.error("matrix::rowcopyto(): v.resize() failed.");
	throw std::bad_alloc();
      }
      
      
      if(typeid(T) == typeid(blas_real<float>)){
	cblas_scopy(x2 - x1 + 1, (const float*)&(data[y*numCols + x1]), 1, (float*)v.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cblas_ccopy(x2 - x1 + 1, (const float*)&(data[y*numCols + x1]), 1, (float*)v.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dcopy(x2 - x1 + 1, (const double*)&(data[y*numCols + x1]), 1, (double*)v.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cblas_zcopy(x2 - x1 + 1, (const double*)&(data[y*numCols + x1]), 1, (double*)v.data, 1);
      }
      else{ // generic vector copy
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=x1;i<=x2;i++)
	  v[i - x1] = data[y*numCols + x1];
	
      }
#endif
    }
    
    
    // copies column data to a given vector, M(y1:y2,x) -> v
    template <typename T>
    void matrix<T>::colcopyto(vertex<T>& v, unsigned int x, unsigned int y1, unsigned int y2)
      const 
    {
      if(y2 >= numRows) y2 = numRows - 1;
      
      if(y2 < y1 || y1 >= numRows || x >= numCols){
	whiteice::logging.error("matrix::colcopyto(): bad index parameter to matrix.");
	throw std::out_of_range("colnorm(): bad indeces to matrix");
      }
      
      
#ifdef CUBLAS
      y2++;

      if(v.resize(y2-y1) != (y2-y1)){
	whiteice::logging.error("matrix::colcopyto(): v.resize() failed.");
	throw std::bad_alloc();
      }

      if(typeid(T) == typeid(blas_real<float>)){
	
	auto s = cublasScopy(cublas_handle, (int)(y2-y1),
			     (const float*)&(data[y1 + x*numRows]), 1,
			     (float*)v.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::colcopyto(): cublasScopy() failed.");
	  throw CUDAException("CUBLAS cublasScopy() failed.");
	}
	
      }
      else if(typeid(T) == typeid(blas_real<double>)){

	auto s = cublasDcopy(cublas_handle, (int)(y2-y1),
			     (const double*)&(data[y1 + x*numRows]), 1,
			     (double*)v.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::colcopyto(): cublasDcopy() failed.");
	  throw CUDAException("CUBLAS cublasDcopy() failed.");
	}
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){

	auto s = cublasCcopy(cublas_handle, (int)(y2-y1),
			     (const cuComplex*)&(data[y1 + x*numRows]), 1,
			     (cuComplex*)v.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::colcopyto(): cublasCcopy() failed.");
	  throw CUDAException("CUBLAS cublasCcopy() failed.");
	}
	
      }
      else if(typeid(T) == typeid(blas_complex<double>)){

	auto s = cublasZcopy(cublas_handle, (int)(y2-y1),
			     (const cuDoubleComplex*)&(data[y1 + x*numRows]), 1,
			     (cuDoubleComplex*)v.data, 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::colcopyto(): cublasZcopy() failed.");
	  throw CUDAException("CUBLAS cublasZcopy() failed.");
	}
	
      }
      else{
#pragma omp parallel for schedule(auto)
	for(unsigned int i=y1;i<y2;i++)
	  v[i - y1] = data[i + x*numRows];
      }

#else      
      
      if(v.resize(y2 - y1 + 1) != y2 - y1 + 1)
	throw std::bad_alloc();
      
      
      if(typeid(T) == typeid(blas_real<float>)){
	cblas_scopy(y2 - y1 + 1, (const float*)&(data[y1*numCols + x]), numCols, (float*)v.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cblas_ccopy(y2 - y1 + 1, (const float*)&(data[y1*numCols + x]), numCols, (float*)v.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dcopy(y2 - y1 + 1, (const double*)&(data[y1*numCols + x]), numCols, (double*)v.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cblas_zcopy(y2 - y1 + 1, (const double*)&(data[y1*numCols + x]), numCols, (double*)v.data, 1);
      }
      else{ // generic copy

#pragma omp parallel for schedule(auto)
	for(unsigned int i=y1;i<=y2;i++)
	  v[i - y1] = data[x + i*numCols];
	
      }
#endif
    }
    
    
    
    template <typename T>
    void matrix<T>::rowcopyfrom(const vertex<T>& v,
				unsigned int y,
				unsigned int x1, unsigned int x2)
      
    {
      if(x2 >= numCols) x2 = numCols - 1;
      
      if(x2 < x1 || x1 >= numCols || y >= numRows || v.size() != x2 - x1 + 1){

	printf("rowcopyfrom(): %d %d %d %d %d %d\n",
	       y, x1, x2, numRows, numCols, v.size());
	
	whiteice::logging.error("matrix::rowcopyfrom(): bad index parameter to matrix.");

	assert(0);
	
	throw std::out_of_range("rowcopyfrom(): bad indeces to matrix");
      }
      
#ifdef CUBLAS
      x2++; // we use [x1,x2[ range where x2 element is not part of the range

      if(typeid(T) == typeid(blas_real<float>)){
	
	auto s = cublasScopy(cublas_handle, (int)(x2-x1),
			     (const float*)v.data, 1,
			     (float*)&(data[y + x1*numRows]), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::rowcopyfrom(): cublasScopy() failed.");
	  throw CUDAException("CUBLAS cublasScopy() failed.");
	}
	
      }
      else if(typeid(T) == typeid(blas_real<double>)){

	auto s = cublasDcopy(cublas_handle, (int)(x2-x1),
			     (const double*)v.data, 1,
			     (double*)&(data[y + x1*numRows]), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::rowcopyfrom(): cublasDcopy() failed.");
	  throw CUDAException("CUBLAS cublasDcopy() failed.");
	}
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){

	auto s = cublasCcopy(cublas_handle, (int)(x2-x1),
			     (const cuComplex*)v.data, 1,
			     (cuComplex*)&(data[y + x1*numRows]), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::rowcopyfrom(): cublasCcopy() failed.");
	  throw CUDAException("CUBLAS cublasCcopy() failed.");
	}
	
      }
      else if(typeid(T) == typeid(blas_complex<double>)){

	auto s = cublasZcopy(cublas_handle, (int)(x2-x1),
			     (const cuDoubleComplex*)v.data, 1,
			     (cuDoubleComplex*)&(data[y + x1*numRows]), numRows);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::rowcopyfrom(): cublasZcopy() failed.");
	  throw CUDAException("CUBLAS cublasZcopy() failed.");
	}
	
      }
      else{
#pragma omp parallel for schedule(auto)
	for(unsigned int i=x1;i<x2;i++)
	  data[y + i*numRows] = v[i - x1];
      }

#else      
      
      if(typeid(T) == typeid(blas_real<float>)){
	cblas_scopy(x2 - x1 + 1, (const float*)v.data, 1, (float*)&(data[y*numCols + x1]), 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cblas_ccopy(x2 - x1 + 1, (const float*)v.data, 1, (float*)&(data[y*numCols + x1]), 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dcopy(x2 - x1 + 1, (const double*)v.data, 1, (double*)&(data[y*numCols + x1]), 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cblas_zcopy(x2 - x1 + 1, (const double*)v.data, 1, (double*)&(data[y*numCols + x1]), 1);
      }
      else{ // generic vector copy

#pragma omp parallel for schedule(auto)
	for(unsigned int i=x1;i<=x2;i++)
	  data[y*numCols + i] = v[i - x1];
	
      }
#endif
    }
    
    
    template <typename T>
    void matrix<T>::colcopyfrom(const vertex<T>& v, unsigned int x, unsigned int y1, unsigned int y2)
      
    {
      if(y2 >= numRows) y2 = numRows - 1;
      
      if(y2 < y1 || y1 >= numRows || x >= numCols || v.size() != y2 - y1 + 1){
	whiteice::logging.error("matrix::colcopyfrom(): bad index parameter to matrix.");
	throw std::out_of_range("colnorm(): bad indeces to matrix");
      }
      
#ifdef CUBLAS
      y2++;

      if(typeid(T) == typeid(blas_real<float>)){
	
	auto s = cublasScopy(cublas_handle, (int)(y2-y1),
			     (const float*)v.data, 1,
			     (float*)&(data[y1 + x*numRows]), 1);
	gpu_sync();
			     

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::colcopyfrom(): cublasScopy() failed.");
	  throw CUDAException("CUBLAS cublasScopy() failed.");
	}
	
      }
      else if(typeid(T) == typeid(blas_real<double>)){

	auto s = cublasDcopy(cublas_handle, (int)(y2-y1),
			     (const double*)v.data, 1,
			     (double*)&(data[y1 + x*numRows]), 1);
	gpu_sync();
			     

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::colcopyfrom(): cublasDcopy() failed.");
	  throw CUDAException("CUBLAS cublasDcopy() failed.");
	}
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	auto s = cublasCcopy(cublas_handle, (int)(y2-y1),
			     (const cuComplex*)v.data, 1,
			     (cuComplex*)&(data[y1 + x*numRows]), 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::colcopyfrom(): cublasCcopy() failed.");
	  throw CUDAException("CUBLAS cublasCcopy() failed.");
	}
	
      }
      else if(typeid(T) == typeid(blas_complex<double>)){

	auto s = cublasZcopy(cublas_handle, (int)(y2-y1),
			     (const cuDoubleComplex*)v.data, 1,
			     (cuDoubleComplex*)&(data[y1 + x*numRows]), 1);
	gpu_sync();

	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("matrix::colcopyfrom(): cublasZcopy() failed.");
	  throw CUDAException("CUBLAS cublasZcopy() failed.");
	}
	
      }
      else{
#pragma omp parallel for schedule(auto)
	for(unsigned int i=y1;i<y2;i++)
	  data[i + x*numRows] = v[i - y1];
      }
      
#else
      
      if(typeid(T) == typeid(blas_real<float>)){
	cblas_scopy(y2 - y1 + 1, (const float*)v.data, 1, (float*)&(data[y1*numCols + x]), numCols);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cblas_ccopy(y2 - y1 + 1, (const float*)v.data, 1, (float*)&(data[y1*numCols + x]), numCols);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dcopy(y2 - y1 + 1, (const double*)v.data, 1, (double*)&(data[y1*numCols + x]), numCols);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cblas_zcopy(y2 - y1 + 1, (const double*)v.data, 1, (double*)&(data[y1*numCols + x]), numCols);
      }
      else{ // generic length calculation

#pragma omp parallel for schedule(auto)
	for(unsigned int i=y1;i<=y2;i++)
	  data[x + i*numCols] = v[i - y1];
	
      }

#endif
    }
    
    
    // creates submatrix M from matrix ([x0,x0+xs-1],[y0:y0+ys-1])
    template <typename T>
    bool matrix<T>::submatrix(matrix<T>& M,
			      unsigned int x0, unsigned int y0,
			      unsigned int xs, unsigned int ys) const
    {
      if(x0+xs > numCols || y0+ys > numRows){
	whiteice::logging.warn("matrix::submatrix(): bad submatrix parameters/fail.");
	return false;
      }

      if(xs == 0 || ys == 0){
	whiteice::logging.warn("matrix::submatrix(): bad submatrix parameters/fail.");
	return false;
      }

      if(M.xsize() != xs || M.ysize() != ys){
	if(M.resize(ys, xs) == false){
	  whiteice::logging.warn("matrix::submatrix(): resize()/function failed.");
	  return false;
	}
      }
      
#ifdef CUBLAS

      const T* from = this->data + (y0 + x0*numRows);
      T* to = M.data;

      // FIXME change to use CUDA 2d array copy for optimized copy
      for(unsigned int i=0;i<xs;i++){
	
	auto s = cudaMemcpy(to, from, ys*sizeof(T), cudaMemcpyDeviceToDevice);

	if(s != cudaSuccess){
	  whiteice::logging.error("matrix::submatrix(): cudaMemcopy() failed.");
	  throw CUDAException("CUBLAS cudaMemcpy() failed.");
	}

	from += numRows;
	to += ys;
      }

      gpu_sync();

      return true;
      
#else
      T* from = this->data + x0 + y0*numCols;
      T* to   = M.data;      
      
      for(unsigned int j=0;j<ys;j++){
	memcpy(to, from, xs*sizeof(T));
	from += numCols;
	to   += xs;
      }
      
      return true;
#endif
    }
    
    
    // writes submatrix M to matrix area ([x0+M.xsize()-1],[y0+M.ysize()-1])
    template <typename T>
    bool matrix<T>::write_submatrix(const matrix<T>& M,
				    unsigned int x0, unsigned int y0)
    {
      const unsigned int ys = M.ysize();
      const unsigned int xs = M.xsize();
      
      if(x0+xs > numCols || y0+ys > numRows){
	whiteice::logging.warn("matrix::write_submatrix(): bad submatrix parameters/fail");
	return false;
      }

#ifdef CUBLAS

      const T* from = M.data;
      T* to = this->data + (y0 + x0*numRows);
      
      
      // FIXME change to use CUDA 2d array copy for optimized copy
      for(unsigned int i=0;i<xs;i++){
	
	auto s = cudaMemcpy(to, from, ys*sizeof(T), cudaMemcpyDeviceToDevice);
	
	if(s != cudaSuccess){
	  gpu_sync();
	  whiteice::logging.error("matrix::write_submatrix(): cudaMemcopy() failed.");
	  throw CUDAException("CUBLAS cudaMemcpy() failed.");
	}

	to += numRows;
	from += ys;
      }

      gpu_sync();

      return true;
#else
      
      T* from = M.data;
      T* to   = this->data + x0 + y0*numCols;
      
      for(unsigned int j=0;j<ys;j++){
	memcpy(to, from, xs*sizeof(T));
	from += xs;
	to   += numCols;
      }

      return true;
#endif
    }
    
    
    
    // writes and reads matrix data to/from vertex
    template <typename T>
    bool matrix<T>::save_to_vertex(vertex<T>& out,
				   unsigned int x0) const
    {
#ifdef CUBLAS
      // NOTE: we store data to vertex as we would have row major matrixes in cublas

      if(out.size() < x0 + numCols*numRows)
	out.resize(x0 + numCols*numRows);
      
      
      for(unsigned int j=0;j<numRows;j++){
	//memcpy(&(out[index]), data[j], numCols*sizeof(T), j_increment += numRows);
	//index += numCols

	if(typeid(T) == typeid(blas_real<float>)){

	  auto s = cublasScopy(cublas_handle, numCols,
			       (const float*)&(data[j]), numRows,
			       (float*)&(out[x0 + j*numCols]), 1);
	  gpu_sync();

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::save_to_vertex(): cublasScopy() failed.");
	    throw CUDAException("CUBLAS cublasScopy() failed.");
	  }
	  
	}
	else if(typeid(T) == typeid(blas_real<double>)){

	  auto s = cublasDcopy(cublas_handle, numCols,
			       (const double*)&(data[j]), numRows,
			       (double*)&(out[x0 + j*numCols]), 1);
	  gpu_sync();

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::save_to_vertex(): cublasDcopy() failed.");
	    throw CUDAException("CUBLAS cublasDcopy() failed.");
	  }
	  
	}
	else if(typeid(T) == typeid(blas_complex<float>)){

	  auto s = cublasCcopy(cublas_handle, numCols,
			       (const cuComplex*)&(data[j]), numRows,
			       (cuComplex*)&(out[x0 + j*numCols]), 1);
	  gpu_sync();

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::save_to_vertex(): cublasCcopy() failed.");
	    throw CUDAException("CUBLAS cublasCcopy() failed.");
	  }
	  
	}
	else if(typeid(T) == typeid(blas_complex<double>)){

	  auto s = cublasZcopy(cublas_handle, numCols,
			       (const cuDoubleComplex*)&(data[j]), numRows,
			       (cuDoubleComplex*)&(out[x0 + j*numCols]), 1);
	  gpu_sync();

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::save_to_vertex(): cublasZcopy() failed.");
	    throw CUDAException("CUBLAS cublasZcopy() failed.");
	  }
	  
	}
	else{
	  // copies whole matrix a whole and don't iterate the loop
	  
	  unsigned int index = x0;
	  
	  for(unsigned int j=0;j<numRows;j++){
	    for(unsigned int i=0;i<numCols;i++, index++){
	      out[index] = (*this)(j,i);
	    }
	  }

	  return true;
	}
	
      }

      return true;
      
#else
      if(out.size() < x0 + numCols*numRows)
	out.resize(x0 + numCols*numRows);
      
      memcpy(&(out.data[x0]), this->data, numCols*numRows*sizeof(T));
      
      return true;
#endif
    }
    
    
    template <typename T>
    bool matrix<T>::load_from_vertex(const vertex<T>& in,
				     unsigned int x0)
    {
      if(in.size() < numCols*numRows + x0)
	return false;

#ifdef CUBLAS

      for(unsigned int j=0;j<numRows;j++){
	//memcpy(data[j], &(out[index]), numCols*sizeof(T), j_increment += numRows)
	//index += numCols

	if(typeid(T) == typeid(blas_real<float>)){

	  auto s = cublasScopy(cublas_handle, numCols,
			       (const float*)&(in[x0 + j*numCols]), 1,
			       (float*)&(data[j]), numRows);
	  gpu_sync();

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::load_from_vertex(): cublasScopy() failed.");
	    throw CUDAException("CUBLAS cublasScopy() failed.");
	  }
	  
	}
	else if(typeid(T) == typeid(blas_real<double>)){

	  auto s = cublasDcopy(cublas_handle, numCols,
			       (const double*)&(in[x0 + j*numCols]), 1,
			       (double*)&(data[j]), numRows);
	  gpu_sync();

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::load_from_vertex(): cublasDcopy() failed.");
	    throw CUDAException("CUBLAS cublasDcopy() failed.");
	  }
	  
	}
	else if(typeid(T) == typeid(blas_complex<float>)){

	  auto s = cublasCcopy(cublas_handle, numCols,
			       (const cuComplex*)&(in[x0 + j*numCols]), 1,
			       (cuComplex*)&(data[j]), numRows);
	  gpu_sync();

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::load_from_vertex(): cublasCcopy() failed.");
	    throw CUDAException("CUBLAS cublasCcopy() failed.");
	  }
	  
	}
	else if(typeid(T) == typeid(blas_complex<double>)){

	  auto s = cublasZcopy(cublas_handle, numCols,
			       (const cuDoubleComplex*)&(in[x0 + j*numCols]), 1,
			       (cuDoubleComplex*)&(data[j]), numRows);
	  gpu_sync();

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::load_from_vertex(): cublasZcopy() failed.");
	    throw CUDAException("CUBLAS cublasZcopy() failed.");
	  }
	  
	}
	else{
	  // copies whole matrix a whole and don't iterate the loop (returns true)
	  
	  unsigned int index = x0;
	  
	  for(unsigned int j=0;j<numRows;j++){
	    for(unsigned int i=0;i<numCols;i++, index++){
	      (*this)(j,i) = in[index];
	    }
	  }

	  return true;
	}
	
      }

      return true;
      
#else
      memcpy(this->data, &(in.data[x0]), numCols*numRows*sizeof(T));
      
      return true;
#endif
    }
    
    
    template <typename T>
    void matrix<T>::normalize() 
    {
#ifdef CUBLAS

      if(typeid(T) == typeid(blas_real<float>)){

	// MULTITHREAD DISABLED: does OpenMP work with exception handling??
	for(unsigned int j=0;j<numRows;j++){
	  float len;

	  auto s = cublasSnrm2(cublas_handle, numCols,
			       (const float*)&(data[j]), numRows,
			       (float*)&len);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::normalize(): cublasSnrm2() failed.");
	    throw CUDAException("CUBLAS cublasSnrm2() failed.");
	  }

	  if(len <= 0.0f)
	    continue; // skip zero length row vectors
	  else
	    len = 1.0f/whiteice::math::sqrt(len);

	  s = cublasSscal(cublas_handle, numCols,
			  (const float*)&len,
			  (float*)&(data[j]), numRows);
	  gpu_sync();

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::normalize(): cublasSscal() failed.");
	      throw CUDAException("CUBLAS cublasSscal() failed.");
	  }
	}
	
      }
      else if(typeid(T) == typeid(blas_real<double>)){

	for(unsigned int j=0;j<numRows;j++){
	  double len;

	  auto s = cublasDnrm2(cublas_handle, numCols,
			       (const double*)&(data[j]), numRows,
			       (double*)&len);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::normalize(): cublasDnrm2() failed.");
	    throw CUDAException("CUBLAS cublasDnrm2() failed.");
	  }

	  if(len <= 0.0)
	    continue; // skip zero length row vectors
	  else
	    len = 1.0f/whiteice::math::sqrt(len);

	  s = cublasDscal(cublas_handle, numCols,
			  (const double*)&len,
			  (double*)&(data[j]), numRows);
	  gpu_sync();

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::normalize(): cublasDscal() failed.");
	      throw CUDAException("CUBLAS cublasDscal() failed.");
	  }
	}
		
      }
      else if(typeid(T) == typeid(blas_complex<float>)){

	for(unsigned int j=0;j<numRows;j++){
	  float len;

	  auto s = cublasScnrm2(cublas_handle, numCols,
				(const cuComplex*)&(data[j]), numRows,
				(float*)&len);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::normalize(): cublasScnrm2() failed.");
	    throw CUDAException("CUBLAS cublasScnrm2() failed.");
	  }

	  if(len <= 0.0f)
	    continue; // skip zero length row vectors
	  else
	    len = 1.0f/whiteice::math::sqrt(len);

	  s = cublasCsscal(cublas_handle, numCols,
			   (const float*)&len,
			   (cuComplex*)&(data[j]), numRows);
	  gpu_sync();

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::normalize(): cublasCsscal() failed.");
	      throw CUDAException("CUBLAS cublasCsscal() failed.");
	  }
	}
	
      }
      else if(typeid(T) == typeid(blas_complex<double>)){

	for(unsigned int j=0;j<numRows;j++){
	  double len;

	  auto s = cublasDznrm2(cublas_handle, numCols,
				(const cuDoubleComplex*)&(data[j]), numRows,
				(double*)&len);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::normalize(): cublasDznrm2() failed.");
	    throw CUDAException("CUBLAS cublasDznrm2() failed.");
	  }

	  if(len <= 0.0f)
	    continue; // skip zero length row vectors
	  else
	    len = 1.0f/whiteice::math::sqrt(len);

	  s = cublasZdscal(cublas_handle, numCols,
			   (const double*)&len,
			   (cuDoubleComplex*)&(data[j]), numRows);
	  gpu_sync();

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("matrix::normalize(): cublasZdscal() failed.");
	      throw CUDAException("CUBLAS cublasZdscal() failed.");
	  }
	}
	
      }
      else{
	
	for(unsigned int j=0;j<numRows;j++){
	  T len = T(0.0f);
	  
	  for(unsigned int i=0;i<numCols;i++){
	    len += (data[i*numRows+j])*whiteice::math::conj(data[i*numRows+j]);
	  }
	  
	  len = whiteice::math::sqrt(len);
	  
	  if(len != T(0.0f)){
	    for(unsigned int i=0;i<numCols;i++){
	      data[i*numRows+j] /= len;
	    }
	  }
	}
	
      }

#else
      // normalizes each row to have unit length
      // normalization of each column: transpose + normalize  + transpose is slow
      // TODO: write code (and test it) which normalizes each column to have unit length
      // (main difference , incX is numCols and length is numRows etc.
      
      if(typeid(T) == typeid(blas_real<float>)){
	float f;
	
	for(unsigned int j=0;j<numRows;j++){
	  f = cblas_snrm2(numCols, (float*)&(data[j*numCols]), 1);
	  
	  if(f == 0.0f) continue;
	  else f = 1.0f/whiteice::math::sqrt(f);
	  
	  cblas_sscal(numCols,  f, (float*)&(data[j*numCols]), 1);
	}
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	float f;
	
	for(unsigned int j=0;j<numRows;j++){
	  f = cblas_scnrm2(numCols, (float*)&(data[j*numCols]), 1);
	  
	  if(f == 0.0f) continue;
	  else f = 1.0f/whiteice::math::sqrt(f);
	  
	  cblas_csscal(numCols,  f, (float*)&(data[j*numCols]), 1);
	}	
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double f;
	
	for(unsigned int j=0;j<numRows;j++){
	  f = cblas_dnrm2(numCols, (double*)&(data[j*numCols]), 1);
	  
	  if(f == 0.0) continue;
	  else f = 1.0/whiteice::math::sqrt(f);
	  
	  cblas_dscal(numCols,  f, (double*)&(data[j*numCols]), 1);
	}
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	double f;
	
	for(unsigned int j=0;j<numRows;j++){
	  f = cblas_dznrm2(numCols, (double*)&(data[j*numCols]), 1);
	  
	  if(f == 0.0) continue;
	  else f = 1.0/whiteice::math::sqrt(f);
	  
	  cblas_zdscal(numCols,  f, (double*)&(data[j*numCols]), 1);
	}
      }
      else{ // generic normalization of rows

#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<numRows;j++){
	  T len = T(0.0);
	  
	  for(unsigned int i=0;i<numCols;i++){
	    len += (data[i+j*numCols])*whiteice::math::conj(data[i+j*numCols]);
	  }
	  
	  len = whiteice::math::sqrt(len);
	  
	  if(len != T(0.0)){
	    for(unsigned int i=0;i<numCols;i++){
	      data[i+j*numCols] /= len;
	    }
	  }
	}
	
      }
      
#endif
    }
    
    
    template <typename T>
    bool matrix<T>::comparable() 
    {
      return false;
    }

    template <typename T>
    void matrix<T>::toString(std::string& line) const 
    {
      if(this->ysize() == 0 && this->xsize() == 0){ line = ""; return; }

      char buffer[30];

      if(typeid(T) == typeid(blas_real<float>) ||
	 typeid(T) == typeid(blas_real<double>)){
	
	if(this->ysize() == 1 && this->xsize() == 1){
	  line = "";
	  double temp = 0.0;
	  whiteice::math::convert(temp, (*this)(0,0));
	  snprintf(buffer, 30, "%f", temp);
	  line += buffer;
	  return;
	}
	
	line = "[";
	double temp = 0.0;
	
	for(unsigned int j=0;j<this->ysize();j++){
	  for(unsigned int i=0;i<this->xsize();i++){
	    whiteice::math::convert(temp, (*this)(j,i));
	    snprintf(buffer, 30, " %f", temp);
	    line += buffer;
	  }
	  
	  line += "; ";
	}
	
	line += "]";

	return;
      }
      else{ // prints complex numbers

	if(this->ysize() == 1 && this->xsize() == 1){
	  line = "";
	  auto r = whiteice::math::real(data[0]);
	  auto i = whiteice::math::conj(data[0]);
	  double temp, temp2;
	  whiteice::math::convert(temp, r);
	  whiteice::math::convert(temp2, i);
	  
	  snprintf(buffer, 30, "%f+%fi", temp, temp2);
	  line += buffer;
	  return;
	}
	
	line = "[";
	double temp, temp2;
	
	for(unsigned int j=0;j<this->ysize();j++){
	  for(unsigned int i=0;i<this->xsize();i++){
	    auto rv = whiteice::math::real((*this)(j,i));
	    auto iv = whiteice::math::imag((*this)(j,i));
	    
	    whiteice::math::convert(temp, rv);
	    whiteice::math::convert(temp2, iv);
	    snprintf(buffer, 30, " %f+%fi", temp, temp2);
	    line += buffer;
	  }
	  
	  line += "; ";
	}
	
	line += " ]";

	return;
      }
    }
    
    ////////////////////////////////////////////////////////////
    // matrix data compression
    // note: compressor destroys possible memory
    // aligmentations
    
    
    template <typename T>
    bool matrix<T>::compress() 
    {
      if(compressor != 0) return false; // already compressed

#ifdef CUBLAS
      whiteice::logging.error("FIXME Memory compression is not supported with GPU memory.");
      return false;
#else
      compressor = new MemoryCompressor();
      
      compressor->setMemory(data, sizeof(T)*numRows*numCols);
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
    bool matrix<T>::decompress() 
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
    bool matrix<T>::iscompressed() const 
    {
      return (compressor != 0);
    }
    
    
    template <typename T>
    float matrix<T>::ratio() const 
    {
      if(compressor == 0) return 1.0f;

#ifdef CUBLAS
      whiteice::logging.error("FIXME Memory compression is not supported with GPU memory.");
      return 1.0f;
#else
      return ( ((float)compressor->getTargetSize()) / ((float)(numRows*numCols*sizeof(T))) );
#endif
    }
    
    
    ////////////////////////////////////////////////////////////
    
    /***************************************************/
    
    
    template <typename T>
    std::ostream& operator<<(std::ostream& ios, const matrix<T>& M)
    {
      ios << "[";
    
      for(unsigned int j=0;j<M.ysize();j++){
	for(unsigned int i=0;i<M.xsize();i++)
	{
	  ios << " " << M(j,i);
	}
	
	ios << "; ";
      }
    
      ios << " ]";
    
      return ios;
    }
    
    
    
    // explicit template instantations
    
    template class matrix<float>;    
    template class matrix<double>;
    template class matrix<complex<float> >;
    template class matrix<complex<double> >;
    
    //template class matrix<int>;
    //template class matrix<char>;
    //template class matrix<unsigned int>;
    //template class matrix<unsigned char>;
        
    template class matrix< blas_real<float> >;
    template class matrix< blas_real<double> >;
    template class matrix< blas_complex<float> >;
    template class matrix< blas_complex<double> >;

    template class matrix< superresolution< blas_real<float>,
					    modular<unsigned int> > >;
    template class matrix< superresolution< blas_real<double>,
					    modular<unsigned int> > >;

    template class matrix< superresolution< blas_complex<float>,
					    modular<unsigned int> > >;
    template class matrix< superresolution< blas_complex<double>,
					    modular<unsigned int> > >;
    
    
    template matrix<float> operator*<float>(const float&, const matrix<float>&) ;
    template matrix<double> operator*<double>(const double&, const matrix<double>&) ;
    template matrix<complex<float> > operator*<complex<float> >(const complex<float>&, const matrix<complex<float> >&)
      ;    
    template matrix<complex<double> > operator*<complex<double> >(const complex<double>&, const matrix<complex<double> >&)
      ;
    
    //template matrix<int> operator*<int>(const int&, const matrix<int>&) ;
    //template matrix<char> operator*<char>(const char&, const matrix<char>&) ;
    //template matrix<unsigned int> operator*<unsigned int>(const unsigned int&, const matrix<unsigned int>&)
    //  ;
    //template matrix<unsigned char> operator*<unsigned char>(const unsigned char&, const matrix<unsigned char>&)
    //  ;
    
    
    template matrix<blas_real<float> > operator*<blas_real<float> >
      (const blas_real<float>&, const matrix<blas_real<float> >&) ;
       
    template matrix<blas_real<double> > operator*<blas_real<double> >
      (const blas_real<double>&, const matrix<blas_real<double> >&) ;
    
    
    template matrix<blas_complex<float> > operator*<blas_complex<float> >
      (const blas_complex<float>&, const matrix<blas_complex<float> >&) ;
    template matrix<blas_complex<double> > operator*<blas_complex<double> >
      (const blas_complex<double>&, const matrix<blas_complex<double> >&) ;


    template matrix<superresolution<blas_real<float>, modular<unsigned int> > > operator*<superresolution<blas_real<float>, modular<unsigned int> > >
    (const superresolution<blas_real<float>, modular<unsigned int> >&,
     const matrix<superresolution<blas_real<float>, modular<unsigned int> > >&) ;
    template matrix<superresolution<blas_real<double>, modular<unsigned int> > > operator*<superresolution<blas_real<double>, modular<unsigned int> > >
    (const superresolution<blas_real<double>, modular<unsigned int> >&,
     const matrix<superresolution<blas_real<double>, modular<unsigned int> > >&) ;
    
    template matrix<superresolution<blas_complex<float>, modular<unsigned int> > > operator*<superresolution<blas_complex<float>, modular<unsigned int> > >
    (const superresolution<blas_complex<float>, modular<unsigned int> >&,
     const matrix<superresolution<blas_complex<float>, modular<unsigned int> > >&) ;
    template matrix<superresolution<blas_complex<double>, modular<unsigned int> > > operator*<superresolution<blas_complex<double>, modular<unsigned int> > >
    (const superresolution<blas_complex<double>, modular<unsigned int> >&,
     const matrix<superresolution<blas_complex<double>, modular<unsigned int> > >&) ;
    
        
    template std::ostream& operator<< <float>(std::ostream& ios, const matrix<float>& M);
    template std::ostream& operator<< <double>(std::ostream& ios, const matrix<double>& M);
    template std::ostream& operator<< <complex<float> >(std::ostream& ios, const matrix<complex<float> >& M);
    template std::ostream& operator<< <complex<double> >(std::ostream& ios, const matrix<complex<double> >& M);
    //template std::ostream& operator<< <int>(std::ostream& ios, const matrix<int>& M);
    //template std::ostream& operator<< <char>(std::ostream& ios, const matrix<char>& M);
    //template std::ostream& operator<< <unsigned int>(std::ostream& ios, const matrix<unsigned int>& M);
    //template std::ostream& operator<< <unsigned char>(std::ostream& ios, const matrix<unsigned char>& M);
    template std::ostream& operator<< <blas_real<float> >(std::ostream& ios, const matrix<blas_real<float> >& M);
    template std::ostream& operator<< <blas_real<double> >(std::ostream& ios, const matrix<blas_real<double> >& M);
    template std::ostream& operator<< <blas_complex<float> >(std::ostream& ios, const matrix<blas_complex<float> >& M);
    template std::ostream& operator<< <blas_complex<double> >(std::ostream& ios, const matrix<blas_complex<double> >& M);

    template std::ostream& operator<< <superresolution< blas_real<float>, modular<unsigned int> > >(std::ostream& ios, const matrix<superresolution< blas_real<float>, modular<unsigned int> > >& M);
    template std::ostream& operator<< <superresolution<blas_real<double>, modular<unsigned int> > >(std::ostream& ios, const matrix<superresolution< blas_real<double>,modular<unsigned int> > >& M);

    template std::ostream& operator<< <superresolution< blas_complex<float>, modular<unsigned int> > >(std::ostream& ios, const matrix<superresolution< blas_complex<float>, modular<unsigned int> > >& M);
    template std::ostream& operator<< <superresolution<blas_complex<double>, modular<unsigned int> > >(std::ostream& ios, const matrix<superresolution< blas_complex<double>,modular<unsigned int> > >& M);
    
  };
};
  
  
#endif
  
