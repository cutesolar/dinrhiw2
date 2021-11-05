
// FIXME: CUDA uses cudaMemcpy() when it should use cudaMemmove() because memory areas overlap!

#ifndef correlation_cpp
#define correlation_cpp

#include "correlation.h"

#include "matrix.h"
#include "vertex.h"
#include "eig.h"
#include "Log.h"

#include <typeinfo>
#include <string.h>

namespace whiteice
{
  namespace math
  {
    
    // calculates autocorrelation matrix from the given data
    template <typename T>
    bool autocorrelation(matrix<T>& R, const std::vector< vertex<T> >& data)
    {
      if(data.size() <= 0)
	return false;
      
      typename std::vector< vertex<T> >::const_iterator i = data.begin();
      const unsigned int N = data[0].size();
      
      if(R.resize(N, N) == false) return false;
      R.zero();
      
      T invs = T(1.0f) / T(data.size());

#ifdef CUBLAS
      
      if(typeid(T) == typeid(blas_real<float>)){	
	      
	while(i != data.end()){
	  
	  auto s = cublasSspr(cublas_handle, CUBLAS_FILL_MODE_LOWER, (int)N,
			      (const float*)&invs, (const float*)(i->data), 1,
			      (float*)R.data);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("autocorrelation(): cublasSspr() failed.");
	    throw CUDAException("CUBLAS cublasSspr() call failed.");
	  }
	  
	  i++;
	}

	// NOT OPTIMIZED
	{
	  // last element of the matrix
	  unsigned int non_packed_index = N*N;
	  unsigned int packed_index = N*(N+1)/2;
	  
	  for(unsigned int i=0;i<(N-1);i++){ // (N-i):th column
	    auto diag_non_packed_index = non_packed_index - (i+1);
	    packed_index -= (i+1);
	    
	    auto e = cudaMemcpy(&(R.data[diag_non_packed_index]), &(R.data[packed_index]),
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
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<N;i++)
	  for(unsigned int j=(i+1);j<N;j++)
	    R(i,j) = R(j,i);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){

	float invsf;
	whiteice::math::convert(invsf, invs);
	
	while(i != data.end()){
	  auto s = cublasChpr(cublas_handle, CUBLAS_FILL_MODE_LOWER, (int)N,
			      (const float*)&invsf, (const cuComplex*)(i->data), 1,
			      (cuComplex*)R.data);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("autocorrelation(): cublasChpr() failed.");
	    throw CUDAException("CUBLAS cublasChpr() call failed.");
	  }
	  
	  i++;
	}

	
	{
	  // last element of the matrix
	  unsigned int non_packed_index = N*N;
	  unsigned int packed_index = N*(N+1)/2;
	  
	  for(unsigned int i=0;i<(N-1);i++){ // (N-i):th column
	    auto diag_non_packed_index = non_packed_index - (i+1);
	    packed_index -= (i+1);
	    
	    auto e = cudaMemcpy(&(R.data[diag_non_packed_index]), &(R.data[packed_index]),
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
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<N;i++)
	  for(unsigned int j=(i+1);j<N;j++)
	    R(i,j) = R(j,i);
	
      }
      else if(typeid(T) == typeid(blas_real<double>)){

	while(i != data.end()){
	  auto s = cublasDspr(cublas_handle, CUBLAS_FILL_MODE_LOWER, (int)N,
			      (const double*)&invs, (const double*)(i->data), 1,
			      (double*)R.data);
	  
	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("autocorrelation(): cublasDspr() failed.");
	    throw CUDAException("CUBLAS cublasDspr() call failed.");
	  }
	}

	
	// NOT OPTIMIZED
	{
	  // last element of the matrix
	  unsigned int non_packed_index = N*N;
	  unsigned int packed_index = N*(N+1)/2;
	  
	  for(unsigned int i=0;i<(N-1);i++){ // (N-i):th column
	    auto diag_non_packed_index = non_packed_index - (i+1);
	    packed_index -= (i+1);
	    
	    auto e = cudaMemcpy(&(R.data[diag_non_packed_index]), &(R.data[packed_index]),
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
#pragma omp parallel for schedule(auto)	
	for(unsigned int i=0;i<N;i++)
	  for(unsigned int j=(i+1);j<N;j++)
	    R(i,j) = R(j,i);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){

	double invsf;
	whiteice::math::convert(invsf, invs);
	
	while(i != data.end()){
	  
	  auto s = cublasZhpr(cublas_handle, CUBLAS_FILL_MODE_LOWER, (int)N,
			      (const double*)&invsf,
			      (const cuDoubleComplex*)(i->data), 1,
			      (cuDoubleComplex*)R.data);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("autocorrelation(): cublasZhpr() failed.");
	    throw CUDAException("CUBLAS cublasZhpr() call failed.");
	  }
	  
	  i++;
	}

	// NOT OPTIMIZED
	{
	  // last element of the matrix
	  unsigned int non_packed_index = N*N;
	  unsigned int packed_index = N*(N+1)/2;
	  
	  for(unsigned int i=0;i<(N-1);i++){ // (N-i):th column
	    auto diag_non_packed_index = non_packed_index - (i+1);
	    packed_index -= (i+1);
	    
	    auto e = cudaMemcpy(&(R.data[diag_non_packed_index]), &(R.data[packed_index]),
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
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<N;i++)
	  for(unsigned int j=(i+1);j<N;j++)
	    R(i,j) = R(j,i);
      }
      else{ // generic code
	
	while(i != data.end()){
	  
	  R += (*i).outerproduct((*i));
	  i++;
	}
	
	R *= invs;
	
      }
      
#else
      
      if(typeid(T) == typeid(blas_real<float>)){	
	
	while(i != data.end()){
	  cblas_sspr(CblasRowMajor, CblasUpper, N, 
		     *((float*)&invs), (float*)(i->data), 1, (float*)R.data);
	  i++;
	}

	// FIXED:
	// (N-i):th row FIXME last i=N-1 row update is not needed
	// for(unsigned int i=0;i<=(N-1);i++){

	// converts symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(R.data[r*N + r]),
		  &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		  sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_scopy(N - i - 1, 
		      (float*)&(R.data[i*N + 1 + i]), 1,
		      (float*)&(R.data[(i+1)*N + i]), N);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	while(i != data.end()){
	  cblas_chpr(CblasRowMajor, CblasUpper, N,
		     *((float*)&invs), (float*)(i->data), 1, (float*)R.data);
	  i++;
	}
	
	// converts symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row (FIXED: was: <=)
	  unsigned int r = N - i - 1;
	  memmove(&(R.data[r*N + r]),
		  &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		  sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_ccopy(N - i - 1, 
		      (float*)&(R.data[i*N + 1 + i]), 1,
		      (float*)&(R.data[(i+1)*N + i]), N);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	while(i != data.end()){
	  cblas_dspr(CblasRowMajor, CblasUpper, N, 
		     *((double*)&invs), (double*)(i->data), 1, (double*)(R.data));
	  i++;
	}
	
	// converts symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(R.data[r*N + r]),
		  &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		  sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_dcopy(N - i - 1, 
		      (double*)&(R.data[i*N + 1 + i]), 1,
		      (double*)&(R.data[(i+1)*N + i]), N);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	while(i != data.end()){
	  cblas_zhpr(CblasRowMajor, CblasUpper, N,
		     *((double*)&invs), (double*)(i->data), 1, (double*)R.data);
	  i++;
	}
	
	// converts symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(R.data[r*N + r]),
		  &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		  sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_zcopy(N - i - 1, 
		      (double*)&(R.data[i*N + 1 + i]), 1,
		      (double*)&(R.data[(i+1)*N + i]), N);
      }
      else{ // generic code
	
	while(i != data.end()){
	  
	  R += (*i).outerproduct((*i));
	  i++;
	}
	
	R *= invs;
	
      }

#endif
      
      
      return true;
    }
    
    
    
    // calculates autocorrelation matrix from W matrix'es row vectors
    template <typename T>
    bool autocorrelation(matrix<T>& R, const matrix<T>& W)
    {
      // autocorrelation is R= s * W^t * W, where s is 
      // one divided by number of rows in W
      
      if(W.ysize() <= 0 || W.xsize() <= 0)
	return false;
      
      const unsigned int N = W.ysize(); // meaning is swapped from cblas_ssyrk() parameters
      const unsigned int K = W.xsize(); // (N is number of vectors here and K is dimension)
      
      if(R.resize(K, K) == false) return false;
      R.zero(); // not necessary..

      T s = T(1.0f) / T((double)N);

#ifdef CUBLAS
      
      const T t = T(0.0f);
      
      if(typeid(T) == typeid(blas_real<float>)){

	auto r = cublasSsyrk(cublas_handle,
			     CUBLAS_FILL_MODE_LOWER,
			     CUBLAS_OP_T,
			     K, N,
			     (const float*)&s,
			     (const float*)W.data, W.ysize(),
			     (const float*)&t,
			     (float*)R.data, R.ysize());

	if(r != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("autocorrelation(): cublasSsyrk() failed.");
	  throw CUDAException("CUBLAS cublasSsyrk() call failed.");
	}

	gpu_sync();

	// OPTIMIZE ME:
	// copies lower triangular elements to upper triangular part of the M matrix
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<K;i++)
	  for(unsigned int j=(i+1);j<K;j++)
	    R(i,j) = R(j,i);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){

	float alpha = 0.0f;
	float beta  = 0.0f;

	whiteice::math::convert(alpha, s);
	whiteice::math::convert(beta, t);

	auto r = cublasCherk(cublas_handle,
			     CUBLAS_FILL_MODE_LOWER,
			     CUBLAS_OP_T,
			     K, N,
			     (const float*)&alpha,
			     (const cuComplex*)W.data, W.ysize(),
			     (const float*)&beta,
			     (cuComplex*)R.data, R.ysize());

	if(r != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("autocorrelation(): cublasCherk() failed.");
	  throw CUDAException("CUBLAS cublasCherk() call failed.");
	}

	gpu_sync();

	// OPTIMIZE ME
	// copies lower triangular elements to upper triangular part of the M matrix
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<K;i++)
	  for(unsigned int j=(i+1);j<K;j++)
	    R(i,j) = R(j,i);
      }
      else if(typeid(T) == typeid(blas_real<double>)){

	auto r = cublasDsyrk(cublas_handle,
			     CUBLAS_FILL_MODE_LOWER,
			     CUBLAS_OP_T,
			     K, N,
			     (const double*)&s,
			     (const double*)W.data, W.ysize(),
			     (const double*)&t,
			     (double*)R.data, R.ysize());

	if(r != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("autocorrelation(): cublasDsyrk() failed.");
	  throw CUDAException("CUBLAS cublasDsyrk() call failed.");
	}
	
	gpu_sync();

	// OPTIMIZE ME
	// copies lower triangular elements to upper triangular part of the M matrix
#pragma omp parallel for schedule(auto)	
	for(unsigned int i=0;i<K;i++)
	  for(unsigned int j=(i+1);j<K;j++)
	    R(i,j) = R(j,i);	
	
      }
      else if(typeid(T) == typeid(blas_complex<double>)){

	double alpha = 0.0f;
	double beta  = 0.0f;

	whiteice::math::convert(alpha, s);
	whiteice::math::convert(beta, t);

	auto r = cublasZherk(cublas_handle,
			     CUBLAS_FILL_MODE_LOWER,
			     CUBLAS_OP_T,
			     K, N,
			     (const double*)&alpha,
			     (const cuDoubleComplex*)W.data, W.ysize(),
			     (const double*)&beta,
			     (cuDoubleComplex*)R.data, R.ysize());

	if(r != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("autocorrelation(): cublasCherk() failed.");
	  throw CUDAException("CUBLAS cublasCherk() call failed.");
	}

	gpu_sync();

	// OPTIMIZE ME: cudaScopy()..
	// copies lower triangular elements to upper triangular part of the M matrix
#pragma omp parallel for schedule(auto)	
	for(unsigned int i=0;i<K;i++)
	  for(unsigned int j=(i+1);j<K;j++)
	    R(i,j) = R(j,i);
	
      }
      else{ // generic code
	matrix<T> V = W;
	V.transpose();
	
	R = V * W;
	
	R *= s;
      }
#else
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, K, N,
		    *((float*)&s), (float*)W.data, W.xsize(),
		    0.0f, (float*)R.data, R.xsize());

	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(K-1);i++)
	  cblas_scopy(K - i - 1, 
		      (float*)&(R.data[i*K + 1 + i]), 1,
		      (float*)&(R.data[(i+1)*K + i]), K);

      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_cherk(CblasRowMajor, CblasUpper, CblasTrans, K, N,
		    *((float*)&s), (float*)W.data, W.xsize(),
		    0.0f, (float*)R.data, R.xsize());
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(K-1);i++)
	  cblas_ccopy(K - i - 1, 
		      (float*)&(R.data[i*K + 1 + i]), 1,
		      (float*)&(R.data[(i+1)*K + i]), K);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, K, N,
		    *((double*)&s), (double*)W.data, W.xsize(),
		    0.0, (double*)R.data, R.xsize());
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(K-1);i++)
	  cblas_dcopy(K - i - 1, 
		      (double*)&(R.data[i*K + 1 + i]), 1,
		      (double*)&(R.data[(i+1)*K + i]), K);

      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zherk(CblasRowMajor, CblasUpper, CblasTrans, K, N,
		    *((double*)&s), (double*)W.data, W.xsize(),
		    0.0, (double*)R.data, R.xsize());
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(K-1);i++)
	  cblas_zcopy(K - i - 1, 
		      (double*)&(R.data[i*K + 1 + i]), 1,
		      (double*)&(R.data[(i+1)*K + i]), K);
      }
      else{ // generic code
	matrix<T> V = W;
	V.transpose();
	
	R = V * W;
	
	R *= s;
      }
#endif
      
      return true;
    }
    
    
    
    // mean_covariance_estimate() is now parallelized (requires extra memory)
    template <typename T>
    bool mean_covariance_estimate(vertex<T>& m, matrix<T>& R,
				  const std::vector< vertex<T> >& data)
    {
      if(data.size() <= 0)
	return false;
      
      const unsigned int N = data[0].size();

      // expect's mem allocation to succeed
      if(R.resize(N, N) == false) return false;
      R.zero();
      
      if(m.resize(N) != N) return false;
      m.zero();
      
      T s = T(1.0f) / T(data.size());

#ifdef CUBLAS
      
      if(typeid(T) == typeid(blas_real<float>)){	

#pragma omp parallel shared(m) shared(R)
	{
	  matrix<T> Ri;
	  vertex<T> mi;

	  Ri.resize(N,N);
	  Ri.zero();

	  mi.resize(N);
	  mi.zero();

#pragma omp for nowait schedule(auto)
	  for(unsigned int index=0;index<data.size();index++){
	    auto& v = data[index];
	    
	    auto r = cublasSspr(cublas_handle, CUBLAS_FILL_MODE_LOWER, (int)N,
				(const float*)&s, (const float*)(v.data), 1,
				(float*)Ri.data);
	    
	    if(r != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("mean_covariance_estimate(): cublasSspr() failed.");
	      throw CUDAException("CUBLAS cublasSspr() call failed.");
	    }
	    
	    mi += v;
	  }

#pragma omp critical (mean_covariance_estimate0)
	  {
	    m += mi;
	    R += Ri;
	  }
	}

	
	// NOT OPTIMIZED
	{
	  // last element of the matrix
	  unsigned int non_packed_index = N*N;
	  unsigned int packed_index = N*(N+1)/2;
	  
	  for(unsigned int i=0;i<(N-1);i++){ // (N-i):th column
	    auto diag_non_packed_index = non_packed_index - (i+1);
	    packed_index -= (i+1);
	    
	    auto e = cudaMemcpy(&(R.data[diag_non_packed_index]), &(R.data[packed_index]),
				sizeof(T)*(i+1), cudaMemcpyDeviceToDevice);

	    if(e != cudaSuccess){
	      whiteice::logging.error("mean_covariance_estimate(): cudaMemcpy() failed.");
	      throw CUDAException("CUBLAS cudaMemcpy() call failed.");
	    }

	    non_packed_index -= N;
	  }
	}

	gpu_sync();
	
	// copies lower triangular elements to upper triangular part of the M matrix
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<N;i++)
	  for(unsigned int j=(i+1);j<N;j++)
	    R(i,j) = R(j,i);
	
	m *= s;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){

	float sf;
	whiteice::math::convert(sf, s);

#pragma omp parallel shared(m) shared(R)
	{
	  matrix<T> Ri;
	  vertex<T> mi;
	  
	  Ri.resize(N,N);
	  Ri.zero();
	  
	  mi.resize(N);
	  mi.zero();

#pragma omp for nowait schedule(auto)
	  for(unsigned int index=0;index<data.size();index++){
	    auto& v = data[index];	  
	    
	    auto r = cublasChpr(cublas_handle, CUBLAS_FILL_MODE_LOWER, (int)N,
				(const float*)&sf, (const cuComplex*)(v.data), 1,
				(cuComplex*)Ri.data);
	    
	    if(r != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("mean_covariance_estimate(): cublasChpr() failed.");
	      throw CUDAException("CUBLAS cublasChpr() call failed.");
	    }
	    
	    mi += v;
	  }

#pragma omp critical (mean_covariance_estimate1)
	  {
	    m += mi;
	    R += Ri;
	  }
	}
	
	// NOT OPTIMIZED
	{
	  // last element of the matrix
	  unsigned int non_packed_index = N*N;
	  unsigned int packed_index = N*(N+1)/2;
	  
	  for(unsigned int i=0;i<(N-1);i++){ // (N-i):th column
	    auto diag_non_packed_index = non_packed_index - (i+1);
	    packed_index -= (i+1);
	    
	    auto e = cudaMemcpy(&(R.data[diag_non_packed_index]), &(R.data[packed_index]),
				sizeof(T)*(i+1), cudaMemcpyDeviceToDevice);

	    if(e != cudaSuccess){
	      whiteice::logging.error("mean_covariance_estimate(): cudaMemcpy() failed.");
	      throw CUDAException("CUBLAS cudaMemcpy() call failed.");
	    }

	    non_packed_index -= N;
	  }
	}

	gpu_sync();

	// copies lower triangular elements to upper triangular part of the M matrix
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<N;i++)
	  for(unsigned int j=(i+1);j<N;j++)
	    R(i,j) = R(j,i);
	  
	m *= s;
      }
      else if(typeid(T) == typeid(blas_real<double>)){

#pragma omp parallel shared(R) shared(m)
	{
	  matrix<T> Ri;
	  vertex<T> mi;

	  Ri.resize(N,N);
	  Ri.zero();

	  mi.resize(N);
	  mi.zero();

#pragma omp for nowait schedule(auto)
	  for(unsigned int index=0;index<data.size();index++){
	    auto& v = data[index];
	    
	    auto r = cublasDspr(cublas_handle, CUBLAS_FILL_MODE_LOWER, (int)N,
				(const double*)&s, (const double*)(v.data), 1,
				(double*)Ri.data);
	    
	    if(r != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("mean_covariance_estimate(): cublasDspr() failed.");
	      throw CUDAException("CUBLAS cublasDspr() call failed.");
	    }
	    
	    mi += v;
	  }

#pragma omp critical (mear_covariance_estimate2)
	  {
	    m += mi;
	    R += Ri;
	  }
	}

	
	// NOT OPTIMIZED
	{
	  // last element of the matrix
	  unsigned int non_packed_index = N*N;
	  unsigned int packed_index = N*(N+1)/2;
	  
	  for(unsigned int i=0;i<(N-1);i++){ // (N-i):th column
	    auto diag_non_packed_index = non_packed_index - (i+1);
	    packed_index -= (i+1);
	    
	    auto e = cudaMemcpy(&(R.data[diag_non_packed_index]), &(R.data[packed_index]),
				sizeof(T)*(i+1), cudaMemcpyDeviceToDevice);

	    if(e != cudaSuccess){
	      whiteice::logging.error("mean_covariance_estimate(): cudaMemcpy() failed.");
	      throw CUDAException("CUBLAS cudaMemcpy() call failed.");
	    }

	    non_packed_index -= N;
	  }
	}

	gpu_sync();

#pragma omp parallel for schedule(auto)
	// copies lower triangular elements to upper triangular part of the M matrix
	for(unsigned int i=0;i<N;i++)
	  for(unsigned int j=(i+1);j<N;j++)
	    R(i,j) = R(j,i);
	
	m *= s;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){

	double sf;
	whiteice::math::convert(sf, s);

#pragma omp parallel shared(R) shared(m)
	{
	  matrix<T> Ri;
	  vertex<T> mi;

	  Ri.resize(N,N);
	  Ri.zero();

	  mi.resize(N);
	  mi.zero();

#pragma omp for nowait schedule(auto)
	  for(unsigned int index=0;index<data.size();index++){
	    auto& v = data[index];
	    
	    auto r = cublasZhpr(cublas_handle, CUBLAS_FILL_MODE_LOWER, (int)N,
				(const double*)&sf,
				(const cuDoubleComplex*)(v.data), 1,
				(cuDoubleComplex*)Ri.data);
	    
	    if(r != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("mean_covariance_estimate(): cublasZhpr() failed.");
	      throw CUDAException("CUBLAS cublasZhpr() call failed.");
	    }
	    
	    mi += v;
	  }

#pragma omp critical (mean_covariance_estimate3)
	  {
	    m += mi;
	    R += Ri;
	  }
	}

	
	// NOT OPTIMIZED
	{
	  // last element of the matrix
	  unsigned int non_packed_index = N*N;
	  unsigned int packed_index = N*(N+1)/2;
	  
	  for(unsigned int i=0;i<(N-1);i++){ // (N-i):th column
	    auto diag_non_packed_index = non_packed_index - (i+1);
	    packed_index -= (i+1);
	    
	    auto e = cudaMemcpy(&(R.data[diag_non_packed_index]), &(R.data[packed_index]),
				sizeof(T)*(i+1), cudaMemcpyDeviceToDevice);

	    if(e != cudaSuccess){
	      whiteice::logging.error("mean_covariance_estimate(): cudaMemcpy() failed.");
	      throw CUDAException("CUBLAS cudaMemcpy() call failed.");
	    }

	    non_packed_index -= N;
	  }
	}
	
	gpu_sync();
	
	// copies lower triangular elements to upper triangular part of the M matrix
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<N;i++)
	  for(unsigned int j=(i+1);j<N;j++)
	    R(i,j) = R(j,i);
	
	m *= s;
      }
      else{ // generic code

#pragma omp parallel shared(R) shared(m)
	{
	  matrix<T> Ri;
	  vertex<T> mi;

	  Ri.resize(N,N);
	  Ri.zero();

	  mi.resize(N);
	  mi.zero();

#pragma omp for nowait schedule(auto)
	  for(unsigned int index=0;index<data.size();index++){
	    auto& v = data[index];
	    
	    Ri += v.outerproduct(v);
	    mi += v;
	  }

#pragma omp critical (mean_covariance_estimate4)
	  {
	    m += mi;
	    R += Ri;
	  }
	}
	
	R *= s;
	m *= s;
      }

#else
      
      if(typeid(T) == typeid(blas_real<float>)){	

#pragma omp parallel // shared(m) shared(R)
	{
	  matrix<T> Ri;
	  vertex<T> mi;

	  Ri.resize(N,N);
	  Ri.zero();

	  mi.resize(N);
	  mi.zero();

#pragma omp for nowait schedule(auto)
	  for(unsigned int index=0;index<data.size();index++){
	    auto& v = data[index];
	    
	    cblas_sspr(CblasRowMajor, CblasUpper, N, 
		       *((float*)&s), (float*)(v.data), 1, (float*)Ri.data);
	    
	    mi += v;
	  }

#pragma omp critical (mean_covariance_estimate5)
	  {
	    m += mi;
	    R += Ri;
	  }
	}
	  
	// converts packed symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(R.data[r*N + r]),
		  &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		  sizeof(T) * (i+1));
	}
	
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_scopy(N - i - 1, 
		      (float*)&(R.data[i*N + 1 + i]), 1,
		      (float*)&(R.data[(i+1)*N + i]), N);
	
	m *= s;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){

#pragma omp parallel shared(m) shared(R)
	{
	  matrix<T> Ri;
	  vertex<T> mi;
	  
	  Ri.resize(N,N);
	  Ri.zero();
	  
	  mi.resize(N);
	  mi.zero();

#pragma omp for nowait schedule(auto)
	  for(unsigned int index=0;index<data.size();index++){
	    auto& v = data[index];	  
	    
	    cblas_chpr(CblasRowMajor, CblasUpper, N,
		       *((float*)&s), (float*)(v.data), 1, (float*)Ri.data);
	    mi += v;
	  }

#pragma omp critical (mean_covariance_estimate6)
	  {
	    m += mi;
	    R += Ri;
	  }
	}
	  
	// converts packed symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(R.data[r*N + r]),
		  &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		  sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_ccopy(N - i - 1, 
		      (float*)&(R.data[i*N + 1 + i]), 1,
		      (float*)&(R.data[(i+1)*N + i]), N);
	  
	m *= s;
      }
      else if(typeid(T) == typeid(blas_real<double>)){

#pragma omp parallel shared(R) shared(m)
	{
	  matrix<T> Ri;
	  vertex<T> mi;

	  Ri.resize(N,N);
	  Ri.zero();

	  mi.resize(N);
	  mi.zero();

#pragma omp for nowait schedule(auto)
	  for(unsigned int index=0;index<data.size();index++){
	    auto& v = data[index];
	    
	    cblas_dspr(CblasRowMajor, CblasUpper, N, 
		       *((double*)&s), (double*)(v.data), 1, (double*)(Ri.data));
	    mi += v;
	  }

#pragma omp critical (mean_covariance_estimate6)
	  {
	    m += mi;
	    R += Ri;
	  }
	}
	
	// converts symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(R.data[r*N + r]),
		  &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		  sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_dcopy(N - i - 1, 
		      (double*)&(R.data[i*N + 1 + i]), 1,
		      (double*)&(R.data[(i+1)*N + i]), N);
	
	m *= s;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){

#pragma omp parallel shared(R) shared(m)
	{
	  matrix<T> Ri;
	  vertex<T> mi;

	  Ri.resize(N,N);
	  Ri.zero();

	  mi.resize(N);
	  mi.zero();

#pragma omp for nowait schedule(auto)
	  for(unsigned int index=0;index<data.size();index++){
	    auto& v = data[index];
	    
	    cblas_zhpr(CblasRowMajor, CblasUpper, N,
		       *((double*)&s), (double*)(v.data), 1, (double*)Ri.data);
	    mi += v;
	  }

#pragma omp critical (mean_covariance_estimate7)
	  {
	    m += mi;
	    R += Ri;
	  }
	}
	
	// converts symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(R.data[r*N + r]),
		  &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		  sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_zcopy(N - i - 1, 
		      (double*)&(R.data[i*N + 1 + i]), 1,
		      (double*)&(R.data[(i+1)*N + i]), N);
	
	m *= s;
      }
      else{ // generic code

#pragma omp parallel shared(R) shared(m)
	{
	  matrix<T> Ri;
	  vertex<T> mi;

	  Ri.resize(N,N);
	  Ri.zero();

	  mi.resize(N);
	  mi.zero();

#pragma omp for nowait schedule(auto)
	  for(unsigned int index=0;index<data.size();index++){
	    auto& v = data[index];
	    
	    Ri += v.outerproduct(v);
	    mi += v;
	  }

#pragma omp critical (mean_covariance_estimate8)
	  {
	    m += mi;
	    R += Ri;
	  }
	}
	
	R *= s;
	m *= s;
      }

#endif
      
      // removes mean from R = E[xx^t]
      
#pragma omp parallel for schedule(auto)
      for(unsigned int j=0;j<N;j++){
	for(unsigned int i=0;i<=j;i++){
	  T tmp = m[j]*m[i];
	  R(j,i) -= tmp;
	  if(j!=i)
	    R(i,j) -= tmp;
	}
      }
      
      return true;
    }
    
    
    
    
    template <typename T>
    bool mean_covariance_estimate(vertex<T>& m, matrix<T>& R, 
				  const std::vector< vertex<T> >& data,
				  const std::vector< whiteice::dynamic_bitset >& missing)
    {
      if(data.size() <= 0)
	return false;
      
      if(data.size() != missing.size())
	return false;
      
      const unsigned int D = data[0].size();
      
      // because some entries are missing calculating
      // E[x] and E[(x - m)(x - m)^t] must happen per dimension or correlation pair
      // basis and because some values are missing Ni or Nij
      // (samples used in estimate SUM(x_i), SUM(x_i*x_j)
      // 
      // however, if/when non-available entries are zeros they contribute nothing
      // to unnormalized sum estimate calculated normally. So normal methods
      // and 
      
      if(!autocorrelation(R, data))
	return false;

      // Ni = number of entries available
      matrix<T> RN(D,D);
      vertex<T> N(D);
      
      m.resize(D);
      m.zero();
      
      for(unsigned int i=0;i<missing.size();i++){
	for(unsigned int j=0;j<D;j++){
	  if(!missing[i][j])
	    N[j] += T(1.0);
	  
	  for(unsigned int k=0;k<D;k++)
	    if((!missing[i][j]) && (!missing[i][k]))
	      RN(j,k) += T(1.0);
	}
	
	m += data[i];
      }

      
      
      
      for(unsigned int i=0;i<D;i++)
	m[i] /= N[i];
      
      // rescales autocorrelation matrix
      for(unsigned int i=0;i<D;i++)
	for(unsigned int j=0;j<D;j++)
	  R(i,j) *= T(data.size())/RN(i,j);
      
      
      return true;
    }


    // calculates crosscorrelation matrix Cyx = E[y*x^h] as well as mean values E[x], E[y]
    template <typename T>
    bool mean_crosscorrelation_estimate(vertex<T>& mx, vertex<T>& my, matrix<T>& Cyx,
					const std::vector< vertex<T> >& xdata,
					const std::vector< vertex<T> >& ydata)
    {
      // not optimized

      if(xdata.size() != ydata.size()) return false;
      if(xdata.size() == 0) return false;

      // calculates Ryx and mean and then Cyx = Ryx - my*mx^t

      mx.resize(xdata[0].size());
      my.resize(ydata[0].size());
      Cyx.resize(ydata[0].size(), xdata[0].size());
      mx.zero();
      my.zero();
      Cyx.zero();

      for(unsigned int i=0;i<xdata.size();i++){
	mx += xdata[i];
	my += ydata[i];

	auto hx = xdata[i];
	hx.hermite();
	for(unsigned int y=0;y<ydata[0].size();y++)
	  for(unsigned int x=0;x<xdata[0].size();x++)
	    Cyx(y,x) += ydata[i][y]*hx[x];
      }

      mx /= T(xdata.size());
      my /= T(xdata.size());
      Cyx /= T(xdata.size());

      auto hmx = mx;
      hmx.hermite();

      for(unsigned int y=0;y<ydata[0].size();y++)
	for(unsigned int x=0;x<xdata[0].size();x++)
	  Cyx(y,x) -= my[y]*hmx[x];

      return true;
    }
    
    
    // calculates PCA dimension reduction using symmetric eigenvalue decomposition
    template <typename T>
    bool pca(const std::vector< vertex<T> >& data, 
	     const unsigned int dimensions,
	     math::matrix<T>& PCA,
	     math::vertex<T>& mxx,
	     T& original_var, T& reduced_var,
	     bool regularizeIfNeeded,
	     bool unitVariance)
    {
      matrix<T> Cxx;

      if(mean_covariance_estimate(mxx, Cxx, data) == false)
	return false;

      
      matrix<T> X;

      
      if(regularizeIfNeeded){
	// not optimized
	
	matrix<T> CC(Cxx);
	T regularizer = T(0.0f);
	
	for(unsigned int j=0;j<CC.ysize();j++)
	  for(unsigned int i=0;i<CC.xsize();i++)
	    regularizer += abs(CC(j,i));
	
	regularizer /= (CC.ysize()*CC.xsize());
	regularizer *= T(0.001);
	if(regularizer <= T(0.0f)) regularizer = T(0.0001f);
	
	unsigned int counter = 0;
	const unsigned int CLIMIT = 20;
	
	
	while(symmetric_eig(CC, X, true) == false){
	  CC = Cxx;
	  
	  for(unsigned int i=0;i<CC.ysize();i++)
	    CC(i,i) += regularizer;
	  
	  regularizer *= T(2.0f);
	  counter++;
	  
	  if(counter >= CLIMIT)
	    return false;
	}

	Cxx = CC;
      }
      else{
	if(symmetric_eig(Cxx, X, true) == false)
	  return false;
      }

      // Cxx = X * D * X^t

      // we want to keep only the top variance vectors

      original_var = T(0.0f);
      reduced_var  = T(0.0f);

      for(unsigned int i=0;i<X.xsize();i++){
	if(i<dimensions) reduced_var += Cxx(i,i);
	original_var += Cxx(i,i);
      }

      if(X.submatrix(PCA, 0, 0, dimensions, X.xsize()) == false)
	return false;

      PCA.transpose();


      if(unitVariance){
	
	T epsilon = T(1e-9f);
	
	for(unsigned int j=0;j<dimensions;j++){
	  
	  T scaling =
	    T(1.0f)/(epsilon + whiteice::math::sqrt(whiteice::math::abs(Cxx(j,j))));
	  
	  for(unsigned int i=0;i<PCA.xsize();i++){
	    PCA(j,i) = scaling*PCA(j,i);
	  }
	}
      }
      
      return true;
    }



    // calculates PCA dimension reduction using symmetric eigenvalue decomposition
    // we keep p% of total variance
    template <typename T>
    bool pca_p(const std::vector< vertex<T> >& data, 
	       const float percent_total_variance,
	       math::matrix<T>& PCA,
	       math::vertex<T>& mxx,
	       T& original_var, T& reduced_var,
	       bool regularizeIfNeeded,
	       bool unitVariance)
    {
      if(percent_total_variance <= 0.0f || percent_total_variance > 1.0f)
	return false;
      
      matrix<T> Cxx;

      if(mean_covariance_estimate(mxx, Cxx, data) == false)
	return false;

      
      matrix<T> X;

      
      if(regularizeIfNeeded){
	// not optimized
	
	matrix<T> CC(Cxx);
	T regularizer = T(0.0f);
	
	for(unsigned int j=0;j<CC.ysize();j++)
	  for(unsigned int i=0;i<CC.xsize();i++)
	    regularizer += abs(CC(j,i));
	
	regularizer /= (CC.ysize()*CC.xsize());
	regularizer *= T(0.001);
	if(regularizer <= T(0.0f)) regularizer = T(0.0001f);
	
	unsigned int counter = 0;
	const unsigned int CLIMIT = 20;
	
	
	while(symmetric_eig(CC, X, true) == false){
	  CC = Cxx;
	  
	  for(unsigned int i=0;i<CC.ysize();i++)
	    CC(i,i) += regularizer;
	  
	  regularizer *= T(2.0f);
	  counter++;
	  
	  if(counter >= CLIMIT)
	    return false;
	}

	Cxx = CC;
      }
      else{
	if(symmetric_eig(Cxx, X, true) == false)
	  return false;
      }

      // Cxx = X * D * X^t

      // we want to keep only the top p% variance of eigenvectors

      original_var = T(0.0f);
      
      for(unsigned int i=0;i<X.xsize();i++){	
	original_var += Cxx(i,i);
      }

      T target_var = T(percent_total_variance)*original_var;

      T current_var = T(0.0f);
      unsigned int dimensions = 1;

      for(unsigned int i=0;i<X.xsize();i++){
	current_var += Cxx(i,i);
	dimensions = (i+1);
	if(current_var >= target_var)
	  break;
      }

      reduced_var = current_var;


      if(X.submatrix(PCA, 0, 0, dimensions, X.xsize()) == false)
	return false;

      PCA.transpose();


      if(unitVariance){
	
	T epsilon = T(1e-9f);
	
	for(unsigned int j=0;j<dimensions;j++){
	  
	  T scaling =
	    T(1.0f)/(epsilon + whiteice::math::sqrt(whiteice::math::abs(Cxx(j,j))));
	  
	  for(unsigned int i=0;i<PCA.xsize();i++){
	    PCA(j,i) = scaling*PCA(j,i);
	  }
	}
      }
      
      return true;
    }
    
    
    
    // explicit template instantations
    
    template bool autocorrelation<float>(matrix<float>& R, const std::vector< vertex<float> >& data);
    template bool autocorrelation<double>(matrix<double>& R, const std::vector< vertex<double> >& data);
    
    template bool autocorrelation<blas_real<float> >(matrix<blas_real<float> >& R,
						      const std::vector< vertex<blas_real<float> > >& data);
    template bool autocorrelation<blas_real<double> >(matrix<blas_real<double> >& R,
						       const std::vector< vertex<blas_real<double> > >& data);
    template bool autocorrelation<blas_complex<float> >(matrix<blas_complex<float> >& R,
							 const std::vector< vertex<blas_complex<float> > >& data);
    template bool autocorrelation<blas_complex<double> >(matrix<blas_complex<double> >& R,
							  const std::vector< vertex<blas_complex<double> > >& data);

    template bool autocorrelation
    <superresolution<blas_real<float>, modular<unsigned int> > >
    (matrix<superresolution<blas_real<float>, modular<unsigned int> > >& R,
     const std::vector< vertex<superresolution<blas_real<float>, modular<unsigned int> > > >& data);
    
    template bool autocorrelation
    <superresolution<blas_real<double>, modular<unsigned int> > >
    (matrix<superresolution<blas_real<double>, modular<unsigned int> > >& R,
     const std::vector< vertex<superresolution<blas_real<double>, modular<unsigned int> > > >& data);

    template bool autocorrelation
    <superresolution<blas_complex<float>, modular<unsigned int> > >
    (matrix<superresolution<blas_complex<float>, modular<unsigned int> > >& R,
     const std::vector< vertex<superresolution<blas_complex<float>, modular<unsigned int> > > >& data);
    
    template bool autocorrelation
    <superresolution<blas_complex<double>, modular<unsigned int> > >
    (matrix<superresolution<blas_complex<double>, modular<unsigned int> > >& R,
     const std::vector< vertex<superresolution<blas_complex<double>, modular<unsigned int> > > >& data);

    
    
    template bool autocorrelation<float>(matrix<float>& R, const matrix<float>& W);
    template bool autocorrelation<double>(matrix<double>& R, const matrix<double>& W);
    
    template bool autocorrelation<blas_real<float> >(matrix<blas_real<float> >& R,
						      const matrix<blas_real<float> >& W);
    template bool autocorrelation<blas_real<double> >(matrix<blas_real<double> >& R,
						       const matrix<blas_real<double> >& W);
    template bool autocorrelation<blas_complex<float> >(matrix<blas_complex<float> >& R,
							 const matrix<blas_complex<float> >& W);
    template bool autocorrelation<blas_complex<double> >(matrix<blas_complex<double> >& R,
							  const matrix<blas_complex<double> >& W);

    template bool autocorrelation
    <superresolution<blas_real<float>, modular<unsigned int> > >
    (matrix<superresolution<blas_real<float>, modular<unsigned int> > >& R,
     const matrix<superresolution<blas_real<float>, modular<unsigned int> > >& W);
    template bool autocorrelation
    <superresolution<blas_real<double>, modular<unsigned int> > >
    (matrix<superresolution<blas_real<double>, modular<unsigned int> > >& R,
     const matrix<superresolution<blas_real<double>, modular<unsigned int> > >& W);
    
    template bool autocorrelation
    <superresolution<blas_complex<float>, modular<unsigned int> > >
    (matrix<superresolution<blas_complex<float>, modular<unsigned int> > >& R,
     const matrix<superresolution<blas_complex<float>, modular<unsigned int> > >& W);
    template bool autocorrelation
    <superresolution<blas_complex<double>, modular<unsigned int> > >
    (matrix<superresolution<blas_complex<double>, modular<unsigned int> > >& R,
     const matrix<superresolution<blas_complex<double>, modular<unsigned int> > >& W);
    

    
    template bool mean_covariance_estimate< float >
    (vertex< float >& m, matrix< float >& R,
     const std::vector< vertex< float > >& data);

    template bool mean_covariance_estimate< double >
    (vertex< double >& m, matrix< double >& R,
     const std::vector< vertex< double > >& data);
    
    template bool mean_covariance_estimate< blas_real<float> >
    (vertex< blas_real<float> >& m, matrix< blas_real<float> >& R,
     const std::vector< vertex< blas_real<float> > >& data);
    
    template bool mean_covariance_estimate< blas_real<double> >
    (vertex< blas_real<double> >& m, matrix< blas_real<double> >& R,
     const std::vector< vertex< blas_real<double> > >& data);
    
    template bool mean_covariance_estimate< blas_complex<float> >
    (vertex< blas_complex<float> >& m, matrix< blas_complex<float> >& R,
     const std::vector< vertex< blas_complex<float> > >& data);
    
    template bool mean_covariance_estimate< blas_complex<double> > 
    (vertex< blas_complex<double> >& m, matrix< blas_complex<double> >& R,
     const std::vector< vertex< blas_complex<double> > >& data);

    template bool mean_covariance_estimate
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (vertex< superresolution<blas_real<float>, modular<unsigned int> > >& m,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& R,
     const std::vector< vertex< superresolution<blas_real<float>, modular<unsigned int> > > >& data);
    
    template bool mean_covariance_estimate
    < superresolution<blas_real<double>, modular<unsigned int> > > 
    (vertex< superresolution<blas_real<double>, modular<unsigned int> > >& m,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& R,
     const std::vector< vertex< superresolution<blas_real<double>, modular<unsigned int> > > >& data);
    
    template bool mean_covariance_estimate
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& m,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& R,
     const std::vector< vertex< superresolution<blas_complex<float>, modular<unsigned int> > > >& data);
    
    template bool mean_covariance_estimate
    < superresolution<blas_complex<double>, modular<unsigned int> > > 
    (vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& m,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& R,
     const std::vector< vertex< superresolution<blas_complex<double>, modular<unsigned int> > > >& data);
    
    
    template bool mean_covariance_estimate< float >
    (vertex< float >& m, matrix< float >& R,
     const std::vector< vertex< float > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);
    
    template bool mean_covariance_estimate< double >
    (vertex< double >& m, matrix< double >& R,
     const std::vector< vertex< double > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);
    
    template bool mean_covariance_estimate< blas_real<float> >
    (vertex< blas_real<float> >& m, matrix< blas_real<float> >& R,
     const std::vector< vertex< blas_real<float> > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);

    template bool mean_covariance_estimate< blas_real<double> >
    (vertex< blas_real<double> >& m, matrix< blas_real<double> >& R,
     const std::vector< vertex< blas_real<double> > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);
    
    template bool mean_covariance_estimate< blas_complex<float> >
    (vertex< blas_complex<float> >& m, matrix< blas_complex<float> >& R,
     const std::vector< vertex< blas_complex<float> > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);
    
    template bool mean_covariance_estimate< blas_complex<double> >
    (vertex< blas_complex<double> >& m, matrix< blas_complex<double> >& R,
     const std::vector< vertex< blas_complex<double> > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);

    
    template bool mean_covariance_estimate
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (vertex< superresolution<blas_real<float>, modular<unsigned int> > >& m,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& R,
     const std::vector< vertex< superresolution<blas_real<float>, modular<unsigned int> > > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);
    
    template bool mean_covariance_estimate
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (vertex< superresolution<blas_real<double>, modular<unsigned int> > >& m,
     matrix< superresolution<blas_real<double>, modular<unsigned int> >  >& R,
     const std::vector< vertex< superresolution<blas_real<double>, modular<unsigned int> > > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);

    
    template bool mean_covariance_estimate
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& m,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& R,
     const std::vector< vertex< superresolution<blas_complex<float>, modular<unsigned int> > > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);
    
    template bool mean_covariance_estimate
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& m,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> >  >& R,
     const std::vector< vertex< superresolution<blas_complex<double>, modular<unsigned int> > > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);
    

    
    template bool mean_crosscorrelation_estimate< float >
    (vertex< float >& mx, vertex< float >& my, matrix< float >& Cyx,
     const std::vector< vertex<float> >& xdata,
     const std::vector< vertex<float> >& ydata);

    template bool mean_crosscorrelation_estimate< double >
    (vertex< double >& mx, vertex< double >& my, matrix< double >& Cyx,
     const std::vector< vertex<double> >& xdata,
     const std::vector< vertex<double> >& ydata);

    
    template bool mean_crosscorrelation_estimate< blas_real<float> >
    (vertex< blas_real<float> >& mx, vertex< blas_real<float> >& my, matrix< blas_real<float> >& Cyx,
     const std::vector< vertex< blas_real<float> > >& xdata,
     const std::vector< vertex< blas_real<float> > >& ydata);

    template bool mean_crosscorrelation_estimate< blas_real<double> >
    (vertex< blas_real<double> >& mx, vertex< blas_real<double> >& my, matrix< blas_real<double> >& Cyx,
     const std::vector< vertex< blas_real<double> > >& xdata,
     const std::vector< vertex< blas_real<double> > >& ydata);

    template bool mean_crosscorrelation_estimate< blas_complex<float> >
    (vertex< blas_complex<float> >& mx, vertex< blas_complex<float> >& my, matrix< blas_complex<float> >& Cyx,
     const std::vector< vertex< blas_complex<float> > >& xdata,
     const std::vector< vertex< blas_complex<float> > >& ydata);

    template bool mean_crosscorrelation_estimate< blas_complex<double> >
    (vertex< blas_complex<double> >& mx, vertex< blas_complex<double> >& my, matrix< blas_complex<double> >& Cyx,
     const std::vector< vertex< blas_complex<double> > >& xdata,
     const std::vector< vertex< blas_complex<double> > >& ydata);

    template bool mean_crosscorrelation_estimate
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (vertex< superresolution<blas_real<float>, modular<unsigned int> > >& mx,
     vertex< superresolution<blas_real<float>, modular<unsigned int> > >& my,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& Cyx,
     const std::vector< vertex< superresolution<blas_real<float>, modular<unsigned int> > > >& xdata,
     const std::vector< vertex< superresolution<blas_real<float>, modular<unsigned int> > > >& ydata);
    
    template bool mean_crosscorrelation_estimate
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (vertex< superresolution<blas_real<double>, modular<unsigned int> > >& mx,
     vertex< superresolution<blas_real<double>, modular<unsigned int> > >& my,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& Cyx,
     const std::vector< vertex< superresolution<blas_real<double>, modular<unsigned int> > > >& xdata,
     const std::vector< vertex< superresolution<blas_real<double>, modular<unsigned int> > > >& ydata);
    
    template bool mean_crosscorrelation_estimate
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& mx,
     vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& my,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& Cyx,
     const std::vector< vertex< superresolution<blas_complex<float>, modular<unsigned int> > > >& xdata,
     const std::vector< vertex< superresolution<blas_complex<float>, modular<unsigned int> > > >& ydata);
    
    template bool mean_crosscorrelation_estimate
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& mx,
     vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& my,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& Cyx,
     const std::vector< vertex< superresolution<blas_complex<double>, modular<unsigned int> > > >& xdata,
     const std::vector< vertex< superresolution<blas_complex<double>, modular<unsigned int> > > >& ydata);

    
    
    template bool pca<float>
      (const std::vector< vertex<float> >& data, 
       const unsigned int dimensions,
       math::matrix<float>& PCA,
       math::vertex<float>& m,
       float& original_var, float& reduced_var,
       bool regularizeIfNeeded,
       bool unitVariance);

    template bool pca<double>
      (const std::vector< vertex<double> >& data, 
       const unsigned int dimensions,
       math::matrix<double>& PCA,
       math::vertex<double>& m,
       double& original_var, double& reduced_var,
       bool regularizeIfNeeded,
       bool unitVariance);

    template bool pca< blas_real<float> >
      (const std::vector< vertex< blas_real<float> > >& data, 
       const unsigned int dimensions,
       math::matrix< blas_real<float> >& PCA,
       math::vertex< blas_real<float> >& m,
       blas_real<float>& original_var, blas_real<float>& reduced_var,
       bool regularizeIfNeeded,
       bool unitVariance);

    template bool pca< blas_real<double> >
      (const std::vector< vertex< blas_real<double> > >& data, 
       const unsigned int dimensions,
       math::matrix< blas_real<double> >& PCA,
       math::vertex< blas_real<double> >& m,
       blas_real<double>& original_var, blas_real<double>& reduced_var,
       bool regularizeIfNeeded,
       bool unitVariance);

    template bool pca< blas_complex<float> >
      (const std::vector< vertex< blas_complex<float> > >& data, 
       const unsigned int dimensions,
       math::matrix< blas_complex<float> >& PCA,
       math::vertex< blas_complex<float> >& m,
       blas_complex<float>& original_var, blas_complex<float>& reduced_var,
       bool regularizeIfNeeded,
       bool unitVariance);

    template bool pca< blas_complex<double> >
      (const std::vector< vertex< blas_complex<double> > >& data, 
       const unsigned int dimensions,
       math::matrix< blas_complex<double> >& PCA,
       math::vertex< blas_complex<double> >& m,
       blas_complex<double>& original_var, blas_complex<double>& reduced_var,
       bool regularizeIfNeeded,
       bool unitVariance);

    template bool pca
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution<blas_real<float>, modular<unsigned int> > > >& data, 
     const unsigned int dimensions,
     math::matrix< superresolution<blas_real<float>, modular<unsigned int> > >& PCA,
     math::vertex< superresolution<blas_real<float>, modular<unsigned int> > >& m,
     superresolution<blas_real<float>, modular<unsigned int> >& original_var,
     superresolution<blas_real<float>, modular<unsigned int> >& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);
    
    template bool pca
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution<blas_real<double>, modular<unsigned int> > > >& data, 
     const unsigned int dimensions,
     math::matrix< superresolution<blas_real<double>, modular<unsigned int> > >& PCA,
     math::vertex< superresolution<blas_real<double>, modular<unsigned int> > >& m,
     superresolution<blas_real<double>, modular<unsigned int> >& original_var,
     superresolution<blas_real<double>, modular<unsigned int> >& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);
    
    template bool pca
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution<blas_complex<float>, modular<unsigned int> > > >& data, 
     const unsigned int dimensions,
     math::matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& PCA,
     math::vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& m,
     superresolution<blas_complex<float>, modular<unsigned int> >& original_var,
     superresolution<blas_complex<float>, modular<unsigned int> >& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);
    
    template bool pca
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution<blas_complex<double>, modular<unsigned int> > > >& data, 
     const unsigned int dimensions,
     math::matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& PCA,
     math::vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& m,
     superresolution<blas_complex<double>, modular<unsigned int> >& original_var,
     superresolution<blas_complex<double>, modular<unsigned int> >& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);
    
    
    
    
    template bool pca_p <float>
    (const std::vector< vertex<float> >& data, 
     const float percent_total_variance,
     math::matrix<float>& PCA,
     math::vertex<float>& m,
     float& original_var, float& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);

    template bool pca_p <double>
    (const std::vector< vertex<double> >& data, 
     const float percent_total_variance,
     math::matrix<double>& PCA,
     math::vertex<double>& m,
     double& original_var, double& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);

    template bool pca_p < blas_real<float> >
    (const std::vector< vertex< blas_real<float> > >& data, 
     const float percent_total_variance,
     math::matrix< blas_real<float> >& PCA,
     math::vertex< blas_real<float> >& m,
     blas_real<float>& original_var, blas_real<float>& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);

    template bool pca_p < blas_real<double> >
    (const std::vector< vertex< blas_real<double> > >& data, 
     const float percent_total_variance,
     math::matrix< blas_real<double> >& PCA,
     math::vertex< blas_real<double> >& m,
     blas_real<double>& original_var, blas_real<double>& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);

    template bool pca_p < blas_complex<float> >
    (const std::vector< vertex< blas_complex<float> > >& data, 
     const float percent_total_variance,
     math::matrix< blas_complex<float> >& PCA,
     math::vertex< blas_complex<float> >& m,
     blas_complex<float>& original_var, blas_complex<float>& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);

    template bool pca_p < blas_complex<double> >
    (const std::vector< vertex< blas_complex<double> > >& data, 
     const float percent_total_variance,
     math::matrix< blas_complex<double> >& PCA,
     math::vertex< blas_complex<double> >& m,
     blas_complex<double>& original_var, blas_complex<double>& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);

    template bool pca_p
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution<blas_real<float>, modular<unsigned int> > > >& data, 
     const float percent_total_variance,
     math::matrix< superresolution<blas_real<float>, modular<unsigned int> > >& PCA,
     math::vertex< superresolution<blas_real<float>, modular<unsigned int> > >& m,
     superresolution<blas_real<float>, modular<unsigned int> >& original_var,
     superresolution<blas_real<float>, modular<unsigned int> >& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);

    template bool pca_p
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution<blas_real<double>, modular<unsigned int> > > >& data, 
     const float percent_total_variance,
     math::matrix< superresolution<blas_real<double>, modular<unsigned int> > >& PCA,
     math::vertex< superresolution<blas_real<double>, modular<unsigned int> > >& m,
     superresolution<blas_real<double>, modular<unsigned int> >& original_var,
     superresolution<blas_real<double>, modular<unsigned int> >& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);

    template bool pca_p
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution<blas_complex<float>, modular<unsigned int> > > >& data, 
     const float percent_total_variance,
     math::matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& PCA,
     math::vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& m,
     superresolution<blas_complex<float>, modular<unsigned int> >& original_var,
     superresolution<blas_complex<float>, modular<unsigned int> >& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);

    template bool pca_p
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution<blas_complex<double>, modular<unsigned int> > > >& data, 
     const float percent_total_variance,
     math::matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& PCA,
     math::vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& m,
     superresolution<blas_complex<double>, modular<unsigned int> >& original_var,
     superresolution<blas_complex<double>, modular<unsigned int> >& reduced_var,
     bool regularizeIfNeeded,
     bool unitVariance);
    
    
  };
};

#endif
