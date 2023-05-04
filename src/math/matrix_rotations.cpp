
#include <typeinfo>
#include <exception>

#include "matrix_rotations.h"
#include "dinrhiw_blas.h"
#include "vertex.h"
#include "matrix.h"
#include "Log.h"


namespace whiteice
{
  namespace math
  {
    
    /* explicit template instantations */
    
    template bool rhouseholder_vector< blas_real<float> >
      (vertex< blas_real<float> >& v,
       const matrix< blas_real<float> >& M,
       unsigned int y, unsigned int x,
       bool rowdir);
    template bool rhouseholder_vector< blas_real<double> >
      (vertex< blas_real<double> >& v,
       const matrix< blas_real<double> >& M,
       unsigned int y, unsigned int x,
       bool rowdir);
    template bool rhouseholder_vector< float >
      (vertex< float >& v,
       const matrix< float >& M,
       unsigned int y, unsigned int x,
       bool rowdir);
    template bool rhouseholder_vector< double >
      (vertex< double >& v,
       const matrix< double >& M,
       unsigned int y, unsigned int x,
       bool rowdir);
    template bool rhouseholder_vector< blas_complex<float> >
      (vertex< blas_complex<float> >& v,
       const matrix< blas_complex<float> >& M,
       unsigned int y, unsigned int x,
       bool rowdir);
    template bool rhouseholder_vector< blas_complex<double> >
      (vertex< blas_complex<double> >& v,
       const matrix< blas_complex<double> >& M,
       unsigned int y, unsigned int x,
       bool rowdir);

    template bool rhouseholder_vector
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (vertex< superresolution<blas_real<float>, modular<unsigned int> > >& v,
     const matrix< superresolution<blas_real<float>, modular<unsigned int> > >& M,
     unsigned int y, unsigned int x,
     bool rowdir);
    
    template bool rhouseholder_vector
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (vertex< superresolution<blas_real<double>, modular<unsigned int> > >& v,
     const matrix< superresolution<blas_real<double>, modular<unsigned int> > >& M,
     unsigned int y, unsigned int x,
     bool rowdir);

    template bool rhouseholder_vector
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& v,
     const matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& M,
     unsigned int y, unsigned int x,
     bool rowdir);
    
    template bool rhouseholder_vector
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& v,
     const matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& M,
     unsigned int y, unsigned int x,
     bool rowdir);

    
    
    template bool rhouseholder_leftrot< blas_real<float> > 
      (matrix< blas_real<float> >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< blas_real<float> >& v);
    template bool rhouseholder_leftrot< blas_real<double> >
      (matrix< blas_real<double> >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< blas_real<double> >& v);
    template bool rhouseholder_leftrot< float > 
      (matrix< float >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< float >& v);
    template bool rhouseholder_leftrot< double >
      (matrix< double >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< double >& v);
    template bool rhouseholder_leftrot< blas_complex<float> > 
      (matrix< blas_complex<float> >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< blas_complex<float> >& v);
    template bool rhouseholder_leftrot< blas_complex<double> >
      (matrix< blas_complex<double> >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< blas_complex<double> >& v);

    template bool rhouseholder_leftrot
    < superresolution<blas_real<float>, modular<unsigned int> > > 
    (matrix< superresolution<blas_real<float>, modular<unsigned int> > >& A,
     const unsigned int i,
     const unsigned int M,
     const unsigned int k,
     vertex< superresolution<blas_real<float>, modular<unsigned int> > >& v);
    
    template bool rhouseholder_leftrot
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<double>, modular<unsigned int> > >& A,
     const unsigned int i,
     const unsigned int M,
     const unsigned int k,
     vertex< superresolution<blas_real<double>, modular<unsigned int> > >& v);

    template bool rhouseholder_leftrot
    < superresolution<blas_complex<float>, modular<unsigned int> > > 
    (matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& A,
     const unsigned int i,
     const unsigned int M,
     const unsigned int k,
     vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& v);
    
    template bool rhouseholder_leftrot
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& A,
     const unsigned int i,
     const unsigned int M,
     const unsigned int k,
     vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& v);
    
    
    
    template bool rhouseholder_rightrot< blas_real<float> >
      (matrix< blas_real<float> >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< blas_real<float> >& v);
    template bool rhouseholder_rightrot< blas_real<double> >
      (matrix< blas_real<double> >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< blas_real<double> >& v);
    template bool rhouseholder_rightrot< float >
      (matrix< float >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< float >& v);
    template bool rhouseholder_rightrot< double >
      (matrix< double >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< double >& v);
    template bool rhouseholder_rightrot< blas_complex<float> >
      (matrix< blas_complex<float> >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< blas_complex<float> >& v);
    template bool rhouseholder_rightrot< blas_complex<double> >
      (matrix< blas_complex<double> >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< blas_complex<double> >& v);

    template bool rhouseholder_rightrot
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<float>, modular<unsigned int> > >& A,
     const unsigned int i,
     const unsigned int M,
     const unsigned int k,
     vertex< superresolution<blas_real<float>, modular<unsigned int> > >& v);
    template bool rhouseholder_rightrot
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<double>, modular<unsigned int> > >& A,
     const unsigned int i,
     const unsigned int M,
     const unsigned int k,
     vertex< superresolution<blas_real<double>, modular<unsigned int> > >& v);


    template bool rhouseholder_rightrot
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& A,
     const unsigned int i,
     const unsigned int M,
     const unsigned int k,
     vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& v);
    template bool rhouseholder_rightrot
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& A,
     const unsigned int i,
     const unsigned int M,
     const unsigned int k,
     vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& v);
    
    
    template void rgivens< blas_real<float> >
      (const blas_real<float>& a, const blas_real<float>& b, vertex< blas_real<float> >& p);
    template void rgivens< blas_real<double> >
      (const blas_real<double>& a, const blas_real<double>& b, vertex< blas_real<double> >& p);
    template void rgivens< float >
      (const float& a, const float& b, vertex< float >& p);
    template void rgivens< double >
      (const double& a, const double& b, vertex< double >& p);

    template void rgivens< blas_complex<float> >
      (const blas_complex<float>& a, const blas_complex<float>& b, vertex< blas_complex<float> >& p);
    template void rgivens< blas_complex<double> >
      (const blas_complex<double>& a, const blas_complex<double>& b, vertex< blas_complex<double> >& p);
    template void rgivens
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (const superresolution<blas_real<float>, modular<unsigned int> >& a,
     const superresolution<blas_real<float>, modular<unsigned int> >& b,
     vertex< superresolution<blas_real<float>, modular<unsigned int> > >& p);
    
    template void rgivens
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (const superresolution<blas_real<double>, modular<unsigned int> >& a,
     const superresolution<blas_real<double>, modular<unsigned int> >& b,
     vertex< superresolution<blas_real<double>, modular<unsigned int> > >& p);

    template void rgivens
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (const superresolution<blas_complex<float>, modular<unsigned int> >& a,
     const superresolution<blas_complex<float>, modular<unsigned int> >& b,
     vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& p);
    
    template void rgivens
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (const superresolution<blas_complex<double>, modular<unsigned int> >& a,
     const superresolution<blas_complex<double>, modular<unsigned int> >& b,
     vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& p);
    
    
    template void rgivens_rightrot< blas_real<float> >
      (matrix< blas_real<float> >& A, const vertex< blas_real<float> >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    template void rgivens_rightrot< blas_real<double> >
      (matrix< blas_real<double> >& A, const vertex< blas_real<double> >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    template void rgivens_rightrot< float >
      (matrix< float >& A, const vertex< float >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    template void rgivens_rightrot< double >
      (matrix< double >& A, const vertex< double >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    template void rgivens_rightrot< blas_complex<float> >
      (matrix< blas_complex<float> >& A, const vertex< blas_complex<float> >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    template void rgivens_rightrot< blas_complex<double> >
      (matrix< blas_complex<double> >& A, const vertex< blas_complex<double> >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);

    template void rgivens_rightrot
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<float>, modular<unsigned int> > >& A,
     const vertex< superresolution<blas_real<float>, modular<unsigned int> > >& p,
     const unsigned int i, const unsigned int j, const unsigned int k);
    
    template void rgivens_rightrot
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<double>, modular<unsigned int> > >& A,
     const vertex< superresolution<blas_real<double>, modular<unsigned int> > >& p,
     const unsigned int i, const unsigned int j, const unsigned int k);

    template void rgivens_rightrot
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& A,
     const vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& p,
     const unsigned int i, const unsigned int j, const unsigned int k);
    
    template void rgivens_rightrot
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& A,
     const vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& p,
     const unsigned int i, const unsigned int j, const unsigned int k);
    
    
    
    template void rgivens_leftrot< blas_real<float> >
      (matrix< blas_real<float> >& A, const vertex< blas_real<float> >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    template void rgivens_leftrot< blas_real<double> >
      (matrix< blas_real<double> >& A, const vertex< blas_real<double> >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    template void rgivens_leftrot< float >
      (matrix< float >& A, const vertex< float >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    template void rgivens_leftrot< double >
      (matrix< double >& A, const vertex< double >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    template void rgivens_leftrot< blas_complex<float> >
      (matrix< blas_complex<float> >& A, const vertex< blas_complex<float> >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    template void rgivens_leftrot< blas_complex<double> >
      (matrix< blas_complex<double> >& A, const vertex< blas_complex<double> >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);

    template void rgivens_leftrot
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<float>, modular<unsigned int> > >& A,
     const vertex< superresolution<blas_real<float>, modular<unsigned int> > >& p,
     const unsigned int i, const unsigned int j, const unsigned int k);
    
    template void rgivens_leftrot
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<double>, modular<unsigned int> > >& A,
     const vertex< superresolution<blas_real<double>, modular<unsigned int> > >& p,
     const unsigned int i, const unsigned int j, const unsigned int k);

    template void rgivens_leftrot
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& A,
     const vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& p,
     const unsigned int i, const unsigned int j, const unsigned int k);
    
    template void rgivens_leftrot
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& A,
     const vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& p,
     const unsigned int i, const unsigned int j, const unsigned int k);    
    
    
    
    /*
     * calculates householder rotation vector for real matrix M
     * rowdir == true: calculates related 
     *   householder row vector for M(y:N, x)
     * rowdir == false: calculates related
     *   householder col vector for M(y, x:N)
     */
    template <typename T>
    bool rhouseholder_vector(vertex<T>& v, const matrix<T>& M,
			     unsigned int y, unsigned int x,
			     bool rowdir)
    {
      try{
	T nrm;
	
	if(rowdir){
	  nrm = M.rownorm(y,x);
	  v.resize(M.xsize() - x);
	  
	  M.rowcopyto(v, y, x);
	}
	else{
	  nrm = M.colnorm(x, y);
	  v.resize(M.ysize() - y);
	  
	  M.colcopyto(v, x, y);
	}
	
	
	if(nrm != T(0.0)){
	  T beta = v[0];
	  
	  if(v[0] > T(0.0))
	    beta += nrm;
	  else
	    beta -= nrm;

#if 1
	  // handle small values causing floating point exception
	  if(typeid(T) == typeid(blas_real<float>) ||
	     typeid(T) == typeid(float)){
	    if(whiteice::math::abs(beta) < T(1e-20f))
	      return false;
	  }
	  else if(typeid(T) == typeid(blas_real<double>) ||
		  typeid(T) == typeid(double)){
	    if(whiteice::math::abs(beta) < T(1e-200))
	      return false;
	  }
#else
	  // HACK: handle small values causing floating point exception
	  // we increase value to be large enough
	  if(typeid(T) == typeid(blas_real<float>) ||
	     typeid(T) == typeid(float)){
	    if(whiteice::math::abs(beta) < 1e-20f){
	      if(beta >= T(0.0f)){
		beta = T(1e-30f);
	      }
	      else if(beta < T(0.0f)){
		beta = T(-1e-30f);
	      }
	    }
	  }
	  else if(typeid(T) == typeid(blas_real<double>) ||
		  typeid(T) == typeid(double)){
	    if(whiteice::math::abs(beta) < 1e-200){
	      if(beta >= T(0.0f)){
		beta = T(1e-300);
	      }
	      else if(beta < T(0.0f)){
		beta = T(-1e-300);
	      }
	    }
	  }
#endif	  
	  
	  v /= beta;
	}
	
	
	v[0] = T(1.0);
	return true;
      }
      catch(std::exception& e){
	std::cout << "failure: " << e.what() << std::endl;
	return false;
      }
    }
    
    
    
    
    /* A = P*A */
    template <typename T>
    bool rhouseholder_leftrot(matrix<T>& A,
			      const unsigned int i,
			      const unsigned int M,
			      const unsigned int k,
			      vertex<T>& v)
    {
      try{
	vertex<T> w;
	vertex<T> vv = v * v;
	T beta = T(-2.0) / vv[0];

	if(w.resize(M) != M || A.ysize() - k != v.size())
	  return false;

#ifdef CUBLAS
	
	const T zero = T(0.0f);
	const T one = T(1.0f);

	if(typeid(T) == typeid(blas_real<float>)){
	  // w = beta * A' * v
	  /*
	  cblas_sgemv(CblasRowMajor, CblasTrans, v.size(), M,
		      *((float*)&beta), (float*)&(A.data[k*A.numCols + i]), A.xsize(),
		      (float*)v.data, 1, 0.0f, (float*)w.data, 1);
	  */

	  auto e = cublasSgemv(cublas_handle, CUBLAS_OP_T,
			       v.size(), M,
			       (const float*)&beta,
			       (const float*)&(A[k + i*A.numRows]), A.ysize(),
			       (const float*)v.data, 1,
			       (const float*)&zero,
			       (float*)w.data, 1);

	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("rhouseholder_leftrot(): cublasSgemv() failed.");
	    throw CUDAException("CUBLAS cublasSgemv() call failed.");
	  }
	  
	  // A += v * w';
	  /*
	  cblas_sger(CblasRowMajor, v.size(), M,
		     1.0f, (float*)v.data, 1, (float*)w.data, 1,
		     (float*)&(A.data[k*A.numCols + i]), A.xsize());
	  */

	  e = cublasSger(cublas_handle, v.size(), M,
			 (const float*)&one,
			 (const float*)v.data, 1, (const float*)w.data, 1,
			 (float*)(&(A.data[k + i*A.numRows])), A.ysize());

	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("rhouseholder_leftrot(): cublasSger() failed.");
	    throw CUDAException("CUBLAS cublasSger() call failed.");
	  }

	  gpu_sync();
	  
	}
	else if(typeid(T) == typeid(blas_real<double>)){
#if 0
	  // w = beta * A' * v
	  cblas_dgemv(CblasRowMajor, CblasTrans, v.size(), M,
		      *((double*)&beta), (double*)&(A.data[k*A.numCols + i]), A.xsize(),
		      (double*)v.data, 1, 0.0, (double*)w.data, 1);
	  
	  // A += v* w';
	  cblas_dger(CblasRowMajor, v.size(), M,
		     1.0, (double*)v.data, 1, (double*)w.data, 1,
		     (double*)&(A.data[k*A.numCols + i]), A.xsize());
#endif

	  auto e = cublasDgemv(cublas_handle, CUBLAS_OP_T,
			       v.size(), M,
			       (const double*)&beta,
			       (const double*)&(A[k + i*A.numRows]), A.ysize(),
			       (const double*)v.data, 1,
			       (const double*)&zero,
			       (double*)w.data, 1);

	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("rhouseholder_leftrot(): cublasDgemv() failed.");
	    throw CUDAException("CUBLAS cublasDgemv() call failed.");
	  }
	  
	  e = cublasDger(cublas_handle, v.size(), M,
			 (const double*)&one,
			 (const double*)v.data, 1, (const double*)w.data, 1,
			 (double*)(&(A.data[k + i*A.numRows])), A.ysize());

	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("rhouseholder_leftrot(): cublasDger() failed.");
	    throw CUDAException("CUBLAS cublasDger() call failed.");
	  }

	  gpu_sync();
	  
	}
	else{
	  // NOT PROPERLY TESTED!
	  // (AND/OR DO THE MATH AGAIN AND CHECK THIS ACTUALLY WORKS)
	  
	  // w = beta * A' * v
	  for(unsigned int l=0;l<w.size();l++){
	    w[l] = T(0.0);
	    
	    for(unsigned int j=0;j<v.size();j++)
	      w[l] += A(k + j, i + l) * v[j];
	    
	    w[l] *= beta;
	  }
	  
	  
	  // A += v * w'
	  matrix<T> deltaA = v.outerproduct(w);
	  
	  for(unsigned int j=0;j<v.size();j++)
	    for(unsigned int l=0;l<w.size();l++)
	      A(k + j, i + l) += deltaA(j,l);
	}
		
#else
	if(typeid(T) == typeid(blas_real<float>)){
	  // w = beta * A' * v
	  cblas_sgemv(CblasRowMajor, CblasTrans, v.size(), M,
		      *((float*)&beta), (float*)&(A.data[k*A.numCols + i]), A.xsize(),
		      (float*)v.data, 1, 0.0f, (float*)w.data, 1);
	  
	  // A += v * w';
	  cblas_sger(CblasRowMajor, v.size(), M,
		     1.0f, (float*)v.data, 1, (float*)w.data, 1,
		     (float*)&(A.data[k*A.numCols + i]), A.xsize());
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  // w = beta * A' * v
	  cblas_dgemv(CblasRowMajor, CblasTrans, v.size(), M,
		      *((double*)&beta), (double*)&(A.data[k*A.numCols + i]), A.xsize(),
		      (double*)v.data, 1, 0.0, (double*)w.data, 1);
	  
	  // A += v* w';
	  cblas_dger(CblasRowMajor, v.size(), M,
		     1.0, (double*)v.data, 1, (double*)w.data, 1,
		     (double*)&(A.data[k*A.numCols + i]), A.xsize());
	}
	else{
	  // NOT PROPERLY TESTED!
	  // (AND/OR DO THE MATH AGAIN AND CHECK THIS ACTUALLY WORKS)
	  
	  // w = beta * A' * v
	  for(unsigned int l=0;l<w.size();l++){
	    w[l] = T(0.0);
	    
	    for(unsigned int j=0;j<v.size();j++)
	      w[l] += A(k + j, i + l) * v[j];
	    
	    w[l] *= beta;
	  }
	  
	  
	  // A += v * w'
	  matrix<T> deltaA = v.outerproduct(w);
	  
	  for(unsigned int j=0;j<v.size();j++)
	    for(unsigned int l=0;l<w.size();l++)
	      A(k + j, i + l) += deltaA(j,l);
	}
#endif
	
	return true;
      }
      catch(std::exception& e){
	std::cout << "failure: " << e.what() << std::endl;
	return false;
      }
    }
    
    
    
    
    
    
    // A(i;i+M,k:k+dim(v)) = A(i+M:k+dim(v))*P
    template <typename T>
    bool rhouseholder_rightrot(matrix<T>& A,
			       const unsigned int i,
			       const unsigned int M,
			       const unsigned int k,
			       vertex<T>& v)
    {
      try{
	vertex<T> w;
	vertex<T> vv = v * v;
	T beta = T(-2.0) / vv[0];

	if(w.resize(M) != M || A.xsize() - k != v.size())
	  return false;
	
#ifdef CUBLAS

	const T one  = T(1.0f);
	const T zero = T(0.0f);

	if(typeid(T) == typeid(blas_real<float>)){
#if 0
	  // w = beta * A * v
	  cblas_sgemv(CblasRowMajor, CblasNoTrans, M, v.size(),
		      *((float*)&beta), (float*)&(A.data[i*A.numCols + k]), A.xsize(),
		      (float*)v.data, 1, 0.0f, (float*)w.data, 1);
	  
	  // A += w * v';
	  cblas_sger(CblasRowMajor, M, v.size(),
		     1.0f, (float*)w.data, 1, (float*)v.data, 1,
		     (float*)&(A.data[i*A.numCols + k]), A.xsize());
#endif

	  auto e = cublasSgemv(cublas_handle, CUBLAS_OP_N,
			       M, v.size(),
			       (const float*)&beta,
			       (const float*)&(A[k + i*A.numRows]), A.ysize(),
			       (const float*)v.data, 1,
			       (const float*)&zero,
			       (float*)w.data, 1);

	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("rhouseholder_rightrot(): cublasSgemv() failed.");
	    throw CUDAException("CUBLAS cublasSgemv() call failed.");
	  }
	  
	  e = cublasSger(cublas_handle, M, v.size(),
			 (const float*)&one,
			 (const float*)w.data, 1, (const float*)v.data, 1,
			 (float*)(&(A.data[k + i*A.numRows])), A.ysize());

	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("rhouseholder_rightrot(): cublasSger() failed.");
	    throw CUDAException("CUBLAS cublasSger() call failed.");
	  }

	  gpu_sync();
	  
	}
	else if(typeid(T) == typeid(blas_real<double>)){
#if 0
	  // w = beta * A * v
	  cblas_dgemv(CblasRowMajor, CblasNoTrans, M, v.size(),
		      *((double*)&beta), (double*)&(A.data[i*A.numCols + k]), A.xsize(),
		      (double*)v.data, 1, 0.0f, (double*)w.data, 1);
	  
	  // A += v* w';
	  cblas_dger(CblasRowMajor, M, v.size(),
		     1.0, (double*)w.data, 1, (double*)v.data, 1,
		     (double*)&(A.data[i*A.numCols + k]), A.xsize());
#endif

	  auto e = cublasDgemv(cublas_handle, CUBLAS_OP_N,
			       M, v.size(),
			       (const double*)&beta,
			       (const double*)&(A[k + i*A.numRows]), A.ysize(),
			       (const double*)v.data, 1,
			       (const double*)&zero,
			       (double*)w.data, 1);

	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("rhouseholder_rightrot(): cublasDgemv() failed.");
	    throw CUDAException("CUBLAS cublasDgemv() call failed.");
	  }
	  
	  e = cublasDger(cublas_handle, M, v.size(),
			 (const double*)&one,
			 (const double*)w.data, 1, (const double*)v.data, 1,
			 (double*)(&(A.data[k + i*A.numRows])), A.ysize());

	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("rhouseholder_rightrot(): cublasDger() failed.");
	    throw CUDAException("CUBLAS cublasDger() call failed.");
	  }

	  gpu_sync();
	  
	}
	else{
	  // NOT PROPERLY TESTED!
	  
	  // w = beta * A * v
	  for(unsigned int l=0;l<w.size();l++){
	    w[l] = T(0.0);
	    
	    for(unsigned int j=0;j<v.size();j++)
	      w[l] += A(i + l, k + j) * v[j];
	    
	    w[l] *= beta;
	  }
	  
	  
	  // A += w * v'
	  matrix<T> deltaA = w.outerproduct(v);
	  
	  for(unsigned int l=0;l<w.size();l++)
	    for(unsigned int j=0;j<v.size();j++)
	      A(i + l, k + j) += deltaA(l,j);
	  
	}
	
#else
	if(typeid(T) == typeid(blas_real<float>)){
	  // w = beta * A * v
	  cblas_sgemv(CblasRowMajor, CblasNoTrans, M, v.size(),
		      *((float*)&beta), (float*)&(A.data[i*A.numCols + k]), A.xsize(),
		      (float*)v.data, 1, 0.0f, (float*)w.data, 1);
	  
	  // A += w * v';
	  cblas_sger(CblasRowMajor, M, v.size(),
		     1.0f, (float*)w.data, 1, (float*)v.data, 1,
		     (float*)&(A.data[i*A.numCols + k]), A.xsize());
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  // w = beta * A * v
	  cblas_dgemv(CblasRowMajor, CblasNoTrans, M, v.size(),
		      *((double*)&beta), (double*)&(A.data[i*A.numCols + k]), A.xsize(),
		      (double*)v.data, 1, 0.0f, (double*)w.data, 1);
	  
	  // A += v* w';
	  cblas_dger(CblasRowMajor, M, v.size(),
		     1.0, (double*)w.data, 1, (double*)v.data, 1,
		     (double*)&(A.data[i*A.numCols + k]), A.xsize());
	}
	else{
	  // NOT PROPERLY TESTED!
	  
	  // w = beta * A * v
	  for(unsigned int l=0;l<w.size();l++){
	    w[l] = T(0.0);
	    
	    for(unsigned int j=0;j<v.size();j++)
	      w[l] += A(i + l, k + j) * v[j];
	    
	    w[l] *= beta;
	  }
	  
	  
	  // A += w * v'
	  matrix<T> deltaA = w.outerproduct(v);
	  
	  for(unsigned int l=0;l<w.size();l++)
	    for(unsigned int j=0;j<v.size();j++)
	      A(i + l, k + j) += deltaA(l,j);
	  
	}
#endif
	
	return true;
      }
      catch(std::exception& e){
	std::cout << "failure: " << e.what() << std::endl;
	return false;
      }
    }



    
    // calculates givens rotation
    template <typename T>
    void rgivens(const T& a, const T& b, vertex<T>& p)
    {
      // p.resize(2); // [c s] MUST HAVE SIZE 2
      
      if(b == T(0.0)){
	p[0] = T(1.0);
	p[1] = T(0.0);
      }
      else{
	T th;

	if(whiteice::math::abs(b) > whiteice::math::abs(a)){
	  th = -a / b;
	  p[1]  = T(1.0)/sqrt(T(1.0) + th*th);
	  p[0]  = p[1]*th;
	}
	else{
	  th = -b / a;
	  p[0]  = T(1.0)/sqrt(T(1.0) + th*th);
	  p[1]  = p[0]*th;
	}
      }
    }
    
        
    // givens right rotation (columns)
    template <typename T>
    void rgivens_rightrot(matrix<T>& A,
			  const vertex<T>& p,
			  const unsigned int i,
			  const unsigned int j,
			  const unsigned int k)
    {
      T t[2];
      
      for(unsigned int l=i;l<j;l++){
	t[0] = A(l,k);
	t[1] = A(l,k+1);
	
	A(l,k)   = p[0]*t[0] - p[1]*t[1];
	A(l,k+1) = p[0]*t[1] + p[1]*t[0];
      }
    }
    
    
    // givens left rotation (rows)
    template <typename T>
    void rgivens_leftrot(matrix<T>& A,
			 const vertex<T>& p,
			 const unsigned int i,
			 const unsigned int j,
			 const unsigned int k)
    {
      T t[2];
      
      for(unsigned int l=i;l<j;l++){
	t[0] = A(k,l);
	t[1] = A(k+1,l);
	
	A(k,l)   = p[0]*t[0] - p[1]*t[1];
	A(k+1,l) = p[0]*t[1] + p[1]*t[0];
      }
    }
    
    
    
    
    
    
#if 0
    /////////////////////////////////////////////////
    
    // calculates householder rotation vectors
    template <typename T>
    void householder_vector(const vertex<T>& v,
			    const unsigned int i,
			    vertex<T>& x,
			    const unsigned int j,
			    bool iscomplex)
    {
      const unsigned int N = v.size() - i;
      x.resize(j+N);
      for(unsigned int k=0;k<N;k++)
	x[j+k] = v[i+k];
      
      T u = v.norm(i,v.size());
      
      if(u != T(0)){
	if(iscomplex){
	  T t, s, z;
	  z = -1; z = sqrt(z);
	  z = whiteice::math::exp(whiteice::math::arg(x[i])*z);
	  
	  t = x[i] + u * z;
	  s = x[i] - u * z;
	  
	  if(whiteice::math::abs(t) > whiteice::math::abs(s)){
	    x[i] = t;
	  }
	  else{
	    x[i] = s;
	  }
	  
	  for(unsigned int k=N-1;k>0;k--)
	    x[i+k] /= x[i];
	}
	else{
	  if(whiteice::math::real(v[i]) >= 0)
	    x[i] += u;
	  else
	    x[i] -= u;
	  
	  for(unsigned int k=N-1;k>0;k--)
	    x[i+k] /= x[i];
	}
      }
      
      x[i] = 1;
    }
    
    
    // rotates (sub)matrix with householder rotation (sub)vector
    
    // rotates from right (columns, A = AP)
    // A(i:(j-1),k:k+N-l) * P( x(l:N) ) , N = length(x)
    template <typename T>
    void householder_rightrot(matrix<T>& A,
			      const unsigned int i, const unsigned int j, const unsigned int k,
			      vertex<T>& x, const unsigned int l)
    {
      vertex<T> w;
      T vv = x.norm(l,x.size());
      vv = vv*vv;
      vv = -2/vv;
      
      w.resize(j-i);
      
      const unsigned int N = x.size();
      const unsigned int M = w.size();
      
      for(unsigned int m=0;m<M;m++){
	w[m] = T(0);
	
	for(unsigned int p=l;p<N;p++)
	  w[m] += A(m+i, k+p-l)*x[p];
	
	w[m] *= vv;
      }
      
      const unsigned int L = N - l;
      
      // saves result into A
      for(unsigned int p=i;p<j;p++)
	for(unsigned int q=k;q<L;q++)
	  A(p,q) += w[p-i]*conj(x[q-i+l]);
    }
    
    
    // rotates from left (rows, A = PA)
    // P( x(l:N) ) * A(k:k+N-l,i:(j-1)), N = length(x)
    template <typename T>
    void householder_leftrot(matrix<T>& A,
			     const unsigned int i, const unsigned int j, const unsigned int k,
			     vertex<T>& x, const unsigned int l)
    {
      vertex<T> w;
      T vv = x.norm(l,x.size());
      vv = vv*vv;
      vv = -2/vv;
      
      w.resize(j-i);
      
      const unsigned int N = x.size();
      const unsigned int M = w.size();
      
      for(unsigned int m=0;m<M;m++){
	w[m] = T(0);
	
	for(unsigned int p=l;p<N;p++)
	  w[m] += conj(x[p]) * A(k+p-l,m+i);
	
	w[m] *= vv;
      }
      
      const unsigned int L = N - l;
      
      // saves result into A
      for(unsigned int p=i;p<j;p++)
	for(unsigned int q=k;q<L;q++)
	  A(p,q) += w[p-i]*x[q-i+l];
    }

    
    /////////////////////////////////////////////////
    
    
    // calculates fast givens rotation parameters
    template <typename T>
    void fastgivens(const vertex<T>& x,
		    vertex<T>& d,
		    vertex<T>& p)
    {
      p.resize(3); // [a b t]
      
      if(x[1] != 0){
	p[0] = - x[0] / x[1];
	p[1] = - p[0] * d[1] / d[0];
	g = p[0]*p[1];
	
	if(g <= 1){
	  T th = d[0];
	  d[0] = (1 + g)*d[1]; // what's the point? BUG?
	  d[1] = (1 + g)*th;
	  
	  p[2] = 1;
	}
	else{
	  p[0] = 1 / p[0];
	  p[1] = 1 / p[1];
	  p[2] = 2;
	  g    = 1 / g;
	  
	  d[0] = (1 + g)*d[0];
	  d[1] = (1 + g)*d[1];
	}
      }
      else{
	p[0] = 0;
	p[1] = 0;
	p[2] = 2;
      }
      
    }
    
    
    
    // fastgivens row rotation of A(k:k+1,i:j-1)
    template <typename T>
    void fastgivens_leftrot(matrix<T>& A,
			    const vertex<T>& p,
			    const unsigned int i, const unsigned int j,
			    const unsigned int k)
    {
      T t[2];
      
      if(p[2] == 1){
	for(unsigned int l=i;l<j;l++){
	  t[0] = A(k,l);
	  t[1] = A(k+1,l);
	  
	  A(k,l)   = p[1]*t[0] + t[1];
	  A(k+1,l) = t[0] + p[0]*t[1];
	}
      }
      else{
	for(unsigned int l=i;l<j;l++){
	  t[0] = A(k,l);
	  t[1] = A(k+1,l);
	  
	  A(k,l)   = t[0] + p[1]*t[1];
	  A(k+1,l) = p[0]*t[0] + t[1];
	}
      }
    }
    
    
    template <typename T>
    void fastgivens_rightrot(matrix<T>& A,
			     const vertex<T>& p,
			     const unsigned int i, const unsigned int j,
			     const unsigned int k)
    {
      // implement me
      // (+ read theory (cannot remember) + test + bugfix )
    }    
    
    
#endif
    
    
    
  }
}






