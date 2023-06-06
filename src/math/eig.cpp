
#include <iostream>
#include <typeinfo>
#include <math.h>
#include <stdlib.h>

#include "eig.h"
#include "blade_math.h"
#include "matrix_rotations.h"
#include "dinrhiw_blas.h"

#include <map>
#include <assert.h>



namespace whiteice
{
  
  namespace math
  {
    
    // explicit template instantations
    
    template bool eig2x2matrix< blas_real<float> >
      (const matrix< blas_real<float> >& A, vertex< blas_real<float> >& d, 
       matrix< blas_real<float> >& X, bool complex_ok);
    template bool eig2x2matrix< blas_real<double> >
      (const matrix< blas_real<double> >& A, vertex< blas_real<double> >& d,
       matrix< blas_real<double> >& X, bool complex_ok);
    template bool eig2x2matrix< blas_complex<float> >
      (const matrix< blas_complex<float> >& A, vertex< blas_complex<float> >& d,
       matrix< blas_complex<float> >& X, bool complex_ok);
    template bool eig2x2matrix< blas_complex<double> >
      (const matrix< blas_complex<double> >& A, vertex< blas_complex<double> >& d,
       matrix< blas_complex<double> >& X, bool complex_ok);
    template bool eig2x2matrix<float>
      (const matrix<float>& A, vertex<float>& d, matrix<float>& X, bool complex_ok);
    template bool eig2x2matrix<double>
      (const matrix<double>& A, vertex<double>& d, matrix<double>& X, bool complex_ok);
    //template bool eig2x2matrix< complex<float> >
    //  (const matrix< complex<float> >& A, vertex< complex<float> >& d,
    //   matrix< complex<float> >& X, bool complex_ok);
    //template bool eig2x2matrix< complex<double> >
    //  (const matrix< complex<double> >& A, vertex< complex<double> >& d,
    //   matrix< complex<double> >& X, bool complex_ok);

    template bool eig2x2matrix
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (const matrix< superresolution<blas_real<float>, modular<unsigned int> > >& A,
     vertex< superresolution<blas_real<float>, modular<unsigned int> > >& d,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& X,
     bool complex_ok);
    
    template bool eig2x2matrix
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (const matrix< superresolution<blas_real<double>, modular<unsigned int> > >& A,
     vertex< superresolution<blas_real<double>, modular<unsigned int> > >& d,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& X,
     bool complex_ok);
    
    template bool eig2x2matrix
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (const matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& A,
     vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& d,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& X,
     bool complex_ok);
    
    template bool eig2x2matrix
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (const matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& A,
     vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& d,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& X,
     bool complex_ok);
    
    
       
       
    template bool hessenberg_reduction< blas_real<float> >
    (matrix< blas_real<float> >& A, matrix< blas_real<float> >& Q);
    template bool hessenberg_reduction< blas_real<double> >
    (matrix< blas_real<double> >& A, matrix< blas_real<double> >& Q);
    template bool hessenberg_reduction<float>
    (matrix<float>& A, matrix<float>& Q);
    template bool hessenberg_reduction<double>
    (matrix<double>& A, matrix<double>& Q);
    template bool hessenberg_reduction< blas_complex<float> >
    (matrix< blas_complex<float> >& A, matrix< blas_complex<float> >& Q);
    template bool hessenberg_reduction< blas_complex<double> >
    (matrix< blas_complex<double> >& A, matrix< blas_complex<double> >& Q);

    template bool hessenberg_reduction
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<float>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& Q);
    template bool hessenberg_reduction
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<double>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& Q);
    
    template bool hessenberg_reduction
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& Q);
    template bool hessenberg_reduction
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& Q);
    
    
    
    template bool qr< blas_real<float> > (matrix< blas_real<float> >&  A,
					  matrix< blas_real<float> >&  Q);
    template bool qr< blas_real<double> >(matrix< blas_real<double> >& A,
					  matrix< blas_real<double> >& Q);
    template bool qr<float> (matrix<float>&  A, matrix<float>&  Q);
    template bool qr<double>(matrix<double>& A, matrix<double>& Q);
    template bool qr< blas_complex<float> > (matrix< blas_complex<float> >&  A,
					     matrix< blas_complex<float> >&  Q);
    template bool qr< blas_complex<double> >(matrix< blas_complex<double> >& A,
					     matrix< blas_complex<double> >& Q);

    template bool qr
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<float>, modular<unsigned int> > >&  A,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >&  Q);
    template bool qr
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<double>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& Q);
    
    template bool qr
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<float>, modular<unsigned int> > >&  A,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >&  Q);
    template bool qr
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& Q);
    
    
    template bool implicit_symmetric_qrstep_wilkinson< blas_real<float> >
      (matrix< blas_real<float> >& A, matrix< blas_real<float> >& X,
       unsigned int e1, unsigned int N);
    template bool implicit_symmetric_qrstep_wilkinson< blas_real<double> >
      (matrix< blas_real<double> >& A, matrix< blas_real<double> >& X,
       unsigned int e1, unsigned int N);
    template bool implicit_symmetric_qrstep_wilkinson<float>
      (matrix<float>& A, matrix<float>& X,
       unsigned int e1, unsigned int N);
    template bool implicit_symmetric_qrstep_wilkinson<double>
      (matrix<double>& A, matrix<double>& X,
       unsigned int e1, unsigned int N);
    template bool implicit_symmetric_qrstep_wilkinson< blas_complex<float> >
      (matrix< blas_complex<float> >& A, matrix< blas_complex<float> >& X,
       unsigned int e1, unsigned int N);
    template bool implicit_symmetric_qrstep_wilkinson< blas_complex<double> >
      (matrix< blas_complex<double> >& A, matrix< blas_complex<double> >& X,
       unsigned int e1, unsigned int N);

    template bool implicit_symmetric_qrstep_wilkinson
    < superresolution<blas_real<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<float>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& X,
     unsigned int e1, unsigned int N);
    template bool implicit_symmetric_qrstep_wilkinson
    < superresolution<blas_real<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<double>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& X,
     unsigned int e1, unsigned int N);

    template bool implicit_symmetric_qrstep_wilkinson
    < superresolution<blas_complex<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& X,
     unsigned int e1, unsigned int N);
    template bool implicit_symmetric_qrstep_wilkinson
    < superresolution<blas_complex<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& X,
     unsigned int e1, unsigned int N);
    
    
    template bool symmetric_eig< blas_real<float> >
    (matrix< blas_real<float> >& A, matrix< blas_real<float> >& D, bool sort);
    template bool symmetric_eig< blas_real<double> >
    (matrix< blas_real<double> >& A, matrix< blas_real<double> >& D, bool sort);
    template bool symmetric_eig<float>(matrix<float>& A, matrix<float>& D, bool sort);
    template bool symmetric_eig<double>(matrix<double>& A, matrix<double>& D, bool sort);
    
    //template bool symmetric_eig< complex<float> >(matrix< complex<float> >& A, matrix< complex<float> >& D, bool sort);
    //template bool symmetric_eig< complex<double> >(matrix< complex<double> >& A, matrix< complex<double> >& D, bool sort);
    template bool symmetric_eig< blas_complex<float> >(matrix< blas_complex<float> >& A, matrix< blas_complex<float> >& D, bool sort);
    template bool symmetric_eig< blas_complex<double> >(matrix< blas_complex<double> >& A, matrix< blas_complex<double> >& D, bool sort);

    template bool symmetric_eig< superresolution<blas_real<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<float>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& D,
     bool sort);
    
    template bool symmetric_eig< superresolution<blas_real<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_real<double>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& D,
     bool sort);

    template bool symmetric_eig< superresolution<blas_complex<float>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& D,
     bool sort);
    
    template bool symmetric_eig< superresolution<blas_complex<double>, modular<unsigned int> > >
    (matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& A,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& D,
     bool sort);
    
    
    template bool svd< blas_real<float> >
      (matrix< blas_real<float> >& A,
       matrix< blas_real<float> >& U, matrix< blas_real<float> >& V);
    template bool svd< blas_real<double> >
      (matrix< blas_real<double> >& A,
       matrix< blas_real<double> >& U, matrix< blas_real<double> >& V);
    template bool svd<float>(matrix<float>& A, matrix<float>& U, matrix<float>& V);
    template bool svd<double>(matrix<double>& A, matrix<double>& U, matrix<double>& V);
    
    
    /***********************************************************************************/
    
    /*
     * solves eigenproblem for 2x2 matrix, AX = XD
     *
     * This is used in special cases and (will be used) 
     * as a final step for diagonalizing matrices which are
     * already in a block-diagonal form. (non-symmetric EVD)
     */
    template <typename T>
    inline bool eig2x2matrix(const matrix<T>& A,
			     vertex<T>& d, matrix<T>& X,
			     bool complex_ok)
    {
      
      // solves eigenvalues
      // 
      // solutions comes from EQs:
      // trace(A) = A(0,0) + A(1,1) = eig1 + eig2
      // det(A)   = A(0,0)*A(1,1) - A(1,0)*A(0,1) = eig1*eig2
      
      d.resize(2);
      T temp = T(0.5);
      
      d[0] = (A(0,0) + A(1,1)) * temp;
      d[1] = (A(0,0) - A(1,1)) * temp;
      
      temp = d[1]*d[1] + A(1,0)*A(0,1);
      
      if(whiteice::math::real(temp) < whiteice::math::real(T(0)) && !complex_ok) // are eigenvalues complex?
	return false;
      
      temp = whiteice::math::sqrt( temp );
      
      d[1] = d[0] + temp;
      d[0] -= temp;
      
      // solves eigenvectors
      
      X.resize(2,2);
      
      if(temp != T(0.0f)){ // different eigenvalues (full rank - unless zero eigenvalues)
	
	for(unsigned int i=0;i<2;i++){
	  if(A(0,1) != T(0.0)){
	    X(0,i) = T(1.0);
	    X(1,i) = (- A(0,0) + d[i])/A(0,1);
	  }
	  else if(A(1,0) != T(0.0)){
	    X(1,i) = (- A(1,1) + d[i])/A(1,0);
	    X(0,i) = T(1.0);
	  }
	  else{
	    if(whiteice::math::real(whiteice::math::abs(A(0,0) - d[i])) >
	       whiteice::math::real(T(0.0)))
	    {
	      X(0,i) = T(0.0);
	      X(1,i) = T(1.0);
	    }
	    else{
	      X(0,i) = T(1.0);
	      X(1,i) = T(0.0);
	    }
	  }
	  
	  // normalizes length of eigenvector
	  temp = whiteice::math::conj(X(0,i))*X(0,i) + whiteice::math::conj(X(1,i))*X(1,i);
	  temp = T(1.0) / whiteice::math::sqrt(temp);
	  
	  X(0,i) *= temp;
	  X(1,i) *= temp;
	}
	
      }
      else{ // same eigenvalues
	
	// eigenvector 1
	if(A(0,1) != T(0.0f)){
	  X(0,0) = T(1.0f);
	  X(1,0) = (- A(0,0) + d[0])/A(0,1);
	}
	else if(A(1,0) != T(0)){
	  X(0,0) = (- A(1,1) + d[0])/A(1,0);
	  X(1,0) = T(1.0f);
	}
	else{
	  // both diagonals are zero -> X = I
	  X.identity();
	  return true;
	}
	
	// normalizes length of eigenvector
	temp = whiteice::math::conj(X(0,0))*X(0,0) + whiteice::math::conj(X(1,0))*X(1,0);
	temp = T(1.0) / whiteice::math::sqrt(temp);
	
	X(0,0) *= temp;
	X(1,0) *= temp;
	
	
	// eigenvector 2
	if(A(0,1) != T(0.0f)){
	  X(0,1) = T(1.0f);
	  X(1,1) = (- A(0,0) + d[1])/A(0,1);
	}
	else if(A(1,0) != T(0)){
	  X(0,1) = (- A(1,1) + d[1])/A(1,0);
	  X(1,1) = T(1.0f);
	}
	else{ // note: should be impossible
	  X(0,1) = T(0.0f);
	  X(1,1) = T(1.0f);
	}
	
	
	// normalizes length of eigenvector
	temp = whiteice::math::conj(X(0,1))*X(0,1) + whiteice::math::conj(X(1,1))*X(1,1);
	temp = T(1.0) / whiteice::math::sqrt(temp);
	
	X(0,1) *= temp;
	X(1,1) *= temp;
	
      } // if ... different/same eigenvectors
      
      return true;
    }
    
    
    /***********************************************************************************/
    
    // solves eigenvectors for eigenvalues
    // with inverse power method
    template <typename T>
    inline void invpowermethod(const matrix<T>& A,
			       const vertex<T>& e,
			       const vertex<T>& n,
			       matrix<T>& X)
    {
      const unsigned int N = A.size();            
      
      X.resize(N,N);
      
      // intitializes X with random data
      for(unsigned int j=0;j<N;j++){
	
	do{
	  for(unsigned int i=0;i<N;i++)
	    X(j,i) = (((double)rand()) / ((double)RAND_MAX)) - 0.5;
	}
	while(!X[j].normalize());
      }
      
      matrix<T> B(N,N);
      
      
      for(unsigned int i=0,k=0;i<e.size();i++){
	
	// calculates related inverse matrix
	// !!! this should be done so that singular
	// matrix can be reversed by estimating 0 = small_epsilon
	// in matrix inverse calculation code
	B = A;	
	for(unsigned int j=0;j<N;j++)
	  B(j,j) -= e[i];
	
	B.inv(); // !! this may fail - code better aprox inv.
	
	// 1st iteration
	for(unsigned int j=0;j<n[i];j++)
	  X[k+j] = B*X[k+j];
	
	if(n[i] > 1) gramschimdt(X,k,k+n[i]);
	else X[k].normalize();
	
	// 2nd iteration
	for(unsigned int j=0;j<n[i];j++)
	  X[k+j] = B*X[k+j];
	
	if(n[i] > 1) gramschimdt(X,k,k+n[i]);
	else X[k].normalize();
	
	//!! eigenvectors are now almost surely
	//   correct (even after 1st iteration).
	//   if exact correctness of
	//   eigenvectors is important one
	//   should check for convergence.
	
	k += n[i];
      }
      
    }

    
    /***********************************************************************************/
        
    // reduces A to hessenberg form Q*H*Q'
    template <typename T>
    inline bool hessenberg_reduction(matrix<T>& A, matrix<T>& Q)
    {
      try{
	const unsigned int N = A.ysize();
	
	if(N <= 2) // already in hessenberg from
	  return true;
	
	vertex<T> v;
	std::vector< vertex<T> > vlist;		
	
	vlist.resize(N-2);
	v.resize(N);
	
	Q.resize(N,N);
	Q.identity();
	
	
	for(unsigned int k=0;k<N-2;k++){
	  if(!rhouseholder_vector(v, A, k+1, k, false)){
	    return false;
	  }
	  
	  if(!rhouseholder_leftrot(A, k, A.xsize() - k, k+1, v)){
	    return false;
	  }
	  
	  if(!rhouseholder_rightrot(A, 0, A.xsize(), k+1, v)){
	    return false;
	  }
	  
	  vlist[k] = v;
	}
	
	
	// calculates Q
	for(unsigned int k=3;k<=N;k++){
	  if(!rhouseholder_leftrot(Q, (N-k)+1,k-1 ,(N-k)+1, vlist[N-k])){
	    return false;
	  }
	}
	
#if 0
	// this should work but isn't optimized yet
	for(unsigned int k=0;k<N-2;k++){
	  if(!rhouseholder_rightrot(Q, 0, Q.xsize(),k+1, vlist[k]))
	    return false;
	}
#endif
	
	return true;
      }
      catch(std::exception& e){
	std::cout << "householder rotation failure: " << e.what() << std::endl;
	return false;
      }
    }
    
    
    /***********************************************************************************/
    
    // calculates A = Q*R factorization
    // affected parts of A will be replaced with R.
    template <typename T>
    inline bool qr(matrix<T>& A, matrix<T>& Q)
    {
      try{
	const unsigned int N = A.xsize();
	
	std::vector< vertex<T> > vlist;
	vertex<T> v;
	
	vlist.resize(N);
	v.resize(N);
	
	if(N <= 1){
	  Q.resize(1,1);
	  Q(1,1) = T(1.0);
	  return true;
	}
	
	
	for(unsigned int k=0;k<N;k++){
	  if(!rhouseholder_vector(v, A, k, k, false))
	    return false;
	  
	  if(!rhouseholder_leftrot(A, k, N - k, k, v))
	    return false;
	  
	  vlist[k] = v;
	}
	
	
	// calculates Q
	Q.resize(N,N);
	Q.identity();
	
	
	// calculates Q
	for(unsigned int k=1;k<=N;k++){
	  if(!rhouseholder_leftrot(Q, (N-k), k ,(N-k), vlist[N-k]))
	    return false;
	}
	
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }    
    
    
    /***********************************************************************************/
    
    
    template <typename T>
    inline bool implicit_symmetric_qrstep_wilkinson(matrix<T>& A, matrix<T>& X,
						    unsigned int e1, unsigned int N)
    {
      try{
	const unsigned int m = e1;
	const unsigned int M = e1+N; // (M-1) is the biggest possible value
	
	// calculates shift
	
	T s;
	T d = (A(M-2,M-2) - A(M-1,M-1))/T(2.0);
	
	const auto zero = abs(T(0.0f));

	const auto div0 = (d + whiteice::math::sqrt(whiteice::math::abs(d*d + A(M-1,M-2)*A(M-1,M-2))));
	
	if(whiteice::math::abs(div0) > zero){
	  s = A(M-1,M-1) - (A(M-1,M-2)*A(M-1,M-2)) / div0;
	}
	else{
	  auto div = (d - whiteice::math::sqrt(whiteice::math::abs(d*d + A(M-1,M-2)*A(M-1,M-2))));

	  // std::cout << div << std::endl;
	  
	  s = A(M-1,M-1) - (A(M-1,M-2)*A(M-1,M-2)) / div;
	}
	
	T x = A(m,m) - s;
	T z = A(m+1,m);
	vertex<T> p; // givens rotation parameters
	p.resize(2);
	
	for(unsigned int k=m;k<(M-1);k++){	  
	  rgivens(x,z,p);	  
	  
	  rgivens_rightrot(A, p, m, M, k); // column rotation
	  rgivens_leftrot (A, p, m, M, k); // row rotation
	  
	  rgivens_rightrot(X, p, 0, X.ysize(), k); // rotates columns of X
	  
	  
	  // so the the last access doesn't access out of range value
	  // TODO: optimize away (handle k==0 separatedly and do update at
	  //       the beginning of the loop
	  if(k < (M-2)){
	    x = A(k+1,k);
	    z = A(k+2,k);
	  }
	  
	}

	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    /***********************************************************************************/

    // helper data structure for sorting eigenvalues in symmetric_eig()
    template <typename T>
    class eigvaluepair
    {
    public:
      int index;
      T value;
    };
    
    
    template <typename T>
    inline bool symmetric_eig(matrix<T>& A, matrix<T>& X, bool sort)
    {
      // KNOWN_BUG: only works with real valued data,
      // as a compilation-hack converts values to real values which don't work
      
      try{
	if(A.xsize() != A.ysize())
	  return false;
	
	if(X.resize(A.xsize(),A.xsize()) == false)
	  return false;
	
	auto TOLERANCE = abs(T(0.000001));
	auto EPSILON   = abs(T(0.000001));
	unsigned int N = A.xsize();
	
	// special cases for small matrices
	if(N == 1){
	  X[0] = T(1.0f);
	  
	  return true;
	}
	else if(N == 2){
	  vertex<T> d;

	  if(typeid(T) == typeid(blas_complex<float>) || typeid(T) == typeid(blas_complex<double>)){
	    const bool complex_solution_ok = true;
	    if(!eig2x2matrix(A, d, X, complex_solution_ok))
	      return false;
	  }
	  else{
	    const bool complex_solution_ok = false;
	    if(!eig2x2matrix(A, d, X, complex_solution_ok))
	      return false;
	  }
	  
	  A(0,0) = d[0];
	  A(0,1) = T(0.0f);
	  A(1,0) = T(0.0f);
	  A(1,1) = d[1];
	  
	  // return true; (need to sort eigenvalues)..
	}
	else{

	  // calculates first hessenberg reduction of A
	  if(!hessenberg_reduction(A, X))
	    return false;

	  
	  // keeps zeroing below diagonal entries,
	  // because of symmetry A is symmetric and diagonal
	  // when algorithm converges (only if A really is symmetric)
	  
	  unsigned int iter = 0;
	  unsigned int f1 = 0, f2 = N-2;
	  unsigned int e1 = 0, e2 = N-2;
	  T error = T(0);
	  
	  for(unsigned int k=0;k<(N-1);k++){
	    double error_double;
	    double absA_double;

	    T err  = real(error);
	    T absA = real(whiteice::math::abs(A(k+1,k)));

	    convert(error_double, err);
	    convert(absA_double, absA);
	    
	    if(absA_double > error_double)
	      error = absA;
	  }
	  
	  
	  while(1){

	    // while(real(error) > real(TOLERANCE)){
	    {
	      double error_double;
	      double tolerance_double;

	      T tol = real(TOLERANCE);
	      T err = real(error);

	      convert(error_double, err);
	      convert(tolerance_double, tol);

	      if(error_double <= tolerance_double)
		break;
	    }
	    
	    // finds submatrix
	    
	    for(unsigned int k=0;k<e2;k++){
	      // if(real(whiteice::math::abs(A(k+1,k))) > real(EPSILON)){
	      double epsilon_double;
	      double absA_double;

	      T absA = real(whiteice::math::abs(A(k+1,k)));
	      T eps  = real(EPSILON);

	      convert(epsilon_double, eps);
	      convert(absA_double, absA);
	      
	      if(absA_double > epsilon_double){
		e1 = k;
		break;
	      }
	    }
	    
	    for(unsigned int k=(N-2);k>=e1;k--){
	      // if(real(whiteice::math::abs(A(k+1,k))) > real(EPSILON)){
	      double epsilon_double;
	      double absA_double;

	      T absA = real(whiteice::math::abs(A(k+1,k)));
	      T eps  = real(EPSILON);

	      convert(epsilon_double, eps);
	      convert(absA_double, absA);
		
	      if(absA_double > epsilon_double){
		e2 = k+1;
		break;
	      }
	    }

	    
	    // calculates qr step for the non-diagonal submatrix
	    if(!implicit_symmetric_qrstep_wilkinson(A, X, e1, (e2 - e1)+1)){
	      std::cout << "IMPLICIT SHIFT FAILED" << std::endl;
	      return false;
	    }

	    if(f1 == e1 && f2 == e2){
	      iter++;
	      
	      if(iter > 10){ // was 50 iterations (was 5)
		TOLERANCE *= 2.0f; // increases TOLERANCE
		EPSILON   *= 2.0f;
		
		iter = 0;
		f1 = e1;
		f2 = e2;
	      }
	    }
	    else{
	      iter = 0;
	      f1 = e1;
	      f2 = e2;
	      
	    }

	    
	    error = T(0.0);
	    for(unsigned int k=e1;k<e2;k++){
	      // if(real(whiteice::math::abs(A(k+1,k))) > real(error))
	      double error_double;
	      double absA_double;

	      T absA = real(whiteice::math::abs(A(k+1,k)));
	      T err  = real(error);

	      convert(error_double, err);
	      convert(absA_double, absA);
	      
	      
	      if(absA_double > error_double)
		error = whiteice::math::abs(A(k+1,k));
	    }
	    
	  }
	}
	
	
	// sorts eigenvectors according to their variances
	if(sort){
	  
	  std::multimap<double, class eigvaluepair<T> > var;
	  
	  for(unsigned int j=0;j<A.xsize();j++){
	    class eigvaluepair<T> data;
	    data.index = j;
	    data.value = A(j,j);

	    double absvalue = 0.0;
	    whiteice::math::convert(absvalue, abs(A(j,j)));
	    
	    var.insert(std::pair<double, class eigvaluepair<T> >(absvalue, data));
	  }
	  
	  int index = A.xsize() - 1;
	  math::vertex<T> t;
	  
	  // this is slow [write faster code]
	  math::matrix<T> XX(X);
	  
	  for(auto& v : var){ // from smallest to the largest eigenvalue
	    
	    if(v.second.index != index){ // we need to move v.second index to index
	      A(index,index) = v.second.value;
	      
	      XX.colcopyto(t, v.second.index);
	      X.colcopyfrom(t, index);
	    }
	    
	    index--;
	  }
	}
	
	return true;
      }
      catch(std::exception& e){
	std::cout << "SYMMETRIC EIGENVALUE SOLVER FAILED. EXCEPTION: " << e.what() << std::endl;
	return false;
      }
    }
    
    
    /***********************************************************************************/
    
    
    /*
     * calculates singular value decomposition of A = U*S*V^T
     * A will be overwriten with singular values.
     *
     *
     * NOTE: this implementation is simple but slow.
     * AA^t = U*S*S*U^T (eig/evd solves U)
     * A^tA = V*S*S*V^T (eig/evd solves V)
     * S = U^T*A*V
     *
     * One must calculate two symmetric evd:s to
     * solve svd.
     *
     * - there are better methods.
     */
    template <typename T>
    inline bool svd(matrix<T>& A, matrix<T>& U, matrix<T>& V)
    {
      // std::cout << "SVD IN: " << A << std::endl;
      
      auto Ah = A;
      Ah.hermite();
      
      auto AAh = A*Ah;
      
      if(symmetric_eig(AAh,U,true) == false)
	return false;

      // std::cout << "AAh = " << AAh << std::endl;

      auto AhA = Ah*A;
      
      if(symmetric_eig(AhA,V,true) == false)
	return false;

      // std::cout << "AhA = " << AhA << std::endl;

      auto Uh = U;
      U.hermite();

      A = Uh*A*V; // singular values

      // std::cout << "S = " << A << std::endl;

      // FIXME: we must reorder A to be diagonal matrix again (+ EIG should have more numeric accuracy to be useful..)

#if 0
      {
	auto Vh = V;
	Vh.hermite();
	std::cout << "SVD OUT: " << (U*A*Vh) << std::endl;
      }
#endif

      return true;

#if 0
      // the code below cannot function because US and VS terms cannot be separated because S is non-square singular matrix..
      
      // A = [N1 x N2] matrix
      const unsigned int N1 = A.ysize();
      const unsigned int N2 = A.xsize();
      
      if(N1 >= N2){
	matrix<T> AA(A);
	AA.transpose();
	AA *= A; // AA = A^t * A
	
	if(symmetric_eig(AA,V) == false)
	  return false;
	
	U = A*V; // result = U S V^t * V = U S [N1 x N1] [N1 x N2] (N2 <= N1)
	A = AA; // A = S^2 (singular values)
	
	for(unsigned int j=0;j<N2;j++){
	  // S^2 -> S (S is diagonal)
	  A(j,j) = whiteice::math::sqrt(whiteice::math::abs(A(j,j)));

	  T scaling = T(1.0);
	  if(A(j,j) != T(0.0))
	    scaling = T(1.0) / A(j,j);
	  
	  for(unsigned int i=0;i<N1;i++) // multiplies U's column. 
	    U(i,j) *= scaling;           // (optimize: matrix::colscale(), matrix::rowscale() )
	}
	
	return true;
      }
      else{
	matrix<T> AA(A);
	AA.transpose();
	AA = A * AA; // AA = A * A^t = USV^t * VSU^t = U S^2 U^t
                     // optimize: this is slow (code templated atlas sped up: A = A*A')
	
	if(symmetric_eig(AA,U) == false)
	  return false;
	
	A.transpose(); // slow, code matrix::transposemulti, B = this^t * A
	V = A * U;     // result = VSU^T * U = VS [N2 x N2] [N2 x N1] (N1 < N2)
	A = AA; // A = S^2
	
	for(unsigned int j=0;j<N1;j++){
	  A(j,j) = whiteice::math::sqrt(whiteice::math::abs(A(j,j)));
	  
	  T scaling = T(1.0);
	  if(A(j,j) != T(0.0))
	    scaling = T(1.0) / A(j,j);

	  for(unsigned int i=0;i<N2;i++)
	    V(i,j) *= scaling;
	}
	
	return true;
      }
#endif 
    }
    
    
    /***********************************************************************************/
    
    // performs implicit francis double shift
    // H is overwritten and rotatation
    // is saved in Z
    template <typename T>
    void francis_qr_step(matrix<T>& H,
			 matrix<T>& Z,
			 const unsigned int i, const unsigned int j,
			 bool iscomplex)
			 
    {
      const unsigned int N = j-i;
      vertex<T> v,w;
      T s, t;
            
      Z.resize(N,N);
      Z.identity();
      
      // implicit double shift strategy
      s = H(i+N-2,i+N-2) + H(i+N-1,i+N-1);
      t = H(i+N-2,i+N-2) * H(i+N-1,i+N-1) - H(i+N-2,i+N-1)*H(i+N-1,i+N-2);	      
      
      if(N > 2){
	unsigned int k = 0;
	
	w.resize(3);
	w[0] = H(i,i) * H(i,i) + H(i,i+1)*H(i+1,i) - s*H(i,i) + t;
	w[1] = H(i+1,i) * ( H(i,i) + H(i+1,i+1) - s);
	w[2] = H(i+1,i) * H(i+2,i+1);
	
	unsigned int r=N;
	if(k+4 < r) r = k+4;
	v.resize(3);
	
	householder_vector(w,0,v,0, iscomplex);
	householder_leftrot(H,i+k+1,i+N,i+k+1,v,0);
	householder_rightrot(H,i,i+r,i+k+1,v,0);
	
	householder_leftrot(Z,k+1,N,k+1,v,0);
      }
      else{
	unsigned int k = 0;
	
	w.resize(2);
	w[0] = H(i,i)*H(i,i) + H(i,i+1)*H(i+1,i) - s*H(i,i) + t;
	w[1] = H(i+1,i)*( H(i,i) + H(i+1,i+1) - s);
	
	unsigned int r=N;
	if(k+4 < r) r = k+4;
	v.resize(2);
	
	householder_vector(w,0,v,0,iscomplex);
	householder_leftrot(H,i+k+1,i+N,i+k+1,v,0);
	householder_rightrot(H,i,i+r,i+k+1,v,0);
	
	householder_leftrot(Z,k+1,N,k+1,v,0);	
      }

      // after implicit francis qr step
      // reduce matrix back to heisenberg form
      
      w.resize(N);
      v.resize(N);
      v[0] = 0;
      
      for(unsigned int k=1;k<N-2;k++){
	// optimize: need only change smaller part
	// of the matrix (see book's implementation)
	
	v[k] = 0;
	
	// optimize: slow
	for(unsigned int l=k+1;l<N;l++)
	  w[l] = H(i+l,k);
	
	householder_vector(w,k+1,v,k+1,iscomplex);
	householder_leftrot(H,i+k,i+N,i+k+1,v,k+1);
	householder_rightrot(H,i+k,i+N,i+k+1,v,k+1);
	
	householder_rightrot(Z,0,N,0,v,0);
      }
      
    }
    
    
    /***********************************************************************************/
    
    // calculates schur form
    template <typename T>
    inline void schur(matrix<T>& A,
		      matrix<T>& X,
		      bool iscomplex)
    {
      const unsigned int N = A.size();      
      
      // TODO: stabilizing values to have
      // same order of magnitude
      
      hessenberg_reduction(A,X,iscomplex);
      
      if(N < 2) return;
      
      // saves rotation of francis qr step
      matrix<T> Z;
      vertex<T> w(N); // temporal vector
      
      // number rotations done in previous
      // iteration
      int pr = 1;
      
      // sd tells which subdiagonals are
      // still non-zero
      std::vector<bool> sd;
      sd.resize(N-1);
      
      for(unsigned int i=0;i<N-1;i++)
	sd[i] = true;
      
      
      // TODO: calculate good tolerance from
      // matrix data
      // this works with (most) floats
      T tolerance = 0.00001 * 0.00001;
      
      while(pr > 0){
	pr = 0;
	
	// checks if subdiagonals have
	// converged to zero
	for(unsigned int i=0;i<N-1;i++){
	  if(abs(A(i+1,i)) < tolerance){
	    A(i+1,i) = 0;
	    sd[i] = false;
	  }
	}
	
	// finds remaining >= 3x3 submatrices
	{
	  unsigned int i=0;
	  while(i<N-2){
	    
	    if(sd[i] && sd[i+1]){
	      
	      // calculates length of block
	      unsigned int sd_len = 1;
	      
	      for(unsigned int j=i+1;j<N-1;j++){
		if(sd[j]) sd_len++;
		else break;
	      }
	      
	      // processes block with francis qr step
	      francis_qr_step(A,Z,i,i+sd_len,iscomplex);
	      
	      // updates X and A
	      // (optimize: this is slow)
	      
	      // updates A
	      w.resize(sd_len);
	      if(i + sd_len < N){
		for(unsigned int k=i+sd_len;k<N;k++){
		  for(unsigned int l=0;l<sd_len;l++)
		    w[l] = A(l+i,k);
		  
		  for(unsigned int l=0;l<sd_len;l++)
		    A(l+i,k) = Z[l] * w;
		}
	      }
	      
	      // so Z can be accessed as vectors
	      Z.transpose();
	      
	      if(i > 0){
		for(unsigned int k=0;k<i-1;k++){
		  for(unsigned int l=0;l<sd_len;l++)
		    w[l] = A(k,l+i);
		
		  for(unsigned int l=0;l<sd_len;l++)
		    A(k,l+i) = w * Z[l];
		}
	      }
	      
	      // updates X
	      for(unsigned int k=0;k<N;k++){
		for(unsigned int l=0;l<sd_len;l++)
		  w[l] = X(k,l+i);
		
		for(unsigned int l=0;l<sd_len;l++)
		  X(k,l+i) = w * Z[l];
	      }
	      
	      pr++;
	      i += sd_len;
	    }
	    
	    i++;
	  } // while(i<N-2)
	
	}
      }
      
    }
    


    /////////////////////////////////////////////////////////////
    // missing "implementations" for unsupported types (SVD, symmetric EVD)
    
    template <> bool svd<int>(matrix<int>& A,
			      matrix<int>& U,
			      matrix<int>& V){ assert(0); return false; }
    
    template <> bool svd<char>(matrix<char>& A,
			       matrix<char>& U,
			       matrix<char>& V){ assert(0); return false; }

    template <> bool svd<unsigned int>(matrix<unsigned int>& A,
				       matrix<unsigned int>& U,
				       matrix<unsigned int>& V){ assert(0); return false; }
    
    template <> bool svd<unsigned char>(matrix<unsigned char>& A,
					matrix<unsigned char>& U,
					matrix<unsigned char>& V){ assert(0); return false; }
    
    // FIXME: THIS SHOULD BE IMPLEMENTED!
    template <> bool svd< blas_complex<float> >(matrix< blas_complex<float> >& A,
						matrix< blas_complex<float> >& U,
						matrix< blas_complex<float> >& V){
      assert(0);
      return false;
    }
    
    // FIXME: THIS SHOULD BE IMPLEMENTED!
    template <> bool svd< blas_complex<double> >(matrix< blas_complex<double> >& A,
						 matrix< blas_complex<double> >& U,
						 matrix< blas_complex<double> >& V){
      assert(0);
      return false;
    }
    
    template <> bool svd< whiteice::math::complex<float> >(matrix< whiteice::math::complex<float> >& A,
							   matrix< whiteice::math::complex<float> >& U,
							   matrix< whiteice::math::complex<float> >& V){
      assert(0);
      return false;
    }
    
    template <> bool svd< whiteice::math::complex<double> >(matrix< whiteice::math::complex<double> >& A,
							    matrix< whiteice::math::complex<double> >& U,
							    matrix< whiteice::math::complex<double> >& V)
    {
      assert(0);
      return false;
    }
    
    template <> bool symmetric_eig<char>
    (matrix<char>& A, matrix<char>& D, bool sort){ assert(0); return false; }
    template <> bool symmetric_eig<int>
    (matrix<int>& A, matrix<int>& D, bool sort){ assert(0); return false; }
    template <> bool symmetric_eig<unsigned char>(matrix<unsigned char>& A, matrix<unsigned char>& D, bool sort){ assert(0); return false; }
    template <> bool symmetric_eig<unsigned int>(matrix<unsigned int>& A, matrix<unsigned int>& D, bool sort){ assert(0); return false; }
    
    
  };
};








