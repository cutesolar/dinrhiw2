/*
 * see implementation notes from the "linear_equations.h"
 */

#include <iostream>
#include <algorithm>
#include <exception>
#include <stdexcept>

#include "linear_equations.h"
#include "matrix.h"
#include "vertex.h"
#include "dinrhiw_blas.h"


namespace whiteice
{
  namespace math
  {
    
    // explicit template instantations
    
    template bool linsolve< blas_real<float> >
    (matrix< blas_real<float> >& A, 
     vertex< blas_real<float> >& x, 
     const vertex< blas_real<float> >& b) ;
    
    template bool linsolve< blas_real<double> >
    (matrix< blas_real<double> >& A, 
     vertex< blas_real<double> >& x, 
     const vertex< blas_real<double> >& b) ;
    
    template bool linsolve<float>
    (matrix<float>& A, 
     vertex<float>& x, 
     const vertex<float>& b) ;
    
    template bool linsolve<double>
    (matrix<double>& A, 
     vertex<double>& x, 
     const vertex<double>& b) ;
    
    
    
    template bool linlsqsolve< blas_real<float> >
    (matrix< blas_real<float> >& A, 
     const vertex< blas_real<float> >& b, 
     vertex< blas_real<float> >& x) ;
    
    template bool linlsqsolve< blas_real<double> >
    (matrix< blas_real<double> >& A, 
     const vertex< blas_real<double> >& b, 
     vertex< blas_real<double> >& x) ;
    
    template bool linlsqsolve<float>
    (matrix<float>& A, 
     const vertex<float>& b, 
     vertex<float>& x) ;
    
    template bool linlsqsolve<double>
    (matrix<double>& A, 
     const vertex<double>& b, 
     vertex<double>& x) ;


    template bool linear_optimization(const std::vector< vertex< blas_real<float> > >& x,
				      const std::vector< vertex< blas_real<float> > >& y,
				      matrix< blas_real<float> >& A,
				      vertex< blas_real<float> >& b,
				      blas_real<float>& error);
    
    template bool linear_optimization(const std::vector< vertex< blas_real<double> > >& x,
				      const std::vector< vertex< blas_real<double> > >& y,
				      matrix< blas_real<double> >& A,
				      vertex< blas_real<double> >& b,
				      blas_real<double>& error);
    
    
    
    template bool cholesky_factorization< blas_real<float> >
    (matrix< blas_real<float> >& A) ;
    template bool cholesky_factorization< blas_real<double> >
    (matrix< blas_real<double> >& A) ;

    template bool cholesky_factorization< blas_complex<float> >
    (matrix< blas_complex<float> >& A) ;
    template bool cholesky_factorization< blas_complex<double> >
    (matrix< blas_complex<double> >& A) ;
    
    template bool cholesky_factorization<float>
    (matrix<float>& A) ;
    template bool cholesky_factorization<double>
    (matrix<double>& A) ;

    template bool cholesky_factorization< superresolution< blas_real<float>, modular<unsigned int> > >
    (matrix< superresolution< blas_real<float>, modular<unsigned int> > >& A) ;
    template bool cholesky_factorization< superresolution< blas_real<double>, modular<unsigned int> > >
    (matrix< superresolution< blas_real<double>, modular<unsigned int> > >& A) ;

    template bool cholesky_factorization< superresolution< blas_complex<float>, modular<unsigned int> > >
    (matrix< superresolution< blas_complex<float>, modular<unsigned int> > >& A) ;
    template bool cholesky_factorization< superresolution< blas_complex<double>, modular<unsigned int> > >
    (matrix< superresolution< blas_complex<double>, modular<unsigned int> > >& A) ;
    
    
    template bool solvegg< blas_real<float> >
    (matrix< blas_real<float> >& C, 
     vertex< blas_real<float> >& x) ;
    
    template bool solvegg< blas_real<double> >
    (matrix< blas_real<double> >& C, 
     vertex< blas_real<double> >& x) ;
    
    template bool solvegg<float>
    (matrix<float>& C, vertex<float>& x) ;
    template bool solvegg<double>
    (matrix<double>& C, vertex<double>& x) ;

    template bool solvegg< superresolution< blas_real<float>, modular<unsigned int> > >
    (matrix< superresolution< blas_real<float>, modular<unsigned int> > >& C,
     vertex< superresolution< blas_real<float>, modular<unsigned int> >  >& x) ;
    
    template bool solvegg< superresolution< blas_real<double>, modular<unsigned int> > >
    (matrix< superresolution< blas_real<double>, modular<unsigned int> > >& C,
     vertex< superresolution< blas_real<double>, modular<unsigned int> > >& x) ;


    
    template bool symmetric_inverse< blas_real<float> >(matrix< blas_real<float> >& A) ;
    template bool symmetric_inverse< blas_real<double> >(matrix< blas_real<double> >& A) ;

    template bool symmetric_inverse< blas_complex<float> >(matrix< blas_complex<float> >& A) ;
    template bool symmetric_inverse< blas_complex<double> >(matrix< blas_complex<double> >& A) ;
    
    template bool symmetric_inverse< float >(matrix< float >& A) ;
    template bool symmetric_inverse< double >(matrix< double >& A) ;

    template bool symmetric_inverse< superresolution< blas_real<float>, modular<unsigned int> > >
    (matrix< superresolution< blas_real<float>, modular<unsigned int> > >& A) ;
    template bool symmetric_inverse< superresolution< blas_real<double>, modular<unsigned int> > >
    (matrix< superresolution< blas_real<double>, modular<unsigned int> > >& A) ;

    template bool symmetric_inverse< superresolution< blas_complex<float>, modular<unsigned int> > >
    (matrix< superresolution< blas_complex<float>, modular<unsigned int> > >& A) ;
    template bool symmetric_inverse< superresolution< blas_complex<double>, modular<unsigned int> > >
    (matrix< superresolution< blas_complex<double>, modular<unsigned int> > >& A) ;
    
    
    template void solve_sylvester< blas_real<float> >
    (const matrix< blas_real<float> >& A,
     const matrix< blas_real<float> >& B,
     matrix< blas_real<float> >& C,
     const unsigned int i, const unsigned int j,
     const unsigned int a, const unsigned int k,
     const unsigned int l, const unsigned int b);
    
    template void solve_sylvester< blas_real<double> >
    (const matrix< blas_real<double> >& A,
     const matrix< blas_real<double> >& B,
     matrix< blas_real<double> >& C,
     const unsigned int i, const unsigned int j,
     const unsigned int a, const unsigned int k,
     const unsigned int l, const unsigned int b);
    
    template void solve_sylvester< blas_complex<float> >
    (const matrix< blas_complex<float> >& A,
     const matrix< blas_complex<float> >& B,
     matrix< blas_complex<float> >& C,
     const unsigned int i, const unsigned int j,
     const unsigned int a, const unsigned int k,
     const unsigned int l, const unsigned int b);
    
    template void solve_sylvester< blas_complex<double> >
    (const matrix< blas_complex<double> >& A,
     const matrix< blas_complex<double> >& B,
     matrix< blas_complex<double> >& C,
     const unsigned int i, const unsigned int j,
     const unsigned int a, const unsigned int k,
     const unsigned int l, const unsigned int b);
    
    template void solve_sylvester<float>
    (const matrix<float>& A,
     const matrix<float>& B,
     matrix<float>& C,
     const unsigned int i, const unsigned int j,
     const unsigned int a, const unsigned int k,
     const unsigned int l, const unsigned int b);
    
    template void solve_sylvester<double>
    (const matrix<double>& A,
     const matrix<double>& B,
     matrix<double>& C,
     const unsigned int i, const unsigned int j,
     const unsigned int a, const unsigned int k,
     const unsigned int l, const unsigned int b);
    
    
    
    /*
     * solves linear Ax = b problem
     * by gaussian elimination with partial pivoting
     * returns false if A is (numerically) singular
     * A is destroyed in a process.
     * (caller may need to make local copies for the call)
     */
    template <typename T>
      bool linsolve(matrix<T>& A, vertex<T>& x, const vertex<T>& b) 
      {
	// initial conditions and initializations
	if(A.xsize() != A.ysize()) return false; // not a square (TODO: calculate pseudoinverse with svd then)
	if(b.size() != A.xsize()) return false;  // b not large enough	
	x = b;
		
	std::vector<unsigned int> permutation;
	permutation.resize(A.ysize()); // permutation of rows
	// permutation p: p(i) = j <-> permutation[i] = j
	
	for(unsigned int i=0;i<permutation.size();i++)
	  permutation[i] = i; // identity permutation		
	
	const unsigned int N = A.ysize();
	
	// calculates upper triangular A 
	// (doesn't really really/numerically
	//  bother to zero lower trianglular part)
	for(unsigned int i=0;i<N;i++) // i:th row
	{
	  // finds biggest value from the column
	  unsigned int index = i;
	  for(unsigned int j=index;j<N;j++)
	    if(A(i,j) > A(index,j)) index = i;
	  
	  if(index != i){ // changes rows
	    // swaps rows
	    // (optimization: should only copy A[i][i..N-1] !)
	    
	    // std::swap< vertex<T> >(A[i], A[index]);
	    // slow swap..
	    for(unsigned int k=0;k<A.xsize();k++){
	      T temp = A(index,k);
	      A(i,k) = temp;
	    }
	    
	    
	    std::swap< T > (x[i], x[index]);
	    
	    // updates permutation
	    std::swap< unsigned int >(permutation[i], permutation[index]);
	  }

	  if(A(i,i) == 0) // TODO: or close to zero! (calculate good tolerance)
	    return false;  // singular matrix
	    
	  // divides i:th row/equation
	  x[i] /= A(i,i);

	  // divides a_ii to be one
	  // (reverse order to set a_ii = 1 last)
	  for(unsigned int j=N;j>i;j--){
	    A(i,j-1) /= A(i,i);
	  }
	  
	  // 'zeroes' bottom rows
	  
	  for(unsigned int j=i+1;j<N;j++){ // j:th row
	    // only alters absolutely needed parts of A's
	    // lower triangular values
	    // -> lower triangular values = (more or less) garbage
	    // (not used anymore)
	    
	    x[j] -= A(j,i)*x[i]; // updates result vector
	    
	    // updates row
	    for(unsigned int k=i+1;k<N;k++) // here: skips actual zeroing of A[j][i]
	      A(j,k) -= A(j,i)*A(i,k);
	  }
	  
	}
	
	// A is upper triangular, solves it iteratively
	// only 'virtually' zeroes upper triangular values
	// as a linear sum of already solved variables
	// (doesn't solve values/variables separately
	//  -> faster + more numerical accuracy)
	
	T sum;
	
	if(N >= 2){
	  for(unsigned int i=N;i>0;i--){ // i:th row
	    
	    // calculates sum of already solved values
	    // for this row and subtracts it
	    // from the other side -> left side variables
	    // are 'virtually' zeroed
	    sum = 0.0f;
	    for(unsigned int j=i;j<N;j++)
	      sum += A(i - 1,j)*x[j];
	    
	    x[i-1] -= sum;
	    x[i-1] /= A(i-1,i-1); // not necessary(?)
	  }
	}
	
	return true;
      }
   
    
    /**************************************************/
    
    /* least squares solution */
    template <typename T>
      bool linlsqsolve(matrix<T>& A, const vertex<T>& b, vertex<T>& x) 
      {
	// A = X^(m x n) , m >= n

	if(!(A.ysize() >= A.xsize()))
	  return false;
	
	// calculates first C = A*A^t (note: should only calculate lower/upper triangular part)
	
	// 1. C = A^t * A (slow implementation)
	matrix<T> At = A;
	At.transpose();	
	matrix<T> C = At * A; // needs only really calculate lower triangular part
	
	// 2. d = A^t * b (reasonable fast implementation possible)
	x = At * b;
	
	// 3. cholesky factorization (quite fast, 
	//    use gaxpy + atlas + partial saving of data (lower part only) for fast impl.)
	if(!cholesky_factorization(C)) return false;
	
	// 4. solve Gy = d  (G is lower triangular)
	// 5. solve G^t*x = y (G^t is upper triangular)
	
	if(!solvegg(C, x)) // solves G*G^t*z = x, saves z in x
	  return false;
	
	return true;
      }


    /* 
     * solves linear optimization problem, min(A,b) E_xy{0.5*(y-Ax-b)^2}
     * regularizes matrixes if they are singular to get solution.
     */
    template <typename T>
    bool linear_optimization(const std::vector< vertex<T> >& x,
			     const std::vector< vertex<T> >& y,
			     matrix<T>& A, vertex<T>& b, T& error)
    {
      if(x.size() != y.size()) return false;
      if(x.size() == 0 || y.size() == 0) return false;

      matrix<T> Cxx, Cxy;
      vertex<T> mx, my;

      Cxx.resize(x[0].size(), x[0].size());
      Cxy.resize(x[0].size(), y[0].size());
      mx.resize(x[0].size());
      my.resize(y[0].size());

      Cxx.zero();
      Cxy.zero();
      mx.zero();
      my.zero();

      bool not_ok = false;
      
#pragma omp parallel
      {
	matrix<T> Cxx_, Cxy_;
	vertex<T> mx_, my_;
	
	Cxx_.resize(x[0].size(), x[0].size());
	Cxy_.resize(x[0].size(), y[0].size());
	mx_.resize(x[0].size());
	my_.resize(y[0].size());
	
	Cxx_.zero();
	Cxy_.zero();
	mx_.zero();
	my_.zero();


#pragma omp for
	for(unsigned int i=0;i<x.size();i++){

	  if(not_ok) continue;
	  
	  // Cxx += x[i].outerproduct();
	  if(addouterproduct(Cxx_, T(1.0), x[i], x[i]) == false){
	    not_ok = true;
	  }
	  
	  // Cxy += x[i].outerproduct(y[i]);
	  if(addouterproduct(Cxy_, T(1.0), x[i], y[i]) == false){
	    not_ok = true;
	  }

	  if((i % 1000) == 0){
	    std::cout << i << std::endl;
	  }
	  
	  mx_ += x[i];
	  my_ += y[i];
	}

#pragma omp critical
	{
	  Cxx += Cxx_;
	  Cxy += Cxy_;

	  mx += mx_;
	  my += my_;
	}
	
      }

      if(not_ok) return false;

      Cxx /= T(x.size());
      Cxy /= T(y.size());
      mx /= T(x.size());
      my /= T(y.size());

      Cxx -= mx.outerproduct();
      Cxy -= mx.outerproduct(my);

      // matrix inverse

      matrix<T> INV;
      T l = T(0.0);

      do{
	std::cout << "calculating matrix-inverse: " << l << std::endl;
	INV = Cxx;

	T trace = T(0.0f);

	for(unsigned int i=0;i<Cxx.xsize();i++){
	  trace += Cxx(i,i);
	  INV(i,i) += l;  
	}

	trace /= Cxx.xsize();

	l += T(1e-20)*trace + T(2.0)*l; 
      }
      while(INV.symmetric_pseudoinverse() == false);

      A = (Cxy.transpose() * INV);
      b = (my - A*mx);

      // calculates error

      T err, e;
      err.zero();

      for(unsigned int i=0;i<x.size();i++){
	auto delta = A*x[i] + b - y[i];

	e = T(0.0);

	for(unsigned int d=0;d<delta.size();d++){
	  e += whiteice::math::abs(delta[d][0]);
	}
	
	e /= T(delta.size());
	

	// e = delta.norm()/delta.size();
	
	/*
	e.zero();

	for(unsigned int d=0;d<delta.size();d++)
	  e += delta[d][0].abs();

	e /= T(delta.size());
	*/

	err += e;
      }

      err /= T(x.size());

      error = err;

      return true;
    }

    
    
    // calculates cholesky factorization of symmetric
    // positive definite matrix, A = G*G^t (G is lower triangular)
    // implementation only uses lower triangular part of A and calculates
    // result to lower triangular part of A (overwrites input data).
    template <typename T>
      bool cholesky_factorization(matrix<T>& A) 
      {
	if(A.xsize() != A.ysize()) return false;
	
	const unsigned int N = A.xsize();
	T scale;
	
	// processes first column
	
	if(A(0,0) == T(0.0))
	  return false;

	
	scale = T(1.0) / sqrt(A(0,0));
	for(unsigned int k=0;k<N;k++)
	  A(k,0) *= scale;
	
	for(unsigned int i=1;i<N;i++){
	  for(unsigned int j=0;j<=(i - 1);j++){
	    for(unsigned int k=j;k<N;k++){
	      A(k,i) -= A(i,j)*A(k,j);
	    }
	  }
	  
	  if(A(i,i) == T(0.0))
	    return false;
	  
	  // normalizes then length of A[j..N][j]
	  scale = T(1.0) / sqrt(A(i,i));
	  
	  for(unsigned int k=i;k<N;k++) // was k=j -> k=i (correct?)
	    A(k,i) *= scale;
	}
	
	return true;
      }
    
    
    template <typename T>
      bool solvegg(matrix<T>& C, vertex<T>& x) 
      {
	// C's lower triangular part = G;
	
	// 4. solves Gy = d  (G is lower triangular)

	const unsigned int N = C.ysize();
	T sum;
	
	for(unsigned int i=0;i<N;i++){ // i:th row
	    
	  // calculates sum of already solved values
	  // for this row and subtracts it
	  // from the other side -> left side variables
	  // are 'virtually' zeroed
	  sum = T(0.0f);
	  for(unsigned int j=0;j<i;j++)
	    sum += C(i,j)*x[j];
	  
	  x[i] -= sum;
	  
	  if(C(i,i) <= T(0.0f)) return false;
	  else x[i] /= C(i,i);
	}
	
	
	// 5. solves G^t*x = y (G^t is upper triangular)
    
	for(unsigned int i=N;i>0;i--){ // i:th row
	  
	  // calculates sum of already solved values
	  // for this row and subtracts it
	  // from the other side -> left side variables
	  // are 'virtually' zeroed
	  sum = T(0.0f);
	  for(unsigned int j=i;j<N;j++)
	    // C[i][j].transpose() * x[j] (because need to get values from lower triangular part)
	    sum += C(j,i - 1)*x[j];
	  
	  x[i - 1] -= sum;
	  
	  if(C(i - 1, i - 1) <= T(0.0f)) return false;
	  else x[i - 1] /= C(i - 1, i - 1);
	}	
	
	return true;
      }
    
    
    /**************************************************/
    
    
    template <typename T>
    bool symmetric_inverse(matrix<T>& A) 
    {
      if(A.ysize() != A.xsize()) return false;  // only square matrixes
      
      if(cholesky_factorization(A) == false) // Cxx = L*L^t
	return false;
      
      math::matrix<T>& L = A;

      // zeroes upper triangular part of L (for testing purposes)
      for(unsigned int j=0;j<L.ysize();j++)
	for(unsigned int i=j+1;i<L.xsize();i++)
	  L(j,i) = T(0.0f);
      
      // inverse is (L^-1)^t * (L^-1)
      // so we only need to calculate inverse of L
      math::matrix<T>  INV;
      INV.resize(A.ysize(), A.xsize());
      INV.identity();
      
      for(unsigned int j=0;j<L.ysize();j++){
	auto t = L(j,j);
	
	for(unsigned int i=0;i<=j;i++){
	  // L(j,i) /= t;
	  INV(j, i) /= t;
	}

	for(unsigned int i=j+1;i<L.ysize();i++){
	  auto k = L(i,j);
	  
	  for(unsigned int r=0;(r<=j);r++){
	    // L(i, r)   -= k * L(j, r);
	    INV(i, r) -= k * INV(j, r);
	  }
	}
      }
      
      auto& LLINV = A;
      LLINV.zero();
      
      // we compute: (L^-1)^t * (L^-1) = UPPER_TRIANGULAR * LOWER_TRIANGULAR
      for(unsigned int j=0;j<LLINV.ysize();j++){
	for(unsigned int i=0;i<LLINV.xsize();i++){
	  unsigned int min_ij = i;
	  if(min_ij < j) min_ij = j;
	  
	  for(unsigned int k=min_ij;k<LLINV.ysize();k++){
	    LLINV(j,i) += INV(k,j)*INV(k,i);
	  }
	}
      }
      
      return true;
    }
    
    
    // FIXME: SYLVESTER EQUATION SOLVER IS CURRENT BROKEN
    // solves sylvester eq and saves result in C
    // A(i:i+a:j:j+a)*X - X*B(k:k+b,l:l+b) = C
    template <typename T>
    void solve_sylvester(const matrix<T>& A,
			 const matrix<T>& B,
			 matrix<T>& C,
			 const unsigned int i,
			 const unsigned int j,
			 const unsigned int a,
			 const unsigned int k,
			 const unsigned int l,
			 const unsigned int b)
    {
      // first iteration, p = 0
      unsigned int p = 0;
      
      if(B(k+p+1,l+p) != T(0)){ // must solve two vectors at once (2x2 block)
	vertex<T> h(2*a);
	
	for(unsigned int m=0;m<a;m++) h[m] = C(m, p);
	for(unsigned int m=a;m<2*a;m++) h[m] = C(m-a, p+1);
	
	// slow: shouldn't generate separated vectors and matrix for
	// linear equations, should write linear equation solver for
	// given data structures instead
	matrix<T> AA(2*a,2*a); // initial matrix is zero matrix (right?)
	
	for(unsigned int m=0;m<a;m++){
	  for(unsigned int n=0;n<a;n++){
	    AA(m,n)     = A(i+m,j+n);
	    AA(a+m,a+n) = A(i+m,j+n);
	  }
	  
	  AA(m,m)     -= B(k+p,l+p);
	  AA(m,a+m)   -= B(k+p+1,l+p);
	  AA(a+m,m)   -= B(k+p+1,l+p);
	  AA(a+m,a+m) -= B(k+p+1,l+p+1);
	}
	
	// slow (optimize: start to use linear equations solvers:
	// AAx = h when they are known to work correctly)
	AA.inv();
	h = AA*h;
	
	for(unsigned int m=0;m<a;m++)   C(m,p)   = h[m];
	for(unsigned int m=a;m<2*a;m++) C(m-a,p+1) = h[m];
	
	p += 2;
      }
      else{ // need to solve only single vector (1x1 block)
	
	// solves (AA - b(k,k)*I)*b = z
	vertex<T> z(a);
	matrix<T> AA(A);
	for(unsigned int m=0;m<a;m++){
	  AA(m,m) -= B(p,p);
	  z[m] = C(m,p);
	}
	
	AA.inv();
	z = AA*z;
	
	for(unsigned int m=0;m<a;m++){
	  C(m,p) = z[m];
	}
	
	p++;
      }
      
      
      // inner parts of the solution matrix
      while(p < b-1){
	
	
	if(B(k+p+1,l+p) != T(0)){ // must solve two vectors at once (2x2 block)
	  vertex<T> h(2*a);
	  
	  for(unsigned int m=0;m<a;m++){
	    h[m] = C(m,p);
	    
	    for(unsigned int n=0;n<p;n++)
	      h[m] += C(m,n)*B(k+n,l+p);
	  }
	  
	  for(unsigned int m=a;m<2*a;m++){
	    h[m] = C(m-a,p+1);
	    
	    for(unsigned int n=0;n<p;n++)
	      h[m] += C(m,n)*B(k+n,l+p+1);
	  }
	  
	  // slow: shouldn't generate separated vectors and matrix for
	  // linear equations, should write linear equation solver for
	  // given data structures instead
	  matrix<T> AA(2*a,2*a); // initial matrix is zero matrix (right?)
	  
	  for(unsigned int m=0;m<a;m++){
	    for(unsigned int n=0;n<a;n++){
	      AA(m,n)     = A(i+m,j+n);
	      AA(a+m,a+n) = A(i+m,j+n);
	    }
	    
	    AA(m,m)     -= B(k+p,l+p);
	    AA(m,a+m)   -= B(k+p+1,l+p);
	    AA(a+m,m)   -= B(k+p+1,l+p);
	    AA(a+m,a+m) -= B(k+p+1,l+p+1);
	  }
	
	  // slow (optimize: start to use linear equations solvers:
	  // AAx = h when they are known to work correctly)
	  AA.inv();
	  h = AA*h;
	  
	  for(unsigned int m=0;m<a;m++)   C(m,p)   = h[m];
	  for(unsigned int m=a;m<2*a;m++) C(m-a,p+1) = h[m];
	  
	  p += 2;
	}
	else{ // need to solve only single vector (1x1 block)
	  
	  for(unsigned int m=0;m<a;m++){
	    for(unsigned int n=0;n<p;n++){
	      C(m,p) += C(m,n)*B(k+n,l+p);
	    }
	  }
	  
	  
	  // solves (AA - b[k][k]*I)*b = z
	  vertex<T> z(a);
	  matrix<T> AA(A);
	  for(unsigned int m=0;m<a;m++){
	    AA(m,m) -= B(p,p);
	    z[m] = C(m,p);
	  }
	  
	  AA.inv();
	  z = AA*z;
	  
	  for(unsigned int m=0;m<a;m++){
	    C(m,p) = z[m];
	  }
	  
	  p++;
	}
	
      }
      
      // last iteration (if needed)
      
      if(p < b){
	
	for(unsigned int m=0;m<a;m++){
	  for(unsigned int n=0;n<p;n++){
	    C(m,p) += C(m,n)*B(k+n,l+p);
	  }
	}
	  
	
	// solves (AA - b[k][k]*I)*b = z
	vertex<T> z(a);
	matrix<T> AA(A);
	for(unsigned int m=0;m<a;m++){
	  AA(m,m) -= B(p,p);
	  z[m] = C(m,p);
	}
	
	AA.inv();
	z = AA*z;
	
	for(unsigned int m=0;m<a;m++){
	  C(m,p) = z[m];
	}
	
	p++;
	
      }
      
    }
    
    
    
  }
}




















