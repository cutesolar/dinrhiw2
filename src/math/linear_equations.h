/*
 * efficient (and algoritmically fast) & 
 * numerically reasonably accurate 
 * (householder based QR factorization would be better
 *  but slower and/or complete pivoting)
 *
 * Ax = b and least squares min ||Ax - b||^2 solvers
 * 
 * ref. Matrix Computations (Golub G.H , Van Loan C. F.)
 *
 * solves Ax = b with square matrix 
 * with modified gaussian (by choosing pivots smartly)
 * (for numerical stability) another possibility would
 * be householder's factorization but it's slower
 * (afaik because Q and R must be explicitely created
 *  (thought this adds only (big) constant into equation))
 *
 * solves overdetermined ||Ax - b|| problem with
 * 'method of normal equations':
 * C = A^T*A
 * d = A^t * b
 * calculates cholesky factorization of C = GG^t
 *
 * solves Gy = d and G^t * x = y  problems
 *
 *
 * note: cholesky factorization assumes that
 *       C is strictly positive definite and symmetric.
 *       In theory C is always positive definite
 *       however, with limited precision C = A^t*A
 *       may not end up being pos. def. from numerical
 *       perspective... this can cause problems
 * 
 */

#ifndef linear_equations_h
#define linear_equations_h

#include "vertex.h"
#include "matrix.h"
#include "dinrhiw_blas.h"


namespace whiteice
{
  namespace math
  {
    /* solves linear Ax = b problem
     * by gaussian elimination with partial pivoting
     * returns false if A is (numerically) singular
     * A is destroyed in a process.
     * (caller may need to make local copies for the call)
     */
    template <typename T>
      bool linsolve(matrix<T>& A, vertex<T>& x, const vertex<T>& b) ;

    /* gives least squares solution to a problem A*x = b,
     * where A is not square matrix
     */
    template <typename T>
      bool linlsqsolve(matrix<T>& A, const vertex<T>& b, vertex<T>& x) ;
   
    
    /* calculates cholesky factorization of symmetric
     * positive definite matrix, A = G*G^t (G is lower triangular)
     * implementation only uses lower triangular part of A and calculates
     * result to lower triangular part of A (overwrites input data).
     */
    template <typename T>
      bool cholesky_factorization(matrix<T>& A) ;
    
    /* needed by cholesky factorization */
    template <typename T>
      bool solvegg(matrix<T>& C, vertex<T>& x) ;
    
    /*
     * solves symmetric matrix inverse problem using cholesky factorization
     */
    template <typename T>
      bool symmetric_inverse(matrix<T>& A) ;
    
    
    
    // FIXME: SYLVESTER EQUATION SOLVER IS CURRENTLY BROKEN
    // solves sylvester eq and saves result in C
    // A(i:i+a:j:j+a)*X - X*B(k:k+b,l:l+b) = C
    template <typename T>
      void solve_sylvester(const matrix<T>& A,
			   const matrix<T>& B,
			   matrix<T>& C,
			   const unsigned int i, const unsigned int j, const unsigned int a,
			   const unsigned int k, const unsigned int l, const unsigned int b);
    
    
    
    extern template bool linsolve< blas_real<float> >
      (matrix< blas_real<float> >& A, vertex< blas_real<float> >& x, const vertex< blas_real<float> >& b) ;
    extern template bool linsolve< blas_real<double> >
      (matrix< blas_real<double> >& A, vertex< blas_real<double> >& x, const vertex< blas_real<double> >& b) ;
    extern template bool linsolve<float>
      (matrix<float>& A, vertex<float>& x, const vertex<float>& b) ;
    extern template bool linsolve<double>
      (matrix<double>& A, vertex<double>& x, const vertex<double>& b) ;
    
    
    extern template bool linlsqsolve< blas_real<float> >
      (matrix< blas_real<float> >& A, const vertex< blas_real<float> >& b, vertex< blas_real<float> >& x) ;
    extern template bool linlsqsolve< blas_real<double> >
      (matrix< blas_real<double> >& A, const vertex< blas_real<double> >& b, vertex< blas_real<double> >& x) ;
    extern template bool linlsqsolve<float>
      (matrix<float>& A, const vertex<float>& b, vertex<float>& x) ;
    extern template bool linlsqsolve<double>
      (matrix<double>& A, const vertex<double>& b, vertex<double>& x) ;
    
    
    extern template bool cholesky_factorization< blas_real<float> >
      (matrix< blas_real<float> >& A) ;
    extern template bool cholesky_factorization< blas_real<double> >
      (matrix< blas_real<double> >& A) ;
    extern template bool cholesky_factorization<float>
      (matrix<float>& A) ;
    extern template bool cholesky_factorization<double>
      (matrix<double>& A) ;
    
    
    extern template bool solvegg< blas_real<float> >
      (matrix< blas_real<float> >& C, vertex< blas_real<float> >& x) ;
    extern template bool solvegg< blas_real<double> >
      (matrix< blas_real<double> >& C, vertex< blas_real<double> >& x) ;
    extern template bool solvegg<float>
      (matrix<float>& C, vertex<float>& x) ;
    extern template bool solvegg<double>
      (matrix<double>& C, vertex<double>& x) ;
    
    extern template bool symmetric_inverse< blas_real<float> >(matrix< blas_real<float> >& A) ;
    extern template bool symmetric_inverse< blas_real<double> >(matrix< blas_real<double> >& A) ;
    extern template bool symmetric_inverse< float >(matrix< float >& A) ;
    extern template bool symmetric_inverse< double >(matrix< double >& A) ;
    
    
    extern template void solve_sylvester< blas_real<float> >
      (const matrix< blas_real<float> >& A,
       const matrix< blas_real<float> >& B,
       matrix< blas_real<float> >& C,
       const unsigned int i, const unsigned int j,
       const unsigned int a, const unsigned int k,
       const unsigned int l, const unsigned int b);
    extern template void solve_sylvester< blas_real<double> >
      (const matrix< blas_real<double> >& A,
       const matrix< blas_real<double> >& B,
       matrix< blas_real<double> >& C,
       const unsigned int i, const unsigned int j,
       const unsigned int a, const unsigned int k,
       const unsigned int l, const unsigned int b);
    extern template void solve_sylvester< blas_complex<float> >
      (const matrix< blas_complex<float> >& A,
       const matrix< blas_complex<float> >& B,
       matrix< blas_complex<float> >& C,
       const unsigned int i, const unsigned int j,
       const unsigned int a, const unsigned int k,
       const unsigned int l, const unsigned int b);
    extern template void solve_sylvester< blas_complex<double> >
      (const matrix< blas_complex<double> >& A,
       const matrix< blas_complex<double> >& B,
       matrix< blas_complex<double> >& C,
       const unsigned int i, const unsigned int j,
       const unsigned int a, const unsigned int k,
       const unsigned int l, const unsigned int b);
    extern template void solve_sylvester<float>
      (const matrix<float>& A,
       const matrix<float>& B,
       matrix<float>& C,
       const unsigned int i, const unsigned int j,
       const unsigned int a, const unsigned int k,
       const unsigned int l, const unsigned int b);
    extern template void solve_sylvester<double>
      (const matrix<double>& A,
       const matrix<double>& B,
       matrix<double>& C,
       const unsigned int i, const unsigned int j,
       const unsigned int a, const unsigned int k,
       const unsigned int l, const unsigned int b);
       
    
  }
}



#endif


















