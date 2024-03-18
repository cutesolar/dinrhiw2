
//#define _ISOC11_SOURCE
//#define _GNU_SOURCE

#include <cstdlib>
#include <stdlib.h>

#include "pocketfft_hdronly.h" // FFT algorithm [buggy with threaded code, DISABLED!!!]
//#include "pocketfft/pocketfft.h"


#include "blade_math.h"
#include "real.h"

#include <cmath>
#include <math.h>
#include <gmp.h>

using namespace pocketfft; // pocketfft_hdronly.h



namespace whiteice
{  
  namespace math
  {
    
    realnumber exp(const realnumber& x)
    {
      /*
       * Taylor series based exponentation implementation
       * with lots of tricks to calculate exp(x) where x
       * is close to zero.
       * 
       * Original MATLAB implementations:
       * m2_exp.m m1_exp.m
       * Theory: AlgoChapter4.html && AlgoChapter5.html (not by me)
       * ohext
       */
      
      signed long int E;
      mpf_get_d_2exp(&E, x.data); // finds E: x = d*2^E
      
      bool negative = false;
      if(E < 0){
	negative = true;
	E = -E;
      }
      
      const unsigned long int P = mpf_get_prec(x.data);
      
      mpf_t d, t, t2, t3, t4, t5;      
      mpf_init2(d, P);
      mpf_init2(t, P);
      mpf_init2(t2, P);
      mpf_init2(t3, P);
      mpf_init2(t4, P);
      mpf_init2(t5, P);
      
      // d = x*2^-E
      
      if(negative == false)
	mpf_div_2exp(t2, x.data, (unsigned)E);
      else
	mpf_set(t2, x.data);
      
      ///////////////////////////////////////////////////////////
      // exponentation for values near zero
      // d = exp(d); (taylor series with extra tricks)
      // 
      
      {
	mpf_div_ui(d, t2, 2); // d = d/2  (0.25, 0.5)
	
	unsigned long int iters = 0;
	
	// calculates number of iterations required
	// for a working accuracy
	if(mpz_sgn(d)){
	  
	  signed long int N;
	  mpf_get_d_2exp(&N, d);
	  N = -N;
	  if(!N)
	    N = 1;
	    
	  iters = P/N;
	  iters += 3;
	}
	
	
	mpf_set_ui(t, 0);  // t = exp(d) approx.
	mpf_set_ui(t2, 1); // t2 = 1/(k!)
	mpf_set_ui(t3, 1); // t3 = d^k
	
	
	iters /= 2; // (unrolled for efficiency: save num vars used)
	
	for(unsigned long int i=1;i<=iters;i++){
	  mpf_div_ui(t4, t2, 2*i - 1); // b = b/i  (so that b(x) = 1/x!)
	  mpf_mul(t2, t3, d);          // z = z*x
	  mpf_mul(t3, t4, t2);         // b*z
	  mpf_add(t5, t, t3);          // y = y + b*z
	  
	  mpf_mul(t3, t2, d);          // z = z*x
	  mpf_div_ui(t2, t4, 2*i);     // b = b/i
	  mpf_mul(t4, t2, t3);         // b*z
	  mpf_add(t,  t5, t4);         // y = y + b*z
	  
	}
	
	// exp(d) = 1 + 2*t + t*t = (t+1)*(t+1)
	mpf_add_ui(t2, t, 1);
	mpf_mul(d, t2, t2);  // both input arguments can be same(?)
      }
      
      
      ///////////////////////////////////////////////////////////
      
      // calculates d^(2^E)
      
      if(negative){ // E < 0:
	// we didn't divide x with 2^E because
	// x was already small enough (x <= 0.5)
	//
	// -> we can directly use the result
	
	mpf_clear(t);
	mpf_clear(t2);
	mpf_clear(t3);
	mpf_clear(t4);
	mpf_clear(t5);

	realnumber res(d);

	mpf_clear(d);
	
	return res;
      }
      else{

	if(E != 0){
	  mpz_t temp, q, r;
	  
	  // note: this is very slow for large exponents
	  // is there smart way to get away from creating so large
	  // exponents?

	  mpz_init2(temp, E+1); // 2^E requires E+1 bits
	  mpz_init2(q, E+1);    // 2^E requires E+1 bits
	  mpz_init2(r, E+1);    // 2^E requires E+1 bits
	  
	  
	  // temp = 2^|E|
	  mpz_ui_pow_ui(temp, 2, (unsigned)E);
	  
	  // implementation supports exponents > MAX_UNSIGNED_LONT_INT
	  
	  // temp = q*MAX_UNSIGNED_LONG_INT + r
	  mpz_tdiv_qr_ui(q, r, temp, (unsigned long int)(-1L));
	  // mpz_clear(temp);
	  
	  // d^temp = (d^(MAX_UNSIGNED_LONG_INT))^q * d^r
	  
	  mpf_set(t2, d);
	  mpf_pow_ui(t, d, (unsigned long int)(-1L));
	  mpf_pow_ui(d, t, mpz_get_ui(q));
	  mpf_pow_ui(t, t2, mpz_get_ui(r));
	  
	  mpf_mul(t2, t, d);
	  
	  mpz_clear(q);
	  mpz_clear(r);
	  mpz_clear(temp);
	  
	  mpf_clear(t);
	  mpf_clear(t3);
	  mpf_clear(t4);
	  mpf_clear(t5);
	  mpf_clear(d);

	  realnumber res(t2);

	  mpf_clear(t2);
	  
	  return res;
	}
	else{
	  mpf_clear(t);
	  mpf_clear(t2);
	  mpf_clear(t3);
	  mpf_clear(t4);
	  mpf_clear(t5);

	  realnumber res(d);

	  mpf_clear(d);
	  
	  return res;
	}
      }
      
    }
    
    
    //////////////////////////////////////////////////////////////////////
    
    
    // logarithm function for realnumber
    inline void __primi_log(mpf_t& x);
    
    realnumber log(const realnumber& x){
      /*
       * Taylor series (at x = 1) based natural logarithm with 
       * lots of tricks to calculate log(x) only when it is close to 1.
       * 
       * Implements matlab code: m1_log.m and m2_log2.m
       */
      
      signed long int E;
      mpf_get_d_2exp(&E, x.data); // finds E: x = d*2^E
      
      mpf_t d, t, t2;
      unsigned long int P = mpf_get_prec(x.data);
      
      mpf_init2(d,  P);
      mpf_init2(t,  P);
      mpf_init2(t2, P);
      
      // d = x*2^-E
      if(E >= 0)
	mpf_div_2exp(d, x.data, (unsigned long int)E);
      else{
	mpf_mul_2exp(d, x.data, (unsigned long int)(-E));
      }
      
      // y = log(d) + ex*log(2)
      
      mpf_set_ui(t, 2);
      
      __primi_log(d);
      __primi_log(t); // can be cached
      
      // we can only multiple with unsigned integers
      if(E >= 0){
	mpf_mul_ui(t2, t, (unsigned long int)E);
	mpf_add(t, d, t2);
      }
      else{
	mpf_mul_ui(t2, t, (unsigned long int)(-E));
	mpf_sub(t, d, t2);
      }
      
      
      mpf_clear(d);
      mpf_clear(t2);

      realnumber res(t);

      mpf_clear(t);
      
      return res;
    }
    
    
    // logarithm which calculates logarithm fastest
    // for values close to 1 but works with any x
    // although calculations can be considerable slower
    inline void __primi_log(mpf_t& x){
      
      bool negate = false;
      unsigned long int iters = 0;
      unsigned long int P = mpf_get_prec(x);
      
      mpf_t y ,t, t2, t3;
      mpf_init2(t, P);
      
      if(mpf_cmp_ui(x, 1UL) > 0){
	mpf_ui_div(t, 1, x); // x = 1/x
	negate = true;
	
	mpf_ui_sub(x, 1, t); // x = 1 - x
      }
      else{
	mpf_ui_sub(t, 1, x); // x = 1 - x
	mpf_swap(t, x);
      }
      
      // x < 1
      
      
      if(mpf_sgn(x)){
	signed long int N;
	mpf_get_d_2exp(&N, x);
	N = -N;
	if(N == 0)
	  N = 1;
	
	iters = P;
	iters /= N;
	iters += 3;
      }
      
      mpf_init2(y, P);
      mpf_init2(t2, P);
      mpf_init2(t3, P);
      
      mpf_set_ui(y, 0);
      mpf_set_ui(t, 1);
      
      iters /= 2; // unrolling: saves number of variables in loop
      
      for(unsigned long int i=1;i<=iters;i++){
	mpf_mul(t2, t, x);    // t2 = x^k-1 * x
	mpf_div_ui(t, t2, 2*i - 1); // t  = x^k / k
	mpf_sub(t3, y, t);    // t3 = y - t
	
	mpf_mul(t, t2, x);    // .. same operations again
	mpf_div_ui(t2, t, 2*i);
	mpf_sub(y, t3, t2);
      }
      
      
      if(negate)
	mpf_neg(x, y);
      else
	mpf_swap(x, y);
      
      mpf_clear(y);
      mpf_clear(t);
      mpf_clear(t2);
      mpf_clear(t3);
      
    }
    
    //////////////////////////////////////////////////////////////////////
    // calculates pi for given precision
    
    realnumber pi(unsigned long int prec){
      // Salamin-Brent algorithm
      
      if(prec < 1)
	prec = 1;
      
      mpf_t ak, bk, ck, sk, pk, t;
      mpf_init2(ak, prec);
      mpf_init2(bk, prec);
      mpf_init2(ck, prec);
      mpf_init2(sk, prec);
      mpf_init2(pk, prec);
      mpf_init2(t, prec);
      
      mpf_set_ui(ak, 1UL);           // a0 = 1
      mpf_div_2exp(sk, ak, 1UL);     // s0 = 1/2
      mpf_sqrt(bk, sk);              // b0 = 1/sqrt(2);
      
      // calculate N from prec
      // 
      // approximatedly (from below): 
      // prec(i) = 2^(1.015*iter)
      // (where prec is in digits, in bits this 10 ~ 8 ~ 2**3 (3 bits)
      // iter    = log(prec)/(1.015*log(2))
      // 
      // it is important to *NOT* to use too many iterations either
      // otherwise roundoff errors can creep into equations and become
      // very large. The equation below has been hand-tuned to give
      // correct results. Usually only the last 5 decimal digits are wrong.
      // 
      // The equation below was found by knowing that precision increases as O(2^x)
      // and by manually testing how well the last digits were calculated
      // by comparing results with one which were computed with higher precision.
      // More formal mathematical optimization approach could probably find somewhat
      // better formula for calculating optimal number of iterations.
      // 
      // For digits < 1000, computer approximations were compared to pi values
      // given in various books.
      // 
      // Too many iterations can really destroy the accuracy in computations.
      // For example, with precision of 50000 and too many iterations you
      // get 4.1xxx*10e-4 as approximated value of pi if you use very many iterations 
      // (150 000 iterations).
      
      const unsigned long int N = (unsigned long int)
	(0.5 + whiteice::math::log((double)prec)/(1.015*whiteice::math::log(2.0)));
      
      for(unsigned long int k=1;k<N;k++){
	// bk = sqrt(a(k-1)*b-(k-1))
	mpf_mul(t, ak, bk); // bk^2 = a(k-1)*b(k-1)	
	
	// ak = (a(k-1) + b(k-1))/2
	mpf_add(pk, ak, bk);
	mpf_div_ui(ak, pk, 2UL);
	mpf_mul(pk, ak, ak); // ak^2
	
	// updates bk when it is no longer needed
	mpf_sqrt(bk, t); // bk = sqrt(a(k-1)*b(k-1))
	
	// ck = ak*ak - bk*bk
	mpf_sub(ck, pk, t);
	
	// sk = sk - 2^k * ck
	mpf_mul_2exp(t, ck, k);
	mpf_sub(ck, sk, t);
	mpf_swap(ck, sk);
      }
      
      // pk = (2*ak*ak)/sk
      mpf_mul_2exp(t, pk, 1);
      mpf_div(pk, t, sk);
      
      
      mpf_clear(ak);
      mpf_clear(bk);
      mpf_clear(ck);
      mpf_clear(sk);
      mpf_clear(t);

      realnumber res(pk);

      mpf_clear(pk);
      
      return res;
    }
    
    
    
    //////////////////////////////////////////////////////////////////////
    
    // conversion function between doubles and floats
    bool convert(float&  B, const float  A) { B = A; return true; }
    bool convert(float&  B, const double A) { B = (float)A; return true; }
    bool convert(double& B, const float  A) { B = (double)A; return true; }
    bool convert(double& B, const double A) { B = A; return true; }
    
    bool convert(double& B, const char A) { B = (double)A; return true; }
    bool convert(double& B, const unsigned char A) { B = (double)A; return true; }
    bool convert(double& B, const int A) { B = (double)A; return true; }
    bool convert(double& B, const unsigned int A) { B = (double)A; return true; }
    
    bool convert(unsigned int& B, const float A)   { B = (unsigned int)A; return true; }
    bool convert(unsigned int& B, const double A)  { B = (unsigned int)A; return true; }
    bool convert(int& B, const float A)            { B = (int)A; return true; }
    bool convert(int& B, const double A)           { B = (int)A; return true; }

    bool convert(unsigned int& B, const blas_real<float> A)      { B = (unsigned int)A.c[0]; return true; }
    bool convert(unsigned int& B, const blas_complex<float> A)   { B = (unsigned int)A.real(); return true; }
    bool convert(unsigned int& B, const blas_real<double> A)     { B = (unsigned int)A.c[0]; return true; }
    bool convert(unsigned int& B, const blas_complex<double> A)  { B = (unsigned int)A.real(); return true; }

    bool convert(int& B, const blas_real<float> A)               { B = (int)A.c[0]; return true; }
    bool convert(int& B, const blas_complex<float> A)            { B = (int)A.real(); return true; }
    bool convert(int& B, const blas_real<double> A)              { B = (int)A.c[0]; return true; }
    bool convert(int& B, const blas_complex<double> A)           { B = (int)A.real(); return true; }

    bool convert(float& B,  const blas_real<float> A) { B = (float)A.c[0]; return true; }
    bool convert(float& B,  const blas_real<double> A) { B = (float)A.c[0]; return true; }
    bool convert(double& B, const blas_real<float> A) { B = (double)A.c[0]; return true; }
    bool convert(double& B, const blas_real<double> A) { B = (double)A.c[0]; return true; }
    
    bool convert(float& B,  const blas_complex<float> A) { B = (float)A.real(); return false; }
    bool convert(float& B,  const blas_complex<double> A) { B = (float)A.real(); return false; }
    bool convert(double& B, const blas_complex<float> A) { B = (double)A.real(); return false; }
    bool convert(double& B, const blas_complex<double> A) { B = (double)A.real(); return false; }


    bool convert(blas_real<float>& B, const blas_complex<float> A)  { B.c[0] = (float)A.c[0]; return true; }
    bool convert(blas_real<float>& B, const blas_complex<double> A) { B.c[0] = (float)A.c[0]; return true; }
    bool convert(blas_real<float>& B, const std::complex<float> A)       { B = (float)A.real(); return true; }
    bool convert(blas_real<float>& B, const std::complex<double> A)      { B = (float)A.real(); return true; }
    bool convert(blas_complex<float>& B, const blas_real<float> A)  { B.c[0] = (float)A.c[0]; B.c[1] = 0.0f; return true; }
    bool convert(blas_complex<double>& B, const blas_real<float> A) { B.c[0] = (double)A.c[0]; B.c[1] = 0.0f; return true; }
    //bool convert(blas_complex<float>& B, const blas_real<double> A)  { B = (float)A.c[0]; return true; }
    //bool convert(blas_complex<double>& B, const blas_real<double> A) { B = (double)A.c[0]; return true; }
    bool convert(std::complex<float>& B, const blas_real<float> A)       { B = (float)A.c[0]; return true; }
    bool convert(std::complex<float>& B, const blas_real<double> A)      { B = (float)A.c[0]; return true; }
    bool convert(std::complex<double>& B, const blas_real<float> A)       { B = (double)A.c[0]; return true; }
    bool convert(std::complex<double>& B, const blas_real<double> A)      { B = (double)A.c[0]; return true; }

    bool convert(std::complex<float>& B, const blas_complex<float> A){
      B = std::complex<float>(A.c[0], A.c[1]);
      return true;
    }
								 
    bool convert(std::complex<float>& B, const blas_complex<double> A){
      B = std::complex<float>(A.c[0], A.c[1]);
      return true;
    }

    bool convert(std::complex<double>& B, const blas_complex<float> A){
      B = std::complex<double>(A.c[0], A.c[1]);
      return true;
    }
								 
    bool convert(std::complex<double>& B, const blas_complex<double> A){
      B = std::complex<double>(A.c[0], A.c[1]);
      return true;
    }
    
    bool convert(blas_real<float>& B, const std::complex< blas_real<float> > A)
    {
      B = A.real();
      return true;
    }
    
    bool convert(blas_real<double>& B, const std::complex< blas_real<double> > A)
    {
      B = A.real();
      return true;
    }

    bool convert(blas_complex<float>& B, const std::complex< blas_real<float> > A)
    {
      B.c[0] = A.real().c[0];
      B.c[1] = A.imag().c[0];
      return true;
    }
    
    bool convert(blas_complex<double>& B, const std::complex< blas_real<double> > A)
    {
      B.c[0] = A.real().c[0];
      B.c[1] = A.imag().c[0];
      return true;
    }
    
    bool convert(complex<blas_real<float> >& B, const blas_complex<float> A)
    {
      B = whiteice::math::complex< blas_real<float> >(A.c[0], A.c[1]);
      return true;
    }
    
    bool convert(complex<blas_real<double> >& B, const blas_complex<double> A)
    {
      B = whiteice::math::complex< blas_real<float> >(A.c[0], A.c[1]);
      return true;
    }


    bool convert(int& B, const math::superresolution< math::blas_real<float>, math::modular<unsigned int> > A)
    {
      B = (int)(A[0].c[0]);
      return true;
    }

    bool convert(int& B, const math::superresolution< math::blas_real<double>, math::modular<unsigned int> > A)
    {
      B = (int)(A[0].c[0]);
      return true;
    }

    bool convert(math::superresolution< math::blas_real<float>, math::modular<unsigned int> >& B, const int A)
    {
      B.zero();
      B[0].c[0] = A;
      return true;
    }

    bool convert(math::superresolution< math::blas_real<double>, math::modular<unsigned int> >& B, const int A)
    {
      B.zero();
      B[0].c[0] = A;
      return true;
    }


    bool convert(superresolution< blas_real<float>, modular<unsigned int> >& B,
		 const superresolution< blas_real<float>, modular<unsigned int> > A)
    {
      B = A;
      return true;
    }

    bool convert(superresolution< blas_real<double>, modular<unsigned int> >& B,
		 const superresolution< blas_real<double>, modular<unsigned int> > A)
    {
      B = A;
      return true;
    }


    bool convert(float& B,
		 const superresolution< blas_real<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(float& B,
		 const superresolution< blas_real<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(double& B,
		 const superresolution< blas_real<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(double& B,
		 const superresolution< blas_real<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }
    
    bool convert(superresolution< blas_real<float>, modular<unsigned int> >& B,
		 const float A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_real<float>, modular<unsigned int> >& B,
		 const double A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_real<double>, modular<unsigned int> >& B,
		 const float A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_real<double>, modular<unsigned int> >& B,
		 const double A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    
    bool convert(complex<float>& B,
		 const superresolution< blas_real<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }    

    bool convert(complex<double>& B,
		 const superresolution< blas_real<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(complex<float>& B,
		 const superresolution< blas_real<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(complex<double>& B,
		 const superresolution< blas_real<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    

    bool convert(superresolution< blas_real<float>, modular<unsigned int> >& B,
		 const std::complex<float> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_real<float>, modular<unsigned int> >& B,
		 const std::complex<double> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_real<double>, modular<unsigned int> >& B,
		 const std::complex<float> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_real<double>, modular<unsigned int> >& B,
		 const std::complex<double> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }
    
    
    bool convert(superresolution< blas_real<float>, modular<unsigned int> >& B,
		 const blas_real<float> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_real<double>, modular<unsigned int> >& B,
		 const blas_real<double> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_real<float>, modular<unsigned int> >& B,
		 const blas_complex<float> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_real<double>, modular<unsigned int> >& B,
		 const blas_complex<double> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_real<double>, modular<unsigned int> >& B,
		 const blas_complex<float> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_real<float>, modular<unsigned int> >& B,
		 const blas_complex<double> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }


    
    
    bool convert(blas_real<float>& B,
		 const superresolution< blas_real<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }
		 
    bool convert(blas_real<double>& B,
		 const superresolution< blas_real<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }
		 
    bool convert(blas_complex<float>& B,
		 const superresolution< blas_real<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }
		 
    bool convert(blas_complex<double>& B,
		 const superresolution< blas_real<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(blas_real<float>& B,
		 const superresolution< blas_real<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(blas_complex<double>& B,
		 const superresolution< blas_real<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(blas_real<double>& B,
		 const superresolution< blas_real<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    

    bool convert(blas_complex<float>& B,
		 const superresolution< blas_complex<double>, modular<unsigned int> > A)
    {
      whiteice::math::convert(B, A[0]);
      return true;
    }

    bool convert(superresolution<whiteice::math::blas_real<float>, modular<unsigned int> >& B,
		 superresolution<whiteice::math::blas_complex<double>, modular<unsigned int> > A)
    {
      for(unsigned int i=0;i<B.size();i++){
	B[i] = A[i][0];
      }

      return true;
    }

    bool convert(superresolution<whiteice::math::blas_real<double>, modular<unsigned int> >& B,
		 superresolution<whiteice::math::blas_complex<float>, modular<unsigned int> > A)
    {
      for(unsigned int i=0;i<B.size();i++){
	B[i] = A[i][0];
      }
      
      return true;
    }
    
    bool convert(superresolution< blas_complex<float>, modular<unsigned int> >& B,
		 superresolution< blas_complex<double>, modular<unsigned int> > A)
    {
      for(unsigned int i=0;i<B.size();i++){
	whiteice::math::convert(B[i], A[i]);
      }
      
      return true;
    }


    bool convert(superresolution<blas_complex<double>, modular<unsigned int> >& B,
		 superresolution<blas_complex<float>, modular<unsigned int> > A)
    {
      for(unsigned int i=0;i<B.size();i++){
	whiteice::math::convert(B[i], A[i]);
      }
      
      return true;
    }

    
    bool convert(superresolution< blas_complex<float>, modular<unsigned int> >& B,
		 const superresolution< blas_complex<float>, modular<unsigned int> > A)
    {
      B = A;
      return true;
    }

    bool convert(superresolution< blas_complex<double>, modular<unsigned int> >& B,
		 const superresolution< blas_complex<double>, modular<unsigned int> > A)
    {
      B = A;
      return true;
    }


    


    bool convert(float& B,
		 const superresolution< blas_complex<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(float& B,
		 const superresolution< blas_complex<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(double& B,
		 const superresolution< blas_complex<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(double& B,
		 const superresolution< blas_complex<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }
    
    bool convert(superresolution< blas_complex<float>, modular<unsigned int> >& B,
		 const float A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_complex<float>, modular<unsigned int> >& B,
		 const double A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_complex<double>, modular<unsigned int> >& B,
		 const float A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_complex<double>, modular<unsigned int> >& B,
		 const double A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    
    bool convert(std::complex<float>& B,
		 const superresolution< blas_complex<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }    

    bool convert(std::complex<double>& B,
		 const superresolution< blas_complex<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(std::complex<float>& B,
		 const superresolution< blas_complex<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(std::complex<double>& B,
		 const superresolution< blas_complex<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    

    bool convert(superresolution< blas_complex<float>, modular<unsigned int> >& B,
		 const std::complex<float> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_complex<float>, modular<unsigned int> >& B,
		 const std::complex<double> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_complex<double>, modular<unsigned int> >& B,
		 const std::complex<float> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_complex<double>, modular<unsigned int> >& B,
		 const std::complex<double> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }
    
    
    bool convert(superresolution< blas_complex<float>, modular<unsigned int> >& B,
		 const blas_real<float> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_complex<double>, modular<unsigned int> >& B,
		 const blas_real<double> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_complex<float>, modular<unsigned int> >& B,
		 const blas_complex<float> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_complex<double>, modular<unsigned int> >& B,
		 const blas_complex<double> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_complex<double>, modular<unsigned int> >& B,
		 const blas_complex<float> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }

    bool convert(superresolution< blas_complex<float>, modular<unsigned int> >& B,
		 const blas_complex<double> A)
    {
      B.zero();
      return whiteice::math::convert(B[0], A);
    }


    
    
    bool convert(blas_real<float>& B,
		 const superresolution< blas_complex<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }
		 
    bool convert(blas_real<double>& B,
		 const superresolution< blas_complex<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }
		 
    bool convert(blas_complex<float>& B,
		 const superresolution< blas_complex<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }
		 
    bool convert(blas_complex<double>& B,
		 const superresolution< blas_complex<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(blas_real<float>& B,
		 const superresolution< blas_complex<double>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(blas_complex<double>& B,
		 const superresolution< blas_complex<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    bool convert(blas_real<double>& B,
		 const superresolution< blas_complex<float>, modular<unsigned int> > A)
    {
      return whiteice::math::convert(B, A[0]);
    }

    
    bool convert(superresolution< blas_real<float>, modular<unsigned int> >& B,
		 const superresolution< blas_complex<float>, modular<unsigned int> > A)
    {
      for(unsigned int i=0;i<B.size();i++){
	B[i].c[0] = A[i].c[0];
      }

      return true;
    }

    bool convert(superresolution< blas_real<double>, modular<unsigned int> >& B,
		 const superresolution< blas_complex<double>, modular<unsigned int> > A)
    {
      for(unsigned int i=0;i<B.size();i++){
	B[i].c[0] = A[i].c[0];
      }

      return true;
    }

    bool convert(superresolution< blas_complex<float>, modular<unsigned int> >& B,
		 const superresolution< blas_real<float>, modular<unsigned int> > A)
    {
      for(unsigned int i=0;i<B.size();i++){
	B[i].c[0] = A[i].c[0];
	B[i].c[1] = 0.0f;
      }

      return true;;
    }

    bool convert(superresolution< blas_complex<double>, modular<unsigned int> >& B,
		 const superresolution< blas_real<double>, modular<unsigned int> > A)
    {
      for(unsigned int i=0;i<B.size();i++){
	B[i].c[0] = A[i].c[0];
	B[i].c[1] = 0.0;
      }

      return true;
    }

    bool convert(superresolution< blas_complex<double>, modular<unsigned int> >& B,
		 const superresolution< blas_real<float>, modular<unsigned int> > A)
    {
      for(unsigned int i=0;i<B.size();i++){
	B[i].c[0] = A[i].c[0];
	B[i].c[1] = 0.0;
      }

      return true;
    }
    
    bool convert(superresolution< blas_complex<float>, modular<unsigned int> >& B,
		 const superresolution< blas_real<double>, modular<unsigned int> > A)
    {
      for(unsigned int i=0;i<B.size();i++){
	B[i].c[0] = (float)(A[i].c[0]);
	B[i].c[1] = 0.0f;
      }

      return true;
    }
    

    bool convert(float& B, const complex<float> A){ B = (float)std::real(A); return true; }
    bool convert(double& B, const complex<double> A){ B = (double)std::real(A); return true; }
    bool convert(double& B, const complex<float> A){ B = (double)std::real(A); return true; }
    bool convert(float& B, const complex<double> A){ B = (float)std::real(A); return true; }
    
    bool convert(blas_real<double>& B, const blas_real<float> A)    { B = (double)A.c[0]; return true; }
    bool convert(blas_real<float>& B, const blas_real<double> A)    { B = (float)A.c[0]; return true; }

    bool convert(blas_real<float>& B, const double A){ B = (float)A; return true; }

    bool convert(complex<float>& B, const float A){ B = (float)A; return true; }
    bool convert(complex<double>& B, const double A){ B = (double)A; return true; }

    
    
    //////////////////////////////////////////////////////////////////////
    
    template <typename T, typename S>
    whiteice::math::superresolution<T,S> sqrt(const whiteice::math::superresolution<T,S> x)
    {
      // Fourier transform approach
      // circ_conv(x,x) = y, Fourier transform => X² = Y => x = F^-1( sqrt(F(y)) )

      whiteice::math::superresolution< math::blas_complex<double>, modular<unsigned int> > z;

      for(unsigned int i=0;i<z.size();i++)
	whiteice::math::convert(z[i], x[i]);
      
      z.fft();
      
      for(unsigned int i=0;i<z.size();i++)
	z[i] = whiteice::math::sqrt(z[i]);
      
      z.inverse_fft();
      
      whiteice::math::superresolution<T,S> result;
      
      for(unsigned int i=0;i<z.size();i++)
	whiteice::math::convert(result[i], z[i]);

#if 0
      // fixes sign to be mostly +
      unsigned int plusses = 0;
      for(unsigned int i=0;i<result.size();i++)
	if(real(result[i]) >= T(0.0f)) plusses++;
      
      // change sign if there are more negative signs => mostly positive signs
      if(plusses < (result.size()/2)) 
	result = -result;
#endif

      // if most of the weight mass is negative we change the sign!
      T mass = T(0.0f);
      for(unsigned int i=0;i<result.size();i++)
	mass += real(result[i]);

      if(real(mass) < 0.0f)
	result = -result; 
      
      return result;
      
      
#if 0
      // polynomial square root algorithm [DO NOT WORK WITH MODULAR POLYNOMIAL ARITHMETIC]

      const unsigned int N = x.size();
      whiteice::math::superresolution<T,S> result(0.0f);
      auto xx = x;

      // first term
      int t = N-1;

      float value = 0.0f;
      whiteice::math::superresolution<T,S> prev(0.0f);
      int pk = t/2;
      
      do{
	convert(value, xx[t]);
	
	if(value){
	  result[t/2] = sqrt(xx[t]);

	  std::cout << "x = " << xx << std::endl;
	  std::cout << "q = " << prev << std::endl;
	  std::cout << "r = " << result << std::endl;
	  
	  pk = t/2;
	  prev[t/2] = result[t/2];
	  xx[t] = T(0.0f);

	  prev[pk] = prev[pk]*T(2.0f);
	}
	
	t--;
      }
      while(!value && t >= 0); 

	    
      for(;t>=0;t--){
	
	convert(value, xx[t]);
	
	if(value){ // non-zero element

	  std::cout << "---------------------------------------------" << std::endl;

	  std::cout << "x = " << xx << std::endl;
	  std::cout << "q = " << prev << std::endl;
	  
	  const int xk = t-pk;
	  const T scale = xx[t]/prev[pk];
	  prev[xk] += scale;

	  std::cout << "q = " << prev << std::endl;
	  std::cout << "scale = " << scale << std::endl;
	  std::cout << "xk = " << xk << std::endl;

	  auto n = prev;

	  n = n*scale;

	  {
	    auto nn = n; 
	    
	    for(int i=((signed)n.size());i>=0;i--)
	      nn[i+xk] = n[i];
	    
	    n = nn;
	  }
	  
#if 0
	  for(int i=((signed)n.size()-xk);i>=0;i--)
	    n[i+xk] = n[i];

	  for(int i=0;i<xk;i++)
	    n[i] = T(0.0f);
#endif
	  
	  std::cout << "n = " << n << std::endl;
	  
	  xx -= n;

	  result[xk] += scale;
	  
	  std::cout << "r = " << result << std::endl;

	  prev[xk] = prev[xk]*T(2.0f);
	}
	
      }

      return result;
#endif
      
#if 0
      // Taylor's series expansion at point c = 1 (Taylor's series DON'T WORK!)

      whiteice::math::superresolution<T,S> result(0.0f);

      // Taylor-8 at c=1 point (gives wrong results for x <= 0 points)  .

      result += superresolution<T,S>(1.0f);

      const T c1 = T(0.5f);
      result += x*c1;

      const T c2 = T(-1.0f/8.0f);
      result += x*x*c2;

      const T c3 = T(1.0f/16.0f);
      result += x*x*x*c3;

      const T c4 = T(-5.0f/128.0f);
      result += x*x*x*x*c4;

      const T c5 = T(7.0f/256.0f);
      result += x*x*x*x*x*c5;

      const T c6 = T(-21.0f/1024.0f);
      result += x*x*x*x*x*x*c6;

      const T c7 = T(66.f/4096.0f);
      result += x*x*x*x*x*x*x*c7;

      const T c8 = T(-429.0f/(24.0f*16384.0f));
      result += x*x*x*x*x*x*x*x*c8;

      
      return result;
#endif
    }


    template whiteice::math::superresolution< blas_real<float>, modular<unsigned int> >
    sqrt(const whiteice::math::superresolution< blas_real<float> , modular<unsigned int> > x);

    template whiteice::math::superresolution< blas_real<double>, modular<unsigned int> >
    sqrt(const whiteice::math::superresolution< blas_real<double> , modular<unsigned int> > x);

    // complex value implementations should NOT work
    template whiteice::math::superresolution< blas_complex<float>, modular<unsigned int> >
    sqrt(const whiteice::math::superresolution< blas_complex<float> , modular<unsigned int> > x);

    template whiteice::math::superresolution< blas_complex<double>, modular<unsigned int> >
    sqrt(const whiteice::math::superresolution< blas_complex<double> , modular<unsigned int> > x);

    
    //////////////////////////////////////////////////////////////////////

    bool isinf(superresolution<blas_complex<float> , modular<unsigned int> > v){
      return false;
    }
    
    bool isinf(superresolution<blas_complex<double>, modular<unsigned int> > v){
      return false;
    }

    bool isnan(superresolution<blas_complex<float> , modular<unsigned int> > v){
      return false;
    }
    
    bool isnan(superresolution<blas_complex<double>, modular<unsigned int> > v){
      return false;
    }
    
    bool isinf(superresolution<blas_real<float> , modular<unsigned int> > v){
      return false;
    }
    
    bool isinf(superresolution<blas_real<double>, modular<unsigned int> > v){
      return false;
    }

    bool isnan(superresolution<blas_real<float> , modular<unsigned int> > v){
      return false;
    }
    
    bool isnan(superresolution<blas_real<double>, modular<unsigned int> > v){
      return false;
    }


    
    // for FFT . bits must be smaller than 8*sizeof(int)
    // (32bits nowadays - assuming 8 bit bytes)
    unsigned int bitreverse(unsigned int index,
			    unsigned int bits)
    {
      unsigned int result = 0;
      
      for(unsigned i=0;i<bits;i++)
      {
	result <<= 1;
	
	if(index & 1) result++;
	
	index >>= 1;
      }
      
      return result;
    }


    
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    //std::mutex plan_mutex;
    //unsigned int has_plan = 0;
    //cfft_plan basic_fft_plan;  // FIXME: global variable should be free'ed at exit [saves computing time]

    
    // slow arbitrary length signal FFT
    template <typename T>
    bool basic_fft(vertex< whiteice::math::blas_complex<T> >& v) 
    {
#if 0
      const unsigned int LEN = v.size();

      double* buffer = (double*)malloc(LEN*sizeof(double)*2);

      if(buffer == NULL){
	return false;
      }

      for(unsigned int i=0;i<LEN;i++){
	buffer[2*i+0] = v[i].c[0];
	buffer[2*i+1] = v[i].c[1];
      }

      {
	/*
	std::lock_guard<std::mutex> lock(plan_mutex);

	cfft_plan plan;
	
	if(has_plan != LEN){
	  if(has_plan) destroy_cfft_plan(basic_fft_plan);
	  basic_fft_plan = make_cfft_plan(LEN);
	  plan = basic_fft_plan;
	  has_plan = LEN;
	}
	else{
	  plan = basic_fft_plan;
	}
	*/

	cfft_plan plan = make_cfft_plan(LEN);
	cfft_forward(plan, buffer, 1.0);
	destroy_cfft_plan(plan);
      }


      for(unsigned int i=0;i<LEN;i++){
	v[i].c[0] = buffer[2*i+0];
	v[i].c[1] = buffer[2*i+1];
      }

      free(buffer);


      return true;
#endif
      
#if 1
      const unsigned int LEN = v.size();
      
      shape_t shape { LEN };
      stride_t stride(1);

      //stride[0] = sizeof(T)*2;
      //size_t tmp = sizeof(T)*2;
      stride[0]  = sizeof(std::complex<T>);
      size_t tmp = sizeof(whiteice::math::blas_complex<T>);
      tmp *= shape[0];

      shape_t axes;
      for(unsigned int i=0;i<shape.size();++i)
	axes.push_back(i);

      c2c(shape, stride, stride, axes, FORWARD,
	  //data.data(), res.data(),
	  (std::complex<T>*)&(v[0]),  (std::complex<T>*)&(v[0]),
	  T(1.0));

      return true;
#endif
    }

    
    template <typename T>
    bool basic_ifft(vertex< whiteice::math::blas_complex<T> >& v) 
    {
#if 0
      const unsigned int LEN = v.size();
      
      double* buffer = (double*)malloc(LEN*sizeof(double)*2);

      if(buffer == NULL){
	return false;
      }
      
      for(unsigned int i=0;i<LEN;i++){
	buffer[2*i+0] = v[i].c[0];
	buffer[2*i+1] = v[i].c[1];
      }

      //double* buffer = (double*)(&(v[0])); 

      {
	/*
	std::lock_guard<std::mutex> lock(plan_mutex);
	
	cfft_plan plan;

	if(has_plan != LEN){
	  if(has_plan) destroy_cfft_plan(basic_fft_plan);
	  basic_fft_plan = make_cfft_plan(LEN);
	  plan = basic_fft_plan;
	  has_plan = LEN;
	}
	else{
	  plan = basic_fft_plan;
	}
	*/

	cfft_plan plan = make_cfft_plan(LEN);
	cfft_backward(plan, buffer, 1.0/LEN);
	destroy_cfft_plan(plan);
      }
      
      
      for(unsigned int i=0;i<LEN;i++){
	v[i].c[0] = buffer[2*i+0];
	v[i].c[1] = buffer[2*i+1];
      }
      
      free(buffer);

      

      return true;
#endif

      
#if 1
      const unsigned int LEN = v.size();

      shape_t shape { LEN };
      stride_t stride(1);

      stride[0] = sizeof(T)*2;
      size_t tmp = sizeof(T)*2;
      tmp *= shape[0];

      shape_t axes;
      for(unsigned int i=0;i<shape.size();++i)
	axes.push_back(i);

      c2c(shape, stride, stride, axes, BACKWARD,
	  (std::complex<T>*)&(v[0]),
	  (std::complex<T>*)&(v[0]),
	  T(1.0/LEN));

      return true;
#endif
    }


    template bool basic_fft<float>(vertex< whiteice::math::blas_complex<float> >& v);
    template bool basic_ifft<float>(vertex< whiteice::math::blas_complex<float> >& v);

    template bool basic_fft<double>(vertex< whiteice::math::blas_complex<double> >& v);
    template bool basic_ifft<double>(vertex< whiteice::math::blas_complex<double> >& v); 

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    
  }
}






