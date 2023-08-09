
#ifndef superresolution_cpp
#define superresolution_cpp

#include "superresolution.h"
#include "modular.h"
#include "blade_math.h"


namespace whiteice
{
  namespace math
  {

    template <typename T, typename U>
    superresolution<T,U>::superresolution()
    {
      // initializes one dimensional scaling basis with zero value
      
      for(unsigned int i=0;i<this->size();i++)
	basis[i] = T(0.0f);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>::superresolution(const T value)
    {
      for(unsigned int i=1;i<this->size();i++)
	basis[i] = T(0.0f);

      basis[0] = value;
    }

    template <typename T, typename U>
    superresolution<T,U>::superresolution(const superresolution<T,U>& s)
    {
      for(unsigned int i=0;i<size();i++)
	this->basis[i] = s.basis[i];
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>::superresolution(const std::vector<T>& values)
    {
      // initializes values.size() - dimensional basis
      if(values.size() != this->size())
	throw illegal_operation("Incorrect basis size in input vector");

      for(unsigned int i=0;i<size();i++)
	basis[i] = values[i];
      
    }   
    
    
    template <typename T, typename U>
    superresolution<T,U>::~superresolution()
    {
      
    }


    // sets superresolution componenets to be all zero
    template <typename T, typename U>
    void superresolution<T,U>::zeros()
    {
      for(unsigned int i=0;i<size();i++)
	basis[i] = T(0.0f);
    }
    
    // sets supperresolution components to be all ones
    template <typename T, typename U>
    void superresolution<T,U>::ones()
    {
      for(unsigned int i=0;i<size();i++)
	basis[i] = T(1.0f);
    }


    // operators
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator+(const superresolution<T,U>& s)
      const 
    {
      superresolution<T,U> t(*this);

      for(unsigned int i=0;i<size();i++){
	t.basis[i] += s.basis[i];
      }
      
#if 0
      U u = U(0);
      t.basis[u[0]] += s.basis[U(0)[0]];

      for(U u=U(1);u != U(0);u++)
	t.basis[u[0]] += s.basis[u[0]];
#endif

      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator-(const superresolution<T,U>& s) const
      
    {
      superresolution<T,U> t(*this);

      for(unsigned int i=0;i<size();i++){
	t.basis[i] -= s.basis[i];
      }

#if 0
      U u = U(0);
      t.basis[u[0]] -= s.basis[u[0]];
      
      for(U u=U(1);u != U(0);u++)
	t.basis[u[0]] -= s.basis[u[0]];
#endif
      
      return t;      
    }
    
    
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator*(const superresolution<T,U>& s) const
      
    {
#if 1
      // for small number lengths direct convolution is faster than FFT. This is O(N^2) thought.
      // with N=31, N^2 = 1000 and 3*N*log(N) + N = 31*5 = 500 multiplications. So FFT is faster??

      superresolution<T,U> result(T(0));

      const unsigned int N = s.size();

#pragma GCC unroll 7 
      for(unsigned int i=0;i<N;i++)
#pragma GCC unroll 7
	for(unsigned int j=0;j<N;j++)
	  result[(i+j)%N] += (s.basis[i])*(this->basis[j]);
      
    return result;
#endif
    
#if 0
      // z = convolution(x, y)
      // z = InvFFT(FFT(x)*FFT(y))
      
      whiteice::math::vertex< whiteice::math::blas_complex<double> > b1, b2;
      b1.resize(this->size());
      b2.resize(this->size());

      for(unsigned int i=0;i<b1.size();i++){
	//b1[i] = this->basis[i];
	//b2[i] = s.basis[i];

	whiteice::math::convert(b1[i], this->basis[i]);
	whiteice::math::convert(b2[i], s.basis[i]);
      }

      // calculates FFT of convolutions
      if(whiteice::math::basic_fft< double >(b1) == false ||
	 whiteice::math::basic_fft< double >(b2) == false)
	throw illegal_operation("FFT failed");

      // inverse computation of convolution
      for(unsigned int i=0;i<DEFAULT_MODULAR_BASIS;i++){
	b1[i] *= b2[i];
      }

      // inverse FFT
      if(whiteice::math::basic_ifft< double >(b1) == false)
	throw illegal_operation("Inverse FFT failed");

      superresolution<T,U> result(T(0));
      
      for(unsigned int i=0;i<b1.size();i++){
	whiteice::math::convert(result.basis[i], b1[i]);
	//result.basis[i] = T(b1[i]);
      }

      return result;
#endif
    }
    
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator/(const superresolution<T,U>& s) const
      
    {
      {
	if(this->size() != s.size())
	  throw illegal_operation("Not same basis");

	// don't check basis dimension is PRIME

#if 0
	if(s.iszero())
	  throw illegal_operation("division by zero");
#endif
      }

      // z = convolution(x, (1/y))
      // z = InvFFT(FFT(x)/FFT(y))
      
      whiteice::math::vertex< whiteice::math::blas_complex<double> > b1, b2;
      b1.resize(this->size());
      b2.resize(this->size());

      for(unsigned int i=0;i<b1.size();i++){
	//b1[i] = this->basis[i];
	//b2[i] = s.basis[i];

	whiteice::math::convert(b1[i], this->basis[i]);
	whiteice::math::convert(b2[i], s.basis[i]);
      }

      // calculates FFT of convolutions
      if(whiteice::math::basic_fft< double >(b1) == false ||
	 whiteice::math::basic_fft< double >(b2) == false)
	throw illegal_operation("FFT failed");

      // inverse computation of convolution
      for(unsigned int i=0;i<b1.size();i++){
	if(b2[i] != whiteice::math::blas_complex<double>(0.0f)){
	  b1[i] = b1[i] / b2[i];
	}
      }

      // inverse FFT
      if(whiteice::math::basic_ifft< double >(b1) == false)
	throw illegal_operation("Inverse FFT failed");

      superresolution<T,U> result(T(0));
      
      for(unsigned int i=0;i<b1.size();i++){
	whiteice::math::convert(result.basis[i], b1[i]);
	//result.basis[i] = T(b1[i]);
      }

      return result;
    }

    
    
    // complex conjugate
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator!() const 
      
    {
      // conjugates both numbers and basises            
      
      superresolution<T,U> s(T(0)); // s = 0 vector initially
      superresolution<T,U> t(T(0));

      t.basis[U(0)[0]] = whiteice::math::conj(this->basis[U(0)[0]]);
      
      // numbers + inits basis
      for(U u=U(1);u!=U(0);u++){
	t.basis[u[0]] = whiteice::math::conj(this->basis[u[0]]);
      }
      
      // basis
      s.basis[whiteice::math::conj(U(0))[0]] = t.basis[U(0)[0]];
      
      for(U u=U(1);u!=U(0);u++){
	s.basis[whiteice::math::conj(u)[0]] += t.basis[u[0]];
      }
      
      return s;
    }


    // complex conjugate
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::conj()
    {
      // conjugates both numbers and basises            
      
      superresolution<T,U> t(T(0));

      t.basis[U(0)[0]] = whiteice::math::conj(this->basis[U(0)[0]]);
      
      // numbers + inits basis
      for(U u=U(1);u!=U(0);u++){
	t.basis[u[0]] = whiteice::math::conj(this->basis[u[0]]);
      }
      
      // basis
      this->basis[whiteice::math::conj(U(0)[0])] = t.basis[U(0)[0]];
      
      for(U u=U(1);u!=U(0);u++){
	this->basis[whiteice::math::conj(u[0])] += t.basis[u[0]];
      }
      
      return (*this);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator-() const
      
    {
      superresolution<T,U> s;

      for(unsigned int i=0;i<size();i++)
	s.basis[i] = -(this->basis[i]);

      return s;
      
#if 0
      superresolution<T,U> s(*this);

      s.basis[U(0)[0]] = -s.basis[U(0)[0]];
      
      for(U u=U(1);u!=U(0);u++){
	s.basis[u[0]] = -(s.basis[u[0]]);
      }
      
      return s;
#endif
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator+=(const superresolution<T,U>& s)
      
    {
      for(unsigned int i=0;i<size();i++)
	this->basis[i] += s.basis[i];

#if 0
      this->basis[U(0)[0]] += s.basis[U(0)[0]];
      
      for(U u=U(1);u!=U(0);u++){
	this->basis[u[0]] += s.basis[u[0]];
      }
#endif
      
      return *this;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator-=(const superresolution<T,U>& s)
      
    {
      for(unsigned int i=0;i<size();i++)
	this->basis[i] -= s.basis[i];
      
#if 0
      this->basis[U(0)[0]] -= s.basis[U(0)[0]];
      
      for(U u=U(1);u!=U(0);u++){
	this->basis[u[0]] -= s.basis[u[0]];
      }
#endif
      
      return *this;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator*=(const superresolution<T,U>& s)
    {
      // small number of dimensions direct convolution is faster than FFT
      
      (*this) = (*this) * s;
      return (*this);

#if 0
      // z = convolution(x, y)
      // z = InvFFT(FFT(x)*FFT(y))
      
      whiteice::math::vertex< whiteice::math::blas_complex<double> > b1, b2;
      b1.resize(this->size());
      b2.resize(this->size());

      for(unsigned int i=0;i<b1.size();i++){
	//b1[i] = this->basis[i];
	//b2[i] = s.basis[i];

	whiteice::math::convert(b1[i], this->basis[i]);
	whiteice::math::convert(b2[i], s.basis[i]);
      }

      // calculates FFT of convolutions
      if(whiteice::math::basic_fft< double >(b1) == false ||
	 whiteice::math::basic_fft< double >(b2) == false)
	throw illegal_operation("FFT failed");

      // inverse computation of convolution
      for(unsigned int i=0;i<DEFAULT_MODULAR_BASIS;i++){
	b1[i] *= b2[i];
      }

      // inverse FFT
      if(whiteice::math::basic_ifft< double >(b1) == false)
	throw illegal_operation("Inverse FFT failed");

      for(unsigned int i=0;i<b1.size();i++){
	whiteice::math::convert(this->basis[i], b1[i]);
	//this->basis[i] = T(b1[i]);
      }

      return (*this);
#endif
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator/=(const superresolution<T,U>& s)
      
    {
      {
	if(this->size() != s.size())
	  throw illegal_operation("Not same basis");

	// don't check basis dimension is PRIME

#if 0
	if(s.iszero())
	  throw illegal_operation("division by zero");
#endif
      }

      // z = convolution(x, (1/y))
      // z = InvFFT(FFT(x)/FFT(y))
      
      whiteice::math::vertex< whiteice::math::blas_complex<double> > b1, b2;
      b1.resize(this->size());
      b2.resize(this->size());

      for(unsigned int i=0;i<b1.size();i++){
	//b1[i] = this->basis[i];
	//b2[i] = s.basis[i];

	whiteice::math::convert(b1[i], this->basis[i]);
	whiteice::math::convert(b2[i], s.basis[i]);
      }
      
      // calculates FFT of convolutions
      if(whiteice::math::basic_fft< double >(b1) == false ||
	 whiteice::math::basic_fft< double >(b2) == false)
	throw illegal_operation("FFT failed");

      // inverse computation of convolution
      for(unsigned int i=0;i<b1.size();i++){
	if(b2[i] != whiteice::math::blas_complex<double>(0))
	  b1[i] /= b2[i];
      }

      // inverse FFT
      if(whiteice::math::basic_ifft< double >(b1) == false)
	throw illegal_operation("Inverse FFT failed");

      for(unsigned int i=0;i<b1.size();i++){
	whiteice::math::convert(this->basis[i], b1[i]);
	//this->basis[i] = b1[i];
      }

      return (*this);
    }

    
    // inner product between elements of superresolutional numbers make sometimes sense!!

    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::innerproduct()
    {
      T dot = T(0.0f);
      for(unsigned int i=0;i<size();i++){
	dot += (this->basis[i])*(this->basis[i]);
	this->basis[i] = T(0.0f);
      }

      this->basis[0] = dot;

      return (*this);
    }

    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::innerproduct(const superresolution<T,U>& s) const
    {
      T dot = T(0.0f);
      
      for(unsigned int i=0;i<size();i++)
	dot += (this->basis[i])*(s.basis[i]);
      
      return superresolution<T,U>(dot);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator=(const superresolution<T,U>& s)
      
    {
      for(unsigned int i=0;i<size();i++)
	this->basis[i] = s.basis[i];
      
      return *this;
    }

    
    template <typename T, typename U>
    bool superresolution<T,U>::operator==(const superresolution<T,U>& s) const 
      
    
    {
      if(this->size() != s.size()){
	return false;
      }

      for(unsigned int i=0;i<size();i++)
	if(this->basis[i] != s.basis[i])
	  return false;

      return true;
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator!=(const superresolution<T,U>& s) const 
      
    {
      return (!((*this) == s));
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator>=(const superresolution<T,U>& s) const
      
    {
      if(s.size() != this->size())
	throw uncomparable("Non same baisis size numbers are uncomparable");

      int i = this->size() - 1;

      do{
	if(basis[i] > s.basis[i])
	  return true;
	else if(basis[i] < s.basis[i])
	  return false;
	
	i--;
      }
      while(i >= 0);

      return true;
    }
    
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator<=(const superresolution<T,U>& s) const 
    {
      if(s.size() != this->size())
	throw uncomparable("Non same basis size numbers are uncomparable");

      int i = this->size() - 1;
      
      do{
	if(basis[i] < s.basis[i])
	  return true;
	else if(basis[i] > s.basis[i])
	  return false;
	
	i--;
      }
      while(i >= 0);
      
      return true;
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator< (const superresolution<T,U>& s) const 
      
    {
      if(s.size() != this->size())
	throw uncomparable("Non same basis basis size numbers are uncomparable");

      int i = this->size() - 1;
      
      do{
	if(basis[i] < s.basis[i])
	  return true;
	else if(basis[i] > s.basis[i])
	  return false;
	
	i--;
      }
      while(i >= 0);

      return false;
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator> (const superresolution<T,U>& s) const 
      
    {
      if(s.size() != this->size())
	throw uncomparable("Non same basis size numbers are uncomparable");

      int i = this->size() - 1;

      do{
	if(basis[i] > s.basis[i])
	  return true;
	else if(basis[i] < s.basis[i])
	  return false;
	
	i--;
      }
      while(i >= 0);


      return false;
    }
    
    
    
    // scalar operation
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator= (const T& s) 
      
    {
      this->zero();

      this->basis[0] = s;

      return (*this);
    }

    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator+ (const T& s) const
    {
      superresolution<T,U> t(*this);
      
      t.basis[0] += s;
      
      return t;
    }

    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator- (const T& s) const
    {
      superresolution<T,U> t(*this);
      
      t.basis[0] -= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator* (const T& s) const 
      
    {
      superresolution<T,U> t(*this);

      for(unsigned int i=0;i<t.size();i++)
	t.basis[i] *= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator/ (const T& s) const 
    {
      superresolution<T,U> t(*this);

      for(unsigned int i=0;i<t.size();i++)
	t.basis[i] /= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator*=(const T& s) 
    {
      superresolution<T,U>& t = (*this);

      for(unsigned int i=0;i<t.size();i++)
	t.basis[i] *= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator/=(const T& s)
      
    {
      superresolution<T,U>& t = (*this);

      for(unsigned int i=0;i<t.size();i++)
	t.basis[i] /= s;
      
      return t;
    }


    // scalar operation
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator= (const T s) 
      
    {
      this->zero();

      this->basis[0] = s;

      return (*this);
    }
    

#if 0
    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator* (const T s) const 
      
    {
      superresolution<T,U> t(*this);

      U i = U(0);
      
      t.basis[i[0]] *= s;

      for(U i=U(1);i != U(0);i++)
	t.basis[i[0]] *= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator/ (const T s) const 
    {
      superresolution<T,U> t(*this);

      t.basis[U(0)[0]] /= s;
      
      for(U i=U(1);i!=U(0);i++)
	t.basis[i[0]] /= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator*=(const T s) 
    {
      superresolution<T,U>& t = (*this);

      t.basis[U(0)[0]] *= s;
      
      for(U i=U(1);i!=U(0);i++)
	t.basis[i[0]] *= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator/=(const T s)
      
    {
      superresolution<T,U>& t = (*this);

      t.basis[U(0)[0]] /= s;
      
      for(U i=U(1);i!=U(0);i++)
	t.basis[i[0]] /= s;
      
      return t;
    }
#endif
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::abs()
    {
      whiteice::math::superresolution<T,U>& z = (*this);
      whiteice::math::superresolution<T,U> zc = (*this);

      {
	// conjugates both numbers and basises            
	
	zc.basis[U(0)[0]] = whiteice::math::conj(zc.basis[U(0)[0]]);
	
	// numbers + inits basis
	for(U u=U(1);u!=U(0);u++){
	  zc.basis[u[0]] = whiteice::math::conj(zc.basis[u[0]]);
	}

	// DO NOT CALCULATE COMPLEX CONJUGATE OF BASIS FUNCTION!
	
      }
      
      // zc.conj;

      z *= zc;
      z = whiteice::math::sqrt(z);
	
      return z;
    }


    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::zero()
    {
      for(unsigned int i=0;i<this->size();i++)
	basis[i] = T(0);
      
      return (*this);
    }


    template <typename T, typename U>
    bool superresolution<T,U>::iszero() const
    {
      for(unsigned int i=0;i<size();i++)
	if(basis[i] != T(0)) return false;

      return true;
    }
    
    
    
    // scales basis - not numbers
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::basis_scaling(const T& s) 
    {
      for(U i=U(1);i != U(0);i++){
	T scaling = whiteice::math::pow(s, T(i[0]));
	this->basis[i[0]] *= scaling;
      }

      return (*this);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::basis_scaling(const std::vector<T>& s)   // non-uniform scaling
    {
      if(this->size() != s.size())
	throw uncomparable("Number basises don't match");

      for(U i=U(1);i != U(0);i++){
	T scaling = whiteice::math::pow(s[i[0]], T(i[0]));
	this->basis[i[0]] *= scaling;
      }
      
      return (*this);
    }
    
    
    // measures with s-(dimensional) measure-function
    template <typename T, typename U>
    T superresolution<T,U>::measure(const U& s) const
    {
      if(s[0] == this->size()-1){
	return T(basis[s[0]]);
      }
      else{
	for(U i=U(this->size()-1);i != U(s);i--)
	  if(basis[i[0]] != T(0)){
	    if(basis[i[0]] < T(0)) return T(-INFINITY);
	    else return T(INFINITY);
	  }

	return basis[s[0]];
      }
    }
    
    // FFT and inverse-FFT only works for complex numbers, stores FFT result to number
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::fft()
    {
      whiteice::math::vertex< whiteice::math::blas_complex<double> > b1;
      b1.resize(this->size());

      for(unsigned int i=0;i<b1.size();i++){
	whiteice::math::convert(b1[i], this->basis[i]);
      }

      // calculates FFT
      if(whiteice::math::basic_fft< double>(b1) == false)
	throw illegal_operation("FFT failed");

      for(unsigned int i=0;i<b1.size();i++){
	whiteice::math::convert(this->basis[i], b1[i]);
      }

      return (*this);
    }

    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::inverse_fft()
    {
      whiteice::math::vertex< whiteice::math::blas_complex<double> > b1;
      b1.resize(this->size());

      for(unsigned int i=0;i<b1.size();i++){
	whiteice::math::convert(b1[i], this->basis[i]);
      }

      // calculates inverse-FFT
      if(whiteice::math::basic_ifft< double >(b1) == false)
	throw illegal_operation("FFT failed");

      for(unsigned int i=0;i<b1.size();i++){
	whiteice::math::convert(this->basis[i], b1[i]);
      }

      return (*this);
    }
    
    // calculates circular convolution: (*this) = (*this) * s, stores circular convolution to number
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::circular_convolution(superresolution<T,U>& s)
    {
      (*this) = (*this) * s; // multiplication is circular convolution in our number system
      return (*this);
    }
    
  }
}

namespace whiteice
{
  namespace math
  {
    template <typename T>
    std::ostream& operator<<(std::ostream& ios,
			     const superresolution<T, modular<unsigned int> > & m)
    {
      if(m.size() == 1){
	ios << m[0];
      }
      else if(m.size() > 1){
	ios << "(";

	for(unsigned int i=0;i<m.size();i++){
	  if(i == 0) ios << m[i];
	  else ios << " " << m[i];
	}

	ios << ")";
      }
      
      return ios;
    }

    
    // DO NOT USE BLAS_REAL BUT BLAS_COMPLEX
    template struct superresolution< whiteice::math::blas_real<float>,
				     whiteice::math::modular<unsigned int> >;
    template struct superresolution< whiteice::math::blas_real<double>,
				     whiteice::math::modular<unsigned int> >;
    
    template struct superresolution< whiteice::math::blas_complex<float>,
				     whiteice::math::modular<unsigned int> >;
    template struct superresolution< whiteice::math::blas_complex<double>,
				     whiteice::math::modular<unsigned int> >;


    template std::ostream& operator<< <whiteice::math::blas_real<float> >
    (std::ostream& ios,
     const whiteice::math::superresolution< whiteice::math::blas_real<float>,
     whiteice::math::modular<unsigned int> >&);

    template std::ostream& operator<< <whiteice::math::blas_real<double> >
    (std::ostream& ios,
     const whiteice::math::superresolution< whiteice::math::blas_real<double>,
     whiteice::math::modular<unsigned int> >&);

    template std::ostream& operator<< <whiteice::math::blas_complex<float> >
    (std::ostream& ios,
     const whiteice::math::superresolution< whiteice::math::blas_complex<float>,
     whiteice::math::modular<unsigned int> >&);

    template std::ostream& operator<< <whiteice::math::blas_complex<double> >
    (std::ostream& ios,
     const whiteice::math::superresolution< whiteice::math::blas_complex<double>,
     whiteice::math::modular<unsigned int> >&);
    
  };
};


#endif
