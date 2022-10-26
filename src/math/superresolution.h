/*
 * superresolutional / dimensional numbers with basis
 * values of type T and exponents/dimensions of type U
 */
#ifndef superresolution_h
#define superresolution_h

namespace whiteice
{
  namespace math
  {
    template <typename T, typename S>
    struct superresolution;
    
  };
};

#include "number.h"
#include "modular.h"
#include "blade_math.h"
#include <vector>


#define NO_PACKED 1


namespace whiteice
{
  namespace math
  {
    /* superresolutional numbers
     * made out of field T with exponent field U
     */
    
    //class superresolution : public number<superresolution<T,U>, T, T, U>
    template <typename T = whiteice::math::blas_real<float>,
	      typename U = whiteice::math::modular<unsigned int> >
    struct superresolution
    {
      superresolution();
      superresolution(const T value);
      // superresolution(const U& resolution);
      superresolution(const superresolution<T,U>& s);
      superresolution(const std::vector<T>& values);
      ~superresolution();
      
      // sets superresolution componenets to be all zero
      void zeros();
      
      // sets supperresolution components to be all ones
      void ones();
      
      // operators
      superresolution<T,U> operator+(const superresolution<T,U>&) const ;
      superresolution<T,U> operator-(const superresolution<T,U>&) const ;
      superresolution<T,U> operator*(const superresolution<T,U>&) const ;
      superresolution<T,U> operator/(const superresolution<T,U>&) const ;

      // complex conjugate (?)
      superresolution<T,U> operator!() const;
      
      superresolution<T,U>& conj();
      
      superresolution<T,U> operator-() const;
      
      
      superresolution<T,U>& operator+=(const superresolution<T,U>&) ;
      superresolution<T,U>& operator-=(const superresolution<T,U>&) ;
      superresolution<T,U>& operator*=(const superresolution<T,U>&) ;
      superresolution<T,U>& operator/=(const superresolution<T,U>&) ;
      
      superresolution<T,U>& operator=(const superresolution<T,U>&) ;
      
      bool operator==(const superresolution<T,U>&) const ;
      bool operator!=(const superresolution<T,U>&) const ;
      
      // WARN! comparision operators assume infinite basis and not cyclic exponent basis
      // this means comparision operators don't work properly b > c => a*b > a*c, a > 0
      // don't hold etc. Comparision work with static numbers but not with operations
      // changing multiple basis numbers
      bool operator>=(const superresolution<T,U>&) const ;
      bool operator<=(const superresolution<T,U>&) const ;
      bool operator< (const superresolution<T,U>&) const ;
      bool operator> (const superresolution<T,U>&) const ;
      
      // scalar operation
      superresolution<T,U>& operator= (const T& s) ;
      superresolution<T,U>  operator+ (const T& s) const ;
      superresolution<T,U>  operator- (const T& s) const ;
      superresolution<T,U>  operator* (const T& s) const ;
      superresolution<T,U>  operator/ (const T& s) const ;
      superresolution<T,U>& operator*=(const T& s) ;
      superresolution<T,U>& operator/=(const T& s) ;
      
      superresolution<T,U>& operator=(const T value) ;
      
      superresolution<T,U>& abs();
      superresolution<T,U>& zero();
      bool iszero() const;
      
      
      inline T& operator[](int index)
      { 
	index %= DEFAULT_MODULAR_BASIS;
	if(index < 0) index += DEFAULT_MODULAR_BASIS;
	
	return basis[index];
      }
      
      
      inline const T& operator[](int index) const
      {
	index %= DEFAULT_MODULAR_BASIS;
	if(index < 0) index += DEFAULT_MODULAR_BASIS;
	
	return basis[index];
      }
      
      inline unsigned int size() const {
	return DEFAULT_MODULAR_BASIS;
      }
      
      inline T& first(){ return basis[0]; }
      
      inline const T& first() const{ return basis[0]; }
      
      inline T& first(const T value){
	basis[0] = value;
	return basis[0];
      }
      
      // superresolution operations
      superresolution<T,U>& basis_scaling(const T& s) ; // uniform
      superresolution<T,U>& basis_scaling(const std::vector<T>& s) ; // non-uniform scaling
      T measure(const U& s) const; // measures with s-(dimensional) measure-function

      // per-element operations
      
      // FFT and inverse-FFT only works for complex numbers, stores FFT result to number
      superresolution<T,U>& fft();
      superresolution<T,U>& inverse_fft();
      
      // calculates circular convolution: (*this) = (*this) * s, stores circular convolution to number
      superresolution<T,U>& circular_convolution(superresolution<T,U>& s);
      
      inline bool comparable(){
	return true; // NOT REALLY COMPARABLE IF OVERFLOW HAPPENS (CIRCULAR CONVOLUTION)
      }
      
      // superresolution basis
      // (todo: switch to 'sparse' representation with (U,T) pairs)
      // and ordered by U (use own avltree implementation)
      
#ifdef NO_PACKED
      T basis[DEFAULT_MODULAR_BASIS];
#else
      T basis[DEFAULT_MODULAR_BASIS] __attribute__ ((packed));
#endif
      
      // std::vector<T> basis;
      
    } 
#ifndef NO_PACKED
      __attribute__ ((packed))
#endif
    ;
    
    
  }
}


// #include "superresolution.cpp"

namespace whiteice{

  namespace math
  {
    template <typename T>
    std::ostream& operator<<(std::ostream& ios, const superresolution<T, modular<unsigned int> > & m);

    
    template <typename T, typename U>
    superresolution<T,U> componentwise_multi(const superresolution<T,U>& a, const superresolution<T,U>& b)
    {
      // calculates a .* b

      superresolution<T,U> result;

      for(unsigned int i=0;i<DEFAULT_MODULAR_BASIS;i++)
	result.basis[i] = a.basis[i]*b.basis[i];

      return result;
    }
			     
    
    extern template struct superresolution< whiteice::math::blas_real<float>,
					    whiteice::math::modular<unsigned int> >;
    extern template struct superresolution< whiteice::math::blas_real<double>,
					    whiteice::math::modular<unsigned int> >;
    
    extern template struct superresolution< whiteice::math::blas_complex<float>,
					    whiteice::math::modular<unsigned int> >;
    extern template struct superresolution< whiteice::math::blas_complex<double>,
					    whiteice::math::modular<unsigned int> >;

    extern template std::ostream& operator<< <whiteice::math::blas_real<float> >
    (std::ostream& ios,
     const whiteice::math::superresolution< whiteice::math::blas_real<float>,
     whiteice::math::modular<unsigned int> >&);

    extern template std::ostream& operator<< <whiteice::math::blas_real<double> >
    (std::ostream& ios,
     const whiteice::math::superresolution< whiteice::math::blas_real<double>,
     whiteice::math::modular<unsigned int> >&);
    
    extern template std::ostream& operator<< <whiteice::math::blas_complex<float> >
    (std::ostream& ios,
     const whiteice::math::superresolution< whiteice::math::blas_complex<float>,
     whiteice::math::modular<unsigned int> >&);

    extern template std::ostream& operator<< <whiteice::math::blas_complex<double> >
    (std::ostream& ios,
     const whiteice::math::superresolution< whiteice::math::blas_complex<double>,
     whiteice::math::modular<unsigned int> >&);

  }
  
};


#endif
