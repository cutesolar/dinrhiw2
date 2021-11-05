/*
 * quaternion class
 */
 
#ifndef quaternion_h
#define quaternion_h

#include "dinrhiw_blas.h"
#include "number.h"
#include <iostream>
#include <exception>
#include <stdexcept>
#include <memory>


namespace whiteice
{
  namespace math
  {

    template <typename T>
      class vertex;
    
    template <typename T>
      class matrix;
    
    
    template <typename T = blas_real<float> >
      class quaternion : public whiteice::number<quaternion<T>, T, T, unsigned int>
    {
      public:
      
      quaternion() ;
      
      quaternion(const quaternion<T>&) ;
      quaternion(const T& s) ;
      virtual ~quaternion() ;
      
      quaternion<T> operator+(const quaternion<T>&) const ;
      quaternion<T> operator-(const quaternion<T>&) const ;
      quaternion<T> operator*(const quaternion<T>&) const ;
      quaternion<T> operator/(const quaternion<T>&) const ;
      
      quaternion<T> operator!() const ;
      quaternion<T> operator-() const ;
      
      quaternion<T>& operator+=(const quaternion<T>&) ;
      quaternion<T>& operator-=(const quaternion<T>&) ;
      quaternion<T>& operator*=(const quaternion<T>&) ;
      quaternion<T>& operator/=(const quaternion<T>&) ;
      
      quaternion<T>& operator=(const quaternion<T>&) ;
      
      bool operator==(const quaternion<T>&) const ;
      bool operator!=(const quaternion<T>&) const ;
      bool operator>=(const quaternion<T>&) const ;
      bool operator<=(const quaternion<T>&) const ;
      bool operator< (const quaternion<T>&) const ;
      bool operator> (const quaternion<T>&) const ;
      
      /* interaction with 'primal' type */
      quaternion<T>& operator= (const T&) ;
      quaternion<T>& operator*=(const T&) ;
      quaternion<T>& operator/=(const T&) ;
      quaternion<T>  operator* (const T&) const ;
      quaternion<T>  operator/ (const T&) const ;
      
      quaternion<T>& operator= (const vertex<T>&) ;
      
      T& operator[](const unsigned int index) ;
      const T& operator[](const unsigned int index) const ;
      
      
      /* creates rotation matrix of quaternion */
      matrix<T> rotation_matrix() const ;
      
      /* calculates quaternion rotator from matrix */
      bool create_quaternion_rotator(const matrix<T>& m) const ;
      
      // setups rotation quaternion
      bool setup_rotation(T& alpha, const quaternion<T>& axis) ;
      
      // inverse quaternion
      quaternion<T> inv() const ;
      
      quaternion<T>& abs() ;
      bool normalize() ;
      
      bool comparable() { return false; }
      
      private:
      
      T* data; // for now..
      
    };
    
    
    template <typename T>
      std::ostream& operator<<(std::ostream& ios, const quaternion<T>& q);
    

    
    extern template class quaternion<float>;
    extern template class quaternion<double>;
    extern template class quaternion< blas_real<float> >;
    extern template class quaternion< blas_real<double> >;
    
    extern template std::ostream& operator<< <float>(std::ostream& ios, const quaternion<float>& q);
    extern template std::ostream& operator<< <double>(std::ostream& ios, const quaternion<double>& q);
    extern template std::ostream& operator<< <blas_real<float> >(std::ostream& ios, const quaternion<blas_real<float> >& q);
    extern template std::ostream& operator<< <blas_real<double> >(std::ostream& ios, const quaternion<blas_real<double> >& q);
    
  }
}

  
#include "matrix.h"
#include "vertex.h"
// #include "quaternion.cpp"


#endif

