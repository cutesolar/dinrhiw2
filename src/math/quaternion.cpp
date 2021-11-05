/*
 * not the most efficient code / abstract / works alright
 */

#ifndef quaternion_cpp
#define quaternion_cpp

#include <new>
#include <cmath>
#include <iostream>
#include <exception>
#include <stdexcept>

#include "quaternion.h"

namespace whiteice
{
  namespace math
  {
    
    template <typename T>
    quaternion<T>::quaternion() 
    {
      data = new T[4];
      
      for(int i=0;i<4;i++)
	data[i] = 0; /* initialize to be zero */
    }
    
  
    template <typename T>
    quaternion<T>::quaternion(const quaternion<T>& q) 
    {
      data = new T[4];
      
      for(int i=0;i<4;i++)
	data[i] = q.data[i];
    }
    
    
    template <typename T>
    quaternion<T>::quaternion(const T& s) 
    {
      data = new T[4];
      
      data[0] = s;
      data[1] = data[2] = data[3] = 0;
    }
    
    
    template <typename T>
    quaternion<T>::~quaternion() 
    {
      delete[] data; data = 0;
    }
    
    
    /***************************************************/  
    
    
    template <typename T>
    quaternion<T> quaternion<T>::operator+(const quaternion<T>& q1) const
      
    {	
      quaternion<T> q;
      const quaternion<T>& q2 = *this;
      
      for(int i=0;i<4;i++)
	q[i] = q1[i] + q2[i];
      
      return q;
    }
    
  
    template <typename T>
    quaternion<T> quaternion<T>::operator-(const quaternion<T>& q1) const
      
    {
      quaternion<T> q;
      const quaternion<T>& q2 = *this;
      
      for(int i=0;i<4;i++)
	q[i] = q1[i] - q2[i];
      
      return q;	
    }
    
    
    template <typename T>
    quaternion<T> quaternion<T>::operator*(const quaternion<T>& q1) const
      
    {
      quaternion<T> q;
      const quaternion<T>& q2 = *this;
      
      q[0] = q1[0]*q2[0] - (q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]); // real part
      
      // '1st imag. dimension'
      q[1] = q1[0]*q2[1] + q2[0]*q1[1]  + q1[2]*q2[3] - q2[2]*q1[3];
            
      // '2nd imag. dimension'
      q[2] = q1[0]*q2[2] + q2[0]*q1[2]  + q1[3]*q2[1] - q2[3]*q1[1];
      
      // '3rd imag. dimension'
      q[3] = q1[0]*q2[3] + q2[0]*q1[3]  + q1[1]*q2[2] - q2[1]*q1[2];
      
      return q;
    }
  
    
    template <typename T>
    quaternion<T> quaternion<T>::operator/(const quaternion<T>& q1) const
      
    {
      illegal_operation e;
      
      throw e; // not possible with quaternions
    }
    
    
    template <typename T>
    quaternion<T> quaternion<T>::operator!() const 
    {
      /* should be: complex conjugate / adjoint (right word?) operator */
      
      quaternion<T> q;
      
      q[0] = (*this)[0];
      
      for(int i=1;i<4;i++)
	q[i] = -(*this)[i];
      
      return q;
    }
    
    
    template <typename T>
    quaternion<T> quaternion<T>::operator-() const 
    {
      quaternion<T> q;
      
      for(int i=1;i<4;i++)
	q[i] = -(*this)[i];
      
      return q;
    }
    
    
    /***************************************************/
    
    template <typename T>
    quaternion<T>& quaternion<T>::operator+=(const quaternion<T>& q)
      
    {	
      for(int i=0;i<4;i++)
	(*this)[i] += q[i];
      
      return (*this);
    }
    
    
    template <typename T>
    quaternion<T>& quaternion<T>::operator-=(const quaternion<T>& q)
      
    {
      for(int i=0;i<4;i++)
	(*this)[i] -= q[i];
      
      return (*this);	
    }
    
    
    template <typename T>
    quaternion<T>& quaternion<T>::operator*=(const quaternion<T>& q1)
      
    {
      quaternion<T>& q2 = (*this);
      quaternion<T> q;
      
      q[0] = q1[0]*q2[0] - (q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]); // real part
      
      // '1st imag. dimension'
      q[1] = q1[0]*q2[1] + q2[0]*q1[1]  + q1[2]*q2[3] - q2[2]*q1[3];
      
      // '2nd imag. dimension'
      q[2] = q1[0]*q2[2] + q2[0]*q1[2]  + q1[3]*q2[1] - q2[3]*q1[1];
      
      // '3rd imag. dimension'
      q[3] = q1[0]*q2[3] + q2[0]*q1[3]  + q1[1]*q2[2] - q2[1]*q1[2];
      
      (*this) = q;
      
      return (*this);
    }

  
    template <typename T>
    quaternion<T>& quaternion<T>::operator/=(const quaternion<T>& q)
      
    {
      illegal_operation e;
      
      throw e; // not possible with quaternions
    }
    
    
    /***************************************************/
    
    template <typename T>
    quaternion<T>& quaternion<T>::operator=(const quaternion<T>& q)
      
    {
      if(this != &q){
	for(int i=0;i<4;i++)
	  (*this)[i] = q[i];
      }
      
      return *this;
    }
    
    /***************************************************/
    
    
    template <typename T>
    bool quaternion<T>::operator==(const quaternion<T>& q1) const
      
    {
      const quaternion<T>& q2 = *this;
    
      for(int i=0;i<4;i++)
	if( q1[i] != q2[i] ) return false;
      
      return true;
    }
    
    
    template <typename T>
    bool quaternion<T>::operator!=(const quaternion<T>& q1) const
      
    {
      const quaternion<T>& q2 = *this;
      
      for(int i=0;i<4;i++)
	if( q1[i] == q2[i] ) return false;
      
      return true;	
    }
    
  
    template <typename T>
    bool quaternion<T>::operator>=(const quaternion<T>& q1) const
      
    {
      illegal_operation e;
      
      throw e; // not possible with quaternions
    }
    
    
    template <typename T>
    bool quaternion<T>::operator<=(const quaternion<T>& q1) const
      
    {
      illegal_operation e;
      
      throw e; // not possible with quaternions
    }
    
    
    template <typename T>
    bool quaternion<T>::operator< (const quaternion<T>& q1) const
      
    {
      illegal_operation e;
      
      throw e; // not possible with quaternions
    }
  
  
    template <typename T>
    bool quaternion<T>::operator> (const quaternion<T>& q1) const
      
    {
      illegal_operation e;
      
      throw e; // not possible with quaternions
    }
    
    /***************************************************/

    
    template <typename T>
    quaternion<T>& quaternion<T>::operator=(const T& k)
      
    {
      quaternion<T>& q = *this;
      
      q[0] = k; // real part
      q[1] = q[2] = q[3] = T(0);
      
      return *this;
    }
    
    
    template <typename T>
    quaternion<T>& quaternion<T>::operator*=(const T& k) 
    {	
      for(int i=0;i<4;i++)
	data[i] *= k;
      
      return (*this);
    }
    
    
    template <typename T>
    quaternion<T>& quaternion<T>::operator/=(const T& k)
      
    {
      for(int i=0;i<4;i++)
	(*this)[i] /= k;
      
      return (*this);
    }
    
    
    template <typename T>
    quaternion<T> quaternion<T>::operator*(const T& k) const 
    {
      quaternion<T> q;
      
      for(int i=0;i<4;i++)
	q[i] = k * (*this)[i];
      
      return q;
    }
    
    
    template <typename T>
    quaternion<T> quaternion<T>::operator/(const T& k) const
      
    {
      quaternion<T> q;
      
      for(int i=0;i<4;i++)
	q[i] = (*this)[i] / k;
      
      return q;
    }
    
    
    /***************************************************/
    
    
    template <typename T>
    quaternion<T>& quaternion<T>::operator=(const vertex<T>& v)
      
    {
      if(v.size() != 4)
	throw std::invalid_argument("quaternion '='-operator: vertex size != 4");
      
      for(unsigned int i=0;i<4;i++)
	data[i] = v[i];
      
      return *this;
    }
    
    
    
    template <typename T>
    T& quaternion<T>::operator[](const unsigned int index)
      
    {        
      if(index > 3) // quaternion is 4-dimensional
	throw std::out_of_range("index too large");
      
      return data[index];
    }
    
    
    template <typename T>
    const T& quaternion<T>::operator[](const unsigned int index) const
      
    {
      if(index > 3) // quaternion is 4-dimensional
	throw std::out_of_range("index too large");
      
      return data[index];
    }
    
    
    /***************************************************/
    
    
    // creates rotation matrix of quaternion
    template <typename T>
    matrix<T> quaternion<T>::rotation_matrix() const 
    {
      T xx = data[0]*data[0];
      T xy = data[0]*data[1];
      T xz = data[0]*data[2];
      T xw = data[0]*data[3];
      T yy = data[1]*data[1];
      T yz = data[1]*data[2];
      T yw = data[1]*data[3];
      T zz = data[2]*data[2];
      T zw = data[2]*data[3];
      
      // from theory it follows:
      // (also in Matrix and Quaternion FAQ)
      
      matrix<T> r(4,4);
      
      r(0, 0) = T(1.0) - T(2.0) * (yy + zz);
      r(0, 1) = T(2.0) * (xy - zw);
      r(0, 2) = T(2.0) * (xz + yw);
      r(0, 3) = T(0.0);
      r(1, 0) = T(2.0) * T(xy + zw);
      r(1, 1) = T(1.0) - T(2.0) * T(xx + zz);
      r(1, 2) = T(2.0) * T(xz + yw);
      r(1, 3) = T(0.0);
      r(2, 0) = T(2.0) * T(xz - yw);
      r(2, 1) = T(2.0) * T(yz - xw);
      r(2, 2) = T(1.0) - T(2.0) * T(xx + yy);
      r(2, 3) = T(0.0);
      r(3, 0) = T(0.0);
      r(3, 1) = T(0.0);
      r(3, 2) = T(0.0);
      r(3, 3) = T(1.0);
      
      return r;
    }
    
    
    /* calculates quaternion rotator from matrix */
    template <typename T>
    bool quaternion<T>::create_quaternion_rotator(const matrix<T>& m)
      const 
    {
      if(m.xsize() != 4 || m.ysize() != 4)
	return false;
      
      // ref: Matrix and Quaternion FAQ
      // - should work - i haven't chedked the math myself
      
      T tr = m(0, 0) + m(1, 1) + m(2, 2) + 1.0;
      T s;
      
      if(tr > 0){
	s = T(0.5) / sqrt(tr);
	
	data[0] = T(s * (m(2, 1) - m(1, 2)));
	data[1] = T(s * (m(0, 2) - m(2, 0)));
	data[2] = T(s * (m(1, 0) - m(0, 1)));
	data[3] = T(0.25) / s;
	return true;
      }
      
      /* finds row with maximum diagonal element */
      
      unsigned int max_index;
      
      {
	T max = m(0, 0);
	max_index = 0;
	
	for(unsigned int i=1;i<3;i++)
	  if(m(i, i) > max) max_index = i;
      }
      
      s = (T)sqrt( T(1.0) + m(0, 0) - m(1, 1) - m(2, 2) );
      T qx = T(0.5) / s;
      T qy = (T)((m(0, 1) + m(1, 0))/s);
      T qz = (T)((m(0, 2) + m(2, 0))/s);
      T qw = (T)((m(1, 2) + m(2, 1))/s);
      
      if(max_index == 0){
	data[0] = qx;
	data[1] = qy;
	data[2] = qz;
	data[3] = qw;
      }
      else if(max_index == 1){
	data[1] = qx;
	data[0] = qy;
	data[3] = qz;
	data[2] = qw;
      }
      else if(max_index == 2){
	data[2] = qx;
	data[3] = qy;
	data[0] = qz;
	data[1] = qw;
      }
      else
	return false;
      
      
      return true;
    }
    
    
    /*
     * calculates 'rotation quaternion' to be used to rotate around axis
     * also useful in smooth interpolation of (rotation) matrixes
     */
    template <typename T>
    bool quaternion<T>::setup_rotation(T& alpha, const quaternion<T>& axis)
       // setup rotation quaternion
    {
      quaternion<T>& q = *this; // nicer
      
      quaternion<T> A = whiteice::math::abs(axis);
      if(A[0] < T(0.995) || A[0] > T(1.005))
	throw std::invalid_argument("axis must be unit quaternion");
      
      q[0] = whiteice::math::cos ( alpha / T(2.0) );
      q[1] = whiteice::math::sin ( alpha / T(2.0) ) * axis[1];
      q[2] = whiteice::math::sin ( alpha / T(2.0) ) * axis[2];
      q[3] = whiteice::math::sin ( alpha / T(2.0) ) * axis[3];
      
      // CRASH HERE
      
      return true;
    }
    
    
    /*
     * inverse quaternion, trivial if |q| = 1 unit
     */
    template <typename T>
    quaternion<T> quaternion<T>::inv() const 
    {
      quaternion<T> q; // (complex) conjugate
      
      q = !(*this);
      
      // q * q' = |q|^2 => q * q'/|q|^2 = 1 => q's inverse is q'/|q|^2
      // |q|^2:
      
      T sum = T(0.0);
      
      for(int i=0;i<4;i++)
	sum += data[i] * data[i];
      
      if(sum == T(0.0)) // zero don't have inverse
	throw std::invalid_argument("tried to computer inverse of zero");
      
      
      for(int i=0;i<4;i++)
	q[i] /= sum;
      
      return q;
    }

    
    // calculates absolute value of quaternion
    // this is (decided to be) |q|^2
    template <typename T>
    quaternion<T>& quaternion<T>::abs() 
    {
      T sum = T(0.0);
      
      for(int i=0;i<4;i++){
	sum += conj((*this)[i]) * (*this)[i];
      }
      
      sum = whiteice::math::sqrt( sum );
      
      for(unsigned int i=1;i<4;i++)
	(*this)[i] = T(0.0);
      
      (*this)[0] = sum;
      
      return (*this);
    }
    
    
    // normalizes length
    template <typename T>
    bool quaternion<T>::normalize() 
    {
      quaternion<T> A = (*this);
      A.abs();
      
      if(A[0] == T(0.0)) // not possible
	return false;
      
      for(int i=0;i<4;i++)
	(*this)[i] /= A[0];
      
      return true;
    }
    
    
    /***************************************************/
    
    
    template <typename T>
    std::ostream& operator<<(std::ostream& ios, const quaternion<T>& q)
    {
      ios << "[";
      
      /* printing style 1.0 + 2.0i - 2.1j - 1.1k */
      
      if(q[0] >= T(0)) ios << "+";      
      ios << q[0] << " ";
      if(q[1] >= T(0)) ios << "+";      
      ios << q[1] << "i ";
      if(q[2] >= T(0)) ios << "+";      
      ios << q[2] << "j ";
      if(q[3] >= T(0)) ios << "+";      
      ios << q[3] << "k ";
      
      ios << "]";
      
      return ios;
    }
    
    
    
    // explicit template instantations
    
    template class quaternion<float>;
    template class quaternion<double>;
    template class quaternion< blas_real<float> >;
    template class quaternion< blas_real<double> >;
    
    template std::ostream& operator<< <float>(std::ostream& ios, const quaternion<float>& q);
    template std::ostream& operator<< <double>(std::ostream& ios, const quaternion<double>& q);
    template std::ostream& operator<< <blas_real<float> >(std::ostream& ios, const quaternion<blas_real<float> >& q);
    template std::ostream& operator<< <blas_real<double> >(std::ostream& ios, const quaternion<blas_real<double> >& q);
    
  }
}
  
#endif


