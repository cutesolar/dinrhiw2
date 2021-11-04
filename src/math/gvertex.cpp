

#ifndef gvertex_cpp
#define gvertex_cpp

#include "ownexception.h"
#include "blade_math.h"
#include "number.h"
#include "gvertex.h"

#include <stdexcept>
#include <iostream>
#include <exception>
#include <vector>
#include <math.h>


namespace whiteice
{
  namespace math
  {
  
    // gvertex ctor, i is dimension of vector
    template <typename T, typename S>
    gvertex<T,S>::gvertex(unsigned int i) 
    {
      c.resize(i);
    }
    
    
    // gvertex ctor - makes copy of v
    template <typename T, typename S>
    gvertex<T,S>::gvertex(const gvertex<T,S>& v) 
    {
      c.resize(v.c.size());
      for(unsigned int i=0;i<v.c.size();i++) c[i] = v.c[i];
    }
    
    
    // gvertex ctor - makes copy of v
    template <typename T, typename S>
    gvertex<T,S>::gvertex(const std::vector<T>& v) 
    {
      c.resize(v.size());
      for(unsigned int i=0;i<v.size();i++)
	c[i] = v[i];
    }
    
    
    // gvertex dtor
    template <typename T, typename S>
    gvertex<T,S>::~gvertex() { }
    
    /***************************************************/
    
    
    // returns gvertex dimension/size
    template <typename T, typename S>
    unsigned int gvertex<T,S>::size() const { return c.size(); }
    
    
    // sets gvertex dimension/size, fills new dimensios with zero
    template <typename T, typename S>
    unsigned int gvertex<T,S>::resize(unsigned int d) 
    {
      unsigned int s = c.size();
      c.resize(d);
      
      if(s < c.size())
	for(;s<c.size();s++) c[s] = T(0);
      
      return c.size();
    }
    
    
    // returns 2nd norm of gvertex
    template <typename T, typename S>
    T gvertex<T,S>::norm() const 
    {
      T sum = 0;
      
      for(unsigned int i=0;i<c.size();i++)
	sum += whiteice::math::conj(c[i])*c[i];
      
      sum = T(whiteice::math::sqrt(sum));
      return sum;
    }

    
    template <typename T, typename S>
    T gvertex<T,S>::norm(unsigned int i, unsigned int j) const 
    {
      T sum = 0;
      
      if(j >= c.size()) j = c.size();
      
      for(unsigned int k=i;k<j;k++)
	sum += whiteice::math::conj(c[k])*c[k];
      
      sum = T(whiteice::math::sqrt(sum));
      return sum;
    }
    
    
    // sets length to zero, zero length -> retuns false
    template <typename T, typename S>
    bool gvertex<T,S>::normalize() 
    {
      T l = norm();
      if(l == T(0)) return false;
      
      for(unsigned int i=0;i<c.size();i++)
	c[i] /= l;
      
      return true;
    }

    
    // calculates sum of gvertexes
    template <typename T, typename S>
    gvertex<T,S> gvertex<T,S>::operator+(const gvertex<T,S>& v) const
      
    {
      if(v.c.size() != c.size())
	throw illegal_operation("vector op: vector dim. mismatch");
      
      gvertex<T,S> r(v.c.size());    
      for(unsigned int i=0;i<v.c.size();i++) r.c[i] = v.c[i] + c[i];
      
      return r;
    }
    
    
    // substracts two gvertexes
    template <typename T, typename S>
    gvertex<T,S> gvertex<T,S>::operator-(const gvertex<T,S>& v) const
      
    {
      if(v.c.size() != c.size())
	throw illegal_operation("vector op: vector dim. mismatch");
      
      gvertex<T,S> r(v.c.size());    
      for(unsigned int i=0;i<v.c.size();i++) r.c[i] = c[i] - v.c[i];
      
      return r;
    }
    
    
    // calculates innerproduct - returns 1-dimension gvertex
    // or calculates scalar product if other vector is one dimensional
    template <typename T, typename S>
    gvertex<T,S> gvertex<T,S>::operator*(const gvertex<T,S>& v) const
      
    {
      if(v.c.size() != c.size()){
	if(v.c.size() == 1)
	{
	  gvertex<T,S> r(c.size());
	  for(unsigned int i=0;i<c.size();i++)
	    r[i] = v.c[0]*c[i];
	  
	  return r;
	}
	else if(c.size() == 1)
	{
	  gvertex<T,S> r(v.c.size());
	  for(unsigned int i=0;i<v.c.size();i++)
	    r[i] = c[0]*v.c[i];
	  
	  return r;
	}
	else
	  throw illegal_operation("vector op: vector dim. mismatch");    
      }
      
      gvertex<T,S> r(1);
      r = S(0.0);
      
      for(unsigned int i=0;i<v.c.size();i++)
	r[0] += c[i]*v.c[i];
      
      return r;
    }
    
    
    // no divide operation
    template <typename T, typename S>
    gvertex<T,S> gvertex<T,S>::operator/(const gvertex<T,S>& v) const
      
    {
      throw illegal_operation("gvertex(): '/'-operator not available");
    }
    
    
    // no "!" operation
    template <typename T, typename S>
    gvertex<T,S> gvertex<T,S>::operator!() const {
      throw illegal_operation("gvertex(): '!'-operation not available");
    }

    
    // changes gvertex sign
    template <typename T, typename S>
    gvertex<T,S> gvertex<T,S>::operator-() const
      
    {
      gvertex<T,S> r(c.size());
      for(unsigned int i=0;i<c.size();i++) r.c[i] = -c[i];    
      return r;
    }
    
    // calculates cross product
    template <typename T, typename S>
    gvertex<T,S> gvertex<T,S>::operator^(const gvertex<T,S>& v) const
      
    {
      if(c.size() != 3 || this->size() != 3)      
	throw illegal_operation("crossproduct: vector dimension != 3");
      
      gvertex<T,S> r(3);
      
      r[0] = c[1]*v.c[2] - c[2]*v.c[1];
      r[1] = c[2]*v.c[0] - c[0]*v.c[2];
      r[2] = c[0]*v.c[1] - c[1]*v.c[0];
      
      return r;    
    }
    
    /***************************************************/
    
    // adds gvertexes
    template <typename T, typename S>
    gvertex<T,S>& gvertex<T,S>::operator+=(const gvertex<T,S>& v)
      
    {
      if(v.c.size() != c.size())
	throw illegal_operation("vector op: vector dim. mismatch");
      
      for(unsigned int i=0;i<v.size();i++) c[i] += v.c[i];
      return *this;
    }
    
    // subtracts gvertexes
    template <typename T, typename S>
    gvertex<T,S>& gvertex<T,S>::operator-=(const gvertex<T,S>& v)
      
    {
      if(v.c.size() != c.size())
	throw illegal_operation("vector op: vector dim. mismatch");
      
      for(unsigned int i=0;i<v.c.size();i++) c[i] -= v.c[i];
      return *this;
    }
    
    // calculates inner product
    template <typename T, typename S>
    gvertex<T,S>& gvertex<T,S>::operator*=(const gvertex<T,S>& v)
      
    {
      if(v.c.size() != c.size())
	throw illegal_operation("vector op: vector dim. mismatch");    
      
      gvertex<T,S> r(1);
      r[0] = S(0);
      
      for(unsigned int i=0;i<v.c.size();i++)
	r[0] += c[i]*v.c[i];
      
      *this = r;
      
      return *this;
    }
    
    // dividing not available
    template <typename T, typename S>
    gvertex<T,S>& gvertex<T,S>::operator/=(const gvertex<T,S>& v)
      {
      throw illegal_operation("gvertex(): '/='-operator not available");
    }
    
    // assigns given gvertex value to this gvertex
    template <typename T, typename S>
    gvertex<T,S>& gvertex<T,S>::operator=(const gvertex<T,S>& v)
      
    {
      if(this != &v){
	if(v.c.size() != c.size()) c.resize(v.c.size());
	
	c.resize(v.c.size());
	for(unsigned int i=0;i<v.c.size();i++) c[i] = v.c[i];      
      }
	
      return *this;
    }
    
    
    /***************************************************/
    
    // compares two gvertexes for equality
    template <typename T, typename S>
    bool gvertex<T,S>::operator==(const gvertex<T,S>& v) const
      
    {
      if(v.c.size() != c.size())
	throw uncomparable("gvertex compare: dimension mismatch");
      
      for(unsigned int i=0;i<v.c.size();i++)
	if(c[i] != v.c[i]) return false;	
      
      return true;
    }
    
    // compares two gvertexes for non-equality
    template <typename T, typename S>
    bool gvertex<T,S>::operator!=(const gvertex<T,S>& v) const
      
    {
      if(v.c.size() != c.size())
	throw uncomparable("gvertex compare: dimension mismatch");
      
      for(unsigned int i=0;i<v.c.size();i++)
	if(c[i] != v.c[i]) return true;
      
      return false;    
    }
    
    // not defined
    template <typename T, typename S>
    bool gvertex<T,S>::operator>=(const gvertex<T,S>& v) const {
      throw uncomparable("gvertex(): '>='-operator defined");
    }
    
    // not defined
    template <typename T, typename S>
    bool gvertex<T,S>::operator<=(const gvertex<T,S>& v) const {
      throw uncomparable("gvertex(): '<='-operator not defined");
    }
    
    // not defined
    template <typename T, typename S>
    bool gvertex<T,S>::operator< (const gvertex<T,S>& v) const {
      throw uncomparable("gvertex(): '<'-operator not defined");
    }
    
    // not defined
    template <typename T, typename S>
    bool gvertex<T,S>::operator> (const gvertex<T,S>& v) const {
      throw uncomparable("gvertex(): '>'-operator not defined");
    }
    
    
    
    // calculates absolute value of each gvertex element
    template <typename T, typename S>
    gvertex<T,S>& gvertex<T,S>::abs() 
    {
      for(unsigned int i=0;i<c.size();i++)
	c[i] = whiteice::math::abs(c[i]);
      
      return (*this);
    }
    
    template <typename T, typename S>
    gvertex<T,S>& gvertex<T,S>::conj() 
    {
      for(unsigned int i=0;i<c.size();i++)
	c[i] = whiteice::math::conj(c[i]);
      
      return (*this);
    }
    
    
    /***************************************************/
    // scalars
    
    
    /* sets all elements of gvertex = given scalar */
    template <typename T, typename S>
    gvertex<T,S>& gvertex<T,S>::operator=(const S& s)
      
    {
      for(unsigned int i=0;i<c.size();i++) c[i] = s;
      return *this;
    }
    
    
    
    // multiples gvertex with scalar */
    template <typename T, typename S>
    gvertex<T,S>  gvertex<T,S>::operator*(const S& s) const 
    {
      gvertex<T,S> r(c.size());
      
      for(unsigned int i=0;i<c.size();i++) r.c[i] = c[i]*s;
      return r;
    }
    
    // multiples gvertex with scalar */
    template <typename T, typename S>
    gvertex<T,S>& gvertex<T,S>::operator*=(const S& s) 
    {
      for(unsigned int i=0;i<c.size();i++) c[i] = c[i]*s;          
      return *this;
    }
    
    
    // multiples gvertex with scalar */
    template <typename T, typename S>
    gvertex<T,S>  gvertex<T,S>::operator/(const S& s) const
      
    {
      gvertex<T,S> r(c.size());
      
      for(unsigned int i=0;i<c.size();i++){
	r.c[i] = c[i]/s;
      }
      
      return r;
    }
    
    // multiples gvertex with scalar */
    template <typename T, typename S>
    gvertex<T,S>& gvertex<T,S>::operator/=(const S& s) 
    {
      for(unsigned int i=0;i<c.size();i++){
	c[i] = c[i]/s;
      }
      
      return *this;
    }
    
    
    
    // scalar times gvertex
    template <typename T, typename S>
    gvertex<T,S> operator*(const S& s, const gvertex<T,S>& v)
    {
      gvertex<T,S> r(v.size());
      
      for(unsigned int i=0;i<v.size();i++)
	r[i] = v[i]*s;
      
      return r;
    }
    
    
    // multiplies gmatrix from left
    template <typename T, typename S>
    gvertex<T,S> gvertex<T,S>::operator* (const gmatrix<T,S>& m) const
      
    {
      if(c.size() != m.ysize())
	throw std::invalid_argument("multiply: gvertex/gmatrix dim. mismatch");
      
      gvertex<T,S> r;
      
      if(m.ysize() > 0) r.resize(m.xsize());
      else { r.resize(0); return r; }
      
      for(unsigned int j=0;j<m.xsize();j++){
	r[j] = (T)0;
	for(unsigned int i=0;i<m.ysize();i++)
	  r[j] += c[i]*m[i][j];
      }
      
      return r;      
    }
    
    /***************************************************/
    
    
    template <typename T, typename S>
    T& gvertex<T,S>::operator[](const unsigned int index)
      
    {
      if(index >= c.size())
	throw std::out_of_range("gvertex[]: index out of range");
      
      return c[index];
    }
    
    
    template <typename T, typename S>
    const T& gvertex<T,S>::operator[](const unsigned int index) const
      
    {
      if(index >= c.size())
	throw std::out_of_range("gvertex[]: index out of range");
      
      return c[index];    
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S> gvertex<T,S>::outerproduct(const gvertex<T,S>& v) const
      
    {
      return outerproduct(*this, v);
    }
    
    
    /* outer product of N length gvertexes */
    template <typename T, typename S>
    gmatrix<T,S> gvertex<T,S>::outerproduct(const gvertex<T,S>& v0,
					  const gvertex<T,S>& v1) const
      
    {
      gmatrix<T,S> m(v0.c.size(), v1.c.size());
      
      for(unsigned int i=0;i<v0.c.size();i++)
	for(unsigned int j=0;j<v1.c.size();j++)
	  m[i][j] = v0.c[i]*v1.c[j];
      
      return m;
    }
    

    template <typename T, typename S> // iterators
    typename gvertex<T,S>::iterator gvertex<T,S>::begin() {
      return c.begin();
    }
    
    
    template <typename T, typename S>
    typename gvertex<T,S>::iterator gvertex<T,S>::end() {
      return c.end();
    }
  
  
    template <typename T, typename S> // iterators
    typename gvertex<T,S>::const_iterator gvertex<T,S>::begin() const {
      return c.begin();
    }
    
    template <typename T, typename S>
    typename gvertex<T,S>::const_iterator gvertex<T,S>::end() const {
      return c.end();
    }
    
    
    /***************************************************/
    
    template <typename T, typename S>
    std::ostream& operator<<(std::ostream& ios,
			     const whiteice::math::gvertex<T,S>& v)
    {
      if(v.size() == 1){
	ios << v[0];
      }
      else if(v.size() > 1){
	
	ios << "[";
	
	for(unsigned int i=0;i<v.size();i++){
	  ios << " " << v[i];
	}
	
	ios << " ]";
      }
      
      return ios;
    }
    
    
    // tries to convert gvertex of type S to gvertex of type T (B = A)
    template <typename T, typename S>
    bool convert(gvertex<T>& B, const gvertex<S>& A) 
    {
      try{
	if(B.resize(A.size()) == false)
	  return false;
	
	for(unsigned int j=0;j<B.size();j++)
	  B[j] = static_cast<T>(A[j]);
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
  }
}





#endif
