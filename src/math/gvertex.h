/* 
 * generic vertex
 * (math vector)
 *
 */

#ifndef gvertex_h
#define gvertex_h

#include "ownexception.h"
#include "number.h"

#include <stdexcept>
#include <exception>
#include <iostream>
#include <vector>


namespace whiteice
{
  namespace math
  {
    
    template <typename T, typename S>
      class gmatrix;

    // T is typename
    // S must be scalar which
    //   can operate with T
    
    
    template <typename T=float, typename S=T>
      class gvertex : public whiteice::number<gvertex<T,S>, S, T, unsigned int>
    {
      public:
      
      explicit gvertex(unsigned int i=4) ;
      gvertex(const gvertex<T,S>& v) ;
      gvertex(const std::vector<T>& v) ;
      virtual ~gvertex() ;
      
      typedef typename std::vector<T>::iterator iterator;
      typedef typename std::vector<T>::const_iterator const_iterator;
      
      unsigned int size() const ;
      unsigned int resize(unsigned int d) ;
      
      // TODO: move norm() to generic number class
      T norm() const ;
      
      // calculates partial norm for gvertex(i:j)
      T norm(unsigned int i, unsigned int j) const ;
      
      bool normalize() ; // length = 1

      gvertex<T,S> operator+(const gvertex<T,S>& v) const ;
      gvertex<T,S> operator-(const gvertex<T,S>& v) const ;

      // inner product - returns gvertex with size 1
      gvertex<T,S> operator*(const gvertex<T,S>& v) const ;
      
      gvertex<T,S> operator/(const gvertex<T,S>& v) const ;
      gvertex<T,S> operator!() const ;
      gvertex<T,S> operator-() const ;

      // cross product
      gvertex<T,S> operator^(const gvertex<T,S>& v) const ;
      
      gvertex<T,S>& operator+=(const gvertex<T,S>& v) ;
      gvertex<T,S>& operator-=(const gvertex<T,S>& v) ;
      gvertex<T,S>& operator*=(const gvertex<T,S>& v) ;
      gvertex<T,S>& operator/=(const gvertex<T,S>& v) ;
      
      gvertex<T,S>& operator=(const gvertex<T,S>& v) ;      
      
      bool operator==(const gvertex<T,S>& v) const ;
      bool operator!=(const gvertex<T,S>& v) const ;
      bool operator>=(const gvertex<T,S>& v) const ;
      bool operator<=(const gvertex<T,S>& v) const ;
      bool operator< (const gvertex<T,S>& v) const ;
      bool operator> (const gvertex<T,S>& v) const ;

      gvertex<T,S>& abs() ;
      gvertex<T,S>& conj() ;
      
      /* scalars */
      gvertex<T,S>& operator= (const S& s) ;
      gvertex<T,S>  operator* (const S& s) const ;
      gvertex<T,S>  operator/ (const S& s) const ;
      gvertex<T,S>& operator*=(const S& s) ;
      gvertex<T,S>& operator/=(const S& s) ;
      
      // multi from right
      template <typename TT, typename SS>
      friend gvertex<TT,SS> operator*(const SS& s, const gvertex<TT,SS>& v);
      
      // multiply from left
      gvertex<T,S>  operator* (const gmatrix<T,S>& m)
        const ;

      // outer product
      gmatrix<T,S> outerproduct(const gvertex<T,S>& v) const ;
      gmatrix<T,S> outerproduct(const gvertex<T,S>& v0,
			       const gvertex<T,S>& v1) const ;
      
      T& operator[](const unsigned int index) ;
      const T& operator[](const unsigned int index) const ;
      
      iterator begin() ; // iterators
      iterator end() ;
      const_iterator begin() const ; // iterators
      const_iterator end() const ;
      
      bool comparable() { return false; }
      
      // TODO: add dummy    static const gvertex<T,S> null; for empty/null gvertex
      
      private:
      
      std::vector<T> c;
      
    };
    
    
    template <typename T, typename S>
      std::ostream& operator<<(std::ostream& ios,
			       const whiteice::math::gvertex<T,S>&);
    
    
    // tries to convert gvertex of type S to gvertex of type T (B = A)
    template <typename T, typename S>
      bool convert(gvertex<T>& B, const gvertex<S>& A) ;
    
  }
}



#include "gmatrix.h"
#include "gvertex.cpp"

#endif
