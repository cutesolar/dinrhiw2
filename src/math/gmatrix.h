/*
 * generic matrix class
 */

#ifndef gmatrix_h
#define gmatrix_h

#include "ownexception.h"
#include "number.h"

#include <stdexcept>
#include <exception>


namespace whiteice
{
  namespace math
  {

    template <typename T, typename S>
      class gvertex;
    
    // S must be scalar which can operate with T
    // T is type of the scalar
    
    template <typename T=float, typename S=T>
      class gmatrix : public whiteice::number< gmatrix<T,S>, S, gvertex<T,S>, unsigned int >
    {
      public:
      
      explicit gmatrix(const unsigned int size_y = 4,
		      const unsigned int size_x = 4);
      gmatrix(const gmatrix<T,S>& M);
      gmatrix(const gvertex<T,S>& diagonal);
      virtual ~gmatrix();
      
      
      gmatrix<T,S> operator+(const gmatrix<T,S>&) const ;
      gmatrix<T,S> operator-(const gmatrix<T,S>&) const ;
      gmatrix<T,S> operator*(const gmatrix<T,S>&) const ;
      gmatrix<T,S> operator/(const gmatrix<T,S>&) const ;
      gmatrix<T,S> operator!() const ;
      gmatrix<T,S> operator-() const ;
      
      gmatrix<T,S>& operator+=(const gmatrix<T,S>&) ;
      gmatrix<T,S>& operator-=(const gmatrix<T,S>&) ;
      gmatrix<T,S>& operator*=(const gmatrix<T,S>&) ;
      gmatrix<T,S>& operator/=(const gmatrix<T,S>&) ;
      
      gmatrix<T,S>& operator=(const gmatrix<T,S>&) ;
                  
      bool operator==(const gmatrix<T,S>&) const ;
      bool operator!=(const gmatrix<T,S>&) const ;
      bool operator>=(const gmatrix<T,S>&) const ;
      bool operator<=(const gmatrix<T,S>&) const ;
      bool operator< (const gmatrix<T,S>&) const ;
      bool operator> (const gmatrix<T,S>&) const ;

      // scalars and gmatrix interaction
      gmatrix<T,S>& operator= (const S&) ;

      gmatrix<T,S>  operator* (const S&) const ;
      
      template <typename TT, typename SS>
      friend gmatrix<TT,SS> operator*(const SS&, const gmatrix<TT,SS>&)
        ;
      
      gmatrix<T,S>  operator/ (const S&) const ;

      gmatrix<T,S>& operator*=(const S&) ;
      gmatrix<T,S>& operator/=(const S&) ;

      gvertex<T,S> operator*(const gvertex<T,S>&) const ;      
      
      gvertex<T,S>& operator[](const unsigned int index)
        ;
      
      const gvertex<T,S>& operator[](const unsigned int index)
        const ;
      
      T& operator()(unsigned int y, unsigned int x) ;
      const T& operator()(unsigned int y, unsigned int x) const ;
      
      
      gmatrix<T,S>& zero(); // zeroes gmatrix
      gmatrix<T,S>& identity();
      
      gmatrix<T,S>& crossproduct(const gvertex<T,S>& v) ;
      
      // euclidean rotation
      gmatrix<T,S>& rotation(const S& xr, const S& yr, const S& zr) ;
      
      // translation
      gmatrix<T,S>& translation(const S& dx, const S& dy, const S& dz) ;
      
      
      gmatrix<T,S>& abs() ;
      gmatrix<T,S>& conj() ;
      gmatrix<T,S>& transpose() ;
      gmatrix<T,S>& hermite() ;
      
      gmatrix<T,S>& inv() ; // inverse
      
      T det() const ; // determinate
      T trace() const ;
      
      unsigned int size() const ;
      unsigned int ysize() const ;      
      unsigned int xsize() const ;
      
      bool resize_x(unsigned int d) ;
      bool resize_y(unsigned int d) ;
      bool resize(unsigned int y, unsigned int x) ;
      
      
      // normalizes each row vector of matrix to have unit length
      void normalize() ;
      
      
      bool comparable() { return false; }
      
      // TODO: add dummy    static const gmatrix<T,S> null; for empty/null gvertex
      
      private:
      
      gvertex< gvertex<T, S>, S > data;
    };
    
    
    template <typename T, typename S>
      std::ostream& operator<<(std::ostream& ios, const gmatrix<T,S>& M);
    
    // tries to convert gmatrix of type L to gmatrix of type T (B = A)
    template <typename T, typename S, typename L, typename M>
      bool convert(gmatrix<T,S>& B, const gmatrix<L,M>& A) ;

  }
}

  
#include "gvertex.h"
#include "gmatrix.cpp"
  
#endif

