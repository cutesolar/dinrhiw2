/*
 * modular arithmetic numbers
 * implementation expects that K is prime
 */

#ifndef modular_h
#define modular_h

#include "number.h"
#include "integer.h"
#include <iostream>

//#define DEFAULT_MODULAR_BASIS 8
//#define DEFAULT_MODULAR_EXP   3 // (2^3 = 8 = 7+1)

#define DEFAULT_MODULAR_BASIS 7 // was: 7 is prime!, was: 31 is prime!, now 11 is prime, was: 11, NOW: 7


namespace whiteice
{
  namespace math
  {

    // class modular : public number< modular<T>, T, T, unsigned int>
    template <typename T=unsigned int>
    struct modular
    {
      public:
      
      modular();
      modular(const modular<T>& a);	
      modular(const T& value, const T& modulo);
      modular(const T& value);
      virtual ~modular();
      
      // operators
      modular<T> operator+(const modular<T>&) const ;
      modular<T> operator-(const modular<T>&) const ;
      modular<T> operator*(const modular<T>&) const ;
      modular<T> operator/(const modular<T>&) const ;
      
      // complex conjugate (?)
      modular<T> operator!() const ;
      modular<T> operator-() const ;
      
      modular<T>& operator+=(const modular<T>&) ;
      modular<T>& operator-=(const modular<T>&) ;
      modular<T>& operator*=(const modular<T>&) ;
      modular<T>& operator/=(const modular<T>&) ;
      
      modular<T>& operator=(const modular<T>&) ;      
      
      bool operator==(const modular<T>&) const ;
      bool operator!=(const modular<T>&) const ;
      bool operator>=(const modular<T>&) const ;
      bool operator<=(const modular<T>&) const ;
      bool operator< (const modular<T>&) const ;
      bool operator> (const modular<T>&) const ;
      
      // scalar operation
      modular<T>& operator= (const T& s) ;
      modular<T>  operator* (const T& s) const ;
      modular<T>  operator/ (const T& s) const ;
      modular<T>& operator*=(const T& s) ;
      modular<T>& operator/=(const T& s) ;

      template <typename A>
      friend modular<A> operator*(const A& s, const modular<A>&) ;

      modular<T>& operator++(int value) ;
      modular<T>& operator--(int value) ;
      
      modular<T>& abs() ;      
      
      T& operator[](const unsigned int& index) 
        ;
      
      const T& operator[](const unsigned int& index) const
        ;
      
      bool comparable() { return false; }
      
      private:
      
      T value;
      T modulo; 	
    };
    
    
    template <typename T>
      modular<T> operator*(const T& s, const modular<T>&) ;
    
    template <typename T>
      std::ostream& operator<<(std::ostream& ios,
			       const whiteice::math::modular<T>&);
    
    
    
    
    extern template struct modular<unsigned int>;
    extern template struct modular<unsigned short>;
    extern template struct modular<unsigned char>;
    extern template struct modular<integer>;
    
    
    extern template modular<unsigned int> operator*<unsigned int>
      (const unsigned int& s, const modular<unsigned int>&) ;
    
    extern template modular<unsigned short> operator*<unsigned short>
      (const unsigned short& s, const modular<unsigned short>&) ;
    
    extern template modular<unsigned char> operator*<unsigned char>
      (const unsigned char& s, const modular<unsigned char>&) ;
    
    extern template modular<integer> operator*<integer>
      (const integer& s, const modular<integer>&) ;
    
    
    extern template std::ostream& operator<< <unsigned int>(std::ostream& ios, const whiteice::math::modular<unsigned int>&);
    extern template std::ostream& operator<< <unsigned short>(std::ostream& ios, const whiteice::math::modular<unsigned short>&);
    extern template std::ostream& operator<< <unsigned char>(std::ostream& ios, const whiteice::math::modular<unsigned char>&);
    extern template std::ostream& operator<< <integer>(std::ostream& ios, const whiteice::math::modular<integer>&);
    
  }
}



#endif
