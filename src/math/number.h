/*
 * interface for number operations
 */

#ifndef number_h
#define number_h

#include <exception>
#include <iostream>
#include <stdexcept>
#include "ownexception.h"


namespace whiteice
{
  // D is datatype, S scalar which can operate with D
  // E is subelement of D and with indexing type U
  template <typename D, typename S, typename E, typename U>
    class number
    {
    public:
      number(){ }
      virtual ~number(){ }
      
      // operators
      virtual D operator+(const D&) const  = 0;
      virtual D operator-(const D&) const  = 0;
      virtual D operator*(const D&) const  = 0;
      virtual D operator/(const D&) const  = 0;
      
      // complex conjugate (?)
      virtual D operator!() const  = 0;
      virtual D operator-() const  = 0;
      
      virtual D& operator+=(const D&)  = 0;
      virtual D& operator-=(const D&)  = 0;
      virtual D& operator*=(const D&)  = 0;
      virtual D& operator/=(const D&)  = 0;
      
      virtual D& operator=(const D&)  = 0;      

      virtual bool operator==(const D&) const  = 0;
      virtual bool operator!=(const D&) const  = 0;
      virtual bool operator>=(const D&) const  = 0;
      virtual bool operator<=(const D&) const  = 0;
      virtual bool operator< (const D&) const  = 0;
      virtual bool operator> (const D&) const  = 0;

      // scalar operation
      virtual D& operator= (const S& s)  = 0;
      //virtual D  operator+ (const S& s) const  = 0;
      //virtual D  operator- (const S& s) const  = 0;
      virtual D  operator* (const S& s) const  = 0;
      virtual D  operator/ (const S& s) const  = 0;
      virtual D& operator*=(const S& s)  = 0;
      virtual D& operator/=(const S& s)  = 0;

      virtual D& abs()  = 0;

      virtual E& operator[](const U index)
	 = 0;

      virtual const E& operator[](const U index) const
	 = 0;
      
      // returns true if >,<,>=,<= are fully defined
      // for "same" type of numbers (can be != same class)
      // returns false if >,<,>=,<= can throw exceptions
      // even with same type of numbers. (*)
      virtual bool comparable()  = 0;

    };
}

// (*) this is mainly needed for superresolutional
// exponent types in order to know if exponent type
// U is comparable or not => can superresolution
// numbers have meaningful ordering

#endif

