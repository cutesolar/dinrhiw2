/*
 * C++ arbitrary precision floating point number class
 * uses GMP library
 */

#include <iostream>
#include <stdio.h>
#include <string>
#include <gmp.h>
#include "number.h"
#include "function.h"

#ifndef __whiteice_gmp_realnumber_h
#define __whiteice_gmp_realnumber_h

namespace whiteice
{
  namespace math
  {
    
    
    class realnumber : // implements number interface
      public whiteice::number<class whiteice::math::realnumber, double, double, unsigned long> 
    {
    public:
      // constructors & conversions
      // 
      // precision 0 means that default precision is used
      // except in case of realnumber and integer parameters
      // where with realnumbers it means that same precision
      // is used and in case of integer it means
      // that higher than default precision is used if it 
      // is necessarily to represent given integer without
      // round off errors
      
      explicit realnumber(unsigned long int prec=0);
      realnumber(const realnumber& r, unsigned long int prec=0);      
      
      explicit realnumber(signed long i, unsigned long int prec=0);
      explicit realnumber(unsigned long i, unsigned long int prec=0);
      realnumber(double d, unsigned long int prec=0);
      // realnumber(const integer& i);
      
      explicit realnumber(const std::string& s,
			  unsigned long int prec=0);
      
    private: 
      // raw assign/copy only for internal functions
      explicit realnumber(const mpf_t& d);
      
    public:
      virtual ~realnumber();
      

      // returns realnumber's working precision in bits
      unsigned long int getPrecision() const ;
      void setPrecision(unsigned long int prec) ;
      
      
      // operators
      realnumber operator+(const realnumber&) const ;
      realnumber operator-(const realnumber&) const ;
      realnumber operator*(const realnumber&) const ;
      realnumber operator/(const realnumber&) const ;
      
      // complex conjugate (?)
      realnumber operator!() const ;
      realnumber operator-() const ;
      
      realnumber& operator+=(const realnumber&) ;
      realnumber& operator-=(const realnumber&) ;
      realnumber& operator*=(const realnumber&) ;
      realnumber& operator/=(const realnumber&) ;
      
      realnumber& operator=(const realnumber&) ;      
      
      // comparisions

      bool operator==(const realnumber&) const ;
      bool operator!=(const realnumber&) const ;
      bool operator>=(const realnumber&) const ;
      bool operator<=(const realnumber&) const ;
      bool operator< (const realnumber&) const ;
      bool operator> (const realnumber&) const ;

      // scalar operations
      realnumber& operator= (const double& s) ;
      realnumber  operator+ (const double& s) const ;
      realnumber  operator- (const double& s) const ;
      realnumber& operator+=(const double& s) ;
      realnumber& operator-=(const double& s) ;
      realnumber  operator* (const double& s) const ;
      realnumber  operator/ (const double& s) const ;
      realnumber& operator*=(const double& s) ;
      realnumber& operator/=(const double& s) ;
      
      friend realnumber operator*(const double s, const realnumber& r);
      
      // scalar comparisions
      bool operator==(const double) const ;
      bool operator!=(const double) const ;
      bool operator>=(const double) const ;
      bool operator<=(const double) const ;
      bool operator< (const double) const ;
      bool operator> (const double) const ;
      
      friend bool operator==(const double, const realnumber& r);
      friend bool operator!=(const double, const realnumber& r);
      friend bool operator>=(const double, const realnumber& r);
      friend bool operator<=(const double, const realnumber& r);
      friend bool operator< (const double, const realnumber& r);
      friend bool operator> (const double, const realnumber& r);
      
      
      bool operator==(const signed long int) const ;
      bool operator!=(const signed long int) const ;
      bool operator>=(const signed long int) const ;
      bool operator<=(const signed long int) const ;
      bool operator< (const signed long int) const ;
      bool operator> (const signed long int) const ;
      
      // basic mathematical functions of for realnumber
      realnumber& abs() ;
      realnumber& ceil() ;
      realnumber& floor() ;
      realnumber& trunc() ;
      realnumber& round() ;

      // returns sign of real number
      // returns 1 if r > 0, 0 if r == 0 and -1 if r < 0
      int sign() const ;

      // overwrites number using [0,1[ given precision number
      // this is SLOW because of mutex lock around __rndstate
      realnumber& random();
      
      
      double& operator[](const unsigned long index)
	;

      const double& operator[](const unsigned long index) const
	;
      
      bool comparable() { return true; }
      
      
      //////////////////////////////////////////////////
      // conversions
      
      // rounds to the closest double
      double  getDouble() const ;
      
      // returns floor(realnumber) conversion to integer
      // integer getInteger() const ; ****** TODO *******
      
      // renders realnumber to a human-digestable and 
      // realnumber() ctor understandable string
      // 
      // ndigits is tells how many digits to show
      // (zero tells to show everthing)
      std::string getString(size_t ndigits = 0) const ;

      // read and write number to FILE using string/text format
      bool printFile(FILE* output) const;
      bool readFile(FILE* input);

      
      
      //////////////////////////////////////////////////////////////////////
      // friend functions
      
      // for some reason defining these as pure functions (PURE_FUNCTION)
      // makes compiler to remove *any* or all calls (wrongly) to
      // these functions so that realnumber.data isn't even correctly
      // initialized (-> segmentation faults, crashes, hang (malloc(really big number) etc.), etc.)
      
      friend realnumber sqrt(const realnumber& r);
      
      friend realnumber  pow(const realnumber& x, const realnumber& y);
      friend realnumber  exp(const realnumber& x);
      friend realnumber  log(const realnumber& x);
      
      friend realnumber  sin(const realnumber& x);
      friend realnumber  cos(const realnumber& x);
      
      friend realnumber pi(unsigned long int prec);
      
      friend realnumber  abs(const realnumber& x);
      friend realnumber  arg(const realnumber& x);
      friend realnumber conj(const realnumber& x);
      friend realnumber real(const realnumber& x);
      friend realnumber imag(const realnumber& x);
      
      
    private:
      
      mpf_t data;
      
    };
    
    
    inline realnumber operator*(const double s, const realnumber& r){
      return (r*s);
    }
    
    
    inline bool operator==(const double d, const realnumber& r){
      return (r == d);
    }
    
    inline bool operator!=(const double d, const realnumber& r){
      return (r != d);
    }
    
    inline bool operator>=(const double d, const realnumber& r){
      return (r <= d);
    }
    
    inline bool operator<=(const double d, const realnumber& r){
      return (r >= d);
    }
    
    inline bool operator< (const double d, const realnumber& r){
      return (r > d);
    }
    
    inline bool operator> (const double d, const realnumber& r){
      return (r < d);
    }
    
    
    // printing
    std::ostream& operator<<(std::ostream& ios,
			     const class whiteice::math::realnumber& r);
    
  };
};



#include "blade_math.h"



#endif // __whiteice_gmp_realnumber_h
