/*
 * vertex - implementation of vector class
 * Uses BLAS or NVIDIA cuBLAS accelerated matrix/vertex math.
 */

#ifndef vertex_h
#define vertex_h

#include "dinrhiw_blas.h"

#include "ownexception.h"
#include "number.h"

#include "Log.h"

#include "compressable.h"
#include "MemoryCompressor.h"

#include <stdexcept>
#include <exception>
#include <iostream>
#include <vector>

#include <assert.h>


#ifdef CUBLAS

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// loading vertex.o object file initializes NVIDIA cuBLAS
extern cublasHandle_t cublas_handle;
extern cublasStatus_t cublas_status;

// GLOBAL: used to set default GPU sync with on with RAM
// DO NOT USE/FIXME: NOTE THIS DOES NOT WORK IN MULTITHREADED CODE!!!
extern volatile bool use_gpu_sync;

inline void gpu_sync(){
  if(use_gpu_sync) cudaDeviceSynchronize();
}

#else

// dummy operator if we are not using GPU and it is accidentally called.
inline void gpu_sync(){ }

#endif


namespace whiteice
{
  class SOM2D;
  
  
  namespace math
  {
    template <typename T> class vertex;
    template <typename T> class matrix;
    template <typename T> class quaternion;
    
    
    
    template <typename T> bool gramschmidt
      (matrix<T>& B,const unsigned int i,const unsigned int j);
        
    // correlation
    template <typename T> bool autocorrelation
      (matrix<T>& R, const std::vector< vertex<T> >& data);
    template <typename T> bool autocorrelation
      (matrix<T>& R, const matrix<T>& W);
    template <typename T> bool mean_covariance_estimate
      (vertex<T>& m, matrix<T>& R, const std::vector< vertex<T> >& data);
    
    // rotations
    template <typename T> bool rhouseholder_leftrot
      (matrix<T>& A, const unsigned int i, const unsigned int M, const unsigned int k, vertex<T>& v);
    
    template <typename T> bool rhouseholder_rightrot
      (matrix<T>& A, const unsigned int i, const unsigned int M, const unsigned int k, vertex<T>& v);


    // outerproduct
    template <typename T>
      bool addouterproduct(matrix<T>& A,
			   const T& scalar,
			   const vertex<T>& a,
			   const vertex<T>& b);
    
    
    template <typename T=blas_real<float> >
      class vertex : public number<vertex<T>, T, T, unsigned int>
      // multiple-inheritance doesn't work with neuralnetwork/tst/test.cpp
      //   simple_dataset_test() etc.  ..  it's probably compiler bug
      //,public compressable
      
    {
      public:
      
      vertex();
      explicit vertex(unsigned int i);
      vertex(const vertex<T>& v);
      vertex(vertex<T>&& t);
      vertex(const std::vector<T>& v);
      virtual ~vertex();
      
      // returns vertex dimension/size
      inline unsigned int size() const {
	return dataSize;
      }
      
      unsigned int resize(unsigned int d) ;
      
      // TODO: move norm() to generic number class
      T norm() const ;
      
      // calculates partial norm for vertex(i:j)
      T norm(unsigned int i, unsigned int j) const ;      
      
      bool normalize() ; // length = 1
      void zero() ; // vertex = 0;

      // hermitean transpose of the vector
      void hermite() ;

      inline void conj(){ this->hermite(); }
      
      vertex<T> operator+(const vertex<T>& v) const ;      
      vertex<T> operator-(const vertex<T>& v) const ;
      
      // if v.size() != 1 this is inner product - returns vertex with size 1
      // if v.size() == 1 this is scalar multiplication
      vertex<T> operator*(const vertex<T>& v) const ;
      
      vertex<T> operator/(const vertex<T>& v) const ;
      vertex<T> operator!() const ;
      vertex<T> operator-() const ;
      
      // cross product
      vertex<T> operator^(const vertex<T>& v) const ;
      
      vertex<T>& operator+=(const vertex<T>& v) ;
      vertex<T>& operator-=(const vertex<T>& v) ;
      vertex<T>& operator*=(const vertex<T>& v) ;
      vertex<T>& operator/=(const vertex<T>& v) ;
      
      vertex<T>& operator=(const vertex<T>& v) ;

      vertex<T>& operator=(const matrix<T>& v) ;

      // template <typename TT, typename UU>
      // friend vertex<TT>& operator=(const vertex<UU>& v) ;
      vertex<T>& operator=(vertex<T>&& t) ;
      
      bool operator==(const vertex<T>& v) const ;
      bool operator!=(const vertex<T>& v) const ;
      bool operator>=(const vertex<T>& v) const ;
      bool operator<=(const vertex<T>& v) const ;
      bool operator< (const vertex<T>& v) const ;
      bool operator> (const vertex<T>& v) const ;
      
      vertex<T>& operator=(const quaternion<T>&) ;
      
      vertex<T>& abs() ;
      vertex<T>& real();
      vertex<T>& imag();
      
      /* scalars */
      vertex<T>& operator= (const T& s) ;
      vertex<T>  operator* (const T& s) const ;
      vertex<T>  operator/ (const T& s) const ;
      vertex<T>& operator*=(const T& s) ;
      vertex<T>& operator/=(const T& s) ;
    
      template <typename TT>
      friend vertex<TT> operator*(const TT& s, const vertex<TT>& v); // multi from right
      
      // multiply from left
      vertex<T>  operator* (const matrix<T>& m) const ;    

      // outer product
      matrix<T> outerproduct() const ;
      matrix<T> outerproduct(const vertex<T>& v) const ;
      matrix<T> outerproduct(const vertex<T>& v0,
			     const vertex<T>& v1) const ;
      
      // element-wise multiplication of vector elements
      vertex<T>& dotmulti(const vertex<T>& v) ;

      // NOTE: If you are using cuBLAS acceleration you have to
      // call gpu_sync() call after modifying vertex values through direct RAM access
      
      inline T& operator[](const unsigned int index) 
      {
#ifdef _GLIBCXX_DEBUG	
	if(index >= dataSize){
	  printf("%d >= %d\n", index, dataSize);
	  whiteice::logging.error("vertex::operator[]: index out of range");
	  assert(0);
	  throw std::out_of_range("vertex index out of range"); }
#endif
	return data[index]; // no range check
      }
      
      inline const T& operator[](const unsigned int index) const 
      {	
#ifdef _GLIBCXX_DEBUG	
	if(index >= dataSize){
	  printf("%d >= %d\n", index, dataSize);
	  whiteice::logging.error("vertex::operator[]: index out of range");
	  assert(0);
	  throw std::out_of_range("vertex index out of range"); }
#endif	
	return data[index]; // no range check
      }

      
      bool subvertex(vertex<T>& v, unsigned int x0, unsigned int len) const ;
      bool write_subvertex(const vertex<T>& v, unsigned int x0) ;
      
      bool comparable() ;
      
      // stores vertex to comma separated ASCII file (.CSV) which can be
      // often imported by data analysis software
      bool saveAscii(const std::string& filename) const ;

      void toString(std::string& line) const ;

      
      //////////////////////////////////////////////////
      // vertex data compression
      
      bool compress() ;
      bool decompress() ;
      bool iscompressed() const ;
      float ratio() const ;
      
      //////////////////////////////////////////////////
      // direct memory access
      
      // copies vertex[start:(start+len-1)] = data[0:(len-1)]
      bool importData(const T* data, unsigned int len=0, unsigned int start=0) ;
      
      // copies data[0:(len-1)] = vertex[start:(start+len-1)]
      bool exportData(T* data, unsigned int len=0, unsigned int start=0) const ;
      
      //////////////////////////////////////////////////
      
      // friend list
      friend class matrix<T>;
      
      friend class whiteice::SOM2D;
      
      friend bool gramschmidt<T>(matrix<T>& B,
				 const unsigned int i,
				 const unsigned int j);
      
      // correlation
      friend bool autocorrelation<T>(matrix<T>& R, const std::vector< vertex<T> >& data);
      friend bool autocorrelation<T>(matrix<T>& R, const matrix<T>& W);
      friend bool mean_covariance_estimate<T>(vertex<T>& m, matrix<T>& R, const std::vector< vertex<T> >& data);
      
      
      // rotations
      friend bool rhouseholder_leftrot<T> (matrix<T>& A,
					   const unsigned int i,
					   const unsigned int M,
					   const unsigned int k,
					   vertex<T>& v);
      
      friend bool rhouseholder_rightrot<T>(matrix<T>& A,
					   const unsigned int i,
					   const unsigned int M,
					   const unsigned int k,
					   vertex<T>& v);
      // outerproduct
      friend bool addouterproduct<T>(matrix<T>& A,
				     const T& scalar,
				     const vertex<T>& a,
				     const vertex<T>& b);
      
      private:
      
      T* data;      
      unsigned int dataSize;
      
      MemoryCompressor* compressor;
      
    };
    
    
    template <typename T>
      vertex<T> operator*(const T& s, const vertex<T>& v);
    
    template <typename T>
      std::ostream& operator<<(std::ostream& ios,
			       const whiteice::math::vertex<T>&);
    
    
    // tries to convert vertex of type S to vertex of type T (B = A)
    template <typename T, typename S>
      bool convert(vertex<T>& B, const vertex<S>& A) 
      {
	try{
	  if(B.resize(A.size()) == false)
	    return false;
	  
	  for(unsigned int i=0;i<B.size();i++){
	    if(whiteice::math::convert(B[i], A[i]) == false)
	      return false;
	    
	    // B[i] = static_cast<T>(A[i])
	  }
	  
	  return true;
	}
	catch(std::exception& e){
	  return false;
	}
      }
  };
};


#include "linear_algebra.h"
#include "correlation.h"
#include "matrix_rotations.h"
#include "outerproduct.h"

namespace whiteice{
  namespace math{
    
    extern template class vertex<float>;
    extern template class vertex<double>;
    extern template class vertex<complex<float> >;
    extern template class vertex<complex<double> >;
    
    //extern template class vertex<int>;
    //extern template class vertex<char>;
    //extern template class vertex<unsigned int>;
    //extern template class vertex<unsigned char>;
        
    extern template class vertex< blas_real<float> >;
    extern template class vertex< blas_real<double> >;
    extern template class vertex< blas_complex<float> >;
    extern template class vertex< blas_complex<double> >;

    extern template class vertex< superresolution< blas_real<float>,
						   modular<unsigned int> > >;
    extern template class vertex< superresolution< blas_real<double>,
						   modular<unsigned int> > >;

    extern template class vertex< superresolution< blas_complex<float>,
						   modular<unsigned int> > >;
    extern template class vertex< superresolution< blas_complex<double>,
						   modular<unsigned int> > >;
    
    
    extern template vertex<float> operator*<float>(const float& s, const vertex<float>& v);
    extern template vertex<double> operator*<double>(const double& s, const vertex<double>& v);
    
    extern template vertex<complex<float> > operator*<complex<float> >
      (const complex<float>& s, const vertex<complex<float> >& v);
    
    extern template vertex<complex<double> > operator*<complex<double> >
      (const complex<double>& s, const vertex<complex<double> >& v);
    
    //extern template vertex<int> operator*<int>(const int& s, const vertex<int>& v);
    //extern template vertex<char> operator*<char>(const char& s, const vertex<char>& v);
    //extern template vertex<unsigned int> operator*<unsigned int>(const unsigned int& s, const vertex<unsigned int>& v);
    // extern template vertex<unsigned char> operator*<unsigned char>(const unsigned char& s, const vertex<unsigned char>& v);
      
    extern template vertex<blas_real<float> > operator*<blas_real<float> >
      (const blas_real<float>& s, const vertex<blas_real<float> >& v);
    
    extern template vertex<blas_real<double> > operator*<blas_real<double> >
      (const blas_real<double>& s, const vertex<blas_real<double> >& v);
									     
    extern template vertex<blas_complex<float> > operator*<blas_complex<float> >
      (const blas_complex<float>& s, const vertex<blas_complex<float> >& v);
    
    extern template vertex<blas_complex<double> > operator*<blas_complex<double> >
      (const blas_complex<double>& s, const vertex<blas_complex<double> >& v);

    
    extern template vertex<superresolution<blas_real<float>, modular<unsigned int> > > operator*<superresolution<blas_real<float>, modular<unsigned int> > >
    (const superresolution<blas_real<float>, modular<unsigned int> >& s,
     const vertex<superresolution<blas_real<float>, modular<unsigned int> > >& v);
    
    extern template vertex<superresolution<blas_real<double>, modular<unsigned int> > > operator*<superresolution<blas_real<double>, modular<unsigned int> > >
    (const superresolution<blas_real<double>, modular<unsigned int> >& s, const vertex<superresolution<blas_real<double>, modular<unsigned int> > >& v);

    
    extern template vertex<superresolution<blas_complex<float>, modular<unsigned int> > > operator*<superresolution<blas_complex<float>, modular<unsigned int> > >
    (const superresolution<blas_complex<float>, modular<unsigned int> >& s,
     const vertex<superresolution<blas_complex<float>, modular<unsigned int> > >& v);
    
    extern template vertex<superresolution<blas_complex<double>, modular<unsigned int> > > operator*<superresolution<blas_complex<double>, modular<unsigned int> > >
    (const superresolution<blas_complex<double>, modular<unsigned int> >& s, const vertex<superresolution<blas_complex<double>, modular<unsigned int> > >& v);
       
    
    
    extern template std::ostream& operator<< <float>(std::ostream& ios, const vertex<float>&);
    extern template std::ostream& operator<< <double>(std::ostream& ios, const vertex<double>&);
    extern template std::ostream& operator<< <complex<float> >(std::ostream& ios, const vertex<complex<float> >&);
    extern template std::ostream& operator<< <complex<double> >(std::ostream& ios, const vertex<complex<double> >&);
    
    extern template std::ostream& operator<< <int>(std::ostream& ios, const vertex<int>&);
    extern template std::ostream& operator<< <char>(std::ostream& ios, const vertex<char>&);
    extern template std::ostream& operator<< <unsigned int>(std::ostream& ios, const vertex<unsigned int>&);
    extern template std::ostream& operator<< <unsigned char>(std::ostream& ios, const vertex<unsigned char>&);
    
    extern template std::ostream& operator<< <blas_real<float> >(std::ostream& ios, const vertex<blas_real<float> >&);
    extern template std::ostream& operator<< <blas_real<double> >(std::ostream& ios, const vertex<blas_real<double> >&);
    extern template std::ostream& operator<< <blas_complex<float> >(std::ostream& ios, const vertex<blas_complex<float> >&);
    extern template std::ostream& operator<< <blas_complex<double> >(std::ostream& ios, const vertex<blas_complex<double> >&);

    extern template std::ostream& operator<< <superresolution<blas_real<float>, modular<unsigned int> > >(std::ostream& ios, const vertex<superresolution<blas_real<float>, modular<unsigned int> > >&);
    extern template std::ostream& operator<< <superresolution<blas_real<double>, modular<unsigned int> > >(std::ostream& ios, const vertex<superresolution<blas_real<double>, modular<unsigned int> > >&);

    
    extern template std::ostream& operator<< <superresolution<blas_complex<float>, modular<unsigned int> > >(std::ostream& ios, const vertex<superresolution<blas_complex<float>, modular<unsigned int> > >&);
    extern template std::ostream& operator<< <superresolution<blas_complex<double>, modular<unsigned int> > >(std::ostream& ios, const vertex<superresolution<blas_complex<double>, modular<unsigned int> > >&);
    
    
  };
};



#endif
