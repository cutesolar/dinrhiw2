/*
 * Hamiltonian Markov chain sampler
 * sampler for sampling from Normal N(0,I) distribution
 * 
 * This is *test*class* to see if the HMC sampling really works
 */


#ifndef HMC_gaussian_h
#define HMC_gaussian_h

#include <vector>
#include <pthread.h>
#include <unistd.h>

#include "vertex.h"
#include "matrix.h"
#include "dinrhiw_blas.h"
#include "HMC_abstract.h"


namespace whiteice
{

  template <typename T = math::blas_real<float> >
  class HMC_gaussian : public HMC_abstract<T>
  {
    public:
    HMC_gaussian(unsigned int dimension);
    ~HMC_gaussian();
    
    virtual bool setTemperature(const T t);
    virtual T getTemperature();


    // probability functions for hamiltonian MC sampling of
    // P ~ exp(-U(q)) distribution
    T U(const math::vertex<T>& q) const;
    math::vertex<T> Ugrad(const math::vertex<T>& q) const;

    // a starting point q for the sampler (may not be random)
    void starting_position(math::vertex<T>& q) const;

    private:

    unsigned int dimension;
  };
  
  
};


namespace whiteice
{
  extern template class HMC_gaussian< float >;
  extern template class HMC_gaussian< double >;
  extern template class HMC_gaussian< math::blas_real<float> >;
  extern template class HMC_gaussian< math::blas_real<double> >;    
};


#endif
