/*
 * Learns weights of dx/dt = f(x,w) differential equation model 
 * using bayesian inference model and HMC sampler.
 *
 */

#ifndef __DiffEq_HMC_h
#define __DiffEq_HMC_h

#include "vertex.h"
#include "matrix.h"

#include "HMC_abstract.h"
#include "nnetwork.h"


namespace whiteice
{
  template <typename T = math::blas_real<float> >
  class DiffEq_HMC : public HMC_abstract<T>
  {
  public:
    
    DiffEq_HMC(const nnetwork<T>& init_net,
	       const whiteice::math::vertex<T>& start,
	       const dataset<T>& data,
	       const std::vector<T>& correct_times,
	       bool storeSamples = true, bool adaptive = true);
    
    ~DiffEq_HMC();
    
    // set "temperature" for probability distribution [used in sampling/training]
    // [t = 1 => no (low) temperature]
    // [t = 0 => high temperature (high degrees of freedom)]
    virtual bool setTemperature(const T t){ temperature = t; return true; }
    
    // get "temperature" of probability distribution
    virtual T getTemperature(){ return temperature; }
    
    // probability functions for hamiltonian MC sampling of
    // P ~ exp(-U(q)) distribution
    virtual T U(const math::vertex<T>& q) const;
    
    virtual math::vertex<T> Ugrad(const math::vertex<T>& q);
    
    // a starting point q for the sampler (may not be random)
    virtual void starting_position(math::vertex<T>& q) const;
    
    private:
    
    T temperature = T(1.0f);
    
    mutable whiteice::nnetwork<T>* nnet = NULL;

    whiteice::math::vertex<T> start;
    whiteice::dataset<T> data;
    std::vector<T> correct_times;
    
  };
  
  
  
};


#endif
