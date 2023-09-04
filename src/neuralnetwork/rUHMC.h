/*
 * Hamiltonian Markov Chain sampler
 * for use with RECURRENT neural networks
 * 
 * (C) Copyright Tomas Ukkonen 2023
 */

#ifndef rUHMC_h
#define rUHMC_h

#include "UHMC.h"


namespace whiteice
{
  template <typename T = math::blas_real<float> >
  class rUHMC : public whiteice::UHMC<T>
  {
  public:

    rUHMC(const whiteice::nnetwork<T>& net, const whiteice::dataset<T>& ds,
	  bool adaptive=false, T alpha = T(0.5), bool store = true, bool restart_sampler = true);
    
    virtual ~rUHMC();


    // probability functions for hamiltonian MC sampling
    T U(const math::vertex<T>& q, bool useRegulizer = true) const;
    
    math::vertex<T> Ugrad(const math::vertex<T>& q, bool useRegulizer = true) const;
    
    // calculates mean error for the latest N samples, 0 = all samples
    T getMeanError(unsigned int latestN = 0) const;

  protected:

    unsigned int RDIM = 0; // recurrent dimensions in neural network
  };
  
};




#endif
