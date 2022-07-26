
#include "DiffEq_HMC.h"


namespace whiteice
{
  template <typename T>
  DiffEq_HMC<T>::DiffEq_HMC(const nnework<T>& init_net,bool storeSamples, bool adaptive) :
    HMC_abstract<T>(storeSamples, adaptive)
  {
    temperature = T(1.0f);
    
    nnet = NULL;
    nnet = new nnetwork<T>(init_net);
  }

  template <typename T>
  DiffEq_HMC<T>::~DiffEq_HMC(){
    if(nnet){ delete nnet; nnet = NULL; }
  }
  
  // probability functions for hamiltonian MC sampling of
  // P ~ exp(-U(q)) distribution
  template <typename T>
  T DiffEq_HMC<T>::U(const math::vertex<T>& q) const
  {
    // 1. simulate T time, N time steps forward using Runge-Kutta and nnet ODE.
    // 2. calculate error

    
  }

  template <typename T>
  math::vertex<T> DiffEq_HMC<T>::Ugrad(const math::vertex<T>& q)
  {
    // 1. simulate T time, N time steps forward using Runge-Kutta and nnet ODE.
    // 2. simulate gradient nnet ODE T time, N steps forward too
    // 3. calculate gradient

    
    
  }
  
  // a starting point q for the sampler (may not be random)
  template <typename T>
  void DiffEq_HMC<T>::starting_position(math::vertex<T>& q) const
  {
    if(nnet) nnet->importdata(q);
  }
  
  
};
