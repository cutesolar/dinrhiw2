
#include <vector>

#include "nnetwork.h"
#include "DiffEQ_HMC.h"
#include "diffeqs.h"


namespace whiteice
{
  template <typename T>
  DiffEq_HMC<T>::DiffEq_HMC(const nnetwork<T>& init_net,
			    const whiteice::math::vertex<T>& start,
			    const whiteice::dataset<T>& data,
			    const std::vector<T>& correct_times,
			    bool storeSamples, bool adaptive) :
    HMC_abstract<T>(storeSamples, adaptive)
  {
    temperature = T(1.0f);
    
    nnet = NULL;
    nnet = new nnetwork<T>(init_net);

    if(nnet->input_size() != start.size()) assert(0);

    this->start = start;
    this->data = data;
    this->correct_times = correct_times;
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

    float TIME_LENGTH = 0.0f;
    convert(TIME_LENGTH, correct_times[correct_times.size()-1]);

    whiteice::math::vertex<T> x0, y, temp, temp2;
    std::vector< whiteice::math::vertex<T> > xdata, inputdata;
    
    T error = T(0.0f);

    

    for(unsigned int i=0;i<data.size(0);i++){
      temp = data.access(i, 0);

      const unsigned int N = temp.size() / correct_times.size();
      x0.resize(N);
      x0.zero();

      for(unsigned int k=0;k<temp.size();k += N){
	temp.subvertex(temp2, k, N);
	inputdata.push_back(temp2);
      }

      if(simulate_diffeq_model2(*nnet, x0, TIME_LENGTH, xdata, correct_times) == false){
	assert(0); // should not happen
      }
      
      if(xdata.size() != inputdata.size()) assert(0); // should not happen
      
      for(unsigned int index=0;index<inputdata.size();index++){
	auto delta = xdata[index] - inputdata[index];
	error += (delta*delta)[0];
      }
    }

    error = T(0.5f)*error/temperature;

    error += T(0.5f)*(q*q)[0];

    return error;
  }
  
  
  template <typename T>
  math::vertex<T> DiffEq_HMC<T>::Ugrad(const math::vertex<T>& q)
  {
    // 1. simulate T time, N time steps forward using Runge-Kutta and nnet ODE.
    // 2. simulate gradient nnet ODE T time, N steps forward too
    // 3. calculate gradient

    float TIME_LENGTH = 0;
    convert(TIME_LENGTH, correct_times[correct_times.size()-1]);

    whiteice::math::vertex<T> x0, y, temp, temp2;
    std::vector< whiteice::math::vertex<T> > xdata, inputdata, gdata;

    whiteice::math::vertex<T> sumgrad;
    

    for(unsigned int i=0;i<data.size(0);i++){
      temp = data.access(i, 0);

      const unsigned int N = temp.size() / correct_times.size();
      x0.resize(N);
      x0.zero();

      for(unsigned int k=0;k<temp.size();k += N){
	temp.subvertex(temp2, k, N);
	inputdata.push_back(temp2);
      }

      if(simulate_diffeq_model2(*nnet, x0, TIME_LENGTH, xdata, correct_times) == false){
	assert(0); // should not happen
      }

      if(xdata.size() != inputdata.size()) assert(0); // should not happen
      
      // calculate deltas
      std::vector< math::vertex<T> > deltas;

      for(unsigned int index=0;index<inputdata.size();index++){
	auto delta = xdata[index] - inputdata[index];
	deltas.push_back(delta);
      }
      

      if(simulate_diffeq_model_nn_gradient2(*nnet, x0,
					    deltas, correct_times,
					    gdata, correct_times) == false){
	assert(0); // should not happen
      }
      
      for(unsigned int index=0;index<gdata.size();index++){
	sumgrad += gdata[index];
      }
    }

    sumgrad = sumgrad/temperature;
    
    sumgrad += q;
    
    
    return sumgrad;
  }
  
  // a starting point q for the sampler (may not be random)
  template <typename T>
  void DiffEq_HMC<T>::starting_position(math::vertex<T>& q) const
  {
    if(nnet) nnet->importdata(q);
  }
  
  
  
  template class DiffEq_HMC< math::blas_real<float> >;
  template class DiffEq_HMC< math::blas_real<double> >;
  
};
