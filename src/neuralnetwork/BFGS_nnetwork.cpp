
#include "BFGS_nnetwork.h"
#include "deep_ica_network_priming.h"
#include "RNG.h"

namespace whiteice
{

  template <typename T>
  BFGS_nnetwork<T>::BFGS_nnetwork(const nnetwork<T>& nn, const dataset<T>& d, bool overfit, bool negativefeedback) :
    whiteice::math::BFGS<T>(overfit), net(nn), data(d)
  {
    this->negativefeedback = negativefeedback;
    
    // divides data to to training and testing sets
    ///////////////////////////////////////////////
    
    dtrain = data;
    dtest  = data;
    
    dtrain.clearData(0);
    dtrain.clearData(1);
    dtest.clearData(0);
    dtest.clearData(1);
    
    
    for(unsigned int i=0;i<data.size(0);i++){
      const unsigned int r = (whiteice::rng.rand() & 1);
      
      if(r == 0){
	math::vertex<T> in  = data.access(0,i);
	math::vertex<T> out = data.access(1,i);
	
	dtrain.add(0, in,  true);
	dtrain.add(1, out, true);
      }
      else{
	math::vertex<T> in  = data.access(0,i);
	math::vertex<T> out = data.access(1,i);
	
	dtest.add(0, in,  true);
	dtest.add(1, out, true);	    
      }
    }
    
  }

  
  template <typename T>
  BFGS_nnetwork<T>::~BFGS_nnetwork()
  {
  }


  template <typename T>
  T BFGS_nnetwork<T>::getError(const math::vertex<T>& x) const
  {
    whiteice::nnetwork<T> nnet(this->net);
    nnet.importdata(x);
    
    math::vertex<T> err;
    T e = T(0.0f);

    // E = SUM 0.5*e(i)^2
    for(unsigned int i=0;i<dtest.size(0);i++){
      nnet.input() = dtest.access(0, i);
      nnet.calculate(false);
      err = dtest.access(1, i) - nnet.output();
      T inv = T(1.0f/err.size());
      err = inv*(err*err);
      e += T(0.5f)*err[0];
    }
    
    e /= T( (float)dtest.size(0) ); // per N

    return e;
  }
  

  template <typename T>
  T BFGS_nnetwork<T>::U(const math::vertex<T>& x) const
  {
    whiteice::nnetwork<T> nnet(this->net);
    
    nnet.importdata(x);
    
    math::vertex<T> err;
    T e = T(0.0f);

    // E = SUM 0.5*e(i)^2
    for(unsigned int i=0;i<dtrain.size(0);i++){
      nnet.input() = dtrain.access(0, i);
      nnet.calculate(false);
      err = dtrain.access(1, i) - nnet.output();
      // T inv = T(1.0f/err.size());
      err = (err*err);
      e += T(0.5f)*err[0];
    }
    
    // e /= T( (float)data.size(0) ); // per N

    {
      T alpha = T(0.01);   // regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
      err = alpha*(x*x);
      e += x[0];
    }
    
    
    return (e);    
  }

  
  template <typename T>
  math::vertex<T> BFGS_nnetwork<T>::Ugrad(const math::vertex<T>& x) const
  {
    whiteice::nnetwork<T> nnet(this->net);
    
    T ninv = T(1.0f); // T(1.0f/data.size(0));
    math::vertex<T> sumgrad, grad, err;

    nnet.importdata(x);

    for(unsigned int i=0;i<dtrain.size(0);i++){
      nnet.input() = dtrain.access(0, i);
      nnet.calculate(true);
      err = nnet.output() - dtrain.access(1,i);
      
      if(nnet.mse_gradient(err, grad) == false){
	std::cout << "gradient failed." << std::endl;
	assert(0); // FIXME
      }
      
      if(i == 0)
	sumgrad = ninv*grad;
      else
	sumgrad += ninv*grad;
    }

    T alpha = T(0.01f);

    sumgrad += alpha*x;

    // TODO: is this really correct gradient to use
    // (we want to use: 0,5*SUM e(i)^2 + alpha*w^2
    
    
    return (sumgrad);    
  }
  
  
  template <typename T>
  bool BFGS_nnetwork<T>::heuristics(math::vertex<T>& x) const
  {
    if(negativefeedback){
      whiteice::nnetwork<T> nnet(this->net);
      nnet.importdata(x);
      
      T alpha = T(0.5f);
      negative_feedback_between_neurons(nnet, dtrain, alpha);
      
      nnet.exportdata(x);
    }
    
    return true;
  }
  
  
  template class BFGS_nnetwork< math::blas_real<float> >;
  template class BFGS_nnetwork< math::blas_real<double> >;

  
};
