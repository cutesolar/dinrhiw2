
#include "LBFGS_nnetwork.h"
#include "deep_ica_network_priming.h"

#include "eig.h"


namespace whiteice
{

  template <typename T>
  LBFGS_nnetwork<T>::LBFGS_nnetwork(const nnetwork<T>& nn, const dataset<T>& d, bool overfit, bool negativefeedback) :
    whiteice::math::LBFGS<T>(overfit), net(nn), data(d)
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
      const unsigned int r = (rand() & 3);
      
      if(r != 0){ // 75% will go to training data
	math::vertex<T> in  = data.access(0,i);
	math::vertex<T> out = data.access(1,i);
	
	dtrain.add(0, in,  true);
	dtrain.add(1, out, true);
      }
      else{ // 25% will go to testing data
	math::vertex<T> in  = data.access(0,i);
	math::vertex<T> out = data.access(1,i);
	
	dtest.add(0, in,  true);
	dtest.add(1, out, true);
      }
    }
    
    // we cannot never have zero training or testing set size
    // in such a small cases (very little data) we just use
    // all the data both for training and testing and overfit
    if(dtrain.size(0) == 0 || dtest.size(0) == 0){
      dtrain = data;
      dtest  = data;
    }
    
  }

  
  template <typename T>
  LBFGS_nnetwork<T>::~LBFGS_nnetwork()
  {
  }


  template <typename T>
  T LBFGS_nnetwork<T>::getError(const math::vertex<T>& x) const
  {
    T e = T(0.0f);
    
#pragma omp parallel shared(e)
    {
      whiteice::nnetwork<T> nnet(this->net);
      nnet.importdata(x);
      
      math::vertex<T> err;
      T esum = T(0.0f);
      
      
      // E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<dtest.size(0);i++){
	// std::cout << "data in  = " << dtest.access(0, i) << std::endl;
	// std::cout << "data out = " << dtest.access(1, i) << std::endl;
	
	nnet.input() = dtest.access(0, i);
	nnet.calculate(false);
	err = dtest.access(1, i) - nnet.output();
	
	err = (err*err);
	esum += T(0.5f)*err[0];
      }

#pragma omp critical (mfdhjgfreouitreqq)
      {
	e += esum;
      }

    }
    
    e /= T( (float)dtest.size(0) ); // per N
    
    return e;
  }
  

  template <typename T>
  T LBFGS_nnetwork<T>::U(const math::vertex<T>& x) const
  {
    T e = T(0.0f);
    
#pragma omp parallel
    {
      whiteice::nnetwork<T> nnet(this->net);
      
      nnet.importdata(x);
      math::vertex<T> err;
      T esum = T(0.0f);
    
      // E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<dtrain.size(0);i++){
	nnet.input() = dtrain.access(0, i);
	nnet.calculate(false);
	err = dtrain.access(1, i) - nnet.output();
	err = (err*err); // /T(dtrain.size(0));
	esum += T(0.5f)*err[0];
      }
      
#pragma omp critical (trweuoifdjkoxvcw)
      {
	e += esum;
      }
      
    }

#if 1    
    {
      T alpha = T(0.01);   // regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
      auto err = T(0.5)*alpha*(x*x);
      e += err[0];
    }
#endif

    e /= T(dtrain.size(0));
    
    return (e);    
  }

  
  template <typename T>
  math::vertex<T> LBFGS_nnetwork<T>::Ugrad(const math::vertex<T>& x) const
  {
    math::vertex<T> sumgrad;
    sumgrad = x;
    sumgrad.zero();

#if 0
    math::matrix<T> sigma2;
    sigma2.resize(net.output_size(), net.output_size());
    sigma2.zero();

    math::vertex<T> m(net.output_size());
    m.zero();
#endif

    // positive phase/gradient
#pragma omp parallel shared(sumgrad)
    {
      whiteice::nnetwork<T> nnet(this->net);
      nnet.importdata(x);
      
      math::vertex<T> sgrad, grad, err;

#if 0
      math::matrix<T> sum_sigma2;
      sum_sigma2.resize(net.output_size(), net.output_size());
      sum_sigma2.zero();
      
      math::vertex<T> sum_m(net.output_size());
      sum_m.zero();
#endif
      
      sgrad = x;
      sgrad.zero();

#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<dtrain.size(0);i++){
	nnet.input() = dtrain.access(0, i);
	nnet.calculate(true);
	err = nnet.output() - dtrain.access(1,i);

	if(nnet.mse_gradient(err, grad) == false){
	  std::cout << "gradient failed." << std::endl;
	  assert(0); // FIXME
	}

#if 0
	sum_sigma2 += err.outerproduct(err);
	// sum_m += err;
#endif
	
	sgrad += grad; // /T(dtrain.size(0));
      }
      
#pragma omp critical (reiogjivzzdf)
      {
	sumgrad += sgrad;
#if 0
	sigma2 += sum_sigma2;
	m += sum_m;
#endif
      }

    }

#if 0
    sigma2 /= T((double)dtrain.size(0));
    m /= T((double)dtrain.size(0));
    sigma2 -= m.outerproduct(m);

    // sigma2 is error covariance matrix
    // (we assume there are only few output dimensions)
    // we need to sample from N(O, Sigma2) solve X*sqrt(D)*n
    auto D = sigma2;
    math::matrix<T> X;
    
    while(symmetric_eig(D, X) == false){
      for(unsigned int i=0;i<m.size();i++)
	sigma2(i,i) += T(1.0); // dummy regularizer..
      D = sigma2;
    }

    sigma2 = D;
    
    for(unsigned int i=0;i<m.size();i++){
      sigma2(i,i) = sqrt(abs(sigma2(i,i)));
    }

    sigma2 = X*sigma2;
    

    // negative phase/gradient
#pragma omp parallel shared(sumgrad)
    {
      whiteice::nnetwork<T> nnet(this->net);
      nnet.importdata(x);
      
      math::vertex<T> sgrad, grad, err;
      
      sgrad = x;
      sgrad.zero();

#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<dtrain.size(0);i++){
	// generates negative particle
	auto x = dtrain.access(0, rng.rand() % dtrain.size(0));
	
	nnet.input() = x;
	nnet.calculate(true);
	auto y = nnet.output();

	math::vertex<T> n(y.size());
	rng.normal(n);
	y += sigma2 * n; // adds properly correlated noise..
	
	err = nnet.output() - y;

	if(nnet.mse_gradient(err, grad) == false){
	  std::cout << "gradient failed." << std::endl;
	  assert(0); // FIXME
	}
	
	sgrad -= grad; // /T(dtrain.size(0));
      }
      
#pragma omp critical (mnveioqaa)
      {
	sumgrad += sgrad;
      }

    }
#endif

#if 1
    {
      T alpha = T(0.01f);
      sumgrad += alpha*x;
    }
#endif
    
    sumgrad /= T(dtrain.size(0));

	  
    // sumgrad.normalize();
    
    return (sumgrad);
  }
  
  
  template <typename T>
  bool LBFGS_nnetwork<T>::heuristics(math::vertex<T>& x) const
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
  
  
  
  template class LBFGS_nnetwork< math::blas_real<float> >;
  template class LBFGS_nnetwork< math::blas_real<double> >;

  
};
