/*
 * BBRBM.cpp
 *
 *  Created on: 22.6.2015
 *      Author: Tomas Ukkonen
 */

#include "BBRBM.h"
#include "dataset.h"
#include "linear_ETA.h"
#include "Log.h"

#include "LBFGS_BBRBM.h"

#include <unistd.h>


namespace whiteice {

template <typename T>
BBRBM<T>::BBRBM()
{
  h.resize(1);
  v.resize(1);
  
  W.resize(1,1);
  a.resize(1);
  b.resize(1);

  initializeWeights();
}

template <typename T>
BBRBM<T>::BBRBM(const BBRBM<T>& rbm)
{
  this->v = rbm.v;
  this->h = rbm.h;

  this->W = rbm.W;
  this->a = rbm.a;
  this->b = rbm.b;

  this->Usamples = rbm.Usamples;
}

// creates 2-layer: V * H network
template <typename T>
BBRBM<T>::BBRBM(unsigned int visible, unsigned int hidden) 
{
  if(visible == 0 || hidden == 0)
    throw std::invalid_argument("invalid network architecture");
    
  v.resize(visible);
  h.resize(hidden);
  
  W.resize(hidden, visible);
  a.resize(visible);
  b.resize(hidden);
    
  initializeWeights();    
}

template <typename T>
BBRBM<T>::~BBRBM()
{

}


template <typename T>
BBRBM<T>& BBRBM<T>::operator=(const BBRBM<T>& rbm)
{
  this->v = rbm.v;
  this->h = rbm.h;

  this->W = rbm.W;
  this->a = rbm.a;
  this->b = rbm.b;
  this->Usamples = rbm.Usamples;

  return (*this);
}

template <typename T>
bool BBRBM<T>::operator==(const BBRBM<T>& rbm) const
{
  if(this->v != rbm.v) return false;
  if(this->h != rbm.h) return false;

  if(this->W != rbm.W) return false;
  if(this->a != rbm.a) return false;
  if(this->b != rbm.b) return false;
  if(this->Usamples != rbm.Usamples) return false;

  return true;
}

template <typename T>
bool BBRBM<T>::operator!=(const BBRBM<T>& rbm) const
{
  if(this->v == rbm.v && this->h == rbm.h && 
     this->W == rbm.W && this->a == rbm.a &&
     this->b == rbm.b && this->Usamples == rbm.Usamples)
    return false;

  return true;
}


template <typename T>
bool BBRBM<T>::resize(unsigned int visible, unsigned int hidden)
{
  if(visible == 0 || hidden == 0)
    return false;
    
  // the last term is always constant: 1 (one)
  v.resize(visible);
  h.resize(hidden);
  
  W.resize(hidden, visible);
  a.resize(visible);
  b.resize(hidden);
  
  initializeWeights(); // overwrites old weights..

  return true;
}

////////////////////////////////////////////////////////////

template <typename T>
unsigned int BBRBM<T>::getVisibleNodes() const
{
  return v.size();
}

template <typename T>
unsigned int BBRBM<T>::getHiddenNodes() const
{
  return h.size();
}
  
template <typename T>
void BBRBM<T>::getVisible(math::vertex<T>& v) const
{
  v = this->v;
}

template <typename T>
bool BBRBM<T>::setVisible(const math::vertex<T>& v)
{
  if(v.size() != this->v.size())
    return false;

  this->v = v;

  return true;
}


template <typename T>
void BBRBM<T>::getHidden(math::vertex<T>& h) const
{
  h = this->h;
}

template <typename T>
bool BBRBM<T>::setHidden(const math::vertex<T>& h)
{
  if(h.size() != this->h.size())
    return false;

  this->h = h;
  
  return true;
}

template <typename T>
math::vertex<T> BBRBM<T>::getBValue() const
{
  return b;
}

template <typename T>
math::vertex<T> BBRBM<T>::getAValue() const
{
  return a;
}

template <typename T>
math::matrix<T> BBRBM<T>::getWeights() const
{
  return W;
}

template <typename T>
bool BBRBM<T>::setBValue(const math::vertex<T>& b)
{
  if(b.size() != this->b.size())
    return false;

  this->b = b;
  
  return true;
}

template <typename T>
bool BBRBM<T>::setAValue(const math::vertex<T>& a)
{
  if(a.size() != this->a.size())
    return false;

  this->a = a;

  return true;
}

template <typename T>  
bool BBRBM<T>::setWeights(const math::matrix<T>& W)
{
  if(W.ysize() != this->W.ysize() ||
     W.xsize() != this->W.xsize())
    return false;

  this->W = W;
  return true;
}


template <typename T>
bool BBRBM<T>::getHiddenResponseField(const math::vertex<T>& v,
				      math::vertex<T>& h) const
{
  if(this->v.size() != v.size()) return false;
  
  h = W*v + b;
  
  for(unsigned int j=0;j<h.size();j++){
    h[j] = T(1.0)/(T(1.0) + math::exp(-h[j], T(20.0f)));
  }

  // no discretization

  return true;
}
  

template <typename T>
bool BBRBM<T>::reconstructData(unsigned int iters)
{
  if(iters == 0) return false;
  
  while(iters > 0){
    h = W*v + b;
    
    // 1. hidden units: calculates sigma(a_j)
    for(unsigned int j=0;j<h.size();j++){
      T aj = T(1.0)/(T(1.0) + math::exp(-h[j], T(20.0f)));

      T r = rng.uniform();

      if(aj > r) h[j] = T(1.0); // discretization step
      else       h[j] = T(0.0);
    }
    
    iters--;
    if(iters <= 0) return true;

    v = h*W + a;
    
    // 1. visible units: calculates sigma(a_j)
    for(unsigned int j=0;j<(v.size()-0);j++){
      T aj = T(1.0)/(T(1.0) + math::exp(-v[j], T(20.0f)));
      
      T r = rng.uniform();
      
      if(aj > r) v[j] = T(1.0); // discretization step
      else       v[j] = T(0.0);
    }
    
    iters--;
    if(iters <= 0) return true;
  }
  
  return true;
}

template <typename T>
bool BBRBM<T>::reconstructData(std::vector< math::vertex<T> >& samples,
			       unsigned int iters)
{
  for(auto& s : samples){
    if(this->setVisible(s) == false)
      return false;
    
    if(this->reconstructData(2*iters) == false)
      return false;

    this->getVisible(s);
  }

  return true;
}

template <typename T>
bool BBRBM<T>::reconstructDataHidden(unsigned int iters)
{
  if(iters == 0) return false;

  while(iters > 0){
    v = h*W + a;
    
    // 1. visible units: calculates sigma(a_j)
    for(unsigned int j=0;j<(v.size()-0);j++){
      T aj = T(1.0)/(T(1.0) + math::exp(-v[j], T(20.0f)));

      T r = rng.uniform();
      
      if(aj > r) v[j] = T(1.0); // discretization step
      else       v[j] = T(0.0);
    }
    
    iters--;
    if(iters <= 0) return true;
    
    h = W*v + b;
    
    // 1. hidden units: calculates sigma(a_j)
    for(unsigned int j=0;j<(h.size()-0);j++){
      T aj = T(1.0)/(T(1.0) + math::exp(-h[j], T(20.0f)));

      T r = rng.uniform();
      
      if(aj > r) h[j] = T(1.0); // discretization step
      else       h[j] = T(0.0);
    }
    
    iters--;
    if(iters <= 0) return true;
  }
    
  return true;
}


  // calculates h = sigmoid(W*v + b) without disretization step
  template <typename T>
  bool BBRBM<T>::calculateHiddenMeanField(const math::vertex<T>& v,
					  math::vertex<T>& h) const
  {
    if(v.size() != a.size())
      return false;

    h = W*v + b;

    sigmoid(h);

    return true;
  }

  
  // calculates v = sigmoid(h*W + a) without discretization step
  template <typename T>
  bool BBRBM<T>::calculateVisibleMeanField(const math::vertex<T>& h,
					   math::vertex<T>& v) const
  {
    if(h.size() != b.size())
      return false;

    v = h*W + a;

    sigmoid(v);

    return true;
  }


template <typename T>
void BBRBM<T>::getParameters(math::matrix<T>& W,
			     math::vertex<T>& a,
			     math::vertex<T>& b) const
{
  W = this->W;
  a = this->a;
  b = this->b;
}


template <typename T>
bool BBRBM<T>::initializeWeights() // initialize weights to small values
{
  // TODO BUG: test this untested code
  
  // use recommendation by Hinton and initialize weights to small random values 0.01*Normal(0,1)
  // const T scaling = T(0.01); // st.dev. scaling [0.01,0.1]
  const T scaling = T(10.0); // st.dev. scaling [0.01,0.1]
  
  for(unsigned int j=0;j<W.ysize();j++){
    for(unsigned int i=0;i<W.xsize();i++){
      T r = scaling*rng.normal();
      W(j,i) = r;
    }
  }
  
  for(unsigned int j=0;j<a.size();j++){
    T r = scaling*rng.normal();
    a[j] = r;
  }
  
  for(unsigned int j=0;j<b.size();j++){
    T r = scaling*rng.normal();
    b[j] = r;
  }
  


#if 0
	// bad initialization is probably why BBRBM code doesn't work properly(?)

  for(unsigned int j=0;j<W.ysize();j++){
    for(unsigned int i=0;i<W.xsize();i++){
      T r = rng.uniform();
      // r   = T(0.02)*r - T(0.01);
      r = T(2.0)*r - T(1.0);
      W(j,i) = r;
    }
  }
    
  for(unsigned int j=0;j<a.size();j++){
    T r = rng.uniform();    
    // r   = T(0.02)*r - T(0.01);
    r = T(2.0)*r - T(1.0);
    a[j] = r;
  }
  
  for(unsigned int j=0;j<b.size();j++){
    T r = rng.uniform();
    // r   = T(0.02)*r - T(0.01);
    r = T(2.0)*r - T(1.0);
    b[j] = r;
  }
#endif

  return true;
}

// returns remaining reconstruction error in samples
template <typename T>
T BBRBM<T>::learnWeights(const std::vector< math::vertex<T> >& samples,
			 const unsigned int EPOCHS,
			 const int verbose, const bool* running)
{
  // implements traditional CD-k (k=2) learning algorithm,
  // which can be used as a reference point
  // and returns reconstruction error as the modelling error..
  
  const unsigned int CDk = 2;  // CD-k algorithm (was 10!)
  T lambda = T(0.00001);       // initial learning rate (was 0.01)
    
  math::matrix<T> PW(W);
  math::matrix<T> NW(W);
  math::vertex<T> pa(a), na(a);
  math::vertex<T> pb(b), nb(b);

  
  T latestError = reconstructionError(samples, 1000, a, b, W);
  
  if(verbose == 1){
    double error = 0.0;
    math::convert(error, latestError);
    
    printf("BBRBM::learnWeights(): %d/%d START. R: %f\n", 0, EPOCHS, error);
    fflush(stdout);
  }
  else if(verbose == 2){
    char buffer[128];
    double error = 0.0;
    math::convert(error, latestError);
    
    snprintf(buffer, 128, "%d/%d START. R: %f\n", 0, EPOCHS, error);
    whiteice::logging.info(buffer);
  }

  if(samples.size() <= 0) return latestError;
  if(samples[0].size() != v.size())
    return latestError; // silent failure

  const unsigned int NUMSAMPLES = 10;
  const unsigned int EPOCHSAMPLES = 1000;
  
  
  for(unsigned int e=0;e<EPOCHS;e++){

    whiteice::linear_ETA<double> eta;
    eta.start(0.0, (double)EPOCHSAMPLES);
    
    for(unsigned int es=0;es<EPOCHSAMPLES;es++){

      if(running) if(*running == false) break;

      eta.update(es);

      if(verbose == 1){
	if(es % 100){
	  printf("\r                                                                             \r");
	  printf("EPOCH SAMPLES %d/%d [ETA: %.2f minutes]", es, EPOCHSAMPLES, eta.estimate()/60.0);
	  fflush(stdout);
	}
      }
      else if(verbose == 2){
	if(es % 100 == 0){ // prints only every 100th iteration..
	  char buffer[128];
	  snprintf(buffer, 128, "BBRBM::learnWeights(): epoch samples %d/%d [ETA: %.2f minutes]",
		   es, EPOCHSAMPLES, eta.estimate()/60.0);
	  whiteice::logging.info(buffer);
	}
      }

      // calculates positive gradient from NUMSAMPLES examples Pdata(v))
      PW.zero();
      pa.zero();
      pb.zero();
      
      // Pdata
      for(unsigned int i=0;i<NUMSAMPLES;i++){
	if(running) if(*running == false) break;
	
	const unsigned int index = rng.rand() % samples.size();
	auto v = samples[index];
	
	auto h = W*v + b;
	
	// hidden units
	for(unsigned int j=0;j<h.size();j++){
	  h[j] = T(1.0)/(T(1.0) + math::exp(-h[j], T(20.0f)));

	  // added discretization DEBUG
#if 1
	  T r = rng.uniform();
	  if(h[j] > r) h[j] = T(1.0);
	  else         h[j] = T(0.0);
#endif
	}
	
	// PW += h.outerproduct(v)/T(NUMSAMPLES);
	T scaling = T(1.0)/T(NUMSAMPLES);
	assert(addouterproduct(PW, scaling, h, v) == true);
	
	pa += v/T(NUMSAMPLES);
	pb += h/T(NUMSAMPLES);
      }
    

      // samples from Pmodel(v) approximatedly
      std::vector< math::vertex<T> > modelsamples; // model samples

      for(unsigned int i=0;i<NUMSAMPLES;i++)
      {
	if(running) if(*running == false) break;
	
	const unsigned int index = rng.rand() % samples.size();
	auto v = samples[index];

	
	auto h = W*v + b;
	
	// 1. hidden units: calculates sigma(a_j)
	for(unsigned int j=0;j<h.size();j++){
	  T aj = T(1.0)/(T(1.0) + math::exp(-h[j], T(20.0f)));

	  T r = rng.uniform();
	  
	  if(aj > r) h[j] = T(1.0); // discretization step
	  else       h[j] = T(0.0);
	}
	
	
	for(unsigned int l=0;l<CDk;l++){
	  v = h*W + a;
	  
	  // 1. visible units: calculates sigma(a_j)
	  for(unsigned int j=0;j<v.size();j++){
	    T aj = T(1.0)/(T(1.0) + math::exp(-v[j], T(20.0f)));

	    T r = rng.uniform();
	    
	    if(aj > r) v[j] = T(1.0); // discretization step
	    else       v[j] = T(0.0);
	  }
	  
	  h = W*v + b;
	  
	  // 2. hidden units: calculates sigma(a_j)
	  for(unsigned int j=0;j<(h.size()-0);j++){
	    T aj = T(1.0)/(T(1.0) + math::exp(-h[j], T(20.0f)));
	    
	    T r = rng.uniform();
	    
	    if(aj > r) h[j] = T(1.0); // discretization step
	    else       h[j] = T(0.0);
	  }
	  
	}

	modelsamples.push_back(v);
      }
    
      // calculates negative gradient by using samples from Pmodel(v)
      NW.zero();
      na.zero();
      nb.zero();
      
      // Pmodel
      for(unsigned int i=0;i<NUMSAMPLES;i++){
	if(running) if(*running == false) break;
	
	// const unsigned int index = rng.rand() % modelsamples.size();
	const unsigned int index = i;
	auto v = modelsamples[index];
	
	auto h = W*v + b;
	
	// hidden units
	for(unsigned int j=0;j<h.size();j++){
	  h[j] = T(1.0)/(T(1.0) + math::exp(-h[j], T(20.0f)));

	  // added discretization DEBUG
#if 1
	  T r = rng.uniform();
	  if(h[j] > r) h[j] = T(1.0);
	  else         h[j] = T(0.0);
#endif
	}
	
	// NW += h.outerproduct(v)/T(NUMSAMPLES);
	T scaling = T(1.0)/T(NUMSAMPLES);
	assert(addouterproduct(NW, scaling, h, v) == true);
	
	
	na += v/T(NUMSAMPLES);
	nb += h/T(NUMSAMPLES);
      }
    
      // updates weights according to CD rule + ADAPTIVE LEARNING RATE
      {
	// W += lambda*(P - N);
#if 0
	W += lambda*(PW - NW);
	a += lambda*(pa - na);
	b += lambda*(pb - nb);

#else
	T coef1 = T(1.0);
	T coef2 = T(1.0/0.9);
	T coef3 = T(0.9);
	
	auto W1 = W + coef1*lambda*(PW - NW);
	auto W2 = W + coef2*lambda*(PW - NW);
	auto W3 = W + coef3*lambda*(PW - NW);
	
	auto a1 = a + coef1*lambda*(pa - na);
	auto a2 = a + coef2*lambda*(pa - na);
	auto a3 = a + coef3*lambda*(pa - na);
	
	auto b1 = b + coef1*lambda*(pb - nb);
	auto b2 = b + coef2*lambda*(pb - nb);
	auto b3 = b + coef3*lambda*(pb - nb);	
	
	T error1 = reconstructionError(samples, 10, a1, b1, W1);
	T error2 = reconstructionError(samples, 10, a2, b2, W2);
	T error3 = reconstructionError(samples, 10, a3, b3, W3);
	
	if(error2 < error1 && error2 < error3){
	  W = W2;
	  a = a2;
	  b = b2;
	  lambda *= coef2;
	}
	else if(error3 < error2 && error3 < error1){
	  W = W3;
	  a = a3;
	  b = b3;
	  lambda *= coef3;
	}
	else{
	  W = W1;
	  a = a1;
	  b = b1;
	  lambda *= coef1;
	}
#endif
	
      }
    }
  
    // calculates mean reconstruction error in samples and looks for convergence
    latestError = reconstructionError(samples, 1000, a, b, W);
    
    if(verbose == 1){
      double error = 0.0;
      math::convert(error, latestError);

      printf("\n");
      printf("%d/%d EPOCH. R: %f\n", e+1, EPOCHS, error);
      fflush(stdout);
    }
    else if(verbose == 2){
      char buffer[128];
      double error = 0.0;
      math::convert(error, latestError);

      snprintf(buffer, 128, "BBRBM::learnWeights(): %d/%d epoch. reconstruction error: %f\n", e+1, EPOCHS, error);
      fflush(stdout);
    }

  }
  
  
  return latestError;
  
  // alternatively we could look at delta parameters when trying to decide if we have converged
  // return (math::frobenius_norm(originalW)/originalW.size()); 
}


// calculates parameters using LBFGS 2nd order optimization and
// CD-3 to estimate gradient
template <typename T>
T BBRBM<T>::learnWeights2(const std::vector< math::vertex<T> >& samples,
			  const unsigned int EPOCHS,
			  const int verbose,
			  const bool* running)
{
  if(EPOCHS <= 0) return T(INFINITY);
  if(samples.size() <= 0) return T(INFINITY);
  if(samples[0].size() != getVisibleNodes()) return T(INFINITY);

  whiteice::dataset<T> ds;

  // this->initializeWeights();
  this->setUData(samples);

  ds.createCluster("input", getVisibleNodes());
  ds.add(0, samples);

  math::vertex<T> x0;
  this->getParametersQ(x0);

  whiteice::LBFGS_BBRBM<T>* optimizer[EPOCHS];
		
  math::vertex<T> x(x0);
  T error;
  int last_iter = -1;
  unsigned int iters = 0;

  std::list<T> errors; // epoch errors... used to detect convergence

  auto bestx = x0;
  auto besterror = T(INFINITY);

  whiteice::linear_ETA<double> eta;
  eta.start(0.0, EPOCHS+1);
  eta.update(0.0);

  for(unsigned int i=0;i<EPOCHS;i++){
    this->getParametersQ(x0);
		  
    optimizer[i] = new whiteice::LBFGS_BBRBM<T>(*this, ds, false);
    optimizer[i]->minimize(x0);

    last_iter = -1;
    iters = 0;
    
    while(true){
      if(!optimizer[i]->isRunning() || optimizer[i]->solutionConverged()){
	break;
      }

      if(running) if(*running == false) break;
		    
      optimizer[i]->getSolution(x, error, iters);
      
      if((signed)iters > last_iter){
	if(verbose == 1){
	  std::cout << "ITER " << iters << ": error = " << error << std::endl;
	  fflush(stdout);
	}
	else if(verbose == 2){
	  char buffer[128];
	  double tmp = 0.0;
	  whiteice::math::convert(tmp, error);
	  
	  snprintf(buffer, 128, "BBRBM::learnWeights(): iter %d: error = %f",
		   iters, tmp);
	  
	  whiteice::logging.info(buffer);
	}
	
	last_iter = iters;
      }
      
      sleep(1);
    }
    
    optimizer[i]->getSolution(x, error, iters);
    this->setParametersQ(x);

    eta.update(i+1); // this epoch has been calculated..

    if(verbose == 1){
      std::cout << "EPOCH " << i << "/" << EPOCHS
		<< ": error = " << error
		<< " ETA " << eta.estimate()/(3600.0) << " hour(s)"
		<< std::endl;
      fflush(stdout);
    }
    else if(verbose == 2){
      char buffer[128];
      double tmp;
      whiteice::math::convert(tmp, error);

      snprintf(buffer, 128, "BBRBM::learnWeights(): epoch %d/%d: error = %f ETA %f hour(s)",
	       i, EPOCHS, tmp, eta.estimate()/3600.0);
      whiteice::logging.info(buffer);
      
    }

    errors.push_back(error);

    // keeps last 20 error terms in epochs
    while(errors.size() > 20)
      errors.pop_front();
    
    // smart convergence detection..
    if(errors.size() > 20){
      T m = 0.0, v = 0.0;
      
      for(auto& e : errors){
	m += e;
	v += e*e;
      }

      m /= T(errors.size());
      v /= T(errors.size());

      v -= m*m;

      T statistic = sqrt(v)/m;

      if(statistic < T(1.005)){ // only 0.5% change in optimized values
	                        // within the latest 20 steps
	delete optimizer[i];
	
	if(verbose == 1){
	  std::cout << "Early stopping.. convergence detected."
		    << std::endl;
	  fflush(stdout);
	}
	else if(verbose == 2)
	  whiteice::logging.info("BBRBM::learnWeights(): Early stopping.. convergence detected.");
	
	break;
      }
	
    }


    if(optimizer[i]->getSolution(x, error, iters)){
      if(error < besterror){
	besterror = error;
	bestx = x;
      }
    }

    delete optimizer[i];
    optimizer[i] = NULL;
  }

  
  this->setParametersQ(bestx);
  
  return besterror;
}

  
// mean error per element per sample (does not increase if dimensions or number of samples increase)
template <typename T>
T BBRBM<T>::reconstructionError(const std::vector< math::vertex<T> >& samples,
				// number of samples to use
				// from samples to estimate reconstruction error
				unsigned int N, 
				const math::vertex<T>& a,
				const math::vertex<T>& b,
				// weight matrix (parameters) to use
				const math::matrix<T>& W) const 
{
  T error = T(0.0);

  for(unsigned int n=0;n<N;n++){
    const unsigned int index = rng.rand() % samples.size();
    const auto& s = samples[index];

    error += reconstructionError(s, a, b, W) / T(N*s.size());
  }

  return error;
}


// weight matrix (parameters) to use
template <typename T>
T BBRBM<T>::reconstructionError(const math::vertex<T>& s,
				const math::vertex<T>& a,
				const math::vertex<T>& b,
				const math::matrix<T>& W) const 
{
  try{
    math::vertex<T> v(this->v);
    math::vertex<T> h(this->h);

    v = s;

    {
      h = W*v + b;
      
      // 1. hidden units: calculates sigma(a_j)
      for(unsigned int j=0;j<(h.size()-0);j++){
	T aj = T(1.0)/(T(1.0) + math::exp(-h[j], T(20.0f)));

	T r = rng.uniform();
	
	if(aj > r) h[j] = T(1.0); // discretization step
	else       h[j] = T(0.0);
      }

      v = h*W + a;
      
      // 1. visible units: calculates sigma(a_j)
      for(unsigned int j=0;j<(v.size()-0);j++){
	T aj = T(1.0)/(T(1.0) + math::exp(-v[j], T(20.0f)));

	T r = rng.uniform();
	
	if(aj > r) v[j] = T(1.0); // discretization step
	else       v[j] = T(0.0);
      }
      
    }

    auto delta = v - s;
    return delta.norm(); // reconstruction error for this vector
  }
  catch(std::exception& e){
    std::cout << "Uncaught exception (reconstructionError()): " << e.what() << std::endl;
    return T(INFINITY); // incorrect data - infinity error
  }
}


////////////////////////////////////////////////////////////
// TODO implement me! (implement LBFGS_BBRBM after gradient descent one works)

template <typename T>
bool BBRBM<T>::setUData(const std::vector< math::vertex<T> >& samples)
{
  if(samples.size() == 0) return false;
  if(samples[0].size() != a.size()) return false;
  
  this->Usamples = samples;
  
  return true;
}

template <typename T>
unsigned int BBRBM<T>::qsize() const  // size of q vector q = [vec(W)]
{
  return (a.size() + b.size() + W.xsize()*W.ysize());
}
  
// converts (W) parameters into q vector
template <typename T>
bool BBRBM<T>::convertParametersToQ(const math::matrix<T>& W,
				    const math::vertex<T>& a,
				    const math::vertex<T>& b, 
				    math::vertex<T>& q) const
{
  q.resize(a.size() + b.size() + W.ysize()*W.xsize());

  q.write_subvertex(a, 0);
  q.write_subvertex(b, a.size());
  W.save_to_vertex(q, a.size()+b.size());

  return true;
}
  
// converts q vector into parameters (W, a, b)
template <typename T>
bool BBRBM<T>::convertQToParameters(const math::vertex<T>& q,
				    math::matrix<T>& W,
				    math::vertex<T>& a,
				    math::vertex<T>& b) const 
				    
{
  a.resize(this->a.size());
  b.resize(this->b.size());
  W.resize(this->W.ysize(), this->W.xsize());
  
  if(q.size() != (a.size()+b.size()+W.ysize()*W.xsize()))
    return false;
  
  try{
    q.subvertex(a, 0, a.size());
    q.subvertex(b, a.size(), b.size());
    math::vertex<T> w(W.ysize()*W.xsize());
    q.subvertex(w, (a.size()+b.size()), w.size());
    W.load_from_vertex(w);
    
    return true;
  }
  catch(std::exception& e){
    return false;
  }
}
  
// sets (W) parameters according to q vector
template <typename T>
bool BBRBM<T>::setParametersQ(const math::vertex<T>& q)
{
  return convertQToParameters(q, W, a, b);
}

template <typename T>
bool BBRBM<T>::getParametersQ(math::vertex<T>& q) const
{
  return convertParametersToQ(W, a, b, q);
}

// keeps parameters within sane levels
// (clips overly large parameters and NaNs)
template <typename T>
void BBRBM<T>::safebox(math::vertex<T>& a, math::vertex<T>& b,
		       math::matrix<T>& W) const
{
  for(unsigned int i=0;i<a.size();i++){
    if(whiteice::math::isnan(a[i])) a[i] = T(0.0); //printf("anan"); }
    if(a[i] < T(-10e6)) a[i] = T(-10e6); //printf("aclip"); }
    if(a[i] > T(+10e6)) a[i] = T(+10e6); //printf("aclip"); }
  }

  for(unsigned int i=0;i<b.size();i++){
    if(whiteice::math::isnan(b[i])) b[i] = T(0.0); //printf("bnan"); }
    if(b[i] < T(-10e6)) b[i] = T(-10e6); //printf("bclip"); }
    if(b[i] > T(+10e6)) b[i] = T(+10e6); //printf("bclip"); }
  }

  for(unsigned int j=0;j<W.ysize();j++){
    for(unsigned int i=0;i<W.xsize();i++){
      if(whiteice::math::isnan(W(j,i))) W(j,i) = T(0.0); //printf("Wnan"); }
      if(W(j,i) < T(-10e6)) W(j,i) = T(-10e6); //printf("Wclip"); }
      if(W(j,i) > T(+10e6)) W(j,i) = T(+10e6); //printf("Wclip"); }
    }
  }
}
  
template <typename T>
T BBRBM<T>::U(const math::vertex<T>& q) const 
{
  math::vertex<T> a;
  math::vertex<T> b;
  math::matrix<T> W;

  convertQToParameters(q, W, a, b);
  safebox(a, b, W);
  // currently uses only reconstruction error to estimate U(p)
  return reconstructionError(Usamples, 1000, a, b, W);
}

// uses CD-1 to estimate gradient
template <typename T>
math::vertex<T> BBRBM<T>::Ugrad(const math::vertex<T>& q) 
{
  math::vertex<T> a;
  math::vertex<T> b;
  math::matrix<T> W;
  
  convertQToParameters(q, W, a, b);
  safebox(a, b, W);

  // calculates gradient vector
  math::vertex<T> ga(a);
  math::vertex<T> gb(b);
  math::matrix<T> gW(W);

  math::vertex<T> pa(a);
  math::vertex<T> pb(b);
  math::matrix<T> PW(W);

  math::vertex<T> na(a);
  math::vertex<T> nb(b);
  math::matrix<T> NW(W);

  const unsigned int NUMSAMPLES = 100;
  const unsigned int CDk = 1;

  {
    // calculates positive gradient from NUMSAMPLES examples Pdata(v))
    PW.zero();
    pa.zero();
    pb.zero();
    
    // Pdata
    for(unsigned int i=0;i<NUMSAMPLES;i++){
      const unsigned int index = rng.rand() % Usamples.size();
      auto v = Usamples[index];
      
      auto h = W*v + b;
      
      // hidden units
      for(unsigned int j=0;j<h.size();j++){
	h[j] = T(1.0)/(T(1.0) + math::exp(-h[j], T(20.0f)));

	// added discretization DEBUG
#if 1
	T r = rng.uniform();
	if(h[j] > r) h[j] = T(1.0);
	else         h[j] = T(0.0);
#endif
      }
      
      // PW += h.outerproduct(v)/T(NUMSAMPLES);
      T scaling = T(1.0)/T(NUMSAMPLES);
      assert(addouterproduct(PW, scaling, h, v) == true);
      
      pa += v/T(NUMSAMPLES);
      pb += h/T(NUMSAMPLES);
    }
    

    // samples from Pmodel(v) approximatedly
    std::vector< math::vertex<T> > modelsamples; // model samples
    
    for(unsigned int i=0;i<NUMSAMPLES;i++)
    {
      const unsigned int index = rng.rand() % Usamples.size();
      auto v = Usamples[index];
      
      
      auto h = W*v + b;
      
      // 1. hidden units: calculates sigma(a_j)
      for(unsigned int j=0;j<(h.size()-0);j++){
	T aj = T(1.0)/(T(1.0) + math::exp(-h[j], T(20.0f)));

	T r = rng.uniform();
	
	if(aj > r) h[j] = T(1.0); // discretization step
	else       h[j] = T(0.0);
      }
	
      
      for(unsigned int l=0;l<CDk;l++){
	v = h*W + a;
	
	// 1. visible units: calculates sigma(a_j)
	for(unsigned int j=0;j<(v.size()-0);j++){
	  T aj = T(1.0)/(T(1.0) + math::exp(-v[j], T(20.0f)));
	  T r = rng.uniform();
	  
	  if(aj > r) v[j] = T(1.0); // discretization step
	  else       v[j] = T(0.0);
	}
	
	h = W*v + b;
	
	// 2. hidden units: calculates sigma(a_j)
	for(unsigned int j=0;j<(h.size()-0);j++){
	  T aj = T(1.0)/(T(1.0) + math::exp(-h[j], T(20.0f)));
	  T r = rng.uniform();
	  
	  if(aj > r) h[j] = T(1.0); // discretization step
	  else       h[j] = T(0.0);
	}
	
      }
      
      modelsamples.push_back(v);
    }
    
    // calculates negative gradient by using samples from Pmodel(v)
    NW.zero();
    na.zero();
    nb.zero();
    
    // Pmodel
    for(unsigned int i=0;i<NUMSAMPLES;i++){
      auto v = modelsamples[i];
      
      auto h = W*v + b;
      
      // hidden units
      for(unsigned int j=0;j<h.size();j++){
	h[j] = T(1.0)/(T(1.0) + math::exp(-h[j], T(20.0f)));
	
	// added discretization DEBUG
#if 1
	T r = rng.uniform();
	if(h[j] > r) h[j] = T(1.0);
	else         h[j] = T(0.0);
#endif
      }
      
      // NW += h.outerproduct(v)/T(NUMSAMPLES);
      T scaling = T(1.0)/T(NUMSAMPLES);
      assert(addouterproduct(NW, scaling, h, v) == true);
      
      na += v/T(NUMSAMPLES);
      nb += h/T(NUMSAMPLES);
    }

    ga = (pa - na);
    gb = (pb - nb);
    gW = (PW - NW);
  }

  math::vertex<T> grad;
  grad.resize(this->qsize());
  
  convertParametersToQ(gW, ga, gb, grad);
  
  return grad;
}

  template <typename T>
  bool BBRBM<T>::diagnostics() const
  {
    whiteice::logging.info("BBRBM::diagnostics()");

    T maxvalue_a = T(-INFINITY);
    T minvalue_a = T(+INFINITY);

    T maxvalue_b = T(-INFINITY);
    T minvalue_b = T(+INFINITY);

    T maxvalue_W = T(-INFINITY);
    T minvalue_W = T(+INFINITY);

    for(unsigned int i=0;i<a.size();i++){
      if(abs(a[i]) > maxvalue_a) maxvalue_a = abs(a[i]);
      if(abs(a[i]) < minvalue_a) minvalue_a = abs(a[i]);
    }

    for(unsigned int i=0;i<b.size();i++){
      if(abs(b[i]) > maxvalue_b) maxvalue_b = abs(b[i]);
      if(abs(b[i]) < minvalue_b) minvalue_b = abs(b[i]);
    }

    for(unsigned int j=0;j<W.ysize();j++){
      for(unsigned int i=0;i<W.xsize();i++){
	if(abs(W(j,i)) > maxvalue_W) maxvalue_W = abs(W(j,i));
	if(abs(W(j,i)) < minvalue_W) minvalue_W = abs(W(j,i));
      }
    }

    double temp[6];

    whiteice::math::convert(temp[0], minvalue_a);
    whiteice::math::convert(temp[1], maxvalue_a);
    whiteice::math::convert(temp[2], minvalue_b);
    whiteice::math::convert(temp[3], maxvalue_b);
    whiteice::math::convert(temp[4], minvalue_W);
    whiteice::math::convert(temp[5], maxvalue_W);

    char buffer[256];
    snprintf(buffer, 256,"a min=%f max=%f b min=%f max=%f W min=%f max=%f",
	     temp[0], temp[1],
	     temp[2], temp[3],
	     temp[4], temp[5]);
    
    whiteice::logging.info(buffer);

    return true;
  }

////////////////////////////////////////////////////////////

// load & saves RBM data from/to file
template <typename T>
bool BBRBM<T>::load(const std::string& filename) 
{
    whiteice::dataset<T> file;

  if(file.load(filename) == false)
    return false;

  if(file.getNumberOfClusters() != 7)
    return false;

  std::vector<std::string> names;
  if(file.getClusterNames(names) == false) return false;

  bool found = false;
  for(auto& n : names)
    if(n == "whiteice::BBRBM file"){
      found = true;
      break;
    }

  if(found == false)
    return false; // unknown filetype
				  
  // tries to load data..
  std::vector< math::vertex<T> > data;

  if(file.getData(0, data) == false) return false;
  a = data[0];
  if(file.getData(1, data) == false) return false;
  b = data[0];
  if(file.getData(2, data) == false) return false;
  W.resize(b.size(), a.size());
  if(W.load_from_vertex(data[0]) == false)
    return false;
  if(file.getData(3, data) == false) return false;
  h = data[0];
  if(file.getData(4, data) == false) return false;
  v = data[0];
  if(file.getData(5, data) == false) return false;
  if(data.size() > 0)
    Usamples = data;
  else
    Usamples.clear();

  // some sanity checks..
  if(b.size() != W.ysize()) return false;
  if(a.size() != W.xsize()) return false;
  if(a.size() != v.size())  return false;
  if(b.size() != h.size())  return false;
  
  return true;
}

template <typename T>
bool BBRBM<T>::save(const std::string& filename) const 
{
  whiteice::dataset<T> file;

  file.createCluster("a", a.size());
  if(file.add(0, a) == false) return false;
  file.createCluster("b", b.size());
  if(file.add(1, b) == false) return false;
  file.createCluster("W", W.size());

  math::vertex<T> vecW(W.xsize()*W.ysize());
  W.save_to_vertex(vecW);
  
  if(file.add(2, vecW) == false) return false;

  if(file.createCluster("h", h.size()) == false) return false;
  file.add(3, h);
  if(file.createCluster("v", v.size()) == false) return false;
  file.add(4, v);

  if(Usamples.size() > 0){
    file.createCluster("Usamples", Usamples[0].size());
    if(file.add(5, Usamples) == false) return false;
  }
  else{
    file.createCluster("Usamples", 0);
  }

  file.createCluster("whiteice::BBRBM file", 1);
  
  return file.save(filename);
}

  
  template <typename T>
  void BBRBM<T>::sigmoid(const math::vertex<T>& input,
			 math::vertex<T>& output) const
  {
    output.resize(input.size());

    for(unsigned int i=0;i<input.size();i++){
      output[i] = T(1.0)/(T(1.0) + math::exp(-input[i], T(20.0f)));
    }
  }

  
  template <typename T>
  void BBRBM<T>::sigmoid(math::vertex<T>& x) const
  {
    for(unsigned int i=0;i<x.size();i++){
      x[i] = T(1.0)/(T(1.0) + math::exp(-x[i], T(20.0f)));
    }
  }

  
  //template class BBRBM< float >;
  //template class BBRBM< double >;
  template class BBRBM< math::blas_real<float> >;
  template class BBRBM< math::blas_real<double> >;

} /* namespace whiteice */
