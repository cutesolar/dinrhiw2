
#include "HMC_diffeq.h"


// probability functions for hamiltonian MC sampling
template <typename T>
T HMC_diffeq<T>::U(const math::vertex<T>& q, bool useRegulizer) const
{
  auto datacp(this->data);
  
  std::vector< math::vertex<T> > xdata;
  
  // now simulate training datapoints
  simulate_diffeq_model2(this->nnet,
			 start_point,
			 (times[times.size()-1]-times[0]).c[0],
			 xdata, times);

  datacp.clearData(0);
  datacp.add(0, xdata);

  whiteice::nnetwork<T> nnet(this->nnet);
  nnet.importdata(q);
  
  T E = T(0.0f);
    
  // E = SUM 0.5*e(i)^2
#pragma omp parallel shared(E)
  {
    math::vertex<T> err, tmp;
    T e = T(0.0f);
    
#pragma omp for nowait schedule(auto)
    for(unsigned int i=0;i<datacp.size(0);i++){
      nnet.calculate(datacp.access(0,i), tmp);
      err = datacp.access(1, i) - tmp;
      e = e  + T(0.5f)*(err*err)[0];
    }
    
#pragma omp critical (mvjrwerfweghx)
    {
      E = E + e;
    }
  }
  
  E /= this->sigma2;
  
  E /= this->temperature;
  
  E += T(0.5)*this->alpha*(q*q)[0];
  
  return (E);
}


template <typename T>
math::vertex<T> HMC_diffeq<T>::Ugrad(const math::vertex<T>& q) const
{
  auto datacp(this->data);
  
  std::vector< math::vertex<T> > xdata;
  
  // now simulate training datapoints
  simulate_diffeq_model2(this->nnet,
			 start_point,
			 (times[times.size()-1]-times[0]).c[0],
			 xdata, times);

  //printf("data sizes: %d %d %d\n", (int)this->data.size(0), (int)this->data.size(1), (int)xdata.size());
  
  datacp.clearData(0);
  datacp.add(0, xdata);

  //printf("data sizes: %d %d %d\n", (int)this->data.size(0), (int)this->data.size(1), (int)xdata.size());
  
  
  math::vertex<T> sum;
  sum.resize(q.size());
  sum.zero();
  
  // positive gradient
#pragma omp parallel shared(sum)
  {
    whiteice::nnetwork<T> nnet(this->nnet);
    nnet.importdata(q);
    
    // const T ninv = T(1.0f); // T(1.0f/data.size(0));
    math::vertex<T> sumgrad, grad, err;
    sumgrad.resize(q.size());
    sumgrad.zero();
    
#pragma omp for nowait schedule(auto)
    for(unsigned int i=0;i<datacp.size(0);i++){
      nnet.input() = datacp.access(0, i);
      nnet.calculate(true);
      err = nnet.output() - datacp.access(1,i);
      
      if(nnet.mse_gradient(err, grad) == false){
	std::cout << "gradient failed." << std::endl;
	assert(0); // FIXME
      }
      
      sumgrad += grad;
    }
    
#pragma omp critical (mfkrewiorweqqqa)
    {
      sum += sumgrad;
    }
  }
  
  sum /= this->sigma2;
  
  sum /= this->temperature; // scales gradient with temperature
  
  
  sum += T(0.5)*this->alpha*q;
  
  // sum.normalize();
  
  return (sum);
}


// calculates mean error for the latest N samples, 0 = all samples
template <typename T>
T HMC_diffeq<T>::getMeanError(unsigned int latestN) const
{
  std::vector< math::vertex<T> > sample;
  
  // copies selected nnetwork configurations
  // from global variable (synchronized) to local memory;
  {
    std::lock_guard<std::mutex> lock(this->solution_lock);
    
    if(!latestN) latestN = this->samples.size();
    if(latestN > this->samples.size()) latestN = this->samples.size();
    
    for(unsigned int i=this->samples.size()-latestN;i<this->samples.size();i++){
      sample.push_back(this->samples[i]);
    }
  }
  
  
  T sumErr = T(0.0f);

  auto datacp(this->data);
  
  for(unsigned int i=0;i<sample.size();i++)
    {
      //////////////////////////////////////////////////////////////////////

      whiteice::nnetwork<T> nnet2(this->nnet);
      std::vector< math::vertex<T> > xdata;
      
      // now simulate training datapoints
      simulate_diffeq_model2(nnet2,
			     start_point,
			     (times[times.size()-1]-times[0]).c[0],
			     xdata, times);
      
      //printf("data sizes: %d %d %d\n", (int)this->data.size(0), (int)this->data.size(1), (int)xdata.size());
      
      datacp.clearData(0);
      datacp.add(0, xdata);
      
      //////////////////////////////////////////////////////////////////////
      
      T E = T(0.0f);
      
      // E = SUM 0.5*e(i)^2
#pragma omp parallel shared(E)
      {
	whiteice::nnetwork<T> nnet(this->nnet);
	nnet.importdata(sample[i]);
      
	math::vertex<T> err;
	T e = T(0.0f);
	
#pragma omp for nowait schedule(auto)
	for(unsigned int i=0;i<datacp.size(0);i++){
	  nnet.input() = datacp.access(0, i);
	  nnet.calculate(false);
	  err = datacp.access(1, i) - nnet.output();
	  // T inv = T(1.0f/err.size());
	  err = (err*err);
	  e = e  + T(0.5f)*err[0];
	}
	
#pragma omp critical (mogjfisdrwe)
	{
	  E = E + e;
	}
      }
      
      sumErr += E;
    }
  
  
  if(sample.size() > 0){
    sumErr /= T((float)sample.size());
    sumErr /= T((float)datacp.size(0));
  }
  
  return sumErr;
}





template class HMC_diffeq< math::blas_real<float> >;
template class HMC_diffeq< math::blas_real<double> >;

