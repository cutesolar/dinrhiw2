/*
 * optimizing weighting between different neural network models
 * 
 */

#include "WeightingOptimizer.h"
#include "Log.h"
#include "nnetwork.h"
#include "dataset.h"


#include <math.h>
#include <vector>
#include <functional>


namespace whiteice
{

  template <typename T>
  WeightingOptimizer<T>::WeightingOptimizer(const dataset<T>& data_,
					    const std::vector< nnetwork<T> >& experts_) :
    data(data_), experts(experts_)
  {
    assert(data.getNumberOfClusters() >= 2);
    assert(experts.size() > 0);
    assert(data.size(0) == data.size(1));
    assert(data.size(0) > 0);
    
    optimizer_thread = nullptr;
    thread_running = false;

    min_error = T(INFINITY);
    hasModelFlag = false;
  }

  
  template <typename T>
  WeightingOptimizer<T>::~WeightingOptimizer()
  {
    this->stopOptimize();
  }
  
  
  template <typename T>
  bool WeightingOptimizer<T>::startOptimize(const nnetwork<T>& weighting)
  {
    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_running || optimizer_thread != nullptr)
      return false;

    if(experts.size() < 0 || data.getNumberOfClusters() < 0) return false;

    if(data.size(0) < 0) return false;

    if(weighting.input_size() != data.dimension(0) ||
       weighting.output_size() != 1)
      return false;

    {
      std::lock_guard<std::mutex> lock(solution_mutex);

      this->weighting = weighting;
      hasModelFlag = false;
      min_error = T(INFINITY);
    }

    thread_running = true;

    try{
      optimizer_thread = new std::thread(std::bind(&WeightingOptimizer<T>::optimizer_loop, this));
    }
    catch(std::exception& e){
      optimizer_thread = nullptr;
      thread_running = false;
    }

    return true;
  }

  
  template <typename T>
  bool WeightingOptimizer<T>::stopOptimize()
  {
    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_running == false) return false;

    thread_running = false;

    if(optimizer_thread){
      optimizer_thread->join();
      delete optimizer_thread;
      optimizer_thread = nullptr;
    }
    
    return true;
  }

  
  template <typename T>
  bool WeightingOptimizer<T>::isRunning() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_running && optimizer_thread) return true;
    else return false;
  }
  
  
  template <typename T>
  bool WeightingOptimizer<T>::getSolution(nnetwork<T>& weighting, T& error) const
  {
    std::lock_guard<std::mutex> lock(solution_mutex);

    if(hasModelFlag == false) return false;
    
    weighting = this->weighting;
    error = min_error;

    return true;
  }


  template <typename T>
  T WeightingOptimizer<T>::getError(const nnetwork<T>& weighting,
				    const dataset<T>& data,
				    const bool regularize) const
  {
    T error = T(0.0f);

    math::vertex<T> w;

    math::matrix<T> M;
    M.resize(this->experts.size(), data.dimension(1));

    for(unsigned int i=0;i<data.size(0);i++){
      const auto& input = data.access(0, i);
      const auto& output = data.access(1, i);

      for(unsigned int k=0;k<experts.size();k++){
	experts[k].calculate(input, w);

	M.rowcopyfrom(w, k);
      }

      weighting.calculate(input, w);

      auto err = output - w*M;

      error += (err*err)[0];
    }

    error *= T(1.0f/data.size(0));
    error *= T(0.50f);

    if(regularize){
      weighting.exportdata(w);

      error += T(0.5f)*regularizer*(w*w)[0];
    }

    return error;
  }

  
  template <typename T>
  void WeightingOptimizer<T>::optimizer_loop()
  {
    // 0. sets optimizer thread priority to minimum background thread
    
    {
      sched_param sch_params;
      int policy = SCHED_FIFO; // SCHED_RR
      
      pthread_getschedparam(pthread_self(), &policy, &sch_params);

#ifdef linux
      policy = SCHED_IDLE; // in linux we can set idle priority
#endif
      sch_params.sched_priority = sched_get_priority_min(policy);
      
      if(pthread_setschedparam(pthread_self(),
			       policy, &sch_params) != 0){
      }

#ifdef WINOS
      SetThreadPriority(GetCurrentThread(),
			THREAD_PRIORITY_IDLE);
#endif
      
    }

    // 1. divides data to training and testing datasets

    whiteice::dataset<T> dtrain, dtest;

    dtrain = data;
    dtest  = data;

    dtrain.clearData(0);
    dtrain.clearData(1);
    
    dtest.clearData(0);
    dtest.clearData(1);

    unsigned int counter = 0;

    while(((dtrain.size(0) == 0) || (dtrain.size(1) == 0) ||
	   (dtest.size(0) == 0) || (dtest.size(1) == 0)) && counter < 10){

      dtrain.clearData(0);
      dtrain.clearData(1);
      
      dtest.clearData(0);
      dtest.clearData(1);

      for(unsigned int i=0;i<data.size(0);i++){
	const unsigned int r = (rng.rand() & 3);
	
	const math::vertex<T>& in  = data.access(0, i);
	const math::vertex<T>& out = data.access(1, i);
	
	if(r != 0){ // 75% will got to training data
	  dtrain.add(0, in,  true);
	  dtrain.add(1, out, true);
	  
	}
	else{
	  dtest.add(0, in,  true);
	  dtest.add(1, out, true);
	}
      }

      counter++;
    }

    if(counter >= 10){
      dtrain = data;
      dtest  = data;
    }

    auto weighting = this->weighting;
    
    // 1. calculates minimum error with current solution
    {
      T err = getError(weighting, dtest, false);

      this->min_error = err;

      {
	char buffer[128];
	snprintf(buffer, 128, "WeightOptimizer::optimizer_loop(). Initial error: %f", real(err).c[0]);
	whiteice::logging.info(buffer);
      }
    }

    unsigned int iterations = 0;
    const unsigned int MAXITERS = 1000;

    
    while(thread_running && iterations < MAXITERS){
      
      T lrate = T(1.0f);

      do{
	// 2. calculates gradient + regularizer

	whiteice::logging.info("WeightOptimizer: calculates gradient");

	math::vertex<T> gradient, w, w0;
	gradient.resize(weighting.exportdatasize());
	gradient.zero();
	
	{
	  math::vertex<T> tmp, w, err;
	  
	  math::matrix<T> M;
	  M.resize(experts.size(), experts[0].output_size());
	  
	  std::vector< math::vertex<T> > bpdata;
	  
	  for(unsigned int i=0;i<dtrain.size(0);i++){
	    
	    // calculates element e = [y_i - w(x_i)*M(x)]*M(x)^t
	    
	    const auto& input = dtrain.access(0, i);
	    const auto& output = dtrain.access(1, i);
	    
	    for(unsigned int k=0;k<experts.size();k++){
	      experts[k].calculate(input, tmp);
	      
	      M.rowcopyfrom(tmp, k);
	    }
	    
	    weighting.calculate(input, w, bpdata);
	    
	    err = (output - w*M);
	    err = M*err;
	    
	    assert(weighting.mse_gradient(err, bpdata, tmp) == true);
	    
	    gradient += tmp;
	  }
	  
	  gradient /= T(dtrain.size(0));
	  
	  
	  // adds a regularizer term
	  {
	    weighting.exportdata(w);
	    
	    gradient += regularizer*w;
	  }	  
	}

	whiteice::logging.info("WeightOptimizer: calculates linesearch..");

	lrate = sqrt(lrate);
	lrate *= T(100.0f);

	w0 = w;

	T delta_value = T(0.0f);

	T err;
	T dtrain_min_err = getError(weighting, dtrain, true);
	T prev_err = dtrain_min_err;

	// 3. calculates step length that will reduce optimized function
	do{
	  w = w0;

	  w -= lrate*gradient;

	  weighting.importdata(w);

	  err = getError(weighting, dtrain, true);
	  
	  delta_value = prev_err - err;
	  
	  if(delta_value <= T(0.0f)){ // error increases => smaller lrate
	    lrate *= T(0.50f);
	  }
	  else if(delta_value > T(0.0f)){ // error decrease => can increase lrate
	    lrate *= T(2.0f);
	  }
	}
	while(delta_value <= T(0.0) && lrate >= T(1e-30) && thread_running);

	iterations++;
	
	{
	  if(err < dtrain_min_err){
	    T test_error = getError(weighting, dtest, false);

	    if(test_error < min_error){
	      std::lock_guard<std::mutex> lock(solution_mutex);

	      min_error = test_error;
	      this->weighting.importdata(w);

	      {
		char buffer[128];
		snprintf(buffer, 128, "WeightOptimizer::optimizer_loop(). Better error found (%d/%d): %f",
			 iterations, MAXITERS,
			 real(err).c[0]);
		whiteice::logging.info(buffer);
	      }
	    }
	  }
	}
	
      }
      while(lrate >= T(1e-30) && iterations < MAXITERS && thread_running);

      // cannot find better solution so we restart with a new random weights
      weighting.randomize();
    }

    
    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      
      if(thread_running) thread_running = false;
    }
    
  }
  

  
  template class WeightingOptimizer<whiteice::math::blas_real<float> >;
  template class WeightingOptimizer<whiteice::math::blas_real<double> >;
  
  //template class WeightingOptimizer<whiteice::math::blas_complex<float> >;
  //template class WeightingOptimizer<whiteice::math::blas_complex<double> >;
  
};
