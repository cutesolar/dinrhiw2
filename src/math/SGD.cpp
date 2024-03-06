/*
 * Implments Stochasic Gradient Descent absract class to be inherited by a specific optimization class.
 * 
 *
 * TODO: allow implementing class to enable dropout heuristic in neural networks.
 *
 */

#include "SGD.h"
#include "linear_equations.h"
#include <iostream>
#include <list>
#include <functional>

#include <unistd.h>

#ifdef _GLIBCXX_DEBUG

#ifndef _WIN32

#undef __STRICT_ANSI__
#include <float.h>
#include <fenv.h>

#endif

#endif


#ifdef WINOS
#include <windows.h>
#endif


namespace whiteice
{
  namespace math
  {

    template <typename T>
    SGD<T>::SGD(bool overfit)
    {
      thread_running = false;
      sleep_mode = false;
      solution_converged = false;
      optimizer_thread = nullptr;
      this->overfit = overfit;
      this->keepWorse = false;
      this->smart_convergence_check = true;
      this->adaptive_lrate = true;
    }
    
 
    template <typename T>
    SGD<T>::~SGD()
    {
      thread_mutex.lock();
      
      if(thread_running){
	thread_running = false;
	
	// waits for thread to stop running
	// std::unique_lock<std::mutex> lock(thread_is_running_mutex);
	// thread_is_running_cond.wait_for(lock, std::chrono::milliseconds(1000)); // 1 second
      }
      
      if(optimizer_thread){
	optimizer_thread->join();
	delete optimizer_thread;
      }
      optimizer_thread = nullptr;

      thread_mutex.unlock();
    }
    
    
    template <typename T>
    bool SGD<T>::minimize(vertex<T> x0,
			  const T lrate,
			  const unsigned int MAX_ITERS,
			  const unsigned int MAX_NO_IMPROVE_ITERS)
    {
      thread_mutex.lock();
      
      if(thread_running || optimizer_thread != nullptr){
	thread_mutex.unlock();
	return false;
      }

      if(lrate <= T(0.0f) || MAX_NO_IMPROVE_ITERS <= 0){
	thread_mutex.unlock();
	return false;
      }

      // calculates initial solution
      solution_mutex.lock();
      {
	heuristics(x0);
	
	this->bestx = x0;
	this->besty = getError(x0);
	
	iterations  = 0;
      }
      solution_mutex.unlock();

      this->lrate = lrate;
      this->MAX_ITERS = MAX_ITERS; // if MAX_ITERS == zero, don't stop until convergence (no improve)
      this->MAX_NO_IMPROVE_ITERS = MAX_NO_IMPROVE_ITERS;
      
      thread_running = true;
      sleep_mode = false;
      solution_converged = false;
      thread_is_running = 0;
      
      try{
	optimizer_thread =
	  new std::thread(std::bind(&SGD<T>::optimizer_loop,
				    this));
	
	// optimizer_thread->detach();
      }
      catch(std::exception& e){
	thread_running = false;
	thread_mutex.unlock();
	return false;
      }
      
      thread_mutex.unlock();
      
      return true;
    }
    
    
    template <typename T>
    bool SGD<T>::getSolution(vertex<T>& x, T& y, unsigned int& iterations) const
    {
      // gets the best found solution
      solution_mutex.lock();
      {
	x = bestx;
	y = besty;
	iterations = this->iterations;
      }
      solution_mutex.unlock();
      
      return true;
    }


    template <typename T>
    bool SGD<T>::getSolutionStatistics(T& y, unsigned int& iterations) const
    {
      std::lock_guard<std::mutex> lock(solution_mutex);

      y = besty;
      iterations = this->iterations;

      return true;
    }
    
    
    // continues, pauses, stops computation
    template <typename T>
    bool SGD<T>::continueComputation()
    {
      sleep_mutex.lock();
      {
	sleep_mode = false;
      }
      sleep_mutex.unlock();
      
      return true;
    }
    
    
    template <typename T>
    bool SGD<T>::pauseComputation()
    {
      sleep_mutex.lock();
      {
	sleep_mode = true;
      }
      sleep_mutex.unlock();
      
      return true;
    }
    
    
    template <typename T>
    bool SGD<T>::stopComputation()
    {
      // FIXME threading sychnronization code is BROKEN!
      
      thread_mutex.lock();
      
      if(thread_running == false){
	thread_mutex.unlock();
	return false;
      }
      
      thread_running = false;
      
      // waits for thread to stop running
      // std::unique_lock<std::mutex> lock(thread_is_running_mutex);
      // thread_is_running_cond.wait_for(lock, std::chrono::milliseconds(1000)); // 1 sec
      
      if(optimizer_thread){
	optimizer_thread->join();
	delete optimizer_thread;
      }
      optimizer_thread = nullptr;
      
      thread_mutex.unlock();
      
      return true;
    }


    // returns true if solution converged and we cannot
    // find better solution
    template <typename T>
    bool SGD<T>::solutionConverged() const
    {
      return solution_converged;
    }
    
    // returns true if optimization thread is running
    template <typename T>
    bool SGD<T>::isRunning() const
    {
      return thread_running;
    }
    
    
    // limit values to sane interval (no too large values)
    template <typename T>
    bool SGD<T>::box_values(vertex<T>& x) const
    {
      // don't allow values larger than 10^4
      for(unsigned int i=0;i<x.size();i++)
	if(x[i] > T(1e4)) x[i] = T(1e4);
	else if(x[i] < T(-1e4)) x[i] = T(-1e4);

      return true;
    }
    
    
    template <typename T>
    void SGD<T>::optimizer_loop()
    {
#ifdef _GLIBCXX_DEBUG  
#ifndef _WIN32    
      {
	// enables FPU exceptions
	feenableexcept(FE_INVALID | FE_DIVBYZERO);
      }
#endif
#endif

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

      
      thread_is_running_cond.notify_all();

      this->iterations = 0;
      unsigned int no_improve_iterations = 0;
      unsigned int no_improve_iterations_count = 0;
      std::list<T> errors; // used for convergence check

      vertex<T> grad;
      
      vertex<T> x(bestx);
      T real_besty = besty;

      const T lrate0 = lrate;
      T mistep_lrate = lrate;
      bool use_mistep_lrate = false;
      int mistep_go_worse = 0;

      bool recalculate_gradient = true;

      
      // stops if given number of iterations has passed or no improvements in N iters
      // or if instructed to stop. Additionally, in the loop there is convergence check
      // to check if to stop computing.
      while((iterations < MAX_ITERS || MAX_ITERS == 0) &&
	    (no_improve_iterations) < MAX_NO_IMPROVE_ITERS &&
	    thread_running)
      {
	if(recalculate_gradient == true)
	  grad = Ugrad(x);

	recalculate_gradient = true; // as the default calculates gradient each iteration..

	auto old_x = x;

	auto lratef = lrate;

	if(use_mistep_lrate){
	  use_mistep_lrate = false;
	  // x -= mistep_lrate*grad; // minimization

	  lratef = mistep_lrate;

	  //std::cout << "lrate = " << mistep_lrate << std::endl;
	}
	else{
	  // x -= lrate*grad; // minimization

	  //std::cout << "lrate = " << lrate << std::endl;
	}

	auto delta_grad = grad;

	for(unsigned int j=0;j<grad.size();j++){
	  for(unsigned int k=0;k<grad[0].size();k++){
	    delta_grad[j][k] *= lratef[0];
	  }
	}

	/*
	//if(mistep_go_worse)
	{
	  // random scaling of the gradient [0.90,1.10]
	  
	  for(unsigned int j=0;j<grad.size();j++){
	    auto scaling = (T(rng.uniform())*T(0.20f) + T(0.90f))[0];
	    for(unsigned int k=0;k<grad[0].size();k++){
	      delta_grad[j][k] *= scaling;
	    }
	  }
	}
	*/


	x -= delta_grad; 
	
	heuristics(x);

	const T ynew = getError(x);

	// std::cout << "SGD::getError() = " << ynew << std::endl;

	if(ynew[0] < (T(0.99999)*real_besty)[0]){ // 0.01% reductions or smaller mean there is no improvement
	  no_improve_iterations = 0;
	  no_improve_iterations_count = 0;
	}
	else{
	  no_improve_iterations++;
	  no_improve_iterations_count++;
	  //std::cout << "NO IMPROVE: " << no_improve_iterations_count << std::endl;
	}
	

	bool worse = false;
	
	if(ynew[0] < besty[0] || keepWorse){
	  {
	    std::lock_guard<std::mutex> lock(solution_mutex);
	    
	    this->besty = ynew;
	    this->bestx = x;
	  }

#if 1
	  if(lrate[0] < 0.01)
	    lrate[0] = sqrt(lrate[0]);
#endif
	}
	else{
	  if((mistep_go_worse > 0 && ynew[0] < (T(1.250)*besty)[0]) /* || 
								      (ynew - besty)[0] < 1e-5*/){ 
	    // go worse direction [just once]
	    if(mistep_go_worse) mistep_go_worse--;
	    worse = true; 
	  }
	  else{
	    if(ynew[0] >= (T(1.250)*besty)[0]){
	      x = old_x; // don't go to worse directions..

	      recalculate_gradient = false; // no need to calculate gradient again.. 
	    }
	    else{
	      worse = true;
	    }
	  }
	}

	if(ynew[0] < real_besty[0]){ // results improved
	  real_besty = ynew;
	  
	  if(adaptive_lrate) // was: increase learning rate by 10%
	    lrate *= T(2.00f);
	}
	else{ // result didn't improve, quickly reduce learning rate by 50%
	  if(adaptive_lrate){
	    if(worse == false) lrate *= T(0.50f); // was 0.50 = 50%
	    else lrate *= T(1.00f); // go to worse direction so increase still lrate

	  }
	}


	
	if(lrate[0] < T(1e-10)[0] || no_improve_iterations_count >= 20){
	  // lrate = T(1e-10);

	  // resets LRATE and goes to worse direction
	  T e = T(0.0f);
	  e = T(0.35)*T(rng.uniform()) + T(0.15);
	  
	  mistep_lrate[0] = pow(lrate[0], e[0]);
	  
	  use_mistep_lrate = true;
	  std::cout << "MISTEP LRATE: " << mistep_lrate << std::endl;
	  lrate[0] = mistep_lrate[0];
	  mistep_go_worse = 2;

	  if((rng.rand()%100) < 10){ // 10% probability to reset x 
	    x = bestx; // resets x too [don't work in practice]
	    std::cout << "RESET X" << std::endl;
	  }
	  
	  no_improve_iterations_count = 0;
	}
	else if(lrate > T(1e10)){
	  lrate = T(1e10);
	  no_improve_iterations_count = 0;
	}

	iterations++;

	
	// smart convergence check: checks if (st.dev. / mean) <= 0.001 (<= 0.1%)
	if(smart_convergence_check){
	  errors.push_back(real_besty); // NOTE: getError() must return >= 0.0 values

	  if(errors.size() >= 100){
	  
	    while(errors.size() > 100)
	      errors.pop_front();

	    // make all values to be positive
	    T min_value = *errors.begin();

	    for(const auto& e : errors)
	      if(e < min_value) min_value = e;

	    if(min_value < T(0.0f)) min_value = min_value - T(1.0f);
	    else min_value = T(-0.01f);
	    
	    
	    T m = T(0.0f);
	    T s = T(0.0f);
	    
	    for(const auto& e : errors){
	      m += (e-min_value);
	      s += (e-min_value)*(e-min_value);
	    }
	    
	    m /= errors.size();
	    s /= errors.size();

	    s -= m*m;
	    s = sqrt(abs(s));

	    T r = T(0.0f);

	    if(m > T(0.0f))
	      r = s/m;

	    if(r[0] <= T(0.005f)[0]){ // convergence: 0.1% st.dev. when compared to mean.
	      solution_converged = true;
	      break;
	    }

	  }
	}

	

	while(sleep_mode && thread_running){
	  std::chrono::milliseconds duration(200);
	  std::this_thread::sleep_for(duration);
	}

      }

      
      {
	solution_converged = true;
	
	// std::lock_guard<std::mutex> lock(thread_mutex); // needed or safe??
	
	thread_running = false; // very tricky here, writing false => false or true => false SHOULD BE ALWAYS SAFE without locks
      }
      // thread_is_running_cond.notify_all(); // waiters have to use wait_for() [timeout milliseconds] as it IS possible to miss notify_all()
    }
    
    
    // explicit template instantations
    
    //template class SGD< float >;
    //template class SGD< double >;
    
    template class SGD< blas_real<float> >;
    template class SGD< blas_real<double> >;    

    
    template class SGD< superresolution<
			  blas_real<float>,
			  modular<unsigned int> > >;
    
    template class SGD< superresolution<
			  blas_real<double>,
			  modular<unsigned int> > >;
    
    
  };
};

