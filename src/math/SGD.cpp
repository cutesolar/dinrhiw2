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

      thread_is_running_cond.notify_all();

      this->iterations = 0;
      unsigned int no_improve_iterations = 0;
      std::list<T> errors; // used for convergence check

      vertex<T> grad;
      
      vertex<T> x(bestx);
      T real_besty = besty;

      T current_lrate = lrate;
      
      // stops if given number of iterations has passed or no improvements in N iters
      // or if instructed to stop. Additionally, in the loop there is convergence check
      // to check if to stop computing.
      while((iterations < MAX_ITERS || MAX_ITERS == 0) &&
	    (no_improve_iterations) < MAX_NO_IMPROVE_ITERS &&
	    thread_running)
      {
	grad = Ugrad(x);

	x -= current_lrate*grad; // minimization

	heuristics(x);

	const T ynew = getError(x);

	if(ynew < real_besty){
	  no_improve_iterations = 0;
	}
	else{
	  no_improve_iterations++;
	}
	
	if(ynew < besty || keepWorse){
	  {
	    std::lock_guard<std::mutex> lock(solution_mutex);
	    
	    this->besty = ynew;
	    this->bestx = x;
	  }
	}

	if(ynew < real_besty){ // results improved
	  real_besty = ynew;
	  
	  if(adaptive_lrate)
	    current_lrate *= T(1.25f);
	}
	else{ // result didn't improve
	  if(adaptive_lrate)
	    current_lrate *= T(0.5f);
	}

	if(current_lrate < T(1e-20))
	  current_lrate = T(1e-20);
	else if(current_lrate > T(1e20))
	  current_lrate = T(1e20);

	iterations++;

	
	// smart convergence check: checks if (st.dev. / mean) <= 0.001 (<= 0.1%)
	if(smart_convergence_check){
	  errors.push_back(real_besty); // NOTE: getError() must return >= 0.0 values

	  if(errors.size() >= 30){
	  
	    while(errors.size() > 30)
	      errors.pop_front();
	    
	    T m = T(0.0f);
	    T s = T(0.0f);
	    
	    for(const auto& e : errors){
	      m += e;
	      s += e*e;
	    }
	    
	    m /= errors.size();
	    s /= errors.size();

	    s -= m*m;
	    s = sqrt(abs(s));

	    T r = T(0.0f);

	    if(m > T(0.0f))
	      r = s/m;

	    if(r <= T(0.005f)){ // convergence: 0.1% st.dev. when compared to mean.
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
    
    template class SGD< float >;
    template class SGD< double >;
    template class SGD< blas_real<float> >;
    template class SGD< blas_real<double> >;    
    
  };
};

