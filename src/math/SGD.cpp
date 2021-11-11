/*
 * Implments Stochasic Gradient Descent absract class to be inherited by a specific optimization class.
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

      if(lrate <= T(0.0f) || MAX_ITERS <= 0 || MAX_NO_IMPROVE_ITERS <= 0){
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
      this->MAX_ITERS = MAX_ITERS;
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

      // TODO: IMPLEMENT Stochastic Gradient Descent!!

      thread_is_running_cond.notify_all();

      this->iterations = 0;
      unsigned int no_improve_iterations = 0;

      vertex<T> grad;
      
      vertex<T> x(bestx);
      T y = besty;

      while((iterations < MAX_ITERS) &&
	    (no_improve_iterations) < MAX_NO_IMPROVE_ITERS &&
	    thread_running)
      {
	grad = Ugrad(x);

	x -= lrate*grad; // minimization

	heuristics(x);

	const T ynew = getError(x);

	if(ynew < y || keepWorse){
	  std::lock_guard<std::mutex> lock(solution_mutex);
	  
	  this->besty = ynew;
	  this->bestx = x;
	  y = ynew;

	  no_improve_iterations = 0;
	}
	else{
	  no_improve_iterations++;
	}
	
	iterations++;

	while(sleep_mode && thread_running){
	  std::chrono::milliseconds duration(200);
	  std::this_thread::sleep_for(duration);
	}

      }

      {
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

