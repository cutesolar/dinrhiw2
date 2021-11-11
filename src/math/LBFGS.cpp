
#include "LBFGS.h"
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
    LBFGS<T>::LBFGS(bool overfit)
    {
      thread_running = false;
      sleep_mode = false;
      solution_converged = false;
      optimizer_thread = nullptr;
      this->overfit = overfit;
      this->onlygradient = false;
      this->use_wolfe = true; // results are not always good unless Wolfe conditions is not used
    }
    
 
    template <typename T>
    LBFGS<T>::~LBFGS()
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
    bool LBFGS<T>::minimize(vertex<T> x0)
    {
    	thread_mutex.lock();

    	if(thread_running || optimizer_thread != nullptr){
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

    	thread_running = true;
    	sleep_mode = false;
    	solution_converged = false;
    	thread_is_running = 0;

    	try{
	  optimizer_thread =
	    new std::thread(std::bind(&LBFGS<T>::optimizer_loop,
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
    bool LBFGS<T>::getSolution(vertex<T>& x, T& y, unsigned int& iterations) const
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
    bool LBFGS<T>::continueComputation()
    {
    	sleep_mutex.lock();
    	{
    		sleep_mode = false;
    	}
    	sleep_mutex.unlock();

    	return true;
    }
    
    
    template <typename T>
    bool LBFGS<T>::pauseComputation()
    {
    	sleep_mutex.lock();
    	{
    		sleep_mode = true;
    	}
    	sleep_mutex.unlock();

    	return true;
    }
    
    
    template <typename T>
    bool LBFGS<T>::stopComputation()
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
    bool LBFGS<T>::solutionConverged() const
    {
      return solution_converged;
    }
    
    // returns true if optimization thread is running
    template <typename T>
    bool LBFGS<T>::isRunning() const
    {
      return thread_running;
    }
    

    template <typename T>
    bool LBFGS<T>::linesearch(vertex<T>& xn,
			      T& scale,
			      const vertex<T>& x,
			      const vertex<T>& d) const
    {
      // finds the correct scale first
      // (exponential search)
      
      // moves scale 2**(e/2) towards 1
      scale = sqrt(scale); // T(1.0); // complete line search..
      // scale = T(1.0); // complete linesearch
      
      vertex<T> localbestx = x + d;
      T localbest  = T(10e20);
      unsigned int found = 0;
      
      vertex<T> t;
      
      T best_alpha = scale * T(::pow(2.0f, -30)); // minimum possible step length
      localbestx = x;
      box_values(localbestx);
      localbest = U(localbestx);
      
      T alpha = T(1.0f);
      
      // k = 0
      {
	alpha  = scale;
	T tvalue;
	
	t = x + alpha*d;
	box_values(t); // limit values	  
	tvalue = U(t);

	if(use_wolfe == true){
	  if(wolfe_conditions(x, alpha, d)){
	    {
	      best_alpha = alpha;
	      localbest = tvalue;
	      localbestx = t;
	    }
	  }
	}
	else{
	  if(tvalue < localbest){
	    {
	      best_alpha = alpha;
	      localbest = tvalue;
	      localbestx = t;
	    }
	  }
	}
	
      }
      
      int k = 1;    	
      
      while(found <= 0 && k <= 30){ // min 2**(-30) = 10e-9 step length

	T tvalue = T(0.0);
	
	if(k <= 20) // was: 10..
	{ // don't allow for large k values
	  alpha  = scale * T(::pow(2.0f, k));
	  
	  t = x + alpha*d;
	  box_values(t); // limit values
	  tvalue = U(t);


	  if(use_wolfe == true){
	    if(wolfe_conditions(x, alpha, d)){
	      {
		best_alpha = alpha;
		localbest = tvalue;
		localbestx = t;
		found++; continue;
	      }
	    }
	  }
	  else{
	    if(tvalue < localbest){
	      {
		best_alpha = alpha;
		localbest = tvalue;
		localbestx = t;
		found++; continue;
	      }
	    }
	  }
	  
	}
	
	alpha  = scale * T(::pow(2.0f, -k));
	
	t = x + alpha*d;
	box_values(t); // limit values
	tvalue = U(t);
	
	if(use_wolfe == true){
	  if(wolfe_conditions(x, alpha, d)){
	    {
	      best_alpha = alpha;
	      localbest = tvalue;
	      localbestx = t;
	      found++; continue;
	    }
	  }
	}
	else{
	  if(tvalue < localbest){
	    {
	      best_alpha = alpha;
	      localbest = tvalue;
	      localbestx = t;
	      found++; continue;
	    }
	  }
	}
	
#if 0
	// HACK: (DISABLED)
	// heuristics: allows going to worse solution with 20% prob
	// (small step length values to gradient direction)
	if((rng.rand() % 5) == 0 && tvalue < T(10.0)){
	  best_alpha = alpha;
	  localbest = tvalue;
	  localbestx = t;
	  found++;
	  break;
	}
#endif
	
	k++;
      }
      
      xn = localbestx;
      scale = best_alpha;
      
      return (found > 0);
    }


    // limit values to sane interval (no too large values)
    template <typename T>
    bool LBFGS<T>::box_values(vertex<T>& x) const
    {
      // don't allow values larger than 10^4
      for(unsigned int i=0;i<x.size();i++)
	if(x[i] > T(1e4)) x[i] = T(1e4);
	else if(x[i] < T(-1e4)) x[i] = T(-1e4);

      return true;
    }
    
    
    template <typename T>
    bool LBFGS<T>::wolfe_conditions(const vertex<T>& x0,
				    const T& alpha,
				    const vertex<T>& p) const
    {
      T c1 = T(0.0001f);
      T c2 = T(0.9f);
      
      bool cond1 = (U(x0 + alpha*p) <= (U(x0) + c1*alpha*(p*Ugrad(x0))[0]));
      bool cond2 = ((p*Ugrad(x0 + alpha*p))[0] >= c2*(p*Ugrad(x0))[0]);

      return (cond1 && cond2);
    }
    
    
    template <typename T>
    void LBFGS<T>::optimizer_loop()
    {
#ifdef _GLIBCXX_DEBUG  
#ifndef _WIN32    
      {
	// enables FPU exceptions
	feenableexcept(FE_INVALID | FE_DIVBYZERO);
      }
#endif
#endif
      
      vertex<T> d, g; // gradient
      vertex<T> x(bestx), xn;
      vertex<T> s, q;
      
      T scale = T(1.0);
      
      T y = besty;
      
      std::list<T> ratios;
      unsigned int reset = 0;
      const unsigned int RESET = 5;

      // history size is large (15) should try value 5 and change to it if results do not become worse.
      const unsigned int M = 10;  // M = MEMORY SIZE (10 could be a good compromise)
      
      std::list< vertex<T> > yk;
      std::list< vertex<T> > sk;
      std::list< T > rk;
      
      thread_is_running_cond.notify_all();
      
      while(thread_running){
	try{
	  // we keep iterating until we converge (later) or
	  // the real error starts to increase
	  
	  if(overfit == false){
	    ratios.push_back(besty);
	    
	    while(ratios.size() > 20)
	      ratios.pop_front();
	    
	    T mean_ratio = 0.0f;
	    T inv = 1.0f/ratios.size();
	    
	    for(auto r : ratios){
	      mean_ratio += (r/besty)*inv;
	    }
	    
	    // mean_ratio = math::pow(mean_ratio, inv);
	    // std::cout << "ratio = " << mean_ratio << std::endl;
	    // std::cout << "ratio = " << mean_ratio << std::endl;
	    
	    
	    // 50% increase from the minimum found
	    //if(mean_ratio > T(1.50f) && iterations > 25){
	    //  break;
	    //}
	    
	    // std::cout << "mean ratio: " << mean_ratio << std::endl;
	    
	    
	    if(mean_ratio < T(1.005f) && iterations > 20){
	      solution_converged = true; // last 20 iterations showed less than 0.5% change..
	      break;
	    }
	  }
	  
	  ////////////////////////////////////////////////////////////
	  g = Ugrad(x);
	  
	  
	  if(thread_running == false) break; // cancellation point
	  
	  // d = -H*g; // linsolve(H, d, -g);
	  // calculates aprox hessian product (L-BFGS method)
	  
	  if(sk.size() > 0 && onlygradient == false){
	    vertex<T> q(g);
	    
	    vertex<T> alpha;
	    vertex<T> beta;
	    
	    alpha.resize(sk.size());
	    beta.resize(sk.size());
	    alpha.zero();
	    beta.zero();
	    
	    {
	      typename std::list< vertex<T> >::iterator si;
	      typename std::list< vertex<T> >::iterator yi;
	      typename std::list< T >::iterator ri;
	      unsigned int i = 0;
	      
	      for(si = sk.begin(), yi = yk.begin(), ri = rk.begin(), i=0;si!=sk.end();si++,yi++,ri++,i++)
	      {
		alpha[i] = (*ri) * ( ((*si)*q)[0] );
		q = q - alpha[i] * (*yi);
	      }
	    }
	    

	    const T epsilon = T(10e-12);
	    T divider = ((*yk.begin())*(*yk.begin()))[0];
	    if(divider < epsilon) divider = epsilon;
	    
	    T Hk = ((*yk.begin()) * (*sk.begin()))[0] / divider;
	    
	    vertex<T> z = Hk*q;
	    
	    {
	      typename std::list< vertex<T> >::reverse_iterator si;
	      typename std::list< vertex<T> >::reverse_iterator yi;
	      typename std::list< T >::reverse_iterator ri;
	      int i = 0;
	      
	      const unsigned int m = sk.size();
	      
	      for(si = sk.rbegin(), yi = yk.rbegin(), ri = rk.rbegin(), i=m-1;si!=sk.rend();si++,yi++,ri++,i--)
	      {
		beta[i] = (*ri) * ((*yi) * z)[0];
		z = z + (*si) * (alpha[i] - beta[i]);
	      }
	      
	    }
	    
	    d = -z;
	  }
	  else{
	    d = -g; // initially just follow the gradient
	  }
	  
	  
	  // cancellation point
	  if(thread_running == false) break;

	  // scale = 1.0
	  
	  // linear search finds xn = x + alpha*d
	  // so that U(xn) is minimized
	  if(linesearch(xn, scale, x, d) == false){
	    // reset => (we try to just follow gradient instead)
	    sk.clear();
	    yk.clear();
	    rk.clear();
	    
#if 1
	    if(reset < RESET){
	      reset++;
	      iterations++;
	      continue;
	    }
	    else{
	      // cannot improve after RESET re-tries
	      // we still cannot improve the result (even after reset)
	      {
		// solution has converged
		
		solution_converged = true;
		
		break; // we stop computation as we cannot find better solution
	      }
	    }
#endif
	  }
	  else{
	    reset = 0;
	  }
	  
	  
	  if(scale <= T(0.0)) scale = 1.0; // fixes the case when scaling goes to zero
	  
	  heuristics(xn); // heuristically improve proposed next xn (might break L-BFGS algorithm!!)
	  
	  
	  // cancellation point
	  if(thread_running == false) break;
	  
	  // y = U(xn);
	  y = getError(xn);
	  
	  
	  if(y < besty){
	    std::lock_guard<std::mutex> lock(solution_mutex);
	    bestx = xn;
	    besty = y;
	  }
	  
	  s = xn - x;
	  vertex<T> yy = Ugrad(xn) - g; // Ugrad(xn) - Ugrad(x)
	  auto syy = (s*yy)[0];
	  
	  T r = T(10e10f); // division by zero work-a-round..
	  if(abs(syy) > T(0.0))
	    r = T(1.0f)/syy;
	  
	  sk.push_front(s);
	  yk.push_front(yy);
	  rk.push_front(r);
	  
	  while(sk.size() > M) sk.pop_back();
	  while(yk.size() > M) yk.pop_back();
	  while(rk.size() > M) rk.pop_back();
	  
	  x = xn;
	  
	  iterations++;
	}
	catch(std::exception& e){
	  std::cout << "ERROR: Unexpected exception: "
		    << e.what() << std::endl;
	  fflush(stdout);
	}
	
	////////////////////////////////////////////////////////////
	
	// checks if thread has been ordered to sleep
	while(sleep_mode){
	  std::chrono::milliseconds duration(1000);
	  std::this_thread::sleep_for(duration);
	}
      }

      
      
      thread_running = false; // very tricky here, writing false => false or true => false SHOULD BE ALWAYS SAFE without locks
      // thread_is_running_cond.notify_all(); // waiters have to use wait_for() [timeout milliseconds] as it IS possible to miss notify_all()
      
    }
    
    
    // explicit template instantations
    
    template class LBFGS< float >;
    template class LBFGS< double >;
    template class LBFGS< blas_real<float> >;
    template class LBFGS< blas_real<double> >;    
    
  };
};

