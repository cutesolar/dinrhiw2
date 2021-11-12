/*
 * SGD - Stochastic Gradient Descent optimizer (abstract class)
 * 
 */

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "RNG.h"


#ifndef __whiteice__SGD_h
#define __whiteice__SGD_h


namespace whiteice
{
  namespace math
  {
    
    template <typename T=blas_real<float> >
      class SGD
      {
      public:
        SGD(bool overfit=false); // overfit: do not use early stopping via getError() function
        virtual ~SGD();
      
      protected:
        /* optimized function */

        virtual T U(const vertex<T>& x) const = 0;
        virtual vertex<T> Ugrad(const vertex<T>& x) const = 0;
      
        // heuristically improve solution x during SGD optimization
        virtual bool heuristics(vertex<T>& x) const = 0;
      
      public:
        /* 
	 * error function we are (indirectly) optimizing)
	 * can be same as U(x) if there are no uncertainties,
	 * but can be different if we are optimizing 
	 * statistical model and want to have early stopping
	 * in order to prevent optimizing to statistical
	 * noise.
	 */
        virtual T getError(const vertex<T>& x) const = 0;
      
      
	// x0 is starting point
        bool minimize(vertex<T> x0,
		      const T lrate = T(1e-2),
		      const unsigned int MAX_ITERS=0, // stop when max no improve iters has passed
		      const unsigned int MAX_NO_IMPROVE_ITERS = 50);

	// do we keep each iterations result as the current best solution..
	void setKeepWorse(bool keepFlag){ keepWorse = keepFlag; }
	bool getKeepWorse() const { return keepWorse; }

	// do we stop when average improvement is less than 0.1% of the mean
	void setSmartConvergenceCheck(bool checkFlag){ smart_convergence_check = checkFlag; }
	bool getSmartConvergenceCheck() const { return smart_convergence_check; }

	// do we adapt lrate if results don't improve
	void setAdaptiveLRate(bool adaptiveFlag){ adaptive_lrate = adaptiveFlag; }
	bool getAdaptiveLRate() const { return adaptive_lrate; }

	// x is the best parameter found, y is training error and
	// iterations is number of training iterations.
        bool getSolution(vertex<T>& x, T& y, unsigned int& iterations) const;

	bool getSolutionStatistics(T& y, unsigned int& iterations) const;
	
	// continues, pauses, stops computation
        bool continueComputation();
        bool pauseComputation();
	bool stopComputation();

        // returns true if solution converged and we cannot
        // find better solution
        bool solutionConverged() const;

        // returns true if optimization thread is running
        bool isRunning() const;

	
      private:
      
	bool box_values(vertex<T>& x) const;
	
            
        // best solution found
	vertex<T> bestx; 
	T besty;
        volatile unsigned int iterations;

	T lrate;
	unsigned int MAX_ITERS;
	unsigned int MAX_NO_IMPROVE_ITERS;
      
        bool overfit = false;
	bool keepWorse = false; // do we save worse solutions
	bool smart_convergence_check = true; // do we stop when average improvement is less than 0.1% of the mean
	bool adaptive_lrate = true; // do we adapt lrate between iterations
	
        volatile bool sleep_mode, thread_running, solution_converged;
      
        volatile int thread_is_running;
        mutable std::mutex thread_is_running_mutex;
        mutable std::condition_variable thread_is_running_cond;
        
        std::thread* optimizer_thread;
        mutable std::mutex sleep_mutex, thread_mutex, solution_mutex;
	
      private:
	void optimizer_loop();
	
      };
    
  };
};



namespace whiteice
{
  namespace math
  {
    
  extern template class SGD< float >;
  extern template class SGD< double >;
  extern template class SGD< blas_real<float> >;
  extern template class SGD< blas_real<double> >;
    
  };
};

#endif
