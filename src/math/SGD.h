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
		      const T lrate = T(1e-6),
		      const unsigned int MAX_ITERS=1,
		      const unsigned int MAX_NO_IMPROVE_ITERS = 100);

	void setKeepWorse(bool keepFlag){ keepWorse = keepFlag; }
	bool getKeepWorse() const { return keepWorse; }

	// x is the best parameter found, y is training error and
	// iterations is number of training iterations.
        bool getSolution(vertex<T>& x, T& y, unsigned int& iterations) const;
	
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
      
        bool overfit;
	bool keepWorse; // do we save worse solutions
	
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
