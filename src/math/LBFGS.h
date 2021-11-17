/*
 * Limited Memory-Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer
 * minimizes the target error function.
 */

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "RNG.h"


#ifndef LBFGS_h
#define LBFGS_h


namespace whiteice
{
  namespace math
  {
    
    template <typename T=blas_real<float> >
      class LBFGS
      {
      public:
        LBFGS(bool overfit=false); // overfit: do not use early stopping via getError() function
        virtual ~LBFGS();
      
      protected:
        /* optimized function */

        virtual T U(const vertex<T>& x) const = 0;
        virtual vertex<T> Ugrad(const vertex<T>& x) const = 0;
      
        // heuristically improve solution x during LBFGS optimization
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
        bool minimize(vertex<T> x0);

	// x is the best parameter found, y is training error (from getError()) and
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

	// NOTE: This DOES NOT work!
	// follow only gradient instead of 2nd order H aprox
	void setGradientOnly(bool gradientOnly=true){
	  this->onlygradient = gradientOnly;
	}

	void setUseWolfeConditions(bool useWolfe = true){
	  this->use_wolfe = useWolfe; // SLOW but should guarantee convergence to grad == zero point.
	}

	// zero disable iterations check
	void setMaxIterations(const unsigned int iters = 0){
	  this->MAXITERS = iters;
	}
	
      private:
      
        bool linesearch(vertex<T>& xn,
			T& scale,
			const vertex<T>& x,
			const vertex<T>& d) const;

	bool box_values(vertex<T>& x) const;
            

        bool wolfe_conditions(const vertex<T>& x0,
			      const T& alpha,
			      const vertex<T>& p) const;

	
	// M = MEMORY SIZE (5 adapt quickly and don't get stuck to wrong gradient like with 10)
	const unsigned int LBFGS_MEMORY = 5;
      
        // best solution found
	vertex<T> bestx; 
	T besty;
        volatile unsigned int iterations;

	unsigned int MAXITERS; // maximum number of iterations to run or zero if continue indefinitely
      
        bool overfit;
	bool onlygradient; // only follow gradient (no 2nd order aprox)
	bool use_wolfe; // do we use wolfe conditions in line search?
	
        volatile bool sleep_mode, thread_running, solution_converged;
      
        volatile int thread_is_running;
        mutable std::mutex thread_is_running_mutex;
        mutable std::condition_variable thread_is_running_cond;
        
        std::thread* optimizer_thread;
        mutable std::mutex sleep_mutex, thread_mutex, solution_mutex;

      protected:
	// whiteice::RNG<T> rng;
	
      private:
	void optimizer_loop();
	
      };
    
  };
};



namespace whiteice
{
  namespace math
  {
    
  extern template class LBFGS< float >;
  extern template class LBFGS< double >;
  extern template class LBFGS< blas_real<float> >;
  extern template class LBFGS< blas_real<double> >;
    
  };
};

#endif
