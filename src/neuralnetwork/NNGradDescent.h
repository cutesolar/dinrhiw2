/*
 * Parallel neural network gradient descent optimizer
 *
 * - keeps looking for the best possible solution forever
 *   (fresh restart after convergence to local minima)
 *
 *
 */


#include <thread>
#include <list>
#include <vector>
#include <map>
#include <mutex>
#include <condition_variable>

#include "dinrhiw_blas.h"
#include "dataset.h"
#include "dinrhiw.h"
#include "nnetwork.h"

#ifndef NNGradDescent_h
#define NNGradDescent_h


namespace whiteice
{
  namespace math
  {
    template <typename T=blas_real<float> >
      class NNGradDescent
      {
      public:

      // deep_pretraining - pretrains new weights of sigmoidal neural network (GBRBM+BBRBM)
      //                    (don't do anything if network is not sigmoidal)
      // 
      NNGradDescent(bool heuristics = false, bool deep_pretraining = false);
      NNGradDescent(const NNGradDescent<T>& grad);
      ~NNGradDescent();

      // sets and gets minibatch settings for estimating gradient
      void setUseMinibatch(bool minibatch = true);
      bool getUseMinibatch() const;

      // sets overfitting (uses all data for training and no separate testing dataset)
      // (overfitting gives much WORSE results so it SHOULD NOT be used.)
      void setOverfit(bool overfit = true);
      bool getOverfit() const;

      // sets non-normalization of error values when normalize_error = false
      void setNormalizeError(bool normalize_error = true);
      bool getNormalizeError();

      // if true uses Minimum Norm Error ||y-f(x)|| instead of the default
      // MSE (Minimum Squared Error) ||y-f(x)||^2 as a error measure
      void setMNE(bool usemne = true);
      bool getUseMNE() const;

      // whether to add alpha*0.5*||w||^2 term to error to keep
      // weight values from exploding. It is RECOMMENDED to enable
      // this for complex valued neural networks because complex
      // non-linearities might easily explode to large values
      //
      // 0.01 seem to work rather well when complex weights are N(0,I)/dim(W(k,:))
      // and data is close to N(0,I) too.
      void setRegularizer(const T alpha = T(0.01f));

      // returns regularizer value, zero means regularizing is disabled
      T getRegularizer() const;
      
      /*
       * starts the optimization process using data as 
       * the dataset as a training and testing data 
       * (implements early stopping)
       *
       * Uses neural network with architecture arch.
       *
       * Executes NTHREADS in parallel when looking for
       * the optimal solution and goes max to 
       * MAXITERS iterations when looking for gradient
       * descent solution
       * 
       * dropout - whether to use dropout heuristics when training
       * initiallyUseNN = true => first try to use parameter nn weights
       */
      bool startOptimize(const whiteice::dataset<T>& data,
			 const whiteice::nnetwork<T>& nn,
			 unsigned int NTHREADS,
			 unsigned int MAXITERS = 0xFFFFFFFF,
			 bool dropout = false,
			 bool initiallyUseNN = true);
      
      /*
       * Returns true if optimizer is running
       */
      bool isRunning();


      /*
       * Returns true if heuristics estimate that optimizer has converged.
       * "stdev/mean" of most recent errors is less than percentage => convergence.
       */
      bool hasConverged(T percentage = T(0.01));
      
      /*
       * returns the best NN solution found so far and
       * its average error in testing dataset and the number
       * of converged solutions so far.
       */
      bool getSolution(whiteice::nnetwork<T>& nn,
		       T& error, unsigned int& Nconverged) const;

      // don't copy nnetwork which might be large (optimization)
      bool getSolutionStatistics(T& error, unsigned int& Nconverged) const;

      bool getSolution(whiteice::nnetwork<T>& nn) const;
      
      /* used to stop the optimization process */
      bool stopComputation();

      // removes solution and resets to empty NNGradDescent<>
      void reset();
	
      private:
      
      T getError(const whiteice::nnetwork<T>& net,
		 const whiteice::dataset<T>& dtest,
		 const bool regularize = true,
		 const bool dropout = false);
	
      
      whiteice::nnetwork<T>* nn; // network architecture and settings
      
      bool heuristics;
      bool dropout; // use dropout heuristics when training	
      bool dont_normalize_error; // don't normalize values when calculating error.
      T regularizer;

      // whether to use minimum norm error ||y-f(x)|| instead of
      // the standard MSE (minimum squared error) 0.5*||y-f(x)||^2 (default)
      bool mne; 
      
      vertex<T> bestx;
      T best_error;
      T best_pure_error;
      unsigned int iterations;
      
      const whiteice::dataset<T>* data;
      
      // flag to indicate this is the first thread to start optimization
      bool first_time;
      std::mutex first_time_lock;
      
      bool deep_pretraining;
      
      unsigned int NTHREADS;
      unsigned int MAXITERS;
      std::vector<std::thread*> optimizer_thread;      
      
      std::map<std::thread::id, std::list<T> > errors; // last errors
      const unsigned int EHISTORY = 20;

      std::map<std::thread::id, bool> convergence;

      // counter per thread to test if there have been no improvements
      std::map<std::thread::id, unsigned int> noimprovements;
      const unsigned int MAX_NOIMPROVEMENT_ITERS = 5;

      whiteice::RNG<T> rng; // we use random numbers
      bool use_minibatch; // use minibatch to estimate gradient
      bool overfit; // use all data to fit to solution (disabled as default)
	
      mutable std::mutex solution_lock, start_lock, errors_lock;
      mutable std::mutex convergence_lock, noimprove_lock;
      
      bool running;
      
      volatile int thread_is_running;
      std::mutex thread_is_running_mutex;
      std::condition_variable thread_is_running_cond;
      bool optimize_started_finished;
      
      void optimizer_loop();
      
      };
    
  };
};


namespace whiteice
{
  namespace math
  {
    extern template class NNGradDescent< blas_real<float> >;
    extern template class NNGradDescent< blas_real<double> >;
    
    extern template class NNGradDescent< blas_complex<float> >;
    extern template class NNGradDescent< blas_complex<double> >;

    extern template class NNGradDescent< superresolution<blas_complex<float>,
							 modular<unsigned int> > >;
    extern template class NNGradDescent< superresolution<blas_complex<double>,
							 modular<unsigned int> > >;
    
    
  };
};



#endif
