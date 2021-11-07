/*
 * Parallel Policy maximizer (gradient ascent) 
 * (maximizes Q(state, action=policy(state)))
 * 
 */


#include <thread>
#include <mutex>
#include <condition_variable>

#include "dinrhiw_blas.h"
#include "dataset.h"
#include "dinrhiw.h"
#include "nnetwork.h"

#include "RNG.h"


#ifndef __whiteice__PolicyGradAscent_h
#define __whiteice__PolicyGradAscent_h

namespace whiteice
{
  
  template <typename T=whiteice::math::blas_real<float> >
  class PolicyGradAscent
  {
  public:
    
    // if errorTerms is true then dataset output values are actual
    // errors rather than correct values
    PolicyGradAscent(bool deep_pretraining = false);
    PolicyGradAscent(const PolicyGradAscent<T>& grad);
    ~PolicyGradAscent();
    
    /*
     * starts the optimization process using data as 
     * the dataset as a training data
     *
     * Uses neural network with architecture arch.
     *
     * Executes NTHREADS in parallel when looking for
     * the optimal solution and goes max to 
     * MAXITERS iterations when looking for gradient
     * descent solution
     * 
     * dropout - whether to use dropout heuristics when training
     */
    bool startOptimize(const whiteice::dataset<T>* data,
		       const whiteice::nnetwork<T>& Q,
		       const whiteice::dataset<T>& Q_preprocess,
		       const whiteice::nnetwork<T>& policy, // optimized policy
		       unsigned int NTHREADS,
		       unsigned int MAXITERS = 10000,
		       bool dropout = false,
		       bool useInitialNN = true);
    
    /*
     * Returns true if optimizer is running
     */
    bool isRunning();

    
    // sets and gets minibatch settings for estimating gradient
    void setUseMinibatch(bool minibatch = true){
      this->use_minibatch = minibatch;
    }
    
    bool getUseMinibatch() const {
      return use_minibatch;
    }
    

    // if lrate is <= 0, disable the SGD (default)
    void setSGD(T sgd_lrate = T(0.0f)){
      if(sgd_lrate <= T(0.0f)){ use_SGD = false; sgd_lrate = T(0.0f); return; }
      use_SGD = true;
      this->sgd_lrate = sgd_lrate;
    }
    
    bool getSGD() const { return use_SGD; }
    
    
    /*
     * returns the best NN solution found so far and
     * its average error in testing dataset and the number
     * of converged solutions so far.
     */
    bool getSolution(whiteice::nnetwork<T>& policy,
		     T& value, unsigned int& iterations) const;

    bool getSolutionStatistics(T& value, unsigned int& iterations) const;

    bool getSolution(whiteice::nnetwork<T>& policy) const;

    bool getDataset(whiteice::dataset<T>& data_) const;
    
    
    /* used to stop the optimization process */
    bool stopComputation();

    // resets data structure to not started state
    void reset();
    
  private:
    // calculates mean Q-value of the policy in dtest dataset (states are inputs)
    T getValue(const whiteice::nnetwork<T>& policy,
	       const whiteice::nnetwork<T>& Q,
	       const whiteice::dataset<T>& Q_preprocess,
	       const whiteice::dataset<T>& dtest) const;


    const whiteice::nnetwork<T>* Q;
    const whiteice::dataset<T>* Q_preprocess;
    whiteice::dataset<T> data; // copy of own dataset
    
    whiteice::nnetwork<T>* policy; // network architecture and settings
    
    
    bool heuristics;
    bool dropout; // use dropout heuristics when training
    bool deep_pretraining;
    
    bool regularize; // use regularizer term..
    T regularizer;

    bool use_minibatch = false; // use minibatch to estimate gradient
    
    bool use_SGD = false; // stochastic gradient descent with fixed learning rate
    T sgd_lrate = T(0.01f);

    bool debug; // debugging messages (disabled)
    
    whiteice::math::vertex<T> bestx; // best policy weights
    T best_value;
    T best_q_value; // without regularizer term
    unsigned int iterations;
    
    
    // flag to indicate this is the first thread to start optimization
    bool first_time;
    std::mutex first_time_lock;
    
    unsigned int NTHREADS;
    unsigned int MAXITERS;
    std::vector<std::thread*> optimizer_thread;
    std::mutex start_lock;
    mutable std::mutex solution_lock;
    
    bool running;
    
    int thread_is_running;
    std::mutex thread_is_running_mutex;
    std::condition_variable thread_is_running_cond;
    
    void optimizer_loop();

    whiteice::RNG<T> rng;
    
  };
  
  
  extern template class PolicyGradAscent< whiteice::math::blas_real<float> >;
  extern template class PolicyGradAscent< whiteice::math::blas_real<double> >;
};



#endif
