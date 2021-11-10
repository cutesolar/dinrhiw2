/*
 * K-means boosting algorithm for neural networks.
 *
 * Algorithm:
 * 1. Assign training samples {(x_i,y_i)} randomly to M neural networks.
 * 2. Train neural networks given training samples
 * 3. Re-assign training points to neural networks that give the smallest error
 * 4. Continue as long as points are re-assigned to different neural networks (K-Means)
 * 5. Train combiner neural network which which assigns weight to each neural network
 *    given x, w(x). Optimize w(x) until convergence.
 * 6. Predict output as y = SUM w_i(x)*nn(x).
 *
 * FIXME: KMBoosting idea does NOT work, rewrite this code to use Gradient Boosting instead.
 *
 * 
 */

#ifndef __whiteice__KMBoosting_h
#define __whiteice__KMBoosting_h

#include <vector>
#include <thread>
#include <mutex>

#include "vertex.h"
#include "dinrhiw_blas.h"
#include "nnetwork.h"
#include "dataset.h"


namespace whiteice
{

  template <typename T = math::blas_real<float> >
  class KMBoosting
  {
  public:

    /*
     * Initialize K-Means neural network boosting algorithm to use 
     * M neural networks and with given architecture arch.
     */
    KMBoosting(const unsigned int M, const std::vector<unsigned int> arch);

    ~KMBoosting();

    bool startOptimize(const whiteice::dataset<T>& data);
    bool stopOptimize();

    bool isRunning() const;

    bool hasModel() const { return hasModelFlag; }

    /*
     * predict output given input
     */
    bool calculate(const math::vertex<T>& input, math::vertex<T>& output) const;

  private:

    std::vector< whiteice::nnetwork<T> > experts; // M experts
    whiteice::nnetwork<T> weighting; // w(x) weight for each expert

    bool hasModelFlag;

    const bool debug = true;

    whiteice::dataset<T> data;
    
    std::thread* optimizer_thread;
    bool thread_running;

    mutable std::mutex solution_mutex;
    mutable std::mutex thread_mutex;
    

    void optimizer_loop();
  };
  


  extern template class KMBoosting<whiteice::math::blas_real<float> >;
  extern template class KMBoosting<whiteice::math::blas_real<double> >;

  //extern template class KMBoosting<whiteice::math::blas_complex<float> >;
  //extern template class KMBoosting<whiteice::math::blas_complex<double> >;
  
};

#endif

