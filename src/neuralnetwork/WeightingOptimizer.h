/*
 * Minimizes weighting problem of neural networks
 *
 * Min(params of w): err(p) = SUM(x,y) 0.5*||y - M(x)*w(x|p)||^2
 *
 * M(x) = [neural_network_1(x);neural_network_2(x);..;neural_network_N(x);]
 *
 */

#ifndef __whiteice__WeightingOptimizer_h
#define __whiteice__WeightingOptimizer_h

#include "nnetwork.h"
#include "dinrhiw_blas.h"

#include <thread>
#include <mutex>


namespace whiteice
{
  template <typename T = math::blas_real<float> >
    class WeightingOptimizer
    {
    public:

      WeightingOptimizer(const dataset<T>& data,
			 const std::vector< nnetwork<T> >& experts);
      ~WeightingOptimizer();
      
      
      bool startOptimize(const nnetwork<T>& weighting);

      bool stopOptimize();

      bool isRunning() const;

      bool hasModel() const { return hasModelFlag; }
      

      bool getSolution(nnetwork<T>& weighting, T& error) const;
      
    private:

      T getError(const nnetwork<T>& weighting,
		 const dataset<T>& data,
		 const bool regularize) const;
      
      
      const dataset<T>& data;
      const std::vector< nnetwork<T> >& experts;

      nnetwork<T> weighting;
      
      const T regularizer = T(0.001f);
      
      
      std::thread* optimizer_thread;
      bool thread_running;

      mutable std::mutex thread_mutex, solution_mutex;

      T min_error;
      bool hasModelFlag;
      
      void optimizer_loop();
      
    };
  

  extern template class WeightingOptimizer<whiteice::math::blas_real<float> >;
  extern template class WeightingOptimizer<whiteice::math::blas_real<double> >;

  //extern template class WeightingOptimizer<whiteice::math::blas_complex<float> >;
  //extern template class WeightingOptimizer<whiteice::math::blas_complex<double> >;
  
};


#endif
