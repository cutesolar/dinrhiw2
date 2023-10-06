/*
 * Global Optimizer for E{y|x} prediction model
 * 
 * - level of detail parameter sets how many datapoints 
 *   there must be at least for each frequent variable combination.  
 * 
 */

#ifndef __whiteice__GlobalOptimizer_h
#define __whiteice__GlobalOptimizer_h

#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <set>

#include "vertex.h"
#include "matrix.h"
#include "superresolution.h"

#include "dynamic_bitset.h"

#include "discretize.h"


namespace whiteice
{
  template <typename T=whiteice::math::blas_real<float> >
  class GlobalOptimizer
  {
  public:
    GlobalOptimizer();
    virtual ~GlobalOptimizer();

    bool startTrain(const std::vector< math::vertex<T> >& xdata,
		    const std::vector< math::vertex<T> >& ydata,
		    T levelOfDetailFreq = T(0.0));
		    
    bool isRunning() const;

    bool stopTrain();
    
    bool getSolutionError(double& error) const;
    
    bool predict(const math::vertex<T>& x, math::vertex<T>& y) const;

    bool save(const std::string& filename) const;
    bool load(const std::string& filename); 
    
  protected:

    mutable std::mutex start_mutex;

    // data
    
    std::vector< math::vertex<T> > xdata;
    std::vector< math::vertex<T> > ydata;

    // model

    std::vector<struct whiteice::discretization> disc;
    std::set<whiteice::dynamic_bitset> f_itemset;

    // linear pseudo global optimizer model
    math::matrix<T> A;
    math::vertex<T> b;

    T levelOfDetailFreq;
    T currentError;
    
    // running
    std::thread* optimizer_thread = nullptr;
    mutable std::mutex thread_mutex, solution_mutex;
    bool thread_running = false;
    
    void optimizer_loop();
  };



  extern template class GlobalOptimizer< math::blas_real<float> >;
  extern template class GlobalOptimizer< math::blas_real<double> >;
  //extern template class GlobalOptimizer< math::blas_complex<float> >;
  //extern template class GlobalOptimizer< math::blas_complex<double> >;

  //extern template class GlobalOptimizer< math::superresolution< math::blas_real<float> > >;
  //extern template class GlobalOptimizer< math::superresolution< math::blas_real<double> > >;
  //extern template class GlobalOptimizer< math::superresolution< math::blas_complex<float> > >;
  //extern template class GlobalOptimizer< math::superresolution< math::blas_complex<double> > >;

  
}


#endif
