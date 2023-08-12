
#ifndef __whiteice_HMC_diffeq_h
#define __whiteice_HMC_diffeq_h

#include "diffeqs.h"

#include "dataset.h"
#include "nnetwork.h"
#include "RungeKutta.h"
#include "HMC.h"

#include <vector>


using namespace whiteice;


template <typename T = math::blas_real<float> >
  class HMC_diffeq : public whiteice::HMC<T>
  {
  public:
  
  HMC_diffeq(const whiteice::nnetwork<T>& net,
	     const whiteice::dataset<T>& ds,
	     const math::vertex<T>& start_point,
	     const std::vector<T>& times, // time-steps for datapoints we use
	     bool adaptive=false, T alpha = T(0.5), bool store = true, bool restart_sampling = true)
    : HMC<T>(net, ds, adaptive, store, restart_sampling)
  {
    this->times = times;
    this->start_point = start_point;
  }
  
  
  // probability functions for hamiltonian MC sampling
  T U(const math::vertex<T>& q, bool useRegulizer = true) const;
  
  math::vertex<T> Ugrad(const math::vertex<T>& q) const;

  // calculates mean error for the latest N samples, 0 = all samples
  T getMeanError(unsigned int latestN = 0) const;  
  
  protected:
  
  math::vertex<T> start_point;
  std::vector<T> times; // time-steps for datapoints we use
  
  };


extern template class HMC_diffeq< math::blas_real<float> >;
extern template class HMC_diffeq< math::blas_real<double> >;


#endif
