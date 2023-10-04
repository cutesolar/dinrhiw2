/*
 * General predictor based on datamined frequency vectors and Linear K-Cluster machine learning model.
 * 
 */

#ifndef __whiteice__GeneralKCluster_h
#define __whiteice__GeneralKCluster_h

#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <set>

#include "vertex.h"
#include "matrix.h"
#include "nnetwork.h"
#include "superresolution.h"

#include "LinearKCluster.h"
#include "dynamic_bitset.h"

#include "discretize.h"


namespace whiteice
{
  template <typename T=whiteice::math::blas_real<float> >
  class GeneralKCluster
  {
  public:
    GeneralKCluster();
    virtual ~GeneralKCluster();

    bool startTrain(const std::vector< math::vertex<T> >& xdata,
		    const std::vector< math::vertex<T> >& ydata);
		    
    bool isRunning() const;

    bool stopTrain();
    
    bool getSolutionError(unsigned int& iters, double& error) const;
    unsigned int getNumberOfClusters() const;
    
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

    LinearKCluster<T>* model = nullptr;
  };



  extern template class GeneralKCluster< math::blas_real<float> >;
  extern template class GeneralKCluster< math::blas_real<double> >;
  //extern template class GeneralKCluster< math::blas_complex<float> >;
  //extern template class GeneralKCluster< math::blas_complex<double> >;

  //extern template class GeneralKCluster< math::superresolution< math::blas_real<float> > >;
  //extern template class GeneralKCluster< math::superresolution< math::blas_real<double> > >;
  //extern template class GeneralKCluster< math::superresolution< math::blas_complex<float> > >;
  //extern template class GeneralKCluster< math::superresolution< math::blas_complex<double> > >;

  
}


#endif
