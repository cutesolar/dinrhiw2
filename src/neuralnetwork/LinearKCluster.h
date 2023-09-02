/*
 * Linear K-Cluster machine learning 
 *
 * Code:
 * 0. Assign data points (x,y) randomly to K-Clusters
 * 1. Train/optimize linear model for points assigned to this cluster
 * 2. Measure error in each cluster model for each datapoint and 
 *    assign datapoints to smallest error cluster.
 * 3. Calculate mean and variance of clusters and reassign datapoints to most probable
 *    cluster based on mean and variance of each cluster
 * 4. Goto 1 if there were signficant changes/no convergence 
 *
 */

#ifndef __whiteice__LinearKCluster_h
#define __whiteice__LinearKCluster_h

#include <vector>
#include <string>
#include <thread>
#include <mutex>

#include "vertex.h"
#include "matrix.h"
#include "nnetwork.h"
#include "superresolution.h"

namespace whiteice
{
  template <typename T=whiteice::math::blas_real<float> >
  class LinearKCluster
  {
  public:
    LinearKCluster(const unsigned int XSIZE, const unsigned int YSIZE);
    virtual ~LinearKCluster();

    bool startTrain(const std::vector< math::vertex<T> >& xdata,
		    const std::vector< math::vertex<T> >& ydata,
		    const unsigned int K = 0); // K = 0 automatically tries to detect good K size

    bool isRunning() const;

    bool stopTrain();
    
    bool getSolutionError(unsigned int& iters, double& error) const;
    unsigned int getNumberOfClusters() const;
    
    bool predict(const math::vertex<T>& x, math::vertex<T>& y) const;

    bool save(const std::string& filename) const;
    bool load(const std::string& filename); 
    
  protected:

    nnetwork<T> architecture;

    // model

    unsigned int K;
    
    std::vector< whiteice::nnetwork<T> > model;

    std::vector<unsigned int> clusterLabels; // clusterLabels[datapoint_index] =  cluster_index_k
    
    double currentError;

    // data
    
    std::vector< math::vertex<T> > xdata;
    std::vector< math::vertex<T> > ydata;

    // running
    std::thread* optimizer_thread = nullptr;
    mutable std::mutex thread_mutex, solution_mutex;
    bool thread_running = false;

    unsigned int iterations = 0;

    void optimizer_loop();
    
  };



  extern template class LinearKCluster< math::blas_real<float> >;
  extern template class LinearKCluster< math::blas_real<double> >;
  extern template class LinearKCluster< math::blas_complex<float> >;
  extern template class LinearKCluster< math::blas_complex<double> >;

  extern template class LinearKCluster< math::superresolution< math::blas_real<float> > >;
  extern template class LinearKCluster< math::superresolution< math::blas_real<double> > >;
  extern template class LinearKCluster< math::superresolution< math::blas_complex<float> > >;
  extern template class LinearKCluster< math::superresolution< math::blas_complex<double> > >;

  
}


#endif
