/*
 * Linear K-Cluster machine learning 
 *
 * Code:
 * 0. Assign data points (x,y) randomly to K-Clusters
 * 1. Train/optimize linear model for points (x_i,y_i) assigned to this cluster k.
 * 2. Measure error in each cluster model for each datapoint and 
 *    assign datapoints to smallest error cluster.
 * 3. Predict code y=f_k(x) finds closest training datapoint x_i and its cluster k which 
 *    model k is used to predict value of y.
 * 4. Goto 1 if there were signficant changes/no convergence in assigning of 
 *    the datapoints x_i to K clusters. 
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

    void setEarlyStopping(bool enabled = true){
      early_stopping = enabled;
    }

    bool getEarlyStopping(){ return early_stopping; }

    bool save(const std::string& filename) const;
    bool load(const std::string& filename); 
    
  protected:

    double calculateError(const std::vector< math::vertex<T> >& x,
			  const std::vector< math::vertex<T> >& y,
			  const whiteice::nnetwork<T>& model) const;

    
    const bool verbose = false; // whether to print debugging messages.
    bool early_stopping = true;

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
