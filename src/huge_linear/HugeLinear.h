/*
 * Implements MSE optimization of linear model y=Ax+b where
 * number of dimensions in x (and y) is extremely large (10^5 or more)
 * which means that covariance matrix cannot be computed and one needs to
 * do gradient descent instead. Also the training sample {(x,y)} can be huge
 * so that all data cannot be kept in memory.
 *
 * FIXME: Only implements float version of the interface to save memory.
 */


#ifndef __HugeLinear_h
#define __HugeLinear_h

#include <vector>
#include <thread>
#include <mutex>

#include "vertex.h"
#include "matrix.h"
#include "RNG.h"


namespace whiteice
{
  // caller must implement interface from which data vectors can be retrieved (possible from the disk)
  class DataSourceInterface {
  public:

    virtual const unsigned long getNumber() const = 0; // number of data vector pairs (x,y)
    virtual const unsigned long getInputDimension() const = 0; // x input vector dimension
    virtual const unsigned long getOutputDimension() const = 0; // y output vector dimension

    // gets index:th data points or return false (bad index or unknown error)
    virtual const bool getData(const unsigned long index,
			       math::vertex< math::blas_real<float> > & x,
			       math::vertex< math::blas_real<float> >& y) const = 0;
  };
  
  
  class HugeLinear {
  public:

    HugeLinear(bool overfit_ = false);
    ~HugeLinear();

    bool startOptimize(DataSourceInterface* data);    
    bool isRunning();
    bool stopOptimize();
    
    bool getSolution(math::matrix< math::blas_real<float> >& A,
		     math::vertex< math::blas_real<float> >& b);
    
    float estimateSolutionMSE(); // mean squared error of the solution
    unsigned int getIterations(); // number of iterations computed so far
    
  private:

    // calculates MSE error for the linear model parameters
    float getError(const DataSourceInterface* data,
		   const std::vector<unsigned long>& dset,
		   const math::matrix< math::blas_real<float> >& A,
		   const math::vertex< math::blas_real<float> >& b) const;

    
    // optimization loop
    void optimizer_loop();

    // threading data
    std::thread* optimizer_thread;
    std::mutex solution_lock, start_lock;    

    bool running; // as long as not converged and runnign == true, keeps iterating results
    bool thread_has_started;
    bool converged;  // true if converged and has stopped computing results

    DataSourceInterface* data;

    // model data
    math::matrix< math::blas_real<float> > A;
    math::vertex< math::blas_real<float> > b;

    bool overfit; // if true overfit to data

    unsigned int iterations;
    float current_solution_mse;

    // random number source
    whiteice::RNG< math::blas_real<float> > rng;
    
  };
  
  
}

#endif
