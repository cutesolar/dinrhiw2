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

namespace whiteice
{
  // caller must implement interface from which data vectors can be retrieved (possible from the disk)
  class DataSourceInterface {
  public:

    virtual const unsigned long getNumber() = 0; // number of data vector pairs (x,y)
    virtual const unsigned long getInputDimension() = 0; // x input vector dimension
    virtual const unsigned long getOutputDimension() = 0; // y output vector dimension

    // gets index:th data points or return false (bad index or unknown error)
    virtual const bool getData(unsigned long index,
			       math::vertex< math::blas_real<float> > & x,
			       math::vertex< math::blas_real<float> >& y) = 0;
  };
  
  
  class HugeLinear {
  public:

    HugeLinear();
    ~HugeLinear();

    bool startOptimize(DataSourceInterface* data);    
    bool isRunning();
    bool stopOptimize();
    
    bool getSolution(math::matrix< math::blas_real<float> >& A,
		     math::vertex< math::blas_real<float> >& b);
    
    float estimateSolutionMSE();

    // removes solution and resets to empty HugeLinear
    void reset();
    
  private:

    void optimizer_loop();

    // threading data
    std::thread* optimizer_thread;
    std::mutex solution_lock, start_lock;

    bool running; // as long as not converged and runnign == true, keeps iterating results
    bool converged;  // true if converged and has stopped computing results

    // model data
    math::matrix< math::blas_real<float> > A;
    math::vertex< math::blas_real<float> > b;
    
  };
  
  
}

#endif
