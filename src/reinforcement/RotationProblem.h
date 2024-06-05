/*
 * Simple rotation test problem: max(x), 

 * when a sphere point in x^2 + y^2 + z^2 = 1 
 * is rotated around origo for 100 iterations.
 * 
 * This test Q can now have negative value (unlike in testcase4).
 * Example seems to work now.
 */

#ifndef __whiteice__rotation_problem_h
#define __whiteice__rotation_problem_h

#include "RIFL_abstract2.h"

#include <condition_variable>


namespace whiteice
{

  template <typename T>
  class RotationProblem : public RIFL_abstract2<T>
  {
  public:
    RotationProblem();
    ~RotationProblem();

    bool rotationIsRunning(){ return this->running; }
      
  protected:
    
    virtual bool getState(whiteice::math::vertex<T>& state);
    
    virtual bool performAction(const whiteice::math::vertex<T>& action,
			       whiteice::math::vertex<T>& newstate,
			       T& reinforcement,
			       bool& endFlag);
    
  protected:

    // resets rotation variables to random values
    void reset();

    T x, y, z;
    
    bool resetLastStep;

    T rotation_x, rotation_y, rotation_z;
    std::mutex rotation_change;
    bool rotation_processed;
    std::condition_variable rotation_processed_cond;

    int iteration;

    volatile bool running;
    std::thread* rotation_thread;
    std::mutex rotation_mutex;

    void rotationLoop();
      
  };

  
  extern template class RotationProblem< math::blas_real<float> >;
  extern template class RotationProblem< math::blas_real<double> >;
};


#endif
