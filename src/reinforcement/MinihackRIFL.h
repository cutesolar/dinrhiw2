/*
 * Continuous reinforcement learning code
 * which uses python script to interact 
 * with enviroment. 
 *
 * It is assumed actions are discrete (8 movement commands) and
 * 8 dimensional action vector (one-hot-encoding) is sampled
 * based on weights p(a_i)=exp(w_i)/SUM(exp(w_i)) to make decision 
 * to do a single action.
 * 
 */

#ifndef __whiteice_minihack_rifl2_h
#define __whiteice_minihack_rifl2_h

#include "RIFL_abstract.h"
#include "RNG.h"

#include <string>
#include <Python.h>

namespace whiteice
{
  template <typename T>
    class MinihackRIFL : public RIFL_abstract<T>
    {
    public:

      MinihackRIFL(const std::string& pythonScript);
      ~MinihackRIFL();

      bool isRunning() const;

    protected:

      virtual bool getState(whiteice::math::vertex<T>& state);
      
      virtual bool performAction(const unsigned int action,
				 whiteice::math::vertex<T>& newstate,
				 T& reinforcement, bool& endFlag);
      
    private:

      std::string filename;
      FILE* pythonFile = NULL;

      PyObject *main_module = NULL, *global_dict = NULL;
      PyObject *getStateFunc = NULL, *performActionFunc = NULL;
      
      //PyThreadState* pystate = NULL;

      // number of errors seen.. (>0 isRunning() == false)
      unsigned int errors = 0;

      //whiteice::RNG<T> rng;
    };


  extern template class MinihackRIFL< math::blas_real<float> >;
  extern template class MinihackRIFL< math::blas_real<double> >;
}

#endif
