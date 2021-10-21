

#include "AdditionProblem.h"
#include "Log.h"

#include <stdio.h>
#include <math.h>

#include <chrono>
#include <functional>

#include <unistd.h>


using namespace std::chrono_literals;


namespace whiteice
{

  template <typename T>
  AdditionProblem<T>::AdditionProblem() : RIFL_abstract2<T>(3, 3)
  {
    {
      reset();
      resetLastStep = false;
    }
    
    running = true;
  }


  template <typename T>
  AdditionProblem<T>::~AdditionProblem()
  {
    running = false;
  }

  
  template <typename T>
  void AdditionProblem<T>::reset()
  {
    whiteice::RNG<T> random;

    state.resize(3);

    random.normal(state);
    state /= state.norm();

    iteration = 0;
    
    // reset in this time step
    resetLastStep = true;
  }
  
  
  template <typename T>
  bool AdditionProblem<T>::getState(whiteice::math::vertex<T>& state_)
  {
    // whiteice::logging.info("AdditionProblem: entering getState()");

    state_ = this->state;
    
    // whiteice::logging.info("AdditionProblem: exiting getState()");
    
    return true;
  }

  
  
  template <typename T>
  bool AdditionProblem<T>::performAction(const whiteice::math::vertex<T>& action,
					 whiteice::math::vertex<T>& newstate,
					 T& reinforcement, bool& endFlag)
  {
    // whiteice::logging.info("AdditionProblem: entering performAction()");
    
    
    assert(action.size() == 3);
    
    {
      iteration++;

      if(iteration > 10){
	iteration = 0;
	reset();
	newstate = state;
	reinforcement = T(5.0)-newstate.norm();
	// reinforcement = T(5.0)-newstate[0];
	// reinforcement = state.norm()-newstate.norm(); [don't work]
	if(reinforcement < T(0.0)) reinforcement = T(0.0);
	std::cout << "ITER " << iteration << " REINFORCEMENT = " << reinforcement << std::endl;
	endFlag = true;

	state = newstate;
	
	return true;
      }
      else{
	newstate = state + action;
	reinforcement = T(5.0)-newstate.norm();
	// reinforcement = T(5.0)-newstate[0];
	// reinforcement = state.norm()-newstate.norm(); [don't work]
	if(reinforcement < T(0.0)) reinforcement = T(0.0);
	std::cout << "ITER " << iteration << " REINFORCEMENT = " << reinforcement << std::endl;
	endFlag = false;

	state = newstate;

	return true;
      }
    }
    
    return true;
  }
 
   
  
  template class AdditionProblem< math::blas_real<float> >;
  template class AdditionProblem< math::blas_real<double> >;
};
