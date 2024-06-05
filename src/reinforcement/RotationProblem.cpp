

#include "RotationProblem.h"
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
  RotationProblem<T>::RotationProblem() : RIFL_abstract2<T>(3, 3)
  {
    {
      reset();
      resetLastStep = false;
    }
    
    
    // starts physics thread
    {
      std::lock_guard<std::mutex> lock(rotation_mutex);
      
      running = true;
      rotation_thread = new std::thread(std::bind(&RotationProblem<T>::rotationLoop, this));
    }
    
  }


  template <typename T>
  RotationProblem<T>::~RotationProblem()
  {
    // stops physics thread
    {
      std::lock_guard<std::mutex> lock(rotation_mutex);

      running = false;
      
      if(rotation_thread){
	rotation_thread->join();
	delete rotation_thread;
      }
      
      rotation_thread = nullptr;
    }
  }

  
  template <typename T>
  void RotationProblem<T>::reset()
  {
    whiteice::math::vertex<T> a;
    whiteice::RNG<T> random;
    
    a.resize(3);
    random.normal(a);
    a /= a.norm();
    
    this->x = a[0];
    this->y = a[1];
    this->z = a[2];
    
    this->rotation_x = 0.0;
    this->rotation_y = 0.0;
    this->rotation_z = 0.0;
    
    iteration = 0;
    rotation_processed = false;
    
    // reset in this time step
    resetLastStep = true;
  }
  
  
  template <typename T>
  bool RotationProblem<T>::getState(whiteice::math::vertex<T>& state)
  {
    // whiteice::logging.info("RotationProblem: entering getState()");
    
    state.resize(this->numStates);
    
    state[0] = x;
    state[1] = y;
    state[2] = z;
    
    // whiteice::logging.info("RotationProblem: exiting getState()");
    
    return true;
  }

  
  
  template <typename T>
  bool RotationProblem<T>::performAction(const whiteice::math::vertex<T>& action,
					 whiteice::math::vertex<T>& newstate,
					 T& reinforcement, bool& endFlag)
  {
    // whiteice::logging.info("RotationProblem: entering performAction()");
    
    
    assert(action.size() == 3);
    
    {
      std::unique_lock<std::mutex> lock(rotation_change);
      
      this->rotation_x = action[0];
      this->rotation_y = action[1];
      this->rotation_z = action[2];
      
      rotation_processed = false;
      rotation_processed_cond.notify_all();
      
      while(rotation_processed == false){
	auto now = std::chrono::system_clock::now();
	rotation_processed_cond.wait_until(lock, now + 100ms);
	if(running == false)
	  return false;
      }
      
      {
	newstate.resize(this->numStates);
	newstate[0] = x;
	newstate[1] = y;
	newstate[2] = z;

	reinforcement = x; // we maximize x=1.0, y=0.0, z=0.0 is the maximum

	// std::cout << iteration << "/100: REINFORCEMENT: " << reinforcement << std::endl;
	
	if(resetLastStep){
	  endFlag = true;
	  resetLastStep = false;
	}
	else{
	  endFlag = false;
	}
      }
    }
    
    return true;
  }

  
  
  template <typename T>
  void RotationProblem<T>::rotationLoop()
  {    
    resetLastStep = false;
    
    
    while(running){
      
      {
	std::unique_lock<std::mutex> lock(rotation_change);
	
	while(rotation_processed == true){
	  auto now = std::chrono::system_clock::now();
	  rotation_processed_cond.wait_until(lock, now + 10*100ms);
	  if(running == false) return;
	}

	whiteice::math::vertex<T> a;
	whiteice::math::matrix<T> R;

	a.resize(4);
	R.resize(4,4);

	a[0] = x;
	a[1] = y;
	a[2] = z;
	a[3] = 1;

	R.rotation(rotation_x, rotation_y, rotation_z);

	a = R*a;

	x = a[0];
	y = a[1];
	z = a[2];

	iteration++;

	if(iteration > 100)
	  reset();

	rotation_processed = true;
	rotation_processed_cond.notify_all();
      }

      // 1x faster than time
      // usleep((unsigned int)(dt.c[0]*1000000.0));  // waits for a single timestep
      
    }
    
  }
  
  
  template class RotationProblem< math::blas_real<float> >;
  template class RotationProblem< math::blas_real<double> >;
};
