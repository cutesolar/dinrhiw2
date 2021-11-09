
#include "CreatePolicyDataset.h"

#include <pthread.h>
#include <sched.h>
#include <functional>

#ifdef WINOS
#include <windows.h>
#endif

#include "Log.h"


namespace whiteice
{
  
  // calculates reinforcement learning training dataset from database
  // uses database_lock for synchronization
  template <typename T>
  CreatePolicyDataset<T>::CreatePolicyDataset(RIFL_abstract2<T> const & rifl_, 
					      std::vector< rifl2_datapoint<T> > const & database_,
					      std::mutex & database_mutex_,
					      whiteice::dataset<T>& data_) : 

    rifl(rifl_),
    database(database_),
    database_mutex(database_mutex_),
    data(data_)
  {
    worker_thread = nullptr;
    running = false;
    completed = false;
  }
  
  
  template <typename T>
  CreatePolicyDataset<T>::~CreatePolicyDataset()
  {
    std::lock_guard<std::mutex> lk(thread_mutex);
    
    if(running || worker_thread != nullptr){
      running = false;
      if(worker_thread) worker_thread->join();
      delete worker_thread;
      worker_thread = nullptr;
    }
  }

  
  // starts thread that creates NUMDATAPOINTS samples to dataset
  template <typename T>
  bool CreatePolicyDataset<T>::start(const unsigned int NUMDATAPOINTS)
  {
    if(NUMDATAPOINTS == 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(running == true || worker_thread != nullptr)
      return false;

    try{
      NUMDATA = NUMDATAPOINTS;
      data.clear();
      data.createCluster("input-state", rifl.numStates);
      
      completed = false;
      
      running = true;
      worker_thread = new std::thread(std::bind(&CreatePolicyDataset<T>::loop, this));
      
    }
    catch(std::exception&){
      running = false;
      if(worker_thread){ delete worker_thread; worker_thread = nullptr; }
      return false;
    }

    return true;
  }
  
  // returns true when computation is completed
  template <typename T>
  bool CreatePolicyDataset<T>::isCompleted() const
  {
    return completed;
  }
  
  // returns true if computation is running
  template <typename T>
  bool CreatePolicyDataset<T>::isRunning() const
  {
    return running;
  }

  template <typename T>
  bool CreatePolicyDataset<T>::stop()
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    if(running || worker_thread != nullptr){
      running = false;
      if(worker_thread) worker_thread->join();
      delete worker_thread;
      worker_thread = nullptr;

      return true;
    }
    else return false;
  }
  
  // returns reference to dataset
  // (warning: if calculations are running then dataset can change during use)
  template <typename T>
  whiteice::dataset<T> const & CreatePolicyDataset<T>::getDataset() const
  {
    return data;
  }
  
  // worker thread loop
  template <typename T>
  void CreatePolicyDataset<T>::loop()
  {
    // set thread priority (non-standard) to low (background thread)
    {
      sched_param sch_params;
      int policy = SCHED_FIFO;
      
      pthread_getschedparam(pthread_self(),
			    &policy, &sch_params);

#ifdef linux
      policy = SCHED_IDLE; // in linux we can set idle priority
#endif
      sch_params.sched_priority = sched_get_priority_min(policy);
      
      if(pthread_setschedparam(pthread_self(),
				 policy, &sch_params) != 0){
	// printf("! SETTING LOW PRIORITY THREAD FAILED\n");
      }
      
#ifdef WINOS
      SetThreadPriority(GetCurrentThread(),
			THREAD_PRIORITY_IDLE);
#endif	
    }

    
#pragma omp parallel for schedule(guided)
    for(unsigned int i=0;i<NUMDATA;i++){

      if(running == false) // we don't do anything anymore..
	continue; // exits OpenMP loop

      database_mutex.lock();
      
      const unsigned int index = rng.rand() % database.size();

      const auto datum = database[index];

      database_mutex.unlock();
      
#pragma omp critical
      {
	data.add(0, datum.state);
      }
      
    }

    if(running == false)
      return; // exit point

#if 1
    // add preprocessing to dataset
    {
      data.preprocess
	(0, whiteice::dataset<T>::dnMeanVarianceNormalization);
    }
#endif

    completed = true;

    {
      // std::lock_guard<std::mutex> lock(thread_mutex);
      running = false;
    }
    
  }
  

  template class CreatePolicyDataset< math::blas_real<float> >;
  template class CreatePolicyDataset< math::blas_real<double> >;
};
