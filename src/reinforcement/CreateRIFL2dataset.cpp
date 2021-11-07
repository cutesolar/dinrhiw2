
#include "CreateRIFL2dataset.h"

#include <pthread.h>
#include <sched.h>

#include <functional>

#ifdef WINOS
#include <windows.h>
#endif

#include "Log.h"

// FIXME?? lagged_Q should be copied to process as it may change while we compute.. (not??)

namespace whiteice
{
  
  // calculates reinforcement learning training dataset from database
  // uses database_lock for synchronization
  template <typename T>
  CreateRIFL2dataset<T>::CreateRIFL2dataset(RIFL_abstract2<T> const & rifl_, 
					    std::vector< rifl2_datapoint<T> > const & database_,
					    std::vector< std::vector< rifl2_datapoint<T> > > const & episodes_,
					    std::mutex & database_mutex_,
					    unsigned int const& epoch_, 
					    whiteice::dataset<T>& data_) : 
  
    rifl(rifl_), 
    database(database_),
    episodes(episodes_),
    database_mutex(database_mutex_),
    epoch(epoch_),
    data(data_)
  {
    worker_thread = nullptr;
    running = false;
    completed = false;

    {
      std::lock_guard<std::mutex> lock(rifl.policy_mutex);
      
      policy_preprocess = rifl.policy_preprocess;
      lagged_policy = rifl.lagged_policy;
    }

    
  }
  
  
  template <typename T>
  CreateRIFL2dataset<T>::~CreateRIFL2dataset()
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
  bool CreateRIFL2dataset<T>::start(const unsigned int NUMDATAPOINTS, const bool smartEpisodes)
  {
    if(NUMDATAPOINTS == 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(running == true || worker_thread != nullptr)
      return false;

    try{
      NUMDATA = NUMDATAPOINTS;
      data.clear();
      data.createCluster("input-state", rifl.numStates + rifl.numActions);
      data.createCluster("output-action", 1);

      this->smartEpisodes = smartEpisodes;
      completed = false;
      
      running = true;
      worker_thread = new std::thread(std::bind(&CreateRIFL2dataset<T>::loop, this));
      
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
  bool CreateRIFL2dataset<T>::isCompleted() const
  {
    return completed;
  }
  
  // returns true if computation is running
  template <typename T>
  bool CreateRIFL2dataset<T>::isRunning() const
  {
    return running;
  }

  template <typename T>
  bool CreateRIFL2dataset<T>::stop()
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
  whiteice::dataset<T> const & CreateRIFL2dataset<T>::getDataset() const
  {
    return data;
  }
  
  // worker thread loop
  template <typename T>
  void CreateRIFL2dataset<T>::loop()
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

    
    // used to calculate avg max abs(Q)-value
    // (internal debugging for checking that Q-values are within sane limits)
    std::vector<T> maxvalues;

    if(smartEpisodes){

      unsigned int counter = 0;

      while(counter < NUMDATA){
	
	if(running == false) // we don't do anything anymore..
	  break; // exits loop

	database_mutex.lock();

	const unsigned int  index = rng.rand() % episodes.size();
	const auto episode = episodes[index];

	database_mutex.unlock();

	T prev_R = T(0.0f);
	
#pragma omp parallel for schedule(guided)
	for(unsigned i=0;i<episode.size();i++){
	  
	  if(running == false) // we don't do anything anymore..
	    continue; // exits OpenMP loop

	  const rifl2_datapoint<T>& datum = episode[episode.size()-1-i];

	  whiteice::math::vertex<T> in(rifl.numStates + rifl.numActions);
	  in.zero();
	  in.write_subvertex(datum.state, 0);
	  in.write_subvertex(datum.action, rifl.numStates);
	  
	  whiteice::math::vertex<T> out(1);
	  out.zero();
	  
	  // calculates updated utility value
	  whiteice::math::vertex<T> y(1);
	  
	  T maxvalue = T(-INFINITY);
	  
	  {
	    whiteice::math::vertex<T> tmp(rifl.numStates + rifl.numActions);
	    
	    assert(tmp.write_subvertex(datum.newstate, 0) == true);
	    
	    {
	      whiteice::math::vertex<T> u; // new action..
	      
	      auto input = datum.newstate;
	      
	      policy_preprocess.preprocess(0, input);
	      
	      lagged_policy.calculate(input, u, 1, 0);
	      
	      policy_preprocess.invpreprocess(1, u); // does nothing..

#if 0
	      // add exploration noise..
	      auto noise = u;
	      // Normal EX[n]=0 StDev[n]=1 [OPTMIZE ME: don't create new RNG everytime but use global one]
	      rng.normal(noise);
	      u += T(0.05)*noise;
#endif
	      
	      assert(tmp.write_subvertex(u, rifl.numStates) == true); // writes policy's action
	    }
	    
	    rifl.Q_preprocess.preprocess(0, tmp);
	    
	    rifl.lagged_Q.calculate(tmp, y, 1, 0);
	    
	    rifl.Q_preprocess.invpreprocess(1, y);
	    
	    if(maxvalue < abs(y[0]))
	      maxvalue = abs(y[0]);
	    
	    if(epoch > 0 && datum.lastStep == false){
	      const T R = rifl.gamma * prev_R + datum.reinforcement;
	      
	      out[0] = R + rifl.gamma*y[0];

	      prev_R = R;
	    }
	    else{ // the first iteration of reinforcement learning do not use Q or if this is last step
	      out[0] = datum.reinforcement;

	      prev_R = out[0];
	    }
	    
	  }
	  
#pragma omp critical (grojreojoGTTVVWAssdfs)
	  {
	    data.add(0, in);
	    data.add(1, out);

	    counter++;
	    
	    maxvalues.push_back(maxvalue);
	  }
	  
	} // for-loop

	
      } // while loop (counter)
      
    }
    else{

#pragma omp parallel for schedule(guided)
      for(unsigned int i=0;i<NUMDATA;i++){
	
	if(running == false) // we don't do anything anymore..
	  continue; // exits OpenMP loop
	
	database_mutex.lock();
	
	const unsigned int index = rng.rand() % database.size();
	
	const auto datum = database[index];
	
	database_mutex.unlock();
	
	whiteice::math::vertex<T> in(rifl.numStates + rifl.numActions);
	in.zero();
	in.write_subvertex(datum.state, 0);
	in.write_subvertex(datum.action, rifl.numStates);
	
	whiteice::math::vertex<T> out(1);
	out.zero();
	
	// calculates updated utility value
	whiteice::math::vertex<T> y(1);
	
	T maxvalue = T(-INFINITY);
	
	{
	  whiteice::math::vertex<T> tmp(rifl.numStates + rifl.numActions);
	  
	  assert(tmp.write_subvertex(datum.newstate, 0) == true);
	  
	  {
	    whiteice::math::vertex<T> u; // new action..
	    
	    auto input = datum.newstate;
	    
	    policy_preprocess.preprocess(0, input);
	    
	    lagged_policy.calculate(input, u, 1, 0);
	    
	    policy_preprocess.invpreprocess(1, u); // does nothing..
	    
	    // add exploration noise..
#if 0
	    auto noise = u;
	    // Normal EX[n]=0 StDev[n]=1 [OPTMIZE ME: don't create new RNG everytime but use global one]
	    rng.normal(noise);
	    u += T(0.05)*noise;
#endif
	    
	    assert(tmp.write_subvertex(u, rifl.numStates) == true); // writes policy's action
	  }
	  
	  rifl.Q_preprocess.preprocess(0, tmp);
	  
	  rifl.lagged_Q.calculate(tmp, y, 1, 0);
	  
	  rifl.Q_preprocess.invpreprocess(1, y);
	  
	  if(maxvalue < abs(y[0]))
	    maxvalue = abs(y[0]);
	  
	  if(epoch >= 10 && datum.lastStep == false){
	    out[0] = datum.reinforcement + rifl.gamma*y[0];
	  }
	  else{ // the first iteration of reinforcement learning do not use Q or if this is last step
	    out[0] = datum.reinforcement;
	  }
	  
	}
	
#pragma omp critical (HGSJIEOeeewcwaWW)
	{
	  data.add(0, in);
	  data.add(1, out);
	  
	  maxvalues.push_back(maxvalue);
	}
	
      }
      
    }

    if(running == false)
      return; // exit point

    // add preprocessing to dataset
#if 1
    {
      data.preprocess
	(0, whiteice::dataset<T>::dnMeanVarianceNormalization);
    
      data.preprocess
	(1, whiteice::dataset<T>::dnMeanVarianceNormalization);
    }
#endif

    
    // for debugging purposes (reports average max Q-value)
    if(maxvalues.size() > 0)
    {
      T sum = T(0.0);
      for(auto& m : maxvalues)
	sum += abs(m);

      sum /= T(maxvalues.size());

      double tmp = 0.0;
      whiteice::math::convert(tmp, sum);

      char buffer[80];
      snprintf(buffer, 80, "CreateRIFL2dataset: avg abs(Q)-value %f",
	       tmp);

      whiteice::logging.info(buffer);
    }

    completed = true;

    {
      // std::lock_guard<std::mutex> lock(thread_mutex);
      running = false;
    }
    
  }
  

  template class CreateRIFL2dataset< math::blas_real<float> >;
  template class CreateRIFL2dataset< math::blas_real<double> >;
};
