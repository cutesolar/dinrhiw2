
#include "CreateRIFL3dataset.h"

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
  // using database_lock
  template <typename T>
  CreateRIFL3dataset<T>::CreateRIFL3dataset(const RIFL_abstract3<T> & rifl_,
					  const std::vector< rifl_datapoint<T> > & database_,
					  const  std::vector< std::vector< rifl_datapoint<T> > >& episodes_,
					  std::mutex & database_mutex_,
					  const unsigned int & epoch_,
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

    useEpisodes = false;
  }

  
  template <typename T>
  CreateRIFL3dataset<T>::~CreateRIFL3dataset()
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
  bool CreateRIFL3dataset<T>::start(const unsigned int NUMDATAPOINTS, const bool useEpisodes)
  {
    if(NUMDATAPOINTS == 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(running == true || worker_thread != nullptr)
      return false;

    try{
      NUMDATA = NUMDATAPOINTS;

      this->useEpisodes = useEpisodes;
      
      data.clear();
      data.createCluster("input-state", rifl.numStates);
      data.createCluster("output-action", rifl.numActions);

      if(useEpisodes){
	data.createCluster("episode-ranges", 2);
      }
      
      completed = false;
      
      running = true;
      worker_thread = new std::thread(std::bind(&CreateRIFL3dataset<T>::loop, this));
      
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
  bool CreateRIFL3dataset<T>::isCompleted() const
  {
    return completed;
  }
  
  // returns true if computation is running
  template <typename T>
  bool CreateRIFL3dataset<T>::isRunning() const
  {
    return running;
  }

  template <typename T>
  bool CreateRIFL3dataset<T>::stop()
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
  whiteice::dataset<T> const & CreateRIFL3dataset<T>::getDataset() const
  {
    return data;
  }
  
  // worker thread loop
  template <typename T>
  void CreateRIFL3dataset<T>::loop()
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

    if(useEpisodes){

      unsigned int counter = 0;

      while(counter < NUMDATA){

	if(running == false)
	  break; // exits loop

	database_mutex.lock();
	
	const unsigned int  index = rng.rand() % episodes.size();
	const auto episode = episodes[index];

	database_mutex.unlock();
	
	// adds episode start and end in dataset (to be added data)
	{
	  const unsigned int START = data.size(0);
	  const unsigned int LENGTH = episode.size();
	  
	  whiteice::math::vertex<T> range;
	  range.resize(2);
	  range[0] = START;
	  range[1] = START+LENGTH;
	  data.add(2, range);
	}

	whiteice::math::vertex<T> recurrent_data;
	recurrent_data.resize(rifl.RECURRENT_DIMENSIONS);
	recurrent_data.zero();

//#pragma omp parallel for schedule(guided)
	for(unsigned i=0;i<episode.size();i++){

	  if(running == false) // we don't do anything anymore..
	    continue; // exits OpenMP loop..
	  
	  const unsigned int action = episode[i].action;
	  const auto& datum = episode[i];

	
	  whiteice::math::vertex<T> in;
	  in = datum.state;
	  
	  whiteice::math::vertex<T> out(rifl.numActions + rifl.RECURRENT_DIMENSIONS);
	  whiteice::math::vertex<T> outaction(rifl.numActions);
	  whiteice::math::vertex<T> u;
	  out.zero();

	  whiteice::math::vertex<T> state = datum.state;
	  whiteice::math::vertex<T> instate;
	  instate.resize(rifl.numStates + rifl.RECURRENT_DIMENSIONS);
	  
	  rifl.preprocess.preprocess(0, state);

	  instate.write_subvertex(state, 0);
	  instate.write_subvertex(recurrent_data, rifl.numStates);
	  
	  assert(rifl.model.calculate(instate, out, 1, 0) == true);

	  out.subvertex(outaction, 0, rifl.numActions);
	  out.subvertex(recurrent_data, rifl.numActions, recurrent_data.size());
	  
	  rifl.preprocess.invpreprocess(1, outaction);
	  
	  // calculates updated utility value
	  
	  T unew_value = T(0.0);
	  T maxvalue = T(-INFINITY);
	  
	  {
	    whiteice::math::vertex<T> input(rifl.numStates);
	    whiteice::math::vertex<T> full_input(rifl.numStates + rifl.RECURRENT_DIMENSIONS);
	    whiteice::math::vertex<T> output(rifl.numActions);
	    
	    input.write_subvertex(datum.newstate, 0);
	    
	    rifl.preprocess.preprocess(0, input);

	    full_input.write_subvertex(input, 0);
	    full_input.write_subvertex(recurrent_data, input.size());
	    
	    assert(rifl.model.calculate(full_input, u, 1, 0) == true);

	    u.subvertex(output, 0, rifl.numActions);
	    
	    rifl.preprocess.invpreprocess(1, output);
	    
	    for(unsigned int i=0;i<output.size();i++)
	      if(maxvalue < output[i])
		maxvalue = output[i];
	    
	    if(epoch <= 10 || datum.lastStep == true){
	      // first iteration always uses pure reinforcement values
	      unew_value = datum.reinforcement;
	    }
	    else{ 
	      unew_value = datum.reinforcement + rifl.gamma*maxvalue;
	    }
	  }
	  
	  outaction[action] = unew_value;
	  
//#pragma omp critical
	  {
	    data.add(0, in);
	    data.add(1, outaction);

	    counter++;
	    
	    maxvalues.push_back(maxvalue);
	  }
	  
	} // for i in episodes (OpenMP loop)
	
	
      } // while loop
      
    }
    else{
    
#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<NUMDATA;i++){
	
	if(running == false) // we don't do anything anymore..
	  continue; // exits OpenMP loop..
	
	database_mutex.lock();
	
	const unsigned int index = rng.rand() % database.size();
	
	const unsigned int action = database[index].action;
	const auto datum = database[index];
	
	database_mutex.unlock();
	
	whiteice::math::vertex<T> in;
	
	in.resize(rifl.numStates);
	in.zero();
	assert(in.write_subvertex(datum.state, 0) == true);
	
	whiteice::math::vertex<T> out(rifl.numActions);
	whiteice::math::vertex<T> u;
	out.zero();
	
	auto instate = datum.state;
	
	rifl.preprocess.preprocess(0, instate);
	assert(rifl.model.calculate(instate, out, 1, 0) == true);
	rifl.preprocess.invpreprocess(1, out);
	
	// calculates updated utility value
	
	T unew_value = T(0.0);
	T maxvalue = T(-INFINITY);
	
	{
	  whiteice::math::vertex<T> input(rifl.numStates);
	  
	  input.zero();
	  input.write_subvertex(datum.newstate, 0);
	  
	  rifl.preprocess.preprocess(0, input);
	  assert(rifl.model.calculate(input, u, 1, 0) == true);
	  rifl.preprocess.invpreprocess(1, u);
	  
	  for(unsigned int i=0;i<u.size();i++)
	    if(maxvalue < u[i])
	      maxvalue = u[i];
	  
	  if(epoch <= 10 || datum.lastStep == true){
	    // first iteration always uses pure reinforcement values
	    unew_value = datum.reinforcement;
	  }
	  else{ 
	    unew_value = datum.reinforcement + rifl.gamma*maxvalue;
	  }
	}
	
	out[action] = unew_value;
	
#pragma omp critical
	{
	  data.add(0, in);
	  data.add(1, out);
	  
	  maxvalues.push_back(maxvalue);
	}
	
      } // for-loop for datapoints (OpenMP)

      
    } // if(useEpisodes) {} else{..}

    if(running == false)
      return; // exit point

#if 1
    // add preprocessing to dataset
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
	sum += m;

      sum /= T(maxvalues.size());

      double tmp = 0.0;
      whiteice::math::convert(tmp, sum);

      char buffer[80];
      snprintf(buffer, 80, "CreateRIFL3dataset: avg max(Q)-value %f",
	       tmp);

      whiteice::logging.info(buffer);
    }

    completed = true;

    {
      //std::lock_guard<std::mutex> lock(thread_mutex);
      running = false;
    }
  }
  

  template class CreateRIFL3dataset< math::blas_real<float> >;
  template class CreateRIFL3dataset< math::blas_real<double> >;
};
