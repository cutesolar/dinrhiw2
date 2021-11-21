
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
					    const  std::vector< std::vector< rifl_datapoint<T> > >& episodes_,
					    //const std::multimap< T, std::vector< rifl_datapoint<T> > >& episodes_,
					    std::mutex & database_mutex_,
					    std::mutex & model_mutex_,
					    const unsigned int & epoch_,
					    whiteice::dataset<T>& data_) :
    rifl(rifl_), 
    episodes(episodes_),
    database_mutex(database_mutex_),
    model_mutex(model_mutex_),
    epoch(epoch_),
    data(data_)
  {
    worker_thread = nullptr;
    running = false;
    completed = false;
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
  
  // starts thread that creates NUM_EPISODES episode samples to dataset
  template <typename T>
  bool CreateRIFL3dataset<T>::start(const unsigned int NUM_EPISODES)
  {
    if(NUM_EPISODES == 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(running == true || worker_thread != nullptr)
      return false;

    try{
      this->NUM_EPISODES = NUM_EPISODES;
      
      data.clear();
      data.createCluster("input-state", rifl.numStates);
      data.createCluster("output-action", rifl.numActions);
      data.createCluster("episode-ranges", 2);
      
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

    // needed??
    whiteice::bayesian_nnetwork<T> model, lagged_Q;
    whiteice::dataset<T> preprocess;

    const T delta = T(0.50f); // amount% of new Q value to be added as new Q value

    // double DQN (lagged_Q network is used to select next action)
    {
      std::lock_guard<std::mutex> lock(model_mutex);

      model = rifl.model;
      lagged_Q = rifl.lagged_Q;
      preprocess = rifl.preprocess;
    }

    
    {

      unsigned int counter = 0;

      for(unsigned int e=0;e<NUM_EPISODES;e++){

	if(running == false)
	  continue; // exits loop

	std::vector< rifl_datapoint<T> > episode;

	database_mutex.lock();

	// random sample from database
	const unsigned int index = rng.rand() % episodes.size();
	episode = episodes[index];

#if 0
	// take samples from 50% lowest performing episodes (lowest score)
	// these are the cases we cannot properly handle

	
	if(episodes.size() >= 500)
	  index = rng.rand() % (episodes.size()/2);
	else
	  index = rng.rand() % episodes.size(); // too little data so sample

	auto iter = episodes.begin(); // from smallest to largest
	std::advance(iter, index);
	episode = iter->second;


	if(rng.rand()&1){
	  auto iter = episodes.rbegin(); // take episodes with largest rewards
	  std::advance(iter, index); // for(unsigned int i=0;i<index;i++) iter++;
	  episode = iter->second;
	}
	else{
	  auto iter = episodes.begin(); // take episodes with smallest rewards
	  std::advance(iter, index); // for(unsigned int i=0;i<index;i++) iter++;
	  episode = iter->second;
	}
#endif

	// const auto episode = episodes[index];

	database_mutex.unlock();
	
	// adds episode start and end in dataset (to be added data)
	{
	  const unsigned int START = data.size(0);
	  const unsigned int LENGTH = episode.size();
	  
	  whiteice::math::vertex<T> range;
	  range.resize(2);
	  range[0] = START;
	  range[1] = START+LENGTH;
	  
	  assert(data.add(2, range) == true);
	}

	whiteice::math::vertex<T> recurrent_data;
	assert(recurrent_data.resize(rifl.RECURRENT_DIMENSIONS) == rifl.RECURRENT_DIMENSIONS);
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
	  
	  assert(preprocess.preprocess(0, state) == true);

	  assert(instate.write_subvertex(state, 0) == true);
	  assert(instate.write_subvertex(recurrent_data, rifl.numStates) == true);
	  
	  assert(model.calculate(instate, out, 1, 0) == true);

	  assert(out.subvertex(outaction, 0, rifl.numActions) == true);
	  assert(out.subvertex(recurrent_data, rifl.numActions, recurrent_data.size()) == true);
	  
	  assert(preprocess.invpreprocess(1, outaction) == true);
	  
	  // calculates updated utility value
	  
	  T unew_value = T(0.0);
	  T maxvalue = T(-INFINITY);
	  
	  {
	    whiteice::math::vertex<T> input(rifl.numStates);
	    whiteice::math::vertex<T> full_input(rifl.numStates + rifl.RECURRENT_DIMENSIONS);
	    whiteice::math::vertex<T> output(rifl.numActions);

	    //full_input.zero(); // recurrent dimensions are ZERO for lagged_Q (fresh start)
	    
	    assert(input.write_subvertex(datum.newstate, 0) == true);
	    
	    assert(preprocess.preprocess(0, input) == true);

	    assert(full_input.write_subvertex(input, 0) == true);
	    assert(full_input.write_subvertex(recurrent_data, input.size()) == true);

#if 0
	    assert(model.calculate(full_input, u, 1, 0) == true);

	    assert(u.subvertex(output, 0, rifl.numActions) == true);
	    
	    assert(preprocess.invpreprocess(1, output) == true);

	    unsigned int next_action = 0;

	    for(unsigned int i=0;i<output.size();i++){
	      if(maxvalue < output[i]){
		maxvalue = output[i];
		next_action = i;
	      }
	    }
	    
	    
	    assert(lagged_Q.calculate(full_input, u, 1, 0) == true);

	    assert(u.subvertex(output, 0, rifl.numActions) == true);
	    
	    assert(preprocess.invpreprocess(1, output) == true);
	    
	    maxvalue = output[next_action];
#else

	    assert(lagged_Q.calculate(full_input, u, 1, 0) == true);

	    assert(u.subvertex(output, 0, rifl.numActions) == true);
	    
	    assert(preprocess.invpreprocess(1, output) == true);

	    auto tmp_logits = output; // TODO: optimize to be outside of loop, not recreated everytime

	    lagged_Q.getNetwork().softmax_output(tmp_logits, 0, tmp_logits.size());
	    
	    const unsigned int next_action = rifl.prob_action_select(tmp_logits);
	    maxvalue = output[next_action];

#if 0
	    for(unsigned int i=0;i<output.size();i++){
	      if(maxvalue < output[i]){
		maxvalue = output[i];
		//next_action = i;
	      }
	    }
#endif
	    
#endif
	    
	    if(epoch <= rifl.WARMUP_ITERS || datum.lastStep == true){
	      // first iteration always uses pure reinforcement values
	      unew_value = datum.reinforcement;
	    }
	    else{ 
	      unew_value = datum.reinforcement + rifl.gamma*maxvalue;
	    }
	  }
	  
	  outaction[action] = (T(1.0f) - delta)*outaction[action] + delta*unew_value;
	  
	  model.getNetwork().softmax_output(outaction, 0, outaction.size());
	  
//#pragma omp critical
	  {
	    assert(data.add(0, in) == true);
	    assert(data.add(1, outaction) == true);

	    counter++;
	    
	    maxvalues.push_back(maxvalue);
	  }
	  
	} // for i in episodes (OpenMP loop)
	
	
      } // for e in NUM_EPISODES
      
    }

    if(running == false)
      return; // exit point

#if 1
    // add preprocessing to dataset
    {
      // TODO: ENABLE INPUT DATASET PREPROCESSING LATER BUT USE WHOLE DATASETS:
      
      // batch dataset used for training is so small that we don't normalize input(??)
      //assert(data.preprocess
      //     (0, whiteice::dataset<T>::dnMeanVarianceNormalization) == true);
      
      
      // assert(data.preprocess
      //(1, whiteice::dataset<T>::dnMeanVarianceNormalization) == true);
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
