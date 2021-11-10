
#include "RIFL_abstract.h"
#include "NNGradDescent.h"

#include <assert.h>
#include <list>
#include <functional>

namespace whiteice
{

  template <typename T>
  RIFL_abstract<T>::RIFL_abstract(const unsigned int numActions,
				  const unsigned int numStates)
  {
    {
      char buffer[128];
      snprintf(buffer, 128, "RIFL_abstract CTOR called (%d, %d)",
	       numActions, numStates);
      whiteice::logging.info(buffer);
    }
    
    // initializes parameters
    {
      gamma = T(0.95);
      epsilon = T(0.80);

      learningMode = true;
      hasModel = 0;
      
      this->numActions        = numActions;
      this->numStates         = numStates;
    }

    
    // initializes neural network architecture and weights randomly
    // neural network is deep 6-layer residual neural network
    {
      // wide neural network..
      std::vector<unsigned int> arch;
      arch.push_back(numStates);
      arch.push_back(50);
      arch.push_back(50);
      arch.push_back(50);
      arch.push_back(50);
      arch.push_back(50);
      arch.push_back(numActions);

      whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
      // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
      // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid);
      nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh);
      
      {
	std::lock_guard<std::mutex> lock(model_mutex);
	
	nn.randomize(2, T(0.50));
	model.importNetwork(nn);
	
	// creates empty preprocessing
	preprocess.createCluster("input-state", numStates);
	preprocess.createCluster("output-action-qs", numActions);
      }
    }

    thread_is_running = 0;
    rifl_thread = nullptr;
    
    whiteice::logging.info("RIFL_abstract CTOR finished");
  }

  
  template <typename T>
  RIFL_abstract<T>::RIFL_abstract(const unsigned int numActions,
				  const unsigned int numStates,
				  std::vector<unsigned int> arch)
  {
    {
      char buffer[128];
      snprintf(buffer, 128, "RIFL_abstract CTOR called (%d, %d)",
	       numActions, numStates);
      whiteice::logging.info(buffer);
    }
    
    // initializes parameters
    {
      gamma = T(0.95);
      epsilon = T(0.80);

      learningMode = true;
      hasModel = 0;
      
      this->numActions        = numActions;
      this->numStates         = numStates;

      if(arch.size() < 2)
	arch.resize(2);

      arch[0] = numStates;
      arch[arch.size()-1] = numActions;
    }

    
    // initializes neural network architecture and weights randomly
    // neural network is deep 6-layer residual neural network
    {

      whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
      // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
      // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid);
      nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh); // was: rectifier
      
      {
	std::lock_guard<std::mutex> lock(model_mutex);
	
	nn.randomize(2, T(0.50));
	model.importNetwork(nn);
	
	// creates empty preprocessing
	preprocess.createCluster("input-state", numStates);
	preprocess.createCluster("output-action-qs", numActions);
      }
    }

    thread_is_running = 0;
    rifl_thread = nullptr;
    
    whiteice::logging.info("RIFL_abstract CTOR finished");
  }
  
  
  template <typename T>
  RIFL_abstract<T>::~RIFL_abstract() 
  {
    // stops executing thread
    {
      if(thread_is_running <= 0) return;

      std::lock_guard<std::mutex> lock(thread_mutex);

      if(thread_is_running <= 0) return;

      thread_is_running--;

      if(rifl_thread){
	rifl_thread->join();
	delete rifl_thread;
      }

      rifl_thread = nullptr;
    }
    
  }

  
  // starts Reinforcement Learning thread
  template <typename T>
  bool RIFL_abstract<T>::start()
  {
    if(thread_is_running != 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_is_running != 0) return false;

    try{
      thread_is_running++;
      rifl_thread = new std::thread(std::bind(&RIFL_abstract<T>::loop, this));
    }
    catch(std::exception& e){
      thread_is_running--;
      rifl_thread = nullptr;

      return false;
    }

    return true;
  }

  
  // stops Reinforcement Learning thread
  template <typename T>
  bool RIFL_abstract<T>::stop()
  {
    if(thread_is_running <= 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_is_running <= 0) return false;

    thread_is_running--;

    if(rifl_thread){
      rifl_thread->join();
      delete rifl_thread;
    }

    rifl_thread = nullptr;
    return true;
  }

  template <typename T>
  bool RIFL_abstract<T>::isRunning() const
  {
    return (thread_is_running > 0);
  }


  // epsilon E [0,1] percentage of actions are chosen according to model
  //                 1-e percentage of actions are random (exploration)
  template <typename T>
  bool RIFL_abstract<T>::setEpsilon(T epsilon) 
  {
    if(epsilon < T(0.0) || epsilon > T(1.0)) return false;
    this->epsilon = epsilon;
    return true;
  }
  

  template <typename T>
  T RIFL_abstract<T>::getEpsilon() const 
  {
    return epsilon;
  }


  template <typename T>
  void RIFL_abstract<T>::setLearningMode(bool learn) 
  {
    learningMode = learn;
  }

  template <typename T>
  bool RIFL_abstract<T>::getLearningMode() const 
  {
    return learningMode;
  }


  template <typename T>
  void RIFL_abstract<T>::setHasModel(unsigned int hasModel) 
  {
    this->hasModel = hasModel;
  }

  template <typename T>
  unsigned int RIFL_abstract<T>::getHasModel() 
  {
    return hasModel;
  }

  
  // saves learnt Reinforcement Learning Model to file
  template <typename T>
  bool RIFL_abstract<T>::save(const std::string& filename) const
  {    

    {
      std::lock_guard<std::mutex> lock(model_mutex);
      
      char buffer[256];

      snprintf(buffer, 256, "%s-model", filename.c_str());

      if(model.save(buffer) == false) return false;

      snprintf(buffer, 256, "%s-preprocess", filename.c_str());

      if(preprocess.save(buffer) == false) return false;
    }

    return true;
  }
  
  // loads learnt Reinforcement Learning Model from file
  template <typename T>
  bool RIFL_abstract<T>::load(const std::string& filename)
  {
    {
      std::lock_guard<std::mutex> lock(model_mutex);
      
      char buffer[256];
	  
      snprintf(buffer, 256, "%s-model", filename.c_str());

      if(model.load(buffer) == false) return false;

      snprintf(buffer, 256, "%s-preprocess", filename.c_str());

      if(preprocess.load(buffer) == false) return false;
    }

    return true;
  }


  // helper function, returns minimum value in v
  template <typename T>
  unsigned int RIFL_abstract<T>::min(const std::vector<unsigned int>& vec)
    const 
  {
    if(vec.size() <= 0) return 0;
    unsigned int min = vec[0];
    for(const auto& v : vec)
      if(v < min) min = v;

    return min;
  }
  

  template <typename T>
  void RIFL_abstract<T>::loop()
  {
    std::vector< rifl_datapoint<T> > database;
    std::mutex database_mutex;

    bool endFlag = false;

    std::vector< rifl_datapoint<T> > episode;
    
    FILE* episodesFile = fopen("episodes-result.txt", "w");
    
    unsigned long episodes_counter = 0;
    
    // gradient descent learning class
    whiteice::math::NNGradDescent<T> grad;

    const T lrate = T(1e-6); // was: 1e-4
    
    grad.setOverfit(false);
    grad.setUseMinibatch(true);
    grad.setSGD(lrate);
    grad.setRegularizer(T(0.001f)); // enable regularizer against large values

    // dataset for learning class
    whiteice::dataset<T> data;

    // used to calculate dataset in background for NNGradDescent..
    whiteice::CreateRIFLdataset<T>* dataset_thread = nullptr;

    unsigned int epoch = 0;
    int old_grad_iterations = 0;

    const unsigned int DATASIZE = 1000000;
    const unsigned int SAMPLESIZE = 5000; // number of samples database must have before use
    const unsigned int BATCHSIZE = 500;
    const unsigned int ITERATIONS = 1; // was 250

    bool first_time = true;
    whiteice::math::vertex<T> state;

    
    while(thread_is_running > 0){

      // 1. gets current state
      {
	auto oldstate = state;

	if(getState(state) == false)
	  state = oldstate;
      }

      // 2. activates neural networks to get utility values for each command
      std::vector<T> U;
      
      {	
	U.resize(numActions);
	
	whiteice::math::vertex<T> u;

	{
	  std::lock_guard<std::mutex> lock(model_mutex);
	  whiteice::math::vertex<T> input;
	  
	  input = state;
	  
	  preprocess.preprocess(0, input);
	  
	  assert(model.calculate(input, u, 1, 0) == true);
	  assert(u.size() == numActions);
	  
	  preprocess.invpreprocess(1, u);

	  for(unsigned int i=0;i<numActions;i++)
	    U[i] = u[i];
	}
      }
      
      // 3. selects action according to probabilities
      unsigned int action = 0;

      {
#if 0
	const T temperature = T(0.1); // was: 1.0 - for random selection of actions.
	
	T psum = T(0.0);
	
	std::vector<T> p;

	for(unsigned int i=0;i<U.size();i++){
	  auto value = U[i];
	  if(value < T(-6.0)) value = T(-6.0);
	  else if(value > T(+6.0)) value = T(+6.0);

	  auto q = exp(value/temperature);
	  psum += q;
	  p.push_back(q);
	}

	for(unsigned int i=0;i<U.size();i++){
	  p[i] /= psum;
	}

	psum = T(0.0f);
	for(unsigned int i=0;i<p.size();i++){
	  auto more = p[i];
	  p[i] += psum;
	  psum += more;
	}

	T r = rng.uniformf();
	
	unsigned int index = 0;

	while(r > p[index]){
	  index++;
	  if(index >= numActions){
	    index = numActions-1;
	    break;
	  }
	}

	action = index;
#else
	

	{ // selects the largest value
	  action = 0;
	  T maxv = U[action];
	  
	  for(unsigned int i=1;i<U.size();i++){
	    if(maxv < U[i]){
	      action = i;
	      maxv = U[i];
	    }
	  }
	}
#endif

	// random selection with (1-epsilon) probability
	// show model with epsilon probability
	{
	  T r = rng.uniformf();
	  
	  if(r > epsilon){
	    action = rng.rand() % (numActions);
	  }
	}

	// if we don't have not yet optimized model, then we make random choices
	if(hasModel == 0)
	  action = rng.rand() % (numActions);

#if 0
	{
	  printf("U = ");
	  for(unsigned int i=0;i<U.size();i++){
	    if(action == i) printf("%e* ", U[i].c[0]);
	    else printf("%e  ", U[i].c[0]);
	  }
	  printf("\n");
	}
#endif

      }
      
      whiteice::math::vertex<T> newstate;
      T reinforcement = T(0.0);

      // 4. perform action 
      {
	if(performAction(action, newstate, reinforcement, endFlag) == false){
	  continue;
	}

	//auto delta = newstate - state;
	//std::cout << "state ||delta||^2 = " << (delta*delta)[0] << std::endl;
      }

      if(learningMode == false){
	continue; // we do not do learning
      }

      // 6. updates database
      {
	struct rifl_datapoint<T> datum;

	datum.state = state;
	datum.newstate = newstate;
	datum.reinforcement = reinforcement;
	datum.action = action;
	datum.lastStep = endFlag;

	episode.push_back(datum);

	if(datum.lastStep){
	  T total_reward = T(0.0f);
	  
	  for(const auto& e : episode)
	    total_reward += e.reinforcement;

	  total_reward /= T(episode.size());

	  char buffer[80];

	  snprintf(buffer, 80, "Episode %d avg reward: %f (%d moves) [%d models]",
		   (int)episodes_counter, total_reward.c[0], (int)episode.size(),
		   hasModel);

	  whiteice::logging.info(buffer);


	  fprintf(episodesFile, "%f\n", total_reward.c[0]);
	  fflush(episodesFile);

	  episode.clear();
	  episodes_counter++;
	}

	// for synchronizing access to database datastructure
	// (also used by CreateRIFLdataset class/thread)
	std::lock_guard<std::mutex> lock(database_mutex);
	
	if(database.size() >= DATASIZE){
	  const unsigned int index = rng.rand() % database.size();
	  database[index] = datum;
	}
	else{
	  database.push_back(datum);
	}
	
      }

      
      // activates batch learning if it is not running
      {
	
	
	if(database.size() >= SAMPLESIZE)
	{
	  whiteice::nnetwork<T> nn;
	  T error;
	  unsigned int iters = 0;

	  
	  if(dataset_thread != nullptr){
	    if(dataset_thread->isCompleted() != true || dataset_thread->isRunning()){
	      continue; // keep running only dataset_thread
	    }
	    else{
	      // dataset_thread computation completed, delete data_thread and start gradient descent
	      dataset_thread->stop();
	      delete dataset_thread;
	      dataset_thread = nullptr;

	      // fetch NN parameters from model
	      {
		// gradient is not yet running.. preparing nn for launch..
		// (no preprocess)
		std::vector< math::vertex<T> > weights;
		
		std::lock_guard<std::mutex> lock(model_mutex);
		
		if(model.exportSamples(nn, weights, 1) == false){
		  assert(0);
		}
		
		assert(weights.size() > 0);
		
		if(nn.importdata(weights[0]) == false){
		  assert(0);
		}

		//nn.setNonlinearity(whiteice::nnetwork<T>::rectifier);
		//nn.setNonlinearity(nn.getLayers()-1,whiteice::nnetwork<T>::rectifier);
	      }
	      
	      const bool dropout = false;
	      const bool useInitialNN = true;

	      grad.setOverfit(false);
	      grad.setUseMinibatch(true);
	      grad.setNormalizeError(false);
	      grad.setSGD(lrate);

	      if(hasModel >= 10){
		grad.startOptimize(data, nn, 1, ITERATIONS, dropout, useInitialNN);
	      }
	      else{
		grad.setUseMinibatch(false);
		grad.setSGD(T(-1.0f)); // disable stochastic gradient descent
		
		grad.startOptimize(data, nn, 1, 5, dropout, useInitialNN);
	      }
	    
	      old_grad_iterations = -1;
	      
	      continue;
	    }
	  }

	  // dataset_thread not running
	  if(grad.isRunning() == false){
	    if(first_time == false){
	      // gradient descent has completed, fetch results and start dataset_thread again here
	      
	      // we do not have proper dataset/model yet so we fetch params
	      grad.getSolution(nn);
	      
	      char buffer[128];
	      double tmp = 0.0;
	      whiteice::math::convert(tmp, error);
	      snprintf(buffer, 128,
		       "RIFL_abstract: new optimized Q-model (%f error, %d iters, epoch %d)",
		       tmp, iters, epoch);
	      whiteice::logging.info(buffer);
	      
	      std::lock_guard<std::mutex> lock(model_mutex);
	      
	      model.importNetwork(nn);
	      
	      nn.diagnosticsInfo();
	      
	      data.clearData(0);
	      data.clearData(1);
	      
	      preprocess = data;
	      
	      whiteice::logging.info("RIFL_abstract: new model imported");

	      grad.reset();
	      hasModel++;
	      epoch++;
	      first_time = false;
	    }
	    
	    // start dataset_thread
	    
	    data.clear();
	    data.createCluster("input-state", numStates);
	    data.createCluster("output-action-q", numActions);
	    
	    dataset_thread = new CreateRIFLdataset<T>(*this,
						      database,
						      database_mutex,
						      epoch,
						      data);
	    dataset_thread->start(BATCHSIZE);
	    
	    whiteice::logging.info("RIFL_abstract: new dataset_thread started");
	    
	    first_time = false;

	    continue;
	  }
	  else{ // grad.isRunning() == true, report progress
	    
	    if(grad.getSolutionStatistics(error, iters)){
	      if(((signed int)iters) > old_grad_iterations){
		char buffer[80];
		
		snprintf(buffer, 80,
			 "RIFL_abstract: epoch %d optimizer %d iters. error: %f hasmodel %d",
			 epoch, iters, error.c[0], hasModel);
		
		whiteice::logging.info(buffer);
		
		old_grad_iterations = (int)iters;
	      }
	    }
	    else{
	      char buffer[80];
	      snprintf(buffer, 80,
		       "RIFL_abstract: epoch %d grad.getSolution() FAILED",
		       epoch);
	      
	      whiteice::logging.error(buffer);
	    }

	    continue;
	  }
	  

#if 0
	  // if gradient thread have stopped or is not yet running..
	  if(grad.isRunning() == false){
	    
	    
	    if(grad.getSolutionStatistics(error, iters) == false){
	      // gradient is not yet running.. or reset() preparing nn for launch..
	    }
	    else{
	      // gradient have stopped running

	      if(dataset_thread == nullptr){
		// we do not have proper dataset/model yet so we fetch params
		grad.getSolution(nn);
			      
		char buffer[128];
		double tmp = 0.0;
		whiteice::math::convert(tmp, error);
		snprintf(buffer, 128,
			 "RIFL_abstract: new optimized Q-model (%f error, %d iters, epoch %d)",
			 tmp, iters, epoch);
		whiteice::logging.info(buffer);

		std::lock_guard<std::mutex> lock(model_mutex);

		model.importNetwork(nn);
		  
		nn.diagnosticsInfo();
		  
		data.clearData(0);
		data.clearData(1);
		
		preprocess = data;
		
		whiteice::logging.info("RIFL_abstract: new model imported");
		
		grad.reset();
		epoch++;
		hasModel++;
	      }
	      
	    }


	    if(dataset_thread == nullptr){
	      data.clear();
	      data.createCluster("input-state", numStates + dimActionFeatures);
	      data.createCluster("output-action-q", 1);
	      
	      dataset_thread = new CreateRIFLdataset<T>(*this,
							database,
							database_mutex,
							epoch,
							data);
	      dataset_thread->start(BATCHSIZE);

	      whiteice::logging.info("RIFL_abstract: new dataset_thread started");
	      
	      continue;
	      
	    }
	    else{
	      if(dataset_thread->isCompleted() != true)
		continue; // we havent computed proper dataset yet..
	    }

	    whiteice::logging.info("RIFL_abstract: new dataset_thread finished");
	    if(dataset_thread) dataset_thread->stop();

	    // fetch NN parameters from model
	    {
	      // gradient is not yet running.. preparing nn for launch..
	      // (no preprocess)
	      std::vector< math::vertex<T> > weights;
	      
	      std::lock_guard<std::mutex> lock(model_mutex);
	      
	      if(model.exportSamples(nn, weights, 1) == false){
		assert(0);
	      }
	      
	      assert(weights.size() > 0);
	      
	      if(nn.importdata(weights[0]) == false){
		assert(0);
	      }

	      nn.setNonlinearity(whiteice::nnetwork<T>::rectifier);
	      nn.setNonlinearity(nn.getLayers()-1,whiteice::nnetwork<T>::rectifer);
	    }
	    
	    const bool dropout = false;
	    bool useInitialNN = true;
	    const unsigned int ITERATIONS=1; // was 250
	    
	    // if(grad.startOptimize(data, nn, 2, 250, dropout) == false){
	    if(grad.startOptimize(data, nn, 1, ITERATIONS, dropout, useInitialNN) == false){
	      whiteice::logging.error("RIFL_abstract: starting grad optimizer FAILED");
	      assert(0);
	    }
	    else{
	      whiteice::logging.info("RIFL_abstract: grad optimizer started");
	    }
	    
	    old_grad_iterations = -1;
	    
	    if(dataset_thread) delete dataset_thread;
	    dataset_thread = nullptr;
	  }
	  else{

	    if(grad.getSolutionStatistics(error, iters)){
	      if(((signed int)iters) > old_grad_iterations){
		char buffer[80];
		
		snprintf(buffer, 80,
			 "RIFL_abstract: epoch %d optimizer %d iters. error: %f hasmodel %d",
			 epoch, iters, error.c[0], hasModel);
		
		whiteice::logging.info(buffer);

		old_grad_iterations = (int)iters;
	      }
	    }
	    else{
	      char buffer[80];
	      snprintf(buffer, 80,
		       "RIFL_abstract: epoch %d grad.getSolution() FAILED",
		       epoch);
	      
	      whiteice::logging.error(buffer);
	    }
	  }
#endif
	}
	
      }
      
    }

    
    grad.stopComputation();

    if(episodesFile) fclose(episodesFile);
    
    if(dataset_thread){
      dataset_thread->stop();
      delete dataset_thread;
      dataset_thread = nullptr;
    }
    
  }

  template class RIFL_abstract< math::blas_real<float> >;
  template class RIFL_abstract< math::blas_real<double> >;
  
};
