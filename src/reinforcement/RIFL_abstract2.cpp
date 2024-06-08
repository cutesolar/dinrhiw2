// Reinforcement learning using continuous state and continuous actions


#include "RIFL_abstract2.h"

#include "NNGradDescent.h"
#include "PolicyGradAscent.h"

#include "Log.h"
#include "linear_ETA.h"
#include "blade_math.h"

#include <assert.h>
#include <functional>
#include <list>


namespace whiteice
{

  template <typename T>
  RIFL_abstract2<T>::RIFL_abstract2(unsigned int numActions,
				    unsigned int numStates)
  {
    // initializes parameters
    {
      // zero = learn pure Q(state,action) = x function which action=policy(state) is optimized
      gamma = T(0.80); // how much weight future values Q() have: was 0.95 WAS: 0.80
      epsilon = T(0.80);

      learningMode = true;
      sleepMode = true;

      hasModel.resize(2);
      hasModel[0] = 0; // Q-network
      hasModel[1] = 0; // policy-network

      latestError = 0.0f;

      assert(numActions > 0);
      assert(numStates > 0);
      
      this->numActions = numActions;
      this->numStates  = numStates;
    }

    
    // initializes neural network architecture and weights randomly
    // neural network is deep 6-layer residual neural network (NOW: 3 layers only)
    {
      std::vector<unsigned int> arch;

      // const unsigned int RELWIDTH = 20; // of the network (20..100)
      
      {
	std::lock_guard<std::mutex> lock(Q_mutex);

	// NOW: 10-layer small width neural network
	arch.push_back(numStates + numActions);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	arch.push_back(1);
	
	{
	  whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid); // tanh, sigmoid, halfLinear
	  //nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh);
	  
	  nn.randomize(2, T(0.8)); // was 1.0
	  nn.setResidual(true);
	  
	  Q.importNetwork(nn);
	  lagged_Q.importNetwork(nn);

	  whiteice::logging.info("RIFL_abstract2: ctor Q diagnostics");
	  Q.diagnosticsInfo();

	  Q_preprocess.createCluster("input-state", numStates + numActions);
	  Q_preprocess.createCluster("output-state", 1); // q-value
	}
      }
      
      
      {
	std::lock_guard<std::mutex> lock(policy_mutex);

	// NOW: 10-layer small width neural network
	arch.clear();
	arch.push_back(numStates);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);	
	arch.push_back(numActions);

	// policy outputs action is (should be) +[-1,+1]^D vector
	{
	  whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::tanh);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::tanh);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid);

	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh);
	  //nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
	  
	  nn.randomize(2, T(0.8)); // was 1.0
	  nn.setResidual(true);
	  
	  policy.importNetwork(nn);
	  lagged_policy.importNetwork(nn);

	  whiteice::logging.info("RIFL_abstract2: ctor policy diagnostics");
	  policy.diagnosticsInfo();

	  policy_preprocess.createCluster("input-state", numStates);
	  policy_preprocess.createCluster("output-state", numActions);
	}
      }
      
    }
    
    
    thread_is_running = 0;
    rifl_thread = nullptr;
  }

  
  template <typename T>
  RIFL_abstract2<T>::RIFL_abstract2(unsigned int numActions,
				    unsigned int numStates,
				    std::vector<unsigned int> Q_arch,
				    std::vector<unsigned int> policy_arch)
  {
    // initializes parameters
    {
      // zero = learn pure Q(state,action) = x function which action=policy(state) is optimized
      gamma = T(0.80); // how much weight future values Q() have: was 0.95 WAS: 0.80
      epsilon = T(0.80);

      learningMode = true;
      sleepMode = true;

      hasModel.resize(2);
      hasModel[0] = 0; // Q-network
      hasModel[1] = 0; // policy-network

      latestError = 0.0f;

      assert(numActions > 0);
      assert(numStates > 0);
      
      this->numActions = numActions;
      this->numStates  = numStates;

      if(Q_arch.size() < 2){
	Q_arch.resize(2);
      }

      Q_arch[0] = numStates + numActions;
      Q_arch[Q_arch.size()-1] = 1;

      if(policy_arch.size() < 2){
	policy_arch.resize(2);
      }

      policy_arch[0] = numStates;
      policy_arch[policy_arch.size()-1] = numActions;
      
    }

    
    // initializes neural network architecture and weights randomly
    // neural network is deep 6-layer residual neural network (NOW: 3 layers only)
    {
      std::vector<unsigned int> arch;

      // const unsigned int RELWIDTH = 20; // of the network (20..100)
      
      {
	std::lock_guard<std::mutex> lock(Q_mutex);

	arch = Q_arch;

	{
	  whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid); // tanh, sigmoid, halfLinear
	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
	  
	  nn.randomize(2, T(1.0)); // was 1.0
	  nn.setResidual(true);
	  
	  Q.importNetwork(nn);

	  whiteice::math::vertex<T> weights;
	  nn.exportdata(weights);
	  weights.zero();
	  nn.importdata(weights);
	  
	  lagged_Q.importNetwork(nn);

	  whiteice::logging.info("RIFL_abstract2: ctor Q diagnostics");
	  Q.diagnosticsInfo();

	  Q_preprocess.createCluster("input-state", numStates + numActions);
	  Q_preprocess.createCluster("output-state", 1); // q-value
	  
	}
      }
      
      
      {
	std::lock_guard<std::mutex> lock(policy_mutex);

	arch.clear();

	arch = policy_arch;

	// policy outputs action is (should be) +[-1,+1]^D vector
	{
	  whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::tanh);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::tanh);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid);
	  // nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh);
	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
	  
	  nn.randomize(2, T(1.0)); // was 1.0
	  nn.setResidual(true);
	  
	  policy.importNetwork(nn);

	  whiteice::math::vertex<T> weights;
	  nn.exportdata(weights);
	  weights.zero();
	  nn.importdata(weights);
	  
	  lagged_policy.importNetwork(nn);

	  whiteice::logging.info("RIFL_abstract2: ctor policy diagnostics");
	  policy.diagnosticsInfo();

	  policy_preprocess.createCluster("input-state", numStates);
	  policy_preprocess.createCluster("output-state", numActions);
	}
      }
      
    }
    
    
    thread_is_running = 0;
    rifl_thread = nullptr;
  }
  

  template <typename T>
  RIFL_abstract2<T>::~RIFL_abstract2() 
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
  bool RIFL_abstract2<T>::start()
  {
    if(thread_is_running != 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_is_running != 0) return false;

    try{
      whiteice::logging.info("RIFL_abstract2: starting main thread");
      
      thread_is_running++;
      rifl_thread = new std::thread(std::bind(&RIFL_abstract2<T>::loop, this));
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
  bool RIFL_abstract2<T>::stop()
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
  bool RIFL_abstract2<T>::isRunning() const
  {
    return (thread_is_running > 0);
  }


  // epsilon E [0,1] percentage of actions are chosen according to model
  //                 1-e percentage of actions are random (exploration)
  template <typename T>
  bool RIFL_abstract2<T>::setEpsilon(T epsilon) 
  {
    if(epsilon < T(0.0) || epsilon > T(1.0)) return false;
    this->epsilon = epsilon;
    return true;
  }
  

  template <typename T>
  T RIFL_abstract2<T>::getEpsilon() const 
  {
    return epsilon;
  }


  template <typename T>
  void RIFL_abstract2<T>::setLearningMode(bool learn) 
  {
    learningMode = learn;
  }

  template <typename T>
  bool RIFL_abstract2<T>::getLearningMode() const 
  {
    return learningMode;
  }

  template <typename T>
  void RIFL_abstract2<T>::setSleepingMode(bool sleep) 
  {
    sleepMode = sleep;
  }

  template <typename T>
  bool RIFL_abstract2<T>::getSleepingMode() const 
  {
    return sleepMode;
  }


  template <typename T>
  void RIFL_abstract2<T>::setHasModel(unsigned int hasModel) 
  {
    this->hasModel[0] = hasModel;
    this->hasModel[1] = hasModel;
  }

  template <typename T>
  unsigned int RIFL_abstract2<T>::getHasModel() 
  {
    if(hasModel[0] < hasModel[1]) return hasModel[0];
    else return hasModel[1];
  }


  template <typename T>
  float RIFL_abstract2<T>::getLatestEpisodeError() const
  {
    return latestError;
  }

  template <typename T>
  unsigned int RIFL_abstract2<T>::getDatabaseSize() const
  {
    std::lock_guard<std::mutex> lock(database_mutex);
    
    return database.size();
  }

  
  // saves learnt Reinforcement Learning Model to file
  template <typename T>
  bool RIFL_abstract2<T>::save(const std::string& filename) const
  {
    std::lock_guard<std::mutex> lock1(Q_mutex);
    std::lock_guard<std::mutex> lock2(policy_mutex);

    char buffer[256];
    
    snprintf(buffer, 256, "%s-q", filename.c_str());    
    if(Q.save(buffer) == false) return false;

    snprintf(buffer, 256, "%s-policy", filename.c_str());
    if(policy.save(buffer) == false) return false;

    snprintf(buffer, 256, "%s-lagged-q", filename.c_str());    
    if(lagged_Q.save(buffer) == false) return false;

    snprintf(buffer, 256, "%s-lagged-policy", filename.c_str());
    if(lagged_policy.save(buffer) == false) return false;

    snprintf(buffer, 256, "%s-q-preprocess", filename.c_str());    
    if(Q_preprocess.save(buffer) == false) return false;

    snprintf(buffer, 256, "%s-policy-preprocess", filename.c_str());
    if(policy_preprocess.save(buffer) == false) return false;

    return true;
  }

  
  // loads learnt Reinforcement Learning Model from file
  template <typename T>
  bool RIFL_abstract2<T>::load(const std::string& filename)
  {
    std::lock_guard<std::mutex> lock1(Q_mutex);
    std::lock_guard<std::mutex> lock2(policy_mutex);

    char buffer[256];
        
    snprintf(buffer, 256, "%s-q", filename.c_str());    
    if(Q.load(buffer) == false) return false;

    snprintf(buffer, 256, "%s-policy", filename.c_str());
    if(policy.load(buffer) == false) return false;

    snprintf(buffer, 256, "%s-lagged-q", filename.c_str());    
    if(lagged_Q.load(buffer) == false) return false;

    snprintf(buffer, 256, "%s-lagged-policy", filename.c_str());
    if(lagged_policy.load(buffer) == false) return false;

    snprintf(buffer, 256, "%s-q-preprocess", filename.c_str());    
    if(Q_preprocess.load(buffer) == false) return false;

    snprintf(buffer, 256, "%s-policy-preprocess", filename.c_str());
    if(policy_preprocess.load(buffer) == false) return false;
    
    return true;
  }


  template <typename T>
  void RIFL_abstract2<T>::onehot_prob_select(const whiteice::math::vertex<T>& action,
					     whiteice::math::vertex<T>& new_action,
					     const T temperature)
  {
    assert(action.size() > 0);
    
    unsigned long ACTION = 0;

    T psum = T(0.0f);
    std::vector<T> p;
    
    for(unsigned int i=0;i<action.size();i++){
      auto value = action[i];
      
      if(value < T(-6.0f)) value = T(-6.0f);
      else if(value > T(+6.0f)) value = T(+6.0f);
      
      auto q = exp(value/temperature);
      psum += q;
      p.push_back(q);
    }
    
    for(unsigned int i=0;i<p.size();i++)
      p[i] /= psum;
    
    psum = T(0.0f);
    for(unsigned int i=0;i<p.size();i++){
      auto more = p[i];
      p[i] += psum;
      psum += more;
    }
    
    T r = rng.uniform();
    
    unsigned long index = 0;
    
    while(r > p[index]){
      index++;
      if(index >= p.size()){
	index = p.size()-1;
	break;
      }
    }
    
    ACTION = index;

    // std::cout << "action = " << action << " => SELECT ACTION: " << ACTION << std::endl;

    new_action.resize(action.size());
    new_action.zero();

#if 0
    for(unsigned int i=0;i<new_action.size();i++){
      new_action[ACTION] = T(-1.0f);
    }
#endif
    
    new_action[ACTION] = T(1.0f);
  }

  
  template <typename T>
  void RIFL_abstract2<T>::loop()
  {
    // number of iteratios to use per epoch for optimization
    const unsigned int Q_OPTIMIZE_ITERATIONS = 5000; // 40, was 1 (dont work), 5, 10, WAS: 500
    const unsigned int P_OPTIMIZE_ITERATIONS = 1000; // 10, was 1 (dont work), 5, 10, WAS: 500
    
    // tau = 1.0 => no lagged neural networks [don't work]
    const T tau = T(1.0); // lagged Q and policy network [keeps tau%=1% of the new weights [was: 0.001, 0.05]
    
    std::vector< std::vector< rifl2_datapoint<T> > > episodes;
    std::vector< rifl2_datapoint<T> > episode;

    FILE* episodesFile = fopen("episodes-result.txt", "w");    

    bool endFlag = false; // did the simulation end during this time step?
    
    whiteice::dataset<T> data;
    whiteice::CreateRIFL2dataset<T>* dataset_thread = nullptr;
    whiteice::math::NNGradDescent<T> grad; // Q(state,action) model optimizer
    
    // deep pretraining using stacked RBMs
    // (requires sigmoidal nnetwork and training
    //  policy nnetwork (calculating gradients) dont work with sigmoid)
    const bool deep = false;
    whiteice::dataset<T> data2;
    whiteice::CreatePolicyDataset<T>* dataset2_thread = nullptr;
    whiteice::PolicyGradAscent<T> grad2(deep);   // policy(state)=action model optimizer

    whiteice::linear_ETA<double> eta, eta2; // estimates how long single epoch of optimization takes
    
    std::vector<unsigned int> epoch;

    epoch.resize(2);
    epoch[0] = 0;
    epoch[1] = 0;

    int old_grad_iterations = -1;
    int old_grad2_iterations = -1;

    const unsigned long DATASIZE = 100000; // was: 100.000 / 1M history of samples
    // assumes each episode length is 100 so this is ~ equal to 1.000.000 samples
    const unsigned long EPISODES_MAX_SIZE = 10000;
    const unsigned long MINIMUM_EPISODE_SIZE = 50;
    const unsigned long MINIMUM_DATASIZE = 10000; // samples required to start learning, was:10000
    const unsigned long SAMPLESIZE = 5000; // number of samples used in learning, was: 5000
    unsigned long database_counter = 0;
    unsigned long episodes_counter = 0;

    latestError = 0.0f;
    
    bool firstTime = true;
    whiteice::math::vertex<T> state;

    whiteice::nnetwork<T> nn;

    unsigned long counter = 0; // N:th iteration
    
    whiteice::logging.info("RIFL_abstract2: starting optimization loop");

    whiteice::logging.info("RIFL_abstract2: initial Q diagnostics");
    Q.diagnosticsInfo();

    
    while(thread_is_running > 0){

      if(sleepMode == true){
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	continue; // we do not do anything and only sleep
      }

      counter++;

      // 1. gets current state
      {
	auto oldstate = state;
      
	if(getState(state) == false){
	  state = oldstate;
	  if(firstTime) continue;

	  whiteice::logging.error("ERROR: RIFL_abstact2::getState() FAILED.");
	}

	firstTime = false;
      }

      // 2. selects action using policy
      // (+ random selection if there is no model or in
      //    1-epsilon probability)
      whiteice::math::vertex<T> action(numActions);
      bool random = false;
      
      {
	std::lock_guard<std::mutex> lock(policy_mutex);

	whiteice::math::vertex<T> u;

	auto input = state;
	policy_preprocess.preprocess(0, input);

	if(lagged_policy.calculate(input, u, 1, 0) == true){
	  if(u.size() != numActions){
	    u.resize(numActions);
	    for(unsigned int i=0;i<numActions;i++){
	      u[i] = T(0.0);
	    }
	  }
	  else{
	    policy_preprocess.invpreprocess(1, u);
	  }
	}
	else{
	  u.resize(numActions);
	  for(unsigned int i=0;i<numActions;i++){
	    u[i] = T(0.0);
	  }
	}

	// it is assumed that action data should have zero mean and is roughly
	// normally distributed (with StDev[n] = 1) so data is close to zero

	// FIXME add better random normally distributed noise (exploration)
	{
	  if(rng.uniform() > epsilon){ // 1-epsilon % are chosen randomly
	    
	    // rng.normal(u); // Normal E[n]=0 StDev[n]=1

	    rng.uniform(u); // [0,1]

	    for(unsigned int i=0;i<u.size();i++)
	      u[i] = T(2.0f)*u[i] - T(1.0f); // [-1,+1]

	    random = true;
	  }
	  else{ // just adds random noise to action [mini-exploration]
	    auto noise = u;
	    rng.normal(noise); // Normal EX[n]=0 StDev[n]=1
	    u += T(0.05)*noise;
	  }
	  
	}

	// if there's no model then make random selection (normally distributed)
#if 1
	if(hasModel[0] == 0 || hasModel[1] == 0){
	  rng.normal(u);
	  random = true;
	}
#endif
	
	action = u;
      }

      
      
      if(oneHotEncodedAction){
	whiteice::math::vertex<T> new_action;

	const T temperature = T(0.10f);

	// maps probabilistic vector values to a single value
	onehot_prob_select(action, new_action, temperature);
	
	action = new_action;
      }

      //std::cout << "action = " << action << " ";
      //std::cout << "random = " << random << std::endl;

      // prints Q value of chosen action
      if(0){
	whiteice::math::vertex<T> u;
	whiteice::math::vertex<T> in(numStates + numActions);
	in.zero();

	in.write_subvertex(state, 0);
	in.write_subvertex(action, numStates);
	
	Q_preprocess.preprocess(0, in);
	
	Q.calculate(in, u, 1, 0);
	
	Q_preprocess.invpreprocess(1, u); // does nothing..

	if(action.size() == state.size()){
	  // ONLY WORKS FOR AdditionProblem! (size(action) == size(state))
	  
	  auto norm1 = state.norm();
	  auto result = action + state;
	  auto norm2 = result.norm();

#if 0
	  if(norm2 < norm1){
	    std::cout << counter << " "
		      << "Q(STATE,POLICY_ACTION) = " << u
		      << ", STATE = " << state
		      << ", ACTION = " << action
		      << "\t NORM DECREASES: " << norm1 << ">" << norm2
		      << " RANDOM: "
		      << random
		      << " MODELS: " << hasModel[0] << " " << hasModel[1]
		      << std::endl;
	  }
	  else{
	    std::cout << counter << " "
		      << "Q(STATE,POLICY_ACTION) = " << u
		      << ", STATE = " << state
		      << ", ACTION = " << action
		      << "\t NORM INCREASES. " << norm1 << "<" << norm2
		      << " RANDOM: "
		      << random
		      << " MODELS: " << hasModel[0] << " " << hasModel[1]
		      << std::endl;
	  }
#endif
	  
	}
	else{
#if 0
	  std::cout << counter << " " 
		    << "Q(STATE,POLICY_ACTION) = " << u
		    << ", STATE = " << state
		    << ", ACTION = " << action
		    << ", RANDOM: "
		    << random
		    << " MODELS: " << hasModel[0] << " " << hasModel[1]
		    << std::endl;
#endif
	}
      }
      
      whiteice::math::vertex<T> newstate;
      T reinforcement = T(0.0);

      // 3. perform action and get newstate and reinforcement
      {
	
	if(performAction(action, newstate, reinforcement, endFlag) == false){
	  //std::cout << "ERROR: RIFL_abstract2::performAction() FAILED." << std::endl;
	  whiteice::logging.error("ERROR: RIFL_abstact::performAction() FAILED.");
	  goto optimization_step;
	}
	
      }

      
      // 4. updates database (of actions and responses)
      {
	struct rifl2_datapoint<T> datum;

	datum.state = state;
	datum.action = action;
	datum.newstate = newstate;
	datum.reinforcement = reinforcement;
	datum.lastStep = endFlag;

	// for synchronizing access to database datastructure
	// (also used by CreateRIFL2dataset class/thread)
	std::lock_guard<std::mutex> lock(database_mutex);

	episode.push_back(datum);

	if(datum.lastStep){

	  T total_reward = T(0.0f);

	  for(const auto& e : episode)
	    total_reward += e.reinforcement;

	  total_reward /= T(episode.size());

	  char buffer[80];

	  snprintf(buffer, 80, "Episode %d avg reward: %f (%d moves) [%d %d models]",
		   (int)episodes_counter, total_reward.c[0], (int)episode.size(),
		   hasModel[0], hasModel[1]);

	  whiteice::logging.info(buffer);


	  fprintf(episodesFile, "%f\n", total_reward.c[0]);
	  fflush(episodesFile);

	  latestError = (float)total_reward.c[0];

	  if(useEpisodes){
	    
	    if(episodes.size() >= EPISODES_MAX_SIZE){
	      const unsigned long index = (episodes_counter % EPISODES_MAX_SIZE);
	      episodes[index] = episode;
	    }
	    else{
	      episodes.push_back(episode);
	    }
	    
	  }

	  episode.clear();
	  episodes_counter++;
	}

	if(database_counter >= DATASIZE)
	  database_counter = database_counter % database.size();

	if(datum.reinforcement.c[0]){

	  if(database.size() >= DATASIZE){
	    const unsigned int index = rng.rand() % database.size();

	    database[index] = datum;
	  }
	  else{
	    database.push_back(datum);
	  }
	  
	}

	database_counter++;
      }

    optimization_step:
      
      if(learningMode == false){
	continue; // we do not do learning
      }
      
      // 5. update/optimize Q(state, action) network
      // activates batch learning if it is not running
      if(database.size() >= MINIMUM_DATASIZE &&
	 (episodes.size() > MINIMUM_EPISODE_SIZE || useEpisodes == false))
      {
	
	// skip if other optimization step (policy network)
	// is behind us
	if(epoch[0] > epoch[1])
	  goto q_optimization_done;
	
	
	T error;
	unsigned int iters;
	
	
	if(grad.isRunning() == false){

	  if(grad.getSolutionStatistics(error, iters) == false){
	  }
	  else{
	    // gradient have stopped running

	    if(dataset_thread == nullptr){

	      grad.getSolution(nn);
	      
	      char buffer[128];
	      double tmp = 0.0;
	      whiteice::math::convert(tmp, error);
	      snprintf(buffer, 128,
		       "RIFL_abstract2: new optimized Q-model (%f error, %d iters, epoch %d)",
		       tmp, iters, epoch[0]);
	      whiteice::logging.info(buffer);
	      
	      {
		std::lock_guard<std::mutex> lock(Q_mutex);
		Q.importNetwork(nn);

		data.clearData(0);
		data.clearData(1);

		Q_preprocess = data;

#if 1
		whiteice::nnetwork<T> nn2;
		std::vector< math::vertex<T> > lagged_weights;
		lagged_Q.exportSamples(nn2, lagged_weights, 1);

		if(lagged_weights.size() > 0){

		  math::vertex<T> weights;
		  nn.exportdata(weights);

		  logging.info("lagged_Q updated");
		  
		  lagged_weights[0] = tau*weights + (T(1.0)-tau)*lagged_weights[0];
		  nn2.importdata(lagged_weights[0]);
		  lagged_Q.importNetwork(nn2);
		}
		else{
		  logging.info("lagged_Q updated: NO LAG");
		  
		  lagged_Q.importNetwork(nn); 
		}
#endif
		
		whiteice::logging.info("RIFL_abstract2: new Q diagnostics");
		lagged_Q.diagnosticsInfo();
		
		whiteice::logging.info("RIFL_abstract2: new Q-model imported");
	      }

	      grad.reset(); // resets gradient to empty gradient descent

	      epoch[0]++;
	      hasModel[0]++;
	    }
	  }


	  // skip if other optimization step (policy network)
	  // is behind us
	  if(epoch[0] > epoch[1])
	    goto q_optimization_done;

	  
	  // const unsigned int NUMSAMPLES = database.size(); // was 1000
	  // const unsigned int NUMSAMPLES = 2000; // was 1000, 128
	  
	  
	  if(dataset_thread == nullptr){

	    {
	      std::lock_guard<std::mutex> lock(database_mutex);
	      
	      data.clear();
	      //data.createCluster("input-state", numStates + numActions);
	      //data.createCluster("output-qvalue", 1);
	      
	      
	      dataset_thread = new CreateRIFL2dataset<T>(*this,
							 database,
							 episodes,
							 database_mutex,
							 epoch[0],
							 data);
	    }
	    
	    dataset_thread->start(SAMPLESIZE, useEpisodes);
	      
	    whiteice::logging.info("RIFL_abstract2: new dataset_thread started (Q)");
	    
	    continue;
      
	  }
	  else{
	    if(dataset_thread->isCompleted() != true){
	      continue; // we havent computed proper dataset yet..
	    }
	  }
	  
	  if(dataset_thread){
	    whiteice::logging.info("RIFL_abstract2: dataset_thread finished (Q)");
	    dataset_thread->stop();
	    delete dataset_thread;
	    dataset_thread = nullptr;
	  }


	  // fetch NN parameters from model
	  whiteice::nnetwork<T> qnn;
	  
	  {
	    std::vector< math::vertex<T> > weights;
	    
	    std::lock_guard<std::mutex> lock(Q_mutex);
	    
	    if(lagged_Q.exportSamples(qnn, weights, 1) == false){ // was: lagged_Q
	      assert(0);
	    }

	    if(weights.size() <= 0)
	      assert(0);

	    if(qnn.importdata(weights[0]) == false){
	      assert(0);
	    }
	  }
	  
	  const bool dropout = false;
	  const bool useInitialNN = true; // WAS: start from scratch everytime
	  
	  grad.setRegularizer(T(0.0f)); // DISABLE REGULARIZER FOR Q-NETWORK (was: 0.001f)
	  grad.setNormalizeError(false); // calculate real error values	  
	  
	  
	  if(hasModel[0] >= 0){
	    eta.start(0.0, Q_OPTIMIZE_ITERATIONS);

	    grad.setUseMinibatch(false);
	    grad.setSGD(T(-1.0f)); // disable stochastic gradient descent
	    
	    grad.startOptimize(data, qnn, 1, Q_OPTIMIZE_ITERATIONS, dropout, useInitialNN);
	  }
	  else{
	    eta.start(0.0, 15);

	    grad.setUseMinibatch(false);
	    grad.setSGD(T(-1.0f)); // disable stochastic gradient descent
	    
	    grad.startOptimize(data, qnn, 1, 15, dropout, useInitialNN);
	  }
	  

	  old_grad_iterations = -1;
	}
	else{
	  T error = T(0.0);
	  unsigned int iters = 0;

	  if(grad.getSolutionStatistics(error, iters)){
	    if(((signed int)iters) > old_grad_iterations){
	      char buffer[128];

	      eta.update(iters);
	      
	      double e;
	      whiteice::math::convert(e, error);
	      
	      snprintf(buffer, 128,
		       "RIFL_abstract2: Q-optimizer epoch %d iter %d error %f hasmodel %d [ETA %.2f mins]",
		       epoch[0], iters, e, hasModel[0], eta.estimate()/60.0);
	      
	      whiteice::logging.info(buffer);

	      old_grad_iterations = (int)iters;
	    }
	  }
	  else{
	    char buffer[80];
	    snprintf(buffer, 80,
		     "RIFL_abstract2: epoch %d grad.getSolution() FAILED",
		     epoch[0]);
	    
	    whiteice::logging.error(buffer);
	  }
	}
      }
      
    q_optimization_done:
      
      
      // 6. update/optimize policy(state) network
      // activates batch learning if it is not running
      
      if(database.size() >= MINIMUM_DATASIZE &&
	 (episodes.size() > MINIMUM_EPISODE_SIZE || useEpisodes == false))
      {
	
	// skip if other optimization step is behind us
	// we only start calculating policy after Q() has been optimized..
	//if(epoch[1] > epoch[0] || epoch[0] == 0)
	//  goto policy_optimization_done;
	if(epoch[0] == 0 || epoch[0] <= epoch[1])
	  goto policy_optimization_done;

	
	whiteice::nnetwork<T> nn;
	T meanq;
	unsigned int iters;

	
	if(grad2.isRunning() == false){


	  if(grad2.getSolutionStatistics(meanq, iters) == false){
	  }
	  else{
	    // gradient has stopped running
	    
	    if(dataset2_thread == nullptr){

	      grad2.getSolution(nn);
	      grad2.getDataset(this->policy_preprocess);
	      
	      char buffer[128];
	      double tmp = 0.0;
	      whiteice::math::convert(tmp, meanq);
	      snprintf(buffer, 128,
		       "RIFL_abstract2: new optimized policy-model (%f mean-q, %d iters, epoch %d)",
		       tmp, iters, epoch[1]);
	      whiteice::logging.info(buffer);

	      {
		std::lock_guard<std::mutex> lock(policy_mutex);
		
		policy.importNetwork(nn);

		policy_preprocess.clearData(0);
		policy_preprocess.clearData(1);
		
#if 1
		whiteice::nnetwork<T> nn2;
		std::vector< math::vertex<T> > lagged_weights;
		lagged_policy.exportSamples(nn2, lagged_weights, 1);

		math::vertex<T> weights;
		nn.exportdata(weights);

		if(lagged_weights.size() > 0){

		  if(weights.size() == lagged_weights[0].size()){

		    logging.info("lagged_policy updated");
				 
		    lagged_weights[0] = tau*weights + (T(1.0)-tau)*lagged_weights[0];
		    nn2.importdata(lagged_weights[0]);
		    lagged_policy.importNetwork(nn2);
		  }
		  else{
		    logging.info("lagged_policy updated: NO LAG");
		    
		    nn2 = nn;
		    lagged_policy.importNetwork(nn2);
		  }
		}
		else{
		  logging.info("lagged_policy updated: NO LAG");
		  
		  nn2 = nn;
		  lagged_policy.importNetwork(nn2);
		}
#endif
		
		whiteice::logging.info("RIFL_abstract2: new policy diagnostics");
		lagged_policy.diagnosticsInfo();
		whiteice::logging.info("RIFL_abstract2: new policy-model imported");
	      }

	      grad2.reset();

	      epoch[1]++;
	      hasModel[1]++;
	    }
	    
	  }

	  
	  // skip if other optimization step is behind us
	  // we only start calculating policy after Q() has been optimized..
	  if(epoch[1] >= epoch[0] || epoch[0] == 0) 
	    goto policy_optimization_done;
	  
	  
	  // const unsigned int BATCHSIZE = database.size(); // was 1000
	  // const unsigned int BATCHSIZE = 1000; // was 128

	  if(dataset2_thread == nullptr){
	    data2.clear();
	    data2.createCluster("input-state", numStates);

	    dataset2_thread = new CreatePolicyDataset<T>(*this,
							 database,
							 database_mutex,
							 data2);
	    dataset2_thread->start(SAMPLESIZE);

	    whiteice::logging.info("RIFL_abstract2: new dataset2_thread started (policy)");

	    continue;
	  }
	  else{
	    if(dataset2_thread->isCompleted() != true)
	      continue; // we havent computed proper dataset yet..
	  }

	  if(dataset2_thread){
	    whiteice::logging.info("RIFL_abstract2: dataset2_thread finished (policy)");
	    dataset2_thread->stop();
	  }
	  
	  
	  // fetch NN parameters from model
	  {
	    whiteice::nnetwork<T> q_nn, nn;
	    whiteice::dataset<T> Q_preprocess_copy;

	    {
	      std::lock_guard<std::mutex> lock(Q_mutex);
	      std::vector< math::vertex<T> > weights;
	      
	      if(lagged_Q.exportSamples(q_nn, weights, 1) == false){ // was: lagged_Q
		assert(0);
	      }
	      
	      assert(weights.size() > 0);
	      
	      if(q_nn.importdata(weights[0]) == false){
		assert(0);
	      }

	      Q_preprocess_copy = Q_preprocess;
	    }

	    {
	      std::vector< math::vertex<T> > weights;
	      
	      std::lock_guard<std::mutex> lock(policy_mutex);
	      
	      if(lagged_policy.exportSamples(nn, weights, 1) == false){ // was: lagged_policy
		assert(0);
	      }
	      
	      assert(weights.size() > 0);
	      
	      if(nn.importdata(weights[0]) == false){
		assert(0);
	      }
	    }

	    const bool dropout = false;
	    const bool useInitialNN = true; // WAS: start from scratch everytime
	    
	    
	    if(hasModel[1] >= 0){
	      eta2.start(0.0, P_OPTIMIZE_ITERATIONS);

	      grad2.setUseMinibatch(false);
	      grad2.setSGD(T(-1.0)); // what is correct learning rate???
	      
	      grad2.startOptimize(&data2, q_nn, Q_preprocess_copy, nn, 1, P_OPTIMIZE_ITERATIONS,
				  dropout, useInitialNN);
	    }
	    else{
	      eta2.start(0.0, 15);

	      grad2.setUseMinibatch(false);
	      grad2.setSGD(T(-1.0f)); // disable stochastic gradient descent
	      
	      grad2.startOptimize(&data2, q_nn, Q_preprocess_copy, nn, 1, 15,
				  dropout, useInitialNN);
	    }

	    
	    

	    
	    old_grad2_iterations = -1;
	    
	    if(dataset2_thread) delete dataset2_thread;
	    dataset2_thread = nullptr;
	  }
	  
	}
	else{
	  
	  if(grad2.getSolutionStatistics(meanq, iters)){
	    if(((signed int)iters) > old_grad2_iterations){
	      char buffer[128];
	      
	      double v;
	      whiteice::math::convert(v, meanq);

	      eta2.update(iters);
	      
	      snprintf(buffer, 128,
		       "RIFL_abstract2: grad2 policy-optimizer epoch %d iter %d mean q-value %f [ETA %.2f mins]",
		       epoch[1], iters, v, eta2.estimate()/60.0);
	      
	      whiteice::logging.info(buffer);

	      old_grad2_iterations = (int)iters;
	    }
	  }
	  else{
	    whiteice::logging.error("grad2.getSolutionStatistics() FAILED.");
	  }
	}
      }
      
    policy_optimization_done:
      
      (1 == 1); // dummy [work-around bug/feature goto requiring expression]
      
    }

    grad.stopComputation();
    grad2.stopComputation();

    if(episodesFile) fclose(episodesFile);
    episodesFile = NULL;

    if(dataset_thread){
      delete dataset_thread;
      dataset_thread = nullptr;
    }

    if(dataset2_thread){
      delete dataset2_thread;
      dataset2_thread = nullptr;
    }
    
  }

  template class RIFL_abstract2< math::blas_real<float> >;
  template class RIFL_abstract2< math::blas_real<double> >;
  
};
