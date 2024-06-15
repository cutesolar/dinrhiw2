
#include "PolicyGradAscent.h"

#include "Log.h"
#include "blade_math.h"

#ifdef WINOS
#include <windows.h>
#endif

#include <functional>
#include <sched.h>

namespace whiteice
{

  template <typename T>
  PolicyGradAscent<T>::PolicyGradAscent(bool deep_pretraining)
  {
    std::lock_guard<std::mutex> lock(solution_lock);
    
    best_value = T(-INFINITY);
    best_q_value = T(-INFINITY);
    iterations = 0;

    Q = NULL;
    Q_preprocess = NULL;
    policy = NULL;
    heuristics = false;
    dropout = false;

    use_minibatch = false;

    this->deep_pretraining = deep_pretraining;
    
    debug = true; // DEBUGing messasges turned ON
    first_time = true;
    
    NTHREADS = 0;
    MAXITERS = 0;

    running = false;
    thread_is_running = 0;

    use_SGD = false; // stochastic gradient descent with fixed learning rate
    sgd_lrate = T(0.01f);

    regularize = false; // REGULARIZER - DISABLED
    //regularizer = T(0.001); // was: 1/10.000, was 0.01
    regularizer = T(0.0); // DISABLED
  }

  
  template <typename T>
  PolicyGradAscent<T>::PolicyGradAscent(const PolicyGradAscent<T>& grad)
  {
    std::lock_guard<std::mutex> lock(solution_lock);
    std::lock_guard<std::mutex> lock2(grad.solution_lock);
    
    best_value = grad.best_value;
    best_q_value = grad.best_q_value;
    iterations = grad.iterations;

    if(grad.Q)
      this->Q = new whiteice::nnetwork<T>(*grad.Q);
    else
      this->Q = NULL;

    data = grad.data;

    if(grad.Q_preprocess)
      this->Q_preprocess = new whiteice::dataset<T>(*grad.Q_preprocess);
    else
      this->Q_preprocess = NULL;

    if(grad.policy)
      this->policy = new whiteice::nnetwork<T>(*grad.policy);
    else
      this->policy = NULL;

    heuristics = grad.heuristics;
    dropout = grad.dropout;
    regularizer = grad.regularizer;
    regularize = grad.regularize;
    deep_pretraining = grad.deep_pretraining;

    use_minibatch = grad.use_minibatch;
    
    use_SGD = grad.use_SGD;
    sgd_lrate = grad.sgd_lrate;

    debug = grad.debug;
    first_time = true;

    NTHREADS = 0;
    MAXITERS = 0;

    running = false;
    thread_is_running = 0;
  }

  
  template <typename T>
  PolicyGradAscent<T>::~PolicyGradAscent()
  {
    start_lock.lock();

    if(running){
      running = false;

      for(unsigned int i=0;i<optimizer_thread.size();i++){
	if(optimizer_thread[i]){
	  optimizer_thread[i]->join();
	  delete optimizer_thread[i];
	  optimizer_thread[i] = NULL;
	}
      }
    }

    start_lock.unlock();
    
    {
      std::lock_guard<std::mutex> lock(solution_lock);
      
      if(Q) delete Q;
      if(Q_preprocess) delete Q_preprocess;
      if(policy) delete policy;
      
      Q = NULL;
      Q_preprocess = NULL;
      policy = NULL;
    }

  }


  
  /*
   * starts the optimization process using data as 
   * the dataset as a training and testing data 
   * (implements early stopping)
   *
   * Executes NTHREADS in parallel when looking for
   * the optimal solution and continues up to MAXITERS iterations.
   */
  template <typename T>
  bool PolicyGradAscent<T>::startOptimize(const whiteice::dataset<T>* data_,
					  const whiteice::nnetwork<T>& Q_,
					  const whiteice::dataset<T>& Q_preprocess_,
					  // optimized policy
					  const whiteice::nnetwork<T>& policy_, 
					  unsigned int NTHREADS,
					  unsigned int MAXITERS,
					  bool dropout,
					  bool initiallyUseNN)
  {
    if(data_ == NULL){
      logging.error("PolicyGradAscent:startOptimize(): data_ == NULL error");
      return false;
    }
    
    if(data_->getNumberOfClusters() != 1){ // dataset only contains state variables
      char buf[256];
      snprintf(buf, 256, "PolicyGradAscent:startOptimize(): data_->getNumberOfclusters() = %d",
	       data_->getNumberOfClusters());
      logging.error(buf);
      return false;
    }
    
    // need at least 1 datapoint(s)
    if(data_->size(0) < 1){
      logging.error("PolicyGradAscent:startOptimize(): data_->size(0) < 1 error");
      return false;
    }
    
    if(data_->dimension(0) != policy_.input_size() ||
       data_->dimension(0) + policy_.output_size() != Q_.input_size())
    {
      char buf[256];
      snprintf(buf, 256, "PolicyGradAscent:startOptimize(): dimensions error: %d %d %d %d",
	       data_->dimension(0), policy_.input_size(), policy_.output_size(),
	       Q_.input_size());
      
      logging.error(buf);
      
      return false;
    }
    
    start_lock.lock();
    
    {
      std::lock_guard<std::mutex> lock(thread_is_running_mutex);
      if(thread_is_running > 0){
	start_lock.unlock();
	logging.error("PolicyGradAscent::startOptimize(): thread_is_running > 0 error");
	return false;
      }
    }

    std::lock_guard<std::mutex> lock(solution_lock);
    
    this->data = *data_;
    
    this->NTHREADS = NTHREADS;
    this->MAXITERS = MAXITERS;
    best_value = T(-INFINITY);
    best_q_value = T(-INFINITY);
    iterations = 0;
    running = true;
    thread_is_running = 0;
    
    {
      std::lock_guard<std::mutex> lock(first_time_lock);
      // first thread uses weights from user supplied NN
      first_time = initiallyUseNN;
    }

    // FIXME can run out of memory and throw exception!
    {
      auto newQ = new nnetwork<T>(Q_); // copies network (settings)      
      auto newpreprocess = new dataset<T>(Q_preprocess_);
      
      if(this->Q) delete this->Q;
      if(this->Q_preprocess) delete this->Q_preprocess;
      
      this->Q = newQ;
      this->Q_preprocess = newpreprocess;


      auto newpolicy = new nnetwork<T>(policy_);
      
      if(this->policy) delete this->policy;
      this->policy = newpolicy;
      
      whiteice::logging.info("PolicyGradAscent: input Q weights diagnostics");
      this->Q->diagnosticsInfo();

      whiteice::logging.info("PolicyGradAscent: input policy weights diagnostics");
      this->policy->diagnosticsInfo();
    }

    this->policy->exportdata(bestx);
    this->dropout = dropout;
    
    best_value = getValue(*(this->policy), *(this->Q), *(this->Q_preprocess), data);
    best_q_value = best_value;

    // std::cout << "INITIAL Q-value for policy: " << best_q_value << std::endl;
    
    for(unsigned int i=0;i<optimizer_thread.size();i++){
      if(optimizer_thread[i]){
	optimizer_thread[i]->join();
	delete optimizer_thread[i];
	optimizer_thread[i] = nullptr;
      }
    }
    
    optimizer_thread.clear();
    
    optimizer_thread.resize(NTHREADS);
    
    for(unsigned int i=0;i<optimizer_thread.size();i++){
      optimizer_thread[i] =
	new std::thread(std::bind(&PolicyGradAscent<T>::optimizer_loop,
				  this));
    }

    // waits for threads to start
    {
      std::unique_lock<std::mutex> lock(thread_is_running_mutex);

      // there is a bug if thread manages to notify and then continue and
      // reduce variable back to zero before this get chance to execute again
      while(thread_is_running == 0) 
	thread_is_running_cond.wait(lock);
    }
    
    
    start_lock.unlock();
    
    return true;
  }
    

  template <typename T>
  bool PolicyGradAscent<T>::isRunning()
  {
    std::lock_guard<std::mutex> lock1(start_lock);
    std::lock_guard<std::mutex> lock2(thread_is_running_mutex);
    return running && (thread_is_running > 0);
  }

  
  /*
   * returns the best policy (nnetwork) solution found so far and
   * its average error in testing dataset and the number
   * iterations optimizer has executed so far
   */
  template <typename T>
  bool PolicyGradAscent<T>::getSolution(whiteice::nnetwork<T>& policy,
					T& value,
					unsigned int& iterations) const
  {
    std::lock_guard<std::mutex> lock(solution_lock);
    
    // checks if the neural network architecture is the correct one
    if(this->policy == NULL) return false;
    
    policy = *(this->policy);
    policy.importdata(bestx);
    
    value = best_q_value;
    iterations = this->iterations;
    
    return true;
  }
  
  
  template <typename T>
  bool PolicyGradAscent<T>::getSolutionStatistics(T& value,
						  unsigned int& iterations) const
  {
    std::lock_guard<std::mutex> lock(solution_lock);

    if(this->policy == NULL) return false;

    value = best_q_value;
    iterations = this->iterations;

    return true;
  }

  
  template <typename T>
  bool PolicyGradAscent<T>::getSolution(whiteice::nnetwork<T>& policy) const
  {
    std::lock_guard<std::mutex> lock(solution_lock);
    
    // checks if the neural network architecture is the correct one
    if(this->policy == NULL) return false;
    
    policy = *(this->policy);
    policy.importdata(bestx);
    
    return true;
  }

  template <typename T>
  bool PolicyGradAscent<T>::getDataset(whiteice::dataset<T>& data_) const
  {
    std::lock_guard<std::mutex> lock(solution_lock);
    data_ = this->data;
    return true;
  }
  
  
  /* used to pause, continue or stop the optimization process */
  template <typename T>
  bool PolicyGradAscent<T>::stopComputation()
  {
    start_lock.lock();
    
    if(thread_is_running == 0 || running == false){
      start_lock.unlock();
      return false; // not running (anymore)
    }
    
    running = false;
    
    {
      std::unique_lock<std::mutex> lock(thread_is_running_mutex);
      
      while(thread_is_running > 0)
	thread_is_running_cond.wait(lock);
    }

    
    for(unsigned int i=0;i<optimizer_thread.size();i++){
      if(optimizer_thread[i]){
	optimizer_thread[i]->join();
	delete optimizer_thread[i];
	optimizer_thread[i] = nullptr;
      }
    }
    
    optimizer_thread.clear();
    
    start_lock.unlock();
    
    return true;
  }


  /* resets computation data structures */
  template <typename T>
  void PolicyGradAscent<T>::reset()
  {
    start_lock.lock();

    if(thread_is_running > 0){ // stop running thread
      running = false;
      
      {
	std::unique_lock<std::mutex> lock(thread_is_running_mutex);
	
	while(thread_is_running > 0)
	  thread_is_running_cond.wait(lock);
      }
    }

    for(unsigned int i=0;i<optimizer_thread.size();i++){
      if(optimizer_thread[i]){
	optimizer_thread[i]->join();
	delete optimizer_thread[i];
	optimizer_thread[i] = nullptr;
      }
    }

    optimizer_thread.clear();
    
    
    if(policy) delete policy;
    if(Q) delete Q;
    if(Q_preprocess) delete Q_preprocess;

    policy = nullptr;
    Q = nullptr;
    Q_preprocess = nullptr;

    first_time = true;
    iterations = 0;
    NTHREADS = 0;
    MAXITERS = 0;
    
    best_value = T(-INFINITY);
    best_q_value = T(-INFINITY);
    
    start_lock.unlock();
  }

  //////////////////////////////////////////////////////////////////////
  
  // calculates mean Q-value of the policy in dtest dataset (states are inputs)
  template <typename T>
  T PolicyGradAscent<T>::getValue(const whiteice::nnetwork<T>& policy,
				  const whiteice::nnetwork<T>& Q,
				  const whiteice::dataset<T>& Q_preprocess,
				  const whiteice::dataset<T>& dtest) const
  {
    T value = T(0.0);
      
    // calculates mean q-value of policy

    if(0){
      logging.info("getValue() Q:");
      Q.diagnosticsInfo();
    }
    
#pragma omp parallel
    {
      T vsum = T(0.0f);

      math::vertex<T> in(policy.input_size() + policy.output_size());
      in.zero();

      whiteice::math::vertex<T> action, state;
      whiteice::math::vertex<T> q(1);
      q.zero();
      
      // calculates mean q-value from the testing dataset
#pragma omp for nowait schedule(guided)
      for(unsigned int i=0;i<dtest.size(0);i++){
	state = dtest.access(0, i);
	
	if(policy.calculate(state, action) == false)
	  assert(0);

	dtest.invpreprocess(0, state);
	//dtest.invpreprocess(1, action);

	if(in.write_subvertex(state, 0) == false) assert(0);
	if(in.write_subvertex(action, state.size()) == false) assert(0);

	if(Q_preprocess.preprocess(0, in) == false) assert(0);

	if(Q.calculate(in, q) == false) assert(0);
	// if(Q.calculate_logged(in, q) == false) assert(0);

	if(Q_preprocess.invpreprocess(1, q) == false) assert(0);
	
	vsum += q[0];
      }
	
      vsum /= T((double)dtest.size(0));
      
#pragma omp critical
      {
	value += vsum;
      }
    }

    
    if(regularize){
      // adds regularizer term (-0.5*||w||^2)
      whiteice::math::vertex<T> w;

      policy.exportdata(w);

      value -= regularizer * T(0.5) * (w*w)[0];
    }
    
    return value;
  }

  //////////////////////////////////////////////////////////////////////

  template <typename T>
  void PolicyGradAscent<T>::optimizer_loop()
  {
    
    {
      sched_param sch_params;
      int policy = SCHED_FIFO; // SCHED_RR
      
      pthread_getschedparam(pthread_self(), &policy, &sch_params);

#ifdef linux
      policy = SCHED_IDLE; // in linux we can set idle priority
#endif
      sch_params.sched_priority = sched_get_priority_min(policy);
      
      if(pthread_setschedparam(pthread_self(),
			       policy, &sch_params) != 0){
      }

#ifdef WINOS
      SetThreadPriority(GetCurrentThread(),
			THREAD_PRIORITY_IDLE);
#endif
      
    }

    // 1. divides data to to training and testing sets
    ///////////////////////////////////////////////////
   
    whiteice::dataset<T> dtrain, dtest;
    
    dtrain = data;
    dtest  = data;
    
    dtrain.clearData(0);
    //dtrain.clearData(1);
    
    dtest.clearData(0);
    //dtest.clearData(1);

    unsigned int counter = 0;
    
    while((dtrain.size(0) == 0 || dtest.size(0)  == 0) && counter < 10){
      dtrain.clearData(0);
      //dtrain.clearData(1);
      
      dtest.clearData(0);
      //dtest.clearData(1);
      
      for(unsigned int i=0;i<data.size(0);i++){
	const unsigned int r = (rand() & 3);
	
	const math::vertex<T>& in  = data.access(0, i);
	//const math::vertex<T>& out = data.access(1, i);
	
	if(r != 0){ // 75% will got to training data
	  dtrain.add(0, in,  true);
	  //dtrain.add(1, out, true);
	  
	}
	else{
	  dtest.add(0, in,  true);
	  //dtest.add(1, out, true);
	}
      }

      counter++;
    }

    if(counter >= 10){
      dtrain = data;
      dtest  = data;
    }

    //dtrain = data;
    //dtest  = data;

    dtrain.diagnostics();

    {
      std::lock_guard<std::mutex> lock(thread_is_running_mutex);
      thread_is_running++;
      thread_is_running_cond.notify_all();
    }
    
    // acquires lock temporally to wait for startOptimizer() to finish
    {
      start_lock.lock();
      start_lock.unlock();
    }


    T value = 0.0; 
    
    unsigned int no_improvements_counter = 0;
    const unsigned int MAX_NO_IMPROVE_ITERS = 300;
    
    whiteice::math::vertex<T> m(policy->exportdatasize());
    whiteice::math::vertex<T> v(policy->exportdatasize());
    
    m.zero();
    v.zero();
    
    // starting position for neural network
    std::unique_ptr< whiteice::nnetwork<T> > policy(new nnetwork<T>(*(this->policy)));      
    
    while(running && iterations < MAXITERS && no_improvements_counter < MAX_NO_IMPROVE_ITERS){
      // keep looking for solution until MAXITERS
      
#if 1
      {
	std::lock_guard<std::mutex> lock(first_time_lock);
	
	// use heuristic to normalize weights to unity (keep input weights) [the first try is always given imported weights]
	if(first_time == false){
	  whiteice::logging.info("PolicyGradAscent: reset policy network");
	  
	  policy->randomize();

	  if(deep_pretraining){
	    auto ptr = policy.release();
	    if(deep_pretrain_nnetwork_full_sigmoid(ptr, dtrain, false, 2, &running) == false)
	      whiteice::logging.error("PolicyGradAscent: deep pretraining FAILED");
	    else
	      whiteice::logging.info("PolicyGradAscent: deep pretraining completed");
	    
	    policy.reset(ptr);
	  }
	  
	  if(heuristics){
	    normalize_weights_to_unity(*policy);
	  }

	}
	else{
	  whiteice::logging.info("PolicyGradAscent: use previous/old policy network");
	  
	  first_time = false;
	}
      }
#endif

      {
	value = getValue(*policy, *Q, *Q_preprocess, dtest);

	solution_lock.lock();
	
	if(value > best_value){
	  // improvement (larger mean q-value of the policy)
	  best_value = value;
	  best_q_value = value; // 
	  policy->exportdata(bestx);
	  this->policy->importdata(bestx);
	  
	  //auto ptr = this->policy;
	  //this->policy = new whiteice::nnetwork<T>(*policy);
	  //delete ptr;
	}
	
	solution_lock.unlock();
      }
      

      
      // 2. normal gradient ascent
      ///////////////////////////////////////
      {
	math::vertex<T> weights, w0;
	  
	T prev_value;
	T delta_value = 0.0f;

	
	T lrate = T(1.0);
	
	do{
	  prev_value = value;

	  whiteice::logging.info("PolicyGradAscent: calculates gradient");
	  
	  // goes through data, calculates gradient
	  // exports weights, weights -= lrate*gradient
	  // imports weights back

	  math::vertex<T> sumgrad;
	  sumgrad.resize(policy->exportdatasize());
	  sumgrad.zero();

	  const unsigned int MINIBATCHSIZE = 100;
	  
	  const T ninv = use_minibatch ? T(1.0f/MINIBATCHSIZE) : T(1.0f/dtrain.size(0));
	  

#pragma omp parallel shared(sumgrad)
	  {
	    math::vertex<T> sgrad, grad;
	    grad.resize(policy->exportdatasize());
	    sgrad.resize(policy->exportdatasize());
	    sgrad.zero();

	    whiteice::nnetwork<T> pnet(*policy);
	    math::vertex<T> err;
	    
	    whiteice::math::vertex<T> in(pnet.input_size() + pnet.output_size());

	    whiteice::math::vertex<T> state, action;
	    whiteice::math::matrix<T> gradP;
	    
	    whiteice::math::matrix<T> gradQ;
	    whiteice::math::vertex<T> Qvalue;

	    whiteice::math::matrix<T> full_gradQ;
	    
	    whiteice::math::matrix<T> Qpostprocess_grad;
	    whiteice::math::matrix<T> Qpreprocess_grad_full;
			
	    whiteice::math::matrix<T> Qpreprocess_grad;
	    
	    if(use_minibatch){
	      
#pragma omp for nowait schedule(guided)
	      for(unsigned int i=0;i<MINIBATCHSIZE;i++){

		const unsigned int index = rng.rand() % dtrain.size(0);
		
		if(dropout) pnet.setDropOut();
		
		// calculates gradients for Q(state, action(state)) and policy(state)
		
		state = dtrain.access(0, index); // preprocessed state vector
		
		pnet.calculate(state, action);
		
		pnet.jacobian(state, gradP);
		
		dtrain.invpreprocess(0, state); // original state for Q network
		// dtrain.invpreprocess(1, action);
		
		{
		  in.write_subvertex(state, 0);
		  in.write_subvertex(action, state.size());
		  
		  this->Q_preprocess->preprocess(0, in);
		  
		  this->Q->calculate(in, Qvalue);
		  
		  this->Q->gradient_value(in, full_gradQ);
		  
		  this->Q_preprocess->preprocess_grad(0, Qpreprocess_grad_full);
		  this->Q_preprocess->invpreprocess_grad(1, Qpostprocess_grad);
		  
		  Qpreprocess_grad_full.submatrix(Qpreprocess_grad,
						  state.size(), 0,
						  action.size(),
						  Qpreprocess_grad_full.ysize());
		  
		  
		  gradQ = Qpostprocess_grad * full_gradQ * Qpreprocess_grad;
		  
		}
		
		grad = gradQ * gradP;
		
		//sgrad += ninv*grad;
		sgrad += grad;
	      }
	      
	      
#pragma omp critical
	      {
		sumgrad += sgrad;
	      }
	      
	    }
	    else{
	      
#pragma omp for nowait schedule(guided)
	      for(unsigned int i=0;i<dtrain.size(0);i++){
		
		if(dropout) pnet.setDropOut();
		
		// calculates gradients for Q(state, action(state)) and action=policy(state)
		
		state = dtrain.access(0, i); // preprocessed state vector

		pnet.calculate(state, action);

		pnet.jacobian(state, gradP);
		
		dtrain.invpreprocess(0, state); // original state for Q network
		// dtrain.invpreprocess(1, action);
		
		{
		  in.write_subvertex(state, 0);
		  in.write_subvertex(action, state.size());
		  
		  this->Q_preprocess->preprocess(0, in);
		  
		  this->Q->calculate(in, Qvalue);
		  
		  this->Q->gradient_value(in, full_gradQ);

		  this->Q_preprocess->preprocess_grad(0, Qpreprocess_grad_full);
		  this->Q_preprocess->invpreprocess_grad(1, Qpostprocess_grad);
		  
		  Qpreprocess_grad_full.submatrix(Qpreprocess_grad,
						  state.size(), 0,
						  action.size(),
						  Qpreprocess_grad_full.ysize());
		  
		  
		  gradQ = Qpostprocess_grad * full_gradQ * Qpreprocess_grad;
		  
		}
		
		grad = gradQ * gradP;
		
		//sgrad += ninv*grad;
		sgrad += grad;
	      }
	      
	      
#pragma omp critical
	      {
		sumgrad += sgrad;
	      }
	      
	    }
	  }

	  sumgrad *= ninv;

	  {
	    char buffer[80];
	    double gradlen = 0.0;
	    
	    whiteice::math::convert(gradlen, sumgrad.norm());

	    snprintf(buffer, 80, "PolicyGradAscent: raw gradient norm: %e",
		     gradlen);
	    
	    whiteice::logging.info(buffer);
	  }
	  
	  
	  // cancellation point
	  {
	    if(running == false){
	      std::lock_guard<std::mutex> lock(thread_is_running_mutex);
	      thread_is_running--;
	      thread_is_running_cond.notify_all();
	      return; // cancels execution
	    }
	  }
	  
	  if(policy->exportdata(weights) == false)
	    whiteice::logging.error("PolicyGradAscent: weight export failed");
	  
	  w0 = weights;
	  
	  
	  // adds regularizer to gradient -(1/2*||w||^2)
	  if(regularize)
	  {
	    // regularizer = T(1.0)/T(sqrt(this->policy->exportdatasize()));
	    
	    sumgrad -= regularizer*w0;
	  }

	  if(debug)
	    whiteice::logging.info("PolicyGradAscent: calculates linesearch..");

	  // sumgrad.normalize(); // normalizes gradient length to unit..
	  
	  lrate = T(0.5f);
	  
	  { // use ADAM optimizer
	    
	    const T alpha = T(0.001);
	    const T beta1 = T(0.9);
	    const T beta2 = T(0.999);
	    const T epsilon = T(1e-8);
	      
	    weights = w0;
	    
#pragma omp parallel for schedule(static)
	    for(unsigned int i=0;i<sumgrad.size();i++){
	      // should ASCEND the error.. (-sumgrad goes to larger values)
	      
	      m[i] = beta1 * m[i] + (T(1.0) - beta1)*(-sumgrad[i]);
	      v[i] = beta2 * v[i] + (T(1.0) - beta2)*sumgrad[i]*sumgrad[i];
	      
	      const T m_hat = m[i] / (T(1.0) - whiteice::math::pow(beta1[0], T(iterations+1)[0]));
	      const T v_hat = v[i] / (T(1.0) - whiteice::math::pow(beta2[0], T(iterations+1)[0]));
	      
	      weights[i] -= (alpha / (whiteice::math::sqrt(v_hat) + epsilon)) * m_hat;
	    }
	    
	    policy->importdata(weights);
	    
	    if(heuristics){
	      normalize_weights_to_unity(*policy);
	    }
	    
	    value = getValue(*policy, *Q, *Q_preprocess, dtest);
	  }

	  
	  {
	    char buffer[128];
	    
	    double v;
	    whiteice::math::convert(v, value);
	    
	    snprintf(buffer, 128,
		     "PolicyGradAscent (%d/%d): policy updated value %f\n",
		     iterations, MAXITERS,
		     v);
	    whiteice::logging.info(buffer);
	  }
	  
	  //policy->exportdata(weights);
	  w0 = weights;
 	  
	  iterations++;
	  
	  // cancellation point
	  {
	    if(running == false){
	      std::lock_guard<std::mutex> lock(thread_is_running_mutex);
	      thread_is_running--;
	      thread_is_running_cond.notify_all();
	      return; // stops execution
	    }
	  }

	  // 3. checks if the result is better
	  //    than the earlier one
	  {
	    solution_lock.lock();
	    
	    if(value > best_value){
	      // improvement (larger mean q-value of the policy)
 
	      best_value = value;
	      best_q_value = value; // getValue(*policy, *Q, *Q_preprocess, dtest);
	      policy->exportdata(bestx);
	      auto ptr = this->policy;
	      this->policy = new whiteice::nnetwork<T>(*policy);
	      delete ptr;
	      
	      {
		char buffer[128];
		
		double b;
		whiteice::math::convert(b, best_q_value);
		
		snprintf(buffer, 128,
			 "PolicyGradAscent: better policy found: %f iter %d",
			 b, iterations);
		whiteice::logging.info(buffer);
	      }

	      no_improvements_counter = 0;
	    }
	    else{
	      no_improvements_counter++; 
	    }
	    
	    solution_lock.unlock();
	  }
	  
	}
	while(iterations < MAXITERS &&
	      no_improvements_counter < MAX_NO_IMPROVE_ITERS &&
	      running);
	
	
      }
      
      
    }
    
    
    std::lock_guard<std::mutex> lock(thread_is_running_mutex);
    thread_is_running--;
    thread_is_running_cond.notify_all();
    
    return;
    
    
  }

  //////////////////////////////////////////////////////////////////////
  

  template class PolicyGradAscent< whiteice::math::blas_real<float> >;
  template class PolicyGradAscent< whiteice::math::blas_real<double> >;
};
