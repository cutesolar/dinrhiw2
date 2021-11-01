
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
    best_value = T(-INFINITY);
    best_q_value = T(-INFINITY);
    iterations = 0;

    Q = NULL;
    Q_preprocess = NULL;
    policy = NULL;
    heuristics = false;
    dropout = false;

    this->deep_pretraining = deep_pretraining;
    
    debug = true; // DEBUGing messasges turned ON
    first_time = true;
    
    NTHREADS = 0;
    MAXITERS = 0;

    running = false;
    thread_is_running = 0;

    regularize = true; // REGULARIZER - ENABLED
    regularizer = T(0.01); // was: 1/10.000, was 0.02
    // regularizer = T(0.0); // DISABLED
  }

  template <typename T>
  PolicyGradAscent<T>::PolicyGradAscent(const PolicyGradAscent<T>& grad)
  {
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

    if(Q) delete Q;
    if(Q_preprocess) delete Q_preprocess;
    if(policy) delete policy;

    Q = NULL;
    Q_preprocess = NULL;
    policy = NULL;

    start_lock.unlock();
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
					  const whiteice::nnetwork<T>& Q,
					  const whiteice::dataset<T>& Q_preprocess,
					  // optimized policy
					  const whiteice::nnetwork<T>& policy, 
					  unsigned int NTHREADS,
					  unsigned int MAXITERS,
					  bool dropout,
					  bool initiallyUseNN)
  {
    if(data_ == NULL) return false;
    
    if(data_->getNumberOfClusters() != 1) // dataset only contains state variables
      return false; 
    
    // need at least 1 datapoint(s)
    if(data_->size(0) < 1) return false;
    
    if(data_->dimension(0) != policy.input_size() ||
       data_->dimension(0) + policy.output_size() != Q.input_size())
      return false;
    
    start_lock.lock();
    
    {
      std::lock_guard<std::mutex> lock(thread_is_running_mutex);
      if(thread_is_running > 0){
	start_lock.unlock();
	return false;
      }
    }

    
    data = *data_;
    
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
      auto newQ = new nnetwork<T>(Q); // copies network (settings)      
      auto newpreprocess = new dataset<T>(Q_preprocess);
      
      if(this->Q) delete this->Q;
      if(this->Q_preprocess) delete this->Q_preprocess;
      
      this->Q = newQ;
      this->Q_preprocess = newpreprocess;


      auto newpolicy = new nnetwork<T>(policy);
      
      if(this->policy) delete this->policy;
      this->policy = newpolicy;
      
      if(initiallyUseNN == false) this->policy->randomize();

      // regularizer = T(0.01)*T(5453.0)/T(this->policy->exportdatasize()); // was: 1/10.000, was 0.02
      
      whiteice::logging.info("PolicyGradAscent: input Q weights diagnostics");
      this->Q->diagnosticsInfo();

      whiteice::logging.info("PolicyGradAscent: input policy weights diagnostics");
      this->policy->diagnosticsInfo();
    }

    this->policy->exportdata(bestx);
    best_value = getValue(*(this->policy), *(this->Q), *(this->Q_preprocess), data);
    best_q_value = getValue(*(this->policy), *(this->Q), *(this->Q_preprocess), data);
    
    this->dropout = dropout;

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
    std::lock_guard<std::mutex>  lock1(start_lock);
    std::unique_lock<std::mutex> lock2(thread_is_running_mutex);
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
    // checks if the neural network architecture is the correct one
    if(this->policy == NULL) return false;
    
    solution_lock.lock();
    
    policy = *(this->policy);
    policy.importdata(bestx);
    
    value = best_q_value;
    iterations = this->iterations;
    
    solution_lock.unlock();
    
    return true;
  }
  
  
  template <typename T>
  bool PolicyGradAscent<T>::getSolutionStatistics(T& value,
						  unsigned int& iterations) const
  {
    if(this->policy == NULL) return false;

    solution_lock.lock();

    value = best_q_value;
    iterations = this->iterations;

    solution_lock.unlock();

    return true;
  }

  
  template <typename T>
  bool PolicyGradAscent<T>::getSolution(whiteice::nnetwork<T>& policy) const
  {
    // checks if the neural network architecture is the correct one
    if(this->policy == NULL) return false;
    
    solution_lock.lock();
    
    policy = *(this->policy);
    policy.importdata(bestx);
    
    solution_lock.unlock();
    
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
#pragma omp parallel
    {
      T vsum = T(0.0f);

      math::vertex<T> in(policy.input_size() + policy.output_size());
      in.zero();

      whiteice::math::vertex<T> action;
      whiteice::math::vertex<T> q;
      
      // calculates mean q-value from the testing dataset
#pragma omp for nowait schedule(auto)	    	    
      for(unsigned int i=0;i<dtest.size(0);i++){
	auto state = dtest.access(0, i);
	
	policy.calculate(state, action);

	dtest.invpreprocess(0, state);
	//dtest.invpreprocess(1, action);

	assert(in.write_subvertex(state, 0) == true);
	assert(in.write_subvertex(action, state.size()) == true);
	
	Q_preprocess.preprocess(0, in);
	
	Q.calculate(in, q);

	Q_preprocess.invpreprocess(1, q);
	
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

      // T reg_term = T(1.0)/T(sqrt(this->policy->exportdatasize()));
      // value -= reg_term * T(0.5) * (w*w)[0];

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
    
    
    while(running && iterations < MAXITERS){
      // keep looking for solution until MAXITERS
	
      // starting position for neural network
      // whiteice::nnetwork<T> policy(*(this->policy));
      std::unique_ptr< whiteice::nnetwork<T> > policy(new nnetwork<T>(*(this->policy)));      
      

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
      
      // 2. normal gradient ascent
      ///////////////////////////////////////
      {
	math::vertex<T> weights, w0;
	  
	T prev_value, value;
	T delta_value = 0.0f;

	value = getValue(*policy, *Q, *Q_preprocess, dtest);

	{
	  solution_lock.lock();
	  
	  if(value > best_value){
	    // improvement (larger mean q-value of the policy)
	      best_value = value;
	      best_q_value = getValue(*policy, *Q, *Q_preprocess, dtest);
	      policy->exportdata(bestx);
	      auto ptr = this->policy;
	      this->policy = new whiteice::nnetwork<T>(*policy);
	      delete ptr;
	  }
	  
	  solution_lock.unlock();
	}

	
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

#pragma omp parallel shared(sumgrad)
	  {
	    T ninv = T(1.0f/dtrain.size(0));
	    math::vertex<T> sgrad, grad;
	    grad.resize(policy->exportdatasize());
	    sgrad.resize(policy->exportdatasize());
	    sgrad.zero();

	    whiteice::nnetwork<T> pnet(*policy);
	    math::vertex<T> err;

	    char buffer[80];
	    double temp = 0.0;
	    
#pragma omp for nowait schedule(auto)
	    for(unsigned int i=0;i<dtrain.size(0);i++){
	      
	      if(dropout) pnet.setDropOut();
	      
	      // calculates gradients for Q(state, action(state)) and policy(state)

	      auto state = dtrain.access(0, i); // preprocessed state vector

#if 0
	      if(debug)
	      {
		whiteice::math::convert(temp, state.norm());
		snprintf(buffer, 80, "PolicyGradientAscent: norm(state) = %e\n", 
			 temp);
		whiteice::logging.info(buffer);
	      }
#endif	      
	      
	      whiteice::math::vertex<T> action;
	      
	      assert(pnet.calculate(state, action) == true);

#if 0
	      if(debug)
	      {
		whiteice::math::convert(temp, action.norm());
		snprintf(buffer, 80, "PolicyGradientAscent: norm(action) = %e\n", 
			 temp);
		whiteice::logging.info(buffer);
	      }
#endif
	      
	      whiteice::math::matrix<T> gradP;
	      
	      pnet.jacobian(state, gradP);

#if 0
	      if(debug)
	      {
		whiteice::math::convert(temp, frobenius_norm(gradP));
		snprintf(buffer, 80, "PolicyGradientAscent: norm(gradP) = %e\n", 
			 temp);
		whiteice::logging.info(buffer);
	      }
#endif
	      
	      dtrain.invpreprocess(0, state); // original state for Q network
	      // dtrain.invpreprocess(1, action);

#if 0
	      if(debug)
	      {
		whiteice::math::convert(temp, state.norm());
		snprintf(buffer, 80, "PolicyGradientAscent: norm(invpreprocess(state)) = %e\n", 
			 temp);
		whiteice::logging.info(buffer);
	      }
#endif
	      
	      whiteice::math::vertex<T> in(state.size() + action.size());
	      
	      in.write_subvertex(state, 0);
	      in.write_subvertex(action, state.size());

#if 0
	      if(debug)
	      {
		whiteice::math::convert(temp, in.norm());
		snprintf(buffer, 80, "PolicyGradientAscent: norm(in) = %e\n", 
			 temp);
		whiteice::logging.info(buffer);
	      }
#endif
	      
	      this->Q_preprocess->preprocess(0, in);

#if 0
	      if(debug)
	      {
		whiteice::math::convert(temp, in.norm());
		snprintf(buffer, 80, "PolicyGradientAscent: norm(preprocess(in)) = %e\n", 
			 temp);
		whiteice::logging.info(buffer);
	      }
#endif
	      
	      whiteice::math::matrix<T> gradQ;
	      whiteice::math::vertex<T> Qvalue;
	      {
		this->Q->calculate(in, Qvalue);
		
		whiteice::math::matrix<T> full_gradQ;
		assert(this->Q->gradient_value(in, full_gradQ) == true);


#if 0
		if(debug)
		{
		  whiteice::logging.info("PolicyGradientAscent: Q-network diagnostics");
		  Q->diagnosticsInfo(); // login
		  
		  whiteice::math::convert(temp, frobenius_norm(full_gradQ));
		  snprintf(buffer, 80,
			   "PolicyGradientAscent: norm(full_gradQ) = %e\n", 
			   temp);
		  whiteice::logging.info(buffer);		  
		}
#endif
		
		whiteice::math::matrix<T> Qpostprocess_grad;
		whiteice::math::matrix<T> Qpreprocess_grad_full;
		
		assert(this->Q_preprocess->preprocess_grad(0, Qpreprocess_grad_full) == true);
		assert(this->Q_preprocess->invpreprocess_grad(1, Qpostprocess_grad) == true);

		whiteice::math::matrix<T> Qpreprocess_grad;
		
		assert(Qpreprocess_grad_full.submatrix(Qpreprocess_grad,
						       state.size(), 0,
						       action.size(),
						       Qpreprocess_grad_full.ysize()) == true);

		
		//std::cout << "Qpostprocess_grad = " << Qpostprocess_grad << std::endl;
		//std::cout << "Qpreprocess_full_grad = " << Qpreprocess_grad_full << std::endl;
		//std::cout << "Qpreprocess_grad = " << Qpreprocess_grad << std::endl;
		//std::cout << "full_gradQ " << full_gradQ << std::endl;


#if 0
		if(debug)
		{
		  whiteice::math::convert(temp, frobenius_norm(Qpreprocess_grad_full));
		  snprintf(buffer, 80, "PolicyGradientAscent: norm(Qpreprocess_grad_full) = %e\n", 
			   temp);
		  whiteice::logging.info(buffer);
		}

		if(debug)
		{
		  whiteice::math::convert(temp, frobenius_norm(Qpreprocess_grad));
		  snprintf(buffer, 80, "PolicyGradientAscent: norm(Qpreprocess_grad) = %e\n", 
			   temp);
		  whiteice::logging.info(buffer);
		}

		if(debug)
		{
		  whiteice::math::convert(temp, frobenius_norm(Qpostprocess_grad));
		  snprintf(buffer, 80, "PolicyGradientAscent: norm(Qpostprocess_grad) = %e\n", 
			   temp);
		  whiteice::logging.info(buffer);
		}
#endif
		
		gradQ = Qpostprocess_grad * full_gradQ * Qpreprocess_grad;

#if 0
		if(debug)
		{
		  whiteice::math::convert(temp, frobenius_norm(gradQ));
		  snprintf(buffer, 80, "PolicyGradientAscent: norm(gradQ) = %e\n", 
			   temp);
		  whiteice::logging.info(buffer);
		}
#endif
		
	      }
	      
	      {
		whiteice::math::matrix<T> g;

		g = gradQ * gradP;
		
		assert(g.xsize() == policy->exportdatasize());
		
		for(unsigned int j=0;j<policy->exportdatasize();j++)
		  grad[j] = g(0, j);
	      }

	    
	      sgrad += ninv*grad;
	      // sgrad += grad;
	    }
	    
#pragma omp critical
	    {
	      sumgrad += sgrad;
	    }
	  }

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
	  
	  // lrate = T(0.5f);
	  lrate = sqrt(lrate);
	  lrate *= T(100.0);
	  
	  do{
	    weights = w0;
	    weights += lrate * sumgrad;
	    
	    if(policy->importdata(weights) == false)
	      whiteice::logging.error("PolicyGradAscent: weight import failed");
	    
	    if(dropout) policy->removeDropOut();
	    
	    if(heuristics){
	      normalize_weights_to_unity(*policy);
	    }
	    
	    value = getValue(*policy, *Q, *Q_preprocess, dtrain);
	    
	    delta_value = (prev_value - value); // negative value means value has increased

	    {
	      char buffer[128];
	      
	      double v, p, d, l;
	      whiteice::math::convert(v, value);
	      whiteice::math::convert(p, prev_value);
	      whiteice::math::convert(d, delta_value);
	      whiteice::math::convert(l, lrate);
	      
	      snprintf(buffer, 128,
		       "PolicyGradAscent: gradstep curvalue %f prevvalue %f delta %e lrate %e\n",
		       v, p, d, l);
	      whiteice::logging.info(buffer);
	    }
	    
	    // if value becomes smaller we reduce learning rate
	    if(delta_value >= T(0.0)){ 
	      lrate *= T(0.50);
	    }
	    // value becomes larger we increase learning rate
	    else if(delta_value < T(0.0)){ 
	      lrate *= T(1.0/0.50);
	    }
	    
	  }
	  while(delta_value > T(0.0) && lrate >= T(10e-30) && running);

	  {
	    char buffer[128];
	    
	    double v, d, l;
	    whiteice::math::convert(v, value);
	    whiteice::math::convert(d, delta_value);
	    whiteice::math::convert(l, lrate);
	    
	    snprintf(buffer, 128,
		     "PolicyGradAscent (%d/%d): policy updated value %f delta %e lrate %e\n",
		     iterations, MAXITERS,
		     v, d, l);
	    whiteice::logging.info(buffer);

#if 0
	    whiteice::logging.info("PolicyGradientAscent: policy-network diagnostics");
	    policy->diagnosticsInfo();
#endif
	  }
	  
	  policy->exportdata(weights);
	  w0 = weights;
 	  
	  iterations++;
	  
	  {
	    solution_lock.lock();
	      
	    if(value > best_value){
	      // improvement (larger mean q-value of the policy)
	      best_value = value;
	      best_q_value = getValue(*policy, *Q, *Q_preprocess, dtest);
	      policy->exportdata(bestx);
	      auto ptr = this->policy;
	      this->policy = new whiteice::nnetwork<T>(*policy);
	      delete ptr;
	      
	      {
		char buffer[128];
		
		double b;
		whiteice::math::convert(b, best_q_value);
		
		snprintf(buffer, 128,
			 "PolicyGradAscent: better policy found: %e iter %d",
			 b, iterations);
		whiteice::logging.info(buffer);
	      }
	      
	    }
	    
	    solution_lock.unlock();
	  }
	  
	  // cancellation point
	  {
	    if(running == false){
	      std::lock_guard<std::mutex> lock(thread_is_running_mutex);
	      thread_is_running--;
	      thread_is_running_cond.notify_all();
	      return; // stops execution
	    }
	  }
	  
	  
	}
	while(lrate >= T(10e-30) && 
	      iterations < MAXITERS &&
	      running);
	
	
	
	// 3. after convergence checks if the result is better
	//    than the earlier one
	{
	  solution_lock.lock();
	  
	  if(value > best_value){
	    // improvement (larger mean q-value of the policy)
	    best_value = value;
	    best_q_value = getValue(*policy, *Q, *Q_preprocess, dtest);
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
	  }
	  
	  solution_lock.unlock();
	}
	
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
