
#include "NNGradDescent.h"
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <functional>

#ifdef WINOS
#include <windows.h>
#endif


#include <chrono>
#include <thread>
#include <sstream>
#include <memory>


#include "pretrain.h"


// instead of INFINITY
#define LARGE_INF_VALUE INFINITY


namespace whiteice
{
  namespace math
  {

    template <typename T>
    NNGradDescent<T>::NNGradDescent(bool heuristics, bool deep_pretraining)
    {
      best_error = T(LARGE_INF_VALUE);
      best_pure_error = T(LARGE_INF_VALUE);
      iterations = 0;
      NTHREADS = 0;
      thread_is_running = 0;
      
      this->heuristics = heuristics;
      this->deep_pretraining = deep_pretraining;
      this->use_minibatch = false;
      this->matrix_factorization_pretraining = false;	

      dropout = false;
      overfit = false;
      dont_normalize_error = false; // don't normalize values when calculating error.
      mne = false; // use minimum squared error as the default error term

      running = false;
      nn = NULL;

      first_time = true;

      use_SGD = false;
      sgd_lrate = T(0.0f);

      // regularizer = T(0.0001); // 1/10.000 (keep weights from becoming large)
      // regularizer = T(1.0); // this works for "standard" cases
      
      regularizer = T(0.0f);    // regularizer is DISABLED
    }


    template <typename T>
    NNGradDescent<T>::NNGradDescent(const NNGradDescent<T>& grad)
    {
      best_error = grad.best_error;
      best_pure_error = grad.best_pure_error;
      iterations = grad.iterations;
      data = grad.data;
      NTHREADS = grad.NTHREADS;
      MAXITERS = grad.MAXITERS;      
      
      this->heuristics = grad.heuristics;
      this->deep_pretraining = grad.deep_pretraining;
      this->use_minibatch = grad.use_minibatch;
      this->matrix_factorization_pretraining = grad.matrix_factorization_pretraining;

      this->use_SGD = grad.use_SGD;
      this->sgd_lrate = grad.sgd_lrate;

      dropout = grad.dropout;
      regularizer = grad.regularizer;
      overfit = grad.overfit;
      mne = grad.mne;
      dont_normalize_error = grad.dont_normalize_error; // don't normalize values when calculating error
      
      running = grad.running;

      bestx = grad.bestx;

      if(grad.nn)
	nn = new whiteice::nnetwork<T>(*grad.nn);
      else
	nn = grad.nn;

      data = grad.data;

      first_time = grad.first_time;

      thread_is_running = 0;
      running = false;
    }

    
    template <typename T>
    NNGradDescent<T>::~NNGradDescent()
    {
      start_lock.lock();

      if(running){
	running = false;
	for(unsigned int i=0;i<optimizer_thread.size();i++){
	  optimizer_thread[i]->join();
	  delete optimizer_thread[i];
	  optimizer_thread[i] = nullptr;
	}

	optimizer_thread.clear();
      }

      if(nn) delete nn;
      nn = nullptr;

      start_lock.unlock();
    }


    template <typename T>
    void NNGradDescent<T>::setMatrixFactorizationPretrainer(bool pretrain)
    {
      this->matrix_factorization_pretraining = pretrain;	
    }
    
    template <typename T>
    bool NNGradDescent<T>::getMatrixFactorizationPretrainer() const
    {
      return this->matrix_factorization_pretraining;
    }

    
    template <typename T>
    void NNGradDescent<T>::setUseMinibatch(bool minibatch)
    {
      use_minibatch = minibatch;
    }

    
    template <typename T>
    bool NNGradDescent<T>::getUseMinibatch() const
    {
      return use_minibatch;
    }

    template <typename T>
    void NNGradDescent<T>::setOverfit(bool overfit)
    {
      this->overfit = overfit;
    }

    template <typename T>
    bool NNGradDescent<T>::getOverfit() const
    {
      return overfit;
    }

    
    // sets non-normalization of error values when normalize_error = false
    template <typename T>
    void NNGradDescent<T>::setNormalizeError(bool normalize_error)
    {
      if(normalize_error) dont_normalize_error = false;
      else dont_normalize_error = true;
    }

    template <typename T>
    bool NNGradDescent<T>::getNormalizeError()
    {
      if(dont_normalize_error) return false;
      else return true;
    }

    template <typename T>
    void NNGradDescent<T>::setMNE(bool usemne)
    {
      this->mne = usemne;
    }

    template <typename T>
    bool NNGradDescent<T>::getUseMNE() const
    {
      return this->mne;
    }

    // whether to add alpha*0.5*||w||^2 term to error to keep
    // weight values from exploding. It is RECOMMENDED to enable
    // this for complex valued neural networks because complex
    // non-linearities might easily explode to large values
    //
    // 0.01 seem to work rather well when complex weights are N(0,I)/dim(W(k,:))
    // and data is close to N(0,I) too.
    template <typename T>
    void NNGradDescent<T>::setRegularizer(const T alpha)
    {
      if(real(alpha) >= real(T(0.0f)))
	this->regularizer = alpha;
    }

    // returns regularizer value, zero means regularizing is disabled
    template <typename T>
    T NNGradDescent<T>::getRegularizer() const
    {
      return regularizer;
    }
    
    /*
     * starts the optimization process using data as 
     * the dataset as a training and testing data 
     * (implements early stopping)
     *
     * Executes NTHREADS in parallel when looking for
     * the optimal solution.
     * 
     * initiallyUseNN = true => initially use given parameter nn weights
     */
    template <typename T>
    bool NNGradDescent<T>::startOptimize(const whiteice::dataset<T>& data,
					 const whiteice::nnetwork<T>& nn,
					 unsigned int NTHREADS,
					 unsigned int MAXITERS,
					 bool dropout,
					 bool initiallyUseNN)
    {
      if(data.getNumberOfClusters() < 2){
	char buffer[256];
	sprintf(buffer, "NNGradDescent::startOptimize(): data.getNumberOfClusters() < 2: %d",
		(int)data.getNumberOfClusters());
	logging.error(buffer);
	return false;
      }

      if(data.size(0) != data.size(1)){
	char buffer[256];
	sprintf(buffer, "NNGradDescent::startOptimize(): data.size(0) != data.size(1): %d %d",
		(int)data.size(0), (int)data.size(1));
	logging.error(buffer);
	return false;
      }

      // need at least 1 datapoint(s)
      if(data.size(0) < 1){
	char buffer[256];
	sprintf(buffer, "NNGradDescent::startOptimize(): data.size(0) < 1: %d",
		(int)data.size(0));
	logging.error(buffer);
	return false;
      }

      if(data.dimension(0) != nn.input_size() ||
	 data.dimension(1) != nn.output_size()){
	char buffer[256];
	sprintf(buffer, "NNGradDescent::startOptimize(): nn dimensions mismatch: %d %d %d %d",
		(int)data.dimension(0), (int)data.dimension(1),
		(int)nn.input_size(), (int)nn.output_size());
	logging.error(buffer);
	return false;
      }

      start_lock.lock();

      if(running == true){
	char buffer[256];
	sprintf(buffer, "NNGradDescent::startOptimize(): running is already true.");
	logging.error(buffer);
	
	start_lock.unlock();
	return false;
      }

      {
	running = false;
	
	std::lock_guard<std::mutex> lock(thread_is_running_mutex);
	if(thread_is_running > 0){
	  start_lock.unlock();

	  char buffer[256];
	  sprintf(buffer, "NNGradDescent::startOptimize(): thread_is_running is positive: %d.",
		  thread_is_running);
	  logging.error(buffer);
	  
	  return false;
	}
	
	for(unsigned int i=0;i<optimizer_thread.size();i++){
	  if(optimizer_thread[i]){
	    optimizer_thread[i]->join();
	    delete optimizer_thread[i];
	    optimizer_thread[i] = nullptr;
	  }
	}

	optimizer_thread.clear();
      }
      

      {
	std::lock_guard<std::mutex> lock(solution_lock);
	
	this->data = data;
	this->NTHREADS = NTHREADS;
	this->MAXITERS = MAXITERS;
	best_error = T(LARGE_INF_VALUE);
	best_pure_error = T(LARGE_INF_VALUE);
	iterations = 0;
	running = true;
	thread_is_running = 0;

	this->nn = new nnetwork<T>(nn); // copies network (settings)
	nn.exportdata(bestx);

	if(this->nn->getBatchNorm()){
	  std::vector< whiteice::math::vertex<T> > xdata;
	  
	  this->data.getData(0, xdata);
	  
	  this->nn->calculateBatchNorm(xdata, 2);
	}
	
	best_error = getError(*(this->nn), this->data, (real(regularizer)>real(T(0.0f))), dropout);
	
	if(dropout){
	  auto nn_without_dropout = *(this->nn);
	  nn_without_dropout.removeDropOut();
	  best_pure_error = getError(nn_without_dropout, this->data, false, false);
	}
	else{
	  best_pure_error = getError(*(this->nn), this->data, false, false);
	}
	
      }

      {
	std::lock_guard<std::mutex> lock(first_time_lock);
	// first thread uses weights from user supplied NN
	first_time = initiallyUseNN;
      }

      
      {
	std::lock_guard<std::mutex> lock(noimprove_lock);
	noimprovements.clear();
      }

      {
	std::lock_guard<std::mutex> lock(convergence_lock);
	convergence.clear();
      }
      
      this->dropout = dropout;
      
      optimizer_thread.resize(NTHREADS);

      {
	std::lock_guard<std::mutex> lock(errors_lock);
	errors.clear();
      }

      this->optimize_started_finished = false;
      
      for(unsigned int i=0;i<optimizer_thread.size();i++){
	optimizer_thread[i] =
	  new std::thread(std::bind(&NNGradDescent<T>::optimizer_loop,
				    this));
      }

      {
	std::unique_lock<std::mutex> lock(thread_is_running_mutex);

	// wait for threads to start
	while(thread_is_running < (int)optimizer_thread.size())
	  thread_is_running_cond.wait(lock);

	optimize_started_finished = true;
      }

      
      start_lock.unlock();

      return true;
    }
    
    template <typename T>
    bool NNGradDescent<T>::isRunning()
    {
      std::lock_guard<std::mutex> lock1(start_lock);
      std::lock_guard<std::mutex> lock2(thread_is_running_mutex); // was unique_lock????
      return running && (thread_is_running > 0);
    }

    
    template <typename T>
    bool NNGradDescent<T>::hasConverged(T percentage)
    {
      if(real(percentage) <= real(T(0.0f))) return true;
      if(real(percentage) >= real(T(1.0f))) return false;

      {
	std::lock_guard<std::mutex> lock(convergence_lock);
	
	if(convergence.size() != NTHREADS) return false;
	unsigned int threadnum = 0;

	for(const auto& c : convergence){
	  if(c.second == true) threadnum++;
	}

	if(threadnum == convergence.size())
	  return true;
      }

#if 0
      // this does not work anymore as we allow continuation
      // for sometime after better values are not found.
      {
	std::vector<bool> convergence;
	
	std::lock_guard<std::mutex> lock(errors_lock);

	if(errors.size() != NTHREADS) return false;
	unsigned int threadnum = 0;
	
	for(const auto& err : errors){
	  if(err.second.size() >= EHISTORY){
	    T m = T(0.0), v = T(0.0);
	    
	    for(const auto& e : err.second){
	      m += e;
	    }
	    
	    m /= err.second.size();

	    for(const auto& e : err.second){
	      v += (e - m)*(e - m);
	    }
	    
	    v /= (err.second.size() - 1);
	    
	    v = sqrt(real(v));

	    std::cout << "THREAD " << threadnum << "/" << (int)errors.size()
		      << " (" << err.second.size() << " samples)"
		      << ": ERROR CONVERGENCE: " << T(100.0)*v/m << "% (convergence: "
		      << T(100.0)*percentage << "%)"
		      << std::endl;

	    if(real(v/m) <= real(percentage)) // 1% is a good value
	      convergence.push_back(true);
	    else
	      convergence.push_back(false);
	  }
	  else{
	    convergence.push_back(false);
	  }

	  threadnum++;
	}
	
	
	unsigned int cnum = 0;
	
	for(const auto& c : convergence){
	  if(c == true) cnum++;
	}
	
	if(cnum == NTHREADS)
	  return true;
      }
#endif

      return false;
    }
    
    
    /*
     * returns the best NN solution found so far and
     * its average error in testing dataset and the number
     * iterations optimizer has executed so far
     */
    template <typename T>
    bool NNGradDescent<T>::getSolution(whiteice::nnetwork<T>& nn,
				       T& error,
				       unsigned int& iterations) const
    {
      // checks if the neural network architecture is the correct one
      if(this->nn == NULL) return false;

      {
	solution_lock.lock();
	
	nn = *(this->nn);
	nn.importdata(bestx);
	
	error = best_pure_error; // FULL DATASET ERROR
	//error = best_error; // VALIDATION DATASET ERROR
	iterations = this->iterations;
	
	solution_lock.unlock();
      }

      return true;
    }


    template <typename T>
    bool NNGradDescent<T>::getSolutionStatistics(T& error,
						 unsigned int& iterations) const
    {
      // checks if the neural network architecture is the correct one
      if(this->nn == NULL) return false;

      {
	solution_lock.lock();
	
	error = best_pure_error; // FULL DATASET ERROR
	// error = best_error; // VALIDATION DATASET ERROR
	iterations = this->iterations;
	
	solution_lock.unlock();
      }
	
      return true;
    }


    template <typename T>
    bool NNGradDescent<T>::getSolution(whiteice::nnetwork<T>& nn) const
    {
      // checks if the neural network architecture is the correct one
      if(this->nn == NULL) return false;

      {
	solution_lock.lock();
	
	nn = *(this->nn);
	nn.importdata(bestx);
	
	solution_lock.unlock();
      }
	
      return true;
    }
    

    
    /* used to pause, continue or stop the optimization process */
    template <typename T>
    bool NNGradDescent<T>::stopComputation()
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

      start_lock.unlock();

      return true;
    }


    // removes solution
    template <typename T>
    void NNGradDescent<T>::reset()
    {
      start_lock.lock();

      if(running){
	running = false;
	for(unsigned int i=0;i<optimizer_thread.size();i++){
	  optimizer_thread[i]->join();
	  delete optimizer_thread[i];
	  optimizer_thread[i] = nullptr;
	}

	optimizer_thread.clear();
      }
      
      if(nn) delete nn;
      nn = nullptr;

      best_error = T(LARGE_INF_VALUE);
      best_pure_error = T(LARGE_INF_VALUE);
      iterations = 0;
      NTHREADS = 0;
      thread_is_running = 0;
      
      this->use_minibatch = false;

      dropout = false;
      overfit = false;
      dont_normalize_error = false;
      mne = false; // use minimum squared error as the default error term

      running = false;
      nn = NULL;

      first_time = true;

      // KEEP THE EXISTING REGULARIZER IF IT HAS BEEN ENABLED
      
      // regularizer = T(0.0001); // 1/10.000 (keep weights from becoming large)
      // regularizer = T(1.0); // this works for "standard" cases
      
      // regularizer = T(0.0f);    // regularizer is DISABLED

      start_lock.unlock();
    }


    template <typename T>
    T NNGradDescent<T>::getError(const whiteice::nnetwork<T>& net,
				 const whiteice::dataset<T>& dtest,
				 const bool regularize,
				 const bool dropout) const
    {

      // debugging
      {
	char buf[256];

	snprintf(buf, 256,
		 "getError() init: data.size(0)=%d use_minibatch=%d regularize=%d dropout=%d\n",
		 dtest.size(0), (int)use_minibatch, (int)regularize, (int)dropout);
	whiteice::logging.info(buf);

	dtest.diagnostics(0);
	dtest.diagnostics(1);
	net.diagnosticsInfo();
      }

      
      // error term is E[ 0.5*||y-f(x)||^2 ]
      // in case mne (minimum norm error) error term is E[||y-f(x)||]
      T error = T(0.0f);

      if(use_minibatch){

	// number of samples used to estimate error
	const unsigned int MINIBATCHSIZE = 1000 > dtest.size(0) ? dtest.size(0) : 1000;

	// calculates error
#pragma omp parallel
	{
	  T esum = T(0.0f);
	  
	  const whiteice::nnetwork<T>& nnet = net;
	  
	  std::vector< std::vector<bool> > net_dropout;
	  
	  math::vertex<T> err;
	  math::vertex<T> out;
	  
	  // calculates error from the testing dataset
#pragma omp for nowait schedule(guided)
	  for(unsigned int i=0;i<MINIBATCHSIZE;i++){
	    const unsigned int index = whiteice::rng.rand() % dtest.size(0);
	    
	    auto yvalue = dtest.access(1, index);
	    
	    if(dropout){
	      nnet.setDropOut(net_dropout);
	      nnet.calculate(dtest.access(0, index), out, net_dropout);
	    }
	    else{
	      nnet.calculate(dtest.access(0, index), out);
	    }
	    
	    if(dont_normalize_error){
	      dtest.invpreprocess(1, yvalue);
	      dtest.invpreprocess(1, out);
	    }
	    
	    err = out - yvalue;
	    
	    if(mne) esum += err.norm();
	    else{
	      auto n = err.norm();
	      esum += T(0.5f)*n*n;
	    }
	    
	  }
	  
	  // esum /= T((float)dtest.size(0));
	  esum /= T((float)MINIBATCHSIZE);
	  
#pragma omp critical
	  {
	    error += esum;
	  }
	}
	
	// divides per output dimension
	// [error is per one dimension so it is more comparable]
	error /= T((float)dtest.access(1,0).size());
	
      }
      else{
	
	// calculates error
#pragma omp parallel
	{
	  T esum = T(0.0f);
	  
	  const whiteice::nnetwork<T>& nnet = net;
	  std::vector< std::vector<bool> > net_dropout;
	  
	  math::vertex<T> err;
	  math::vertex<T> out;
	  
	  // calculates error from the testing dataset
#pragma omp for nowait schedule(guided)
	  for(unsigned int i=0;i<dtest.size(0);i++){
	    const unsigned int index = i; // rng.rand() % dtest.size(0);
	    
	    auto yvalue = dtest.access(1, index);
	    
	    if(dropout){
	      nnet.setDropOut(net_dropout);
	      nnet.calculate(dtest.access(0, index), out, net_dropout);
	    }
	    else{
	      nnet.calculate(dtest.access(0, index), out);
	    }
	    
	    if(dont_normalize_error){
	      dtest.invpreprocess(1, yvalue);
	      dtest.invpreprocess(1, out);
	    }
	    
	    err = out - yvalue;
	    
	    if(mne) esum += err.norm();
	    else{
	      auto n = err.norm();
	      esum += T(0.5f)*n*n;
	    }

	    	
	    //if(esum >= T(10e9)){
	    //  assert(0);
	    //}
	    
	  }
	  
	  esum /= T((float)dtest.size(0));
	  // esum /= T((float)MINIBATCHSIZE);
	  
#pragma omp critical
	  {
	    error += esum;
	  }
	}
	
	// divides per output dimension
	// [error is per one dimension so it is more comparable]
	error /= T((float)dtest.access(1,0).size());
	
      }

      if(regularize){
	whiteice::math::vertex<T> w;

	net.exportdata(w);

	auto n = w.norm();

	error += regularizer * T(0.5f) * n*n;
      }

      // shouldn't needed but should help with wierd number problems
      {
	error = abs(error); 
      }

      return error;
    }



    
    template <typename T>
    void NNGradDescent<T>::optimizer_loop()
    {
      try{

      
      // set thread priority (non-standard)
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

      {
	std::lock_guard<std::mutex> lock(convergence_lock);
	
	convergence[std::this_thread::get_id()] = false;
      }

      
      // 1. divides data to to training and testing sets
      ///////////////////////////////////////////////////
      
      whiteice::dataset<T> dtrain, dtest;
      
      dtrain = this->data;
      dtest  = this->data;

      if(overfit == false){ // divides data to separate training and testing datasets
	
	dtrain.clearData(0);
	dtrain.clearData(1);
	dtest.clearData(0);
	dtest.clearData(1);
	
	int counter = 0;
	
	while((dtrain.size(0) == 0 || dtrain.size(1) == 0 ||
	       dtest.size(0)  == 0 || dtest.size(1)  == 0) && counter < 10){
	  
	  dtrain.clearData(0);
	  dtrain.clearData(1);
	  dtest.clearData(0);
	  dtest.clearData(1);
	  
	  for(unsigned int i=0;i<data.size(0);i++){
	    const unsigned int r = (whiteice::rng.rand() & 3);
	    
	    if(r != 0){ // 75% will go to training data
	      math::vertex<T> in  = data.access(0,i);
	      math::vertex<T> out = data.access(1,i);
	      
	      dtrain.add(0, in,  true);
	      dtrain.add(1, out, true);
	    }
	    else{ // 25% will go to testing data
	      math::vertex<T> in  = data.access(0,i);
	      math::vertex<T> out = data.access(1,i);
	      
	      dtest.add(0, in,  true);
	      dtest.add(1, out, true);
	    }
	    
	  }
	  
	  counter++;
	}
	
	if(counter >= 10){ // too little data to divive datasets
	  dtrain = this->data;
	  dtest  = this->data;
	}
      }

      {
	std::lock_guard<std::mutex> lock(thread_is_running_mutex);
	thread_is_running++;
	thread_is_running_cond.notify_all();
      }

      {
	// wait for startOptimizer() to finish
	while(optimize_started_finished == false){
	  std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}	
      }
      
      if(use_adam){
	// Use Adam Optimizer instead which should often give better results..
	
	const T alpha = T(0.001);
	const T beta1 = T(0.9);
	const T beta2 = T(0.999);
	const T epsilon = T(1e-8);
	
	std::list<T> errors; // used for convergence check

	std::unique_ptr< nnetwork<T> > nn(new nnetwork<T>(*(this->nn)));

	{
	  std::lock_guard<std::mutex> lock(first_time_lock);

	  if(first_time == false){ // don't use initial weights..
	    nn->randomize(); // NO pretraining for now
	    first_time = true;
	  }
	}
	
	vertex<T> x;
	T real_besty, pure_real_besty;

	{
	  nn->exportdata(x);

	  if(dropout){
	    auto nn_without_dropout = *nn;
	    nn_without_dropout.removeDropOut();
	    pure_real_besty = getError(nn_without_dropout, dtest, false, false);
	  }
	  else{
	    pure_real_besty = getError(*nn, dtest, false, false);
	  }

	  real_besty = pure_real_besty;

	  {
	    std::lock_guard<std::mutex> lock(solution_lock);
	    
	    this->best_error = pure_real_besty;
	    this->best_pure_error = pure_real_besty;
	  }
	}

	unsigned int no_improve_iterations = 0;

	const bool smart_convergence_check = true;
	
	vertex<T> sumgrad(x.size());
	
	vertex<T> m(x.size());
	vertex<T> v(x.size());
	
	m.zero();
	v.zero();
	
	
	while(running && iterations < MAXITERS){

	  {
	    std::lock_guard<std::mutex> lock(solution_lock);
	    
	    iterations++;
	  }

	  if(nn->getBatchNorm()){
	    std::vector< whiteice::math::vertex<T> > xdata;
	    
	    dtrain.getData(0, xdata);
	    
	    nn->calculateBatchNorm(xdata, 2);
	  }

	  
	  
	  // calculates gradient 
	  {
	    sumgrad.zero();
	    
	    // number of samples used to estimate gradient in minibatch mode
	    const unsigned int MINIBATCHSIZE = 1000 > dtrain.size(0) ? dtrain.size(0) : 1000;
	      
	    if(use_minibatch){
	      const T ninv = T(1.0f/MINIBATCHSIZE);
		
#pragma omp parallel shared(sumgrad)
	      {
		math::vertex<T> sgrad, grad;
		sgrad.resize(nn->exportdatasize());
		sgrad.zero();
		
		const whiteice::nnetwork<T>& nnet = *nn;
		std::vector< math::vertex<T> > bpdata;
		std::vector< std::vector<bool> > net_dropout; // NOTE: dropout code don't work?
		  
		math::vertex<T> err;
		math::vertex<T> output;
		
#pragma omp for nowait schedule(guided)
		for(unsigned int i=0;i<MINIBATCHSIZE;i++){
		  const unsigned int index = rng.rand() % dtrain.size(0);
		  // const unsigned intnet index = i;
		  
		  if(dropout){
		    nnet.setDropOut(net_dropout);
		    if(nnet.calculate(dtrain.access(0, index), output,
				      net_dropout, bpdata) == false){
		      std::cout << "calculate failed (1)." << std::endl;
		      assert(0);
		    }
		  }
		  else{
		    if(nnet.calculate(dtrain.access(0, index), output,
				      bpdata) == false){
		      std::cout << "calculate failed (2)." << std::endl;
		      assert(0);
		    }
		  }
		  
		  err = output - dtrain.access(1, index);
		  
		  if(mne) err.normalize(); // minimum norm error gradient instead
		  
		  if(dropout){
		    if(nnet.mse_gradient(err, bpdata, net_dropout, grad) == false){
		      std::cout << "gradient failed (1)." << std::endl;
		      assert(0);
		    }
		  }
		  else{
		    if(nnet.mse_gradient(err, bpdata, grad) == false){
		      std::cout << "gradient failed (2)." << std::endl;
		      assert(0);
		    }
		  }
		  
		  // sgrad += ninv*grad;
		  sgrad += grad;
		}
		
#pragma omp critical
		{
		  sumgrad += sgrad;
		}
		
	      }
	      
	      sumgrad *= ninv;
	    }
	    else{ // do not use minibatch but ALL data is used to compute gradient
		
	      const T ninv = T(1.0f/dtrain.size(0));
	      
#pragma omp parallel shared(sumgrad)
	      {
		math::vertex<T> sgrad, grad;
		sgrad.resize(nn->exportdatasize());
		sgrad.zero();
		
		const whiteice::nnetwork<T>& nnet = *nn;
		std::vector< math::vertex<T> > bpdata;
		std::vector< std::vector<bool> > net_dropout;
		
		math::vertex<T> output;
		math::vertex<T> err;
		
#pragma omp for nowait schedule(guided)
		for(unsigned int i=0;i<dtrain.size(0);i++){
		  const unsigned int index = i;
		  
		  if(dropout){
		    nnet.setDropOut(net_dropout);
		    if(nnet.calculate(dtrain.access(0, index), output,
				      net_dropout, bpdata) == false){
		      std::cout << "calculate failed. (3)" << std::endl;
		      assert(0);
		    }
		  }
		  else{
		    if(nnet.calculate(dtrain.access(0, index), output,
				      bpdata) == false){
		      std::cout << "calculate failed. (4)" << std::endl;
		      assert(0);
		    }
		  }
		  
		  err = output - dtrain.access(1, index);
		  
		  if(mne) err.normalize(); // minimum norm error gradient instead
		  
		  if(dropout){
		    if(nnet.mse_gradient(err, bpdata, net_dropout, grad) == false){
		      std::cout << "gradient failed (3)." << std::endl;
		      assert(0);
		    }
		  }
		  else{
		    if(nnet.mse_gradient(err, bpdata, grad) == false){
		      std::cout << "gradient failed (4)." << std::endl;
		      assert(0);
		    }
		  }
		  
		  // sgrad += ninv*grad;
		  sgrad += grad;
		}
		
#pragma omp critical
		{
		  sumgrad += sgrad;
		}
		
	      }
	      
	      sumgrad *= ninv;
	    }
	    
	    
	  }

#pragma omp parallel for schedule(static)
	  for(unsigned int i=0;i<sumgrad.size();i++){
	    m[i] = beta1 * m[i] + (T(1.0) - beta1)*sumgrad[i];
	    v[i] = beta2 * v[i] + (T(1.0) - beta2)*sumgrad[i]*sumgrad[i];

	    const T m_hat = m[i] / (T(1.0) - whiteice::math::pow(beta1[0], T(iterations)[0]));
	    const T v_hat = v[i] / (T(1.0) - whiteice::math::pow(beta2[0], T(iterations)[0]));

	    x[i] -= (alpha / (whiteice::math::sqrt(v_hat) + epsilon)) * m_hat;
	  }

	  // if(heuristics) heuristics(x);
	  
	  nn->importdata(x);

	  
	  T new_error = T(0.0);

	  if(dropout){
	    auto nn_without_dropout = *nn;
	    nn_without_dropout.removeDropOut();
	    new_error = getError(nn_without_dropout, dtest, false, false);
	  }
	  else{
	    new_error = getError(*nn, dtest, false, false);
	  }	  

	  {
	    std::lock_guard<std::mutex> lock(solution_lock);
	    
	    if(new_error >= real_besty){
	      no_improve_iterations++;
	    }
	    else{
	      if(new_error < this->best_pure_error){
		this->bestx = x;
		this->best_error = new_error;
		this->best_pure_error = new_error;
		(*(this->nn)) = *nn;
	      }
	      
	      real_besty = new_error;
	      pure_real_besty = new_error;
	      
	      no_improve_iterations = 0;
	    }
	  }

	  
	  if(smart_convergence_check){
	    errors.push_back(pure_real_besty); // NOTE: getError() must return >= 0.0 values
	    
	    if(errors.size() >= 250){
	      
	      while(errors.size() > 250)
		errors.pop_front();
	      
	      // make all values to be positive
	      
	      T m = T(0.0f);
	      
	      for(const auto& e : errors)
		m += abs(e);
	      
	      m /= errors.size();
	      
	      T s = T(0.0f);
	      
	      for(const auto& e : errors){
		s += abs(e-m)/sqrt(T(errors.size()));
	      }
	      
	      T r = T(0.0f);
	      
	      if(m > T(0.0f))
		r = s/m;
	      else
		r = s;
	      
	      if(r[0] <= T(0.005f)[0]){ // convergence: 0.1% st.dev. when compared to mean.
		convergence[std::this_thread::get_id()] = true;
		//solution_converged = true;
		break;
	      }
	      
	    }
	  }
	  
	}
	
	
	
      }
      else{ // don't use_ADAM
      
	while(running && iterations < MAXITERS){ // keep looking for solution (forever)
	  
	// starting location for neural network
	  std::unique_ptr< nnetwork<T> > nn(new nnetwork<T>(*(this->nn)));
	  
	  {
	    char buffer[256];
	    
	    std::ostringstream ss;
	    ss << std::this_thread::get_id();
	    std::string str_id = ss.str();
	    
	    whiteice::math::vertex<T> w;
	    nn->exportdata(w);
	    
	    snprintf(buffer, 256, "NNGradDescent: %d/%d (%s) reset/fresh neural network. param norm %f.", iterations, MAXITERS, str_id.c_str(), w.norm()[0].c[0]);
	    whiteice::logging.info(buffer);
	  }
	  
	  
	  {
	    std::lock_guard<std::mutex> lock(first_time_lock);
	    
	    // use heuristic to normalize weights to unity
	    // (keep input weights) [the first try is always given imported weights]
	    
	    if(first_time == false){
	      
	      nn->randomize();
	      
	      whiteice::logging.info("NNGradDescent: resets neural network (nn::randomize() called)");

	      
	      if(typeid(T) == typeid(whiteice::math::blas_real<float>) ||
		 typeid(T) == typeid(whiteice::math::blas_real<double>)){
		
		if(this->matrix_factorization_pretraining){
		  
#if 0
		  bool residual = nn->getResidual();
		  std::vector< enum whiteice::nnetwork<T>::nonLinearity > nonlins;
		  nn->getNonlinearity(nonlins);
		  
		  nn->setResidual(false);
		  nn->setNonlinearity(whiteice::nnetwork< T >::pureLinear);
#endif
		  
		  
		  whiteice::logging.info("NNGradDescent: starting Matrix Factorization Pretrainer.");
		  
		  whiteice::PretrainNN<T> pretrainer;
		  
		  pretrainer.setMatrixFactorization(true);
		  
		  if(pretrainer.startTrain(*nn, dtrain, 50)){
		    
		    while(pretrainer.isRunning()){
		      sleep(1);
		    }
		    
		    pretrainer.stopTrain();
		    
		    pretrainer.getResults(*nn);
		  }
		  
#if 0
		  nn->setResidual(residual);
		  nn->setNonlinearity(nonlins);
#endif
		  
		  whiteice::logging.info("NNGradDescent: stopping Matrix Factorization Pretrainer.");
		  
		}
		
	      }
	      
	      
#if 0
	      // disable deep pretraining and normalize weights to unity as they
	      // don't implement complex numbers..
	      
	      if(deep_pretraining){
		auto ptr = nn.release();
		// verbose = 2: logs training to log file..
		if(deep_pretrain_nnetwork(ptr, dtrain, false, 2, &running) == false)
		  whiteice::logging.error("NNGradDescent: deep pretraining FAILED");
		else
		  whiteice::logging.info("NNGradDescent: deep pretraining completed");
		
		nn.reset(ptr);
	      }
	      
	      
	      if(heuristics){
		//normalize_weights_to_unity(*nn);
		
		/*
		  if(whiten1d_nnetwork(*nn, dtrain) == false)
		  printf("ERROR: whiten1d_nnetwork failed\n");
		  
		*/
		/*
		  normalize_weights_to_unity(*nn);
		  T alpha = T(0.5f);
		  negative_feedback_between_neurons(*nn, dtrain, alpha);
		*/
	      }
#endif
	      
	    }
	    else{
	      first_time = false;
	      
	      
	      if(typeid(T) == typeid(whiteice::math::blas_real<float>) ||
		 typeid(T) == typeid(whiteice::math::blas_real<double>)){
		
		if(this->matrix_factorization_pretraining){
		  
#if 0
		  bool residual = nn->getResidual();
		  std::vector< enum whiteice::nnetwork<T>::nonLinearity > nonlins;
		  
		  nn->getNonlinearity(nonlins);
		  
		  nn->setResidual(false);
		  nn->setNonlinearity(whiteice::nnetwork< T >::pureLinear);
#endif
		  
		  whiteice::logging.info("NNGradDescent: starting Matrix Factorization Pretrainer.");
		  
		  whiteice::PretrainNN<T> pretrainer;
		  
		  pretrainer.setMatrixFactorization(true);
		  
		  if(pretrainer.startTrain(*nn, dtrain, 50)){
		    
		    while(pretrainer.isRunning()){
		      sleep(1);
		    }
		    
		    pretrainer.stopTrain();
		    
		    pretrainer.getResults(*nn);
		  }
		  
#if 0
		  nn->setResidual(residual);
		  nn->setNonlinearity(nonlins);
#endif
		  whiteice::logging.info("NNGradDescent: stopping Matrix Factorization Pretrainer.");
		  
		}
	      }
	      
	    }
	    
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
	  
	  
	  
	  // 2. normal gradient descent
	  ///////////////////////////////////////
	  {
	    math::vertex<T> weights, w0;
	    
	    T prev_error, error;	  
	    T delta_error = T(0.0f);
	    
	    
	    error = getError(*nn, dtest, (real(regularizer)>real(T(0.0f))), dropout);
	    
	    {
	      solution_lock.lock();
	      
	      if(real(error) < real(best_error)){
		// improvement (smaller error with early stopping)
		
		if(dropout){
		  auto nn_without_dropout = *nn;
		  nn_without_dropout.removeDropOut();
		  
		  const T gerror = getError(nn_without_dropout, data,
					    false, false);
		  
		  if(real(gerror) < real(best_pure_error)){
		    nn_without_dropout.exportdata(bestx);
		    best_error = error;
		    best_pure_error = gerror;
		  }
		}
		else{
		  const T gerror = getError(*nn, data, false, false);
		  
		  if(real(gerror) < real(best_pure_error)){
		    nn->exportdata(bestx);
		    best_error = error;
		    best_pure_error = gerror;
		  }
		}
		
	      }
	      
	      solution_lock.unlock();
	    }
	    
	    prev_error = error;

	    
	    // resets no improvement counter to check for convergence
	    // and sets best error for this loop iteration
	    T local_thread_best_error = T(LARGE_INF_VALUE);
	    
	    {
	      std::lock_guard<std::mutex> lock(noimprove_lock);
	      
	      noimprovements[std::this_thread::get_id()] = 0;
	      local_thread_best_error = error;
	    }
	    
	    
	    T lrate = T(0.01f);
	    T ratio = T(1.0f);
	    
	    error = getError(*nn, dtrain, (real(regularizer)>real(T(0.0f))), dropout);
	    
	    do
	    {
	      prev_error = error;
	      
	      // goes through data, calculates gradient
	      // exports weights, weights -= lrate*gradient
	      // imports weights back
	      
	      math::vertex<T> sumgrad;
	      sumgrad.resize(nn->exportdatasize());
	      sumgrad.zero();
	      
	      // number of samples used to estimate gradient in minibatch mode
	      const unsigned int MINIBATCHSIZE = 1000 > dtrain.size(0) ? dtrain.size(0) : 1000;
	      
	      if(use_minibatch){
		const T ninv = T(1.0f/MINIBATCHSIZE);
		
#pragma omp parallel shared(sumgrad)
		{
		  math::vertex<T> sgrad, grad;
		  sgrad.resize(nn->exportdatasize());
		  sgrad.zero();
		  
		  const whiteice::nnetwork<T>& nnet = *nn;
		  std::vector< math::vertex<T> > bpdata;
		  std::vector< std::vector<bool> > net_dropout;
		  
		  math::vertex<T> err;
		  math::vertex<T> output;
		  
#pragma omp for nowait schedule(guided)
		  for(unsigned int i=0;i<MINIBATCHSIZE;i++){
		    const unsigned int index = rng.rand() % dtrain.size(0);
		    // const unsigned intnet index = i;
		    
		    if(dropout){
		      nnet.setDropOut(net_dropout);
		      if(nnet.calculate(dtrain.access(0, index), output,
					net_dropout, bpdata) == false){
			std::cout << "calculate failed (1)." << std::endl;
			assert(0);
		      }
		    }
		    else{
		      if(nnet.calculate(dtrain.access(0, index), output,
					bpdata) == false){
			std::cout << "calculate failed (2)." << std::endl;
			assert(0);
		      }
		    }
		    
		    err = output - dtrain.access(1, index);
		    
		    if(mne) err.normalize(); // minimum norm error gradient instead
		    
		    if(dropout){
		      if(nnet.mse_gradient(err, bpdata, net_dropout, grad) == false){
			std::cout << "gradient failed (1)." << std::endl;
			assert(0);
		      }
		    }
		    else{
		      if(nnet.mse_gradient(err, bpdata, grad) == false){
			std::cout << "gradient failed (2)." << std::endl;
			assert(0);
		      }
		    }
		    
		    // sgrad += ninv*grad;
		    sgrad += grad;
		  }
		  
#pragma omp critical
		  {
		    sumgrad += sgrad;
		  }
		  
		}
		
		
		sumgrad *= ninv;
	      }
	      else{ // do not use minibatch but ALL data is used to compute gradient
		
		const T ninv = T(1.0f/dtrain.size(0));
		
#pragma omp parallel shared(sumgrad)
		{
		  math::vertex<T> sgrad, grad;
		  sgrad.resize(nn->exportdatasize());
		  sgrad.zero();
		  
		  const whiteice::nnetwork<T>& nnet = *nn;
		  std::vector< math::vertex<T> > bpdata;
		  std::vector< std::vector<bool> > net_dropout;
		  
		  math::vertex<T> output;
		  math::vertex<T> err;
		  
#pragma omp for nowait schedule(guided)
		  for(unsigned int i=0;i<dtrain.size(0);i++){
		    const unsigned int index = i;
		    
		    if(dropout){
		      nnet.setDropOut(net_dropout);
		      if(nnet.calculate(dtrain.access(0, index), output,
					net_dropout, bpdata) == false){
			std::cout << "calculate failed. (3)" << std::endl;
			assert(0);
		      }
		    }
		    else{
		      if(nnet.calculate(dtrain.access(0, index), output,
					bpdata) == false){
			std::cout << "calculate failed. (4)" << std::endl;
			assert(0);
		      }
		    }
		    
		    err = output - dtrain.access(1, index);
		    
		    if(mne) err.normalize(); // minimum norm error gradient instead
		    
		    if(dropout){
		      if(nnet.mse_gradient(err, bpdata, net_dropout, grad) == false){
			std::cout << "gradient failed (3)." << std::endl;
			assert(0);
		      }
		    }
		    else{
		      if(nnet.mse_gradient(err, bpdata, grad) == false){
			std::cout << "gradient failed (4)." << std::endl;
			assert(0);
		      }
		    }
		    
		    // sgrad += ninv*grad;
		    sgrad += grad;
		  }
		  
#pragma omp critical
		  {
		    sumgrad += sgrad;
		  }
		  
		}
		
		sumgrad *= ninv;
	      }
	      
	      
	      
	      {
		char buffer[256];
		double tmp = 0.0;
		whiteice::math::convert(tmp, sumgrad.norm());
		
		snprintf(buffer, 256, "NNGradDescent: %d/%d gradient norm: %f", iterations, MAXITERS, tmp);
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
	      
	      if(nn->exportdata(weights) == false)
		std::cout << "export failed." << std::endl;
	      
	      w0 = weights;
	      
	      // sumgrad.normalize();
	      
	      // adds regularizer to gradient (1/2*||w||^2)
	      
	      sumgrad += regularizer*w0;
	      
	      
	      // restarts gradient descent from lrate = 0.50
	      
	      lrate = sqrt(lrate);
	      lrate *= T(100.0);
	      
	      if(lrate <= T(1e-3)) // now very small learning rates
		lrate = T(1e-3);
	      
	      // lrate = 0.01f;
	      
	      
	      // line search: (we should maybe increase lrate to both directions lrate_next = 2.0*lrate and lrate_next2 = 0.5*lrate...
	      do{
		if(use_SGD){ // just single step towards gradient
		  weights = w0;
		  weights -= sgd_lrate * sumgrad;
		  
		  nn->importdata(weights);
		  
		  break; // don't do linesearch..
		}
		
		weights = w0;
		weights -= lrate * sumgrad;
		
		if(nn->importdata(weights) == false)
		  std::cout << "import failed." << std::endl;
		
		if(heuristics){
		  //normalize_weights_to_unity(*nn);
#if 0
		  if(whiten1d_nnetwork(*nn, dtrain) == false)
		    printf("ERROR: whiten1d_nnetwork failed\n");
		  
		  // using negative feedback heuristic 
		  T alpha = T(0.5f); // lrate;
		  negative_feedback_between_neurons(*nn, dtrain, alpha);
#endif
		}
		
		error = getError(*nn, dtrain, (real(regularizer)>real(T(0.0f))), dropout);
		
		delta_error = (prev_error - error); // positive value means error has decreased
		ratio = real(delta_error) / real(error);
		
		if(real(delta_error) < real(T(0.0f))){ // if error grows we reduce learning rate
		  lrate *= T(0.50);
		}
		else if(real(delta_error) > real(T(0.0))){ // error becomes smaller we increase learning rate
		  lrate *= T(1.0/0.50);
		}
		
		// break;
		
		{
		  char buffer[256];
		  double tmp1, tmp2, tmp3, tmp4;
		  whiteice::math::convert(tmp1, error);
		  whiteice::math::convert(tmp2, delta_error);
		  whiteice::math::convert(tmp4, ratio);
		  whiteice::math::convert(tmp3, lrate);
		  
		  std::ostringstream ss;
		  ss << std::this_thread::get_id();
		  std::string str_id = ss.str();
		  
		  snprintf(buffer, 256,
			   "NNGradDescent: %d/%d (%s) linesearch error: %f delta-error: %f ratio: %f lrate: %e",
			   iterations, MAXITERS,
			   str_id.c_str(),
			   tmp1, tmp2, tmp4, tmp3);
		  whiteice::logging.info(buffer);
		}
		
#if 0
		// leaky error reduction, we sometimes allow jump to worse
		// position in gradient direction
		if((rng.rand() % 100) == 0 && real(error) < real(T(1.00f))){ // was 0.50f
		  logging.info("NNGradDescent: random early stopping of linesearch");
		  break;
		}
#endif
	      }
	      while(real(delta_error) <= T(0.0) && real(lrate) >= real(T(10e-25)) && running);
	      
	      
	      // replaces error with TESTing set error
	      error = getError(*nn, dtest, (real(regularizer)>real(T(0.0f))), dropout);
	      
	      if(use_SGD == false){
		
		if(real(error) >= real(local_thread_best_error)){
		  // no improvement for this iteration
		  std::lock_guard<std::mutex> lock(noimprove_lock);
		  
		  noimprovements[std::this_thread::get_id()]++;
		}
		else{ // improvement in this iteration
		  // resets no improvement counter for this thread
		  local_thread_best_error = error;
		  
		  std::lock_guard<std::mutex> lock(noimprove_lock);
		  
		  noimprovements[std::this_thread::get_id()] = 0;
		}
	      }
	      
	      
	      if(use_SGD == false){
		char buffer[256];
		double tmp1, tmp2, tmp3, tmp4;
		whiteice::math::convert(tmp1, error);
		whiteice::math::convert(tmp2, delta_error);
		whiteice::math::convert(tmp4, ratio);
		whiteice::math::convert(tmp3, lrate);
		
		std::ostringstream ss;
		ss << std::this_thread::get_id();
		std::string str_id = ss.str();
		
		snprintf(buffer, 256,
			 "NNGradDescent: %d/%d (%s) linesearch STOP noimprove counter: %d error: %f delta-error: %f ratio: %f lrate: %e",
			 iterations, MAXITERS,
			 str_id.c_str(),
			 noimprovements[std::this_thread::get_id()],
			 tmp1, tmp2, tmp4, tmp3);
		whiteice::logging.info(buffer);
	      }
	      
	      
	      //nn->exportdata(weights);
	      w0 = weights;
	      
	      if(use_SGD == false){
		if(real(error) < real(best_error)){
		  std::lock_guard<std::mutex> lock(solution_lock);
		  
		  if(dropout){
		    auto nn_without_dropout = *nn;
		    nn_without_dropout.removeDropOut();
		    
		    const T gerror = getError(nn_without_dropout, data,
					      false, false);
		    
		    if(real(gerror) < real(best_pure_error)){
		      nn_without_dropout.exportdata(bestx);
		      best_error = error;
		      best_pure_error = gerror;
		    }
		  }
		  else{
		    const T gerror = getError(*nn, data, false, false);
		    
		    if(real(gerror) < real(best_pure_error)){
		      nn->exportdata(bestx);
		      best_error = error;
		      best_pure_error = gerror;
		    }
		  }
		}
	      }
	      else{
		const T gerror = getError(*nn, data, false, false);
		
		std::lock_guard<std::mutex> lock(solution_lock);
		
		if(real(gerror) < real(best_pure_error)){
		  nn->exportdata(bestx);
		  best_error = error;
		  best_pure_error = gerror;
		}
	      }
	      
	      
	      // TODO: PUSH ALWAYS BEST ERROR TO SEE CONVERGENCE OF THREAD
	      // ONLY TRUE IMPROVEMENTS ALLOW KEEPING ITERATING
	      {
		std::lock_guard<std::mutex> lock(errors_lock);
		
		auto& e = errors[std::this_thread::get_id()];
		e.push_back(error);
		while(e.size() > EHISTORY) e.pop_front();
	      }
	      
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
	      
	      
	    }
	    while(real(error) > real(T(0.00001f)) &&
		  noimprovements[std::this_thread::get_id()] < MAX_NOIMPROVEMENT_ITERS &&
		  iterations < MAXITERS &&
		running);
	    
	    {
	      // marks first convergence by this thread
	      // (restart from random point)
	      std::lock_guard<std::mutex> lock(convergence_lock);
	      
	      convergence[std::this_thread::get_id()] = true;
	    }
	    
	    
	    // 3. after convergence checks if the result is better
	    //    than the earlier one
	    {
	      solution_lock.lock();
	      
	      if(real(error) < real(best_error)){
		// improvement (smaller error with early stopping)
		
		if(dropout){
		  auto nn_without_dropout = *nn;
		  nn_without_dropout.removeDropOut();
		  
		  const T gerror = getError(nn_without_dropout, data,
					    false, false);
		  
		  if(real(gerror) < real(best_pure_error)){
		    nn_without_dropout.exportdata(bestx);
		    best_error = error;
		    best_pure_error = gerror;
		  }
		}
		else{
		  const T gerror = getError(*nn, data, false, false);
		  
		  if(real(gerror) < real(best_pure_error)){
		    nn->exportdata(bestx);
		    best_error = error;
		    best_pure_error = gerror;
		  }
		}
		
	      }
	      
	      solution_lock.unlock();
	    }
	  }
	  
	  
	}

      }
      
      }
      catch(std::exception& e){
	std::cout << "NNGradDescent::optimizer_loop(). Unexpected exception: " << e.what() << std::endl;
	
      }

      
      std::lock_guard<std::mutex> lock(thread_is_running_mutex);
      thread_is_running--;
      thread_is_running_cond.notify_all();
      
      return;
    }


    
    template class NNGradDescent< blas_real<float> >;
    template class NNGradDescent< blas_real<double> >;
    
    template class NNGradDescent< blas_complex<float> >;
    template class NNGradDescent< blas_complex<double> >;

    template class NNGradDescent< superresolution<blas_complex<float>,
						  modular<unsigned int> > >;
    template class NNGradDescent< superresolution<blas_complex<double>,
						  modular<unsigned int> > >;
    
  };
};
