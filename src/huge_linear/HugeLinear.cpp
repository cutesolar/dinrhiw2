
#include "HugeLinear.h"

#include <chrono>
#include <functional>
#include <set>

#include <math.h>


namespace whiteice
{
  
  HugeLinear::HugeLinear()
  {
    current_solution_mse = INFINITY;
    iterations = 0;

    data = NULL;
    optimizer_thread = NULL;

    running = false;
    converged = false;
  }
  
  HugeLinear::~HugeLinear()
  {
    if(isRunning()) stopOptimize();
  }

  bool HugeLinear::startOptimize(DataSourceInterface* data)
  {
    if(data == NULL) return false;
    if(data->getNumber() < 10) return false;
    if(data->getInputDimension() == 0 || data->getOutputDimension() == 0)
      return false;

    std::lock_guard<std::mutex> lock(start_lock);

    if(running) return false;
    if(optimizer_thread != NULL) return false;

    {
      running = true;
      converged = false;
      iterations = 0;
      current_solution_mse = INFINITY;
      this->data = data;

      thread_has_started = false;

      try{
	optimizer_thread = new std::thread(std::bind(&HugeLinear::optimizer_loop, this));

	unsigned int counter = 0;

	while(thread_has_started == false && counter < 1000){
	  std::this_thread::sleep_for(std::chrono::milliseconds(50));
	  counter++;
	}

	if(counter >= 1000){
	  running = false;
	  delete optimizer_thread; // don't join for a failed thread (safe?)
	  return false;
	}

	return true;
      }
      catch(std::system_error& thread_error){
	running = false;
	optimizer_thread = NULL;
	return false;
      }
    }
    
    return false;
  }
  
  bool HugeLinear::isRunning()
  {
    return (running && (optimizer_thread != NULL));
  }
  
  bool HugeLinear::stopOptimize()
  {
    std::lock_guard<std::mutex> lock(start_lock);

    if(running == false) return false;

    running = false;
    
    if(optimizer_thread){
      optimizer_thread->join();
      delete optimizer_thread;
      optimizer_thread = NULL;
    }
    
    return true;
  }
  
  bool HugeLinear::getSolution(math::matrix< math::blas_real<float> >& A,
			       math::vertex< math::blas_real<float> >& b)
  {
    std::lock_guard<std::mutex> lock(solution_lock);
    
    if(iterations == 0) return false;

    A = this->A;
    b = this->b;
    
    return true;
  }
  
  float HugeLinear::estimateSolutionMSE()
  {
    return current_solution_mse;
  }

  unsigned int HugeLinear::getIterations() // number of iterations computed so far
  {
    return iterations;
  }


  // calculates MSE error for the linear model parameters
  float HugeLinear::getError(const DataSourceInterface* data,
			     const std::vector<unsigned long>& dset,
			     const math::matrix< math::blas_real<float> >& A,
			     const math::vertex< math::blas_real<float> >& b) const
  {
    if(data == NULL) return INFINITY;

    float error = 0.0f;

    math::vertex< math::blas_real<float> > x, y;

    for(unsigned long i=0;i<dset.size();i++){

      if(data->getData(dset[i], x, y) == false){
	return INFINITY;
      }

      auto delta = y - A*x - b;
      
      error += (delta*delta)[0].c[0];
    }

    error /= (float)(data->getNumber());

    return error;
  }
  
  
  void HugeLinear::optimizer_loop()
  {
    if(running && data)
      thread_has_started = true;
    else{
      thread_has_started = false;
      return;
    }

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

    // divides data to training and testing sets [50.000 samples or number of samples from each]
    std::vector<unsigned long> dtrain, dtest;
    // 100.000 samples from dataset
    unsigned long NUMSAMPLES = data->getNumber() > 100000 ? 100000 : data->getNumber(); 

    if(NUMSAMPLES > 10){
      for(unsigned long i=0;i<NUMSAMPLES;i++){
	
	const unsigned long index = (unsigned long)(((unsigned long)rng.rand64()) % data->getNumber());
	const unsigned int r = (rng.rand() & 1);
	
	if(r == 0){
	  dtrain.push_back(index);
	}
	else{
	  dtest.push_back(index);
	}
      
      }
    }
    else{
      // uses all data for low data cases
      for(unsigned long i=0;i<NUMSAMPLES;i++){
	dtrain.push_back(i);
	dtest.push_back(i);
      }
    }

    if(running == false)
      return;
    
    // parameters
    // (FIXME: should use unsigned long as dimension size, size maybe more than 2^32)

    math::matrix< math::blas_real<float> > A;
    math::vertex< math::blas_real<float> > b;
    
    A.resize(data->getOutputDimension(), data->getInputDimension());
    b.resize(data->getOutputDimension());

    // normally distributed variables
    rng.normal(A);
    rng.normal(b);

    if(running == false)
      return;

    const unsigned int MAXITERS = 1000000; // 1.000.000 iterations max
    float curError = INFINITY;

    {
      std::lock_guard<std::mutex> lock(solution_lock);
      
      curError = getError(data, dtest, A, b);
      current_solution_mse = curError;

      this->A = A;
      this->b = b;
    }

    // calculates constant statistics for gradient computation
    
    math::matrix< math::blas_real<float> > yx;
    math::vertex< math::blas_real<float> > mx, my;

    yx.resize(data->getOutputDimension(), data->getInputDimension());
    mx.resize(data->getInputDimension());
    my.resize(data->getOutputDimension());
    
    yx.zero();
    mx.zero();
    my.zero();

    for(unsigned long index=0;index<dtrain.size() && running;index++){
      
      math::vertex< math::blas_real<float> > x, y;

      if(data->getData(dtrain[index], x, y) == false){
	running = false;
	return;
      }
      
      // calculates the y*x^t term (constant during computations)
      for(unsigned int j=0;j<data->getOutputDimension();j++){
	for(unsigned int i=0;i<data->getInputDimension();i++){
	  yx(j,i) += y[j]*x[i];
	}
      }
      
      // calculates mx term (constant during computations)
      mx += x;
      
      // calculates my term (constant during computations)
      my += y;
    }

    
    mx /= math::blas_real<float>(dtrain.size());
    my /= math::blas_real<float>(dtrain.size());
    yx /= math::blas_real<float>(dtrain.size());

    
    {
      char buffer[256];
      
      snprintf(buffer, 256, "HugeLinear: %d/%d starting optimization loop.",
	       iterations, MAXITERS);
      
      whiteice::logging.info(buffer);
    }

    while(running && iterations < MAXITERS){

      // calculates gradient
      math::matrix< math::blas_real<float> > gradA(data->getOutputDimension(),data->getInputDimension());;
      math::vertex< math::blas_real<float> > gradB(data->getOutputDimension());
      
      math::matrix< math::blas_real<float> > Axx;
      
      Axx.resize(data->getOutputDimension(), data->getInputDimension());
      Axx.zero();
      
      for(unsigned long index=0;index<dtrain.size();index++){
	math::vertex< math::blas_real<float> > x, y;
	
	if(data->getData(dtrain[index], x, y) == false){
	  running = false;
	  return;
	}
	
	// calculates Axx term (NOT a constant during computations, must be updated during every iteration)
	auto Ax = A*x;
	
	for(unsigned int j=0;j<data->getOutputDimension();j++){
	  for(unsigned int i=0;i<data->getInputDimension();i++){
	    Axx(j,i) += Ax[j]*x[i];
	  }
	}
      }
      
      Axx /= math::blas_real<float>(dtrain.size());
      
      // gradient
      gradA = Axx - yx;
      
      for(unsigned int j=0;j<data->getOutputDimension();j++){
	for(unsigned int i=0;i<data->getInputDimension();i++){
	  gradA(j,i) += b[j]*mx[i];
	}
      }
      
      gradB = A*mx + b - my;

      
      // gradient line search
      float error = curError;
      math::blas_real<float> lrate = 1.0f; // learning rate
      math::matrix< math::blas_real<float> > AA;
      math::vertex< math::blas_real<float> > BB;

      do{
	lrate *= 0.5f;
	
	AA = A - lrate*gradA; // negative direction to find smaller error
	BB = b - lrate*gradB;

	error = getError(data, dtest, AA, BB);

	//printf("ITER %d LINE SEARCH: lrate=%f error=%f (current best error=%f)\n",
	//       iterations, lrate.c[0], error, curError);
      }
      while(error >= curError && lrate >= 1e-20);

      //printf("ITER %d FOUND STEP: lrate=%f error=%f (current best error=%f)\n",
      //	     iterations, lrate.c[0], error, curError);

      A = AA;
      b = BB;

      {
	char buffer[256];
	
	snprintf(buffer, 256, "HugeLinear: %d/%d current error: %f.",
		 iterations, MAXITERS, error);
	
	whiteice::logging.info(buffer);
      }
      
      if(lrate > 1e-20){
	curError = error;
	
	if(curError < current_solution_mse){

	  std::lock_guard<std::mutex> lock(solution_lock);
	  
	  current_solution_mse = curError;
	  this->A = A;
	  this->b = b;
	}
      }
      else{

	{
	  char buffer[256];
	  
	  snprintf(buffer, 256, "HugeLinear: %d/%d STOP: convergence or cannot improve error anymore.",
		   iterations, MAXITERS);
	  
	  whiteice::logging.info(buffer);
	}
	
	// convergence or cannot improve error anymore
	break;
      }
      
      iterations++;
    }

    running = false;
    converged = true;
  }
  
  
};
