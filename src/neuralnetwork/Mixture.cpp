
#include "Mixture.h"
#include "nnetwork.h"
#include "rLBFGS_nnetwork.h"
#include "RNG.h"

#include <chrono>


namespace whiteice
{
  
  template <typename T>
  Mixture<T>::Mixture(const unsigned int NUM,
		      const unsigned int depth,
		      const bool overfit_,
		      const bool negfeedback_
		      ) : N(NUM),
			  SIMULATION_DEPTH(depth),
			  overfit(overfit_),
			  negfeedback(negfeedback_)
  {
    // N = number of experts
    std::lock_guard<std::mutex> lock1(solutionMutex);
    std::lock_guard<std::mutex> lock2(threadMutex);
    
    thread_running = false;
    worker_thread  = nullptr;
    global_iterations = 0;
    running_iterations = 0;
    converged = false;
    model = nullptr;

    solutions.resize(N);
    y.resize(N);
    percent.resize(N);
  }

  
  template <typename T>
  Mixture<T>::~Mixture()
  {
    thread_running = false;
    if(worker_thread != nullptr)
      worker_thread->join();

    if(model) delete model;
    model = nullptr;
  }

  //////////////////////////////////////////////////

  template <typename T>
  bool Mixture<T>::minimize(const whiteice::nnetwork<T>& m,
			    const whiteice::dataset<T>& data)
  {
    std::lock_guard<std::mutex> lock(threadMutex);
    
    if(thread_running) return false; // already running
    if(N <= 1) return false; // bad value
    
    // input and output clusters
    if(data.getNumberOfClusters() != 2)
      return false;

    if(data.size(0) != data.size(1))
      return false;

    // dataset must have at least 10*N points (10 points to train model)
    if(data.size(0) <= 10*N)
      return false;

    try{
      thread_running = true;
      converged = false;
      this->data = data;

      this->model = new whiteice::nnetwork<T>(m);

      if(worker_thread != nullptr)
	delete worker_thread; // deletes old thread

      worker_thread =
	new std::thread(&Mixture<T>::optimizer_loop, this);
    }
    catch(std::exception& e){
      if(model) delete model;
      model = nullptr;
      thread_running = false;
      converged = false;
      worker_thread = nullptr;
      return false; // creating worker thread failed.
    }

    return true;
  }
  
  template <typename T>
  bool Mixture<T>::getSolution(std::vector< math::vertex<T> >& x,
			       std::vector< T >& y,
			       std::vector< T >& percent,
			       unsigned int& iterations,
			       unsigned int& changes) const
  {
    std::lock_guard<std::mutex> lock(solutionMutex);
    
    if(global_iterations+running_iterations <= 0)
      return false; // no solution (yet)
    
    x = solutions;
    y = this->y;
    percent = this->percent;
    iterations = global_iterations + running_iterations;
    changes = latestChanges;

    return true;
  }

  template <typename T>
  bool Mixture<T>::getMajoritySolution(math::vertex<T>& x,
				       T& p)
  {
    std::lock_guard<std::mutex> lock(solutionMutex);
    
    if(global_iterations <= 0) return false; // no solution
    if(percent.size() <= 0) return false;

    unsigned int bestIndex = 0;
    auto bestP = percent[0];

    for(unsigned int n=1;n<percent.size();n++){
      if(percent[n] > bestP){
	bestP = percent[n];
	bestIndex = n;
      }
    }

    p = bestP;
    x = solutions[bestIndex];

    return true;
  }

  
  template <typename T>
  bool Mixture<T>::stopComputation()
  {
    std::lock_guard<std::mutex> lock(threadMutex);
    
    if(thread_running == false) return false; // nothing to do
    
    thread_running = false;
    if(worker_thread != nullptr)
      worker_thread->join();    

    worker_thread = nullptr;
    if(model) delete model;
    model = nullptr;

    return true;
  }

  template <typename T>
  bool Mixture<T>::solutionConverged() const 
  {
    return converged;
  }

  template <typename T>
  bool Mixture<T>::isRunning() const 
  {
    return thread_running;
  }

  template <typename T>
  void Mixture<T>::optimizer_loop()
  {
    converged = false;
    std::vector< whiteice::nnetwork<T>* > models(N);
    std::vector<unsigned int> assignments;
    std::vector<unsigned int> counts(N);
    
    for(unsigned int m=0;m<models.size();m++){
      models[m] = new whiteice::nnetwork<T>(*model);
      models[m]->randomize(); // random initial states

      percent[m] = T(1.0) / T(models.size());
    }

    {
      std::lock_guard<std::mutex> lock(solutionMutex);
      solutions.resize(N);
      y.resize(N);
      global_iterations = 0;
      running_iterations = 0;
      
      for(unsigned int m=0;m<models.size();m++){
	assert(models[m]->exportdata(solutions[m]) == true);
	y[m] = T(INFINITY);
      }

      latestChanges = 0;
    }

    assignments.resize(data.size(0));

    bool first_time = true;

    while(thread_running){
      // 0. import current solution to models
      {
	for(unsigned int m=0;m<models.size();m++){
	  assert(models[m]->importdata(solutions[m]) == true);
	  counts[m] = 0;
	}
      }
      
      
      // 1. divide data to N sets
      std::vector< whiteice::dataset<T> > sets(N);
      unsigned int changes = 0;

      for(unsigned int n=0;n<N;n++)
	sets[n].copyAllButData(data);

      // assign each data point according to minimum error
      // (also calculates mean error)
      math::vertex<T> sum_output;
      sum_output.resize(models[0]->output_size());
      T mean_error = T(0.0);
      
      for(unsigned int i=0;i<data.size(0);i++){
	unsigned int index = 0;
	unsigned int nn = 0;
	
	math::vertex<T> input, output;
	input = data.access(0, i);
	models[index]->calculate(input, output);

	sum_output.zero();
	
	if(!whiteice::math::isinf(y[index])){	  
	  sum_output += output;
	  nn++;
	}
	
	auto delta = (data.access(1, i) - output);
	T minError = (delta*delta)[0];
	
	for(unsigned int n=index;n<N;n++){
	  
	  input = data.access(0, i);
	  models[n]->calculate(input, output);

	  if(!whiteice::math::isinf(y[n])){
	    sum_output += output;
	    nn++;
	  }
	    
	  auto delta = (data.access(1, i) - output);
	  auto error = (delta*delta)[0];
	  
	  if(error < minError){
	    error = minError;
	    index = n;
	  }
	  
	}

	sum_output /= T(nn);
	delta = (data.access(1, i) - sum_output);
	mean_error += (delta*delta)[0] / T(data.size(0));

	// random assingments on the first round..
	if(first_time)
	  index = whiteice::rng.rand() % N;

	if(index != assignments[i])
	  changes++;

	// printf("%d ", index); fflush(stdout);
				      
	sets[index].add(0, data.access(0, i));
	sets[index].add(1, data.access(1, i));
	assignments[i] = index;
	counts[index]++;
      }

      // updates percentages
      {
	std::lock_guard<std::mutex> lock(solutionMutex);

	for(unsigned int n=0;n<N;n++)
	  percent[n] = T(counts[n]) / T(data.size(0));
	
      }
      std::cout << "MEAN ERROR " << mean_error << std::endl;
      

      if(changes == 0 && first_time == false){ // convergence
	std::lock_guard<std::mutex> lock(threadMutex);
	converged = true;
	thread_running = false;
	
	for(unsigned int m=0;m<models.size();m++){
	  delete models[m];
	}
	
	if(model) delete model;
	model = nullptr;
	return;
      }
      else{
	latestChanges = changes;
      }
      
      first_time = false;
      
      // 2. optimize each network with non-zero set
      
      std::vector< rLBFGS_nnetwork<T>* > optimizers(N);

      for(unsigned int index=0;index<optimizers.size();index++){
	if(sets[index].size(0) > 0){
	  auto& o = optimizers[index];
	  o = new rLBFGS_nnetwork<T>(*models[index],
				     sets[index],
				     SIMULATION_DEPTH,
				     overfit,
				     negfeedback);
	  math::vertex<T> x0;
	  models[index]->exportdata(x0);
	  
	  if(o->minimize(x0) == false){
	    for(unsigned int i=0;i<index;i++){
	      delete optimizers[i];
	      optimizers[i] = nullptr;
	    }

	    thread_running = false;
	    break; // jump out of the system (internal error)
	  }
	}
      }

      
      // 3. wait until convergence of each optimizer
      {
	bool waiting = true;

	while(waiting){
	  
	  // gets current solutions
	  {
	    unsigned int sumiterations = 0;

	    math::vertex<T> x;
	    T y;
	    unsigned int iterations = 0;

	    for(unsigned int i=0;i<optimizers.size();i++){
	      if(optimizers[i]){
		if(optimizers[i]->getSolution(x, y, iterations) == true){
		  sumiterations += iterations;
		  this->y[i] = y;
		  std::cout << "OPT " << i << " " << y << " : " << iterations << std::endl;
		}
	      }
	    }

	    std::lock_guard<std::mutex> lock(solutionMutex);
	    running_iterations = sumiterations;
	  }
	  
	  unsigned int counter = 0;
	  unsigned int total = 0;

	  for(unsigned int i=0;i<optimizers.size();i++){
	    if(optimizers[i]){
	      if(optimizers[i]->isRunning() == false)
		counter++;
	      total++;
	    }
	  }

	  if(counter == total){
	    waiting = false;
	  }
	  else{
	    // 500ms
	    std::chrono::milliseconds duration(500); 
	    std::this_thread::sleep_for(duration);
	  }

	  if(thread_running == false){

	    for(unsigned int i=0;i<optimizers.size();i++){
	      if(optimizers[i]) delete optimizers[i];
	      optimizers[i] = nullptr;
	    }

	    std::lock_guard<std::mutex> lock2(solutionMutex);
	    running_iterations = 0;
	    
	    break;
	  }
	  
	}	
      }

      // 4. copy model parameters to our solution
      {
	std::lock_guard<std::mutex> lock(solutionMutex);
	running_iterations = 0;

	for(unsigned int i=0;i<optimizers.size();i++){
	  if(optimizers[i]){
	    math::vertex<T> x;
	    T y;
	    unsigned int iterations = 0;
	    
	    if(optimizers[i]->getSolution(x, y, iterations) == true){
	      solutions[i] = x;
	      this->y[i] = y;
	      global_iterations += iterations;
	    }
	    else{
	      {
		std::lock_guard<std::mutex> lock(threadMutex);
		
		thread_running = false;
		
		for(unsigned int i=0;i<optimizers.size();i++){
		  delete optimizers[i];
		  optimizers[i] = nullptr;
		}

		break;
	      }
	    
	    }
	  }
	  else{ // copies new random values to the solution
	    
	    models[i]->randomize(); // random initial states
	    assert(models[i]->exportdata(solutions[i]) == true);
	    y[i] = T(INFINITY);
	    
	  }
	}
	
      }

      // 5. cleanup
      {
	for(unsigned int i=0;i<optimizers.size();i++){
	  if(optimizers[i]){
	    delete optimizers[i];
	    optimizers[i] = nullptr;
	  }
	}
      }
      
    }

    
    for(unsigned int m=0;m<models.size();m++){
      delete models[m];
    }
    
    thread_running = false;
    if(model) delete model;
    model = nullptr;
  }
      


  template class Mixture< whiteice::math::blas_real<float> >;
  template class Mixture< whiteice::math::blas_real<double> >;
  
};
