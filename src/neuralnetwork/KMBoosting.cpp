/*
 * K-Means neural network boosting
 * 
 */

#include "KMBoosting.h"
#include "WeightingOptimizer.h"
#include "NNGradDescent.h"

#include <mutex>
#include <unistd.h>
#include <math.h>
#include <functional>


namespace whiteice
{
  
  /*
   * Initialize K-Means neural network boosting algorithm to use 
   * M neural networks and with given architecture arch.
   */
  template <typename T>
  KMBoosting<T>::KMBoosting(const unsigned int M, const std::vector<unsigned int> arch)
  {
    assert(M > 0);
    assert(arch.size() >= 2);

    experts.resize(M);

    for(unsigned int i=0;i<experts.size();i++)
      experts[i].setArchitecture(arch);

    // combiner neural network must be small so it doesn't overfit
    // in practice handling outliers is problem for combiner neural network

    std::vector<unsigned int> combiner_arch;
    combiner_arch.push_back(arch[0]);
    combiner_arch.push_back(50);
    combiner_arch.push_back(50);
    combiner_arch.push_back(1);

    weighting.setArchitecture(combiner_arch);
    weighting.setNonlinearity(weighting.getLayers()-1, whiteice::nnetwork<T>::softmax);

    optimizer_thread = nullptr;
    thread_running = false;

    hasModelFlag = false;
  }


  template <typename T>
  KMBoosting<T>::~KMBoosting()
  {
    // stop optimizer thread if it is running

    this->stopOptimize();

    if(optimizer_thread) delete optimizer_thread;
  }


  template <typename T>
  bool KMBoosting<T>::startOptimize(const whiteice::dataset<T>& data)
  {
    std::lock_guard<std::mutex> lock(thread_mutex);

    if(data.getNumberOfClusters() < 2) return false;
    if(data.dimension(0) != experts[0].input_size()) return false;
    if(data.dimension(1) != experts[0].output_size()) return false;
    if(data.dimension(0) != weighting.input_size()) return false;
    if(data.size(0) != data.size(1)) return false;
    if(data.size(0) < 10*experts.size()) return false;

    if(thread_running || optimizer_thread != nullptr){
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(solution_mutex);

      if(experts.size() < 0) return false;

      for(unsigned int i=0;i<experts.size();i++)
	experts[i].randomize();

      weighting.randomize();

      hasModelFlag = false;
    }

    this->data = data;
    thread_running = true;
    

    try{
      optimizer_thread = new std::thread(std::bind(&KMBoosting<T>::optimizer_loop, this));
    }
    catch(std::exception& e){
      optimizer_thread = nullptr;
      thread_running = false;
    }

    return true;
  }


  template <typename T>
  bool KMBoosting<T>::stopOptimize()
  {
    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_running == false) return false;

    thread_running = false;

    if(optimizer_thread){
      optimizer_thread->join();
      delete optimizer_thread;
      optimizer_thread = nullptr;
    }
    
    return true;
  }

  
  template <typename T>
  bool KMBoosting<T>::isRunning() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_running && optimizer_thread) return true;
    else return false;
  }
  
  /*
   * predict output given input
   */
  template <typename T>
  bool KMBoosting<T>::calculate(const math::vertex<T>& input, math::vertex<T>& output) const
  {
    std::lock_guard<std::mutex> lock(solution_mutex);
    
    if(hasModelFlag == false) return false;

    whiteice::math::vertex<T> w, tmp;

    if(weighting.calculate(input, w) == false) return false;
    if(w.size() != experts.size()) return false;

    output.resize(experts[0].output_size());
    output.zero();

    for(unsigned int i=0;i<experts.size();i++){
      if(experts[i].calculate(input, tmp) == false) return false;

      output += w[i]*tmp;
    }

    return true;
  }
  


  template <typename T>
  void KMBoosting<T>::optimizer_loop()
  {
    if(thread_running == false) return;

    std::vector< whiteice::nnetwork<T> > experts; // M experts
    whiteice::nnetwork<T> weighting; // w(x) weight for each expert
    

    {
      std::lock_guard<std::mutex> lock(solution_mutex);
      
      for(unsigned int i=0;i<experts.size();i++)
	experts[i] = this->experts[i];

      weighting = this->weighting;
    }

    // step 1. assign data randomly among experts
    std::vector< whiteice::dataset<T> > datapoints;

    {
      datapoints.resize(experts.size());

      for(unsigned int i=0;i<datapoints.size();i++){
	datapoints[i].createCluster("input", data.dimension(0));
	datapoints[i].createCluster("output", data.dimension(1));
      }

      bool points_ok = false;

      while(points_ok == false){

	for(unsigned int i=0;i<datapoints.size();i++){
	  datapoints[i].clearAll();
	}

	for(unsigned int i=0;i<data.size(0);i++){
	  unsigned int index = rng.rand() % datapoints.size();

	  datapoints[index].add(0, data.access(0, i));
	  datapoints[index].add(1, data.access(1, i));
	}

	bool ok = true;
	for(unsigned int i=0;i<datapoints.size();i++){
	  if(datapoints[i].size(0) == 0) ok = false;
	}

	if(ok == false) continue;
	else points_ok = true;

	for(unsigned int i=0;i<datapoints.size();i++){
	  datapoints[i].preprocess(0);
	  datapoints[i].preprocess(1);
	}

      }
      
    }

    if(thread_running == false) return;


    // first step, learn data points by assigned them as in K-Means
    // to a neural network k which have the smallest error. After assignment
    // recompute neural network weights. Continue as long as less than 1% of the datapoints
    // change cluster
    while(true){

      for(unsigned int i=0;i<experts.size();i++){

	if(datapoints[i].size(0) == 0){
	  experts[i].randomize();
	  continue;
	}

	whiteice::math::NNGradDescent<T> grad;
	
	grad.setUseMinibatch(true);
	grad.setOverfit(false);
	grad.setNormalizeError(false);
	grad.setRegularizer(T(0.01));

	const unsigned int MAXITERS = 0xFFFFFFFF;
	const bool dropout = false;
	const bool initiallyUseNN = true;

	if(grad.startOptimize(datapoints[i], experts[i],
			      1, MAXITERS, dropout, initiallyUseNN) == false){
	  {
	    std::lock_guard<std::mutex> lock(thread_mutex);
	    thread_running = false;
	  }
	  
	  return;
	}

	while(grad.isRunning()){
	  sleep(1);

	  if(thread_running == false){
	    grad.stopComputation();
	    return;
	  }
	}

	T error;
	unsigned int Nconverged;

	if(grad.getSolution(experts[i], error, Nconverged) == false){
	  {
	    std::lock_guard<std::mutex> lock(thread_mutex);
	    thread_running = false;
	  }

	  return;
	}
	
      }

      // now reassigns datapoints to a neural network which gives the smallest error
      {
	std::vector< whiteice::dataset<T> > clusters;
	clusters.resize(experts.size());

	for(unsigned int i=0;i<clusters.size();i++){
	  clusters[i].createCluster("input", data.dimension(0));
	  clusters[i].createCluster("output", data.dimension(1));
	}

	unsigned int deltas = 0;

	for(unsigned int j=0;j<datapoints.size();j++){

	  if(thread_running == false) return; // stop running thread

	  for(unsigned int i=0;i<datapoints[j].size(0);i++){
	    
	    const auto& input = datapoints[j].access(0,i);
	    const auto& output = datapoints[j].access(1,i);

	    T min_error;
	    unsigned int choice = 0;

	    whiteice::math::vertex<T> tmp;

	    experts[0].calculate(input, tmp);
	    tmp -= output;
	    min_error = (tmp*tmp)[0]; // squared error

	    for(unsigned int k=1;k<experts.size();k++){
	      experts[k].calculate(input, tmp);
	      tmp -= output;
	      T error = (tmp*tmp)[0]; // squared error

	      if(error < min_error){
		min_error = error;
		choice = k;
	      }
	    }

	    if(choice != j) deltas++;

	    clusters[choice].add(0, input);
	    clusters[choice].add(1, output);
	  }
	}

	if(deltas <= data.size(0)/100){
	  break; // only less than 1% of datapoints change cluster => stop optimizing
	}

	if(thread_running == false) return; // stop running thread

	for(unsigned int i=0;i<clusters.size();i++){
	  clusters[i].preprocess(0);
	  clusters[i].preprocess(1);
	}

	if(thread_running == false) return; // stop running thread

	datapoints = clusters;
      }
      
    }

    
    if(thread_running == false) return; // stop running thread


    // now we have experts, next compute a combiner neural network
    {

      class WeightingOptimizer<T> optimizer(data, experts);

      assert(optimizer.startOptimize(weighting) == true);

      while(optimizer.isRunning()){
	sleep(1);

	if(thread_running == false){
	  optimizer.stopOptimize();
	  return;
	}
      }

      optimizer.stopOptimize();

      T error = T(0.0f);

      assert(optimizer.getSolution(weighting, error) == true);
    }
    
    // in the exit copy parameters to global ones
    {
      
      {
	std::lock_guard<std::mutex> lock(solution_mutex);
	
	for(unsigned int i=0;i<experts.size();i++)
	  this->experts[i] = experts[i];
	
	this->weighting = weighting;

	hasModelFlag = true;
      }
      
      
      {
	std::lock_guard<std::mutex> lock(thread_mutex);
	
	thread_running = false;
      }
    }
  }


  //////////////////////////////////////////////////////////////////////

  template class KMBoosting<whiteice::math::blas_real<float> >;
  template class KMBoosting<whiteice::math::blas_real<double> >;
  
  //template class KMBoosting<whiteice::math::blas_complex<float> >;
  //template class KMBoosting<whiteice::math::blas_complex<double> >;
  
};
