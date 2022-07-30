

#include "HMC_abstract.h"
#include "RNG.h"
#include <random>
#include <list>



namespace whiteice
{
  template <typename T>
  HMC_abstract<T>::HMC_abstract(bool storeSamples, bool adaptive)
  {
    this->adaptive = adaptive;
    this->storeSamples = storeSamples;
    
    sum_N = 0;
    sum_mean.zero();
    // sum_covariance.zero();
    
    running = false;
    paused = false;
    
    
  }
  
  
  template <typename T>
  HMC_abstract<T>::~HMC_abstract()
  {
    std::lock_guard<std::mutex> lock(start_lock);
    
    if(running){
      running = false;
      
      for(auto t : sampling_thread){
	t->join();
	delete t;
			}
      
      sampling_thread.clear();
    }
    
    running = false;
  }
  
  
  
  template <typename T>
  bool HMC_abstract<T>::startSampler()
  {
    std::lock_guard<std::mutex> lock(start_lock);
    
    if(running)
      return false; // already running
    
    running = true;
    paused = false;
    
    sampling_thread.clear();
    
    try{
      starting_position(q); // initializes starting position before starting the thread here just to be sure..
      std::thread* t = new std::thread(&HMC_abstract<T>::sampler_loop, this);
      sampling_thread.push_back(t);
      
      return true;
    }
    catch(std::exception& e){
      std::cout << "ERROR: unexpected exception: " << e.what() << std::endl;
      running = false;
      paused = false;
      
      for(auto t : sampling_thread){
	t->join();
	delete t;
      }
      
      sampling_thread.clear();
      
      return false;
    }
  }
  
  
  template <typename T>
  bool HMC_abstract<T>::pauseSampler()
  {
    if(!running) return false;
    
    paused = true;
    return true;
  }
  
  
  template <typename T>
  bool HMC_abstract<T>::continueSampler()
  {
    if(!running) return false;
    
    paused = false;
    return true;
  }
  
  
  template <typename T>
  bool HMC_abstract<T>::stopSampler()
  {
    std::lock_guard<std::mutex> lock(start_lock);
    
    if(!running)
      return false;
    
    running = false;
    paused = false;
    
    for(auto t : sampling_thread){
      t->join();
      delete t;
    }
    
    sampling_thread.clear();
    return true;
  }
  
  
  template <typename T>
  unsigned int HMC_abstract<T>::getSamples(std::vector< math::vertex<T> >& samples) const
  {
    std::lock_guard<std::mutex> lock(solution_lock);
    
    samples = this->samples;
    unsigned int N = this->samples.size();
    
    return N;
  }
  
  
  template <typename T>
  unsigned int HMC_abstract<T>::getNumberOfSamples() const
  {
    return samples.size(); // calling size() should be thread-safe... ???
  }
  
  
  template <typename T>
  bool HMC_abstract<T>::getCurrentSample(math::vertex<T>& s)
  {
    std::lock_guard<std::mutex> lock(updating_sample);
    s = q;
    return true;
    
  }
  
  template <typename T>
  bool HMC_abstract<T>::setCurrentSample(const math::vertex<T>& s)
  {
    std::lock_guard<std::mutex> lock(updating_sample);
    q_overwritten = true; // signals the sampler_loop that it should use updated q and not overwrite global q
    q = s;
    return true;
    
  }
  
  template <typename T>
  void HMC_abstract<T>::setUpdated(bool updated)
  {
    std::lock_guard<std::mutex> lock(updating_sample);
    q_updated = updated;
  }
  
  template <typename T>
  bool HMC_abstract<T>::getUpdated() // are there new sample since last call to setUpdated(false) ??
  {
    std::lock_guard<std::mutex> lock(updating_sample);
    return q_updated;
  }
  
  template <typename T>
  math::vertex<T> HMC_abstract<T>::getMean() const
  {
    std::lock_guard<std::mutex> lock(solution_lock);
    
    if(sum_N > 0){
      T inv = T(1.0f)/T(sum_N);
      math::vertex<T> m = inv*sum_mean;
      
      return m;
    }
    else{
      math::vertex<T> m;
      m.zero();
      
      return m;
    }
  }
  
  /*
    template <typename T>
    math::matrix<T> HMC_abstract<T>::getCovariance() const
    {
    pthread_mutex_lock( &solution_lock );
    
    if(sum_N > 0){
    T inv = T(1.0f)/T(sum_N);
    math::vertex<T> m = inv*sum_mean;
      math::matrix<T> C = inv*sum_covariance;
      
      C -= m.outerproduct();
      
      pthread_mutex_unlock( &solution_lock );
      
      return C;
      }
      else{
      math::matrix<T> C;
      C.zero();
      
      pthread_mutex_unlock( &solution_lock );
      
      return C;
      }
      }
  */
  

  template <typename T>
  T HMC_abstract<T>::getMeanError(unsigned int latestN) const
  {
    std::lock_guard<std::mutex> lock(solution_lock);;
    
    if(!latestN) latestN = samples.size();
    if(latestN > samples.size()) latestN = samples.size();
    
    T sumErr = T(0.0f);
    
    for(unsigned int i=samples.size()-latestN;i<samples.size();i++)
      sumErr += U(samples[i]);
    
    if(latestN > 0)
      sumErr /= T((float)latestN);
    
    return sumErr;
  }
  
  
  template <typename T>
  void HMC_abstract<T>::sampler_loop()
  {
    samples.clear();
    sum_mean.zero();
    // sum_covariance.zero();
    sum_N = 0;
    
    // q = location, p = momentum, H(q,p) = hamiltonian
    math::vertex<T> p; // q is global variable
    
    {
      std::lock_guard<std::mutex> lock(updating_sample);
      //starting_position(q); // random starting position q [ALREADY DONE WHEN THE THREAD STARTS!]
      q_updated = true;
      
      p.resize(q.size()); // momentum is initially zero
    }
    
    p.zero();
    
    // epsilon = epsilon0/sqrt(D) in order to keep distance ||x(n+1) - x(n)|| = epsilon0 for all dimensions dim(x) = D
    T epsilon = T(0.01f); ///math::sqrt(q.size()); .. NOT!!
    unsigned int L = 3; // HMC-10 and HMC-3 has been used, was: HMC-20 is probably a overkill, HMC-1 don't work correctly

    // sets epsilon similarly to HMC.pp [NOT!]
    {
      epsilon = T(0.001f);
      epsilon /= whiteice::math::sqrt((float)q.size());
    }
    
    whiteice::RNG<T> rng;
    
    // used to adaptively finetune step length epsilon based on accept rate
    // the aim of the adaptation is to keep accept rate near optimal 70%
    // L is fixed to rather large value 20
    T accept_rate = T(0.0f);
    unsigned int accept_rate_samples = 0;
    
    // heuristics: we don't store any samples until number of accepts
    // has been 5, this is used to wait for epsilon parameter to adjust
    // correctly so that the probability of accept per iteration is reasonable
    // (we don't store rejects during initial epsilon parameter learning)
    unsigned int number_of_accepts = 0;
    const unsigned int EPSILON_LEARNING_ACCEPT_LIMIT = 0; // BUGGY!! was: 5, we set this to zero
    const T MAX_EPSILON = T(1.0f);
    
    const bool use_difference = false;
    
    
    while(running) // keep sampling forever
    {
      math::vertex<T> q; // local copy of q during this iteration
      
      {
	std::lock_guard<std::mutex> lock(updating_sample);
	q = this->q; // reads the global q
	q_overwritten = false; // detect if somebody have changed q during computation
      }

      {
	char buffer[128];
	double value = 0.0f;
	whiteice::math::convert(value, epsilon);
	sprintf(buffer, "HMC_abstract::sampler_loop(). epsilon=%e.", value);
	whiteice::logging.info(buffer);
      }
      
      rng.normal(p); // Normal distribution
      
      math::vertex<T> old_q = q;
      math::vertex<T> current_p = p;
      
      // leap frog algorithm
      {
	p -= T(0.5f) * epsilon * Ugrad(q);
	
	for(unsigned int i=0;i<L;i++){
	  q += epsilon * p;
	  if(i != L-1) p -= epsilon*Ugrad(q);
	}
	
	p -= T(0.5f) * epsilon * Ugrad(q);
	
	p = -p;
      }
      
	
      T deltaU = T(0.0);
	
      if(use_difference == false){
	T current_U  = U(old_q);
	T proposed_U = U(q);
	deltaU = current_U - proposed_U;
      }
      else{
	deltaU = -Udiff(q, old_q);
#if 0
	// we approximate difference: deltaU = U(q) - U(q_old) ~ (q-q_old)*grad(U(q_old))
	auto deltaq = q - old_q;
	auto gU = Ugrad(old_q);
	deltaU = (deltaq*gU)[0];
#endif
      }
      
      T current_K  = T(0.0f);
      T proposed_K = T(0.0f);
      
      for(unsigned int i=0;i<p.size();i++){
	current_K  += T(0.5f)*current_p[i]*current_p[i];
	proposed_K += T(0.5f)*p[i]*p[i];
      }
      
      const T r = rng.uniform();
      const T p_accept = exp(deltaU+current_K-proposed_K);
      // const T p_accept = exp(deltaU); // HACK TO ALLOW MOVES MOSTLY TO BETTER DIRECTION

      std::cout << "p_accept: " << p_accept << std::endl;
      std::cout << "epsilon: " << epsilon << std::endl;
      
      if(r <= p_accept && !whiteice::math::isnan(p_accept))
	{
	  // accept (q)
	  {
	    std::lock_guard<std::mutex> lock(updating_sample);
	    
	    if(q_overwritten == false){
	      this->q = q; // writes the global q
	    }
	    
	    q_updated = true;
	  }
	  std::cout << "ACCEPT" << std::endl;
	  
	  number_of_accepts++;
	  
	  if(number_of_accepts > EPSILON_LEARNING_ACCEPT_LIMIT){
	    std::lock_guard<std::mutex> lock(solution_lock);
	    
	    if(sum_N > 0){
	      sum_mean += q;
	      // sum_covariance += q.outerproduct();
	      sum_N++;
	    }
	    else{
	      sum_mean = q;
	      // sum_covariance = q.outerproduct();
	      sum_N++;
	    }
	    
	    if(storeSamples)
	      samples.push_back(q);

	    whiteice::logging.info("HMC::sampler_loop(). sample ACCEPTED.");
	  }

	  
	  if(adaptive){
	    accept_rate++;
	    accept_rate_samples++;
	  }
	}
      else{
	// reject (keep old_q)
	printf("REJECT\n");
	{
	  std::lock_guard<std::mutex> lock(updating_sample);
	  if(q_overwritten == false){
	    this->q = old_q; // writes the global q
	  }
	  
	  q_updated = true;
	}
	
	
	if(number_of_accepts > EPSILON_LEARNING_ACCEPT_LIMIT){
	  std::lock_guard<std::mutex> lock(solution_lock);
	  
	  if(sum_N > 0){
	    sum_mean += q;
	    // sum_covariance += q.outerproduct();
	    sum_N++;
	  }
	  else{
	    sum_mean = q;
				    // sum_covariance = q.outerproduct();
	    sum_N++;
	  }
	  
	  
	  if(storeSamples)
	    samples.push_back(q);
	}
	
	if(adaptive){
	  // accept_rate;
	  accept_rate_samples++;
	}
      }
      
      
      updating_sample.unlock();
      
      
      if(adaptive){
	// use accept rate to adapt epsilon
	// adapt sampling rate every N iteration (sample)
	if(accept_rate_samples >= 30) // was: 20
	  {
	    accept_rate /= accept_rate_samples;
	    
	    // std::cout << "ACCEPT RATE: " << accept_rate << std::endl;
	    // std::cout << "EPSILON:     " << epsilon << std::endl;

	    {
	      char buffer[128];
	      double value = 0.0f;
	      whiteice::math::convert(value, accept_rate);
	      sprintf(buffer, "HMC_abstract::sampler_loop(). accept_rate=%.2f.", value);
	      whiteice::logging.info(buffer);
	    }

	    // we target to 50% accept rate because it will give
	    // maximum amount of information max H(accept) which hopefully
	    // leads to faster convergence of estimates (to correct value,
	    // the variance of estimates should be higher though).
	    // Also: changing epsilon breaks MCMC sampling jump symmetry
	    // but if we have converged we hopefully are close to 50%
	    // all the time meaning that epsilon don't change that much +
	    // I'm currently using MCMC samplers a bit more like random search
	    // optimizer methods meaning that not reaching the true distribution
	    // is not that serious..
	    
	    if(accept_rate < T(0.651f)){ // 65.1% was shown to be optimal in one research paper
	      epsilon = T(0.6)*epsilon;
	      // std::cout << "NEW SMALLER EPSILON: " << epsilon << std::endl;
	      
	    }
	    else{
	      // important: sampler can diverge because of adaptive epsilon so we FORCE
	      //            epsilon to be small and sampler cannot diverge??
	      
	      auto new_epsilon  = T(1.0/0.6)*epsilon;
	      if(new_epsilon < MAX_EPSILON)
		epsilon = new_epsilon;
	      
	      // std::cout << "NEW LARGER  EPSILON: " << epsilon << std::endl;
	    }
	    
	    accept_rate = T(0.0f);
	    accept_rate_samples = 0;
	  }
      }
      
      // printf("SAMPLES: %d\n", samples.size());
      
      while(paused && running) // pause
	sleep(1);

    }
    
  }
  
  
};


namespace whiteice
{  
  template class HMC_abstract< float >;
  template class HMC_abstract< double >;
  template class HMC_abstract< math::blas_real<float> >;
  template class HMC_abstract< math::blas_real<double> >;    
};

