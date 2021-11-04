
#include "RNN_RBM.h"
#include "matrix.h"
#include "vertex.h"
#include "dataset.h"
#include "bayesian_nnetwork.h"

#include <stdio.h>
#include <functional>

namespace whiteice
{

  
  template <typename T>
  RNN_RBM<T>::RNN_RBM(unsigned int dimVisible,
		      unsigned int dimHidden,
		      unsigned int dimRecurrent)
  {
    this->dimVisible = dimVisible;
    this->dimHidden  = dimHidden;
    this->dimRecurrent = dimRecurrent;

    if(dimVisible == 0 || dimHidden == 0 || dimRecurrent == 0)
      throw std::invalid_argument("RNN_RBM ctor: zero dimension not possible");

    this->running = false;
    this->optimization_thread = nullptr;
    this->optimization_threads = 0;
    this->best_error = T(INFINITY);
    this->iterations = 0;
    

    // initializes neural network
    {
      // NN inputs are visible elements v and recurrent elements
      // NN outputs are two RBM bias elements and recurrent elements
      std::vector<unsigned int> arch;
      arch.push_back(dimVisible + dimRecurrent);
      arch.push_back(dimVisible + dimHidden + dimRecurrent);
      arch.push_back(dimVisible + dimHidden + dimRecurrent);

      nn.setArchitecture(arch, whiteice::nnetwork<T>::halfLinear);
      // nn.setArchitecture(arch, whiteice::nnetwork<T>::pureLinear);
      
      nn.randomize();
    }

    // initializes BBRBM
    {      
      rbm.resize(dimVisible, dimHidden);
      
      rbm.initializeWeights();
    }

    // sets synthesization variables
    {
      synthIsInitialized = false;
      vprev.resize(dimVisible);
      rprev.resize(dimRecurrent);
      
      vprev.zero();
      rprev.zero();
    }
    
  }
  

  template <typename T>
  RNN_RBM<T>::RNN_RBM(const whiteice::RNN_RBM<T>& rbm)
  {
    std::lock_guard<std::mutex> lock(rbm.model_mutex);
    
    this->dimVisible = rbm.dimVisible;
    this->dimHidden  = rbm.dimHidden;
    this->dimRecurrent = rbm.dimRecurrent;

    this->nn = rbm.nn;
    this->rbm = rbm.rbm;

    this->synthIsInitialized = rbm.synthIsInitialized;
    this->vprev = rbm.vprev;
    this->rprev = rbm.rprev;

    this->best_error = rbm.best_error;
    this->iterations = rbm.iterations;

    this->running = false;
    this->optimization_threads = 0;
    this->optimization_thread = nullptr;
  }
  
  
  template <typename T>
  RNN_RBM<T>::~RNN_RBM()
  {
    synthIsInitialized = false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    running = false;
    
    if(optimization_thread){
      optimization_thread->join();
      delete optimization_thread;
    }

    optimization_thread = nullptr;
  }
  

  template <typename T>
  RNN_RBM<T>& RNN_RBM<T>::operator=(const whiteice::RNN_RBM<T>& rbm) 
  {
    std::lock_guard<std::mutex> lock1(thread_mutex);

    if(optimization_threads > 0)
      throw std::invalid_argument("RNN_RBM::operator= cannot assign while optimization is running");

    std::lock_guard<std::mutex> lock2(rbm.model_mutex);

    this->dimVisible = rbm.dimVisible;
    this->dimHidden  = rbm.dimHidden;
    this->dimRecurrent = rbm.dimRecurrent;

    this->nn = rbm.nn;
    this->rbm = rbm.rbm;

    this->synthIsInitialized = rbm.synthIsInitialized;
    this->vprev = rbm.vprev;
    this->rprev = rbm.rprev;

    this->best_error = rbm.best_error;
    this->iterations = rbm.iterations;

    this->running = false;
    this->optimization_threads = 0;
    this->optimization_thread = nullptr;

    return (*this);
  }


  template <typename T>
  unsigned int RNN_RBM<T>::getVisibleDimensions() const
  {
    return dimVisible;
  }

  
  template <typename T>
  unsigned int RNN_RBM<T>::getHiddenDimensions() const
  {
    return dimHidden;
  }


  template <typename T>
  unsigned int RNN_RBM<T>::getRecurrentDimensions() const
  {
    return dimRecurrent;
  }


  template <typename T>
  void RNN_RBM<T>::getRNN(whiteice::nnetwork<T>& nn) const
  {
    std::lock_guard<std::mutex> lock(model_mutex);
    nn = this->nn;
  }


  template <typename T>
  void RNN_RBM<T>::getRBM(whiteice::BBRBM<T>& rbm) const
  {
    std::lock_guard<std::mutex> lock(model_mutex);
    rbm = this->rbm;
  }
  
  
  template <typename T>
  bool RNN_RBM<T>::startOptimize(const std::vector< std::vector< whiteice::math::vertex<T> > >& timeseries, bool randomize)
  {
    if(running) return false; // already running

    if(timeseries.size() <= 0) return false;
    if(timeseries[0].size() <= 0) return false;

    for(unsigned int i=0;i<timeseries.size();i++)
      for(unsigned int j=0;j<timeseries[i].size();j++)
	if(timeseries[i][j].size() != dimVisible)
	  return false;
    
    
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    if(running || optimization_threads > 0) return false;

    try{
      
      {
	std::lock_guard<std::mutex> lock(model_mutex);

	if(randomize){
	  nn.randomize();
	  rbm.initializeWeights();
	}

	best_error = this->reconstructionError(rbm, nn, timeseries);

	if(best_error == T(0.0) || best_error == T(INFINITY))
	  return false;
      }

      this->timeseries = timeseries;
      
      iterations = 0;
      running = true;

      std::unique_lock<std::mutex> lock2(optimize_mutex);
      
      optimization_thread = new std::thread(std::bind(&RNN_RBM<T>::optimize_loop, this));

      // do not exit startOptimize() until thread has started
      while(optimization_threads == 0){
	optimization_threads_cond.wait(lock2);
      }
      
    }
    catch(std::exception& e){
      running = false;
      optimization_thread = nullptr;
      
      return false;
    }

    return true;
  }


  template <typename T>
  bool RNN_RBM<T>::getOptimizeError(unsigned int& iterations, T& error)
  {
    std::lock_guard<std::mutex> lock(model_mutex);

    error = best_error;
    iterations = this->iterations;

    return true;
  }

  
  template <typename T>
  bool RNN_RBM<T>::isRunning() // optimization loop is running
  {
    return (running && optimization_threads > 0);
  }

  
  template <typename T>
  bool RNN_RBM<T>::stopOptimize()
  {
    if(running == false) return false; // already (being) stopped

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(running == false || optimization_threads == 0)
      return false; // already stopped

    running = false;

    if(optimization_thread){
      optimization_thread->join();
      delete optimization_thread;
    }

    optimization_thread = nullptr;
    timeseries.clear();
    
    return true;
  }
  
  
  // resets timeseries synthetization parameters
  template <typename T>
  void RNN_RBM<T>::synthStart()
  {
    std::lock_guard<std::mutex> lock(synth_mutex);
    
    vprev.resize(dimVisible);
    rprev.resize(dimRecurrent);
    vprev.zero();
    rprev.zero();

    synthIsInitialized = true;
  }

  
  // synthesizes next timestep by using the model
  template <typename T>
  bool RNN_RBM<T>::synthNext(whiteice::math::vertex<T>& vnext)
  {
    std::lock_guard<std::mutex> lock1(synth_mutex);
    std::lock_guard<std::mutex> lock2(model_mutex);
    
    if(synthIsInitialized == false) return false;
	
    // const unsigned int CDk = 10;
    
    
    whiteice::math::vertex<T> input(dimVisible + dimRecurrent);
    input.write_subvertex(vprev, 0);
    input.write_subvertex(rprev, dimVisible);
    
    whiteice::math::vertex<T> output;
    
    nn.calculate(input, output);
    
    whiteice::math::vertex<T> a(dimVisible), b(dimHidden);
    
    output.subvertex(a, 0, dimVisible);
    output.subvertex(b, dimVisible, dimHidden);

    whiteice::math::vertex<T> vstar(dimVisible);
    
    // uses CD-k to calculate v* (CD-k estimate)
    {
      rbm.setBValue(b);
      rbm.setAValue(a);
      
      rbm.setVisible(vprev);
      rbm.reconstructData(2*CDk);
      
      rbm.getVisible(vstar);
      
      vnext = vstar;
    }

    output.subvertex(rprev, dimVisible + dimHidden, dimRecurrent);

    synthIsInitialized = false;
    
    return true;
  }

  
  // synthesizes N next candidates using the probabilistic model
  template <typename T>
  bool RNN_RBM<T>::synthNext(unsigned int N, std::vector< whiteice::math::vertex<T> >& vnext)
  {
    std::lock_guard<std::mutex> lock1(synth_mutex);
    std::lock_guard<std::mutex> lock2(model_mutex);
    
    if(synthIsInitialized == false) return false;
    
    // const unsigned int CDk = 10;
    
    
    whiteice::math::vertex<T> input(dimVisible + dimRecurrent);
    input.write_subvertex(vprev, 0);
    input.write_subvertex(rprev, dimVisible);
    
    whiteice::math::vertex<T> output;
    
    nn.calculate(input, output);
    
    whiteice::math::vertex<T> a(dimVisible), b(dimHidden);
    
    output.subvertex(a, 0, dimVisible);
    output.subvertex(b, dimVisible, dimHidden);

    whiteice::math::vertex<T> vstar(dimVisible);
    
    // uses CD-k to calculate v* (CD-k estimate)
    {
      rbm.setBValue(b);
      rbm.setAValue(a);

      for(unsigned int i=0;i<N;i++){
	rbm.setVisible(vprev);
	rbm.reconstructData(2*CDk);
	
	rbm.getVisible(vstar);

	vnext.push_back(vstar);
      }
    }

    output.subvertex(rprev, dimVisible + dimHidden, dimRecurrent);

    synthIsInitialized = false;
    
    return true;
  }

  
  // selects given v as the next step in time-series
  // (needed to be called before calling again synthNext())
  template <typename T>
  bool RNN_RBM<T>::synthSetNext(whiteice::math::vertex<T>& v)
  {
    std::lock_guard<std::mutex> lock(synth_mutex);
    
    vprev = v;
    synthIsInitialized = true;

    return true;
  }


  template <typename T>
  bool RNN_RBM<T>::save(const std::string& basefilename) const
  {
    // model cannot change while saving..
    std::lock_guard<std::mutex> lock(model_mutex); 

    
    // saves generic variables
    {
      whiteice::dataset<T> conf;
      whiteice::math::vertex<T> v;

      if(conf.createCluster("dimensions", 1) == false) return false;
      v.resize(1);
      v.zero();

      v[0] = T(this->dimVisible);
      if(conf.add(0, v) == false) return false;

      v[0] = T(this->dimHidden);
      if(conf.add(0, v) == false) return false;

      v[0] = T(this->dimRecurrent);
      if(conf.add(0, v) == false) return false;

      v[0] = T((int)(this->synthIsInitialized));
      if(conf.add(0, v) == false) return false;

      if(conf.createCluster("vprev", this->dimVisible) == false) return false;
      if(conf.add(1, this->vprev) == false) return false;

      if(conf.createCluster("rprev", this->dimRecurrent) == false) return false;
      if(conf.add(2, this->rprev) == false) return false;

      if(conf.save(basefilename) == false) return false;
    }

    // saves model data
    {
      char buffer[256];
      snprintf(buffer, 256, "%s.rnn", basefilename.c_str());

      whiteice::bayesian_nnetwork<T> bnet;
      if(bnet.importNetwork(this->nn) == false) return false;

      if(bnet.save(buffer) == false) return false;

      snprintf(buffer, 256, "%s.bbrbm", basefilename.c_str());

      if(this->rbm.save(buffer) == false) return false;
    }

    return true;
  }


  template <typename T>
  bool RNN_RBM<T>::load(const std::string& basefilename)
  {
    // tries to load RNN_RBM
    std::lock_guard<std::mutex> lock1(thread_mutex);
    std::lock_guard<std::mutex> lock2(optimize_mutex);

    // cannot load while optimization thread is running..
    if(optimization_threads > 0) 
      return false; 

    unsigned int dimVisible;
    unsigned int dimHidden;
    unsigned int dimRecurrent;
    bool synthIsInitialized;

    whiteice::math::vertex<T> vprev;
    whiteice::math::vertex<T> rprev;

    // generic variables
    {
      whiteice::dataset<T> conf;
      whiteice::math::vertex<T> v;

      if(conf.load(basefilename) == false) return false;
      if(conf.getNumberOfClusters() != 3) return false;

      if(conf.size(0) != 4) return false;
      if(conf.dimension(0) != 1) return false;

      v.resize(1);
      v.zero();

      v = conf.access(0, 0); // dimVisible
      if(v[0] < T(0.0)) return false;
      whiteice::math::convert(dimVisible, floor(v[0]));

      v = conf.access(0, 1); // dimHidden
      if(v[0] < T(0.0)) return false;
      whiteice::math::convert(dimHidden, floor(v[0]));
      
      v = conf.access(0, 2); // dimRecurrent
      if(v[0] < T(0.0)) return false;
      whiteice::math::convert(dimRecurrent, floor(v[0]));
      
      v = conf.access(0, 3); // synthIsInitialized
      if(v[0] < T(0.0)) return false;
      int tmp = 0;
      whiteice::math::convert(tmp, floor(v[0]));
      synthIsInitialized = (bool)tmp;

      v.resize(dimVisible); // vprev
      if(conf.dimension(1) != dimVisible) return false;
      v = conf.access(1, 0);
      vprev = v;

      v.resize(dimRecurrent); // vprev
      if(conf.dimension(2) != dimRecurrent) return false;
      v = conf.access(2, 0);
      rprev = v;
    }

    // tries to load model data
    whiteice::nnetwork<T> nn;
    whiteice::BBRBM<T> rbm;
    
    {
      whiteice::bayesian_nnetwork<T> bnet;
      std::vector< whiteice::math::vertex<T> > weights;

      char buffer[256];
      snprintf(buffer, 256, "%s.rnn", basefilename.c_str());

      if(bnet.load(buffer) == false) return false;

      if(bnet.exportSamples(nn, weights, 0) == false) return false;
      if(weights.size() <= 0) return false;

      if(nn.input_size() != dimVisible + dimRecurrent) return false;
      if(nn.output_size() != dimVisible + dimHidden + dimRecurrent) return false;

      if(nn.importdata(weights[0]) == false) return false;

      
      snprintf(buffer, 256, "%s.bbrbm", basefilename.c_str());

      if(rbm.load(buffer) == false) return false;

      if(rbm.getVisibleNodes() != dimVisible) return false;
      if(rbm.getHiddenNodes() != dimHidden) return false;
    }

    // data loaded successfully: sets global variables
    {
      std::lock_guard<std::mutex> lock(model_mutex);
      
      this->dimVisible = dimVisible;
      this->dimHidden  = dimHidden;
      this->dimRecurrent = dimRecurrent;
      
      this->synthIsInitialized = synthIsInitialized;      
      this->vprev = vprev;
      this->rprev = rprev;

      this->nn = nn;
      this->rbm = rbm;

      this->running = false;
      this->optimization_threads = 0;
      this->timeseries.clear();
      this->iterations = 0;
      this->best_error = T(0.0);
    }

    return true;
  }
  

  template <typename T>
  T RNN_RBM<T>::reconstructionError(whiteice::BBRBM<T>& rbm,
				    whiteice::nnetwork<T>& nn, 
				    const std::vector< std::vector< whiteice::math::vertex<T> > >& timeseries) const
  {
    if(timeseries.size() <= 0) return T(0.0);
    if(timeseries[0].size() <= 0) return T(0.0);
    if(timeseries[0][0].size() != dimVisible) return T(INFINITY);

    T error = T(0.0);
    unsigned int counter = 0;

#pragma omp parallel shared(error) shared(counter)
    {
      T e = T(0.0);
      unsigned int c = 0;

      whiteice::BBRBM<T> rbm(this->rbm);
      
#pragma omp for nowait schedule(auto)
      for(unsigned int n=0;n<timeseries.size();n++){
	
	whiteice::math::vertex<T> r(dimRecurrent);
	r.zero();
	
	whiteice::math::vertex<T> v(dimVisible);
	v.zero();
	
	for(unsigned int i=0;i<timeseries[n].size();i++){
	  whiteice::math::vertex<T> input(dimVisible + dimRecurrent);
	  input.write_subvertex(v, 0);
	  input.write_subvertex(r, dimVisible);
	  
	  whiteice::math::vertex<T> output;
	  
	  nn.calculate(input, output);
	  
	  whiteice::math::vertex<T> a(dimVisible), b(dimHidden);
	  
	  output.subvertex(a, 0, dimVisible);
	  output.subvertex(b, dimVisible, dimHidden);
	  
	  v = timeseries[n][i]; // visible element
	  
	  whiteice::math::vertex<T> vstar(dimVisible);
	  
	  // uses CD-k to calculate v* (CD-k estimate)
	  {
	    rbm.setBValue(b);
	    rbm.setAValue(a);
	    
	    rbm.setVisible(v);
	    rbm.reconstructData(2*CDk);
	    
	    rbm.getVisible(vstar);
	  }
	  
	  e += (v - vstar).norm();
	  c++;
	  
	  output.subvertex(r, dimVisible + dimHidden, dimRecurrent);	
	}
      }

#pragma omp critical (rewirwohtrieohcvhgr)
      {
	error += e;
	counter += c;
      }

    }
      
    error /= T(counter);

    return error;
  }


  template <typename T>
  void RNN_RBM<T>::optimize_loop()
  {
    {
      std::unique_lock<std::mutex> lock(optimize_mutex);
      
      optimization_threads++;
      optimization_threads_cond.notify_all();
    }

    // const unsigned int CDk = 1;
    
    bool verbose = false;
    T epsilon = T(0.01); // initial step length

    std::list<T> errors;
    
    whiteice::nnetwork<T> nn;
    whiteice::BBRBM<T> rbm;

    // gets local copy of model
    {
      std::lock_guard<std::mutex> lock(model_mutex);

      nn = this->nn;
      rbm = this->rbm;
    }

    // negative gradient heuristics
    // [we use random samples to form another negative grad not related to CD algo]
    const bool negative_gradient = false;
    const T p_negative = 0.70; // 70% change of being 1 (random noise)
    
    std::vector< std::vector< whiteice::math::vertex<T> > > negativeseries;

    if(negative_gradient){
      for(unsigned int n=0;n<negativeseries.size();n++){
	for(unsigned int t=0;t<negativeseries[n].size();t++){
	  for(unsigned int k=0;k<negativeseries[n][t].size();k++){
	    if(rbm.rng.uniform() <= p_negative){
	      negativeseries[n][t][k] = T(1.0);
	    }
	    else{
	      negativeseries[n][t][k] = T(0.0);
	    }
	  }
	}
      }
    }
    

    
    
    while(running){

      whiteice::math::matrix<T> grad_W(dimHidden, dimVisible);
      whiteice::math::vertex<T> grad_w(nn.exportdatasize());
      unsigned int numgradients = 0;

      grad_W.zero();
      grad_w.zero();


#pragma omp parallel shared(grad_W) shared(grad_w) shared(numgradients)
      {
	whiteice::math::matrix<T> th_grad_W(dimHidden, dimVisible);
	whiteice::math::vertex<T> th_grad_w(nn.exportdatasize());
	unsigned int th_numgradients = 0;

	th_grad_W.zero();
	th_grad_w.zero();

	whiteice::BBRBM<T> rbm(this->rbm);
	
#pragma omp for nowait schedule(auto)
	for(unsigned int n=0;n<timeseries.size();n++){

	  if(running == false) continue;
	  
	  whiteice::math::vertex<T> r(dimRecurrent);
	  r.zero();
	  
	  whiteice::math::vertex<T> v(dimVisible);
	  v.zero();
	  
	  whiteice::math::matrix<T> ugrad(dimVisible + dimHidden + dimRecurrent,
					  nn.exportdatasize());
	  ugrad.zero();
	  
	  for(unsigned int i=0;i<timeseries[n].size() && running;i++){
	    whiteice::math::vertex<T> input(dimVisible + dimRecurrent);
	    input.write_subvertex(v, 0);
	    input.write_subvertex(r, dimVisible);
	    
	    whiteice::math::vertex<T> output;
	    
	    nn.calculate(input, output);
	    
	    whiteice::math::vertex<T> a(dimVisible), b(dimHidden);
	    
	    output.subvertex(a, 0, dimVisible);
	    output.subvertex(b, dimVisible, dimHidden);
	    
	    v = timeseries[n][i]; // visible element
	    
	    whiteice::math::vertex<T> vstar(dimVisible), hstar(dimHidden);
	    whiteice::math::vertex<T> h(dimHidden);
	    
	    // uses CD-k to calculate v* and h* (CD-k estimates) and h response to v
	    {
	      rbm.setBValue(b);
	      rbm.setAValue(a);

	      rbm.getHiddenResponseField(v, h); // v->h
	      // rbm.setVisible(v);
	      // rbm.reconstructData(1);
	      // rbm.getHidden(h);
	      

	      rbm.setVisible(v);
	      rbm.reconstructData(2*CDk);
	      
	      rbm.getVisible(vstar);
	      
	      // rbm.getHidden(hstar);
	      rbm.getHiddenResponseField(vstar, hstar);
	    }
	    
	    
	    // calculates error gradients of recurrent neural network
	    {
	      // du(n)/dw = df/dw + df/dr * Gr * du(n-1)/dw, Gr matrix selects r
	      
	      whiteice::math::matrix<T> fgrad_w;
	      
	      nn.jacobian(input, fgrad_w);
	      
	      whiteice::math::matrix<T> fgrad_input;
	      
	      nn.gradient_value(input, fgrad_input);
	      
	      whiteice::math::matrix<T> fgrad_r(nn.output_size(), dimRecurrent);
	      
	      fgrad_input.submatrix(fgrad_r,
				    dimVisible+dimHidden, 0,
				    dimRecurrent, nn.output_size());
	      
	      whiteice::math::matrix<T> ugrad_r(dimRecurrent, nn.exportdatasize());
	      
	      ugrad.submatrix(ugrad_r,
			      0, dimVisible+dimHidden,
			      nn.exportdatasize(), dimRecurrent);
	      
	      ugrad = fgrad_w + fgrad_r * ugrad_r;
	    }
	    
	    
	    // calculates gradients of log(probability)
	    {
	      // calculates rbm W weights gradient: h*v^T - E[h*v^T]
	      // TODO optimize computations
	      auto gW = (h.outerproduct(v) - hstar.outerproduct(vstar)); 
	      
	      // calculates RNN weights gradient:
	      // dlog(p)/dw = dlog(p)/da * da/dw + dlog(p)/db * db/dw
	      
	      // da/dw
	      whiteice::math::matrix<T> ugrad_a(dimVisible, nn.exportdatasize());
	      
	      ugrad.submatrix(ugrad_a,
			      0, 0,
			      nn.exportdatasize(), dimVisible);
	      
	      // db/dw
	      whiteice::math::matrix<T> ugrad_b(dimHidden, nn.exportdatasize());
	      
	      ugrad.submatrix(ugrad_b,
			      0, dimVisible,
			      nn.exportdatasize(), dimHidden);
	      
	      auto dlogp_a = v - vstar;
	      auto dlogp_b = h - hstar;
	      
	      auto dlogp_w = dlogp_a * ugrad_a + dlogp_b * ugrad_b;
	      
	      th_grad_W += gW;
	      th_grad_w += dlogp_w;
	      
	      th_numgradients++;
	    }
	    
	    output.subvertex(r, dimVisible + dimHidden, dimRecurrent);	  
	  }
	  
	}

#pragma omp critical (qwefjiovrjvkfpofep)
	{
	  numgradients += th_numgradients;
	  grad_W += th_grad_W;
	  grad_w += th_grad_w;
	}
	
      }

      if(numgradients > 0){
	grad_W /= T(numgradients);
	grad_w /= T(numgradients);
      }


      if(negative_gradient && running){

	// negative gradient for minimizing probability of randomized data
	whiteice::math::matrix<T> n_grad_W(dimHidden, dimVisible);
	whiteice::math::vertex<T> n_grad_w(nn.exportdatasize());
	unsigned int n_numgradients = 0;
	
	n_grad_W.zero();
	n_grad_w.zero();


#pragma omp parallel shared(n_grad_W) shared(n_grad_w) shared(n_numgradients)
	{
	  whiteice::math::matrix<T> th_grad_W(dimHidden, dimVisible);
	  whiteice::math::vertex<T> th_grad_w(nn.exportdatasize());
	  unsigned int th_numgradients = 0;
	  
	  th_grad_W.zero();
	  th_grad_w.zero();

	  whiteice::BBRBM<T> rbm(this->rbm);

	  // calculates normal log probability gradient but changes gradient sign
	  // so we reduce probability of randomly generated time-series

#pragma omp for nowait schedule(auto)
	  for(unsigned int n=0;n<negativeseries.size();n++){

	    if(running == false) continue;
	    
	    whiteice::math::vertex<T> r(dimRecurrent);
	    r.zero();
	    
	    whiteice::math::vertex<T> v(dimVisible);
	    v.zero();
	    
	    whiteice::math::matrix<T> ugrad(dimVisible + dimHidden + dimRecurrent,
					    nn.exportdatasize());
	    ugrad.zero();
	    
	    for(unsigned int i=0;i<negativeseries[n].size() && running;i++){
	      whiteice::math::vertex<T> input(dimVisible + dimRecurrent);
	      input.write_subvertex(v, 0);
	      input.write_subvertex(r, dimVisible);
	      
	      whiteice::math::vertex<T> output;
	      
	      nn.calculate(input, output);
	      
	      whiteice::math::vertex<T> a(dimVisible), b(dimHidden);
	      
	      output.subvertex(a, 0, dimVisible);
	      output.subvertex(b, dimVisible, dimHidden);
	      
	      v = negativeseries[n][i]; // visible element
	      
	      whiteice::math::vertex<T> vstar(dimVisible), hstar(dimHidden);
	      whiteice::math::vertex<T> h(dimHidden);
	      
	      // uses CD-k to calculate v* and h* (CD-k estimates) and h response to v
	      {
		rbm.setBValue(b);
		rbm.setAValue(a);
		
		// rbm.setVisible(v);
		// rbm.reconstructData(1);
		// rbm.getHidden(h);
		rbm.getHiddenResponseField(v, h); // v->h
		
		rbm.setVisible(v);
		rbm.reconstructData(2*CDk);
		
		rbm.getVisible(vstar);
		// rbm.getHidden(hstar);
		rbm.getHiddenResponseField(vstar, hstar); // v->h
	      }
	      
	      
	      // calculates error gradients of recurrent neural network
	      {
		// du(n)/dw = df/dw + df/dr * Gr * du(n-1)/dw, Gr matrix selects r
		
		whiteice::math::matrix<T> fgrad_w;
		
		nn.jacobian(input, fgrad_w);
		
		whiteice::math::matrix<T> fgrad_input;
		
		nn.gradient_value(input, fgrad_input);
		
		whiteice::math::matrix<T> fgrad_r(nn.output_size(), dimRecurrent);
		
		fgrad_input.submatrix(fgrad_r,
				      dimVisible+dimHidden, 0,
				      dimRecurrent, nn.output_size());
		
		whiteice::math::matrix<T> ugrad_r(dimRecurrent, nn.exportdatasize());
		
		ugrad.submatrix(ugrad_r,
				0, dimVisible+dimHidden,
				nn.exportdatasize(), dimRecurrent);
		
		ugrad = fgrad_w + fgrad_r * ugrad_r;
	      }
	      
	      
	      // calculates gradients of log(probability)
	      {
		// calculates rbm W weights gradient: h*v^T - E[h*v^T]
		// TODO optimize computations
		auto gW = (h.outerproduct(v) - hstar.outerproduct(vstar)); 
		
		// calculates RNN weights gradient:
		// dlog(p)/dw = dlog(p)/da * da/dw + dlog(p)/db * db/dw
		
		// da/dw
		whiteice::math::matrix<T> ugrad_a(dimVisible, nn.exportdatasize());
		
		ugrad.submatrix(ugrad_a,
				0, 0,
				nn.exportdatasize(), dimVisible);
		
		// db/dw
		whiteice::math::matrix<T> ugrad_b(dimHidden, nn.exportdatasize());
		
		ugrad.submatrix(ugrad_b,
				0, dimVisible,
				nn.exportdatasize(), dimHidden);
		
		auto dlogp_a = v - vstar;
		auto dlogp_b = h - hstar;
		
		auto dlogp_w = dlogp_a * ugrad_a + dlogp_b * ugrad_b;
		
		th_grad_W += gW;
		th_grad_w += dlogp_w;
		
		th_numgradients++;
	      }
	      
	      output.subvertex(r, dimVisible + dimHidden, dimRecurrent);	  
	    }
	  
	  }

#pragma omp critical (foewpoefwoaaaoneiove)
	  {
	    n_numgradients += th_numgradients;
	    n_grad_W += th_grad_W;
	    n_grad_w += th_grad_w;
	  }

	}
	
	
	if(n_numgradients > 0){
	  n_grad_W /= T(n_numgradients);
	  n_grad_w /= T(n_numgradients);
	  
	  // 10% of the gradient is negative gradient (regularizer value)
	  // 10% don't work (36-16-2), 50% (36-50-10) works but has too
	  //                           little variation (p = 70%)
	  // 30% don'y work (36-16-2) [p = 50%]
	  // 50% (36-16-2) p=50% do not work
	  // 50% (35-16-2) p=70% 
	  
	  grad_W = T(0.5)*(grad_W - n_grad_W);
	  grad_w = T(0.5)*(grad_w - n_grad_w);
	}
      }
      
      

      // after we have calculated gradients through time-series
      // updates parameters can computes reconstruction error and reports it
      if(numgradients > 0 && running)
      {
	T error = T(0.0);

	{
	  whiteice::math::matrix<T> W;
	  whiteice::math::vertex<T> w;

	  whiteice::BBRBM<T> rbm1(rbm), rbm2(rbm), rbm3(rbm);
	  whiteice::nnetwork<T> nn1(nn), nn2(nn), nn3(nn);

	  W = rbm.getWeights();
	  W += epsilon*grad_W;
	  rbm1.setWeights(W);

	  W = rbm.getWeights();
	  W += T(0.90)*epsilon*grad_W;
	  rbm2.setWeights(W);

	  W = rbm.getWeights();
	  W += T(1.0/0.90)*epsilon*grad_W;
	  rbm3.setWeights(W);
	  
	  nn.exportdata(w);
	  w += epsilon*grad_w;
	  nn1.importdata(w);

	  nn.exportdata(w);
	  w += T(0.90)*epsilon*grad_w;
	  nn2.importdata(w);

	  nn.exportdata(w);
	  w += T(1.0/0.9)*epsilon*grad_w;
	  nn3.importdata(w);
	  
	  T error1 = reconstructionError(rbm1, nn1, timeseries);
	  T error2 = reconstructionError(rbm2, nn2, timeseries);
	  T error3 = reconstructionError(rbm3, nn3, timeseries);

	  if(error1 <= error2 && error1 <= error3){
	    error = error1;
	  }
	  else if(error2 <= error1 && error2 <= error3){
	    error = error2;
	    epsilon = T(0.90)*epsilon;
	  }
	  else{ // error3 is the smallest
	    error = error3;
	    epsilon = T(1.0/0.90)*epsilon;
	  }

	  
	  W = rbm.getWeights();
	  W += epsilon*grad_W;
	  rbm.setWeights(W);
	}

	if(verbose){
	  printf("RNN_RBM ITER %d RECONSTRUCTION ERROR: %f EPSILON: %f\n",
		 iterations, error.c[0], epsilon.c[0]);
	  fflush(stdout);
	}
	  
	if(running)
	{
	  std::lock_guard<std::mutex> lock(model_mutex);

	  if(error < best_error){
	    best_error = error;
	    this->rbm = rbm;
	    this->nn  = nn;
	  }
	}

	errors.push_back(error);
	
      }
      else{
	{
	  std::unique_lock<std::mutex> lock(optimize_mutex);
	  
	  optimization_threads--;
	  optimization_threads_cond.notify_all();
	}

	return; // something is wrong (no gradients : no data to calculate)
      }

      
      // checks for stopping criteria
      // (error standard deviation is 1% of mean value)
      {
	while(errors.size() > 20)
	  errors.pop_front();

	if(errors.size() >= 10){
	  T me = T(0.0), se = T(0.0);
	  
	  for(const auto& e : errors){
	    me += abs(e) / T(errors.size());
	    se += e*e / T(errors.size());
	  }

	  se = sqrt(abs(se - me*me)); // st.dev.

	  T ratio = se / (me + T(10e-10));

	  if(verbose){
	    printf("STOPPING CRITERIA RATIO: %f\n", ratio.c[0]);
	    fflush(stdout);
	  }

	  if(ratio <= T(0.0001)*epsilon){

	    std::unique_lock<std::mutex> lock(optimize_mutex);
	    
	    optimization_threads--;
	    optimization_threads_cond.notify_all();

	    return; // stops computation
	  }
	  
	}
      }
      
      iterations++;
    }

    
    {
      std::unique_lock<std::mutex> lock(optimize_mutex);
      
      optimization_threads--;
      optimization_threads_cond.notify_all();
    }
    
  }

  
  template class RNN_RBM< whiteice::math::blas_real<float> >;
  template class RNN_RBM< whiteice::math::blas_real<double> >;
}
