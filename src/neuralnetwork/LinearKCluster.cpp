
#include "LinearKCluster.h"
#include <functional>
#include <math.h>

#include "RNG.h"
#include "correlation.h"
#include "linear_equations.h"
#include "dataset.h"
#include "nnetwork.h"
#include "bayesian_nnetwork.h"


namespace whiteice
{

  template <typename T>
  LinearKCluster<T>::LinearKCluster(const unsigned XSIZE, const unsigned int YSIZE)
  {
    assert(XSIZE >= 1 && YSIZE >= 1); 
    
    K = 0;

    std::vector<unsigned int> arch;
    
    arch.push_back(XSIZE);
    arch.push_back(YSIZE);
    
    architecture.setArchitecture
      (arch, nnetwork<T>::pureLinear);
    
    architecture.randomize();
  }

  template <typename T>
  LinearKCluster<T>::~LinearKCluster()
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    thread_running = false;

    if(optimizer_thread){
      optimizer_thread->join();
      delete optimizer_thread;
      optimizer_thread = nullptr;
    }

    this->K = 0;
  }
  
  template <typename T>
  bool LinearKCluster<T>::startTrain(const std::vector< math::vertex<T> >& xdata,
				     const std::vector< math::vertex<T> >& ydata,
				     const unsigned int K)
  {
    try{
      if(xdata.size() == 0 ||  ydata.size() == 0) return false;
      if(xdata.size() != ydata.size()) return false;
      if(K > xdata.size()) return false;
      
      std::lock_guard<std::mutex> lock(thread_mutex);
      
      if(thread_running) return false;
      
      {
	std::lock_guard<std::mutex> lock(solution_mutex);
	this->K = 0;
	this->model.clear();
	currentError = (double)(INFINITY);
	clusterLabels.clear();
	
	// selects at most 1.000.000 datapoints
	{
	  const unsigned int MAXNUMBER = 1000000;
	  
	  if(xdata.size() <= MAXNUMBER){
	    this->xdata = xdata;
	    this->ydata = ydata;
	  }
	  else{
	    this->xdata.clear();
	    this->ydata.clear();
	    
	    while(this->xdata.size() < MAXNUMBER){
	      const unsigned int index = whiteice::rng.rand() % xdata.size();
	      this->xdata.push_back(xdata[index]);
	      this->ydata.push_back(ydata[index]);
	    }
	  }
	}
	
	
	if(K == 0){
	  this->K = this->xdata.size()/10000;
	  if(this->K < 1) this->K = 1;
	  
	  std::cout << "Using K=" << this->K << " cluster(s) in optimization." << std::endl;
	}
      }
      
      thread_running = true;
      
      try{
	if(optimizer_thread){ delete optimizer_thread; optimizer_thread = nullptr; }
	optimizer_thread = new std::thread(std::bind(&LinearKCluster<T>::optimizer_loop, this));
      }
      catch(std::exception& e){
	thread_running = false;
	optimizer_thread = nullptr;
	return false;
      }
    }
    catch(std::bad_alloc& e){
      thread_running = false;
      optimizer_thread = nullptr;
      this->xdata.clear();
      this->ydata.clear();
      this->K = 0;

      return false;
    }
      
    return true;
    
  }

  template <typename T>
  bool LinearKCluster<T>::isRunning() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    if(thread_running) return true;
    else return false; 
  }

  template <typename T>
  bool LinearKCluster<T>::stopTrain()
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    thread_running = false;
    
    if(optimizer_thread){
      optimizer_thread->join();
      delete optimizer_thread;
      optimizer_thread = nullptr;
    }

    return true;
    
  }

  template <typename T>
  bool LinearKCluster<T>::getSolutionError(unsigned int& iters, double& error) const
  {
    std::lock_guard<std::mutex> lock(solution_mutex);
    iters = this->iterations;
    error = this->currentError;

    return true;
  }

  template <typename T>
  unsigned int LinearKCluster<T>::getNumberOfClusters() const
  {
    std::lock_guard<std::mutex> lock(solution_mutex);
    return this->K;
  }

  template <typename T>
  bool LinearKCluster<T>::predict(const math::vertex<T>& x,
				  math::vertex<T>& y) const
  {
    std::lock_guard<std::mutex> lock(solution_mutex);

    if(K == 0 || model.size() != K) return false;

    double best_distance = INFINITY;
    unsigned int kbest = 0;

#pragma omp parallel
    {
      double best_distance_thread = best_distance;
      unsigned int kbest_thread = kbest;
      
#pragma omp for schedule(auto) nowait
      for(unsigned int i=0;i<xdata.size();i++){
	auto delta = xdata[i] - x;
	
	double d = 0.0;
	whiteice::math::convert(d, whiteice::math::abs(delta.norm())[0]);
	if(d<best_distance_thread){
	  best_distance_thread = d;
	  kbest_thread = i;
	}
      }
      
#pragma omp critical
      {
	if(best_distance > best_distance_thread){
	  best_distance = best_distance_thread;
	  kbest = kbest_thread;
	}
      }
    }

    const unsigned int k = clusterLabels[kbest];

    model[k].calculate(x, y);
    
    return true;
  }

  template <typename T>
  bool LinearKCluster<T>::save(const std::string& filename) const
  {
    if(filename.size() <= 0) return false;

    std::lock_guard<std::mutex> lock(solution_mutex);
    
    if(K <= 0) return false;

    whiteice::dataset<T> data;
    
    if(data.createCluster("K", 1) == false) return false;
    if(data.createCluster("xdata", xdata[0].size()) == false) return false;
    if(data.createCluster("x-cluster-labels", 1) == false) return false;
    if(data.createCluster("model error", 1) == false) return false;

    math::vertex<T> v;

    {
      v.resize(1);
      v[0] = T(this->K);
      if(data.add(0, v) == false) return false;
    }

    {
      v.resize(xdata[0].size());

      for(unsigned int i=0;i<xdata.size();i++){
	v = xdata[i];
	if(data.add(1, v) == false) return false;
      }
    }

    {
      v.resize(1);

      for(unsigned int i=0;i<xdata.size();i++){
	v[0] = T(clusterLabels[i]);
	if(data.add(2, v) == false) return false;
      }
    }
      

    {
      v.resize(1);

      v[0] = T(this->currentError);
      if(data.add(3, v) == false) return false;
    }


    whiteice::bayesian_nnetwork<T> bn;
    
    {
      std::vector< math::vertex<T> > weights;
      
      v.resize(model[0].exportdatasize());

      for(unsigned int i=0;i<model.size();i++){
	model[i].exportdata(v);
	weights.push_back(v);
      }

      if(bn.importSamples(architecture, weights) == false) return false;
    }

    const std::string bnet_filename = filename + ".nnetworks";

    if(bn.save(bnet_filename) == false) return false;
    if(data.save(filename) == false) return false;
    
    return true;
  }
  

  template <typename T>
  bool LinearKCluster<T>::load(const std::string& filename)
  {
    whiteice::dataset<T> data;
    whiteice::bayesian_nnetwork<T> bn;

    const std::string bnet_filename = filename + ".nnetworks";

    if(data.load(filename) == false) return false;
    if(bn.load(bnet_filename) == false) return false;

    if(data.getNumberOfClusters() != 4) return false;
    if(data.dimension(0) != 1) return false;
    if(data.size(0) != 1) return false;
    if(data.dimension(3) != 1) return false;

    unsigned int k = 0; 

    math::vertex<T> v;
    
    v = data.access(0, 0);
    if(v.size() != 1) return false;
    T value;
    double kd;
    whiteice::math::convert(value, v[0]);
    whiteice::math::convert(kd, value);
    k = (unsigned int)kd;

    if(k == 0) return false;

    if(bn.getNumberOfSamples() != k) return false;

    whiteice::nnetwork<T> net;
    std::vector< math::vertex<T> > weights;
    
    {
      if(bn.exportSamples(net, weights) == false)
	return false;

      if(weights.size() != k) return false;

      if(net.input_size() != data.dimension(1)) return false;
    }
    
    const unsigned int xsize = data.dimension(1);
    const unsigned int ysize = net.output_size();
    if(1 != data.dimension(2)) return false;
    
    if(data.size(3) != 1) return false;

    if(xsize*ysize > 2000000000) return false; // sanity check, 2 GB max size

    // looks good, try to load parameters
    {
      std::lock_guard<std::mutex> lock(solution_mutex);

      this->K = k;
      xdata.clear();
      ydata.clear();
      clusterLabels.clear();

      for(unsigned int i=0;i<data.size(1);i++){
	v = data.access(1, i);
	xdata.push_back(v);
	v = data.access(2, i);
	unsigned int k = 0;
	double kd = 0.0;
	whiteice::math::convert(kd, v[0][0]);
	whiteice::math::convert(k, kd);
	
	clusterLabels.push_back(k);
      }

      v = data.access(3, 0);
      if(v.size() != 1) return false;
      whiteice::math::convert(currentError, v[0]);

      model.resize(K);
      for(unsigned int k=0;k<K;k++){
	model[k] = net;
	if(model[k].importdata(weights[k]) == false)
	  return false;
      }
      architecture = net;
    }

    return true;
  }


  template <typename T>
  double LinearKCluster<T>::calculateError(const std::vector< math::vertex<T> >& x,
					   const std::vector< math::vertex<T> >& y,
					   const whiteice::nnetwork<T>& model) const
  {
    double error = 0.0;
    
#pragma omp parallel shared(error)
    {
      double ei = 0.0;
      
#pragma omp for schedule(auto)
      for(unsigned int i=0;i<x.size();i++){
	math::vertex<T> delta;
	  
	model.calculate(x[i], delta);
	delta -= y[i];
	
	// keeps only real part of the error
	for(unsigned int n=0;n<delta.size();n++){
	  delta[n] = T(delta[n][0]);
	}
	
	double e = INFINITY;
	whiteice::math::convert(e, whiteice::math::abs(delta.norm()[0]));
	ei += e;
      }
      
#pragma omp critical
      {
	error += ei;
      }
    }
    
    error /= x.size();
    error /= y[0].size();

    return error;
  }
  

  template <typename T> 
  void LinearKCluster<T>::optimizer_loop()
  {
    try{
      if(thread_running == false || K == 0) return;
      
      {
	std::lock_guard<std::mutex> lock(solution_mutex);
      
	model.resize(K);
	
	for(unsigned int k=0;k<model.size();k++){
	  model[k] = this->architecture;
	  model[k].randomize();
	}
	
	for(unsigned int i=0;i<xdata.size();i++)
	clusterLabels.push_back(whiteice::rng.rand()%K);
	
	iterations = 0;
	currentError = INFINITY;
      }
      
      std::vector< unsigned int> datacluster;
      
      for(unsigned int i=0;i<xdata.size();i++){
	datacluster.push_back(rng.rand() % this->K);
      }
      
      // local copy of solutions
      
      std::vector< whiteice::nnetwork<T> > M;
      
      M.resize(K);
      
      {
	std::lock_guard<std::mutex> lock(solution_mutex);
	
	for(unsigned int k=0;k<M.size();k++){
	  M[k] = model[k];
	}
      }
      
      
      // calculate solution error
      double error = 0.0;
      
#pragma omp parallel shared(error)
      {
	double ei = 0.0;
	
#pragma omp for schedule(auto)
	for(unsigned int i=0;i<xdata.size();i++){
	  
	  const unsigned int k = datacluster[i];
	  
	  math::vertex<T> delta;
	  
	  M[k].calculate(xdata[i], delta);
	  delta -= ydata[i];
	  
	  // keeps only real part of the error
	  for(unsigned int n=0;n<delta.size();n++){
	    delta[n] = T(delta[n][0]);
	  }
	  
	  double e = INFINITY;
	  whiteice::math::convert(e, whiteice::math::abs(delta.norm()[0]));
	  ei += e;
	}
	
#pragma omp critical
	{
	  error += ei;
	}
      }
      
      error /= xdata.size();
      error /= ydata[0].size();
      
      if(verbose) std::cout << "INITIAL ERROR: " << error << std::endl;
      
      
      
      // convergence checking code..
      const unsigned int CONV_LIMIT = 30;
      std::vector<double> convergence_errors;
      
      
      
      while(true){
	
	{
	  if(thread_running == false)
	    break; // out from the loop and finish
	}
	
	
#pragma omp parallel for schedule(auto)
	for(unsigned int k=0;k<K;k++){
	  //* 1. Train/optimize linear model for points assigned to this cluster
	  
	  std::vector< math::vertex<T> > x, y;
	  
	  for(unsigned int i=0;i<datacluster.size();i++){
	    if(datacluster[i] == k){
	      x.push_back(xdata[i]);
	      y.push_back(ydata[i]);
	    }
	  }
	  
	  math::vertex<T> sumgrad;
	  sumgrad.resize(M[k].exportdatasize());
	  sumgrad.zero();
	  
	  
	  for(unsigned int i=0;i<x.size();i++){
	    
	    math::vertex<T> err;
	    
	    M[k].calculate(x[i], err);
	    err -= y[i];
	    
	    math::matrix<T> DF, cDF;
	    
	    M[k].jacobian(x[i], DF);
	    DF.conj(); // ADDED..
	    cDF.resize(DF.ysize(),DF.xsize());
	    
	    for(unsigned int j=0;j<DF.size();j++){
	      whiteice::math::convert(cDF[j], DF[j]);
	      cDF[j].fft();
	    }
	    
	    math::vertex<T> ce, cerr;
	    
	    ce.resize(err.size());
	    
	    for(unsigned int j=0;j<err.size();j++){
	      whiteice::math::convert(ce[j], err[j]);
	      ce[j].fft();
	    }
	    
	    cerr.resize(DF.xsize());
	    cerr.zero();
	    
	    for(unsigned int j=0;j<DF.xsize();j++){
	      auto ctmp = ce;
	      for(unsigned int l=0;l<DF.ysize();l++){
		cerr[j] += ctmp[l].circular_convolution(cDF(l,j));
	      }
	    }
	    
	    T one;
	    one.zero();
	    one = T(1.0);
	    one.fft();
	    
	    for(unsigned int j=0;j<cerr.size();j++)
	    cerr[j].circular_convolution(one);
	    
	    err.resize(cerr.size());
	    
	    for(unsigned int j=0;j<err.size();j++){
	      cerr[j].inverse_fft();
	      for(unsigned int l=0;l<err[j].size();l++)
		whiteice::math::convert(err[j][l], cerr[j][l]);
	    }
	    
	    const auto& grad = err;
	    
	    sumgrad += grad;
	  }
	  
	  
	  if(x.size() > 0){

	    sumgrad *= T(1.0/((double)x.size()));

	    auto err0 = calculateError(x, y, M[k]);
	    auto m = M[k];

	    double lrate = 0.01;
	    unsigned int counter = 0;

	    while(counter <= 30){
	      auto grad = sumgrad*T(lrate);
	    
	      math::vertex<T> w;
	      
	      M[k].exportdata(w);
	      
	      w -= grad;
	    
	      m.importdata(w);

	      auto err = calculateError(x, y, m);

	      if(err < err0){
		M[k] = m;
		break;
	      }
	      else{
		lrate /= 2.0;
		counter++;
	      }
	    }

	    // does nothing if counter has reached maximum value
	    
	  }
	  else{
	    M[k].randomize();
	  }

	  x.clear();
	  y.clear();
	  
	}
	
	
	//* 2. Measure error in each cluster model for each datapoint and 
	//*    assign datapoints to the cluster with smallest error.
	
	// datacluster.clear();
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<xdata.size();i++){
	  
	  if((whiteice::rng.rand() & 1)) continue; // only 50% of the points are reassigned..
	  
	  std::vector<double> errors;
	  
	  for(unsigned int k=0;k<K;k++){
	    
	    math::vertex<T> delta;
	    M[k].calculate(xdata[i], delta);
	    delta -= ydata[i];;
	    
	    for(unsigned int i=0;i<delta.size();i++)
	      delta[i] = T(delta[i][0]);
	    
	    auto err = whiteice::math::abs(delta.norm()[0]);
	    
	    double e = INFINITY;
	    whiteice::math::convert(e, err);
	    
	    errors.push_back(e);
	  }

	  double mean_errors = 0.0;
	  
	  for(const auto& e : errors)
	    mean_errors += e;
	  
	  mean_errors /= errors.size();
	  
	  double sump = 0.0;

	  for(unsigned int i=0;i<errors.size();i++){
	    if((errors[i]-mean_errors) < 600.0 && (errors[i]-mean_errors) > -600.0) 
	      errors[i] = whiteice::math::exp(-(errors[i]-mean_errors));
	    else if((errors[i]-mean_errors) < -600.0)
	      errors[i] = whiteice::math::exp(600.0);
	    else
	      errors[i] = whiteice::math::exp(-600.0);
	    
	    sump += errors[i];
	  }
	  
	  for(unsigned int i=0;i<errors.size();i++){
	    errors[i] /= sump;
	  }
	  
	  sump = 0.0;
	  unsigned int bestk = 0;
	  
	  double r = whiteice::rng.uniform().c[0];
	  
	  for(unsigned int i=0;i<errors.size();i++){
	    sump += errors[i];
	    
	    if(r <= sump){ bestk = i; break; }
	    
	    //if(errors[i] > bestp){ bestp = errors[i]; bestk = i; }
	  }
	  
	  datacluster[i] = bestk;
	}
	
	// calculate solution error
	double error = 0.0;
	
#pragma omp parallel shared(error)
	{
	  double ei = 0.0;
	  
#pragma omp for schedule(auto)
	  for(unsigned int i=0;i<xdata.size();i++){
	    
	    const unsigned int k = datacluster[i];

	    math::vertex<T> delta;
	    M[k].calculate(xdata[i], delta);
	    delta -= ydata[i];
	    
	    double e = INFINITY;

	    T sume = T(0.0);
	    
	    for(unsigned int i=0;i<delta.size();i++){
	      sume += whiteice::math::abs(T(delta[i][0]));
	    }
	    
	    whiteice::math::convert(e, sume);
	    ei += e;
	  }
	  
#pragma omp critical
	  {
	    error += ei;
	  }
	}
	
	// error is error per element in vector
	error /= xdata.size(); // number of vectors
	error /= ydata[0].size(); // vector dimensions
	
	
	
	
	//* 4. Goto 1 if there were significant changes/no convergence 
	{
	  if(error <= currentError)
	    {
	      std::lock_guard<std::mutex> lock(solution_mutex);
	      
	      model = M;
	      currentError = error;
	      clusterLabels = datacluster;
	    }
	  
	  
	  iterations++;
	  
	  // converegence detection
	  {
	    convergence_errors.push_back(currentError);
	    
	    while(convergence_errors.size() > CONV_LIMIT){
	      convergence_errors.erase(convergence_errors.begin());
	    }
	    
	    if(convergence_errors.size() >= CONV_LIMIT){
	      double m = 0.0, s = 0.0;
	      
	      for(const auto& c : convergence_errors){
		m += fabs(c);
		s += c*c;
	      }
	      
	      // mean and variance of the data
	      m /= convergence_errors.size();
	      s /= convergence_errors.size();
	      s = fabs(s - m*m);
	      
	      // mean estimator st.dev.
	      s /= convergence_errors.size();
	      s = whiteice::math::sqrt(s); 
	      
	      const double conv = s/(m + 1e-3);
	      
	      // std::cout << "convergence: " << conv << std::endl;

	      if(early_stopping)
		if(conv < 0.0001) break; // mean error is 0.01% of the mean value
	    }
	  }
	  
	}
	
      }
      
      
      {
	if(thread_running){
	  std::lock_guard<std::mutex> lock(thread_mutex);
	  thread_running = false;
	}
      }
    }
    catch(std::bad_alloc& e){
      std::cout << "ERROR: LinearKCluster optimizer out of memory: " << e.what() << "." << std::endl;
      thread_running = false;
      this->K = 0;
      xdata.clear();
      ydata.clear();
      model.clear();
    }
    catch(std::exception& e){
      std::cout << "ERROR: LinearKCluster exception: " << e.what() << "." << std::endl;
      thread_running = false;
      this->K = 0;
      xdata.clear();
      ydata.clear();
      model.clear();
    }
  }



  template class LinearKCluster< math::blas_real<float> >;
  template class LinearKCluster< math::blas_real<double> >;
  template class LinearKCluster< math::blas_complex<float> >;
  template class LinearKCluster< math::blas_complex<double> >;

  template class LinearKCluster< math::superresolution< math::blas_real<float> > >;
  template class LinearKCluster< math::superresolution< math::blas_real<double> > >;
  template class LinearKCluster< math::superresolution< math::blas_complex<float> > >;
  template class LinearKCluster< math::superresolution< math::blas_complex<double> > >;

  
};

