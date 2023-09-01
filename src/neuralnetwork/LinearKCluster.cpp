
#include "LinearKCluster.h"


namespace whiteice
{

  template <typename T>
  LinearKCluster<T>::LinearKCluster()
  {
    K = 0;
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
  bool LinearKCluster<T>::startTrain(const unsigned int K,
				     const std::vector< math::vertex<T> >& xdata,
				     const std::vector< math::vertex<T> >& ydata)
  {
    if(K< 1 || xdata.size() == 0 ||  ydata.size() == 0) return false;
    if(xdata.size() != ydata.size()) return false;
    if(K > xdata.size()) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_running) return false;

    {
      std::lock_guard<std::mutex> lock(solution_mutex);
      this->K = 0;
      A.clear();
      b.clear();
      xmean.clear();
      xvariance.clear();
      currentError = T(INFINITY);

      this->K = K;
      this->xdata = xdata;
      this->ydata = ydata;
    }

    thread_running = true;
    this->K = K;
    this->xdata = xdata;
    this->ydata = ydata;

    try{
      if(optimizer_thread){ delete optimizer_thread; optimizer_thread = nullptr; }
      optimizer_thread = new std::thread(std::bind(&LinearKCluster<T>::optimizer_loop, this));
    }
    catch(std::exception& e){
      thread_running = false;
      optimizer_thread = nullptr;
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
  double LinearKCluster<T>::getSolutionError() const
  {
    std::lock_guard<std::mutex> lock(solution_mutex);
    return this->currentError;
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

    if(K == 0 || A.size() == 0) return false;

    double best_distance = INFINITY;
    unsigned int kbest = 0;

    for(unsigned int i=0;i<xmean.size();i++){
      auto delta = xmean[i] - x;
      for(unsigned int j=0;j<delta.size();j++)
	if(xvariance[j] != T(0.0f))
	  delta[j] /= xvariance[j];

      double d = 0.0;
      whiteice::math::convert(d, delta.norm());
      if(d < best_distance){
	best_distance = d;
	kbest = i;
      }
    }

    y = A[kbest]*x + b[kbest];

    return true;
  }

  template <typename T>
  bool LinearKCluster<T>::save(const std::string& filename) const
  {
  }

  template <typename T>
  bool LinearKCluster<T>::load(const std::string& filename)
  {
    
  }

  template <typename T> 
  void LinearKCluster<T>::optimizer_loop()
  {
    if(thread_running == false) return;
    
    //* 0. Assign data points (x,y) randomly to K-Clusters
    
    //* 3. Calculate mean and variance of clusters and reassign datapoints to most probable
    //*    cluster based on mean and variance of each cluster
    //* 4. Goto 1 if there were signficant changes/no convergence

    {
      std::lock_guard<std::mutex> lock(solution_mutex);
      
      A.resize(K);
      b.resize(K);
      xmean.resize(K);
      xvariance.resize(K);

      for(unsigned int k=0;k<K;k++){
	A[k].resize(ydata[0].size(),xdata[0].size());
	b[k].resize(ydata[0].size());
	xmean[k].resize(xdata[0].size());
	xvariance[k].resize(xdata[0].size());

	A[k].zero();
	b[k].zero();
	xmean[k].zero();
	xvariance[k].zero();
      }
    }

    std::vector<unsigned int> datacluster, old_datacluster;
    for(unsigned int i=0;i<xdata.size();i++){
      datacluster.push_back(whiteice::rng.rand() % K); // random assignment
    }

    // local copy of solutions
    std::vector< math::matrix<T> > AA;
    std::vector< math::vertex<T> > bb;
    std::vector< math::vertex<T> > xxmean;
    std::vector< math::vertex<T> > xxvariance;

    AA.resize(K);
    bb.resize(K);
    xxmean.resize(K);
    xxvariance.resize(K);
    

    while(true){
      {
	std::lock_guard<std::mutex> lock(thread_mutex);
	
	if(thread_running == false)
	  break; // out from the loop and finish
      }

      for(unsigned int k=0;k<K;k++){
	//* 1. Train/optimize linear model for points assigned to this cluster

	std::vector< math::vertex<T> > x, y;

	for(unsigned int i=0;i<datacluster.size();i++){
	  if(datacluster[i] == k){
	    x.push_back(xdata[i]);
	    y.push_back(ydata[i]);
	  }
	}

	math::matrix<T> Cxx, Cyx;
	math::vertex<T> mx, my;

	math::mean_covariance_estimate(mx, Cxx, x);
	math::mean_crosscorrelation_estimate(mx, my, Cyx, x, y);

	math::matrix<T> INV;
	T l = T(1e-20);

	do{
	  INV = Cxx;

	  T trace = T(0.0f);
      
	  for(unsigned int i=0;(i<(Cxx.xsize()) && (i<Cxx.ysize()));i++){
	    trace += Cxx(i,i);
	    INV(i,i) += l; // regularizes Cxx (if needed)
	  }
      
	  if(Cxx.xsize() < Cxx.ysize())	  
	    trace /= Cxx.xsize();
	  else
	    trace /= Cxx.ysize();
      
	  l += T(0.1)*trace + -T(2.0f)*l; // keeps "scale" of the matrix same
	}
	while(whiteice::math::symmetric_inverse(INV) == false);

	AA[k] = Cyx*INV;
	bb[k] = my - AA[k]*mx;
      }

      //* 2. Measure error in each cluster model for each datapoint and 
      //*    assign datapoints to the cluster with smallest error.

      datacluster.clear();
      
      for(unsigned int i=0;i<xdata.size();i++){

	unsigned int kbest = 0;
	double bestErrorr = INFINITY;
	
	for(unsigned int k=0;k<K;k++){
	  auto err = (AA[k]*xdata[i] + bb[k] - ydata[i]).norm();

	  double e = INFINITY;
	  whiteice::math::convert(e, err);
	  if(e < bestError){
	    kbest = k;
	    bestError = e;
	  }
	}

	datacluster.push_back(kbest);
      }
      
      //* 3. Calculate mean and variance of clusters and reassign datapoints to most probable
      //*    cluster based on mean and variance of each cluster

      {
	std::vector<unsigned int> counts;
	counts.resize(xxmean.size());
	
	for(unsigned int k=0;k<xxmean.size();k++){
	  xxmean[k].resize(xdata[0].size());
	  xxvariance[k].resize(xdata[0].size());
	  xxmean[k].zero();
	  xxvariance[k].zero();
	  counts[k] = 0;
	}
	
	
	
	for(unsigned int i=0;i<datacluster.size();i++){
	  const unsigned int k = datacluster[i];
	  xxmean[k] += xdata[i];
	  counts[k]++;
	  
	  for(unsigned int j=0;j<xxvariance[k].size();j++)
	    xxvariance[k][j] += xdata[i][j]*xdata[i][j];
	}
	
	for(unsigned int k=0;k<xxmean.size();k++){
	  if(counts[k]){
	    xxmean[k] /= counts[k];
	    xxvariance[k] /= counts[k];

	    for(unsigned int j=0;j<xxvariance[k].size();j++)
	      xxvariance[k][j] =
		whiteice::math::sqrt(whiteice::math::abs(xxvariance[k][j] - xxmean[k][j]*xxmean[k][j]));
	  }
	}
      }

      // reassign datapoints based on cluster mean and variance
      {
	datacluster.clear();

	for(unsigned int i=0;i<xdata.size();i++){

	  unsigned int kbest = 0;
	  double bestError = INFINITY;

	  for(unsigned int k=0;k<K;k++){
	    auto delta = xdata[i] - xxmean[k];

	    for(unsigned int j=0;j<delta.size();j++){
	      if(xxvariance[k][j] != T(0.0f))
		delta[j] = delta[j]/xxvariance[k][j];
	    }

	    auto n = delta.norm();
	    double distance = INFINITY;
	    whitiece::math::convert(distance, n);

	    if(bestError > distance){
	      kbest = k;
	      bestError = distance;
	    }
	    
	  }

	  datacluster.push_back(kbest);
	}
	
      }

      
      // calculate solution error
      double error = 0.0;
      
      {
	for(unsigned int i=0;i<xdata.size();i++){
	  const unsigned int k = datacluster[i];

	  auto delta = AA[k]*xdata[i] + bb[k] - ydata[i];
	  double e = INFINITY;
	  whiteice::math::convert(e, delta.norm());
	  error += e;
	}

	error /= xdata.size();
      }


      //* 4. Goto 1 if there were signficant changes/no convergence 
      {
	{
	  std::lock_guard<std::mutex> lock(solution_mutex);
	  
	  A = AA;
	  b = bb;
	  xmean = xxmean;
	  xvariance = xxvariance;

	  currentError = error;
	}

	if(old_datacluster.size() > 0){
	  
	  unsigned int changes = 0;
	  
	  for(unsigned int i=0;i<datacluster.size();i++){
	    if(datacluster[i] != old_datacluster[i])
	      changes++
	  }

	  if(changes <= datacluster.size()/100){ // only 1% of points change => convergence
	    std::lock_guard<std::mutex> lock(thread_mutex);
	    thread_running = false;
	    continue;
	  }
	}

	old_datacluster = datacluster;
      }
    }


    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      thread_running = false;
    }
  }


  
  template class KMeans< math::blas_real<float> >;
  template class KMeans< math::blas_real<double> >;
  template class KMeans< math::blas_complex<float> >;
  template class KMeans< math::blas_complex<double> >;

  template class KMeans< math::superresolution< math::blas_real<float> > >;
  template class KMeans< math::superresolution< math::blas_real<double> > >;
  template class KMeans< math::superresolution< math::blas_complex<float> > >;
  template class KMeans< math::superresolution< math::blas_complex<double> > >;
  
};

