
#include "KMeans.h"
#include "conffile.h"
#include "dinrhiw_blas.h"
#include "blade_math.h"

#include <stdio.h>
#include <thread>
#include <mutex>
#include <functional>
#include <set>

#ifndef KMeans_cpp
#define KMeans_cpp


namespace whiteice
{
  
  template <typename T>
  KMeans<T>::KMeans(T learning_rate, bool optimalmode)
  {
    kmeans.clear();
    this->learning_rate = learning_rate;
    this->goodmode = optimalmode;
  }

  template <typename T>
  KMeans<T>::KMeans(const KMeans<T>& model)
  {
    this->goodmode = model.goodmode;
    this->samplesetsize = model.samplesetsize;
    this->learning_rate = model.learning_rate;

    this->kmeans = model.kmeans;
  }
  
  template <typename T>
  KMeans<T>::~KMeans(){  }

  template <typename T>
  KMeans<T>& KMeans<T>::operator=(const KMeans<T>& model)
  {
    this->goodmode = model.goodmode;
    this->samplesetsize = model.samplesetsize;
    this->learning_rate = model.learning_rate;

    this->kmeans = model.kmeans;

    return (*this);
  }


  // selects random points from data as starting cluster locations
  template <typename T>
  void KMeans<T>::randomize(std::vector<std::vector<T> >& kmeans,
			    const std::vector< whiteice::math::vertex<T> >& data) const
  {
    if(kmeans.size() <= 0 || data.size() <= 0) return;

    if(kmeans.size() >= data.size()){
      for(unsigned int i=0;i<kmeans.size();i++){
	kmeans[i].resize(data[0].size());
	for(unsigned int j=0;j<kmeans[i].size();j++)
	  kmeans[i][j] = T(0.0f);
      }
      
      for(unsigned int i=0;i<data.size()&&i<kmeans.size();i++){
	kmeans[i].resize(data[i].size());
	for(unsigned int j=0;j<data[i].size();j++)
	  kmeans[i][j] = data[i][j];
      }
      return;
    }
    
    std::set<unsigned int> selected; // don't pick same element twice
    unsigned int first_one = 0;

    for(unsigned int k=0;k<kmeans.size();k++){
      unsigned int index = 0;

      do{
	index = rng.rand() % data.size();
      }
      while(selected.find(index) != selected.end());

      if(k == 0) first_one = index;

      kmeans[k].resize(data[index].size());

      for(unsigned int i=0;i<kmeans[k].size();i++){
	kmeans[k][i] = data[index][i];
      }

      selected.insert(index);
    }

    selected.clear();
    selected.insert(first_one);

    for(unsigned int k=1;k<kmeans.size();k++){
      kmeans[k].resize(data[0].size());

      unsigned int found = 0;
      T best_distance = T(0.0f);

      for(unsigned int i=0;i<data.size();i++){

	if(selected.find(i) != selected.end())
	  continue; // don't reselect same data point again

	T min_distance = INFINITY;
	unsigned int k_min_distance = 0;

	for(unsigned int l=0;l<k;l++){
	  T d = calc_distance(kmeans[l], data[i]);

	  if(d < min_distance){
	    d = min_distance;
	    k_min_distance = i;
	  }
	}

	if(min_distance > best_distance){
	  best_distance = min_distance;
	  found = k_min_distance;
	}
	
      }

      for(unsigned int i=0;i<data[found].size();i++)
	kmeans[k][i] = data[found][i];

      selected.insert(found);
    }
  }
  
  
  // selects random points from data as starting cluster locations
  template <typename T>
  void KMeans<T>::randomize(std::vector< whiteice::math::vertex<T> >& kmeans,
			    const std::vector< whiteice::math::vertex<T> >& data) const
  {
    if(kmeans.size() <= 0 || data.size() <= 0) return;

    if(kmeans.size() >= data.size()){
      for(unsigned int i=0;i<kmeans.size();i++)
	kmeans[i].zero();
      
      for(unsigned int i=0;i<data.size()&&i<kmeans.size();i++){
	kmeans[i] = data[i];
      }
      return;
    }

    std::set<unsigned int> selected; // don't pick same element twice
    unsigned int first_one = 0;
    
    for(unsigned int k=0;k<kmeans.size();k++){
      unsigned int index = 0;
      
      do{
	index = rng.rand() % data.size();
      }
      while(selected.find(index) != selected.end());

      if(k == 0) first_one = index;

      kmeans[k].resize(data[index].size());

      for(unsigned int i=0;i<kmeans[k].size();i++){
	kmeans[k][i] = data[index][i];
      }

      selected.insert(index);

    }
    
    selected.clear();
    selected.insert(first_one);
    
    for(unsigned int k=1;k<kmeans.size();k++){
      kmeans[k].resize(data[0].size());
      
      unsigned int found = 0;
      T best_distance = T(0.0f);

      for(unsigned int i=0;i<data.size();i++){

	if(selected.find(i) != selected.end())
	  continue; // don't reselect same data point again

	T min_distance = INFINITY;
	unsigned int k_min_distance = 0;

	for(unsigned int l=0;l<k;l++){
	  T d = calc_distance(kmeans[l], data[i]);

	  if(d < min_distance){
	    d = min_distance;
	    k_min_distance = i;
	  }
	}

	if(min_distance > best_distance){
	  best_distance = min_distance;
	  found = k_min_distance;
	}
	
      }

      kmeans[k] = data[found];

      selected.insert(found);
    }
    
  }

  
  template <typename T>
  bool KMeans<T>::means_changed(const std::vector< math::vertex<T> >& means1,
				const std::vector< math::vertex<T> >& means2) const
  {
    if(means1.size() != means2.size())
      return true;

    T error = T(0.0f);

    for(unsigned int k=0;k<means1.size();k++){
      if(means1[k].size() != means2[k].size())
	return true;

      auto delta = means1[k] - means2[k];
      error += delta.norm();
    }

    if(error > T(0.0f)) return true;

    return false;
  }
  
  template <typename T>
  T KMeans<T>::error(const std::vector< std::vector<T> >& data) const
  {
    T e, error = T(0.0);
    
    if(data.size() <= 1000){
      for(unsigned int i=0;i<data.size();i++){
	unsigned int j = 0;
	
	e = calc_distance(kmeans[j], data[i]);
	
	for(unsigned j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[i]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(data.size());
    }
    else{
      for(unsigned int i=0;i<1000;i++){
	unsigned int index = rng.rand() % data.size();
	unsigned int j=0;
	
	e = calc_distance(kmeans[j], data[index]);
	
	for(j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[index]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(1000.0);
    }
    
    return error;
  }
  
  
  template <typename T>
  T KMeans<T>::error(const std::vector< whiteice::math::vertex<T> >& data) const
  {
    T e, error = T(0.0);
    
    if(data.size() <= 1000){
      for(unsigned int i=0;i<data.size();i++){
	unsigned int j = 0;
	
	e = calc_distance(kmeans[j], data[i]);
	
	for(unsigned j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[i]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(data.size());
    }
    else{
      for(unsigned int i=0;i<1000;i++){
	unsigned int index = rng.rand() % data.size();
	unsigned int j=0;
	
	e = calc_distance(kmeans[j], data[index]);
	
	for(unsigned int j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[index]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(1000.0);
    }
    
    return error;
  }
  

  template <typename T>
  T KMeans<T>::error(const std::vector< whiteice::math::vertex<T> >& kmeans,
		     const std::vector< whiteice::math::vertex<T> >& data) const
  {
    T e, error = T(0.0);
    
    if(data.size() <= 1000){
      for(unsigned int i=0;i<data.size();i++){
	unsigned int j = 0;
	
	e = calc_distance(kmeans[j], data[i]);
	
	for(unsigned j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[i]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(data.size());
    }
    else{
      for(unsigned int i=0;i<1000;i++){
	unsigned int index = rng.rand() % data.size();
	unsigned int j=0;
	
	e = calc_distance(kmeans[j], data[index]);
	
	for(unsigned int j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[index]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(1000.0);
    }
    
    return error;
  }
  
  
  template <typename T>
  bool KMeans<T>::learn(const unsigned int k,
			std::vector<std::vector<T> >& data) 
  {
    if(this->startTrain(k, data) == false) return false;

    while(this->isRunning())
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    this->stopTrain();

    return true;
  }
  
  
  
  template <typename T>
  bool KMeans<T>::learn(const unsigned int k, std::vector< whiteice::math::vertex<T> >& data) 
  {
    if(this->startTrain(k, data) == false) return false;

    while(this->isRunning())
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

    this->stopTrain();

    return true;
  }


  /**
   * starts internal thread for computing the results
   */
  template <typename T>
  bool KMeans<T>::startTrain(const unsigned int K,
			     std::vector< whiteice::math::vertex<T> >& data)
  {
    if(K < 1 || data.size() == 0) return false;
    
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    if(thread_running){
      return false; // thread is already running
    }

    {
      std::lock_guard<std::mutex> lock(solution_mutex);

      best_kmeans.resize(K);

      for(unsigned int i=0;i<K;i++){
	best_kmeans[i].resize(data[0].size());
	best_kmeans[i].zero();
      }
	  
      this->best_error = INFINITY;
      
      randomize(best_kmeans, data);
    }

    thread_running = true;
    this->verbose = true;

    if(data.size() <= 50000){ // 50.000 points
      this->data = data;
    }
    else{
      this->data.clear();
      for(unsigned int i=0;i<50000;i++){ // samples 50.000 points from data.
	const unsigned int index = rng.rand() % data.size();
	this->data.push_back(data[index]);
      }
    }


    try{
      if(optimizer_thread){ delete optimizer_thread; optimizer_thread = nullptr; }
      optimizer_thread = new std::thread(std::bind(&KMeans<T>::optimizer_loop, this));
    }
    catch(std::exception& e){
      thread_running = false;
      optimizer_thread = nullptr;
      return false;
    }
    
    return true;
  }

  template <typename T>
  bool KMeans<T>::startTrain(const unsigned int K,
			     std::vector< std::vector<T> >& data)
  {
    if(K <= 0) return false;

    std::vector< whiteice::math::vertex<T> > d;

    d.resize(data.size());
    for(unsigned int j=0;j<d.size();j++){
      d[j].resize(data[j].size());
      for(unsigned int i=0;i<d[j].size();i++)
	d[j][i] = data[j][i];
    }

    return startTrain(K, d);
  }
  
  // returns true if optimizer thread is running
  template <typename T>
  bool KMeans<T>::isRunning()
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    if(thread_running || computing_stopped == false)
      return true;
    else
      return false;

    return false;
  }
  
  // returns current solution error
  template <typename T>
  double KMeans<T>::getSolutionError()
  {
    std::lock_guard<std::mutex> lock(solution_mutex);
    return this->best_error;
  }
  
  // stops training K-means clustering
  template <typename T>
  bool KMeans<T>::stopTrain()
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
  void KMeans<T>::optimizer_loop()
  {
    try{
      computing_stopped = false;
      
      auto best_kmeans_current = best_kmeans;
      auto err = T(INFINITY);
      whiteice::math::convert(best_error, err);

      unsigned int noimprove = 0;

      
      while(thread_running){
	
	std::vector<unsigned int > TOTAL;
	TOTAL.resize(best_kmeans.size());
	auto kmeans_sum  = best_kmeans_current;

	for(unsigned int i=0;i<best_kmeans_current.size();i++){
	  kmeans_sum[i].zero();
	  TOTAL[i] = 0;
	}

	const unsigned int DATASIZE = 
	  (best_kmeans.size()*70) > data.size() ? data.size() : (best_kmeans.size()*70);

#pragma omp parallel
	{
	  auto kmeans_i = best_kmeans_current;
	  std::vector<unsigned int> N;
	  N.resize(kmeans_i.size());
	  
	  for(unsigned int i=0;i<kmeans_i.size();i++){
	    kmeans_i[i].zero();
	    N[i] = 0;
	  }

#pragma omp for nowait schedule(auto)
	  for(unsigned int i=0;i<DATASIZE;i++){ // was: data.size()

	    const unsigned int index = rng.rand() % data.size();
	    
	    unsigned int winner = 0;
	    T error = T(INFINITY);

	    for(unsigned int k=0;k<best_kmeans_current.size();k++){
	      auto delta = best_kmeans_current[k] - data[index];
	      auto e = delta.norm();

	      if(e < error){
		winner = k;
		error = e;
	      }
	    }

	    kmeans_i[winner] += data[index];
	    N[winner]++;
	  }
	  
#pragma omp critical (mfdskfweiporwe)
	  for(unsigned int i=0;i<best_kmeans_current.size();i++){
	    kmeans_sum[i] += kmeans_i[i];
	    TOTAL[i] += N[i];
	  }	  
	}

	for(unsigned int i=0;i<best_kmeans_current.size();i++){
	  if(TOTAL[i] > 0)
	    kmeans_sum[i] /= TOTAL[i];
	}


	// check if error has decreased
	auto err = error(kmeans_sum, data);

	// check if means has changed [DISABLED]
	// bool change = means_changed(kmeans_sum, best_kmeans_current);
	best_kmeans_current = kmeans_sum;

	
	if(err < best_error){
	  std::lock_guard<std::mutex> lock(solution_mutex);
	  
	  whiteice::math::convert(best_error, err);
	  best_kmeans = kmeans_sum;

	  noimprove = 0;
	}
	else{ // checked earlier too that k-means has changed.. [DISABLED]
	  noimprove++;

	  if(noimprove > 10)
	    break;
	}
	
      }

      
      {
	std::lock_guard<std::mutex> lock(solution_mutex);
	
	// this->kmeans = best_kmeans;
	this->kmeans.resize(best_kmeans.size());

	for(unsigned int k=0;k<best_kmeans.size();k++){
	  this->kmeans[k].resize(best_kmeans[k].size());
	  for(unsigned int i=0;i<best_kmeans[k].size();i++)
	    this->kmeans[k][i] = best_kmeans[k][i];
	}
      }
      
      thread_running = false;
      computing_stopped = true;
    }
    catch(std::exception& e){
      thread_running = false;
      computing_stopped = true;
    }
  }
  
  
  template <typename T>
  unsigned int KMeans<T>::size() const 
  {
    return kmeans.size();
  }
  
  
  template <typename T>
  math::vertex<T>& KMeans<T>::operator[](unsigned int index)
  {
    std::lock_guard<std::mutex> lock(solution_mutex);
    return kmeans[index];
  }
  
  
  template <typename T>
  const math::vertex<T>& KMeans<T>::operator[](unsigned int index) const
  {
    std::lock_guard<std::mutex> lock(solution_mutex);
    return kmeans[index];
  }

  template <typename T>
  unsigned int KMeans<T>::getClusterIndex(const whiteice::math::vertex<T>& x) const
    
  {
    if(kmeans.size() <= 0)
      throw std::logic_error("KMeans: No clustering available");

    if(x.size() != kmeans[0].size())
      throw std::logic_error("KMeans::getClusterIndex(): input data dimension mismatch with kmeans clusters."); 

    T best_distance = T(INFINITY);
    unsigned int best_index = 0;

    for(unsigned int i=0;i<kmeans.size();i++){

      T distance = T(0.0);
      
      for(unsigned int j=0;j<kmeans[i].size();j++){
	distance += (x[j] - kmeans[i][j])*(x[j] - kmeans[i][j]);
      }

      if(distance < best_distance){
	best_distance = distance;
	best_index = i;
      }
    }

    return best_index;
  }
  
  template <typename T>
  unsigned int KMeans<T>::getClusterIndex(const std::vector<T>& x) const
    
  {
    if(kmeans.size() <= 0)
      throw std::logic_error("KMeans: No clustering available");

    T best_distance = T(INFINITY);
    unsigned int best_index = 0;

    for(unsigned int i=0;i<kmeans.size();i++){

      T distance = T(0.0);
      
      for(unsigned int j=0;j<kmeans[i].size();j++){
	distance += (x[j] - kmeans[i][j])*(x[j] - kmeans[i][j]);
      }

      if(distance < best_distance){
	best_distance = distance;
	best_index = i;
      }
    }

    return best_index;
  }
  
  // calculates squared distance
  template <typename T>
  T KMeans<T>::calc_distance(const std::vector<T>& u,
			     const std::vector<T>& v) const
  {
    unsigned int i;
    const unsigned int max = u.size();
    T e, error = T(0.0);
    
    for(i=0;i<max;i++){
      e = (u[i] - v[i]);
      error += e*e;
    }
    
    return error;
  }
  
  
  template <typename T>
  T KMeans<T>::calc_distance(const std::vector<T>& u,
			     const whiteice::math::vertex<T>& v) const
  {
    unsigned int i;
    const unsigned int max = u.size();
    T e, error = T(0.0);
    
    for(i=0;i<max;i++){
      e = (u[i] - v[i]);
      error += e*e;
    }
    
    return error;
  }

  
  template <typename T>
  T KMeans<T>::calc_distance(const whiteice::math::vertex<T>& u,
			     const whiteice::math::vertex<T>& v) const
  {
    auto delta = (u - v);
    return delta.norm();
  }
  
  
  template <typename T>
  bool KMeans<T>::save(const std::string& filename) 
  {
    whiteice::conffile configuration;
    std::vector<int> ints;
    std::vector<float> floats;
    std::vector<std::string> strings;
    
    if(kmeans.size() <= 0) return false;
    
    // version number
    ints.clear();
    ints.push_back(1);
    if(!configuration.set("KMEANS_VERSION", ints)) return false;

    const unsigned int dimension = kmeans[0].size();
    
    // number of vectors and their dimensions    
    ints.clear();
    ints.push_back(kmeans.size()); // K
    ints.push_back(dimension); // dimension
    if(!configuration.set("KMEANS_SIZES", ints)) return false;
    
    // sets kmeans vector data
    char buf[100];    
    floats.resize(dimension);
    
    unsigned int counter = 0;
    typename std::vector< math::vertex<T> >::iterator i;
    //typename std::vector<T>::iterator j;
    std::vector<float>::iterator l;
    
    for(i=kmeans.begin();i!=kmeans.end();i++, counter++){
      sprintf(buf, "KMEANS_VECTOR%d", counter);            
      
      l = floats.begin();
      for(unsigned int j=0;j<floats.size();j++,l++)
	math::convert( (*l), (*i)[j] );
      
      if(!configuration.set(buf,floats)){
	return false;
      }
    }
    
    return configuration.save(filename);
  }
  

  // loads k-means clustering data from a file
  template <typename T>
  bool KMeans<T>::load(const std::string& filename) 
  {
    whiteice::conffile configuration;
    std::vector<int> ints;
    std::vector<float> floats;
    std::vector<std::string> strings;
    
    kmeans.clear();
    
    if(!configuration.load(filename))
      return false;

    // version number
    ints.clear();
    if(!configuration.get("KMEANS_VERSION", ints)) return false;
    if(ints.size() != 1) return false;
    
    // version number check
    if(ints[0] != 1) return false;

    
    // number of vectors and their dimensions        
    ints.clear();
    if(!configuration.get("KMEANS_SIZES", ints)) return false;
    if(ints.size() != 2) return false;
    
    kmeans.resize(ints[0]); // K
    for(unsigned int i=0;i<kmeans.size();i++)
      kmeans[i].resize(ints[1]);
    
    // sets kmeans vector data
    char buf[100];    
    floats.clear();
    
    unsigned int counter = 0;
    typename std::vector< math::vertex<T> >::iterator i;
    //typename std::vector<T>::iterator j;
    std::vector<float>::iterator l;
    
    for(i=kmeans.begin();i!=kmeans.end();i++, counter++){
      sprintf(buf, "KMEANS_VECTOR%d", counter);
      
      if(!configuration.get(buf, floats)){	
	return false;
      }
      
      //j = i->begin();
      unsigned int j = 0;
      for(l = floats.begin();l!=floats.end();j++,l++){
	//*j = T(*l);
	(*i)[j] = T(*l);
      }
    }
    
    return true;
  }
  
  
  
  // calculates aprox error from the latter part of data
  template <typename T>    
  T KMeans<T>::second_half_error(const std::vector< std::vector<T> >& data) const
  {
    T e, error = T(0.0);
    
    if(data.size() <= 1000){
      for(unsigned int i=(data.size()/2);i<data.size();i++){
	unsigned int j = 0;
	
	e = calc_distance(kmeans[j], data[i]);
	
	for(unsigned j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[i]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(data.size());
    }
    else{
      for(unsigned int i=0;i<1000;i++){
	unsigned int index = (rng.rand() % (data.size()/2)) + (data.size()/2);
	unsigned int j=0;
	
	e = calc_distance(kmeans[j], data[index]);
	
	for(unsigned int j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[index]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(1000.0);
    }
    
    return error;
  }
  
  
  // calculates aprox error from the latter part of data
  template <typename T>    
  T KMeans<T>::second_half_error(const std::vector< whiteice::math::vertex<T> >& data) const
  {
    T e, error = T(0.0);
    
    if(data.size() <= 1000){
      for(unsigned int i=(data.size()/2);i<data.size();i++){
	unsigned int j = 0;
	
	e = calc_distance(kmeans[j], data[i]);
	
	for(unsigned j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[i]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(data.size());
    }
    else{
      for(unsigned int i=0;i<1000;i++){
	unsigned int index = (rng.rand() % (data.size()/2)) + (data.size()/2);
	unsigned int j=0;
	
	e = calc_distance(kmeans[j], data[index]);
	
	for(unsigned int j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[index]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(1000.0);
    }
    
    return error;
  }

  
  
  
  
  template <typename T>
  T& KMeans<T>::rate() {
    return learning_rate;
  }
  
  
  template <typename T>
  const T& KMeans<T>::rate() const {
    return learning_rate;
  }
  
  
  template <typename T>
  bool& KMeans<T>::getOptimalMode() {
    return goodmode;
  }
  
  
  template <typename T>
  const bool& KMeans<T>::getOptimalMode() const {
    return goodmode;
  }
  
  
  template <typename T>
  unsigned int& KMeans<T>::numSamplingSteps() {
    return samplesetsize;
  }
  
  
  template <typename T>
  const unsigned int& KMeans<T>::numSamplingSteps() const {
    return samplesetsize;
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class KMeans< float >;  
  template class KMeans< double >;  
  template class KMeans< math::blas_real<float> >;
  template class KMeans< math::blas_real<double> >;
  
}
  
#endif



