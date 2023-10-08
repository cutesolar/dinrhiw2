
#include "GlobalOptimizer.h"
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
  GlobalOptimizer<T>::GlobalOptimizer()
  {
    currentError = T(INFINITY);
  }

  template <typename T>
  GlobalOptimizer<T>::~GlobalOptimizer()
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    thread_running = false;

    if(optimizer_thread){
      optimizer_thread->join();
      delete optimizer_thread;
      optimizer_thread = nullptr;
    }
    
  }
  
  template <typename T>
  bool GlobalOptimizer<T>::startTrain(const std::vector< math::vertex<T> >& xdata,
				      const std::vector< math::vertex<T> >& ydata,
				      T levelOfDetailFreq)
  {
    try{
      if(xdata.size() == 0 ||  ydata.size() == 0) return false;
      if(xdata.size() != ydata.size()) return false;
      if(levelOfDetailFreq < 0.0 || levelOfDetailFreq >= 1.0) return false;
      
      std::lock_guard<std::mutex> lock(thread_mutex);
      
      if(thread_running) return false;
      
      {
	std::lock_guard<std::mutex> lock(solution_mutex);
	
	currentError = (double)(INFINITY);
	this->A.resize(1,1);
	this->b.resize(1);
	this->levelOfDetailFreq = levelOfDetailFreq;
	
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
	
      }
      
      thread_running = true;
      
      try{
	if(optimizer_thread){ delete optimizer_thread; optimizer_thread = nullptr; }
	optimizer_thread = new std::thread(std::bind(&GlobalOptimizer<T>::optimizer_loop, this));
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
      
      return false;
    }
      
    return true;
    
  }

  template <typename T>
  bool GlobalOptimizer<T>::isRunning() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    if(thread_running) return true;
    else return false; 
  }

  template <typename T>
  bool GlobalOptimizer<T>::stopTrain()
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
  bool GlobalOptimizer<T>::getSolutionError(double& error) const
  {
    std::lock_guard<std::mutex> lock(solution_mutex);

    whiteice::math::convert(error, this->currentError);

    return true;
  }

  
  template <typename T>
  bool GlobalOptimizer<T>::predict(const math::vertex<T>& x,
				   math::vertex<T>& y) const
  {
    std::lock_guard<std::mutex> lock(solution_mutex);

    if(currentError == T(INFINITY)) return false;
    if(A.xsize() == 1 && A.ysize() == 1) return false;

    // discretize x
    
    math::vertex<T> xdisc;

    std::vector< std::vector<std::string> > sdata;
    std::vector< std::vector<double> > bindata;
    std::vector< std::vector<double> > results;
    
    std::vector<std::string> vec;

    for(unsigned int k=0;k<x.size();k++){
      char buf[80];
      sprintf(buf, "%f", x[k].c[0]);
      vec.push_back(std::string(buf));
    }

    sdata.push_back(vec);

    whiteice::binarize(sdata, this->disc, bindata);
    whiteice::enrich_data_again(bindata, this->f_itemset, results);

    xdisc.resize(results[0].size());

    for(unsigned int i=0;i<xdisc.size();i++)
      xdisc[i] = results[0][i];

    y = A*xdisc + b;
    
    return true;
  }

  template <typename T>
  bool GlobalOptimizer<T>::save(const std::string& filename) const
  {
    std::lock_guard<std::mutex> lock(solution_mutex);

    if(filename.size() == 0) return false;
    if(currentError == T(INFINITY)) return false;
  
    // save disc and f_itemset, + XSIZE, YSIZE here!
    
    whiteice::dataset<T> data;

    data.createCluster("XSIZE", 1);
    data.createCluster("YSIZE", 1);
    data.createCluster("f_itemset", f_itemset.begin()->size());
    data.createCluster("TYPE", 1);
    data.createCluster("bins", 1000);
    data.createCluster("str_elems", 4096);
    data.createCluster("A", this->A.size());
    data.createCluster("b", this->b.size());
    data.createCluster("error", 1);

    math::vertex<T> v;

    {
      v.resize(1);
      v[0] = f_itemset.size();
      if(data.add(0, v) == false) return false;
    }

    {
      v.resize(1);
      v[0] = this->ydata[0].size();
      if(data.add(1, v) == false) return false;
    }

    {
      for(const auto& f : f_itemset){
	v.resize(f.size());
	for(unsigned int i=0;i<f.size();i++){
	  if(f[i]) v[i] = 1.0f;
	  else v[i] = 0.0f;
	}

	if(data.add(2, v) == false) return false;
      }
    }

    {
      v.resize(1);
      for(unsigned int k=0;k<disc.size();k++){
	v[0] = disc[k].TYPE;

	if(data.add(3, v) == false) return false;
      }
    }

    {
      v.resize(1000);
      for(unsigned int i=0;i<disc.size();i++){
	v[0] = disc[i].bins.size();
	for(unsigned int k=1;k<=disc[i].bins.size();k++){
	  v[k] = disc[i].bins[k-1];
	}

	if(data.add(4, v) == false) return false;
      }
    }

    {
      v.resize(4096);
      
      for(unsigned int i=0;i<disc.size();i++){
	v[0] = disc[i].elem.size();
	for(unsigned int k=1;k<=disc[i].elem.size();k++){
	  unsigned int index = 1;
	  for(unsigned int l=0;l<=disc[i].elem[k].size();l++){
	    v[index+l] = disc[i].elem[k][l];
	  }
	  
	  index += disc[i].elem[k].size();
	}

	if(data.add(5, v) == false) return false;
      }
    }

    {
      v.resize(this->A.size());

      for(unsigned int i=0;i<this->A.size();i++){
	v[i] = A[i];
      }

      if(data.add(6, v) == false) return false;
    }

    {
      v.resize(this->b.size());

      for(unsigned int i=0;i<this->b.size();i++){
	v[i] = b[i];
      }

      if(data.add(7, v) == false) return false;
    }

    {
      v.resize(1);

      v[0] = currentError;

      if(data.add(8, v) == false) return false;
    }


    if(data.save(filename) == false) return false;

    return true;
  }
  

  template <typename T>
  bool GlobalOptimizer<T>::load(const std::string& filename)
  {
    std::lock_guard<std::mutex> lock(solution_mutex);

    whiteice::dataset<T> data;

    if(data.load(filename) == false) return false;
    
    if(data.getNumberOfClusters() != 9) return false;
    
    // load disc here! + XSIZE, YSIZE
    unsigned int XSIZE, YSIZE;
    std::set< whiteice::dynamic_bitset > fset;
    std::vector< struct whiteice::discretization > cd;
    math::matrix<T> AA;
    math::vertex<T> bb;
    T error = T(INFINITY);

    math::vertex<T> v;

    {
      v = data.access(0, 0);
      if(v.size() != 1) return false;
      XSIZE = (unsigned int)v[0].c[0];
    }

    {
      v = data.access(1, 0);
      if(v.size() != 1) return false;
      YSIZE = (unsigned int)v[0].c[0];
    }

    AA.resize(YSIZE, XSIZE);
    bb.resize(YSIZE);

    {
      for(unsigned int i=0;i<data.size(2);i++){
	v = data.access(2, i);
	whiteice::dynamic_bitset b;
	b.resize(v.size());
	b.reset();

	for(unsigned int k=0;k<v.size();k++){
	  if(v[k] != 0.0f) b.set(k, true);
	  else b.set(k, false);
	}

	fset.insert(b);
      }
    }

    {
      cd.resize(data.size(3));

      for(unsigned int i=0;i<data.size(3);i++){
	v = data.access(3, i);
	cd[i].TYPE = v[0].c[0];
      }
    }

    {
      if(cd.size() != data.size(4)) return false;

      for(unsigned int i=0;i<data.size(4);i++){
	v = data.access(4, i);

	cd[i].bins.resize(v[0].c[0]);

	for(unsigned int k=1;k<=cd[i].bins.size();k++){
	  cd[i].bins[k-1] = v[k].c[0];
	}
      }
    }

    {
      if(cd.size() != data.size(5)) return false;
      
      for(unsigned int i=0;i<data.size(5);i++){
	v = data.access(5, i);
	cd[i].elem.resize((unsigned int)v[0].c[0]);

	unsigned int index = 0;
	unsigned int vstart = 1;
	unsigned int vindex = 1;

	while(index < cd[i].elem.size()){

	  if(v[vindex] != '\0') vindex++;
	  else{
	    
	    char buffer[2048];

	    for(unsigned int k=vstart;k<=vindex;k++){
	      buffer[k-vstart] = (char)v[vstart].c[0];
	    }

	    cd[i].elem[index] = std::string(buffer);

	    index++;
	    
	    vstart = vindex+1;
	    vindex++;
	  }
	}
      }
    }

    {
      v = data.access(6, 0);

      if(v.size() != AA.size()) return false;

      for(unsigned int i=0;i<AA.size();i++){
	AA[i] = v[i];
      }
    }

    {
      v = data.access(7, 0);
      
      if(v.size() != bb.size()) return false;

      for(unsigned int i=0;i<bb.size();i++){
	bb[i] = v[i];
      }
    }

    {
      v = data.access(8, 0);

      if(v.size() != 1) return false;

      error = v[0];
    }

    
    
    this->f_itemset = fset;
    this->disc = cd;
    this->A = AA;
    this->b = bb;
    this->currentError = error;

    return true;
  }


  template <typename T> 
  void GlobalOptimizer<T>::optimizer_loop()
  {
    try{
      if(thread_running == false) return;
      
      std::vector< std::vector<std::string> > data;
      std::vector< std::vector<double> > bindata;
      std::vector< std::vector<double> > xresults;

      for(unsigned int i=0;i<xdata.size();i++){
	std::vector<std::string> row;
	row.resize(xdata[i].size());

	for(unsigned int k=0;k<row.size();k++){
	  char buffer[80];
	  sprintf(buffer, "%f", xdata[i][k].c[0]);
	  row[k] = std::string(buffer);
	}

	data.push_back(row);

	if(thread_running == false) return;
      }

      {
	std::lock_guard<std::mutex> lock(solution_mutex);
	
	if(whiteice::calculate_discretize(data, this->disc) == false){
	  return;
	}

	if(thread_running == false) return;
	
	if(whiteice::binarize(data, this->disc, bindata) == false){
	  return;
	}

	if(thread_running == false) return;

	double freq_limit = 0.0;
	whiteice::math::convert(freq_limit, this->levelOfDetailFreq);
	
	if(whiteice::enrich_data(bindata, this->f_itemset, xresults, freq_limit) == false){
	  return;
	}
	
	if(thread_running == false) return;
      }

      this->xdata.resize(xresults.size());

      for(unsigned int i=0;i<xresults.size();i++){
	whiteice::math::vertex<T> x;
	x.resize(xresults[i].size());

	for(unsigned int k=0;k<x.size();k++){
	  x[k] = xresults[i][k];
	}

	this->xdata[i] = x;

	if(thread_running == false) return;
      }

      // don't normalize y data

      std::cout << "xdata: " << xdata.size() << " " << xdata[0].size() << std::endl;
      std::cout << "ydata: " << ydata.size() << " " << ydata[0].size() << std::endl;


      //////////////////////////////////////////////////////////////////////
      // now we have preprocessed data, calculates linear model.. 

      {
	if(thread_running == false) return;
	
	std::lock_guard<std::mutex> lock(solution_mutex);
	
	if(linear_optimization(this->xdata, this->ydata,
			       this->A, this->b, this->currentError) == false){
	  currentError = (double)(INFINITY);
	  std::cout << "WARN: linear_optimization FAILED." << std::endl;
	  return;
	}
      }
      
      thread_running = false;
      return;
    }
    catch(std::exception& e){
      std::cout << "ERROR: GlobalOptimizer exception: " << e.what() << "." << std::endl;
      thread_running = false;
      currentError = T(INFINITY);
      xdata.clear();
      ydata.clear();
    }
  }



  template class GlobalOptimizer< math::blas_real<float> >;
  template class GlobalOptimizer< math::blas_real<double> >;
  //template class GlobalOptimizer< math::blas_complex<float> >;
  //template class GlobalOptimizer< math::blas_complex<double> >;

  //template class GlobalOptimizer< math::superresolution< math::blas_real<float> > >;
  //template class GlobalOptimizer< math::superresolution< math::blas_real<double> > >;
  //template class GlobalOptimizer< math::superresolution< math::blas_complex<float> > >;
  //template class GlobalOptimizer< math::superresolution< math::blas_complex<double> > >;

  
};

