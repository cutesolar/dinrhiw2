
#include "LinearKCluster.h"
#include "GeneralKCluster.h"

#include <functional>
#include <math.h>

#include "discretize.h"

#include "RNG.h"
#include "correlation.h"
#include "linear_equations.h"
#include "dataset.h"
#include "nnetwork.h"
#include "bayesian_nnetwork.h"


namespace whiteice
{

  template <typename T>
  GeneralKCluster<T>::GeneralKCluster()
  {
  }

  template <typename T>
  GeneralKCluster<T>::~GeneralKCluster()
  {
    std::lock_guard<std::mutex> lock(start_mutex);

    if(model){
      delete model;
      model = nullptr;
    }
    
  }
  
  template <typename T>
  bool GeneralKCluster<T>::startTrain(const std::vector< math::vertex<T> >& xdata,
				      const std::vector< math::vertex<T> >& ydata)
  {
    try{
      std::lock_guard<std::mutex> lock(start_mutex);
      
      if(xdata.size() == 0 ||  ydata.size() == 0) return false;
      if(xdata.size() != ydata.size()) return false;

      std::vector< std::vector<std::string> > data;
      std::vector< std::vector<double> > bindata;
      std::vector< std::vector<double> > xresults;

      std::cout << xdata.size() << std::endl;

      for(unsigned int i=0;i<xdata.size();i++){
	std::vector<std::string> row;
	row.resize(xdata[i].size());

	for(unsigned int k=0;k<row.size();k++){
	  char buffer[80];
	  sprintf(buffer, "%f", xdata[i][k].c[0]);
	  row[k] = std::string(buffer);
	}

	data.push_back(row);
      }

      if(whiteice::calculate_discretize(data, this->disc) == false){
	return false;
      }

      if(whiteice::binarize(data, this->disc, bindata) == false){
	return false;
      }

      if(whiteice::enrich_data(bindata, this->f_itemset, xresults) == false){
	return false;
      }

      this->xdata.resize(xresults.size());

      for(unsigned int i=0;i<xresults.size();i++){
	whiteice::math::vertex<T> x;
	x.resize(xresults[i].size());

	for(unsigned int k=0;k<x.size();k++){
	  x[k] = xresults[i][k];
	}

	this->xdata[i] = x;
      }

      
      // zero means and unit variances y-data
      if(0){
	dataset<T> dset;
	dset.createCluster("y", ydata[0].size());
	dset.add(0, ydata);
	dset.preprocess(0);
	dset.getData(0, this->ydata);
      }
      else{
	this->ydata = ydata;
      }
      
#if 0
      // reports linear optimization error 
      {
	math::matrix<T> A;
	math::vertex<T> b;
	T error = T(0.0);
	
	if(linear_optimization(this->xdata, this->ydata,
			       A, b, error)){
	  std::cout << "Linear optimization error is: " << error << std::endl;
	}
      }
#endif

      const unsigned int XSIZE = this->xdata[0].size();
      const unsigned int YSIZE = this->ydata[0].size();

      if(model){
	delete model;
	model = nullptr;
      }
      
      model = new LinearKCluster<T>(XSIZE, YSIZE);
      
      model->setEarlyStopping(false);

      model->startTrain(this->xdata, this->ydata);
    }
    catch(std::exception& e){
      return false;
    }

      
    return true;
    
  }

  template <typename T>
  bool GeneralKCluster<T>::isRunning() const
  {
    std::lock_guard<std::mutex> lock(start_mutex);
    
    if(model) return model->isRunning();
    else return false;
  }

  template <typename T>
  bool GeneralKCluster<T>::stopTrain()
  {
    std::lock_guard<std::mutex> lock(start_mutex);

    if(model) return model->stopTrain();
    else return false;
  }

  template <typename T>
  bool GeneralKCluster<T>::getSolutionError(unsigned int& iters, double& error) const
  {
    std::lock_guard<std::mutex> lock(start_mutex);

    if(model) return model->getSolutionError(iters, error);
    else return false;
  }

  template <typename T>
  unsigned int GeneralKCluster<T>::getNumberOfClusters() const
  {
    std::lock_guard<std::mutex> lock(start_mutex);

    if(model) return model->getNumberOfClusters();
    else return false;
  }

  template <typename T>
  bool GeneralKCluster<T>::predict(const math::vertex<T>& x,
				   math::vertex<T>& y) const
  {
    std::lock_guard<std::mutex> lock(start_mutex);

    if(model == nullptr) return false;

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
    
    return model->predict(xdisc, y);
  }

  template <typename T>
  bool GeneralKCluster<T>::save(const std::string& filename) const
  {
    std::lock_guard<std::mutex> lock(start_mutex);

    if(model == nullptr) return false;
    
    // save disc and f_itemset, + XSIZE, YSIZE here!

    whiteice::dataset<T> data;

    data.createCluster("XSIZE", 1);
    data.createCluster("YSIZE", 1);
    data.createCluster("f_itemset", f_itemset.begin()->size());
    data.createCluster("TYPE", 1);
    data.createCluster("bins", 1000);
    data.createCluster("str_elems", 4096);

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

    std::string extrafilename = filename + ".general-k";

    if(data.save(extrafilename) == false) return false;

    if(model) return model->save(filename);
    else return false;
  }
  

  template <typename T>
  bool GeneralKCluster<T>::load(const std::string& filename)
  {
    std::lock_guard<std::mutex> lock(start_mutex);

    std::string extrafilename = filename + ".general-k";

    whiteice::dataset<T> data;

    if(data.load(extrafilename) == false) return false;

    if(data.getNumberOfClusters() != 6) return false;
    
    // load disc here! + XSIZE, YSIZE
    unsigned int XSIZE, YSIZE;
    std::set< whiteice::dynamic_bitset > fset;
    std::vector< struct whiteice::discretization > cd;

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

    if(model) delete model;

    model = new LinearKCluster<T>(XSIZE, YSIZE);
	
    if(model->load(filename) == false) return false;
    
    this->f_itemset = fset;
    this->disc = cd;

    return true;
  }



  template class GeneralKCluster< math::blas_real<float> >;
  template class GeneralKCluster< math::blas_real<double> >;
  
  //template class GeneralKCluster< math::blas_complex<float> >;
  //template class GeneralKCluster< math::blas_complex<double> >;

  //template class GeneralKCluster< math::superresolution< math::blas_real<float> > >;
  //template class GeneralKCluster< math::superresolution< math::blas_real<double> > >;
  //template class GeneralKCluster< math::superresolution< math::blas_complex<float> > >;
  //template class GeneralKCluster< math::superresolution< math::blas_complex<double> > >;

  
};

