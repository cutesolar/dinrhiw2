/*
 * data set implementation
 *
 * FIXME: save()/load() 
 *        assumes sizeof(float) == 32bits and sizeof(int) == 32bits
 *
 * FIXME: dataset load()/save() mysteriously FAILs on mingw/win32
 */

#ifndef dataset_cpp
#define dataset_cpp

#include <stdexcept>
#include <exception>
#include <vector>
#include <typeinfo>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <iostream>
#include <string>
#include <locale>
#include <codecvt>

#if defined(WINNT) || defined(WIN32) || defined(_WIN32)
#ifndef UNICODE
#define UNICODE
#endif
#endif


#include "dinrhiw_blas.h"
#include "vertex.h"
#include "eig.h"
#include "dataset.h"
#include "blade_math.h"
#include "norms.h"
#include "Log.h"

/**************************************************/

namespace whiteice
{
  
  // static member definition
  template <typename T> 
  const char* dataset<T>::FILEID_STRING = "whiteice::dataset";
  
  
  template <typename T>
  dataset<T>::dataset()
  {
    clusters.clear();
    namemapping.clear();
  }
  
  
  // creates dataset with given dimensionality
  // data is set of vectors
  template <typename T>
  dataset<T>::dataset(unsigned int dimension) 
  {
    // dimension can be zero, this means
    // default cluster cannot be used
    
    clusters.resize(1);
    clusters[0].cname  = std::string("default");
    clusters[0].cindex = 0;
    clusters[0].preprocessings.clear();
    clusters[0].data_dimension = dimension;
    namemapping["default"] = 0;
  }
  
  
  template <typename T>
  dataset<T>::dataset(const dataset<T>& d)
  {
    clusters.resize(d.clusters.size());

    for(unsigned int i=0;i<clusters.size();i++){
      clusters[i] = d.clusters[i];
    }

    this->namemapping = d.namemapping;
  }


  template <typename T>
  dataset<T>& dataset<T>::operator=(const dataset<T>& d)
  {
    clusters.resize(d.clusters.size());

    for(unsigned int i=0;i<clusters.size();i++){
      clusters[i] = d.clusters[i];
    }

    this->namemapping = d.namemapping;

    return (*this);
  }
  
  
  template <typename T>
  dataset<T>::~dataset() {
    clusters.clear();
    namemapping.clear();
  }


  // copies preprocessing and other information to dataset but no data (perfect copy but no data)
  template <typename T>
  void dataset<T>::copyAllButData(const dataset<T>& d)
  {
    clusters.resize(d.clusters.size());

#pragma omp parallel for schedule(auto)
    for(unsigned int i=0;i<clusters.size();i++){
      clusters[i].cname = d.clusters[i].cname;
      clusters[i].cindex = d.clusters[i].cindex;

      clusters[i].data_dimension = d.clusters[i].data_dimension;
      clusters[i].data.clear(); // DO NOT COPY DATA
      
      clusters[i].preprocessings = d.clusters[i].preprocessings;

      clusters[i].mean = d.clusters[i].mean;
      clusters[i].variance = d.clusters[i].variance;

      clusters[i].softmax_parameter = d.clusters[i].softmax_parameter;

      clusters[i].Rxx = d.clusters[i].Rxx;
      clusters[i].Wxx = d.clusters[i].Wxx;
      clusters[i].invWxx = d.clusters[i].invWxx;
      
      clusters[i].ICA = d.clusters[i].ICA;
      clusters[i].invICA = d.clusters[i].invICA;
      
      namemapping[clusters[i].cname] = i;
    }
  }
  
  
  ////////////////////////////////////////////////////////////
  // cluster code
  
  template <typename T>
  bool dataset<T>::createCluster(const std::string& name, const unsigned int dimension)
  {
    if(name.length() <= 0)
      return false;
    
    if(namemapping.find(name) != namemapping.end())
      return false;
    
    unsigned int n = clusters.size();
    clusters.resize(clusters.size()+1);
    clusters[n].cname = name;
    clusters[n].cindex = n;
    clusters[n].data_dimension = dimension;
    clusters[n].preprocessings.clear();
    namemapping[name] = n;
    
    return true;
  }

  template <typename T>
  bool dataset<T>::resetCluster(const unsigned int index, const std::string& name, const unsigned int dimension)
  {
    if(index >= clusters.size()) return false;

    if(name.length() <= 0)
      return false;

    auto it = namemapping.find(name);
    
    if(it != namemapping.end())
      if(it->second != index)
	return false;
    
    clusters[index].cname = name;
    clusters[index].cindex = index;
    clusters[index].data_dimension = dimension;
    clusters[index].preprocessings.clear();
    namemapping[name] = index;

    // removes data and preprocessing information
    clusters[index].data.clear();
    clusters[index].preprocessings.clear();

    return true;
  }

  
  template <typename T>
  bool dataset<T>::getClusterNames(std::vector<std::string>& names) const
  {
    names.clear();
    
    typename std::vector<cluster>::const_iterator i;
    i = clusters.begin();
    
    while(i != clusters.end()){
      names.push_back(i->cname);
      i++;
    }
    
    return true;
  }
  
  
  template <typename T>
  unsigned int dataset<T>::getCluster(const std::string& name) const
  {
    typename std::map<std::string, unsigned int>::const_iterator i;
    i = namemapping.find(name);
    
    if(i == namemapping.end())
      return (unsigned int)(-1);
    
    return (i->second);
  }
  
  
  template <typename T>
  std::string dataset<T>::getName(unsigned int index) const {
    if(index >= clusters.size()) return "";
    
    return clusters[index].cname;
  }


  template <typename T>
  bool dataset<T>::setName(const unsigned int index, const std::string& name) {
    if(index >= clusters.size()) return false;
    
    if(name.size() > 0)
      clusters[index].cname = name;
    else
      return false;
    
    return true;
  }
  
  
  template <typename T>
  unsigned int dataset<T>::getNumberOfClusters() const
  {
    return clusters.size();
  }
  
  
  template <typename T>
  bool dataset<T>::removeCluster(std::string& name)
  {
    typename std::map<std::string, unsigned int>::iterator i = 
      namemapping.find(name);
    
    if(i == namemapping.end())
      return false;
    
    return removeCluster(i->second);
  }
  
  
  template <typename T>
  bool dataset<T>::removeCluster(unsigned int index)
  {
    if(index >= clusters.size())
      return false;
    
    // not very good implementation, ok because
    // number of clusters is usually small 
    // ( < 10, typically 1 or 2)
    
    // removes cluster
    typename std::vector<cluster>::iterator i = clusters.begin();
    for(unsigned int j=0;j<index;j++,i++);
    clusters.erase(i);
    
    // recalculates *->cindex and namemapping entries
    
    namemapping.clear();
    
    i = clusters.begin();
    index = 0;
    
    while(i != clusters.end()){
      i->cindex = index;
      namemapping[i->cname] = index;
      index++;
      i++;
    }
    
    return true;
  }
  
  
  
  // adds data examples
  template <typename T>
  bool dataset<T>::add(const math::vertex<T>& input, bool nopreprocess) {
    return add(0, input, nopreprocess);
  }
  
  
  template <typename T>
  bool dataset<T>::add(const std::vector<math::vertex<T> >& inputs, bool nopreprocess) 
  {
    return add(0, inputs, nopreprocess);
  }
  
  
  template <typename T>  
  bool dataset<T>::add(const std::string& input, bool nopreprocess) {
    return add(0, input, nopreprocess);
  }
  
  
  template <typename T>  
  bool dataset<T>::add(const std::vector<std::string>& inputs, bool nopreprocess) 
  {
    return add(0, inputs, nopreprocess);
  }
  
  
  
  // adds data to clusters
  template <typename T>
  bool dataset<T>::add(unsigned int index,
		       const math::vertex<T>& input, bool nopreprocess) 
  {
    if(index >= clusters.size()) 
      // slow.. (internal calls need only single check)
      return false;
    
    try{ 
      if(input.size() != clusters[index].data_dimension)
	return false;
      
      clusters[index].data.push_back(input);
      unsigned int n = clusters[index].data.size() - 1;
      
      if(nopreprocess == false){
	// preprocesses if it is needed
	if(!preprocess(index, clusters[index].data[n])){
	  clusters[index].data.resize(n);
	  return false;
	}
      }
      
      return true;
    }
    catch(std::exception& e){
      std::cout << "unexpected exception: "
		<< e.what() << std::endl;
      
      return false;
    }
    
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::add(unsigned int index,
		       const std::vector<math::vertex<T> >& inputs, bool nopreprocess) 
  {
    if(index >= clusters.size())
      return false;
    
    if(inputs.size() <= 0)
      return true;
    
    {
      const unsigned int DD = clusters[index].data_dimension;
      
      for(unsigned int i=0;i<inputs.size();i++)
	if(inputs[i].size() != DD)
	  return false;
    }

    typename std::vector<math::vertex<T> >::const_iterator i;
    
    {
      unsigned int last = clusters[index].data.size();
      
      for(i=inputs.begin();i!=inputs.end();i++,last++){
	clusters[index].data.push_back(*i);
	if(nopreprocess == false)
	  preprocess(index, clusters[index].data[last]);
      }
    }
    
    return true;
  }

  
  template <typename T>
  bool dataset<T>::add(unsigned int index, const std::vector<T>& input,
		       bool nopreprocess) 
  {
    math::vertex<T> v(input.size());
    
    for(unsigned int i=0;i<v.size();i++)
      v[i] = input[i];
    
    return this->add(index, v, nopreprocess);
  }
  
  

  
  
  template <typename T>
  bool dataset<T>::add(unsigned int index,
		       const std::string& input, bool nopreprocess) 
  {
    if(index >= clusters.size()) // slow.. (internal calls)
      return false;
    
    if(clusters[index].data_dimension != input.length())
      return false;
    
    try{
      math::vertex<T> vec;
      vec.resize(clusters[index].data_dimension);
      
      for(unsigned int i=0;i<input.length();i++)
	vec[i] = T(input[i]);
      
      return add(index, vec);
    }
    catch(std::exception& e){ return false; }
    
  }
  
  
  template <typename T>
  bool dataset<T>::add(unsigned int index,
		       const std::vector<std::string>& inputs,
		       bool nopreprocess) 
  {
    typename std::vector<std::string >::const_iterator i;
    
    for(i=inputs.begin();i!=inputs.end();i++){
      if(!add(index, *i)) return false;
    }
    
    return true;
  }
  
  // creates empty dataset
  template <typename T>
  bool dataset<T>::clear()
  {
    clusters.clear();
    namemapping.clear();
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::clearData(unsigned int index)
  {
    if(index >= clusters.size())
      return false;
    
    clusters[index].data.clear();
    return true;
  }
  
  
  template <typename T> // reduces size of data
  bool dataset<T>::resize(unsigned int index,
			  unsigned int nsize) 
  {
    if(index >= clusters.size())
      return false;
    
    if(clusters[index].data.size() < nsize)
      return false;
    
    clusters[index].data.resize(nsize);
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::clearAll(unsigned int index)
  {
    if(index >= clusters.size())
      return false;
    
    clusters[index].data.clear();
    
    clusters[index].preprocessings.clear();
    
    clusters[index].Rxx.resize(1,1);
    clusters[index].Wxx.resize(1,1);
    clusters[index].invWxx.resize(1,1);
    clusters[index].mean.resize(1);
    clusters[index].variance.resize(1);

    clusters[index].ICA.resize(1,1);
    
    return true;
  }

  
  /*
   * downsamples all clusters jointly down to samples samples.
   * fails if clusters have different sizes or samples == 0 or
   * it is larger that given samples. 
   */
  template <typename T>
  bool dataset<T>::downsampleAll(unsigned int samples) 
  {
    if(clusters.size() <= 0) return false;

    const unsigned int N = clusters[0].data.size();

    for(unsigned int i=0;i<clusters.size();i++){
      if(clusters[i].data.size() != N)
	return false;
    }

    if(samples >= N){
      return false;
    }
    else if(samples == 0){
      for(unsigned int i=0;i<clusters.size();i++){
	clusters[i].data.clear();
      }
      return true;
    }
    
    std::vector<dataset<T>::cluster> d;
    d.resize(clusters.size());

    for(unsigned int i=0;i<samples;i++){
      unsigned int index = rand() % N;
      
      for(unsigned int j=0;j<clusters.size();j++){
	d[j].data.push_back(clusters[j].data[index]);
      }
    }

    for(unsigned int j=0;j<clusters.size();j++)
      clusters[j].data = d[j].data;

    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::removeBadData()
  {
    if(clusters.size() <= 0)
      return true; // nothing to do
    
    // checks all cluster sizes are equal
    const unsigned int N = clusters[0].data.size();

    for(unsigned int i=0;i<clusters.size();i++){
      if(clusters[i].data.size() != N)
	return false;
    }
    
    
    std::vector<dataset<T>::cluster> d;
    d = clusters;
    
    for(unsigned int k=0;k<clusters.size();k++){
      d[k].data.clear();
    }
    
    for(unsigned int i=0;i<N;i++){
      bool bad_data = false;
      
      for(unsigned int k=0;k<clusters.size();k++){
	const auto& v = clusters[k].data[i];
	
	// std::cout << "test:  " << v << std::endl;
	// std::cout << "vsize: " << v.size() << std::endl;
	
	for(unsigned int d=0;d<v.size();d++){
	  // std::cout << v[d] << std::endl;
	  // std::cout << whiteice::math::tohex(v[d]) << std::endl;
	  
	  if(whiteice::math::isnan(v[d])){
	    // printf("NaN detected\n");
	    bad_data = true;
	  }
	  else if(whiteice::math::isinf(v[d])){
	    bad_data = true;
	  }
	}
	
	// std::cout << "bad data status: " << bad_data << std::endl;
      }
      
      // if(bad_data) printf("BAD DATA DETECTED\n");
      
      
      if(bad_data == false){
	for(unsigned int k=0;k<clusters.size();k++){
	  d[k].data.push_back(clusters[k].data[i]);
	}
      }
    }
    
    clusters = d;
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::getData(unsigned int index, std::vector< math::vertex<T> >& data) const 
  {
    if(index >= clusters.size())
      throw std::out_of_range("cluster index too big");
    
    data = clusters[index].data;
    
    return true;
  }
  
  
  // iterators for dataset
  template <typename T>
  typename dataset<T>::iterator dataset<T>::begin(unsigned int index) 
  {
    if(index >= clusters.size())
      throw std::out_of_range("cluster index too big.");
    
    return clusters[index].data.begin();
  }
  
  template <typename T>
  typename dataset<T>::iterator dataset<T>::end(unsigned int index) 
  {
    if(index >= clusters.size())
      throw std::out_of_range("cluster index too big.");
    
    return clusters[index].data.end();
  }
  
  template <typename T>
  typename dataset<T>::const_iterator dataset<T>::begin(unsigned int index) const 
  {
    if(index >= clusters.size())
      throw std::out_of_range("Cluster index too big.");
    
    return clusters[index].data.begin();
  }
  
  template <typename T>
  typename dataset<T>::const_iterator dataset<T>::end(unsigned int index) const 
  {
    if(index >= clusters.size())
      throw std::out_of_range("Cluster index too big.");
    
    return clusters[index].data.end();
  }
  
  
  
  template <typename T>
  bool dataset<T>::load(const std::string& filename) 
  {
    // loads datasets which have been saved in
    // dataset format = 1 (older dataset format 0 is not supported)
    //
    //  FILEID_STRING : char[]
    //  version : INT
    //  cnum    : INT  number of clusters
    //  namelen : INT  length of names section in bytes (includes padding)
    //  names   : [list of NULL terminated strings] (cnum names)
    //  padding : (=> address dividable by 4)
    //  clusters: [list of CLUSTER]
    //
    // CLUSTER:
    //  datasize: INT  number of data vectors
    //  dimen.  : INT
    //  NORMFLAG: INT  normalization flags
    //  softmax : float
    //  [mean]  : float[DIM]
    //  [var]   : float[DIM]
    //  [Rxx]   : float[DIM]x[DIM]
    //  [ICA]   : float[DIM]x[DIM]
    //  data    : float[datasize]x[DIM]
    
    if(filename.length() <= 0)
      return false;
    
    FILE* fp = NULL;

#ifdef UNICODE
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> converter;
    std::wstring wstr = converter.from_bytes(filename.data());

    std::string fmode = "rb";
    std::wstring wmode = converter.from_bytes(fmode.data());

    fp = (FILE*)_wfopen(wstr.c_str(), wmode.c_str());
#else
    fp = (FILE*)fopen(filename.c_str(), "rb");
#endif
    
    if(fp == 0) return false;
    if(feof(fp) || ferror(fp))
      return false;
    
    // printf("L: %d\n", (int)ftell(fp));
    
    unsigned int version = 0xFFFFFFFF;
    unsigned int cnum = 0;
    
    // checks file id string
    {
      char line[128] = "";

      // printf("F: %d\n", (int)strlen(FILEID_STRING)+1);
      const size_t s = strlen(FILEID_STRING)+1;
      
      if(fread(line, 1, s, fp) != s)
      {
	fclose(fp);
	return false;
      }

      if(strcmp(line, FILEID_STRING) != 0){
	fclose(fp);
	return false;
      }
    }

    // printf("L: %d\n", (int)ftell(fp));
    
    if(fread(&version, 4, 1, fp) != 1){
      fclose(fp);
      return false;
    }
    
    if(fread(&cnum, 4, 1, fp) != 1){
      fclose(fp);
      return false;
    }

    if(cnum > 1000000){ // only supports up to 1.000.000 clusters
      fclose(fp);
      return false;
    }

    // we only support version 2
    if(version != 2){
      fclose(fp);
      return false;
    }
    
    
    clusters.resize(cnum);

    // reads names
    if(cnum > 0){
      // gets names section length
      
      unsigned int namesSectionSize = 0;
      
      if(fread(&namesSectionSize, 4, 1, fp) != 1){
	clusters.resize(0);
	fclose(fp);
	return false;
      }

      if(namesSectionSize > 10000000){ // only supports 10 million characters
	clusters.resize(0);
	fclose(fp);
	return false;
      }
      
      char* names = (char*)malloc(namesSectionSize);
      if(names == 0){
	clusters.resize(0);
	fclose(fp);
	return false;
      }
      
      if(fread(names, namesSectionSize, 1, fp) != 1){
	clusters.resize(0);
	free(names);
	fclose(fp);
	return false;
      }
      
      char *ptr, *endptr;
      ptr = names;
      endptr = &(names[namesSectionSize]);
      
      unsigned int index = 0;
      while(ptr < endptr && index <clusters.size()){
	char* begin = ptr;
	
	while(ptr < endptr && *ptr != '\0') ptr++;
	
	// found null
	if(*ptr == '\0'){
	  clusters[index].cname = begin; // copies string
	  index++;
	}
	
	ptr++;
      }
      
      free(names);
      
      if(index != clusters.size()){
	clusters.resize(0);
	fclose(fp);
	return false;
      }
      
    }
    
    //////////////////////////////////////////////////////////////////////
    // reads in each cluster
    
    for(unsigned int i=0;i<clusters.size();i++){
      unsigned int datasize = 0;
      unsigned int data_dimension = 0;
      unsigned int flags = 0;
      float d32_softmax = 0.0f;
      
      // reads basic cluster information
      
      if(fread(&datasize, 4, 1, fp) != 1){
	clusters.resize(0);
	fclose(fp);
	return false;
      }

      if(fread(&data_dimension, 4, 1, fp) != 1){
	clusters.resize(0);
	fclose(fp);
	return false;	
      }

      if(fread(&flags, 4, 1, fp) != 1){
	clusters.resize(0);
	fclose(fp);
	return false;	
      }
      
      if(fread(&d32_softmax, 4, 1, fp) != 1){
	clusters.resize(0);
	fclose(fp);
	return false;	
      }

      
      clusters[i].data_dimension = data_dimension;
      clusters[i].softmax_parameter = T(d32_softmax);
      
      clusters[i].preprocessings.clear();
      
      if(flags & 0x02)
	clusters[i].preprocessings.push_back(dnSoftMax);
      if(flags & 0x01)
	clusters[i].preprocessings.push_back(dnMeanVarianceNormalization);
      if(flags & 0x04)
	clusters[i].preprocessings.push_back(dnCorrelationRemoval);
      if(flags & 0x08)
	clusters[i].preprocessings.push_back(dnLinearICA);	
      
      
      
      //////////////////////////////////////////////////////////////////////
      // reads cluster statistics

      float* buffer = (float*)calloc(2*clusters[i].data_dimension, 4);
      
      if(flags & 0x01){
	clusters[i].mean.resize(clusters[i].data_dimension);
	clusters[i].variance.resize(clusters[i].data_dimension);
	
	if(fread(buffer, 4, 2*clusters[i].mean.size(), fp) != 2*clusters[i].mean.size())
	{
	  clusters.resize(0);
	  fclose(fp);
	  free(buffer);
	  return false;
	}
	
	
	for(unsigned int j=0;j<clusters[i].mean.size();j++){
	  if(sizeof(abs(T(0.0f))) == sizeof(T(0.0f))){ // single value per number (real)
	    whiteice::math::convert(clusters[i].mean[j], buffer[2*j]);
	  }
	  else{ // complex number (two values per number)
	    auto value = whiteice::math::blas_complex<double>(buffer[2*j],buffer[2*j+1]);
	    whiteice::math::convert(clusters[i].mean[j], value);
	  }
	}
	
	
	if(fread(buffer, 4, 2*clusters[i].variance.size(), fp) != 2*clusters[i].variance.size())
	{
	  clusters.resize(0);
	  fclose(fp);
	  free(buffer);
	  return false;
	}
	
	for(unsigned int j=0;j<clusters[i].variance.size();j++){
	  if(sizeof(abs(T(0.0f))) == sizeof(T(0.0f))){ // single value per number (real)
	    whiteice::math::convert(clusters[i].variance[j], buffer[2*j]);
	  }
	  else{ // complex number (two values per number)
	    auto value = whiteice::math::blas_complex<double>(buffer[2*j],buffer[2*j+1]);
	    whiteice::math::convert(clusters[i].variance[j], value);
	  }
	}
      }
      
      if(flags & 0x04){
	clusters[i].Rxx.resize(clusters[i].data_dimension,
			       clusters[i].data_dimension);
	
	for(unsigned int a=0;a<data_dimension;a++){
	  
	  if(fread(buffer, 4, 2*data_dimension, fp) != 2*data_dimension){
	    clusters.resize(0);
	    fclose(fp);
	    free(buffer);
	    return false;
	  }
	  
	  for(unsigned int b=0;b<data_dimension;b++){
	    if(sizeof(abs(T(0.0f))) == sizeof(T(0.0f))){ // single value per number (real)
	      whiteice::math::convert(clusters[i].Rxx(a,b), buffer[2*b]);
	    }
	    else{ // complex number (two values per number)
	      auto value = whiteice::math::blas_complex<double>(buffer[2*b],buffer[2*b+1]);
	      whiteice::math::convert(clusters[i].Rxx(a,b), value);
	    }
	  }
	}
	
	
	// recalculates Wx and invWx vectors
	math::matrix<T> D(clusters[i].Rxx);
	math::matrix<T> V, Vh, invD;
	
	if(symmetric_eig(D, V) == false){
	  clusters.resize(0);
	  fclose(fp);
	  free(buffer);
	  return false;
	}
	
	invD = D;
	
	for(unsigned int j=0;j<D.ysize();j++){
	  T d = abs(invD(j,j));

	  auto epsilon = abs(T(1e-8f));

	  if(d > epsilon){
	      invD(j,j) = whiteice::math::sqrt(T(1.0)/d);
	      D(j,j)    = whiteice::math::sqrt(d);
	  }
	  else{
	    invD(j,j) = T(0.0f);
	    D(j,j)    = T(0.0f);
	  }
	  
	}
	
	Vh = V;
	Vh.hermite();
	
	clusters[i].Wxx = V * invD * Vh;
	clusters[i].invWxx = V * D * Vh;
	
      }


      if(flags & 0x08){
	clusters[i].ICA.resize(clusters[i].data_dimension,
			       clusters[i].data_dimension);
	
	for(unsigned int a=0;a<data_dimension;a++){
	  
	  if(fread(buffer, 4, 2*data_dimension, fp) != 2*data_dimension){
	    clusters.resize(0);
	    fclose(fp);
	    free(buffer);
	    return false;
	  }
	  
	  for(unsigned int b=0;b<data_dimension;b++){
	    if(sizeof(abs(T(0.0f))) == sizeof(T(0.0f))){ // single value per number (real)
	      whiteice::math::convert(clusters[i].ICA(a,b), buffer[2*b]);
	    }
	    else{ // complex number (two values per number)
	      auto value = whiteice::math::blas_complex<double>(buffer[2*b],buffer[2*b+1]);
	      whiteice::math::convert(clusters[i].ICA(a,b), value);
	    }
	  }
	}

	clusters[i].invICA = clusters[i].ICA;
	if(clusters[i].invICA.inv() == false)
	  return false; // calculating inverse of ICA failed.
      }      

      //////////////////////////////////////////////////////////////////////
      // reads cluster data
      
      clusters[i].data.resize(datasize);
      
      for(unsigned int a=0;a<clusters[i].data.size();a++){
	
	if(fread(buffer, 4, 2*clusters[i].data_dimension, fp) != 2*clusters[i].data_dimension)
	{
	  clusters.resize(0);
	  fclose(fp);
	  free(buffer);
	  return false;
	}
	
	clusters[i].data[a].resize(data_dimension);
	for(unsigned int b=0;b<data_dimension;b++){
	  if(sizeof(abs(T(0.0f))) == sizeof(T(0.0f))){ // single value per number (real)
	    whiteice::math::convert(clusters[i].data[a][b], buffer[2*b]);
	  }
	  else{ // complex number (two values per number)
	    auto value = whiteice::math::blas_complex<double>(buffer[2*b],buffer[2*b+1]);
	    whiteice::math::convert(clusters[i].data[a][b], value);
	  }
	}
      }
      
      
      free(buffer);
    }
    
    
    if(ferror(fp)){
      clusters.resize(0);
      fclose(fp);      
      return false;
    }
    
    
    fclose(fp);
    
    
    // sets up rest of data structures
    namemapping.clear();
    
    for(unsigned int i=0;i<clusters.size();i++){
      clusters[i].cindex = i;
      namemapping[clusters[i].cname] = i;
    }
    
    
    return true;
  }
  
  
  
  
  template <typename T>
  bool dataset<T>::save(const std::string& filename) const 
  {
    // dataset is saved as binary file in following format.
    // all data is either 32bit unsigned integers or 32bit floats
    // 
    //  FILEID_STRING : char[]
    //  version : INT
    //  cnum    : INT  number of clusters
    //  namelen : INT  length of names list in bytes (incl. padding)
    //  names   : [list of NULL terminated strings] (cnum names)
    //  padding : (address divisable by 4)
    //  clusters: [list of CLUSTER]
    //
    // CLUSTER:
    //  datasize: INT  number of data vectors
    //  dimen.  : INT
    //  NORMFLAG: INT  normalization flags
    //  softmax : float
    //  [mean]  : float[DIM] (two values per element)
    //  [var]   : float[DIM] (two values per element)
    //  [Rxx]   : float[DIM]x[DIM] (two values per element)
    //  [ICA]   : float[DIM]x[DIM] (two values per element)
    //  data    : float[datasize]x[DIM] (two values per element)
    
    if(filename.length() <= 0)
      return false;
    
    FILE* fp = NULL;

#ifdef UNICODE
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> converter;
    std::wstring wstr = converter.from_bytes(filename.data());

    std::string fmode = "wb";
    std::wstring wmode = converter.from_bytes(fmode.data());

    fp = (FILE*)_wfopen(wstr.c_str(), wmode.c_str());
#else
    fp = (FILE*)fopen(filename.c_str(), "wb");
#endif
    fflush(stdout);
     
    if(fp == NULL){
    	return false;
    }

    if(ferror(fp)){
    	fclose(fp);
    	return false;
    }

    // version 0 was initial version number for previous
    // version 1 did not support complex numbers
    // version 2 saves/loads values to disk as complex numbers
    const unsigned int version = 2; 
    // dataset fileformat (not supported)
    const unsigned int cnum    = clusters.size();

    if(fwrite(FILEID_STRING, 1, strlen(FILEID_STRING)+1, fp) != strlen(FILEID_STRING)+1){
      fclose(fp);
      remove(filename.c_str());
      return false;
    }


    if(fwrite(&version, 4, 1, fp) != 1){
      fclose(fp);
      remove(filename.c_str());
      return false;
    }


    if(fwrite(&cnum, 4, 1, fp) != 1){
      fclose(fp);
      remove(filename.c_str());
      return false;
    }


    // writes names list
    if(clusters.size() > 0){
      unsigned int namesSectionSize = 0;

      for(unsigned int i=0;i<clusters.size();i++)
	namesSectionSize += (clusters[i].cname.length() + 1);

      // adds padding
      namesSectionSize = ((namesSectionSize+3)/4)*4;

      if(fwrite(&namesSectionSize, 4, 1, fp) != 1){
	fclose(fp);
	remove(filename.c_str());
	return false;
      }

      char* names = (char*)calloc(1, namesSectionSize);
      if(names == 0){	
	fclose(fp);
	remove(filename.c_str());
	return false;
      }

      char *ptr;
      ptr = names;
      
      for(unsigned int i=0;i<clusters.size();i++){
	for(unsigned int j=0;j<(clusters[i].cname.length()+1);j++, ptr++){
	  *ptr = clusters[i].cname[j];
	}
      }
      
      
      if(fwrite(names, namesSectionSize, 1, fp) != 1){
	fclose(fp);
	free(names);
	remove(filename.c_str());
	return false;
      }

      free(names);
    }


    //////////////////////////////////////////////////////////////////////
    // writes each cluster

    for(unsigned int i=0;i<clusters.size();i++){
      const unsigned int datasize = clusters[i].data.size();
      const unsigned int data_dimension = clusters[i].data_dimension;

      unsigned int flags = 0;
      // flags are (order is always same):
      // bit 0 = dnMeanVarianceNormalization
      // bit 1 = dnSoftMax
      // bit 2 = dnCorrelationRemoval
      // bit 3 = dnLinearICA

      for(unsigned int j=0;j<clusters[i].preprocessings.size();j++){
	if(clusters[i].preprocessings[j] == dnMeanVarianceNormalization)
	  flags |= 0x01;
	else if(clusters[i].preprocessings[j] == dnSoftMax)
	  flags |= 0x02;
	else if(clusters[i].preprocessings[j] == dnCorrelationRemoval)
	  flags |= 0x04;
	else if(clusters[i].preprocessings[j] == dnLinearICA)
	  flags |= 0x08;
      }

      float d32_softmax = 0.0f;
      math::convert(d32_softmax, clusters[i].softmax_parameter);

      // writes basic cluster information

      if(fwrite(&datasize, 4, 1, fp) != 1){
	fclose(fp);
	remove(filename.c_str());
	return false;
      }

      if(fwrite(&data_dimension, 4, 1, fp) != 1){
	fclose(fp);
	remove(filename.c_str());
	return false;
      }

      if(fwrite(&flags, 4, 1, fp) != 1){
	fclose(fp);
	remove(filename.c_str());
	return false;
      }

      if(fwrite(&d32_softmax, 4, 1, fp) != 1){
	fclose(fp);
	remove(filename.c_str());
	return false;
      }


      //////////////////////////////////////////////////////////////////////
      // writes cluster statistics
      
      
      float* buffer = (float*)calloc(4, 2*clusters[i].data_dimension);

      if(buffer == 0){
	fclose(fp);
	remove(filename.c_str());
	return false;
      }
      
      
      if(flags & 0x01){
	for(unsigned int j=0;j<clusters[i].mean.size();j++){
	  auto realpart = math::real(clusters[i].mean[j]);
	  auto imagpart = math::imag(clusters[i].mean[j]);
	  
	  math::convert(buffer[2*j+0], realpart);
	  math::convert(buffer[2*j+1], imagpart);
	}
	
	if(fwrite(buffer, 4, 2*clusters[i].mean.size(), fp) != 2*clusters[i].mean.size())
	  {
	    fclose(fp);
	    remove(filename.c_str());
	    free(buffer);
	    return false;
	  }
	
	
	for(unsigned int j=0;j<clusters[i].variance.size();j++){
	  auto realpart = math::real(clusters[i].variance[j]);
	  auto imagpart = math::imag(clusters[i].variance[j]);
	  
	  math::convert(buffer[2*j+0], realpart);
	  math::convert(buffer[2*j+1], imagpart);
	}
	
	if(fwrite(buffer, 4, 2*clusters[i].variance.size(), fp) != 2*clusters[i].variance.size())
	{
	  remove(filename.c_str());
	  fclose(fp);
	  free(buffer);
	  return false;
	}
      }
      
      
      if(flags & 0x04){
	for(unsigned int a=0;a<data_dimension;a++){
	  
	  for(unsigned int b=0;b<data_dimension;b++){
	    auto realpart = math::real(clusters[i].Rxx(a,b));
	    auto imagpart = math::imag(clusters[i].Rxx(a,b));
	    
	    math::convert(buffer[2*b+0], realpart);
	    math::convert(buffer[2*b+1], imagpart);
	  }
	  
	  if(fwrite(buffer, 4, 2*data_dimension, fp) != 2*data_dimension){
	    remove(filename.c_str());
	    fclose(fp);
	    free(buffer);
	    return false;
	  }
	}
      }


      if(flags & 0x08){
	for(unsigned int a=0;a<data_dimension;a++){
	  
	  for(unsigned int b=0;b<data_dimension;b++){
	    auto realpart = math::real(clusters[i].ICA(a,b));
	    auto imagpart = math::imag(clusters[i].ICA(a,b));
	    
	    math::convert(buffer[2*b+0], realpart);
	    math::convert(buffer[2*b+1], imagpart);
	  }
	  
	  if(fwrite(buffer, 4, 2*data_dimension, fp) != 2*data_dimension){
	    remove(filename.c_str());
	    fclose(fp);
	    free(buffer);
	    return false;
	  }
	}
      }
      
      
      //////////////////////////////////////////////////////////////////////
      // writes cluster data
      
      for(unsigned int a=0;a<clusters[i].data.size();a++){
	
	for(unsigned int b=0;b<data_dimension;b++){
	  auto realpart = math::real(clusters[i].data[a][b]);
	  auto imagpart = math::imag(clusters[i].data[a][b]);
	  
	  math::convert(buffer[2*b+0], realpart);
	  math::convert(buffer[2*b+1], imagpart);
	}
	
	if(fwrite(buffer, 4, 2*data_dimension, fp) != 2*data_dimension){
	  remove(filename.c_str());
	  fclose(fp);
	  free(buffer);
	  return false;
	}
      }
      
      free(buffer);
      
    }
    
    
    if(ferror(fp))
      return false;
    
    fclose(fp);
    
    return true;
  }
  

  template <typename T>
  bool dataset<T>::exportAscii(const std::string& filename,
			       const unsigned int cluster_index,
			       const bool writeHeaders,
			       const bool raw) const 
  {
    if(filename.length() <= 0)
      return false;

    if(cluster_index >= clusters.size())
      return false;
    
    FILE* fp = NULL;

#ifdef UNICODE
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> converter;
    std::wstring wstr = converter.from_bytes(filename.data());

    std::string fmode = "wt";
    std::wstring wmode = converter.from_bytes(fmode.data());

    fp = (FILE*)_wfopen(wstr.c_str(), wmode.c_str());
#else
    fp = (FILE*)fopen(filename.c_str(), "wt");
#endif
    
    if(fp == 0) return false;
    if(ferror(fp)){ fclose(fp); return false; }
    
    const unsigned int BUFSIZE = 2048;
    
    char* buffer = (char*)malloc(BUFSIZE*sizeof(char));
    if(buffer == NULL){ fclose(fp); return false; }

    const auto index = cluster_index; // rename variable
    
    {
      if(writeHeaders){
	snprintf(buffer, BUFSIZE, "# cluster %d: %d datapoints %d dimension(s).\n",
		 index, (int)clusters[index].data.size(), clusters[index].data_dimension);
	fputs(buffer, fp);

	snprintf(buffer, BUFSIZE,
		 "# complex number format = 'real(x[0]) imag(x[0]) real(x[1]) imag(x[1])..'.\n");
	fputs(buffer, fp);
      }
      
      // dumps data in this cluster to ascii format
      for(auto d : clusters[index].data){
	if(raw == false) // removes possible preprocessing from data
	  this->invpreprocess(index, d); 
	
	if(clusters[index].data_dimension > 0){
	  float rvalue = 0.0f, ivalue = 0.0f;
	  whiteice::math::convert(rvalue, math::real(d[0]));
	  whiteice::math::convert(ivalue, math::imag(d[0]));
	  
	  snprintf(buffer, BUFSIZE, "%+f %+f", rvalue, ivalue);
	  fputs(buffer, fp);
	  
	  for(unsigned int i=1;i<d.size();i++){
	    whiteice::math::convert(rvalue, math::real(d[i]));
	    whiteice::math::convert(ivalue, math::imag(d[i]));

	    snprintf(buffer, BUFSIZE, " %+f %+f", rvalue, ivalue);
	    fputs(buffer, fp);
	  }
	  
	  fputs("\n", fp);
	}
      }
    }
    
    free(buffer);
    if(ferror(fp)){ fclose(fp); return false;}
    
    fclose(fp);
    return true;
  }

  
  /*
   * imports space, "," or ";" separated floating point numbers as vectors into cluster 0
   * which will be overwritten. Ignores the first line which may contain headers and
   * reads at most LINES of vertex data or unlimited amount of data (if set to 0).
   *
   * if realData is true then import real data and not complex data
   * * (one number per number instead of 2 numbers per number)
   */
  template <typename T>
  bool dataset<T>::importAscii(const std::string& filename,
			       const int cluster_index,
			       const unsigned int LINES,
			       const bool realData) 
  {
    std::vector< math::vertex<T> > import;

    if(cluster_index >= 0)
      if(cluster_index >= (int)getNumberOfClusters())
	return false;

    FILE* fp = NULL;

#ifdef UNICODE
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> converter;
    std::wstring wstr = converter.from_bytes(filename.data());

    std::string fmode = "rt";
    std::wstring wmode = converter.from_bytes(fmode.data());

    fp = (FILE*)_wfopen(wstr.c_str(), wmode.c_str());
#else
    fp = fopen(filename.c_str(), "rt");
#endif

    if(fp == 0 || ferror(fp)){
      if(fp) fclose(fp);
      return false;
    }

    const unsigned int BUFLEN = 1000000;
    char* buffer = (char*)malloc(BUFLEN);
    if(buffer == NULL){
      fclose(fp);
      return false;
    }
    
    
    // import format is
    // <file> = (<line>"\n")*
    // <line> = <vector> | <comment>
    // <vector> = "%f %f %f %f ... "
    // <comment> = "# any string"
    // where separators are either spaces, ";" or "," numbers are assumed to have form -12.3101

    unsigned int lines = 0;

    while(!feof(fp)){
      if(LINES){
	if(lines >= LINES)
	  break; // we stop after we have processed LINES lines
      }

      if(fgets(buffer, BUFLEN, fp) != buffer){
	break; // silently fails if we cannot read new line
      }

      // intepretes buffer as a vector
      std::vector<double> line;
      char* s = buffer;
      
      if(*s == '#') // ignore comment lines (only possible as the first character in line)
	continue;

      while(*s == ' ' || *s == ',' || *s == ';' || *s == '\t' || *s == '|') s++;
    
      while(*s != '\n' && *s != '\0' && *s != '\r'){	
	char* prev = s;
	double v = strtod(s, &s);
	if(s == prev){
	  break; // no progress
	}
	
	if(whiteice::math::isnan(v) || whiteice::math::isinf(v)){
	  // bad data
	  
	  if(fp) fclose(fp);
	  free(buffer);
	  return false; 
	}

	line.push_back(v);
	if(realData) line.push_back(0.0); // adds zero complex element
	
	while(*s == ' ' || *s == ',' || *s == ';' || *s == '\t' || *s == '|')
	  s++;
      }

      if(import.size() > 0 && line.size() > 0){
	if(line.size() != 2*import[0].size()){ // number of dimensions must match for all lines
	  fclose(fp);
	  free(buffer);
	  return false; // we just give up if there is strange/bad file
	}
      }

      if(line.size() > 0){
	// converts  vector of double values to vertex

	math::vertex<T> vec; // line
	vec.resize(line.size()/2);
	
	if(sizeof(abs(T(0.0f))) == sizeof(T(0.0f))){ // single value per number (real)
	  for(unsigned int i=0;i<vec.size();i++){
	    whiteice::math::convert(vec[i], line[2*i]);
	  }
	}
	else{ // complex number (two values)
	  if((line.size() & 1) == 1){ // odd number means error
	    fclose(fp);
	    free(buffer);
	    return false; // we just give up if there is strange/bad file
	  }

	  vec.resize(line.size()/2);

	  for(unsigned int i=0;i<(line.size()/2);i++){
	    auto value = whiteice::math::blas_complex<double>(line[2*i],line[2*i+1]);
	    whiteice::math::convert(vec[i], value);
	  }
	  
	}
	
	
	import.push_back(vec);
	lines++;
      }
    }
    
    free(buffer);
    fclose(fp);

    if(import.size() <= 0)
      return false;

    // clears cluster 0 and adds new data
    if(this->getNumberOfClusters() > 0 && cluster_index >= 0){
      this->clearAll(cluster_index);

      typename std::map<std::string, unsigned int>::const_iterator i;
      i = namemapping.find(clusters[cluster_index].cname);
      
      std::string name = "data import";
      clusters[cluster_index].data_dimension = import[0].size();
      clusters[cluster_index].cname = name;
      clusters[cluster_index].cindex = 0;
      
      if(i != namemapping.end())
	namemapping.erase(i);
      
      namemapping[name] = cluster_index;
      clusters[cluster_index].data = import;
      
      return true;
    }
    else{
      if(this->createCluster("data import", import[0].size()) == false)
	return false;

      if(this->add(0, import) == false){
	this->removeCluster(0);
	return false;
      }
      
      return true;
    }
  }
  
  
  // accesses zero cluster
  template <typename T>
  const math::vertex<T>& dataset<T>::operator[](unsigned int index) const
    
  {
    if(clusters.size() <= 0){
      std::string error = "dataset::operator[]: cluster zero doesn't exist.";
      throw std::out_of_range(error);
    }
    
    if(index >= clusters[0].data.size()){
      throw std::out_of_range("dataset::operator[]: cluster index out of range");
    }
    
    return clusters[0].data[index];
  }
  
  template <typename T>
  const math::vertex<T>& dataset<T>::access(unsigned int cluster, unsigned int data) const 
  {
    if(cluster >= clusters.size())
      throw std::out_of_range("dataset::access(): cluster index out of range");
    
    if(data >= clusters[cluster].data.size())
      throw std::out_of_range("dataset::access(): data index out of range");

    return clusters[cluster].data[data];
  }

  template <typename T>
  math::vertex<T>& dataset<T>::access(unsigned int cluster, unsigned int data)
  {
    if(cluster >= clusters.size())
      throw std::out_of_range("dataset::access(): cluster index out of range");
    
    if(data >= clusters[cluster].data.size())
      throw std::out_of_range("dataset::access(): data index out of range");
    
    return clusters[cluster].data[data];
  }
  
  template <typename T>
  const math::vertex<T>& dataset<T>::accessName(const std::string& clusterName, unsigned int dataElem) 
  {
    typename std::map<std::string, unsigned int>::const_iterator i;
    i = namemapping.find(clusterName);
    
    if(i == namemapping.end())
      throw std::out_of_range("dataset: cannot find cluster name");
    
    const unsigned int cluster = i->second;
    
    if(dataElem >= clusters[cluster].data.size())
      throw std::out_of_range("data index out of range");
    
    return clusters[cluster].data[dataElem];
  }


  template <typename T>
  unsigned int dataset<T>::size(unsigned int index) const   // dataset size  
  {
    if(index >= clusters.size())
      return (unsigned int)(-1);
    
    return clusters[index].data.size();
  }
  
  template <typename T>
  bool dataset<T>::clear(unsigned int index)   // data set clear  
  {
    if(index >= clusters.size())
      return false;
    
    clusters[index].data.clear();
    return true;
  }
  
  template <typename T>
  unsigned int dataset<T>::dimension(unsigned int index) const   // dimension of data vectors
  {
    if(index >= clusters.size())
      return (unsigned int)(-1);
    
    return clusters[index].data_dimension;
  }
  
  
  template <typename T>
  bool dataset<T>::getPreprocessings(unsigned int cluster,
				     std::vector<data_normalization>& preprocessings) const 
  {
    if(cluster >= clusters.size())
      return false;
    
    preprocessings.clear();
    preprocessings = clusters[cluster].preprocessings;
    
    return true;
  }
  
  
  // data preprocessing
  template <typename T>
  bool dataset<T>::preprocess(unsigned int index, enum data_normalization norm) 
  {
    if(index >= clusters.size())
      return false;
    
    try{
      if(norm == dnMeanVarianceNormalization){
	
	if(is_normalized(index, dnMeanVarianceNormalization))
	  return true;
	
	// sets up mean&variance vectors
	{
	  clusters[index].mean.resize(clusters[index].data_dimension);
	  clusters[index].variance.resize(clusters[index].data_dimension);
	  
	  clusters[index].mean.zero();
	  clusters[index].variance.zero();
	}
	
	// calculates mean & variance
	// (calculates first E[X] and E[X^2]
	// -> mean,variance (almost correct)
	{
	  typename std::vector<math::vertex<T> >::iterator i;
	  
	  i = clusters[index].data.begin();
	  
	  while(i != clusters[index].data.end()){
	    clusters[index].mean += (*i);
	    
	    for(unsigned int k=0;k<clusters[index].data_dimension;k++)
	      clusters[index].variance[k] += ((*i)[k]) * math::conj((*i)[k]);
	    
	    i++;
	  }
	  
	  // calculates E[X] and sqrt(Var[X]) = sqrt(E[X^2] - E[X]**2)
	  
	  if(clusters[index].data.size() > 0){
	    clusters[index].mean /= T(clusters[index].data.size());
	    clusters[index].variance /= T(clusters[index].data.size());
	  }
	  
	  // not needed
	  //mean.resize(data_dimension);
	  //variance.resize(data_dimension);
	  
	  
	  {
	    unsigned int k = 0;
	    
	    while(k < clusters[index].data_dimension){
	      if(k < clusters[index].data_dimension){
		clusters[index].variance[k] -=
		  (clusters[index].mean[k])*math::conj(clusters[index].mean[k]);

		auto epsilon = abs(T(0.0000001));

		math::blas_real<double> e, vk;
		whiteice::math::convert(e, epsilon);
		whiteice::math::convert(vk, abs(clusters[index].variance[k]));

		//if(abs(clusters[index].variance[k]) < epsilon)
		if(vk < e)
		  clusters[index].variance[k] = T(epsilon);
		
		clusters[index].variance[k]  =
		  T(whiteice::math::sqrt(whiteice::math::abs(clusters[index].variance[k])));
		k++;
	      }
	      else{
		break;
	      }
	    }
	    
	  }
	  
	}
	
	{
#pragma omp parallel for schedule(auto)
	  for(unsigned int i=0;i<clusters[index].data.size();i++){
	    mean_variance_removal(index, clusters[index].data[i]);
	  }

#pragma omp barrier
	}
	
	clusters[index].preprocessings.push_back(dnMeanVarianceNormalization);

	return true;
      }
      else if(norm == dnSoftMax){
	const T r = T(0.80);

	// softmax code here requires that we have something like N(0,1) distributed data
	if(is_normalized(index, dnMeanVarianceNormalization) == false){
	  if(preprocess(index, dnMeanVarianceNormalization) == false){
	    return false;
	  }
	}
	
	if(is_normalized(index, dnSoftMax))
	  return true;

	clusters[index].softmax_parameter = r;
	
	// soft max
	{
#pragma omp parallel for schedule(auto)
	  for(unsigned int i=0;i<clusters[index].data.size();i++)
	    soft_max(index, clusters[index].data[i]);
	}
	
	clusters[index].preprocessings.push_back(dnSoftMax);
	return true;
      }
      else if(norm == dnCorrelationRemoval){

	if(is_normalized(index, dnCorrelationRemoval))
	  return true;

	if(is_normalized(index, dnMeanVarianceNormalization) == false)
	  if(preprocess(index, dnMeanVarianceNormalization) == false)
	    return false;

	if(typeid(T) == typeid(whiteice::math::blas_complex<float>) || 
	   typeid(T) == typeid(whiteice::math::blas_complex<double>))
	{
	  printf("Warning/FIXME: dataset PCA correlation removal currently FAILS with complex data.\n");
	  return false;
	}

	// we can use autocorrelation because mean is already zero
	if(autocorrelation(clusters[index].Rxx, clusters[index].data) == false){
	  clusters[index].Rxx.resize(clusters[index].data_dimension, clusters[index].data_dimension);
	  clusters[index].Rxx.identity();
	}

	
	// std::cout << "Rxx = " << clusters[index].Rxx << std::endl;


	// regularizes the problem by adding variance to diagonal
	// in order to be able to compute eig()
	
	math::matrix<T> V, Vh, invD, D(clusters[index].Rxx);
	T dd = T(1e-2f);
	unsigned int counter = 0;

	
	while(symmetric_eig(D, V) == false){
	  D = clusters[index].Rxx;

	  for(unsigned int i=0;i<D.ysize();i++){
	    D(i,i) += dd;
	  }

	  dd = T(2.0)*dd;
	  counter++;

	  if(counter >= 10){
	    D.resize(clusters[index].data_dimension, clusters[index].data_dimension);
	    D.identity();
	    V.resize(clusters[index].data_dimension, clusters[index].data_dimension);
	    V.identity();

	    std::cout << "Calculating symmetric eigenvalue decomposition failed."
		      << std::endl;

	    // don't do silent failure anymore
	    // break;
	    return false;
	    
	  }
	}

	
	// std::cout << "typeinfo = " << typeid(T).name() << std::endl;
	// std::cout << "D = " << D << std::endl;
	// std::cout << "V = " << V << std::endl;
	
	invD = D;
	
	for(unsigned int i=0;i<invD.ysize();i++){
	  T d = abs(invD(i,i));

	  const auto epsilon = abs(T(1e-8f));
	  
	  if(d > epsilon){
	    invD(i,i) = whiteice::math::sqrt(T(1.0)/(epsilon + d));
	    D(i,i)    = whiteice::math::sqrt(d);
	  }
	  else{
	    invD(i,i) = T(0.0f);
	    D(i,i)    = T(0.0f);
	  }
	  
	}
	
	
	// std::cout << "invD = " << invD << std::endl;
	
	Vh = V;
	Vh.hermite();
	
	clusters[index].Wxx = V * invD * Vh;
	clusters[index].invWxx = V * D * Vh;

	// std::cout << "Wxx      = " << clusters[index].Wxx << std::endl;
	// std::cout << "inv(Wxx) = " << clusters[index].invWxx << std::endl;

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<clusters[index].data.size();i++)
	  whiten(index, clusters[index].data[i]);
	
	clusters[index].preprocessings.push_back(dnCorrelationRemoval);

	return true;
      }
      else if(norm == dnLinearICA)
      {
	if(is_normalized(index, dnLinearICA))
	  return true; // already preprocessed

	if(clusters[index].data.size() <= 1)
	  return false; // cannot work

	// we must have correlation removal so do it first
	if(is_normalized(index, dnCorrelationRemoval) == false)
	  if(preprocess(index, dnCorrelationRemoval) == false)
	    return false;

	if(math::ica(clusters[index].data, clusters[index].ICA) == false){
	  std::cout << "Calculating independent component analysis failed."
		    << std::endl;
	  return false;
	}
	else{
	  // calcualtes inverse of ICA
	  clusters[index].invICA = clusters[index].ICA;
	  if(clusters[index].invICA.inv() == false){
	    std::cout << "Cannot compute inverse of ICA." << std::endl;
	    std::cout << "ICA = " << clusters[index].ICA << std::endl;
	    return false; // calculating of ICA preprocessing failed.
	  }
	    
	}

	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<clusters[index].data.size();i++)
	  ica(index, clusters[index].data[i]);
	
	clusters[index].preprocessings.push_back(dnLinearICA);
	return true;
      }
      else return false;
    }
    catch(std::exception& e){
      std::cout << "dataset::preprocess: fatal error (exception): " << e.what() << std::endl;
      return false;
    }
  }
  
  
  template <typename T>
  bool dataset<T>::preprocess(enum data_normalization norm) {
    if(clusters.size() > 0)
      return preprocess(0, norm);
    else
      return false;
  }
  
  
  
  // inverse preprocess everything, calculates new preprocessing parameters
  // and preprocesses everything with parameter data from the whole dataset
  // (dataset may grow after preprocessing)
  template <typename T>
  bool dataset<T>::repreprocess(unsigned int index) 
  { 
    if(index >= clusters.size())
      return false;
    
    // saves preprocesings
    std::vector<enum data_normalization > old_preprocessings;
    typename std::vector<enum data_normalization >::iterator i;
    
    for(i=clusters[index].preprocessings.begin();
	i!=clusters[index].preprocessings.end();i++)
      old_preprocessings.push_back(*i);
    
    // inverse preprocess whole data
    if(!invpreprocess(index, clusters[index].data))
      return false;  
    
    clusters[index].preprocessings.clear();
    
    // apply preprocessings to whole data again
    for(i = old_preprocessings.begin();i!=old_preprocessings.end();i++){
      if(!preprocess(index, *i))
	return false;
    }
    
    return true;
  }
  
  
  // converts data with same preprocessing as with dataset vectors
  template <typename T>
  bool dataset<T>::preprocess(unsigned int index, math::vertex<T>& vec) const 
  {
    if(index >= clusters.size()) // this is slow (optimize internal calls)
      return false;

    if(vec.size() != clusters[index].data_dimension)
      return false;
    
    typename std::vector<enum data_normalization>::const_iterator i;
    
    for(i=clusters[index].preprocessings.begin();i!=clusters[index].preprocessings.end();i++){

      if(*i == dnLinearICA){
	ica(index, vec);
      }
      else if(*i == dnCorrelationRemoval){
    	  whiten(index, vec);
      }
      else if(*i == dnMeanVarianceNormalization){
    	  mean_variance_removal(index, vec);
      }
      else if(*i == dnSoftMax){
    	  soft_max(index, vec);
      }
      else return false;
    }
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::preprocess(unsigned int index,
			      std::vector< math::vertex<T> >& group) const 
  {
    if(index >= clusters.size())
      return false;

    bool ok = true;

#pragma omp parallel for schedule(auto)
    for(unsigned int i=0;i<group.size();i++){
      if(ok == false) continue;
      if(!preprocess(index, group[i])) ok = false;
    }

    return ok;
  }
  
  
  template <typename T>
  bool dataset<T>::preprocess(math::vertex<T>& vec) const {
    return preprocess(0, vec);
  }
  
  template <typename T>
  bool dataset<T>::preprocess(std::vector< math::vertex<T> >& group) const {
    return preprocess(0, group);
  }
  
  
  // inverse preprocess given data vector
  template <typename T>
  bool dataset<T>::invpreprocess(unsigned int index, math::vertex<T>& vec) const 
  {
    if(index >= clusters.size()) // this is slow (optimize internal calls)
      return false;

    if(vec.size() != clusters[index].data_dimension)
      return false;
    
    typename std::vector<enum data_normalization>::const_reverse_iterator i;
    
    for(i=clusters[index].preprocessings.rbegin();
	i!=clusters[index].preprocessings.rend();i++){
      if(*i == dnLinearICA){
	inv_ica(index, vec);
      }
      else if(*i == dnCorrelationRemoval){
	inv_whiten(index, vec);
      }
      else if(*i == dnMeanVarianceNormalization){
	inv_mean_variance_removal(index, vec);
      }
      else if(*i == dnSoftMax){
	inv_soft_max(index, vec);
      }
      else return false;
    }

    return true;  
  }


  // inverse preprocess given data vector m and covariance matrix C
  template <typename T>
  bool dataset<T>::invpreprocess(unsigned int index, 
				 math::vertex<T>& m,
				 math::matrix<T>& C) const 
  {
    if(index >= clusters.size())
      return false;

    if(m.size() != clusters[index].data_dimension || 
       C.ysize() != clusters[index].data_dimension ||
       C.xsize() != clusters[index].data_dimension)
      return false;
    
    typename std::vector<enum data_normalization>::const_reverse_iterator i;
    
    for(i=clusters[index].preprocessings.rbegin();
	i!=clusters[index].preprocessings.rend();i++){
      if(*i == dnLinearICA){
	inv_ica(index, m);
	inv_ica_cov(index, C);
      }
      else if(*i == dnCorrelationRemoval){
	inv_whiten(index, m);
	inv_whiten_cov(index, C);
      }
      else if(*i == dnMeanVarianceNormalization){
	inv_mean_variance_removal(index, m);
	inv_mean_variance_removal_cov(index, C);
      }
      else if(*i == dnSoftMax){
	inv_soft_max(index, m);
	inv_soft_max_cov(index,C);
      }
      else return false;
    }

    return true;  
  }
  
  
  template <typename T>
  bool dataset<T>::invpreprocess(unsigned int index,
				 std::vector<math::vertex<T> >& group) const 
  {
    if(index >= clusters.size())
      return false;

    bool ok = true;

#pragma omp parallel for schedule(auto)
    for(unsigned int i=0;i<group.size();i++){
      if(ok == false) continue;
      if(!invpreprocess(index, group[i])) ok = false;
    }

    return ok;
  }
  
  
  template <typename T>
  bool dataset<T>::invpreprocess(math::vertex<T>& vec) const {
    return invpreprocess(0, vec);
  }
  
  
  template <typename T>
  bool dataset<T>::invpreprocess(std::vector< math::vertex<T> >& group) const {
    return invpreprocess(0, group);
  }
  
  
  // removes all preprocessings from data
  template <typename T>
  bool dataset<T>::convert(unsigned int index) 
  {
    if(index >= clusters.size())
      return false;

    // inverse preprocess whole data
    if(!invpreprocess(index, clusters[index].data)) return false;
    clusters[index].preprocessings.clear();
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::convert(unsigned int index,
			   std::vector<enum data_normalization> plist)
  {
    if(index >= clusters.size())
      return false;
    
    // checks that list has only legitime values
    // and it doesn't have duplicates
    
    for(unsigned int i=0;i<plist.size();i++){
      if(plist[i] != dnMeanVarianceNormalization &&
	 plist[i] != dnSoftMax && 
	 plist[i] != dnCorrelationRemoval &&
	 plist[i] != dnLinearICA) return false;
      
      for(unsigned int j=0;j<plist.size();j++){
	if(i == j) continue;
	if(plist[i] == plist[j]) return false;
      }
    }
    
    // checks how far the both lists are same and 
    // invert only preprocessings that don't match
    
    {
      unsigned int i=0;
      while(i < plist.size() && i < clusters[index].preprocessings.size()){
	if(plist[i] != clusters[index].preprocessings[i])
	  break;
	else i++;
      }
      
      typename std::vector< math::vertex<T> >::iterator k;
      
      for(unsigned int j=i;j<clusters[index].preprocessings.size();j++){
	if(clusters[index].preprocessings[j] == dnMeanVarianceNormalization){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    inv_mean_variance_removal(index, *k);
	}
	else if(clusters[index].preprocessings[j] == dnSoftMax){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    inv_soft_max(index, *k);
	}
	else if(clusters[index].preprocessings[j] == dnCorrelationRemoval){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    inv_whiten(index, *k);
	}
	else if(clusters[index].preprocessings[j] == dnLinearICA){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    inv_ica(index, *k);
	}	
      }
      
      // preprocesses with plist[restoflist]
      
      for(unsigned int j=i;j<plist.size();j++){
	if(clusters[index].preprocessings[j] == dnMeanVarianceNormalization){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    mean_variance_removal(index, *k);
	}
	else if(clusters[index].preprocessings[j] == dnSoftMax){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    soft_max(index, *k);
	}
	else if(clusters[index].preprocessings[j] == dnCorrelationRemoval){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    whiten(index, *k);
	}
	else if(clusters[index].preprocessings[j] == dnLinearICA){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    ica(index, *k);
	}
      }
      
      
      clusters[index].preprocessings = plist;
    }
    
    
    return true;
  }


  template <typename T>
  bool dataset<T>::preprocess_grad(unsigned int index, math::matrix<T>& W) const 
  {
    if(index >= clusters.size())
      return false;

    for(unsigned int j=0;j<clusters[index].preprocessings.size();j++)
      if(clusters[index].preprocessings[j] == dnSoftMax)
	return false; // cannot compute linear gradient for softmax

    W.resize(this->dimension(index), this->dimension(index));
    W.identity();
    
    for(unsigned int j=0;j<clusters[index].preprocessings.size();j++){
      if(clusters[index].preprocessings[j] == dnMeanVarianceNormalization){
	for(unsigned int i=0;i<dimension(index);i++)
	  if(clusters[index].variance[i] > T(10e-8))
	    for(unsigned int k=0;k<dimension(index);k++)
	      W(k,i) /= (T(2.0)*clusters[index].variance[i]);
      }
      else if(clusters[index].preprocessings[j] == dnCorrelationRemoval){
	W = clusters[index].Wxx * W;
      }
      else if(clusters[index].preprocessings[j] == dnLinearICA){
	W = clusters[index].ICA * W;
      }
    }

    return true;
  }

  template <typename T>
  bool dataset<T>::invpreprocess_grad(unsigned int index, math::matrix<T>& W) const 
  {
    if(index >= clusters.size())
      return false;
    
    for(unsigned int i=0;i<clusters[index].preprocessings.size();i++)
      if(clusters[index].preprocessings[i] == dnSoftMax)
	return false; // cannot compute linear gradient for softmax

    W.resize(this->dimension(index), this->dimension(index));
    W.identity();
    
    for(unsigned int j=0;j<clusters[index].preprocessings.size();j++){
      if(clusters[index].preprocessings[j] == dnMeanVarianceNormalization){
	for(unsigned int i=0;i<dimension(index);i++)
	  if(clusters[index].variance[i] > T(10e-8))
	    for(unsigned int k=0;k<dimension(index);k++)
	      W(k,i) *= (T(2.0)*clusters[index].variance[i]);
      }
      else if(clusters[index].preprocessings[j] == dnCorrelationRemoval){
	W = clusters[index].invWxx * W;
      }
      else if(clusters[index].preprocessings[j] == dnLinearICA){
	W = clusters[index].invICA * W;
      }
    }

    return true;
  }

  template <typename T>
  bool dataset<T>::diagnostics(const int cluster, const bool verbose) const 
  {
    if(cluster >= (int)clusters.size()) return false;
    
    whiteice::logging.info("dataset::diagnostics()");
    if(verbose){
      printf("dataset::diagnostics()\n");
      fflush(stdout);
    }

    const unsigned int BUFLEN=8192;
    char buffer[BUFLEN];
    
    for(unsigned int c=0;c<clusters.size();c++){
      if(cluster >= 0)
	if(((unsigned int)cluster) != c)
	  continue; // if cluster is positive show only that cluster
      
      // prints min and max value
      {
	auto maxvalue = -abs(T(INFINITY));
	auto minvalue = abs(T(INFINITY));
	for(auto& d : clusters[c].data){
	  for(unsigned int i=0;i<d.size();i++){
	    if(abs(d[i]) > maxvalue) maxvalue = abs(d[i]);
	    if(abs(d[i]) < minvalue) minvalue = abs(d[i]);
	  }
	}
	
	double temp1 = 0.0, temp2 = 0.0;
	whiteice::math::convert(temp1, maxvalue);
	whiteice::math::convert(temp2, minvalue);
	
	snprintf(buffer, BUFLEN, "Cluster %d (N=%d) max abs(value): %f min abs(value): %f\n",
		 c, (int)clusters[c].data.size(), temp1, temp2);
	if(verbose){
	  printf("%s", buffer);
	  fflush(stdout);
	}
	whiteice::logging.info(buffer);
      }

      // prints preprocessing parameters
      {
	double temp = 0.0f;
	whiteice::math::convert(temp, clusters[c].softmax_parameter);
	
	snprintf(buffer, BUFLEN, "Cluster %d: softmax value: %f\n", c, temp);
	if(verbose) printf("%s", buffer);
	whiteice::logging.info(buffer);
	
	std::string line;
	clusters[c].mean.toString(line);
	T nrm = clusters[c].mean.norm();
	whiteice::math::convert(temp, nrm);
	snprintf(buffer, BUFLEN, "Cluster: %d: ||mean|| = %f. mean = %s\n",
		 c, temp, line.c_str());
	if(verbose) printf("%s", buffer);
	whiteice::logging.info(buffer);

	clusters[c].variance.toString(line);
	nrm = clusters[c].variance.norm();
	whiteice::math::convert(temp, nrm);
	snprintf(buffer, BUFLEN, "Cluster: %d: ||stdev|| = %f. stdev = %s\n",
		 c, temp, line.c_str());
	if(verbose) printf("%s", buffer);
	whiteice::logging.info(buffer);

	clusters[c].Rxx.toString(line);
	nrm = math::frobenius_norm(clusters[c].Rxx);
	whiteice::math::convert(temp, nrm);
	snprintf(buffer, BUFLEN, "Cluster: %d: ||Rxx|| = %f. Rxx = %s\n",
		 c, temp, line.c_str());
	if(verbose) printf("%s", buffer);
	whiteice::logging.info(buffer);

	clusters[c].Wxx.toString(line);
	nrm = math::frobenius_norm(clusters[c].Wxx);
	whiteice::math::convert(temp, nrm);
	snprintf(buffer, BUFLEN, "Cluster: %d: ||Wxx|| = %f. Wxx = %s\n",
		 c, temp, line.c_str());
	if(verbose) printf("%s", buffer);
	whiteice::logging.info(buffer);

	clusters[c].invWxx.toString(line);
	nrm = math::frobenius_norm(clusters[c].invWxx);
	whiteice::math::convert(temp, nrm);
	snprintf(buffer, BUFLEN, "Cluster: %d: ||invWxx|| = %f. Wxx = %s\n",
		 c, temp, line.c_str());
	if(verbose) printf("%s", buffer);
	whiteice::logging.info(buffer);

	clusters[c].ICA.toString(line);
	nrm = math::frobenius_norm(clusters[c].ICA);
	whiteice::math::convert(temp, nrm);
	snprintf(buffer, BUFLEN, "Cluster: %d: ||ICA|| = %f. ICA = %s\n",
		 c, temp, line.c_str());
	if(verbose) printf("%s", buffer);
	whiteice::logging.info(buffer);

	clusters[c].invICA.toString(line);
	nrm = math::frobenius_norm(clusters[c].invICA);
	whiteice::math::convert(temp, nrm);
	snprintf(buffer, BUFLEN, "Cluster: %d: ||invICA|| = %f. invICA = %s\n",
		 c, temp, line.c_str());
	if(verbose) printf("%s", buffer);
	whiteice::logging.info(buffer);
	
	
	if(verbose) fflush(stdout);
      }
      
    }

    

    return true;
  }
  
  
  /**************************************************/
  
  // is data normalized with given operator?
  template <typename T>
  bool dataset<T>::is_normalized(unsigned int index,
				 enum data_normalization norm) const 
  {
    typename std::vector<enum data_normalization>::const_iterator i;
    
    for(i=clusters[index].preprocessings.begin();
	i!=clusters[index].preprocessings.end();i++){
      if(*i == norm) return true;
    }
    
    return false;
  }
  
  
  
  // sets variance to be something like 0.50**2 -> good input/outputs for nn
  template <typename T>
  void dataset<T>::mean_variance_removal(unsigned int index, math::vertex<T>& vec) const
  {
    // x = (x - mean)/sqrt(var) -> new var = 1.0
    
    vec -= clusters[index].mean;

    //const auto epsilon = abs(T(1e-8f));
    
    for(unsigned int i=0;i<vec.size();i++){
      //if(abs(clusters[index].variance[i]) > epsilon){
      vec[i] /= abs(clusters[index].variance[i]);
      //}
    }
    
  }
  
  
  template <typename T>
  void dataset<T>::inv_mean_variance_removal(unsigned int index,
					     math::vertex<T>& vec) const
  {
    // [x' * sqrt(var)] + mean

    //const auto epsilon = abs(T(1e-8f));
    
    for(unsigned int i=0;i<vec.size();i++){
      //if(abs(clusters[index].variance[i]) > epsilon){
	vec[i] *= abs(clusters[index].variance[i]);
	//}
    }
    
    vec += clusters[index].mean;
  }

  
  template <typename T>
  void dataset<T>::inv_mean_variance_removal_cov(unsigned int index,
						 math::matrix<T>& C) const
  {
    // x(i) = [x(i)' * sqrt(var(i))] + mean(i)
    // Var[x(i)] = Var(x'(i)*sqrt(var(i)))
    // Var[x(i)] = Var(eye(diag(sqrt(var)))*x') = eye(diag(var))*Var(x')

    //const auto epsilon = abs(T(1e-8f));

    auto ItrVar = C;
    
    for(unsigned int j=0;j<ItrVar.ysize();j++)
      for(unsigned int i=0;i<ItrVar.xsize();i++)
	if(i == j) ItrVar(j,i) = sqrt(abs(clusters[index].variance[i]));
	else ItrVar(j,i) = T(0.0f);

    C = ItrVar*C*ItrVar;
  }
  
  
  template <typename T>
  void dataset<T>::soft_max(unsigned int index,
			    math::vertex<T>& vec) const
  {
    const unsigned int N = vec.size();
    
    for(unsigned int i=0;i<N;i++){
      T value = -(vec[i]/clusters[index].softmax_parameter);
      value   = (T(1.0f) + whiteice::math::exp(value));
      vec[i] = T(1.0f) / value;
    }
    
  }
  
  template <typename T>
  void dataset<T>::inv_soft_max_cov(unsigned int index,
				    math::matrix<T>& C) const
  {
    // FIXME: implement me!
    assert(0);
  }
  
  template <typename T>
  void dataset<T>::inv_soft_max(unsigned int index,
				math::vertex<T>& vec) const
  {
    const unsigned int N = vec.size();
    
    for(unsigned int i=0;i<N;i++){
      vec[i] = -clusters[index].softmax_parameter *
	whiteice::math::log( (T(1.0f) / vec[i]) - T(1.0f) );
    }
  }
  
  
  template <typename T>
  void dataset<T>::whiten(unsigned int index, math::vertex<T>& vec) const
  {
    vec = clusters[index].Wxx * vec;
  }
  
  
  template <typename T>
  void dataset<T>::inv_whiten(unsigned int index,
			      math::vertex<T>& vec) const
  {
    vec = clusters[index].invWxx * vec;
  }

  
  template <typename T>
  void dataset<T>::inv_whiten_cov(unsigned int index,
			      math::matrix<T>& C) const
  {
    auto At = clusters[index].invWxx;
    At.transpose();
    
    C = clusters[index].invWxx * C * At;
  }



  template <typename T>
  void dataset<T>::ica(unsigned int index,
		       math::vertex<T>& vec) const
  {
    vec = clusters[index].ICA * vec;
  }
  
  
  template <typename T>
  void dataset<T>::inv_ica(unsigned int index,
			   math::vertex<T>& vec) const
  {
    
    vec = clusters[index].invICA * vec;
  }  
  
  template <typename T>
  void dataset<T>::inv_ica_cov(unsigned int index,
			       math::matrix<T>& C) const
  {
    auto At = clusters[index].invICA;
    At.transpose();
    
    C = clusters[index].invICA * C * At;
  }  
  
  
  
  //////////////////////////////////////////////////////////////////////
  
  
  template class dataset< whiteice::math::blas_real<float> >;
  template class dataset< whiteice::math::blas_real<double> >;
  
  template class dataset< whiteice::math::blas_complex<float> >;
  template class dataset< whiteice::math::blas_complex<double> >;

  
  template class dataset< whiteice::math::superresolution<
			    whiteice::math::blas_real<float>,
			    whiteice::math::modular<unsigned int> > >;
  template class dataset< whiteice::math::superresolution<
			    whiteice::math::blas_real<double>,
			    whiteice::math::modular<unsigned int> > >;

  template class dataset< whiteice::math::superresolution<
			    whiteice::math::blas_complex<float>,
			    whiteice::math::modular<unsigned int> > >;
  template class dataset< whiteice::math::superresolution<
			    whiteice::math::blas_complex<double>,
			    whiteice::math::modular<unsigned int> > >;
  
  template class dataset< float >;
  template class dataset< double >;    
  
}
  
#endif
