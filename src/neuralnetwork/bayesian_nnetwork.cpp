

#include "bayesian_nnetwork.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <exception>
#include <stdexcept>
#include <typeinfo>
#include <set>

#include "nnetwork.h"
#include "dataset.h"
#include "dinrhiw_blas.h"
#include "Log.h"


namespace whiteice
{

  template <typename T>
  bayesian_nnetwork<T>::bayesian_nnetwork()
  {
    
  }

  template <typename T>
  bayesian_nnetwork<T>::bayesian_nnetwork(const bayesian_nnetwork<T>& bnet)
  {
    this->nnets.resize(bnet.nnets.size());

    for(unsigned int i=0;i<nnets.size();i++){
      if(bnet.nnets[i] != NULL)
	nnets[i] = new whiteice::nnetwork<T>(*(bnet.nnets[i]));
      else
	nnets[i] = NULL;
    }
  }

  
  template <typename T>
  bayesian_nnetwork<T>::~bayesian_nnetwork()
  {
    for(unsigned int i=0;i<nnets.size();i++){
      if(this->nnets[i]){
	delete this->nnets[i];
	this->nnets[i] = NULL;
      }
    }

    nnets.clear();
  }

  
  template <typename T>
  bayesian_nnetwork<T>& bayesian_nnetwork<T>::operator=(const bayesian_nnetwork<T>& bnet)
  {
    this->nnets.resize(bnet.nnets.size());

    for(unsigned int i=0;i<nnets.size();i++){
      if(bnet.nnets[i] != NULL)
	nnets[i] = new whiteice::nnetwork<T>(*(bnet.nnets[i]));
      else
	nnets[i] = NULL;
    }
    
    return (*this);
  }
  

  template <typename T>
  void bayesian_nnetwork<T>::printInfo() const // mostly for debugging.. prints NN information/data.
  {
    printf("BNN contains %d samples\n", (int)nnets.size());
    
    if(nnets.size() > 0)
      nnets[0]->printInfo();
  }

  
  template <typename T>
  void bayesian_nnetwork<T>::diagnosticsInfo() const 
  {
    char buffer[80];

    for(unsigned int i=0;i<nnets.size();i++){ 
      snprintf(buffer, 80, "BNN NETWORK %d/%d", i+1, (int)nnets.size());
      whiteice::logging.info(buffer);

      nnets[i]->diagnosticsInfo();
    }
    
    
  }
  
  
  // number of samples in BNN
  template <typename T>
  unsigned int bayesian_nnetwork<T>::getNumberOfSamples() const {
	  return nnets.size();
  }

  
  /*
   * imports and exports samples of p(w) to and from nnetwork
   */
  template <typename T>
  bool bayesian_nnetwork<T>::importSamples(const whiteice::nnetwork<T>& nn,
					   const std::vector< math::vertex<T> >& weights)
  {
    if(weights.size() <= 0) return false;
    
    std::vector< nnetwork<T>* > nnnets;
    nnnets.resize(weights.size());

    for(unsigned int i=0;i<nnnets.size();i++){
      nnnets[i] = new nnetwork<T>(nn); // FIXME handle alloc that FAILs..
      
      if(nnnets[i]->importdata(weights[i]) == false){
	for(unsigned int j=0;j<=i;j++){
	  delete nnnets[i];
	  nnnets[i] = NULL;
	}
	
	return false;
      }
    }

    // remove old data
    for(unsigned int i=0;i<this->nnets.size();i++){
      if(this->nnets[i]){
	delete this->nnets[i];
	this->nnets[i] = NULL;
      }
    }

    nnets.clear();

    this->nnets = nnnets; // copies new pointers over old data

    return true;
  }


  template <typename T>
  bool bayesian_nnetwork<T>::importNetwork(const nnetwork<T>& net)
  {
    std::vector< math::vertex<T> > weight;
    weight.resize(1);

    if(net.exportdata(weight[0]) == false)
      return false;
    
    return importSamples(net, weight);
  }
  

  template <typename T>
  bool bayesian_nnetwork<T>::exportSamples(whiteice::nnetwork<T>& nn, 
					   std::vector< math::vertex<T> >& weights,
					   int latestN) const
  {
    if(nnets.size() <= 0) return false;
    if(latestN > (signed)nnets.size()) return false;
    if(latestN <= 0) latestN = nnets.size();

    nn = (*nnets[0]);

    weights.resize(latestN);

    for(unsigned int i=(nnets.size() - latestN);i<nnets.size();i++)
      if(nnets[i]->exportdata(weights[i]) == false){
	weights.clear();
	return false;
      }
    
    return true;
  }
  
  template <typename T>
  bool bayesian_nnetwork<T>::importSamples(const whiteice::nnetwork<T>& nn,
					   const std::vector< math::vertex<T> >& weights,
					   const std::vector< math::vertex<T> >& bndatas)
  {
    if(weights.size() <= 0) return false;
    if(weights.size() != bndatas.size()) return false;
    
    std::vector< nnetwork<T>* > nnnets;
    nnnets.resize(weights.size());

    for(unsigned int i=0;i<nnnets.size();i++){
      nnnets[i] = new nnetwork<T>(nn); // FIXME handle alloc that FAILs..

      nnnets[i]->setBatchNorm(true);
      
      if(nnnets[i]->importdata(weights[i]) == false){
	for(unsigned int j=0;j<=i;j++){
	  delete nnnets[i];
	  nnnets[i] = NULL;
	}
	
	return false;
      }

      if(nnnets[i]->importBNdata(bndatas[i]) == false){
	for(unsigned int j=0;j<=i;j++){
	  delete nnnets[i];
	  nnnets[i] = NULL;
	}
	
	return false;
      }
      
    }

    // remove old data
    for(unsigned int i=0;i<this->nnets.size();i++){
      if(this->nnets[i]){
	delete this->nnets[i];
	this->nnets[i] = NULL;
      }
    }

    nnets.clear();

    this->nnets = nnnets; // copies new pointers over old data

    return true;
  }

  template <typename T>
  bool bayesian_nnetwork<T>::exportSamples(whiteice::nnetwork<T>& nn, 
					   std::vector< math::vertex<T> >& weights,
					   std::vector< math::vertex<T> >& bndatas,
					   int latestN) const
  {
    if(nnets.size() <= 0) return false;
    if(latestN > (signed)nnets.size()) return false;
    if(latestN <= 0) latestN = nnets.size();

    nn = (*nnets[0]);

    weights.resize(latestN);
    bndatas.resize(latestN);

    for(unsigned int i=(nnets.size() - latestN);i<nnets.size();i++){
      if(nnets[i]->exportdata(weights[i]) == false){
	weights.clear();
	bndatas.clear();
	return false;
      }

      if(nnets[i]->exportBNdata(bndatas[i]) == false){
	weights.clear();
	bndatas.clear();
	return false;
      }
    }
    
    return true;
  }

  
  template <typename T>
  bool bayesian_nnetwork<T>::exportNetworks(std::vector< whiteice::nnetwork<T>* >& nnlist,
					    int latestN) const
  {
    if(nnets.size() <= 0) return false;
    if(latestN > (signed)nnets.size()) return false;
    if(latestN <= 0) latestN = nnets.size();

    nnlist.resize(latestN);

    for(unsigned int i=(nnets.size() - latestN);i<nnets.size();i++){
      nnlist[i] = new whiteice::nnetwork<T>(*nnets[i]);
    }
    
    return true;
  }
  
  
  template <typename T>
  bool bayesian_nnetwork<T>::getArchitecture(std::vector<unsigned int>& arch) const
  {
    if(nnets.size() <= 0) return false;

    nnets[0]->getArchitecture(arch);

    return true;
  }

  template <typename T>
  bool bayesian_nnetwork<T>::editArchitecture(std::vector<unsigned int>& arch,
					      typename nnetwork<T>::nonLinearity nl)
  {
    if(nnets.size() <= 0) return true; // nothing to do..

    std::vector<whiteice::nnetwork<T>*> nets;

    for(unsigned int i=0;i<nnets.size();i++){
      // creates new network with given arch
      whiteice::nnetwork<T>* nn = new nnetwork<T>(arch, nl);
      
      std::vector<unsigned int> oldarch;
      nnets[i]->getArchitecture(oldarch);
      
      for(unsigned int l=0;l<nn->getLayers()&&l<nnets[i]->getLayers();l++)
      {
	if(arch[l] == oldarch[l] && arch[l+1] == oldarch[l+1]){
	  nn->setNonlinearity(l, nnets[i]->getNonlinearity(l));
	  nn->setFrozen(l, true);

	  whiteice::math::matrix<T> W;
	  whiteice::math::vertex<T> b;

	  nnets[i]->getWeights(W, l);
	  nnets[i]->getBias(b, l);

	  nn->setWeights(W, l);
	  nn->setBias(b, l);
	}
	else{
	  break; // archtecture stops matching the new one
	}
      }

      nn->randomize(); // sets (new) weights to random values
      nets.push_back(nn);
    }

    
    if(nets.size() == nnets.size()){
      for(auto p : nnets) delete p;
      nnets = nets;
      return true;
    }
    else{
      for(auto p : nets) delete p;
      return false;
    }
      
  }

  template <typename T>
  bayesian_nnetwork<T>* bayesian_nnetwork<T>::createSubnet(const unsigned int fromLayer)
  {
    std::vector<whiteice::nnetwork<T>*> nets;

    for(unsigned int i=0;i<nnets.size();i++){
      // creates new network 
      whiteice::nnetwork<T>* nn = nnets[i]->createSubnet(fromLayer);      
      nets.push_back(nn);
    }

    auto bn = new bayesian_nnetwork<T>();

    bn->nnets = nets;

    return bn;
  }

  
  template <typename T>
  bool bayesian_nnetwork<T>::injectSubnet(const unsigned int fromLayer,
					  bayesian_nnetwork<T>* bnn)
  {
    if(nnets.size() == 0 || bnn->nnets.size() == 0) return false;

    if(nnets.size() <= 0) return true; // nothing to do

    if(nnets[0]->getLayers() != bnn->nnets[0]->getLayers() + fromLayer)
      return false;

    std::vector<whiteice::nnetwork<T>*> nets;

    for(unsigned int i=0;i<bnn->nnets.size();i++){
      nets.push_back(new nnetwork<T>(*nnets[rand() % nnets.size()]));
      if(nets[i]->injectSubnet(fromLayer, bnn->nnets[i]) == false)
	return false;
    }

    // delete old nnets data
    for(unsigned int i=0;i<nnets.size();i++)
      delete nnets[i];

    nnets = nets;

    return true;
  }

  
  template <typename T>
  bool bayesian_nnetwork<T>::setNonlinearity(typename nnetwork<T>::nonLinearity nl){
    for(unsigned int i=0;i<nnets.size();i++)
      if(nnets[i])
	if(nnets[i]->setNonlinearity(nl) == false)
	  return false;
    
    return true;
  }

  template <typename T>
  void bayesian_nnetwork<T>::getNonlinearity(std::vector< typename nnetwork<T>::nonLinearity >& nl)
  { 
    if(nnets.size() > 0)
      if(nnets[0])
	return nnets[0]->getNonlinearity(nl);

    assert(0); // we should never reach here..
  }
  
  
  template <typename T>
  bool bayesian_nnetwork<T>::downsample(unsigned int N)
  {
    if(N == 0) return false;
    if(N >= nnets.size())
      return true;
    
    std::set<unsigned int> samples;
    for(unsigned int i=0;i<nnets.size();i++)
      samples.insert(i);
    
    std::vector< nnetwork<T>* > nn;
    
    while(nn.size() < N){ 
      auto iter = samples.begin();
      int i = rand() % samples.size();
      while(i > 0){ iter++; i--; }
      const int index = *iter;
      
      samples.erase(iter);
      nn.push_back(nnets[index]);
    }
    
    // deletes rest of the nnetworks
    {
      auto iter = samples.begin();
      
      while(iter != samples.end()){
	const int index = (*iter);
	delete nnets[index];
	iter++;
      }
    }
    
    nnets.clear();
    nnets = nn;
    
    return true;
  }
  
#if 1
  // calculates E[f(input,w)] and Var[f(x,w)] for given input
  template <typename T>
  bool bayesian_nnetwork<T>::calculate(const math::vertex<T>& input,
				       math::vertex<T>& mean,
				       math::matrix<T>& covariance,
				       unsigned int SIMULATION_DEPTH, // for recurrent use of nnetworks..
				       int latestN) const
  {
    if(nnets.size() <= 0) return false;
    if(latestN > (signed)nnets.size()) return false;
    if(latestN <= 0) latestN = nnets.size();
    if(SIMULATION_DEPTH == 0) return false;

    const int RDIM = nnets[0]->input_size() - input.size();
    
    if(SIMULATION_DEPTH > 1){
      if(((int)nnets[0]->output_size()) - RDIM <= 0 || RDIM <= 0)
	return false;
    }
    else{
      if(input.size() != nnets[0]->input_size())
	return false;
    }

    const unsigned int DIM = nnets[0]->output_size() - RDIM;
    mean.resize(DIM);
    covariance.resize(DIM,DIM);
    
    mean.zero();
    covariance.zero();

    //if(latestN <= (signed)DIM)
    //  covariance.identity(); // regularizer term for small datasize
    
#pragma omp parallel shared(mean, covariance)
    {
      math::matrix<T> cov;
      math::vertex<T> m;
      
      m.resize(DIM);
      cov.resize(DIM,DIM);
      m.zero();
      cov.zero();
      
      T ninv  = T(1.0f/latestN);

#pragma omp for nowait schedule(auto)
      for(unsigned int i=(nnets.size() - latestN);i<nnets.size();i++){

	const unsigned int index = rng.rand() % nnets.size();
	
	math::vertex<T> in(nnets[0]->input_size());
	math::vertex<T> out(DIM), out_nn(nnets[0]->output_size());
	math::vertex<T> rdim(RDIM);

	in.zero();
	out.zero();
	out_nn.zero();
	rdim.zero();

	in.write_subvertex(input, 0); // writes input section
	
	for(unsigned int d=0;d<SIMULATION_DEPTH;d++){
	  if(SIMULATION_DEPTH > 1){
	    in.write_subvertex(rdim, input.size());
	  }
	  
	  nnets[index]->calculate(in, out_nn); // recurrent calculations if needed
	  
	  if(SIMULATION_DEPTH > 1){
	    out_nn.subvertex(rdim, DIM, rdim.size());
	  }
	}
	
	if(SIMULATION_DEPTH > 1){
	  out_nn.subvertex(out, 0, DIM);
	}
	else{
	  out = out_nn;
	}
	
	m += ninv*out;
	cov += ninv*out.outerproduct();
      }
      
#pragma omp critical
      {
	mean += m;
	covariance += cov;
      }
      
    }

    // should divide by N-1 but we ignore this in order to have a result for N=1
    covariance -= mean.outerproduct(); 

    return true;
  }
#endif

  // calculates E[f(input,w)] for given input
  template <typename T>
  bool bayesian_nnetwork<T>::calculate(const math::vertex<T>& input,
				       math::vertex<T>& mean,
				       unsigned int SIMULATION_DEPTH, // for recurrent use of nnetworks..
				       int latestN) const
  {
    if(nnets.size() <= 0) return false;
    if(latestN > (signed)nnets.size()) return false;
    if(latestN <= 0) latestN = nnets.size();

    const int RDIM = nnets[0]->input_size() - input.size();
    
    if(SIMULATION_DEPTH > 1){
      if(((int)nnets[0]->output_size()) - RDIM <= 0 || RDIM <= 0)
	return false;
    }
    else{
      if(input.size() != nnets[0]->input_size())
	return false;
    }

    const unsigned int DIM = nnets[0]->output_size() - RDIM;
    mean.resize(DIM);
    
    mean.zero();
    
#pragma omp parallel shared(mean)
    {
      math::vertex<T> m;
      
      m.resize(DIM);
      m.zero();
      
      T ninv  = T(1.0f/latestN);

#pragma omp for nowait schedule(auto)
      for(unsigned int i=(nnets.size() - latestN);i<nnets.size();i++){

	const unsigned int index = rng.rand() % nnets.size();
	
	math::vertex<T> in(nnets[0]->input_size());
	math::vertex<T> out(DIM), out_nn(nnets[0]->output_size());
	math::vertex<T> rdim(RDIM);

	in.zero();
	out.zero();
	out_nn.zero();
	rdim.zero();

	in.write_subvertex(input, 0); // writes input section
	
	for(unsigned int d=0;d<SIMULATION_DEPTH;d++){
	  if(SIMULATION_DEPTH > 1){
	    in.write_subvertex(rdim, input.size());
	  }
	  
	  nnets[index]->calculate(in, out_nn); // recurrent calculations if needed
	  
	  if(SIMULATION_DEPTH > 1){
	    out_nn.subvertex(rdim, DIM, rdim.size());
	  }
	}
	
	if(SIMULATION_DEPTH > 1){
	  out_nn.subvertex(out, 0, DIM);
	}
	else{
	  out = out_nn;
	}
	
	m += out;
      }

      m *= ninv;
      
#pragma omp critical
      {
	mean += m;
      }
      
    }

    
    return true;
  }


  template <typename T>
  unsigned int bayesian_nnetwork<T>::outputSize() const 
  {
    if(nnets.size() <= 0) return 0;

    return nnets[0]->output().size();
  }

  template <typename T>
  unsigned int bayesian_nnetwork<T>::inputSize() const 
  {
    if(nnets.size() <= 0) return 0;

    return nnets[0]->input().size();
  }


#define FNN_VERSION_CFGSTR          "FNN_CONFIG_VERSION"
#define FNN_NUMWEIGHTS_CFGSTR       "FNN_NUM_WEIGHTS"
#define FNN_ARCH_CFGSTR             "FNN_ARCH"
#define FNN_WEIGHTS_CFGSTR          "FNN_WEIGHTS%d"  
#define FNN_NONLINEARITY_CFGSTR     "FNN_NONLINEARITY"
#define FNN_FROZEN_CFGSTR           "FNN_FROZEN"
#define FNN_RESIDUAL_CFGSTR         "FNN_RESIDUAL"
#define FNN_BATCH_NORM_CFGSTR       "FNN_BATCHNORM"
#define FNN_BN_DATA_CFGSTR          "FNN_BN_DATA"
  
  // stores and loads bayesian nnetwork to a text file
  // (saves all samples into files)
  template <typename T>
  bool bayesian_nnetwork<T>::load(const std::string& filename) 
  {
    try{
      // whiteice::conffile configuration;
      whiteice::dataset<T> configuration;
      math::vertex<T> data;
      // unsigned int cluster = 0;
      
      std::vector<int> ints;
      std::vector<float> floats;
      std::vector<std::string> strings;
      bool residual = false;
      bool batchnorm = false;
      
      if(configuration.load(filename) == false)
	return false;
      
      int versionid = 0;
      
      // checks version
      {
	//if(!configuration.get(FNN_VERSION_CFGSTR, ints))
	//return false;
	data = configuration.accessName(FNN_VERSION_CFGSTR, 0);
	ints.resize(data.size());
	for(unsigned int i=0;i<data.size();i++){
	  double ad = -1.0;
	  math::convert(ad, data[i]);
	  ints[i] = (int)ad;
	}
	
	
	if(ints.size() != 1)
	  return false;
	
	versionid = ints[0];
	
	ints.clear();
      }

      if(typeid(T) == typeid(whiteice::math::superresolution< whiteice::math::blas_real<float>, whiteice::math::modular<unsigned int> >) ||
	 typeid(T) == typeid(whiteice::math::superresolution< whiteice::math::blas_real<double>, whiteice::math::modular<unsigned int> >)){
	
	if(versionid != (3800 + 1)) // v3.8 datafile (3.8 adds superresolutional numbers which add 1 to version number) (3.7 adds batch norm data)
	  return false;
      }
      else{

	if(versionid != (3800 + 0)) // v3.8 datafile (3.8 adds superresolutional numbers which add 10.000 to version number) (3.7 adds batch norm data)
	  return false;
      }
      
      std::vector<unsigned int> arch;
      
      // gets architecture
      {
	data = configuration.accessName(FNN_ARCH_CFGSTR, 0);
	ints.resize(data.size());
	for(unsigned int i=0;i<data.size();i++){
	  double ad = -1.0;
	  math::convert(ad, data[i]);
	  ints[i] = (int)ad;
	}
	
	if(ints.size() < 2)
	  return false;
	
	arch.resize(ints.size());
	
	for(unsigned int i=0;i<ints.size();i++){
	  if(ints[i] <= 0) return false;
	  arch[i] = (unsigned int)ints[i];
	}
      }

      // gets nonlinearity of nnetworks
      std::vector< typename whiteice::nnetwork<T>::nonLinearity > nl;
      
      {
	nl.resize(arch.size()-1);
	for(unsigned int l=0;l<nl.size();l++){
	  nl[l] = whiteice::nnetwork<T>::tanh;
	}

	// nl[nl.size()-1] = whiteice::nnetwork<T>::pureLinear;
	
	data = configuration.accessName(FNN_NONLINEARITY_CFGSTR, 0);
	ints.resize(data.size());
	
	for(unsigned int i=0;i<data.size();i++){
	  double ad = -1.0;
	  math::convert(ad, data[i]);
	  ints[i] = (int)ad;
	}
	
	if(ints.size() != nl.size())
	  return false;

	for(unsigned int l=0;l<nl.size();l++){
	  if(ints[l] == 0)
	    nl[l] = whiteice::nnetwork<T>::sigmoid;
	  else if(ints[l] == 1)
	    nl[l] = whiteice::nnetwork<T>::stochasticSigmoid;
	  else if(ints[l] == 2)
	    nl[l] = whiteice::nnetwork<T>::halfLinear;
	  else if(ints[l] == 3)
	    nl[l] = whiteice::nnetwork<T>::pureLinear;
	  else if(ints[l] == 4)
	    nl[l] = whiteice::nnetwork<T>::tanh;
	  else if(ints[l] == 5)
	    nl[l] = whiteice::nnetwork<T>::rectifier;
	  else if(ints[l] == 6)
	    nl[l] = whiteice::nnetwork<T>::softmax;
	  else if(ints[l] == 7)
	    nl[l] = whiteice::nnetwork<T>::tanh10;
	  else if(ints[l] == 8)
	    nl[l] = whiteice::nnetwork<T>::hermite;
	  else
	    return false; // bad data
	}
      }


      // gets frozen layers of nnetwork
      std::vector< bool > frozen;
      {
	frozen.resize(arch.size()-1);
	for(unsigned int l=0;l<frozen.size();l++){
	  frozen[l] = false;
	}
	
	data = configuration.accessName(FNN_FROZEN_CFGSTR, 0);
	ints.resize(data.size());
	
	for(unsigned int i=0;i<data.size();i++){
	  double ad = -1.0;
	  math::convert(ad, data[i]);
	  ints[i] = (int)ad;
	}
	
	if(ints.size() != frozen.size())
	  return false;

	for(unsigned int l=0;l<nl.size();l++){
	  if(ints[l] == 0)
	    frozen[l] = false;
	  else
	    frozen[l] = true;
	}
      }


      // gets residual information for nnetwork
      {
	data = configuration.accessName(FNN_RESIDUAL_CFGSTR, 0);

	if(data.size() != 1) return false;
	if(data[0] == T(0.0f)) residual = false;
	else if(data[0] == T(1.0f)) residual = true;
	else return false;
      }


      // gets batch norm information for nnetwork
      {
	data = configuration.accessName(FNN_BATCH_NORM_CFGSTR, 0);

	if(data.size() != 1) return false;
	if(data[0] == T(0.0f)) batchnorm = false;
	else if(data[0] == T(1.0f)) batchnorm = true;
	else return false;
      }

      
      
      // reads number of samples information
      int numberOfSamples = 0;
      
      {
	data = configuration.accessName(FNN_NUMWEIGHTS_CFGSTR, 0);
	ints.resize(data.size());
	for(unsigned int i=0;i<data.size();i++){
	  double ad = 0.0;
	  math::convert(ad, data[i]);
	  ints[i] = (int)ad;
	}
	
	if(ints.size() != 1)
	  return false;
	
	numberOfSamples = ints[0];
	
	ints.clear();
      }
      
      if(numberOfSamples <= 0)
	return false;
      
      
      std::vector< nnetwork<T>* > nets;
      
      nets.resize(numberOfSamples);
      
      // weights: we just import a big vertex vector
      for(unsigned int index=0;index<nets.size();index++)
      {
	nets[index] = new nnetwork<T>(arch);
	if(nets[index]->setNonlinearity(nl) == false)
	  return false;
	
	if(nets[index]->setFrozen(frozen) == false)
	  return false;

	nets[index]->setResidual(residual);
	
	math::vertex<T> w;
	
	w = configuration.accessName(FNN_WEIGHTS_CFGSTR, index);
	
	if(nets[index]->importdata(w) == false){
	  for(unsigned int j=0;j<=index;j++)
	    delete nets[j];
	  return false;
	}

	if(batchnorm){
	  nets[index]->setBatchNorm(true);
	  
	  w = configuration.accessName(FNN_BN_DATA_CFGSTR, index);
	  
	  if(nets[index]->importBNdata(w) == false){
	    for(unsigned int j=0;j<=index;j++)
	      delete nets[j];
	    return false;
	  }
	}
	
	floats.clear();
      }
      
      for(unsigned int i=0;i<nnets.size();i++)
	delete nnets[i]; // deletes old networks
      
      nnets = nets; // saves the loaded nnetworks
      
      return true;
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception "
		<< "File: " << __FILE__ << " "
		<< "Line: " << __LINE__ << " "
		<< e.what() << std::endl;
      
      return false;
    }
  }

  
  
  template <typename T>
  bool bayesian_nnetwork<T>::save(const std::string& filename) const 
  {
    try{
      if(nnets.size() <= 0) return false;

      // whiteice::conffile configuration;
      whiteice::dataset<T> configuration;
      math::vertex<T> data;

      std::vector<int> ints;
      std::vector<float> floats;
      std::vector<std::string> strings;
      

      // writes version information
      {
	// version number = integer/1000
	
	if(typeid(T) == typeid(whiteice::math::superresolution< whiteice::math::blas_real<float>, whiteice::math::modular<unsigned int> >) ||
	   typeid(T) == typeid(whiteice::math::superresolution< whiteice::math::blas_real<double>, whiteice::math::modular<unsigned int> >)){

	  // v3.8 datafile (3.8 adds superresolutional numbers which adds 1 to version number) (3.7 adds batch norm data)
	  ints.push_back(3800 + 1);
	  
	}
	else{
	  // v3.8 datafile (3.8 adds superresolutional numbers which adds 1 to version number) (3.7 adds batch norm data)
	  ints.push_back(3800 + 0);
	}
	      
	
	

	configuration.createCluster(FNN_VERSION_CFGSTR, ints.size());
	data.resize(ints.size());
	for(unsigned int i=0;i<ints.size();i++){
	  whiteice::math::convert(data[i], (double)ints[i]);
	  //data[i] = ints[i];
	}
	configuration.add(configuration.getCluster(FNN_VERSION_CFGSTR), data);
	
	//if(!configuration.set(FNN_VERSION_CFGSTR, ints))
	//  return false;
	
	ints.clear();
      }

      
      std::vector<unsigned int> arch;

      nnets[0]->getArchitecture(arch);
      
      // writes architecture information
      {
	for(unsigned int i=0;i<arch.size();i++)
	  ints.push_back(arch[i]);

	configuration.createCluster(FNN_ARCH_CFGSTR, ints.size());
	data.resize(ints.size());
	for(unsigned int i=0;i<ints.size();i++){
	  whiteice::math::convert(data[i], (double)ints[i]);
	  //data[i] = ints[i];
	}
	configuration.add(configuration.getCluster(FNN_ARCH_CFGSTR), data);
	
	//if(!configuration.set(FNN_ARCH_CFGSTR, ints))
	//  return false;
	
	ints.clear();
      }

      // writes number of samples information
      {
	ints.push_back(nnets.size());

	configuration.createCluster(FNN_NUMWEIGHTS_CFGSTR, ints.size());
	data.resize(ints.size());
	for(unsigned int i=0;i<ints.size();i++){
	  whiteice::math::convert(data[i], (double)ints[i]);
	  //data[i] = ints[i];
	}
	configuration.add(configuration.getCluster(FNN_NUMWEIGHTS_CFGSTR), data);
	
	ints.clear();
      }

      
      // writes non-linearity information (assumes all nets have same nonlinearities!)
      {
	std::vector< typename whiteice::nnetwork<T>::nonLinearity > nl;
	
	if(nnets.size() > 0){
	  nnets[0]->getNonlinearity(nl);

	  for(unsigned int l=0;l<nl.size();l++){

	    if(nl[l] == whiteice::nnetwork<T>::sigmoid)
	      ints.push_back(0);
	    else if(nl[l] == whiteice::nnetwork<T>::stochasticSigmoid)
	      ints.push_back(1);
	    else if(nl[l] == whiteice::nnetwork<T>::halfLinear)
	      ints.push_back(2);
	    else if(nl[l] == whiteice::nnetwork<T>::pureLinear)
	      ints.push_back(3);
	    else if(nl[l] == whiteice::nnetwork<T>::tanh)
	      ints.push_back(4);
	    else if(nl[l] == whiteice::nnetwork<T>::rectifier)
	      ints.push_back(5);
	    else if(nl[l] == whiteice::nnetwork<T>::softmax)
	      ints.push_back(6);
	    else if(nl[l] == whiteice::nnetwork<T>::tanh10)
	      ints.push_back(7);
	    else if(nl[l] == whiteice::nnetwork<T>::hermite)
	      ints.push_back(8);
	    else
	      return false;
	  }
	}
	else return false;

	configuration.createCluster(FNN_NONLINEARITY_CFGSTR, ints.size());
	data.resize(ints.size());
	for(unsigned int i=0;i<ints.size();i++){
	  whiteice::math::convert(data[i], (double)ints[i]);
	  //data[i] = ints[i];
	}
	configuration.add(configuration.getCluster(FNN_NONLINEARITY_CFGSTR), data);
	
	ints.clear();
      }

      // writes layer frozen information (assumes all nets have same nonlinearities!)
      {
	std::vector< bool > frozen;
	
	if(nnets.size() > 0){
	  nnets[0]->getFrozen(frozen);

	  for(unsigned int l=0;l<frozen.size();l++){
	    if(frozen[l] == false)
	      ints.push_back(0);
	    else
	      ints.push_back(1);
	  }
	}
	else return false;

	configuration.createCluster(FNN_FROZEN_CFGSTR, ints.size());
	data.resize(ints.size());
	for(unsigned int i=0;i<ints.size();i++){
	  whiteice::math::convert(data[i], (double)ints[i]);
	  //data[i] = ints[i];
	}
	configuration.add(configuration.getCluster(FNN_FROZEN_CFGSTR), data);
	
	ints.clear();
      }

      // weights: we just convert everything to a big vertex vector and write it
      math::vertex<T> w;
      if(nnets[0]->exportdata(w) == false)
    	  return false;

      configuration.createCluster(FNN_WEIGHTS_CFGSTR, w.size());

      for(unsigned int index=0;index<nnets.size();index++)
      {
	// char buffer[80];
	math::vertex<T> w;
	
	if(nnets[index]->exportdata(w) == false)
	  return false;
	
	configuration.add(configuration.getCluster(FNN_WEIGHTS_CFGSTR), w);

      }      

      // residual neural network information
      {
	configuration.createCluster(FNN_RESIDUAL_CFGSTR, 1);
	
	data.resize(1);
	if(nnets[0]->getResidual()){
	  data[0] = T(1.0f);
	}
	else{
	  data[0] = T(0.0f);
	}

	configuration.add(configuration.getCluster(FNN_RESIDUAL_CFGSTR), data);
      }

      // batch norm neural network information
      {
	configuration.createCluster(FNN_BATCH_NORM_CFGSTR, 1);
	
	data.resize(1);
	if(nnets[0]->getBatchNorm()){
	  data[0] = T(1.0f);
	}
	else{
	  data[0] = T(0.0f);
	}

	configuration.add(configuration.getCluster(FNN_BATCH_NORM_CFGSTR), data);
      }
      
      
      if(nnets[0]->getBatchNorm()){
	// weights: we just convert everything to a big vertex vector and write it
	math::vertex<T> w;
	if(nnets[0]->exportBNdata(w) == false)
    	  return false;
	
	configuration.createCluster(FNN_BN_DATA_CFGSTR, w.size());
	
	for(unsigned int index=0;index<nnets.size();index++)
	{
	  // char buffer[80];
	  math::vertex<T> w;
	  
	  if(nnets[index]->exportBNdata(w) == false)
	    return false;
	  
	  configuration.add(configuration.getCluster(FNN_BN_DATA_CFGSTR), w);
	  
	}
	
      }
      else{
	configuration.createCluster(FNN_BN_DATA_CFGSTR, 1);

	math::vertex<T> w;
	w.resize(1);
	w[0] = T(0.0f);

	configuration.add(configuration.getCluster(FNN_BN_DATA_CFGSTR), w);
      }
      
      return configuration.save(filename);
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception "
		<< "File: " << __FILE__ << " "
		<< "Line: " << __LINE__ << " "
		<< e.what() << std::endl;
      
      return false;
    }
    
  }

  
  template class bayesian_nnetwork< math::blas_real<float> >;
  template class bayesian_nnetwork< math::blas_real<double> >;
  template class bayesian_nnetwork< math::blas_complex<float> >;
  template class bayesian_nnetwork< math::blas_complex<double> >;

  template class bayesian_nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >;
  template class bayesian_nnetwork< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >;
  template class bayesian_nnetwork< math::superresolution< math::blas_complex<float>, math::modular<unsigned int> > >;
  template class bayesian_nnetwork< math::superresolution< math::blas_complex<double>, math::modular<unsigned int> > >;
  
};
