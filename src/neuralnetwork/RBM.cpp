
#include "RBM.h"
#include "norms.h"
#include "RNG.h"

namespace whiteice
{
  
  // creates 1x1 network, used to load some useful network later
  template <typename T>
  RBM<T>::RBM()
  {
    v.resize(2);
    h.resize(2);
    W.resize(2,2);

    initializeWeights();
  }
  
  template <typename T>
  RBM<T>::RBM(const RBM<T>& rbm)
  {
    this->v = rbm.v;
    this->h = rbm.h;
    this->W = rbm.W;
  }
  
  
  // creates 2-layer: V * H network
  template <typename T>
  RBM<T>::RBM(unsigned int visible, unsigned int hidden)
    
  {
    if(visible == 0 || hidden == 0)
      throw std::invalid_argument("invalid network architecture");
    
    // the last term is always constant: 1 (one)
    v.resize(visible + 1);
    h.resize(hidden + 1);
    W.resize(hidden + 1, visible + 1);
    
    initializeWeights();    
  }
  
  template <typename T>
  RBM<T>::~RBM()
  {
    // nothing to do
  }
  
  template <typename T>  
  RBM<T>& RBM<T>::operator=(const RBM<T>& rbm)
  {
    this->v = rbm.v;
    this->h = rbm.h;
    this->W = rbm.W;
    
    return (*this);
  }
  
  
  template <typename T>
  bool RBM<T>::resize(unsigned int visible, unsigned int hidden)
  {
    if(visible == 0 || hidden == 0)
      return false;
    
    // the last term is always constant: 1 (one)
    v.resize(visible + 1);
    h.resize(hidden + 1);
    W.resize(hidden + 1, visible + 1);
    
    initializeWeights();    
    
    return true;
  }
  
  
  template <typename T>
  math::vertex<T> RBM<T>::getVisible() const
  {
    math::vertex<T> t = v;
    t.resize(v.size() - 1);
    
    return t;
  }
  
  template <typename T>
  bool RBM<T>::setVisible(const math::vertex<T>& v)
  {
    if(this->v.size() != (v.size() + 1))
      return false;
    
    for(unsigned int j=0;j<v.size();j++)
      this->v[j] = v[j];
    
    this->v[v.size()] = T(1.0);
    
    return true;
  }
  
  template <typename T>
  math::vertex<T> RBM<T>::getHidden() const
  {
    math::vertex<T> t = h;
    t.resize(h.size() - 1);
    
    return t;
  }
  
  
  template <typename T>
  bool RBM<T>::setHidden(const math::vertex<T>& h)
  {
    if(this->h.size() != (h.size() + 1))
      return false;
    
    for(unsigned int j=0;j<h.size();j++)
      this->h[j] = h[j];
    
    this->h[h.size()] = T(1.0);
    
    return true;
  }
  
  
  template <typename T>
  bool RBM<T>::reconstructData(unsigned int iters)
  {
    if(iters == 0) return false;
    
    math::matrix<T> Wt = W;
    Wt.transpose();
    
    while(iters > 0){
      h = W*v;
      
      // 1. hidden units: calculates sigma(a_j)
      for(unsigned int j=0;j<(h.size()-0);j++){
	T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
	T r = T(whiteice::rng.uniformf());
	
	if(aj > r) h[j] = T(1.0); // discretization step
	else       h[j] = T(0.0);
      }
      
      iters--;
      if(iters <= 0) return true;
    
      v = Wt*h;
      
      // 1. visible units: calculates sigma(a_j)
      for(unsigned int j=0;j<(v.size()-0);j++){
	T aj = T(1.0)/(T(1.0) + math::exp(-v[j]));
	T r = T(whiteice::rng.uniformf());
	
	if(aj > r) v[j] = T(1.0); // discretization step
	else       v[j] = T(0.0);
      }
      
      iters--;
      if(iters <= 0) return true;
    }
    
    return true;
  }
  
  
  template <typename T>
  bool RBM<T>::reconstructDataHidden(unsigned int iters)
  {
    if(iters == 0) return false;
    
    math::matrix<T> Wt = W;
    Wt.transpose();
    
    while(iters > 0){
      v = Wt*h;
      
      // 1. visible units: calculates sigma(a_j)
      for(unsigned int j=0;j<(v.size()-0);j++){
	T aj = T(1.0)/(T(1.0) + math::exp(-v[j]));
	T r = T(whiteice::rng.uniformf());
	
	if(aj > r) v[j] = T(1.0); // discretization step
	else       v[j] = T(0.0);
      }
      
      iters--;
      if(iters <= 0) return true;
      
      h = W*v;
      
      // 1. hidden units: calculates sigma(a_j)
      for(unsigned int j=0;j<(h.size()-0);j++){
	T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
	T r = T(whiteice::rng.uniformf());
	
	if(aj > r) h[j] = T(1.0); // discretization step
	else       h[j] = T(0.0);
      }
      
      iters--;
      if(iters <= 0) return true;
    

    }
    
    return true;
  }
  
  
  template <typename T>
  math::matrix<T> RBM<T>::getWeights() const
  {
    return W;
  }
  
  template <typename T>
  bool RBM<T>::initializeWeights(){ // initialize weights to small values
    
    for(unsigned int j=0;j<W.ysize();j++)
      for(unsigned int i=0;i<W.xsize();i++)
	W(j,i) = randomValue();
    
    v[v.size()-1] = T(1.0);
    h[h.size()-1] = T(1.0);

    return true;
  }
  
  
  template <typename T>
  T RBM<T>::learnWeights(const std::vector< math::vertex<T> >& samples)
  {
    const unsigned int L = 10; // CD-10 algorithm
    const T lambda = T(0.01);
    
    math::matrix<T> P(W);
    math::matrix<T> N(W);
    math::matrix<T> originalW = W;
    math::matrix<T> Wt = W;
    Wt.transpose();
    
    
    for(unsigned int i=0;i<samples.size();i++){
      const unsigned int index = whiteice::rng.rand() % samples.size();
      
      if(samples[index].size() + 1 != v.size())
	continue; // silently ignores bad data.. (throw exception instead?)

      // v = samples[index];
      v.importData(&(samples[index][0]), samples[index].size());
      v[v.size()-1] = T(1.0);
      
      
      {
	h = W*v;
	
	// 1. hidden units: calculates sigma(a_j)
	for(unsigned int j=0;j<(h.size()-0);j++){
	  T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
	  T r = T(whiteice::rng.uniformf());
	  
	  if(aj > r) h[j] = T(1.0); // discretization step
	  else       h[j] = T(0.0);
	}
	
	for(unsigned int y=0;y<h.size();y++)
	  for(unsigned int x=0;x<v.size();x++)
	    P(y,x) = h[y]*v[x];

	
	for(unsigned int l=0;l<L;l++){
	  v = Wt*h;
	  
	  // 1. visible units: calculates sigma(a_j)
	  for(unsigned int j=0;j<(v.size()-0);j++){
	    T aj = T(1.0)/(T(1.0) + math::exp(-v[j]));
	    T r = T(whiteice::rng.uniformf());
	    
	    if(aj > r) v[j] = T(1.0); // discretization step
	    else       v[j] = T(0.0);
	  }
	  
	  // v[v.size()-1] = T(1.0); // DO NOT FORCE THE HIDDEN INPUT TO ONE DURING STIM
	  
	  
	  h = W*v;
	  
	  // 2. hidden units: calculates sigma(a_j)
	  for(unsigned int j=0;j<(h.size()-0);j++){
	    T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
	    T r = T(whiteice::rng.uniformf());
	    
	    if(aj > r) h[j] = T(1.0); // discretization step
	    else       h[j] = T(0.0);
	  }
	  
	}
	
	
	for(unsigned int y=0;y<h.size();y++)
	  for(unsigned int x=0;x<v.size();x++)
	    N(y,x) = h[y]*v[x];
      }
      
      // updates weights according to CD rule
      W += lambda*(P - N);
      Wt = W;
      Wt.transpose();
    }
    
    
    originalW -= W;
    
    return (math::frobenius_norm(originalW)/originalW.size());
  }
  
  ////////////////////////////////////////////////////////////
  
#define RBM_VERSION_CFGSTR          "RBM_CONFIG_VERSION"
#define RBM_ARCH_CFGSTR             "RBM_ARCH"
#define RBM_WEIGHTS_CFGSTR          "RBM_WEIGHTS"
  
  template <typename T>
  bool RBM<T>::load(const std::string& filename) 
  {
    try{
      whiteice::conffile configuration;
      
      std::vector<std::string> strings;
      std::vector<float> floats;
      std::vector<int> ints;
      
      if(!configuration.load(filename))
	return false;
      
      // checks version
      {
	int versionid = 0;
	
	if(!configuration.get(RBM_VERSION_CFGSTR, ints))
	  return false;
	
	if(ints.size() != 1)
	  return false;
	
	versionid = ints[0];
	
	ints.clear();
	
	if(versionid != 1000) // v1.0 datafile
	  return false;
      }
      
      // loads architecture
      std::vector<int> arch;
      {
	if(!configuration.get(RBM_ARCH_CFGSTR,ints))
	  return false;
	
	if(ints.size() < 2)
	  return false;
	
	for(unsigned int i=0;i<ints.size();i++)
	  if(ints[i] <= 0) return false;
	
	arch = ints;
      }
      
      // tries to load weight matrix V
      math::matrix<T> V;
      {
	if(!configuration.get(RBM_WEIGHTS_CFGSTR, floats))
	  return false;
	
	if(floats.size() != (unsigned int)(arch[0]*arch[1]))
	  return false;
	
	V.resize(arch[1], arch[0]);
	
	for(unsigned int i=0;i<(unsigned int)(arch[1]*arch[0]);i++)
	  V[i] = T(floats[i]);
      }
      
      
      // if everything went ok then activate the changes
      v.resize(arch[0]);
      h.resize(arch[1]);
      W = V;
      
      return true;
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception: "
		<< "File: " << __FILE__ << " "
		<< "Line: " << __LINE__ << " "
		<< e.what() << std::endl;
      return false;
    }
  }
  
  template <typename T>
  bool RBM<T>::save(const std::string& filename) const 
  {
    try{
      whiteice::conffile configuration;
      
      std::vector<std::string> strings;
      std::vector<float> floats;
      std::vector<int> ints;
      
      // sets version
      {
	int versionid = 1000; // v1.0 datafile
	
	ints.clear();
	ints.push_back(versionid);
	
	if(!configuration.set(RBM_VERSION_CFGSTR, ints))
	  return false;
	
	ints.clear();
      }
      
      // sets architecture
      {
	ints.clear();
	ints.push_back(v.size());
	ints.push_back(h.size());
	
	if(!configuration.set(RBM_ARCH_CFGSTR,ints))
	  return false;
	
	ints.clear();
      }
      
      // sets weight matrix W
      {
	floats.clear();
	floats.resize(v.size()*h.size());

	for(unsigned int i=0;i<(unsigned int)(v.size()*h.size());i++){
	  float f;
	  if(math::convert(f, W[i]) == false)
	    return false; // cannot convert to floats meaningfully => failure in save
	  floats[i] = f;
	}
	
	if(!configuration.set(RBM_WEIGHTS_CFGSTR, floats))
	  return false;
      }
      
      // if everything went ok saves the conf file to disk
      if(!configuration.save(filename))
	return false;
      
      return true;
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception: "
		<< "File: " << __FILE__ << " "
		<< "Line: " << __LINE__ << " "
		<< e.what() << std::endl;
      return false;
    }
  }
  
  
  template <typename T>
  T RBM<T>::randomValue()
  {
    T r = T(whiteice::rng.uniformf());
    
    r = T(0.02)*r - T(0.01);
    
    return r;
  }
  
  
  
  
  template class RBM< float >;
  template class RBM< double >;  
  template class RBM< math::blas_real<float> >;
  template class RBM< math::blas_real<double> >;
  
};
