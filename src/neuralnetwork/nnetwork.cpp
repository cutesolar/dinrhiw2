// TODO  optimize, nnetwork takes the most of the time in computations
// FIXED jacobian() routines again use OpenMP for loops which should give 2% performance increase(??)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <exception>
#include <stdexcept>
#include <typeinfo>

#include "nnetwork.h"
#include "dataset.h"
#include "dinrhiw_blas.h"
#include "Log.h"



namespace whiteice
{
  
  
  template <typename T>
  nnetwork<T>::nnetwork()
  {
    arch.resize(2);    
    
    arch[0] = 1;
    arch[1] = 1;

    maxwidth = 0;
    size = 0;
    for(unsigned int i=0;i<arch.size();i++){
      if(i > 0)
	size += (arch[i-1] + 1)*arch[i];
      
      if(arch[i] > maxwidth)
	maxwidth = arch[i];
    }

    W.resize(arch.size()-1);
    b.resize(arch.size()-1);

    for(unsigned int i=1;i<arch.size();i++){
      W[i-1].resize(arch[i], arch[i-1]);
      b[i-1].resize(arch[i]);
    }

    // there are arch.size()-1 layers in our network
    // which are all optimized as the default
    frozen.resize(arch.size()-1);
    for(unsigned int i=0;i<frozen.size();i++)
      frozen[i] = false;

    nonlinearity.resize(arch.size()-1);
    for(unsigned int i=0;i<nonlinearity.size();i++)
      nonlinearity[i] = rectifier;
    
    nonlinearity[nonlinearity.size()-1] = pureLinear;

    
    inputValues.resize(1);
    outputValues.resize(1);

    retain_probability = T(1.0);

    // residual = false;
    residual = true; // ENABLES residual neural networks as the default.

    batchnorm = false;

    randomize();
  }
  
  
  template <typename T>
  nnetwork<T>::nnetwork(const nnetwork<T>& nn)
  {
    inputValues = nn.inputValues;
    outputValues = nn.outputValues;

    hasValidBPData = nn.hasValidBPData;
    maxwidth = nn.maxwidth;
    size = nn.size;

    arch   = nn.arch;
    bpdata = nn.bpdata;
    W      = nn.W;
    b      = nn.b;
    bn_mu  = nn.bn_mu;
    bn_sigma = nn.bn_sigma;
    
    nonlinearity = nn.nonlinearity;
    frozen = nn.frozen;
    retain_probability = nn.retain_probability;
    dropout = nn.dropout;
    residual = nn.residual;
    batchnorm = nn.batchnorm;
    
  }
  
  
  template <typename T>
  nnetwork<T>::nnetwork(const std::vector<unsigned int>& nnarch,
			const typename nnetwork<T>::nonLinearity nl)
    
  {
    if(nnarch.size() < 2)
      throw std::invalid_argument("invalid network architecture");

    maxwidth = 0;
    
    for(unsigned int i=0;i<nnarch.size();i++){
      if(nnarch[i] <= 0)
	throw std::invalid_argument("invalid network architecture");
      if(nnarch[i] > maxwidth)
	maxwidth = nnarch[i];
    }

    // sets up architecture
    arch = nnarch;

    nonlinearity.resize(arch.size()-1);
    for(unsigned int i=0;i<nonlinearity.size();i++)
      nonlinearity[i] = nl;

    // HERE WE ALWAYS SET LAST LAYERS NONLINEARITY TO BE LINEAR FOR NOW..
    nonlinearity[nonlinearity.size()-1] = pureLinear; 
    
    W.resize(arch.size()-1);
    b.resize(arch.size()-1);

    size = 0;
    
    for(unsigned int i=1;i<arch.size();i++){
      W[i-1].resize(arch[i], arch[i-1]);
      b[i-1].resize(arch[i]);

      size += arch[i]*(1 +arch[i-1]);
    }
    
    inputValues.resize(arch[0]);
    outputValues.resize(arch[arch.size()-1]);
    
    // there are arch.size()-1 layers in our network
    // which are all optimized as the default
    frozen.resize(arch.size()-1);
    for(unsigned int i=0;i<frozen.size();i++)
      frozen[i] = false;

    hasValidBPData = false;

    retain_probability = T(1.0);

    // residual = false;
    residual = true; // ENABLES residual neural networks as the default.

    batchnorm = false;

    randomize();
  }
  
  
  template <typename T>
  nnetwork<T>::~nnetwork()
  {
    
  }

  
  template <typename T>
  nnetwork<T>& nnetwork<T>::operator=(const nnetwork<T>& nn)
  {
    hasValidBPData = nn.hasValidBPData;
    arch = nn.arch;
    maxwidth = nn.maxwidth;
    size = nn.size;

    nonlinearity = nn.nonlinearity;
    frozen = nn.frozen;
    retain_probability = nn.retain_probability;
    dropout = nn.dropout;
    residual = nn.residual;
    batchnorm = nn.batchnorm;
    
    W      = nn.W;
    b      = nn.b;
    bn_mu  = nn.bn_mu;
    bn_sigma = nn.bn_sigma;
    bpdata = nn.bpdata;

    inputValues = nn.inputValues;
    outputValues = nn.outputValues;
    
    samples = nn.samples;

    return (*this);
  }

  template <typename T>
  void nnetwork<T>::printInfo() const
  {
    // prints nnetwork information (mostly for debugging purposes)

    printf("NETWORK LAYOUT (%d): \n", getLayers());

    if(residual)
      printf("Residual neural network (skip every 2 layers)\n");

    for(unsigned int l=0;l<getLayers();l++){
      bool frozen = this->getFrozen(l);
      unsigned int nl = (unsigned int)this->getNonlinearity(l);
      unsigned int inputsize = this->getInputs(l);
      unsigned int width = this->getNeurons(l);

      if(frozen)
	printf("%d->%d (F%d) ", inputsize, width, nl);
      else
	printf("%d->%d( %d) ", inputsize, width, nl);
    }

    printf("\n");
    fflush(stdout);


#if 0
    printf("NEURAL NETWORK LAYER WEIGHTS:\n");

    for(unsigned int l=0;l<getLayers();l++){
      
      std::cout << "W(" << l << ") = " << W[l] << std::endl;
      std::cout << "b(" << l << ") = " << b[l] << std::endl;
    }
#endif
      
  }

  
  template <typename T>
  void nnetwork<T>::diagnosticsInfo() const
  {
    char buffer[128];    
    snprintf(buffer, 128, "nnetwork: DIAGNOSTIC/MAXVALUE (%d layers):",
	     getLayers());
    whiteice::logging.info(buffer);


    for(unsigned int l=0;l<getLayers();l++){
      auto maxvalueW = -abs(T(INFINITY));
      auto maxvalueb = -abs(T(INFINITY));
      auto minvalueW = abs(T(+INFINITY));
      auto minvalueb = abs(T(+INFINITY));

      const math::matrix<T>& W = this->W[l];
      const math::vertex<T>& b = this->b[l];
      
      for(unsigned int i=0;i<b.size();i++){
	if(maxvalueb < abs(b[i]))
	  maxvalueb = abs(b[i]);
	if(minvalueb > abs(b[i]))
	  minvalueb = abs(b[i]);
      }

      for(unsigned int j=0;j<W.ysize();j++){
	for(unsigned int i=0;i<W.xsize();i++){
	  if(maxvalueW < abs(W(j, i)))
	    maxvalueW = abs(W(j, i));
	  if(minvalueW > abs(W(j, i)))
	    minvalueW = abs(W(j, i));
	}
      }


      double temp = 0.0;
      whiteice::math::convert(temp, minvalueW);

      double temp2 = 0.0;
      whiteice::math::convert(temp2, maxvalueW);

      double temp3 = 0.0;
      whiteice::math::convert(temp3, minvalueb);

      double temp4 = 0.0;
      whiteice::math::convert(temp4, maxvalueb);

      snprintf(buffer, 128, "nnetwork: LAYER %d/%d %d-%d MIN/MAX ABS-VALUE W=%f/%f b=%f/%f",
	       l+1, getLayers(), getInputs(l), getNeurons(l),
	       temp, temp2, temp3, temp4);
      whiteice::logging.info(buffer);
      
    }
  }

  
  ////////////////////////////////////////////////////////////

  // returns input and output dimensions of neural network
  template <typename T>
  unsigned int nnetwork<T>::input_size() const {
    if(arch.size() > 0) return arch[0];
    else return 0;
  }
  
  template <typename T>
  unsigned int nnetwork<T>::output_size() const {
    unsigned int index = arch.size()-1;
    
    if(arch.size() > 0) return arch[index];
    else return 0;
  }

  template <typename T>
  unsigned int nnetwork<T>::gradient_size() const 
  {
    return size; // number of parameters in neural network
  }


  template <typename T>
  void nnetwork<T>::getArchitecture(std::vector<unsigned int>& nn_arch) const
  {
    nn_arch = this->arch;
  }

  // invalidates all data and essentially creates a new network over previous one
  template <typename T>
  bool nnetwork<T>::setArchitecture(const std::vector<unsigned int>& nnarch,
				    const typename nnetwork<T>::nonLinearity nl)
  {
    if(nnarch.size() <= 1) return false;

    if(arch.size() == nnarch.size()){
      unsigned int same = true;

      for(unsigned int i=0;i<nnarch.size();i++){
	if(nnarch[i] <= 0) return false;
	if(arch[i] != nnarch[i]){
	  same = false;
	  break;
	}
      }

      for(unsigned int i=0;i<nonlinearity.size();i++){
	if(nonlinearity[i] != nl){
	  same = false;
	  break;
	}
      }
	  
      
      if(same == true)
	return true; // we already have the target NN architecture
    }

    ///////////////////////////////////////////////////////////////////////////
    // resets nnetwork<T> parameters
    
    maxwidth = 0;
    
    for(unsigned int i=0;i<nnarch.size();i++){
      if(nnarch[i] > maxwidth)
	maxwidth = nnarch[i];
    }

    // sets up architecture
    arch = nnarch;

    nonlinearity.resize(arch.size()-1);
    for(unsigned int i=0;i<nonlinearity.size();i++)
      nonlinearity[i] = nl;

    // HERE WE ALWAYS SET LAST LAYERS NONLINEARITY TO BE LINEAR FOR NOW..
    nonlinearity[nonlinearity.size()-1] = pureLinear; 
    
    unsigned int memuse = 0;

    W.resize(arch.size()-1);
    b.resize(arch.size()-1);
    
    for(unsigned int i=1;i<arch.size();i++){
      W[i-1].resize(arch[i], arch[i-1]);
      b[i-1].resize(arch[i]);
      
      memuse += (arch[i-1] + 1)*arch[i];
    }
    
    size = memuse;
    
    inputValues.resize(arch[0]);
    outputValues.resize(arch[arch.size()-1]);
    
    // there are arch.size()-1 layers in our network
    // which are all optimized as the default
    frozen.resize(arch.size()-1);
    for(unsigned int i=0;i<frozen.size();i++)
      frozen[i] = false;

    dropout.clear(); // removes dropout data after architecture change
    
    hasValidBPData = false;

    randomize();

    return true;
  }


  /*
   * calculates output for input
   * if gradInfo = true also saves needed
   * information for calculating gradient (bpdata)
   */
  template <typename T>
  bool nnetwork<T>::calculate(bool gradInfo, bool collectSamples)
  {
    // TODO write cblas and cuBLAS optimized version which uses
    // direct accesses to matrix/vertex memory areas
    
    math::vertex<T> state = inputValues;

    if(collectSamples)
      samples.resize(getLayers());

    if(gradInfo){
      bpdata.resize(getLayers()+1);
      bpdata[0] = state; // input value
    }

    math::vertex<T> skipValue;

    if(residual) skipValue = state;

    for(unsigned int l=0;l<getLayers();l++){
      if(collectSamples)
	samples[l].push_back(state);

      const bool residualActive =
	(residual && (l % 2) == 0 && l != 0 && W[l].ysize() == skipValue.size());

      if(residualActive)
	state = W[l]*state + b[l] + skipValue;
      else
	state = W[l]*state + b[l];

      if(gradInfo) // saves local field information
	bpdata[l+1] = state;

      for(unsigned int i=0;i<state.size();i++){
	state[i] = nonlin(state[i], l, i);
      }

      if(residual && (l % 2) == 0 && l != 0)
	skipValue = state;
    }


    outputValues = state;

    hasValidBPData = gradInfo;

    return true;
  }


  // calculates inverse function from output to input [assumes inverse function of
  // non-linearities exists and that linear operators has some kind of
  // pseudo-inverse with regularization)
  template <typename T>
  bool nnetwork<T>::inv_calculate(math::vertex<T>& input,
				  const math::vertex<T>& output,
				  const std::vector< std::vector<bool> >& dropout,
				  bool collectSamples)
  {
    // TODO write cblas and cuBLAS optimized version which uses
    // direct accesses to matrix/vertex memory areas

    // FIXME: repeatedly calculates matrix inverse everytime this function is called!!!
    
    // FORWARD PASS TO LEARN ABOUT RESIDUAL VALUES FOR BACKWARD PASS

    if(input.size() != input_size() || output.size() != output_size())
      return false;

    if(dropout.size() != getLayers()) return false;
    
    math::vertex<T> state = input;

    std::vector< math::vertex<T> > xsamples;
    
    if(collectSamples){
      samples.resize(getLayers());
    }

    math::vertex<T> skipValue;

    if(residual) skipValue = state;

    for(unsigned int l=0;l<getLayers();l++){
      xsamples.push_back(state);

      const bool residualActive =
	(residual && (l % 2) == 0 && l != 0 && W[l].ysize() == skipValue.size());

      if(residualActive)
	state = W[l]*state + b[l] + skipValue;
      else
	state = W[l]*state + b[l];

      for(unsigned int i=0;i<state.size();i++){
	if(dropout[l][i]) state[i] = T(0.0f);
	else state[i] = nonlin_nodropout(state[i], l, i);
      }

      if(residual && (l % 2) == 0 && l != 0)
	skipValue = state;
    }

    // BACKWARD PASS FROM OUTPUT TO INPUT AND COLLECT SAMPLES IF NEEDED

    state = output;
    if(residual){
      int index = 0;
      index = ((getLayers()-1)&(0xFFFFFFFF-1))-2;
      if(index >= 0) 
	skipValue = xsamples[index];
      else
	skipValue = input; 
    }

    math::matrix<T> A;
    math::vertex<T> c;

    for(int l=getLayers()-1;l>=0;l--){
      if(collectSamples)
	samples[l].push_back(state);
      
      for(unsigned int i=0;i<state.size();i++){
	state[i] = inv_nonlin_nodropout(state[i], l, i); // ASSUMES NO DROPOUT FOR INVERSE FUNCTION
      }

      const bool residualActive =
	(residual && (l % 2) == 0 && l != 0 && W[l].ysize() == skipValue.size());

      // regularizes A if it don't have pseudoinverse
      {
	bool donthaveinverse = true;
	T regularizer = T(1e-4);
	
	while(donthaveinverse){
	  
	  if(residualActive){
	    A = W[l];
	    c = b[l] + skipValue;
	  }
	  else{
	    A = W[l];
	    c = b[l];
	  }

	  for(unsigned int i=0;i<A.ysize() && i <A.xsize();i++){
	    A(i,i) += regularizer;
	  }
	  
	  if(A.pseudoinverse() == false){
	    // printf("PSEUDOINVERSE FAILED\n");
	    regularizer *= T(4.0);
	  }
	  else{
	    donthaveinverse = false;
	  }
	}
      }
      
      state = A*(state - c);
      
      if(residual && ((l-2) % 2) == 0 && l-2 >= 0){
	skipValue = xsamples[l-2];
      }
      
    }

    input = state; // saves predicted x value from output value
    

    return true;
  }
  
   
  // simple thread-safe version
  // [parallelizable version of calculate: don't calculate gradient nor collect samples]
  template <typename T>
  bool nnetwork<T>::calculate(const math::vertex<T>& input, math::vertex<T>& output) const
  {
    if(input.size() != input_size()) return false; // input vector has wrong dimension
    
    // TODO write cblas and cuBLAS optimized version which uses
    // direct accesses to matrix/vertex memory areas

    output = input;
    auto& state = output;

    math::vertex<T> skipValue;

    if(residual) skipValue = state;
    
    for(unsigned int l=0;l<getLayers();l++){
      
      if(residual && (l % 2) == 0 && l != 0 && W[l].ysize() == skipValue.size())
	state = W[l]*state + b[l] + skipValue;
      else
	state = W[l]*state + b[l];

      for(unsigned int i=0;i<state.size();i++){
	state[i] = nonlin(state[i], l, i);
      }

      if(residual && (l % 2) == 0 && l != 0)
	skipValue = state;
    }

    return true;
  }

  
  // thread safe calculate(). Uses dropout data provided by user.
  // This allows same nnetwork<> object to be used in thread safe manner (const).
  template <typename T>
  bool nnetwork<T>::calculate(const math::vertex<T>& input, math::vertex<T>& output,
			      const std::vector< std::vector<bool> >& dropout) const
  {
    // TODO write cblas and cuBLAS optimized version which uses
    // direct accesses to matrix/vertex memory areas

    if(input.size() != input_size()) return false;
    if(dropout.size() != getLayers()) return false;

    output = input;
    math::vertex<T>& state = output;

    math::vertex<T> skipValue;

    if(residual) skipValue = state;
    
    for(unsigned int l=0;l<getLayers();l++){
      
      if(residual && (l % 2) == 0 && l != 0 && W[l].ysize() == skipValue.size())
	state = W[l]*state + b[l] + skipValue;
      else
	state = W[l]*state + b[l];
      
      for(unsigned int i=0;i<state.size();i++){
	if(dropout[l][i]) state[i] = T(0.0f);
	else state[i] = nonlin_nodropout(state[i], l, i);
      }
      
      if(residual && (l % 2) == 0 && l != 0)
	skipValue = state;
    }
    
    return true;
  }

  
  // thread safe calculate(). Use internal dropout if available.
  // This allows same nnetwork<> object to be used in thread safe manner (const).
  template <typename T>
  bool nnetwork<T>::calculate(const math::vertex<T>& input, math::vertex<T>& output,
			      std::vector< math::vertex<T> >& bpdata) const
  {
    // TODO write cblas and cuBLAS optimized version which uses
    // direct accesses to matrix/vertex memory areas

    if(input.size() != input_size()) return false;
    
    output = input;
    math::vertex<T>& state = output;

    bpdata.resize(getLayers()+1);
    bpdata[0] = state; // input value

    math::vertex<T> skipValue;

    if(residual) skipValue = state;

    for(unsigned int l=0;l<getLayers();l++){

      if(residual && (l % 2) == 0 && l != 0 && W[l].ysize() == skipValue.size())
	state = W[l]*state + b[l] + skipValue;
      else
	state = W[l]*state + b[l];

      // stores neuron's local field
      bpdata[l+1] = state;
      
      for(unsigned int i=0;i<state.size();i++){
	state[i] = nonlin(state[i], l, i);
      }

      if(residual && (l % 2) == 0 && l != 0)
	skipValue = state;
    }
    
    return true;
  }


  // thread safe calculate call which also stores backpropagation data
  // bpdata can be used calculate mse_gradient() with backpropagation
  // in a const nnetwork<> class so that same nnetwork<> object can
  // be used with multiple threads. If dropout vector has data also
  // does dropout heuristics. This allows same nnetwork<> object to be
  // used in thread safe manner.
  template <typename T>
  bool nnetwork<T>::calculate(const math::vertex<T>& input, math::vertex<T>& output,
			      const std::vector< std::vector<bool> >& dropout,
			      std::vector< math::vertex<T> >& bpdata) const
  {
    if(input.size() != input_size()) return false;
    if(dropout.size() != getLayers()) return false;
    
    // TODO write cblas and cuBLAS optimized version which uses
    // direct accesses to matrix/vertex memory areas

    output = input;
    math::vertex<T>& state = output;

    bpdata.resize(getLayers()+1);
    bpdata[0] = state; // input value

    math::vertex<T> skipValue;

    if(residual) skipValue = state;

    for(unsigned int l=0;l<getLayers();l++){

      const bool residualActive =
	(residual && (l % 2) == 0 && l != 0 && W[l].ysize() == skipValue.size());

      if(residualActive)
	state = W[l]*state + b[l] + skipValue;
      else
	state = W[l]*state + b[l];
      
      // stores neuron's local field
      bpdata[l+1] = state;
      
      for(unsigned int i=0;i<state.size();i++){
	if(dropout[l][i]) state[i] = T(0.0f);
	else state[i] = nonlin_nodropout(state[i], l, i);
      }
      
      if(residual && ((l % 2) == 0) && l != 0){
	skipValue = state;
      }
    }
    
    return true;
  }

  
  template <typename T> // number of layers+1
  unsigned int nnetwork<T>::length() const {
    return arch.size();
  }
  
  
  template <typename T>
  bool nnetwork<T>::randomize(const unsigned int type,
			      const T EXTRA_SCALING)
  {
    // was 0.50f for L=40 layers and 0.75 for L=10 layers. L=100 is maybe 0.25???
    // 0.25 was used as default for 2 layers! (no problem)
    const float SUPERRESOLUTION_METRIC_SCALING_FACTOR = 0.25f; // s^d scaling for superreso values [was: 0.25f, 0.75 gives worse results]
    
    if(type == 0){
      const whiteice::math::blas_complex<double> ar(2.0f,0.0f), br(1.0f, 0.0f);
      const whiteice::math::blas_complex<double> ai(0.0f,2.0f), bi(0.0f, 1.0f);

      for(unsigned int l=0;l<getLayers();l++){

	if(frozen[l]) continue; // skip frozen layers
	
	for(unsigned int i=0;i<W[l].size();i++){
	  // RNG is real valued, a and b are complex
	  // this means value is complex valued [-1,+1]+[-1,+1]i
	  const auto value =
	    EXTRA_SCALING*((T(ar)*T(rng.uniformf()) - T(br)) + (T(ai)*T(rng.uniformf()) - T(bi)));
	  
	  whiteice::math::convert(W[l][i], value);
	}

	for(unsigned int i=0;i<b[l].size();i++){
	  const auto value =
	    EXTRA_SCALING*((T(ar)*T(rng.uniformf()) - T(br)) + (T(ai)*T(rng.uniformf()) - T(bi)));
	  
	  whiteice::math::convert(b[l][i], value);
	}
	
      }      
    }
    else if(type == 1)
    {
      const whiteice::math::blas_complex<double> ar(2.0f,0.0f), br(1.0f, 0.0f);
      const whiteice::math::blas_complex<double> ai(0.0f,2.0f), bi(0.0f, 1.0f);
      const whiteice::math::blas_complex<double> bias_scaling(0.01f,0.0f);
      
      for(unsigned int l=0;l<getLayers();l++){

	if(frozen[l]) continue; // skip frozen layers

	// this initialization is as described in the paper of Xavier Glorot
	// "Understanding the difficulty of training deep neural networks"
	
	T var = math::sqrt(math::abs(T(6.0f) / T(arch[l] + arch[l+1])));

	var *= EXTRA_SCALING;

	for(unsigned int i=0;i<W[l].size();i++){
	  // RNG is real valued, a and b are complex
	  // this means value is complex valued var*([-1,+1]+[-1,+1]i)
	  const auto value = ((T(rng.uniformf())*ar - br) + (T(rng.uniformf())*ai - bi))*var;
	  
	  whiteice::math::convert(W[l][i], value);
	}

	// NOTE: bias is now set to small [-1,+1] value but not zero
	// b[l].zero(); // bias terms are set to be zero

	for(unsigned int i=0;i<b[l].size();i++){
	  const auto value =
	    ((T(rng.uniformf())*ar - br) + (T(rng.uniformf())*ai - bi))*var*bias_scaling;
	  
	  whiteice::math::convert(b[l][i], value);
	}

	  
      }
      
    }
    else if(type == 2){ // type = 2

      for(unsigned int l=0;l<getLayers();l++){

	if(frozen[l]) continue; // skip frozen layers

	// this initialization is as described in the paper of Xavier Glorot
	// "Understanding the difficulty of training deep neural networks"

	for(unsigned int i=0;i<W[l].size();i++){
	  if(typeid(T) == typeid(math::blas_real<float>) ||
	     typeid(T) == typeid(math::blas_real<double>) ||
	     typeid(T) == typeid(math::superresolution< math::blas_real<float>,
				 math::modular<unsigned int> >) ||
	     typeid(T) == typeid(math::superresolution< math::blas_real<double>,
				 math::modular<unsigned int> >))
	  {
	    
	    float var  = math::sqrt(math::abs((1.0f) / (arch[l])));

	    float extra = 1.0f;
	    whiteice::math::convert(extra, EXTRA_SCALING);
	    
	    var *= extra;

	    // scales higher dimensions in superresolutional numbers to be smaller..
	    float alpha = SUPERRESOLUTION_METRIC_SCALING_FACTOR;
	    float factor = 1.0f;
	    
	    for(unsigned int k=0;k<W[l][i].size();k++){
	      // RNG is is complex normal value if needed
	      const auto value = (rng.normalf())*var*factor;

	      whiteice::math::convert(W[l][i][k], value);

	      factor *= alpha;
	    }
	  }
	  else{ // complex valued numbers:
	    
	    T var  = math::sqrt(T(1.0f) / arch[l]);
	    T ivar = math::sqrt(T(-1.0f) / arch[l]);
	    
	    var *= EXTRA_SCALING;
	    ivar *= EXTRA_SCALING;
	    
	    // RNG is is complex normal value if needed
	    const T scaling = math::sqrt(T(0.5f)); // CN(0,1) = N(0,0.5^2) + N(0,0.5^2)*i
	    
	    const auto value = (T(rng.normalf())*var + T(rng.normalf())*ivar)*scaling;
	    
	    whiteice::math::convert(W[l][i], value);
	  }
	}

	// NOTE: bias is now set to small [-1,+1] value but not zero
	// b[l].zero(); // bias terms are set to be zero
	
	for(unsigned int i=0;i<b[l].size();i++){
	  if(typeid(T) == typeid(math::blas_real<float>) ||
	     typeid(T) == typeid(math::blas_real<double>) ||
	     typeid(T) == typeid(math::superresolution< math::blas_real<float>,
				 math::modular<unsigned int> >) ||
	     typeid(T) == typeid(math::superresolution< math::blas_real<double>,
				 math::modular<unsigned int> >))
	  {
	    
	    float var  = math::sqrt(math::abs((1.0f) / arch[l]));
	    
	    float bias_scaling, extra;
	    whiteice::math::convert(bias_scaling, 0.01f);
	    whiteice::math::convert(extra, EXTRA_SCALING);

	    var *= extra;

	    // scales higher dimensions in superresolutional numbers to be smaller..
	    float alpha = SUPERRESOLUTION_METRIC_SCALING_FACTOR;
	    float factor = 1.0f;

	    for(unsigned int k=0;k<b[l][i].size();k++){
	      // RNG is is complex normal value if neededq
	      const auto value = (rng.normalf())*var*bias_scaling*factor;
	      
	      whiteice::math::convert(b[l][i][k], value);

	      factor *= alpha;
	    }
	    
	  }
	  else{ // complex valued numbers:
	    
	    T var  = math::sqrt(T(1.0f) / arch[l]);
	    T ivar = math::sqrt(T(-1.0f) / arch[l]);
	    
	    T bias_scaling;
	    whiteice::math::convert(bias_scaling, 0.01f);

	    var *= EXTRA_SCALING;
	    ivar *= EXTRA_SCALING;
	    
	    // RNG is is complex normal value if needed
	    const T scaling = math::sqrt(T(0.5f)); // CN(0,1) = N(0,0.5^2) + N(0,0.5^2)*i
	    
	    const auto value = (T(rng.normalf())*var + T(rng.normalf())*ivar)*scaling*bias_scaling;
	    
	    whiteice::math::convert(b[l][i], value);
	  }
	}

      }
      
    }
    else{
      return false;
    }
    
    return true;
  }
  

  // set parameters to fit the data from dataset (we set weights to match data values)
  // [experimental code]
  template <typename T>
  bool nnetwork<T>::presetWeightsFromData(const whiteice::dataset<T>& ds)
  {
    if(ds.getNumberOfClusters() < 2) return false;
    if(ds.dimension(0) != input_size()) return false;
    if(ds.size(0) <= 0) return false;

    if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
       typeid(T) == typeid(whiteice::math::blas_complex<double>)){
      printf("Warning/ERROR: nnetwork::presetWeightsFromData() don't work with complex data.\n");
      printf("make symmetric_pseudoinverse() work to get this function working.\n");
      return false;
    }

    this->randomize(); // sets biases to random values

    // set weights to match to data so that inner product is matched exactly to some data
    std::vector< whiteice::math::vertex<T> > inputdata;
    std::vector< whiteice::math::vertex<T> > outputdata;
    std::vector< whiteice::math::vertex<T> > realoutput;

    for(unsigned int i=0;i<ds.size(0);i++)
      inputdata.push_back(ds.access(0, i));

    for(unsigned int i=0;i<ds.size(1);i++)
      realoutput.push_back(ds.access(1, i));
    

    for(unsigned int l=0;l<getLayers();l++){
      

      if(l != getLayers()-1){ // not the last layer
	for(unsigned int j=0;j<W[l].ysize();j++){
	  unsigned int k1 = rand() % inputdata.size();
	  unsigned int k2 = rand() % inputdata.size();

	  T norm1 = inputdata[k1].norm(); norm1 = norm1*norm1;
	  T norm2 = inputdata[k2].norm(); norm2 = norm2*norm2;
	  if(norm1 <= T(0.0f)) norm1 = T(1.0f);
	  if(norm2 <= T(0.0f)) norm2 = T(1.0f);

	  for(unsigned int i=0;i<W[l].xsize();i++){
	    W[l](j,i) = T(0.5f)*inputdata[k1][i]/norm1 + T(0.5f)*inputdata[k2][i]/norm2;
	  }
	}

	b[l].zero();
      }
      else{ // the last layer, calculate linear mapping y=A*x + b
	// W = Cyx*inv(Cxx)
	// b = E[y] - W*E[x]
	
	whiteice::math::vertex<T> mx, my;
	whiteice::math::matrix<T> Cxx, Cyx;

	mean_covariance_estimate(mx, Cxx, inputdata);
	mean_crosscorrelation_estimate(mx, my, Cyx, inputdata, realoutput);

	if(Cxx.symmetric_pseudoinverse() == false){
	  printf("ERROR: nnetwork<>::presetWeightsFromData(): symmetric pseudoinverse FAILED.\n");
	  fflush(stdout);
	  assert(0);
	}

	W[l] = Cyx*Cxx;
	b[l] = my - W[l]*mx;
      }

      outputdata.resize(inputdata.size());

      // processes data in parallel
#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<inputdata.size();i++){
	auto out = W[l]*inputdata[i] + b[l];
	for(unsigned int n=0;n<out.size();n++)
	  out[n] = nonlin(out[n], l, n);

	outputdata[i] = out;
      }

      inputdata = outputdata;
      outputdata.clear();
    }

    return true;
  }
  

  // set parameters to fit the data from dataset but uses
  // random weights except for the last layer
  // [experimental code]
  template <typename T>
  bool nnetwork<T>::presetWeightsFromDataRandom(const whiteice::dataset<T>& ds)
  {
    if(ds.getNumberOfClusters() < 2) return false;
    if(ds.dimension(0) != input_size()) return false;
    if(ds.size(0) <= 0) return false;

    if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
       typeid(T) == typeid(whiteice::math::blas_complex<double>)){
      printf("Warning/ERROR: nnetwork::presetWeightsFromDataRandom() don't work with complex data.\n");
      printf("make symmetric_pseudoinverse() work to get this function working.\n");
      return false;
    }
    
    // set weights to match to data so that inner product is matched exactly to some data
    std::vector< whiteice::math::vertex<T> > inputdata;
    std::vector< whiteice::math::vertex<T> > outputdata;
    std::vector< whiteice::math::vertex<T> > realoutput;
    
    for(unsigned int i=0;i<ds.size(0);i++)
      inputdata.push_back(ds.access(0, i));
    
    for(unsigned int i=0;i<ds.size(1);i++)
      realoutput.push_back(ds.access(1, i));
    
    for(unsigned int l=0;l<getLayers();l++){
      
      if(l != getLayers()-1){ // not the last layer

	const T var = math::sqrt(T(1.0f) / W[l].xsize());
	
	for(unsigned int j=0;j<W[l].ysize();j++){
	  for(unsigned int i=0;i<W[l].xsize();i++){
	    W[l](j,i) = var*T(rng.normalf());
	  }
	}
	
	b[l].zero();
      }
      else{ // the last layer, calculate linear mapping y=A*x + b
	// W = Cyx*inv(Cxx)
	// b = E[y] - W*E[x]
	
	whiteice::math::vertex<T> mx, my;
	whiteice::math::matrix<T> Cxx, Cyx;
	
	mean_covariance_estimate(mx, Cxx, inputdata);
	mean_crosscorrelation_estimate(mx, my, Cyx, inputdata, realoutput);
	
	if(Cxx.symmetric_pseudoinverse() == false){
	  printf("ERROR: nnetwork<>::presetWeightsFromDataRandom(): symmetric pseudoinverse FAILED.\n");
	  fflush(stdout);
	  assert(0);
	}
	
	W[l] = Cyx*Cxx;
	b[l] = my - W[l]*mx;
      }
      
      outputdata.resize(inputdata.size());
      
      // processes data in parallel
#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<inputdata.size();i++){
	auto out = W[l]*inputdata[i] + b[l];
	for(unsigned int n=0;n<out.size();n++)
	  out[n] = nonlin(out[n], l, n);
	
	outputdata[i] = out;
      }
      
      inputdata = outputdata;
      outputdata.clear();
    }
    
    return true;
  }
  

  // calculates MSE error of the dataset or negative in case of error
  template <typename T>
  T nnetwork<T>::mse(const whiteice::dataset<T>& data) const
  {
    T error = T(0.0f);

    if(data.getNumberOfClusters() < 2) return T(-1.0f);
    
    if(data.dimension(0) != input_size() || data.dimension(1) != output_size())
      return T(-1.0f);

    if(data.size(0) != data.size(1))
      return T(-1.0f);

    // TODO OpenMP parallelize me (easy) 
    for(unsigned int i=0;i<data.size(0);i++){
      const auto& input = data.access(0, i);
      math::vertex<T> output;

      if(this->calculate(input, output) == false)
	return T(-1.0);

      output -= data.access(1, i);
      T n = output.norm();

      error += n*n;
    }

    if(data.size(0)) error /= data.size(0);

    error = T(0.5)*error; // E{ 0.5*||error||^2 }

    // normalizes per dimension
    error /= data.dimension(1);
    
    return error;
  }
  
  
  // calculates gradient of [ 1/2*(network(last_input|w) - last_output)^2 ]
  // => error*GRAD[function(x,w)]
  // uses values stored by previous computation
  // accesses to the neural network matrix and vector parameter data (memory block)
  template <typename T>
  bool nnetwork<T>::mse_gradient(const math::vertex<T>& error,
				 math::vertex<T>& grad) const
  {
    if(!hasValidBPData)
      return false;

    if(error.size() != arch[arch.size()-1])
      return false;
    
    bool complex_data = false;
    
    if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
       typeid(T) == typeid(whiteice::math::blas_complex<double>) ||
       typeid(T) == typeid(whiteice::math::superresolution<
			   whiteice::math::blas_complex<float>,
			   whiteice::math::modular<unsigned int> >) ||
       typeid(T) == typeid(whiteice::math::superresolution<
			   whiteice::math::blas_complex<double>,
			   whiteice::math::modular<unsigned int> >))
    {
      // with complex data we need to take conjugate of gradient values
      complex_data = true;
    }

    int layer = getLayers()-1;

    // initial local gradient is error[i]*NONLIN'(v)
    math::vertex<T> lgrad(error);
    
    for(unsigned int i=0;i<lgrad.size();i++){
      if(complex_data) lgrad[i].conj();

      lgrad[i] *= Dnonlin(bpdata[layer+1][i], layer, i);
    }

    grad.resize(size);
    unsigned int gindex = grad.size();

    if(residual){
      std::vector< math::vertex<T> > lgrad_prev;
      lgrad_prev.resize(3);
      lgrad_prev[0] = lgrad;
      lgrad_prev[1] = lgrad;
      lgrad_prev[2] = lgrad;

      while(layer >= 0){
	const unsigned int gsize = W[layer].size() + b[layer].size();
	gindex -= gsize;
	
	// delta W = (lgrad * input^T) [input for the layer is bpdata's localfield]
	// delta b =  lgrad;
	
	if(frozen[layer] == false){
	  
	  if(layer > 0){
	    for(unsigned int y=0;y<W[layer].ysize();y++){
	      for(unsigned int x=0;x<W[layer].xsize();x++){
		grad[gindex] = lgrad[y] * nonlin(bpdata[layer][x], layer-1, x);
		gindex++;
	      }
	    }
	  }
	  else{ // input layer
	    for(unsigned int y=0;y<W[layer].ysize();y++){
	      for(unsigned int x=0;x<W[layer].xsize();x++){
		grad[gindex] = lgrad[y] * bpdata[layer][x];
		gindex++;
	      }
	    }
	  }
	  
	  for(unsigned int y=0;y<b[layer].size();y++){
	    grad[gindex] = lgrad[y];
	    gindex++;
	  }
	  
	  gindex -= gsize;
	  
	}
	else{ // sets gradient for frozen layer zero
	  memset(&(grad[gindex]), 0, sizeof(T)*gsize);
	}
	
	lgrad_prev[2] = lgrad_prev[1];
	lgrad_prev[1] = lgrad_prev[0];
	lgrad_prev[0] = lgrad;

	if(layer > 0){ // no need to calculate next local gradient for the input layer
	  if(residual == false || 
	     layer % 2 != 0 || 
	     W[layer].xsize() != lgrad_prev[1].size() || 
	     layer+1 >= (int)W.size())
	  {
	    //printf("NON-RESIDUAL NNETWORK LAYER: %d\n", layer); fflush(stdout);
	    
	    // for hidden layers: local gradient is:
	    // lgrad[n] = diag(..g'(v[i])..)*(W^t * lgrad[n+1])
	    
	    lgrad = lgrad * W[layer];
	    
	    for(unsigned int i=0;i<lgrad.size();i++){
	      lgrad[i] *= Dnonlin(bpdata[layer][i], layer-1, i);
	    }
	  }
	  else{
	    //printf("RESIDUAL NNETWORK LAYER: %d\n", layer); fflush(stdout);
	    
	    lgrad = lgrad * W[layer];

	    lgrad += lgrad_prev[1];
	    
	    for(unsigned int i=0;i<lgrad.size();i++){
	      lgrad[i] *= Dnonlin(bpdata[layer][i], layer-1, i);
	    }
	    
#if 0
	    // residual new local grad
	    // lgrad[n] = diag(..g'..)*
	    // lgrad[n+1] + W[layer]^t*diag(..h'..)*W[layer+1]^t*lgrad[n+1]
	    // 
	    
	    auto e = lgrad_prev[1]*W[layer+1];
	    
	    for(unsigned int i=0;i<e.size();i++){
	      e[i] *= Dnonlin(bpdata[layer+1][i], layer, i);
	    }
	    
	    auto wdw = e * W[layer];

	    wdw += lgrad_prev[1];

	    for(unsigned int i=0;i<wdw.size();i++){
	      wdw[i] *= Dnonlin(bpdata[layer][i], layer-1, i);
	    }
	    
	    lgrad = wdw;
#endif	    
	  }
	}
	
	layer--;
      }
	    
    }
    else{

      while(layer >= 0){
	const unsigned int gsize = W[layer].size() + b[layer].size();
	gindex -= gsize;
	
	// delta W = (lgrad * input^T) [input for the layer is bpdata's localfield]
	// delta b =  lgrad;
	
	if(frozen[layer] == false){
	  
	  if(layer > 0){
	    for(unsigned int y=0;y<W[layer].ysize();y++){
	      for(unsigned int x=0;x<W[layer].xsize();x++){
		grad[gindex] = lgrad[y] * nonlin(bpdata[layer][x], layer-1, x);
		gindex++;
	      }
	    }
	  }
	  else{ // input layer
	    for(unsigned int y=0;y<W[layer].ysize();y++){
	      for(unsigned int x=0;x<W[layer].xsize();x++){
		grad[gindex] = lgrad[y] * bpdata[layer][x];
		gindex++;
	      }
	    }
	  }
	  
	  for(unsigned int y=0;y<b[layer].size();y++){
	    grad[gindex] = lgrad[y];
	    gindex++;
	  }
	  
	  gindex -= gsize;
	  
	}
	else{ // sets gradient for frozen layer zero
	  memset(&(grad[gindex]), 0, sizeof(T)*gsize);
	}
	
	if(layer > 0){ // no need to calculate next local gradient for the input layer
	  // for hidden layers: local gradient is:
	  // lgrad[n] = diag(..g'(v[i])..)*(W^t * lgrad[n+1])
	  
	  lgrad = lgrad * W[layer];
	  
	  for(unsigned int i=0;i<lgrad.size();i++){
	    lgrad[i] *= Dnonlin(bpdata[layer][i], layer-1, i);
	  }
	}
	
	layer--;
      }
      
    }

    assert(gindex == 0);
    
    // for complex neural networks we need to calculate conjugate value of
    // the whole gradient (for this to work we need to calculate conjugate value
    // of error term (f(z)-y) as the input for the gradient operation
    if(complex_data){
      grad.conj();
    }
    
    return true;
    
  }

  
  // calculates gradient of parameter weights w f(v|w) when using squared error: 
  // grad(0,5*error^2) = grad(output - right) = nn(x) - y
  // uses backpropagation data provided by user
  template <typename T>
  bool nnetwork<T>::mse_gradient(const math::vertex<T>& error,
				 const std::vector< math::vertex<T> >& bpdata,
				 math::vertex<T>& grad) const
  {

    if(error.size() != arch[arch.size()-1]){
      printf("FAIL 1: %d != %d\n", error.size(), arch[arch.size()-1]);
      return false;
    }

    if(bpdata.size() != getLayers()+1){ // no backpropagation data
      printf("FAIL 2: %d != %d\n", (int)bpdata.size(), getLayers()+1);
      return false;
    }
    
    bool complex_data = false;
    
    if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
       typeid(T) == typeid(whiteice::math::blas_complex<double>) ||
       typeid(T) == typeid(whiteice::math::superresolution<
			   whiteice::math::blas_complex<float>,
			   whiteice::math::modular<unsigned int> >) ||
       typeid(T) == typeid(whiteice::math::superresolution<
			   whiteice::math::blas_complex<double>,
			   whiteice::math::modular<unsigned int> >))
    {
      // with complex data we need to take conjugate of gradient values
      complex_data = true;
    }

    int layer = getLayers()-1;

    // initial local gradient is error[i]*NONLIN'(v)
    math::vertex<T> lgrad(error);
    
    for(unsigned int i=0;i<lgrad.size();i++){
      if(complex_data) lgrad[i].conj();

      lgrad[i] *= Dnonlin(bpdata[layer+1][i], layer, i);
    }

    grad.resize(size);
    unsigned int gindex = grad.size();

    std::vector< math::vertex<T> > lgrad_prev;
    lgrad_prev.resize(3);
    lgrad_prev[0] = lgrad;
    lgrad_prev[1] = lgrad;
    lgrad_prev[2] = lgrad;
    
    while(layer >= 0){
      const unsigned int gsize = W[layer].size() + b[layer].size();
      gindex -= gsize;
      
      // delta W = (lgrad * input^T) [input for the layer is bpdata's localfield]
      // delta b =  lgrad;
      
      if(frozen[layer] == false){
	
	if(layer > 0){
	  for(unsigned int y=0;y<W[layer].ysize();y++){
	    for(unsigned int x=0;x<W[layer].xsize();x++){
	      grad[gindex] = lgrad[y] * nonlin(bpdata[layer][x], layer-1, x);
	      gindex++;
	    }
	  }
	}
	else{ // input layer
	  for(unsigned int y=0;y<W[layer].ysize();y++){
	    for(unsigned int x=0;x<W[layer].xsize();x++){
	      grad[gindex] = lgrad[y] * bpdata[layer][x];
	      gindex++;
	    }
	  }
	}
	
	for(unsigned int y=0;y<b[layer].size();y++){
	  grad[gindex] = lgrad[y];
	  gindex++;
	}
	
	gindex -= gsize;
	
      }
      else{ // sets gradient for frozen layer zero
	memset(&(grad[gindex]), 0, sizeof(T)*gsize);
      }

      lgrad_prev[2] = lgrad_prev[1];
      lgrad_prev[1] = lgrad_prev[0];
      lgrad_prev[0] = lgrad;
      
      if(layer > 0){ // no need to calculate next local gradient for the input layer
	if(residual == false || 
	   layer % 2 != 0 || 
	   W[layer].xsize() != lgrad_prev[1].size() || 
	   layer+1 >= (int)W.size())
	{
	  //printf("NON-RESIDUAL NNETWORK LAYER: %d\n", layer); fflush(stdout);
	  
	  // for hidden layers: local gradient is:
	  // lgrad[n] = diag(..g'(v[i])..)*(W^t * lgrad[n+1])
	  
	  lgrad = lgrad * W[layer];
	  
	  for(unsigned int i=0;i<lgrad.size();i++){
	    lgrad[i] *= Dnonlin(bpdata[layer][i], layer-1, i);
	  }
	}
	else{
	  //printf("RESIDUAL NNETWORK LAYER: %d\n", layer); fflush(stdout);

	  lgrad = lgrad * W[layer];

	  lgrad += lgrad_prev[1];
	  
	  for(unsigned int i=0;i<lgrad.size();i++){
	    lgrad[i] *= Dnonlin(bpdata[layer][i], layer-1, i);
	  }
	  
#if 0
	  // residual new local grad
	  // lgrad[n] = diag(..g'..)*
	  // lgrad[n+1] + W[layer]^t*diag(..h'..)*W[layer+1]^t*lgrad[n+1]
	  // 
	  
	  auto e = lgrad_prev[1]*W[layer+1];
	  
	  for(unsigned int i=0;i<e.size();i++){
	    e[i] *= Dnonlin(bpdata[layer+1][i], layer, i);
	  }
	  
	  auto wdw = e * W[layer];
	  
	  wdw += lgrad_prev[1];
	  
	  for(unsigned int i=0;i<wdw.size();i++){
	    wdw[i] *= Dnonlin(bpdata[layer][i], layer-1, i);
	  }
	  
	  lgrad = wdw;
#endif	  
	}
      }
      
      
      layer--;
    }
    
    assert(gindex == 0);
    
    // for complex neural networks we need to calculate conjugate value of
    // the whole gradient (for this to work we need to calculate conjugate value
    // of error term (f(z)-y) as the input for the gradient operation
    if(complex_data){
      grad.conj();
    }
    
    return true;
    
  }
  

  // calculates gradient of [ 1/2*(network(last_input|w) - last_output)^2 ]
  // => error*GRAD[function(x,w)]
  // used backpropagation bpdata provided by caller (use calculate() with bpdata) and 
  // dropout heuristic.
  template <typename T>
  bool nnetwork<T>::mse_gradient(const math::vertex<T>& error,
				 const std::vector< math::vertex<T> >& bpdata,
				 const std::vector< std::vector<bool> >& dropout,
				 math::vertex<T>& grad) const
  {
    if(error.size() != arch[arch.size()-1])
      return false;
    
    if(bpdata.size() != getLayers()+1)
      return false; // no backpropagation data

    if(dropout.size() != getLayers())
      return this->mse_gradient(error, bpdata, grad);
    
    bool complex_data = false;
    
    if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
       typeid(T) == typeid(whiteice::math::blas_complex<double>) ||
       typeid(T) == typeid(whiteice::math::superresolution<
			   whiteice::math::blas_complex<float>,
			   whiteice::math::modular<unsigned int> >) ||
       typeid(T) == typeid(whiteice::math::superresolution<
			   whiteice::math::blas_complex<double>,
			   whiteice::math::modular<unsigned int> >))
    {
      // with complex data we need to take conjugate of gradient values
      complex_data = true;
    }

    
    int layer = getLayers()-1;

    // initial local gradient is error[i]*NONLIN'(v)
    math::vertex<T> lgrad(error);
    
    for(unsigned int i=0;i<lgrad.size();i++){
      if(complex_data) lgrad[i].conj();

      lgrad[i] *= Dnonlin_nodropout(bpdata[layer+1][i], layer, i);
    }

    grad.resize(size);
    unsigned int gindex = grad.size();

    std::vector< math::vertex<T> > lgrad_prev;
    lgrad_prev.resize(3);
    lgrad_prev[0] = lgrad;
    lgrad_prev[1] = lgrad;
    lgrad_prev[2] = lgrad;

    
    while(layer >= 0){
      const unsigned int gsize = W[layer].size() + b[layer].size();
      gindex -= gsize;

      // delta W = (lgrad * input^T) [input for the layer is bpdata's localfield]
      // delta b =  lgrad;

      if(frozen[layer] == false){

	if(layer > 0){
	  for(unsigned int y=0;y<W[layer].ysize();y++){
	    for(unsigned int x=0;x<W[layer].xsize();x++){
	      
	      if(dropout[layer-1][x]) grad[gindex] = T(0.0f);
	      else grad[gindex] = lgrad[y] * nonlin_nodropout(bpdata[layer][x], layer-1, x);
	      
	      gindex++;
	    }
	  }
	}
	else{ // input layer
	  for(unsigned int y=0;y<W[layer].ysize();y++){
	    for(unsigned int x=0;x<W[layer].xsize();x++){
	      grad[gindex] = lgrad[y] * bpdata[layer][x];
	      gindex++;
	    }
	  }
	}
	
	for(unsigned int y=0;y<b[layer].size();y++){
	  grad[gindex] = lgrad[y];
	  gindex++;
	}

	gindex -= gsize;

      }
      else{ // sets gradient for frozen layer zero
	memset(&(grad[gindex]), 0, sizeof(T)*gsize);
      }

      lgrad_prev[2] = lgrad_prev[1];
      lgrad_prev[1] = lgrad_prev[0];
      lgrad_prev[0] = lgrad;
      
      if(layer > 0){ // no need to calculate next local gradient for the input layer
	if(residual == false || 
	   layer % 2 != 0 || 
	   W[layer].xsize() != lgrad_prev[1].size() || 
	   layer+1 >= (int)W.size())
	{
	  //printf("NON-RESIDUAL NNETWORK LAYER: %d\n", layer); fflush(stdout);
	  
	  // for hidden layers: local gradient is:
	  // lgrad[n] = diag(..g'(v[i])..)*(W^t * lgrad[n+1])
	  
	  lgrad = lgrad * W[layer];
	  
	  for(unsigned int i=0;i<lgrad.size();i++){
	    lgrad[i] *= Dnonlin(bpdata[layer][i], layer-1, i);
	  }
	}
	else{
	  //printf("RESIDUAL NNETWORK LAYER: %d\n", layer); fflush(stdout);

	  lgrad = lgrad * W[layer];

	  lgrad += lgrad_prev[1];
	  
	  for(unsigned int i=0;i<lgrad.size();i++){
	    lgrad[i] *= Dnonlin(bpdata[layer][i], layer-1, i);
	  }
	  
#if 0
	  // residual new local grad
	  // lgrad[n] = diag(..g'..)*
	  // lgrad[n+1] + W[layer]^t*diag(..h'..)*W[layer+1]^t*lgrad[n+1]
	  // 
	  
	  auto e = lgrad_prev[1]*W[layer+1];
	  
	  for(unsigned int i=0;i<e.size();i++){
	    e[i] *= Dnonlin(bpdata[layer+1][i], layer, i);
	  }
	  
	  auto wdw = e * W[layer];
	  
	  wdw += lgrad_prev[1];
	  
	  for(unsigned int i=0;i<wdw.size();i++){
	    wdw[i] *= Dnonlin(bpdata[layer][i], layer-1, i);
	  }
	  
	  lgrad = wdw;
#endif	  
	}
      }

      layer--;
    }

    assert(gindex == 0);
    
    // for complex neural networks we need to calculate conjugate value of
    // the whole gradient (for this to work we need to calculate conjugate value
    // of error term (f(z)-y) as the input for the gradient operation
    if(complex_data){
      grad.conj();
    }
    
    return true;
  }
    

  /* 
   * calculates jacobian/gradient of parameter weights w f(v|w)
   *
   * For math documentation read docs/neural_network_gradient.tm
   *
   */
  template <typename T>
  bool nnetwork<T>::jacobian(const math::vertex<T>& input,
			     math::matrix<T>& grad) const
  {
    if(input.size() != this->input_size()) return false;
    
    // local fields for each layer (and input not stored)
    std::vector< whiteice::math::vertex<T> > v;

    auto x = input;

    // forward pass: calculates local fields v
    int l = 0;
    math::vertex<T> skipValue;

    if(residual) skipValue = x;

    for(l=0;l<(signed)getLayers();l++){

      if(residual && (l % 2) == 0 && l != 0 && W[l].ysize() == skipValue.size())
	x = W[l]*x + b[l] + skipValue;
      else
	x = W[l]*x + b[l];

      v.push_back(x); // stores local field

      for(unsigned int i=0;i<getNeurons(l);i++){
	x[i] = nonlin(x[i], l, i);
      }

      if(residual && (l % 2) == 0 && l != 0)
	skipValue = x;
    }

    /////////////////////////////////////////////////
    // backward pass: calculates gradients

    l--;

    grad.resize(output_size(), gradient_size());
    grad.zero(); // REMOVE ME: for debugging..

    whiteice::math::matrix<T> lgrad; // calculates local gradient
    lgrad.resize(output_size(), output_size());
    lgrad.zero();

    for(unsigned int i=0;i<output_size();i++){
      lgrad(i,i) = Dnonlin(v[l][i], l, i);
    }

    unsigned int index = gradient_size();

    std::vector< math::matrix<T> > lgrad_prev;
    lgrad_prev.resize(3);
    lgrad_prev[0] = lgrad;
    lgrad_prev[1] = lgrad;
    lgrad_prev[2] = lgrad;

    for(;l>0;l--){
      
      // calculates gradient [gradient of W is always in ROW MAJOR format!]
      {
	index -= W[l].ysize()*W[l].xsize() + b[l].size();

	// weight matrix gradient
#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<W[l].ysize();j++){
	  const unsigned int jindex = index + j*W[l].xsize();
	  
	  for(unsigned int i=0, iindex=jindex;i<W[l].xsize();i++,iindex++){
	    
	    // TODO optimize with vector math
	    //#pragma omp parallel for schedule(auto)
	    for(unsigned int k=0;k<grad.ysize();k++){
	      grad(k, iindex) = lgrad(k,j)*nonlin(v[l-1][i], l-1, i);
	    }
	    
	  }
	}

	index += W[l].ysize()*W[l].xsize();

	// bias vector gradient
#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<b[l].size();j++){
	  const unsigned int bindex = index + j;
	  
	  // TODO optimize with vector math
	  //#pragma omp parallel for schedule(auto)
	  for(unsigned int k=0;k<grad.ysize();k++)
	    grad(k, bindex) = lgrad(k, j);	  
	}

	index += b[l].size();

	index -= W[l].ysize()*W[l].xsize() + b[l].size();
      }

      lgrad_prev[2] = lgrad_prev[1];
      lgrad_prev[1] = lgrad_prev[0];
      lgrad_prev[0] = lgrad;
      
      // updates local gradient
      if(residual == false ||
	 l % 2 != 0 || 
	 W[l].xsize() != lgrad_prev[1].xsize() ||
	 l+1 >= (int)W.size())
      {
	
	auto temp = lgrad * W[l];
	lgrad.resize(temp.ysize(), getNeurons(l-1));
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<lgrad.xsize();i++){
	  const auto Df = Dnonlin(v[l-1][i], l-1, i);
	  for(unsigned int j=0;j<lgrad.ysize();j++){
	    lgrad(j,i) = temp(j,i)*Df;
	  }
	}
	
      }
      else{

	auto temp = lgrad * W[l];
	lgrad.resize(temp.ysize(), getNeurons(l-1));
	
	temp += lgrad_prev[1];
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<lgrad.xsize();i++){
	  const auto Df = Dnonlin(v[l-1][i], l-1, i);
	  for(unsigned int j=0;j<lgrad.ysize();j++){
	    lgrad(j,i) = temp(j,i)*Df;
	  }
	}
      }
      
      
    }


    // l = 0 layer (input layer)
    {
      
      // calculates gradient
      {
	index -= W[0].ysize()*W[0].xsize() + b[0].size();

	// weight matrix gradient
#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<W[0].ysize();j++){
	  const unsigned int jindex = index + j*W[0].xsize();
	  for(unsigned int i=0, iindex=jindex;i<W[0].xsize();i++,iindex++){
	    
	    // TODO optimize with vector math
	    //#pragma omp parallel for schedule(auto)
	    for(unsigned int k=0;k<grad.ysize();k++)
	      grad(k, iindex) = lgrad(k,j)*input[i];
	  }
	}

	index += W[0].ysize()*W[0].xsize();

	// bias vector gradient
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<b[0].size();i++){
	  const unsigned int bindex = index + i;
	  
	  // TODO optimize with vector math
	  //#pragma omp parallel for schedule(auto)
	  for(unsigned int k=0;k<grad.ysize();k++)
	    grad(k, bindex) = lgrad(k, i);

	}

	index += b[0].size();

	index -= W[0].ysize()*W[0].xsize() + b[0].size();
      }
      
    }
    
    assert(index == 0);

    return true;
  }


    /* 
   * calculates jacobian/gradient of parameter weights w f(v|w)
   *
   * For math documentation read docs/neural_network_gradient.tm
   *
   */
  template <typename T>
  bool nnetwork<T>::jacobian(const math::vertex<T>& input,
			     math::matrix<T>& grad,
			     const std::vector< std::vector<bool> >& dropout) const
  {
    if(input.size() != this->input_size()) return false;

    if(dropout.size() != getLayers())
      return this->jacobian(input, grad);
    
    // local fields for each layer (and input not stored)
    std::vector< whiteice::math::vertex<T> > v;

    auto x = input;

    // forward pass: calculates local fields
    int l = 0;
    math::vertex<T> skipValue;

    if(residual) skipValue = x;

    for(l=0;l<(signed)getLayers();l++){

      if(residual && (l % 2) == 0 && l != 0 && W[l].ysize() == skipValue.size())
	x = W[l]*x + b[l] + skipValue;
      else
	x = W[l]*x + b[l];

      v.push_back(x); // stores local field

      for(unsigned int i=0;i<getNeurons(l);i++){
	if(dropout[l][i]) x[i] = T(0.0f);
	else x[i] = nonlin_nodropout(x[i], l, i);
      }

      if(residual && (l % 2) == 0 && l != 0)
	skipValue = x;
    }

    /////////////////////////////////////////////////
    // backward pass: calculates gradients

    l--;

    grad.resize(output_size(), gradient_size());
    grad.zero(); // REMOVE ME: for debugging..

    whiteice::math::matrix<T> lgrad; // calculates local gradient
    lgrad.resize(output_size(), output_size());
    lgrad.zero();

    for(unsigned int i=0;i<output_size();i++){
      lgrad(i,i) = Dnonlin_nodropout(v[l][i], l, i);
    }

    unsigned int index = gradient_size();

    std::vector< math::matrix<T> > lgrad_prev;
    lgrad_prev.resize(3);
    lgrad_prev[0] = lgrad;
    lgrad_prev[1] = lgrad;
    lgrad_prev[2] = lgrad;

    for(;l>0;l--){
      
      // calculates gradient [gradient of W is always in ROW MAJOR format!]
      {
	index -= W[l].ysize()*W[l].xsize() + b[l].size();

	// weight matrix gradient
#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<W[l].ysize();j++){
	  const unsigned int jindex = index + j*W[l].xsize();
	  
	  for(unsigned int i=0, iindex=jindex;i<W[l].xsize();i++,iindex++){
	    
	    // TODO optimize with vector math
	    //#pragma omp parallel for schedule(auto)
	    for(unsigned int k=0;k<grad.ysize();k++){
	      if(dropout[l-1][i]) grad(k, iindex) = T(0.0f);
	      else grad(k, iindex) = lgrad(k,j)*nonlin_nodropout(v[l-1][i], l-1, i);
	    }
	    
	  }
	}

	index += W[l].ysize()*W[l].xsize();

	// bias vector gradient
#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<b[l].size();j++){
	  const unsigned int bindex = index + j;
	  
	  // TODO optimize with vector math
	  //#pragma omp parallel for schedule(auto)
	  for(unsigned int k=0;k<grad.ysize();k++)
	    grad(k, bindex) = lgrad(k, j);	  
	}

	index += b[l].size();

	index -= W[l].ysize()*W[l].xsize() + b[l].size();
      }

      lgrad_prev[2] = lgrad_prev[1];
      lgrad_prev[1] = lgrad_prev[0];
      lgrad_prev[0] = lgrad;

      // updates local gradient
      if(residual == false ||
	 l % 2 != 0 || 
	 W[l].xsize() != lgrad_prev[1].ysize() ||
	 l+1 >= (int)W.size())
      {
	auto temp = lgrad * W[l];
	lgrad.resize(temp.ysize(), getNeurons(l-1));
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<lgrad.xsize();i++){
	  const auto Df = dropout[l-1][i] ? T(0.0f) : Dnonlin_nodropout(v[l-1][i], l-1, i);
	  for(unsigned int j=0;j<lgrad.ysize();j++){
	    lgrad(j,i) = temp(j,i)*Df;
	  }
	}
      }
      else{
	auto temp = lgrad * W[l];
	lgrad.resize(temp.ysize(), getNeurons(l-1));
	
	temp += lgrad_prev[1];
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<lgrad.xsize();i++){
	  const auto Df = dropout[l-1][i] ? T(0.0f) : Dnonlin_nodropout(v[l-1][i], l-1, i);
	  for(unsigned int j=0;j<lgrad.ysize();j++){
	    lgrad(j,i) = temp(j,i)*Df;
	  }
	}
	
      }
      
    }


    // l = 0 layer (input layer)
    {
      
      // calculates gradient
      {
	index -= W[0].ysize()*W[0].xsize() + b[0].size();

	// weight matrix gradient
#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<W[0].ysize();j++){
	  const unsigned int jindex = index + j*W[0].xsize();
	  for(unsigned int i=0, iindex=jindex;i<W[0].xsize();i++,iindex++){
	    
	    // TODO optimize with vector math
	    //#pragma omp parallel for schedule(auto)
	    for(unsigned int k=0;k<grad.ysize();k++)
	      grad(k, iindex) = lgrad(k,j)*input[i];
	  }
	}

	index += W[0].ysize()*W[0].xsize();

	// bias vector gradient
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<b[0].size();i++){
	  const unsigned int bindex = index + i;
	  
	  // TODO optimize with vector math
	  //#pragma omp parallel for schedule(auto)
	  for(unsigned int k=0;k<grad.ysize();k++)
	    grad(k, bindex) = lgrad(k, i);

	}

	index += b[0].size();

	index -= W[0].ysize()*W[0].xsize() + b[0].size();
      }
      
    }

    
    
    assert(index == 0);

    return true;
  }
  
  

  
  
  template <typename T> // non-linearity used in neural network
  inline T nnetwork<T>::nonlin(const T& input, unsigned int layer, unsigned int neuron) const 
  {
    assert(layer < getLayers());
    assert(neuron < getNeurons(layer));

    if(dropout.size() > 0){
      if(dropout[layer][neuron])
	return T(0.0f);
    }

    const float RELUcoef = 0.01f; // original was 0.01f

    if(nonlinearity[layer] == softmax){
      const T k = T(1.50f);

      T in = input;
      if(abs(in) > abs(T(+20.0f)))
	in = abs(T(+20.0f))*in/abs(in);

      const T value = T(1.0f) + whiteice::math::exp(k*in);

      T output = whiteice::math::log(T(1.0f) + whiteice::math::exp(k*in))/k;

      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == sigmoid){
      // non-linearity motivated by restricted boltzman machines..
      T in = input;

      if(abs(in) > abs(T(+20.0f)))
	in = abs(T(+20.0f))*in/abs(in);
      
      T output = T(1.0f) / (T(1.0f) + math::exp(-in));

      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == stochasticSigmoid){
      // non-linearity motivated by restricted boltzman machines..
      whiteice::math::blas_complex<double> in, out;
      whiteice::math::convert(in, input);

      if(abs(in) > abs(math::blas_complex<double>(+20.0f)))
	in = abs(math::blas_complex<double>(+20.0f))*in/abs(in);
      
      out = math::blas_complex<double>(1.0f) / (math::blas_complex<double>(1.0f) + math::exp(-in));

      // IS THIS REALLY CORRECT(?)
      const auto rand_real = abs(T(((double)rand())/((double)RAND_MAX)));
      const auto rand_imag = abs(T(((double)rand())/((double)RAND_MAX)));

      whiteice::math::blas_complex<double> value;

      if(abs(math::real(out)) > rand_real.first()){ value.real(1.0f); }
      else{ value.real(0.0f); }
      
      if(abs(math::imag(out)) > rand_imag.first()){ value.imag(1.0f); }
      else{ value.imag(0.0f); }

      T output_value;
      whiteice::math::convert(output_value, T(value));

      if(batchnorm && layer != getLayers()-1){
	return (output_value - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output_value;
      }
    }
    else if(nonlinearity[layer] == tanh){

      const T a = T(1.7159f);
      const T b = T(2.0f/3.0f);
      
      T in = input;
      
      if(abs(in) > abs(T(+10.0f))) in = abs(T(+10.0))*in/abs(in);
      
      const T e2x = whiteice::math::exp(T(2.0f)*b*in);
      const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));
      const T output = a*tanhbx;

      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == tanh10){

      const T a = T(10.0f);
      const T b = T(2.0f/3.0f);
      
      T in = input;
      
      if(abs(in) > abs(T(+10.0f))) in = abs(T(+10.0))*in/abs(in);
      
      const T e2x = whiteice::math::exp(T(2.0f)*b*in);
      const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));
      const T output = a*tanhbx;

      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == halfLinear){
      // tanh(x) + 0.5x: from a research paper statistically
      // better gradiets for deep neural networks
      {
	const T a = T(1.7159f); // suggested by Haykin's neural network book (1999)
	const T b = T(2.0f/3.0f);
	
	if(abs(input) > abs(T(+10.0f))){
	  T output = a*input/abs(input) + T(0.5f)*a*b*input;

	  if(batchnorm && layer != getLayers()-1){
	    return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	}
	
	// for real valued data
	//if(input > T(10.0)) return a + T(0.5f)*a*b*input;
	//else if(input < T(-10.0)) return -a + T(0.5)*a*b*input;
	
	const T e2x = whiteice::math::exp(T(2.0f)*b*input);
	const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));
	T output = a*tanhbx;
	output = (output + T(0.5f)*a*b*input);

	if(batchnorm && layer != getLayers()-1){
	  return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	}
	else{
	  return output;
	}
      }
      
    }
    else if(nonlinearity[layer] == hermite){ // Hermite polynomials

      // no large values..
      T in = input;

      for(unsigned int n=0;n<input.size();n++){
        if(input[n] > T(+10.0f)[0]) in[n] = T(+10.0f)[0];
        else if(input[n] < T(-10.0f)[0]) in[n] = T(-10.0f)[0];
      }
      
      
      T output = in;
      
      const unsigned int hermite_degree = 1 + (neuron % 3);

      if(hermite_degree == 1){
	output = T(2.0f)*in;
      }
      else if(hermite_degree == 2){
	output = T(4.0f)*in*in - T(2.0f);
      }
      else if(hermite_degree == 3){
	output = T(8.0f)*in*in*in - T(12.0f)*in;
      }


      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
      
    }
    else if(nonlinearity[layer] == pureLinear){
      T output = input; // all layers/neurons are linear..

      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
      
    }
    else if(nonlinearity[layer] == rectifier){

      if(typeid(T) == typeid(whiteice::math::blas_real<float>) ||
	 typeid(T) == typeid(whiteice::math::blas_real<double>)){
	if(input.first().real() < 0.0f){
	  T output = T(RELUcoef*input.first().real());
	  
	  if(batchnorm && layer != getLayers()-1){
	    return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	}
	else{
	  T output = T(input.first().real());

	  if(batchnorm && layer != getLayers()-1){
	    return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	}
      }
      else if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
	      typeid(T) == typeid(whiteice::math::blas_complex<double>))
      {
	math::blas_complex<double> out;
	out.real(input.first().real());
	out.imag(input.first().imag());
	
	if(input.first().real() < 0.0f){
	  out.real(RELUcoef*out.real());
	}
	
	if(input.first().imag() < 0.0f){
	  out.imag(RELUcoef*out.imag());
	}

	T output = T(out);

	if(batchnorm && layer != getLayers()-1){
	  return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	}
	else{
	  return output;
	}
      }
#if 1
      else if(typeid(T) == typeid(whiteice::math::superresolution<
				  whiteice::math::blas_real<float>,
				  whiteice::math::modular<unsigned int> >) ||
	      typeid(T) == typeid(whiteice::math::superresolution<
				  whiteice::math::blas_real<double>,
				  whiteice::math::modular<unsigned int> >))
      {
	// extension of ReLU to superresolutional numbers [apply ReLU to each dimension]
	
	T output = input;

#if 1
	for(unsigned int i=0;i<output.size();i++){ // was only 1
	  if(output[0].real() < 0.0f)
	    output[i] *= RELUcoef;
	}
#endif

#if 0
	if(input[0].real() < 0.0f){
	  output = T(RELUcoef)*input;
	}
	else{
	  //return input;
	}
#endif

	if(batchnorm && layer != getLayers()-1){
	  return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	}
	else{
	  return output;
	}
      }
#endif
      else{ // superresolution class [almost linear because superresolution numbers are non-linear already]
	if(input[0].real() < 0.0f){
	  T output = T(RELUcoef)*input;

	  if(batchnorm && layer != getLayers()-1){
	    return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	  
	}
	else{
	  T output = input;

	  if(batchnorm && layer != getLayers()-1){
	    return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	  
	}
	
      }
    }
    else{
      assert(0);
    }

    return T(0.0);
  }
  
  
  template <typename T> // derivat of non-linearity used in neural network
  inline T nnetwork<T>::Dnonlin(const T& input, unsigned int layer, unsigned int neuron) const 
  {
    assert(layer < getLayers());
    assert(neuron < getNeurons(layer));

    if(dropout.size() > 0){ // drop out is activated
      if(dropout[layer][neuron])
	return T(0.0f); // this neuron is disabled 
    }

    const float RELUcoef = 0.01f; // original was 0.01f

    if(nonlinearity[layer] == softmax){
      const T k = T(1.50f);

      T in = input;
      
      if(abs(in) > abs(T(+20.0f)))
	in = abs(T(+20.0f))*in/abs(in);

      const T divider = T(1.0f) + whiteice::math::exp(-k*in);

      T output = T(1.0f)/divider;

      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
	
    else if(nonlinearity[layer] == sigmoid){
      // non-linearity motivated by restricted boltzman machines..
      T in = input;

      if(abs(in) > abs(T(+20.0f))) in = abs(T(+20.0))*in/abs(in);

      T output = T(1.0f) + math::exp(-in);
      output = math::exp(-in) / (output*output);

      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == stochasticSigmoid){
      // FIXME: what is "correct" derivate here? I guess we should calculate E{g'(x)} or something..
      // in general stochastic layers should be frozen so that they are optimized
      // through other means than following the gradient..
      T in = input;

      if(abs(in) > abs(T(+20.0f))) in = abs(T(+20.0))*in/abs(in);

      // non-linearity motivated by restricted boltzman machines..
      T output = T(1.0f) + math::exp(-in);
      output = math::exp(-in) / (output*output);


      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == tanh){

      const T a = T(1.7159f);
      const T b = T(2.0f/3.0f);
      
      T in = input;

      if(abs(in) > abs(T(+10.0f))) in = abs(T(+10.0))*in/abs(in);

      // for real valued data:
      //if(in > T(+10.0f)) in = T(+10.0);
      //else if(in < T(-10.0f)) in = T(-10.0f);
      
      const T e2x = whiteice::math::exp(T(2.0f)*b*in);
      const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));

      T output = a*b*(T(1.0f) - tanhbx*tanhbx);

      
      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == tanh10){

      const T a = T(10.0f);
      const T b = T(2.0f/3.0f);
      
      T in = input;

      if(abs(in) > abs(T(+10.0f))) in = abs(T(+10.0))*in/abs(in);

      // for real valued data:
      //if(in > T(+10.0f)) in = T(+10.0);
      //else if(in < T(-10.0f)) in = T(-10.0f);
      
      const T e2x = whiteice::math::exp(T(2.0f)*b*in);
      const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));

      T output = a*b*(T(1.0f) - tanhbx*tanhbx);

      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == halfLinear){

      // tanh(x) + 0.5x: from a research paper statistically
      // better gradiets for deep neural networks
      {
	const T a = T(1.7159f); // suggested by Haykin's neural network book (1999)
	const T b = T(2.0f/3.0f);
	
	if(abs(input.first()).first() > abs(T(+10.0f)).first()){
	  T output = T(0.0f) + T(0.5f)*a*b;

	  if(batchnorm && layer != getLayers()-1){
	    return output/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	}
	
	// for real valued data
	//if(input > T(10.0)) return T(0.0) + T(0.5)*a*b;
	//else if(input < T(-10.0)) return T(0.0) + T(0.5)*a*b;
	
	const T e2x = whiteice::math::exp(T(2.0f)*b*input);
	const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));
	
	T output = a*b*(T(1.0f) - tanhbx*tanhbx);
	output = (output + T(0.5f)*a*b);

	
	if(batchnorm && layer != getLayers()-1){
	  return output/bn_sigma[layer][neuron];
	}
	else{
	  return output;
	}
      }      
    }
    else if(nonlinearity[layer] == hermite){ // Hermite polynomials

      // no large values..
      T in = input;

      for(unsigned int n=0;n<input.size();n++){
        if(input[n] > T(+10.0f)[0]) in[n] = T(+10.0f)[0];
        else if(input[n] < T(-10.0f)[0]) in[n] = T(-10.0f)[0];
      }
      

      T output = in;

      const unsigned int hermite_degree = 1 + (neuron % 3);

      if(hermite_degree == 1){
	output = T(2.0f);
      }
      else if(hermite_degree == 2){
	output = T(8.0f)*in;
      }
      else if(hermite_degree == 3){
	output = T(24.0f)*in*in - T(12.0f);
      }


      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
      
    }
    else if(nonlinearity[layer] == pureLinear){
      T output = T(1.0f); // all layers/neurons are linear..

      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == rectifier){

      if(typeid(T) == typeid(whiteice::math::blas_real<float>) ||
	 typeid(T) == typeid(whiteice::math::blas_real<double>)){
	if(input.first().real() < 0.0f){
	  T output = T(RELUcoef);

	  if(batchnorm && layer != getLayers()-1){
	    return output/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	}
	else{
	  T output = T(1.00f);

	  if(batchnorm && layer != getLayers()-1){
	    return output/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	}
      }
      else if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
	      typeid(T) == typeid(whiteice::math::blas_complex<double>)){
	
	math::blas_complex<double> out;
	out.real(input.first().real());
	out.imag(input.first().imag());
	
	if(input.first().real() < 0.0f){
	  out.real(RELUcoef*out.real());
	}
	
	if(input.first().imag() < 0.0f){
	  out.imag(RELUcoef*out.imag());
	}

	// const T epsilon = T(1e-6);
	const math::blas_complex<double> epsilon = math::blas_complex<double>(1e-6);

	// correct derivate is Df(z) = f(z)/z
	if(abs(input.first().real()) > 1e-9)
	  out /= (input.first());
	else
	  out /= (input.first() + epsilon);

	T output = T(out);

	if(batchnorm && layer != getLayers()-1){
	  return output/bn_sigma[layer][neuron];
	}
	else{
	  return output;
	}
	
      }
      else{ // superresolution

	// in superresolution, we only use leaky ReLU to the zeroth real component and keep other values linear..
	// this should mean that derivate exists because we are only non-linear in real line

#if 1
	// WRONG but WORKS BETTER!

	auto output = input;

	if(input[0].real() < 0.0f)
	  output = T(RELUcoef);
	else
	  output = T(1.0f);
#endif

#if 0
	// CORRECT but WORKS MUCH WORSE!!! (gets stuck to local optimums)
	const double epsilon = 1e-30;
	
	T h;
	
	for(unsigned int i=0;i<h.size();i++)
	  h[i] = epsilon;
	
	T output = (nonlin(input+h, layer, neuron) -
		    nonlin(input, layer, neuron))/h;
#endif

	// return output;
	
	if(batchnorm && layer != getLayers()-1){
	  return output/bn_sigma[layer][neuron];
	}
	else{
	  return output;
	}
	
      }
      
    }
    else{
      assert(0);
    }

    return T(0.0);
  }



  template <typename T> // non-linearity used in neural network
  inline T nnetwork<T>::nonlin_nodropout(const T& input, unsigned int layer, unsigned int neuron) const 
  {
    // no dropout checking
    
    const float RELUcoef = 0.01f; // original was 0.01f
    
    if(nonlinearity[layer] == softmax){
      const T k = T(1.50f);

      T in = input;
      if(abs(in) > abs(T(+20.0f)))
	in = abs(T(+20.0f))*in/abs(in);

      const T value = T(1.0f) + whiteice::math::exp(k*in);

      T output = whiteice::math::log(T(1.0f) + whiteice::math::exp(k*in))/k;
      
      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
      
    }
    else if(nonlinearity[layer] == sigmoid){
      // non-linearity motivated by restricted boltzman machines..
      T in = input;

      if(abs(in) > abs(T(+20.0f)))
	in = abs(T(+20.0f))*in/abs(in);
      
      T output = T(1.0f) / (T(1.0f) + math::exp(-in));

      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == stochasticSigmoid){
      // non-linearity motivated by restricted boltzman machines..
      whiteice::math::blas_complex<double> in, out;
      whiteice::math::convert(in, input);

      if(abs(in) > abs(math::blas_complex<double>(+20.0f)))
	in = abs(math::blas_complex<double>(+20.0f))*in/abs(in);
      
      out = math::blas_complex<double>(1.0f) / (math::blas_complex<double>(1.0f) + math::exp(-in));

      // IS THIS REALLY CORRECT(?)
      const auto rand_real = abs(T(((double)rand())/((double)RAND_MAX)));
      const auto rand_imag = abs(T(((double)rand())/((double)RAND_MAX)));

      whiteice::math::blas_complex<double> value;

      if(abs(math::real(out)) > rand_real.first()){ value.real(1.0f); }
      else{ value.real(0.0f); }
      
      if(abs(math::imag(out)) > rand_imag.first()){ value.imag(1.0f); }
      else{ value.imag(0.0f); }

      T output;
      whiteice::math::convert(output, T(value));
      
      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == tanh){

      const T a = T(1.7159f);
      const T b = T(2.0f/3.0f);
      
      T in = input;
      
      if(abs(in) > abs(T(+10.0f))) in = abs(T(+10.0))*in/abs(in);
      
      const T e2x = whiteice::math::exp(T(2.0f)*b*in);
      const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));
      const T output = a*tanhbx;

      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == tanh10){

      const T a = T(10.0f);
      const T b = T(2.0f/3.0f);
      
      T in = input;
      
      if(abs(in) > abs(T(+10.0f))) in = abs(T(+10.0))*in/abs(in);
      
      const T e2x = whiteice::math::exp(T(2.0f)*b*in);
      const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));
      const T output = a*tanhbx;

      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == halfLinear){
      // tanh(x) + 0.5x: from a research paper statistically
      // better gradiets for deep neural networks
      {
	const T a = T(1.7159f); // suggested by Haykin's neural network book (1999)
	const T b = T(2.0f/3.0f);
	
	if(abs(input) > abs(T(+10.0f))){
	  return a*input/abs(input) + T(0.5f)*a*b*input;
	}
	
	// for real valued data
	//if(input > T(10.0)) return a + T(0.5f)*a*b*input;
	//else if(input < T(-10.0)) return -a + T(0.5)*a*b*input;
	
	const T e2x = whiteice::math::exp(T(2.0f)*b*input);
	const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));
	T output = a*tanhbx;
	output = (output + T(0.5f)*a*b*input);
	
	if(batchnorm && layer != getLayers()-1){
	  return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	}
	else{
	  return output;
	}
      }
      
    }
    else if(nonlinearity[layer] == hermite){ // Hermite polynomials

      // no large values..
      T in = input;

      for(unsigned int n=0;n<input.size();n++){
        if(input[n] > T(+10.0f)[0]) in[n] = T(+10.0f)[0];
        else if(input[n] < T(-10.0f)[0]) in[n] = T(-10.0f)[0];
      }
      

      T output = in;

      const unsigned int hermite_degree = 1 + (neuron % 3);

      if(hermite_degree == 1){
	output = T(2.0f)*in;
      }
      else if(hermite_degree == 2){
	output = T(4.0f)*in*in - T(2.0f);
      }
      else if(hermite_degree == 3){
	output = T(8.0f)*in*in*in - T(12.0f)*in;
      }


      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
      
    }    
    else if(nonlinearity[layer] == pureLinear){
      T output = input; // all layers/neurons are linear..
      
      if(batchnorm && layer != getLayers()-1){
	return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == rectifier){

      if(typeid(T) == typeid(whiteice::math::blas_real<float>) ||
	 typeid(T) == typeid(whiteice::math::blas_real<double>)){
	if(input.first().real() < 0.0f){
	  T output = T(RELUcoef*input.first().real());

	  if(batchnorm && layer != getLayers()-1){
	    return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	}
	else{
	  T output = T(input.first().real());
	  
	  if(batchnorm && layer != getLayers()-1){
	    return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	}
      }
      else if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
	      typeid(T) == typeid(whiteice::math::blas_complex<double>)){
	math::blas_complex<double> out;
	out.real(input.first().real());
	out.imag(input.first().imag());
	
	if(input.first().real() < 0.0f){
	  out.first().real(RELUcoef*out.real());
	}
	
	if(input.first().imag() < 0.0f){
	  out.first().imag(RELUcoef*out.imag());
	}

	T output = T(out);

	if(batchnorm && layer != getLayers()-1){
	  return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	}
	else{
	  return output;
	}
      }
      else{ // superresolution class
	T output = input;

	for(unsigned int i=0;i<output.size();i++){ // was only 1
	  if(output[0].real() < 0.0f)
	    output[i] *= RELUcoef;
	}

	if(batchnorm && layer != getLayers()-1){
	  return (output - bn_mu[layer][neuron])/bn_sigma[layer][neuron];
	}
	else{
	  return output;
	}
	
#if 0
	if(input[0].real() < 0.0f){
	  const T output = T(RELUcoef)*input;
	  return output;
	}
	else{
	  return input;
	}
#endif
	
      }
    }
    else{
      assert(0);
    }

    return T(0.0);
  }
  
  
  template <typename T> // derivate of non-linearity used in neural network
  inline T nnetwork<T>::Dnonlin_nodropout(const T& input, unsigned int layer, unsigned int neuron) const 
  {
    // no dropout checking

    const float RELUcoef = 0.01f; // original was 0.01f

    if(nonlinearity[layer] == softmax){
      const T k = T(1.50f);

      T in = input;
      
      if(abs(in) > abs(T(+20.0f)))
	in = abs(T(+20.0f))*in/abs(in);

      const T divider = T(1.0f) + whiteice::math::exp(-k*in);

      T output = T(1.0f)/divider;
      
      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
	
    else if(nonlinearity[layer] == sigmoid){
      // non-linearity motivated by restricted boltzman machines..
      T in = input;

      if(abs(in) > abs(T(+20.0f))) in = abs(T(+20.0))*in/abs(in);

      T output = T(1.0f) + math::exp(-in);
      output = math::exp(-in) / (output*output);

      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == stochasticSigmoid){
      // FIXME: what is "correct" derivate here? I guess we should calculate E{g'(x)} or something..
      // in general stochastic layers should be frozen so that they are optimized
      // through other means than following the gradient..
      T in = input;

      if(abs(in) > abs(T(+20.0f))) in = abs(T(+20.0))*in/abs(in);

      // non-linearity motivated by restricted boltzman machines..
      T output = T(1.0f) + math::exp(-in);
      output = math::exp(-in) / (output*output);
      
      
      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == tanh){

      const T a = T(1.7159f);
      const T b = T(2.0f/3.0f);
      
      T in = input;

      if(abs(in) > abs(T(+10.0f)))
	in = abs(T(+10.0))*in/abs(in);

      // for real valued data:
      //if(in > T(+10.0f)) in = T(+10.0);
      //else if(in < T(-10.0f)) in = T(-10.0f);
      
      const T e2x = whiteice::math::exp(T(2.0f)*b*in);
      const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));

      T output = a*b*(T(1.0f) - tanhbx*tanhbx);

      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == tanh10){

      const T a = T(10.0f);
      const T b = T(2.0f/3.0f);
      
      T in = input;

      if(abs(in) > abs(T(+10.0f)))
	in = abs(T(+10.0))*in/abs(in);

      // for real valued data:
      //if(in > T(+10.0f)) in = T(+10.0);
      //else if(in < T(-10.0f)) in = T(-10.0f);
      
      const T e2x = whiteice::math::exp(T(2.0f)*b*in);
      const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));

      T output = a*b*(T(1.0f) - tanhbx*tanhbx);

      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == halfLinear){

      // tanh(x) + 0.5x: from a research paper statistically
      // better gradiets for deep neural networks
      {
	const T a = T(1.7159f); // suggested by Haykin's neural network book (1999)
	const T b = T(2.0f/3.0f);
	
	if(abs(input.first()).first() > abs(T(+10.0f)).first()){
	  T output = T(0.0f) + T(0.5f)*a*b;
	  
	  if(batchnorm && layer != getLayers()-1){
	    return output/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	}
	
	// for real valued data
	//if(input > T(10.0)) return T(0.0) + T(0.5)*a*b;
	//else if(input < T(-10.0)) return T(0.0) + T(0.5)*a*b;
	
	const T e2x = whiteice::math::exp(T(2.0f)*b*input.first());
	const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));
	
	T output = a*b*(T(1.0f) - tanhbx*tanhbx);
	output = (output + T(0.5f)*a*b);

	if(batchnorm && layer != getLayers()-1){
	  return output/bn_sigma[layer][neuron];
	}
	else{
	  return output;
	}
      }      
    }
    else if(nonlinearity[layer] == hermite){ // Hermite polynomials

      // no large values..
      T in = input;

      for(unsigned int n=0;n<input.size();n++){
        if(input[n] > T(+10.0f)[0]) in[n] = T(+10.0f)[0];
        else if(input[n] < T(-10.0f)[0]) in[n] = T(-10.0f)[0];
      }
      

      T output = in;
      
      const unsigned int hermite_degree = 1 + (neuron % 3);

      if(hermite_degree == 1){
	output = T(2.0f);
      }
      else if(hermite_degree == 2){
	output = T(8.0f)*in;
      }
      else if(hermite_degree == 3){
	output = T(24.0f)*in*in - T(12.0f);
      }


      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
      
    }
    else if(nonlinearity[layer] == pureLinear){
      T output = T(1.0f); // all layers/neurons are linear..

      if(batchnorm && layer != getLayers()-1){
	return output/bn_sigma[layer][neuron];
      }
      else{
	return output;
      }
    }
    else if(nonlinearity[layer] == rectifier){

      if(typeid(T) == typeid(whiteice::math::blas_real<float>) ||
	 typeid(T) == typeid(whiteice::math::blas_real<double>)){
	if(input.first().real() < 0.0f){
	  T output = T(RELUcoef);
	  
	  if(batchnorm && layer != getLayers()-1){
	    return output/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	}
	else{
	  T output = T(1.00f);
	  
	  if(batchnorm && layer != getLayers()-1){
	    return output/bn_sigma[layer][neuron];
	  }
	  else{
	    return output;
	  }
	  
	}
      }
      else if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
	      typeid(T) == typeid(whiteice::math::blas_complex<double>)){
	math::blas_complex<double> out;
	out.real(input.first().real());
	out.imag(input.first().imag());
	
	if(input.first().real() < 0.0f){
	  out.real(RELUcoef*out.real());
	}
	
	if(input.first().imag() < 0.0f){
	  out.imag(RELUcoef*out.imag());
	}

	//const T epsilon = T(1e-6);
	const math::blas_complex<double> epsilon = math::blas_complex<double>(1e-6);

	// correct derivate is Df(z) = f(z)/z
	if(abs(input.first().real()) > 1e-9)
	  out /= (input.first());
	else
	  out /= (input.first() + epsilon);

	T output = T(out);

	if(batchnorm && layer != getLayers()-1){
	  return output/bn_sigma[layer][neuron];
	}
	else{
	  return output;
	}
      }
#if 0
      else if(typeid(T) == typeid(whiteice::math::superresolution<
				  whiteice::math::blas_real<float>,
				  whiteice::math::modular<unsigned int> >) ||
	      typeid(T) == typeid(whiteice::math::superresolution<
				  whiteice::math::blas_real<double>,
				  whiteice::math::modular<unsigned int> >))
      {
	const double epsilon = 1e-30;

	T h;

	for(unsigned int i=0;i<h.size();i++)
	  h[i] = epsilon;
	
	const T output = (nonlin_nodropout(input+h, layer, neuron) - nonlin_nodropout(h, layer, neuron))/h;

	return output;
      }
#endif
      else{ // superresolution

	// in superresolution, we only use leaky ReLU to the zeroth real component and keep other values linear..
	// this should mean that derivate exists because we are only non-linear in real line

#if 1
	// WRONG BUT WORKS BETTER!
	
	T output;

	if(input[0].real() < 0.0f){
	  output = T(RELUcoef);
	}
	else{
	  output = T(1.0f);
	}
#endif
	
#if 0
	// CORRECT but WORKS MUCH WORSE!!! (gets stuck to local optimums)
	const double epsilon = 1e-25;
	
	T h;
	
	for(unsigned int i=0;i<h.size();i++)
	  h[i] = epsilon;
	
	T output = (nonlin_nodropout(input+h, layer, neuron) -
		    nonlin_nodropout(input, layer, neuron))/h;
#endif
	
	if(batchnorm && layer != getLayers()-1){
	  return output/bn_sigma[layer][neuron];
	}
	else{
	  return output;
	}
	
      }
      
    }
    else{
      assert(0);
    }

    return T(0.0);
  }


  
  
  template <typename T>
  inline T nnetwork<T>::inv_nonlin(const T& input, unsigned int layer, unsigned int neuron) const 
  {
    // inverse of non-linearity used
#if 0
    // sinh non-linearity..
    // (sinh()) non-linearity is maybe a bit better non-linearity..
    T output = math::sinh(input); 
#endif
    // THIS DO NOT WORK CURRENTLY
    
    T output = T(0.0f);

    assert(0); // there is NO inverse function
    
    return output;
  }


  template <typename T>
  inline T nnetwork<T>::inv_nonlin_nodropout(const T& input,
					     unsigned int layer,
					     unsigned int neuron) const 
  {
    // inverse of non-linearity used
    T output = input;

    const float RELUcoef = 0.01f; // original was 0.01f

    if(nonlinearity[layer] == pureLinear){
      if(batchnorm && layer != getLayers()-1){
	output = output*bn_sigma[layer][neuron] + bn_mu[layer][neuron];
      }

      return output;
    }
    else if(nonlinearity[layer] == rectifier){
      if(batchnorm && layer != getLayers()-1){
	output = output*bn_sigma[layer][neuron] + bn_mu[layer][neuron];
      }

      if(typeid(T) == typeid(whiteice::math::blas_real<float>) ||
	 typeid(T) == typeid(whiteice::math::blas_real<double>)){
	if(input.first().real() < 0.0f)
	  output = output/T(RELUcoef);
	else
	  output = output;

	if(output.first().real() < -10.0f) output = T(-10.0f);
	else if(output > T(+10.0f)) output = T(+10.0f);

	return output;
      }
      else if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
	      typeid(T) == typeid(whiteice::math::blas_complex<double>)){
	math::blas_complex<double> out;
	
	out.real(output.first().real());
	out.imag(output.first().imag());
	
	if(output.first().real() < 0.0f){
	  out.first().real(out.real()/RELUcoef);
	}
	
	if(output.first().imag() < 0.0f){
	  out.first().imag(out.imag()/RELUcoef);
	}

	output = T(out);
	
	if(output.first().real() < (-10.0f)) output.first().real(-10.0f);
	else if(output.first().real() > (+10.0f)) output.first().real(+10.0f);

	if(output.first().imag() < (-10.0f)) output.first().imag(-10.0f);
	else if(output.first().imag() > (+10.0f)) output.first().imag(+10.0f);

	return output;
      }
      else{ // superresolution class

	for(unsigned int i=0;i<output.size();i++){ // was only 1
	  if(output[0].real() < 0.0f)
	    output[i] /= RELUcoef;
	  
	  if(output[i].real() < -10.0f) output[i] = -10.0f;
	  else if(output[i].real() > +10.0f) output[i] = +10.0f;
	}

	return output;
      }
      
    }
    else{
      output = T(0.0f);
      assert(0); // FIXME: there is NO inverse function IMPLEMENTATION FOR MANY CASES
    }
    
    return output;
  }
  

  /*
   * calculates gradient of input value GRAD[f(v|w)] while keeping weights w constant
   */
  template <typename T>
  bool nnetwork<T>::gradient_value(const math::vertex<T>& input,
				   math::matrix<T>& grad) const
  {
    if(input.size() != input_size()) return false;
    
    const unsigned int L = getLayers();
    
    grad.resize(input_size(), input_size());
    grad.identity();

    auto hgrad = grad;
    
    math::vertex<T> x = input;
    math::vertex<T> skipValue;
    
    if(residual) skipValue = x;

    
    for(unsigned int l=0;l<L;l++){

      //printf("L = %d. W*HG = %d %d x %d %d\n", l,
      //     W[l].ysize(), W[l].xsize(), hgrad.ysize(), hgrad.xsize());
      
      hgrad = W[l]*hgrad;

      //printf("MATMUL DONE\n");

      if(residual && (l % 2) == 0 && l != 0 && W[l].ysize() == skipValue.size())
	x = W[l]*x + b[l] + skipValue;
      else
	x = W[l]*x + b[l];

      
      if(residual && l % 2 == 0 && l != 0 && hgrad.ysize() == hgrad.xsize()){
	// hgrad += I

	for(unsigned int i=0;i<hgrad.xsize();i++)
	  hgrad(i,i) += T(1.0f);
	
#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<hgrad.ysize();j++){
	  for(unsigned int i=0;i<hgrad.xsize();i++){
	    hgrad(j,i) *= Dnonlin(x[j], l, j);
	  }
	}
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<x.size();i++){
	  x[i] = nonlin(x[i], l, i);
	}

	//printf("1:L = %d. G = HG*G = %d %d x %d %d\n", l,
	//       hgrad.ysize(), hgrad.xsize(),
	//       grad.ysize(), grad.xsize());

	grad = hgrad*grad;

	unsigned int s = hgrad.ysize();
	hgrad.resize(s,s);
	hgrad.identity();

	//printf("HG MUL DONE\n");
      }
      else if(residual && l % 2 == 0 && l != 0){ // no same layer size (no skip)
	
#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<hgrad.ysize();j++){
	  for(unsigned int i=0;i<hgrad.xsize();i++){
	    hgrad(j,i) *= Dnonlin(x[j], l, j);
	  }
	}
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<x.size();i++){
	  x[i] = nonlin(x[i], l, i);
	}

	//printf("2:L = %d. G = HG*G = %d %d x %d %d\n", l,
	//       hgrad.ysize(), hgrad.xsize(),
	//       grad.ysize(), grad.xsize());
	
	grad = hgrad*grad;

	unsigned int s = hgrad.ysize();
	hgrad.resize(s,s);
	hgrad.identity();
	
	//printf("HG MUL DONE\n");
      }
      else{

#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<hgrad.ysize();j++){
	  for(unsigned int i=0;i<hgrad.xsize();i++){
	    hgrad(j,i) *= Dnonlin(x[j], l, j);
	  }
	}
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<x.size();i++){
	  x[i] = nonlin(x[i], l, i);
	}
	
      }

      
      if(residual && (l % 2) == 0 && l != 0)
	skipValue = x;
      
    }

    //printf("FINAL MATMUL..\n"),
    grad = hgrad*grad;
    //printf("FINAL MATMUL.. DONE\n");
    // hgrad.identity();
    
    return true;
  }


  /*
   * calculates gradient of input value GRAD[f(v|w)] while keeping weights w constant
   * Uses caller provided dropout table. Returns false if it is invalid.
   */
  template <typename T>
  bool nnetwork<T>::gradient_value(const math::vertex<T>& input,
				   math::matrix<T>& grad,
				   const std::vector< std::vector<bool> >& dropout) const
  {
    printf("nnetwork::gradient_value(dropout) called (%d %d)\n", input.size(), input_size());
    fflush(stdout);
    
    if(input.size() != input_size()) return false;
    if(dropout.size() != getLayers()) return false;
    
    const unsigned int L = getLayers();
    
    grad.resize(input_size(), input_size());
    grad.identity();

    auto hgrad = grad;
    
    math::vertex<T> x = input;
    math::vertex<T> skipValue = x;

    for(unsigned int l=0;l<L;l++){
      
      hgrad = W[l]*hgrad;

      if(residual && (l % 2) == 0 && l != 0 && W[l].ysize() == skipValue.size())
	x = W[l]*x + b[l] + skipValue;
      else
	x = W[l]*x + b[l];

      if(residual && l % 2 == 0 && l != 0 && hgrad.xsize() == hgrad.ysize()){
	// hgrad += I

	for(unsigned int i=0;i<hgrad.xsize();i++)
	  hgrad(i,i) += T(1.0f);

#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<hgrad.ysize();j++){
	  for(unsigned int i=0;i<hgrad.xsize();i++){
	    if(dropout[l][j]) hgrad(j,i) = T(0.0f);
	    else hgrad(j,i) *= Dnonlin_nodropout(x[j], l, j);
	  }
	}
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<x.size();i++){
	  if(dropout[l][i]) x[i] = T(0.0f);
	  else x[i] = nonlin_nodropout(x[i], l, i);
	}
	
	grad = hgrad*grad;
	hgrad.identity();
      }
      else if(residual && l % 2 == 0 && l != 0){ // no same layer size (no skip)

#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<hgrad.ysize();j++){
	  for(unsigned int i=0;i<hgrad.xsize();i++){
	    if(dropout[l][j]) hgrad(j,i) = T(0.0f);
	    else hgrad(j,i) *= Dnonlin_nodropout(x[j], l, j);
	  }
	}
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<x.size();i++){
	  if(dropout[l][i]) x[i] = T(0.0f);
	  else x[i] = nonlin_nodropout(x[i], l, i);
	}

	grad = hgrad*grad;
	hgrad.identity();
	
      }
      else{

#pragma omp parallel for schedule(auto)
	for(unsigned int j=0;j<hgrad.ysize();j++){
	  for(unsigned int i=0;i<hgrad.xsize();i++){
	    if(dropout[l][j]) hgrad(j,i) = T(0.0f);
	    else hgrad(j,i) *= Dnonlin_nodropout(x[j], l, j);
	  }
	}
	
#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<x.size();i++){
	  if(dropout[l][i]) x[i] = T(0.0f);
	  else x[i] = nonlin_nodropout(x[i], l, i);
	}
	
      }
      
      if(residual && (l % 2) == 0 && l != 0)
	skipValue = x;
      
    }

    grad = hgrad*grad;
    // hgrad.identity();
    
    return true;
  }


  // calculates softmax values (p-values) of output elements (output(START)..output(END-1))
  // this is needed to probabilistically select a single action from outputs
  template <typename T>
  bool nnetwork<T>::softmax_output(math::vertex<T>& output,
				   const unsigned int START, const unsigned int END) const
  {
    // no proper error handling
    if(START >= END) return false;
    if(START > output.size() || END > output.size()) return false;

    T total_sum = T(0.0f);

    for(unsigned int i=START;i<END;i++){
      T value = output[i];
      if(value >= T(+25.0f)) value = T(+25.0f);
      else if(value <= T(-25.0f)) value = T(-25.0f);

      output[i] = math::exp(value);
      
      total_sum += output[i];
    }

    for(unsigned int i=START;i<END;i++){
      output[i] /= total_sum;
    }

    return true;
  }
  
  
  // calculates softmax of output vertex dimensions [output(start)..output(end-1)] elements
  // and calculates their entropy [used by entropy regularization code]
  template <typename T>
  T nnetwork<T>::entropy(const math::vertex<T>& output,
			 const unsigned int START, const unsigned int END) const
  {
    // no proper error handling
    if(START >= END) return T(0.0f);
    if(START > output.size() || END > output.size()) return T(0.0f);

    std::vector<T> softmax_values;
    T total_sum = T(0.0f);

    for(unsigned int i=START;i<END;i++){
      T value = output[i];
      if(value >= T(+25.0f)) value = T(+25.0f);
      else if(value <= T(-25.0f)) value = T(-25.0f);

      value = math::exp(value);

      softmax_values.push_back(value);
      total_sum += value;
    }

    T H = T(0.0f), p; // entropy
    const T epsilon = T(1e-20); // avoid div by zero values and add small epsilon to p-values

    for(const auto& h : softmax_values){
      p = h/total_sum;
      H += -p*math::log(p + epsilon);
    }

    return H;
  }


  // calculates entropy gradient given output and backpropagation data
  // [so no need for large and slow jacobian matrix]
  template <typename T>
  bool nnetwork<T>::entropy_gradient(const math::vertex<T>& output,
				     const unsigned int START, const unsigned int END,
				     const std::vector< math::vertex<T> >& bpdata,
				     math::vertex<T>& entropy_gradient) const
  {
    if(START >= END) return false;
    if(START >= output.size() || END > output.size()) return false;
    if(output.size() != output_size()) return false;

    // calculates softmax probabilities
    std::vector<T> softmax_values;
    T total_sum = T(0.0f);

    for(unsigned int i=START;i<END;i++){
      T value = output[i];
      if(value >= T(+25.0f)) value = T(+25.0f);
      else if(value <= T(-25.0f)) value = T(-25.0f);

      value = math::exp(value);

      softmax_values.push_back(value);
      total_sum += value;
    }

    // entropy error term is e = S*h, calculates h first
    math::vertex<T> h;
    h.resize(END - START);
    
    for(unsigned int i=0;i<softmax_values.size();i++){
      h[i] = -(T(1.0f) + math::log(softmax_values[i]/total_sum));
    }

    // calculates S^t matrix
    math::matrix<T> St;
    St.resize(softmax_values.size(), output.size());
    St.zero();

    for(unsigned int i=START;i<END;i++){
      const T scaling = softmax_values[i-START]/(total_sum*total_sum);

      for(unsigned int j=START;j<END;j++){
	if(i==j) St(i,j) = scaling*(total_sum - softmax_values[j]);
	else St(i,j) = scaling*(-softmax_values[j]);
      }
      
    }

    // calculates error vector e used for a backpropagation input

    const math::vertex<T> e = h*St;
    
    return mse_gradient(e, bpdata, entropy_gradient);
  }
  
  
  // calculates entropy output gradient given jacobian matrix of neural network.
  // calculates softmax values from output values and use they as probabilities for entropy
  // [used by entropy regularization code]
  template <typename T>
  bool nnetwork<T>::entropy_gradient_j(const math::vertex<T>& output,
				       const unsigned int START, const unsigned int END,
				       const math::matrix<T>& grad, // jacobian matrix
				       math::vertex<T>& entropy_gradient) const
  {
    if(START >= END) return false;
    if(START >= output.size() || END > output.size()) return false;
    if(START >= grad.ysize() || END > grad.ysize()) return false;
    
    // 1. first calculates exp(output) values (unscaled probabilities) q_i

    std::vector<T> qvalues;
    T total_sum = T(0.0f);
    
    for(unsigned int i=START;i<END;i++){
      T value = output[i];
      
      if(value >= T(+25.0f)) value = T(+25.0f);
      else if(value <= T(-25.0f)) value = T(-25.0f);
      
      value = math::exp(value);
      
      qvalues.push_back(value);
      total_sum += value;
    }

    // 2. next calculates grad p_i values

    std::vector< math::vertex<T> > grad_p;
    math::vertex<T> tmp;
    math::vertex<T> grad_y_j;
    
    grad_p.resize(END - START);
    tmp.resize(grad.xsize());
    tmp.zero();

    // calculates constant for each row
    for(unsigned int j=0;j<(END-START);j++){
      grad.rowcopyto(grad_y_j, START+j);
      
      tmp += qvalues[j]*grad_y_j;
    }

    
    for(unsigned int i=0;i<(END-START);i++){
      grad_p[i].resize(grad.xsize());
      
      grad.rowcopyto(grad_p[i], START+i);
      
      grad_p[i] *= total_sum;
      grad_p[i] -= tmp;
      grad_p[i] *= qvalues[i]/(total_sum*total_sum);
    }

    // 3. then calculates entropy gradient
    entropy_gradient.resize(grad.xsize());
    entropy_gradient.zero();

    const T epsilon = T(1e-20); // avoid div by zero values and add small epsilon to p-values

    for(unsigned int i=0;i<(END-START);i++){
      const auto pvalue = (qvalues[i]/total_sum) + epsilon;
      entropy_gradient -= grad_p[i]*(T(1.0f) + math::log(pvalue));
    }

    return true;
  }
  
  
  // calculates Kullback-Leibler divergence between output and correct values
  // assumes output must be converted using softmax and only elements START..END-1 are used
  // assumes correct_pvalues is END-START long p-values vector which sums to one.
  template <typename T>
  T nnetwork<T>::kl_divergence(const math::vertex<T>& output,
			       const unsigned int START, const unsigned int END,
			       const math::vertex<T>& correct_pvalues) const
  {
    // no proper error handling
    if(START >= END) return T(0.0f);
    if(START > output.size() || END > output.size()) return T(0.0f);
    if(END-START != correct_pvalues.size()) return T(0.0f);

    std::vector<T> softmax_values;
    T total_sum = T(0.0f);

    for(unsigned int i=START;i<END;i++){
      T value = output[i];
      if(value >= T(+25.0f)) value = T(+25.0f);
      else if(value <= T(-25.0f)) value = T(-25.0f);

      value = math::exp(value);

      softmax_values.push_back(value);
      total_sum += value;
    }

    T KL = T(0.0f), p; // K-L divergence
    const T epsilon = T(1e-20); // avoid div by zero values and add small epsilon to p-values

    for(unsigned int i=0;i<softmax_values.size();i++){
      p = softmax_values[i]/total_sum;
      
      if(correct_pvalues[i] > T(0.0f)){
	KL -= correct_pvalues[i]*whiteice::math::log((p+epsilon)/correct_pvalues[i]);
      }
    }

    return KL;
  }

  
  // calculates Kullback-Leibler divergence gradient
  // given output (raw outputs) and backpropagation data
  // assumes correct_pvalues is END-START long p-values vector which sums to one.
  template <typename T>
  bool nnetwork<T>::kl_divergence_gradient(const math::vertex<T>& output,
					   const unsigned int START, const unsigned int END,
					   const math::vertex<T>& correct_pvalues,
					   const std::vector< math::vertex<T> >& bpdata,
					   math::vertex<T>& entropy_gradient) const
  {
    if(START >= END) return false;
    if(START >= output.size() || END > output.size()) return false;
    if(output.size() != output_size()) return false;

    // calculates softmax probabilities
    std::vector<T> softmax_values;
    T total_sum = T(0.0f);
    const T epsilon = T(1e-20);

    for(unsigned int i=START;i<END;i++){
      T value = output[i];
      if(value >= T(+25.0f)) value = T(+25.0f);
      else if(value <= T(-25.0f)) value = T(-25.0f);

      value = math::exp(value);

      softmax_values.push_back(value);
      total_sum += value;
    }

    // entropy error term is e = S*h, calculates h first
    math::vertex<T> h;
    h.resize(END - START);
    
    for(unsigned int i=0;i<softmax_values.size();i++){
      h[i] = -correct_pvalues[i]/((softmax_values[i]/total_sum) + epsilon);
    }

    // calculates S^t matrix
    math::matrix<T> St;
    St.resize(softmax_values.size(), output.size());
    St.zero();

    for(unsigned int i=START;i<END;i++){
      const T scaling = softmax_values[i-START]/(total_sum*total_sum);

      for(unsigned int j=START;j<END;j++){
	if(i==j) St(i,j) = scaling*(total_sum - softmax_values[j]);
	else St(i,j) = scaling*(-softmax_values[j]);
      }
      
    }

    // calculates error vector e used for a backpropagation input

    const math::vertex<T> e = h*St;
    
    return mse_gradient(e, bpdata, entropy_gradient);
  }



  // calculates Kullback-Leibler divergence gradient
  // given output (raw outputs) and backpropagation data
  // assumes correct_pvalues is END-START long p-values vector which sums to one.
  template <typename T>
  bool nnetwork<T>::kl_divergence_gradient_j(const math::vertex<T>& output,
					     const unsigned int START, const unsigned int END,
					     const math::vertex<T>& correct_pvalues,
					     const math::matrix<T>& grad, // jacobian matrix
					     math::vertex<T>& entropy_gradient) const
  {
    if(START >= END) return false;
    if(START >= output.size() || END > output.size()) return false;
    if(output.size() != output_size()) return false;
    if(output.size() != grad.ysize()) return false;

    // calculates softmax probabilities
    std::vector<T> softmax_values;
    T total_sum = T(0.0f);
    const T epsilon = T(1e-20);

    for(unsigned int i=START;i<END;i++){
      T value = output[i];
      if(value >= T(+25.0f)) value = T(+25.0f);
      else if(value <= T(-25.0f)) value = T(-25.0f);

      value = math::exp(value);

      softmax_values.push_back(value);
      total_sum += value;
    }

    // entropy error term is e = S*h, calculates h first
    math::vertex<T> h;
    h.resize(END - START);
    
    for(unsigned int i=0;i<softmax_values.size();i++){
      h[i] = -correct_pvalues[i]/((softmax_values[i]/total_sum) + epsilon);
    }

    // calculates S^t matrix
    math::matrix<T> St;
    St.resize(softmax_values.size(), output.size());
    St.zero();

    for(unsigned int i=START;i<END;i++){
      const T scaling = softmax_values[i-START]/(total_sum*total_sum);

      for(unsigned int j=START;j<END;j++){
	if(i==j) St(i,j) = scaling*(total_sum - softmax_values[j]);
	else St(i,j) = scaling*(-softmax_values[j]);
      }
      
    }

    entropy_gradient = (h*St)*grad;
    
    return true;
  }


  // calculates REVERSE Kullback-Leibler divergence between output and correct values
  // assumes output must be converted using softmax and only elements START..END-1 are used
  // assumes correct_pvalues is END-START long p-values vector which sums to one.
  template <typename T>
  T nnetwork<T>::reverse_kl_divergence(const math::vertex<T>& output,
				       const unsigned int START, const unsigned int END,
				       const math::vertex<T>& correct_pvalues) const
  {
    // no proper error handling
    if(START >= END) return T(0.0f);
    if(START > output.size() || END > output.size()) return T(0.0f);
    if(END-START != correct_pvalues.size()) return T(0.0f);

    std::vector<T> softmax_values;
    T total_sum = T(0.0f);

    for(unsigned int i=START;i<END;i++){
      T value = output[i];
      if(value >= T(+25.0f)) value = T(+25.0f);
      else if(value <= T(-25.0f)) value = T(-25.0f);

      value = math::exp(value);

      softmax_values.push_back(value);
      total_sum += value;
    }

    T KL = T(0.0f), p; // K-L divergence
    const T epsilon = T(1e-20); // avoid div by zero values and add small epsilon to p-values

    for(unsigned int i=0;i<softmax_values.size();i++){
      p = softmax_values[i]/total_sum;
      
      if(p > T(0.0f)){
	KL -= p*whiteice::math::log((epsilon+correct_pvalues[i])/p);
      }
    }

    return KL;
  }

  
  // calculates REVERSE Kullback-Leibler divergence gradient
  // given output (raw outputs) and backpropagation data
  // assumes correct_pvalues is END-START long p-values vector which sums to one.
  template <typename T>
  bool nnetwork<T>::reverse_kl_divergence_gradient(const math::vertex<T>& output,
						   const unsigned int START, const unsigned int END,
						   const math::vertex<T>& correct_pvalues,
						   const std::vector< math::vertex<T> >& bpdata,
						   math::vertex<T>& entropy_gradient) const
  {
    if(START >= END) return false;
    if(START >= output.size() || END > output.size()) return false;
    if(output.size() != output_size()) return false;

    // calculates softmax probabilities
    std::vector<T> softmax_values;
    T total_sum = T(0.0f);
    const T epsilon = T(1e-20);

    for(unsigned int i=START;i<END;i++){
      T value = output[i];
      if(value >= T(+25.0f)) value = T(+25.0f);
      else if(value <= T(-25.0f)) value = T(-25.0f);

      value = math::exp(value);

      softmax_values.push_back(value);
      total_sum += value;
    }

    // entropy error term is e = S*h, calculates h first
    math::vertex<T> h;
    h.resize(END - START);
    
    for(unsigned int i=0;i<softmax_values.size();i++){
      h[i] = T(1.0) + math::log(softmax_values[i]/total_sum + epsilon) - math::log(correct_pvalues[i] + epsilon);
    }

    // calculates S^t matrix
    math::matrix<T> St;
    St.resize(softmax_values.size(), output.size());
    St.zero();

    for(unsigned int i=START;i<END;i++){
      const T scaling = softmax_values[i-START]/(total_sum*total_sum);

      for(unsigned int j=START;j<END;j++){
	if(i==j) St(i,j) = scaling*(total_sum - softmax_values[j]);
	else St(i,j) = scaling*(-softmax_values[j]);
      }
      
    }

    // calculates error vector e used for a backpropagation input

    const math::vertex<T> e = h*St;
    
    return mse_gradient(e, bpdata, entropy_gradient);
  }

  
  // calculates REVERSE Kullback-Leibler divergence gradient
  // given output (raw outputs) and backpropagation data
  // assumes correct_pvalues is END-START long p-values vector which sums to one.
  template <typename T>
  bool nnetwork<T>::reverse_kl_divergence_gradient_j(const math::vertex<T>& output,
						     const unsigned int START, const unsigned int END,
						     const math::vertex<T>& correct_pvalues,
						     const math::matrix<T>& grad, // jacobian matrix
						     math::vertex<T>& entropy_gradient) const
  {
    if(START >= END) return false;
    if(START >= output.size() || END > output.size()) return false;
    if(output.size() != output_size()) return false;

    // calculates softmax probabilities
    std::vector<T> softmax_values;
    T total_sum = T(0.0f);
    const T epsilon = T(1e-20);

    for(unsigned int i=START;i<END;i++){
      T value = output[i];
      if(value >= T(+25.0f)) value = T(+25.0f);
      else if(value <= T(-25.0f)) value = T(-25.0f);

      value = math::exp(value);

      softmax_values.push_back(value);
      total_sum += value;
    }

    // entropy error term is e = S*h, calculates h first
    math::vertex<T> h;
    h.resize(END - START);
    
    for(unsigned int i=0;i<softmax_values.size();i++){
      h[i] = T(1.0) + math::log(softmax_values[i]/total_sum + epsilon) - math::log(correct_pvalues[i] + epsilon);
    }

    // calculates S^t matrix
    math::matrix<T> St;
    St.resize(softmax_values.size(), output.size());
    St.zero();

    for(unsigned int i=START;i<END;i++){
      const T scaling = softmax_values[i-START]/(total_sum*total_sum);

      for(unsigned int j=START;j<END;j++){
	if(i==j) St(i,j) = scaling*(total_sum - softmax_values[j]);
	else St(i,j) = scaling*(-softmax_values[j]);
      }
      
    }

    // calculates error vector e used for a backpropagation input

    entropy_gradient = (h*St)*grad;
    
    return true;
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
#define FNN_VERSION_CFGSTR          "FNN_CONFIG_VERSION"
#define FNN_ARCH_CFGSTR             "FNN_ARCH"
#define FNN_VWEIGHTS_CFGSTR         "FNN_WEIGHTS"
#define FNN_LAYER_W_CFGSTR          "FNN_LWEIGHT%d"
#define FNN_LAYER_B_CFGSTR          "FNN_LBIAS%d"
#define FNN_AFUNS_CFGSTR            "FNN_AFUNCTIONS" // symbolic names for activation function
#define FNN_LAYER_AFUN_PARAM_CFGSTR "FNN_LAFPARAM%d"
#define FNN_MOMENS_CFGSTR           "FNN_MOMENTS"
#define FNN_LRATES_CFGSTR           "FNN_LRATES"

  // version 3.5 labels
#define FNN_NUMWEIGHTS_CFGSTR       "FNN_NUM_WEIGHTS"
#define FNN_WEIGHTS_CFGSTR          "FNN_WEIGHTS%d"

#define FNN_STOCHASTIC_CFGSTR       "FNN_STOCHASTIC"
#define FNN_NONLINEARITY_CFGSTR     "FNN_NONLINEAR"
#define FNN_FROZEN_CFGSTR           "FNN_FROZEN"
#define FNN_TIMESTAMP_CFGSTR        "FNN_TIMESTAMP"
#define FNN_RETAIN_CFGSTR           "FNN_RETAIN"
#define FNN_RESIDUAL_CFGSTR         "FNN_RESIDUAL"
  
#define FNN_BATCH_NORM_CFGSTR       "FNN_BATCHNORM"
#define FNN_BN_DATA_CFGSTR          "FNN_BN_DATA"

  //////////////////////////////////////////////////////////////////////

  template <typename T>
  bool nnetwork<T>::save(const std::string& filename) const
  {
    try{
      whiteice::dataset<T> conf;
      whiteice::math::vertex<T> data;

      // writes version information
      {
	if(conf.createCluster(FNN_VERSION_CFGSTR, 1) == false) return false;
	// version number = float
	data.resize(1);
	data[0] = T(3.200); // version 3.2 with BN data
	if(conf.add(0, data) == false) return false;
      }

      // writes architecture information
      {
	if(conf.createCluster(FNN_ARCH_CFGSTR, arch.size()) == false) return false;
	data.resize(arch.size());
	for(unsigned int i=0;i<arch.size();i++)
	  data[i] = T((float)arch[i]);
	if(conf.add(1, data) == false) return false;
      }

      // weights: we just convert everything to a big vertex vector and write it
      {
	if(this->exportdata(data) == false) return false;
	
	if(conf.createCluster(FNN_VWEIGHTS_CFGSTR, data.size()) == false) return false;
	if(conf.add(2, data) == false) return false;
      }

      // used non-linearity
      {
	data.resize(this->nonlinearity.size());

	for(unsigned int l=0;l<nonlinearity.size();l++){
	  if(nonlinearity[l] == sigmoid){
	    data[l] = T(0.0);
	  }
	  else if(nonlinearity[l] == stochasticSigmoid){
	    data[l] = T(1.0);
	  }	  
	  else if(nonlinearity[l] == halfLinear){
	    data[l] = T(2.0);
	  }
	  else if(nonlinearity[l] == pureLinear){
	    data[l] = T(3.0);
	  }
	  else if(nonlinearity[l] == tanh){
	    data[l] = T(4.0);
	  }
	  else if(nonlinearity[l] == rectifier){
	    data[l] = T(5.0);
	  }
	  else if(nonlinearity[l] == softmax){
	    data[l] = T(6.0);
	  }
	  else if(nonlinearity[l] == tanh10){
	    data[l] = T(7.0);
	  }
	  else if(nonlinearity[l] == hermite){
	    data[l] = T(8.0);
	  }
	  else return false; // error!
	}

	if(conf.createCluster(FNN_NONLINEARITY_CFGSTR, data.size()) == false) return false;
	if(conf.add(3, data) == false) return false;
      }

      // frozen status of each layer
      {
	data.resize(this->frozen.size());
	
	for(unsigned int l=0;l<frozen.size();l++){
	  if(frozen[l] == false){
	    data[l] = T(0.0);
	  }
	  else{
	    data[l] = T(1.0);
	  }
	}

	if(conf.createCluster(FNN_FROZEN_CFGSTR, data.size()) == false) return false;
	if(conf.add(4, data) == false) return false;
      }

      // saves retain probability and other parameters
      {
	data.resize(1);
	data[0] = this->retain_probability;

	if(conf.createCluster(FNN_RETAIN_CFGSTR, data.size()) == false) return false;
	if(conf.add(5, data) == false) return false;
      }

      // saves boolean flags (residual neural network for now)
      {
	data.resize(1);
	data[0] = T((float)this->residual);

	if(conf.createCluster(FNN_RESIDUAL_CFGSTR, data.size()) == false) return false;
	if(conf.add(6, data) == false) return false;
      }
      
      // saves boolean flags (batch norm flag for now)
      {
	data.resize(1);
	data[0] = T((float)this->batchnorm);

	if(conf.createCluster(FNN_BATCH_NORM_CFGSTR, data.size()) == false) return false;
	if(conf.add(7, data) == false) return false;
      }

      // weights: we just convert everything to a big vertex vector and write it
      {
	if(batchnorm){
	  if(this->exportBNdata(data) == false) return false;
	}
	else{
	  data.resize(1);
	  data[0] = T(0.0f);
	}
	
	if(conf.createCluster(FNN_BN_DATA_CFGSTR, data.size()) == false) return false;
	if(conf.add(8, data) == false) return false;
      }

      // timestamp
      {
	char buffer[128];
	time_t now = time(0);
	snprintf(buffer, 128, "%s", ctime(&now));
	
	std::string timestamp = buffer;

	if(conf.createCluster(FNN_TIMESTAMP_CFGSTR, timestamp.length()) == false)
	  return false;
	if(conf.add(9, timestamp) == false) return false;
      }

      // don't save dropout or retain probability
      
      return conf.save(filename);
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception "
		<< "File: " << __FILE__ << " "
		<< "Line: " << __LINE__ << " "
		<< e.what() << std::endl;
      
      return false;
    }

    return false;
  } 
  
  
  ///////////////////////////////////////////////////////////////////////////
  

  // load neuralnetwork data from file
  template <typename T>
  bool nnetwork<T>::load(const std::string& filename)
  {
    try{
      whiteice::dataset<T> conf;
      whiteice::math::vertex<T> conf_data;

      // loaded parameters of neural network
      
      std::vector<unsigned int> conf_arch;
      whiteice::math::vertex<T> conf_weights;
      std::vector<nonLinearity> conf_nonlins;
      std::vector<bool> conf_frozen;
      T conf_retain = T(1.0);
      bool conf_residual = false;
      bool conf_batchnorm = false;
      whiteice::math::vertex<T> conf_bn_data;
      
      if(conf.load(filename) == false) return false;

      // checks version number
      {
	const unsigned int cluster = conf.getCluster(FNN_VERSION_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false;
	if(conf.dimension(cluster) != 1) return false;

	conf_data = conf.access(cluster, 0);
	
	if(conf_data[0] != T(3.200)) // only handles version 3.2 files (3.2 has batch norm info)
	  return false;
      }

      // checks number of clusters (10 in version 3.2 files)
      {
	if(conf.getNumberOfClusters() != 10) return false;
      }

      // gets architecture information
      {
	const int unsigned cluster = conf.getCluster(FNN_ARCH_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) < 2) return false; // bad architecture size

	conf_data = conf.access(cluster, 0);

	if(conf_data.size() < 2) return false; // bad architecture size;

	conf_arch.resize(conf_data.size());

	for(unsigned int i=0;i<conf_arch.size();i++){
	  int value = 0;
	  whiteice::math::convert(value, conf_data[i].first());

	  if(value <= 0) return false; // bad data

	  conf_arch[i] = (unsigned int)value;
	}
      }

      // gets weights parameter vector
      {
	const unsigned int cluster = conf.getCluster(FNN_VWEIGHTS_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) < 2) return false; // bad weight size
	
	conf_weights = conf.access(cluster, 0);
      }

      // gets non-linearities
      {
	const unsigned int cluster = conf.getCluster(FNN_NONLINEARITY_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) != conf_arch.size()-1)
	  return false; // number of layers don't match architecture size

	conf_data = conf.access(cluster, 0);

	conf_nonlins.resize(conf_data.size());

	for(unsigned int l=0;l<conf_data.size();l++){
	  if(conf_data[l] == T(0.0)) conf_nonlins[l] = sigmoid;
	  else if(conf_data[l] == T(1.0)) conf_nonlins[l] = stochasticSigmoid;
	  else if(conf_data[l] == T(2.0)) conf_nonlins[l] = halfLinear;
	  else if(conf_data[l] == T(3.0)) conf_nonlins[l] = pureLinear;
	  else if(conf_data[l] == T(4.0)) conf_nonlins[l] = tanh;
	  else if(conf_data[l] == T(5.0)) conf_nonlins[l] = rectifier;
	  else if(conf_data[l] == T(6.0)) conf_nonlins[l] = softmax;
	  else if(conf_data[l] == T(7.0)) conf_nonlins[l] = tanh10;
	  else if(conf_data[l] == T(8.0)) conf_nonlins[l] = hermite;
	  else return false; // unknown non-linearity
	}
      }

      // gets frozen status of each layer
      {
	const unsigned int cluster = conf.getCluster(FNN_FROZEN_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) != conf_arch.size()-1)
	  return false; // number of layers don't match architecture size
	
	conf_data = conf.access(cluster, 0);

	conf_frozen.resize(conf_data.size());

	for(unsigned int l=0;l<conf_frozen.size();l++){
	  if(conf_data[l] == T(0.0)) conf_frozen[l] = false;
	  else if(conf_data[l] == T(1.0)) conf_frozen[l] = true;
	  else return false; // unknown value
	}
      }

      // gets misc parameters (retain_probability)
      {
	const unsigned int cluster = conf.getCluster(FNN_RETAIN_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) != 1)
	  return false; // only single parameter saved
	
	conf_data = conf.access(cluster, 0);

	if(conf_data.size() != 1) return false;

	conf_retain = conf_data[0];

	if(abs(conf_retain) < abs(T(0.0)) || abs(conf_retain) > abs(T(1.0)))
	  return false; // correct interval for data
      }

      // gets boolean parameters (residual flag)
      {
	const unsigned int cluster = conf.getCluster(FNN_RESIDUAL_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) != 1)
	  return false; // only single parameter saved
	
	conf_data = conf.access(cluster, 0);

	if(conf_data.size() != 1) return false;

	if(conf_data[0] == T(0.0f)) conf_residual = false;
	else if(conf_data[0] == T(1.0f)) conf_residual = true;
	else return false;
      }

      // gets boolean parameters (batchnorm flag)
      {
	const unsigned int cluster = conf.getCluster(FNN_BATCH_NORM_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) != 1)
	  return false; // only single parameter saved
	
	conf_data = conf.access(cluster, 0);

	if(conf_data.size() != 1) return false;

	if(conf_data[0] == T(0.0f)) conf_batchnorm = false;
	else if(conf_data[0] == T(1.0f)) conf_batchnorm = true;
	else return false;
      }

      // gets batch norm parameters vector
      {
	const unsigned int cluster = conf.getCluster(FNN_BN_DATA_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) < 1) return false; // bad weight size
	
	conf_bn_data = conf.access(cluster, 0);
      }

      
      // don't check timestamp metainformation

      // parameters where successfully loaded from the disk.
      // now calculates all the parameters and set them as neural network values
      {
	this->arch = conf_arch;

	this->W.resize(arch.size()-1);
	this->b.resize(arch.size()-1);

	unsigned int memuse = 0;
	maxwidth = arch[0];
	
	unsigned int i = 1;
	while(i < arch.size()){
	  memuse += (arch[i-1] + 1)*arch[i];

	  this->W[i-1].resize(arch[i], arch[i-1]);
	  this->b[i-1].resize(arch[i]);
			      
	  if(arch[i] > maxwidth)
	    maxwidth = arch[i];
	  i++;
	}
	
	hasValidBPData = false;
	bpdata.clear();
	size = memuse;
	
	inputValues.resize(arch[0]);
	outputValues.resize(arch[arch.size() - 1]);

	this->frozen = conf_frozen;
	this->nonlinearity = conf_nonlins;
	this->retain_probability = conf_retain;
	this->residual = conf_residual;
	this->batchnorm = conf_batchnorm;

	if(this->importdata(conf_weights) == false) // this should never fail
	  return false;
	
	if(this->batchnorm){
	  this->setBatchNorm(true);
	  if(this->importBNdata(conf_bn_data) == false) return false;
	}
	
	
	dropout.clear(); // dropout is disabled in saved networks
      }

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

  
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  
  
  // exports and imports neural network parameters to/from vertex
  template <typename T>
  bool nnetwork<T>::exportdata(math::vertex<T>& v) const
  {
    v.resize(size);

    unsigned int index = 0;

    for(unsigned int l=0;l<getLayers();l++){
      if(W[l].save_to_vertex(v, index) == false)
	return false;

      index += W[l].size();

      if(b[l].exportData(&(v[index])) == false)
	return false;

      index += b[l].size();
    }

    return true;
    
    // NN exports TO vertex, vertex imports FROM NN
    // return v.importData(&(data[0]), size, 0);
  }
  
  template <typename T>
  bool nnetwork<T>::importdata(const math::vertex<T>& v)
  {
    if(v.size() != size)
      return false;

    unsigned int index = 0;
    
    for(unsigned int l=0;l<getLayers();l++){
      if(W[l].load_from_vertex(v, index) == false)
	return false;

      index += W[l].size();

      if(b[l].importData(&(v[index])) == false)
	return false;

      index += b[l].size();
    }

#ifdef _GLIBCXX_DEBUG
    // ONLY DO RANGE CHECKS IF DEBUGGING FLAG IS DEFINED
    
    const bool verbose = false;

    // slow temp copy of data back for inspection
    math::vertex<T> data;
    if(this->exportdata(data) == false) return false;
    bool bad_data = false;
    
    // "safebox" (keeps data always within sane levels)
    for(unsigned int i=0;i<data.size();i++){
      if(whiteice::math::isnan(data[i])){
	if(verbose)
	  std::cout << "nnetwork::importdata() warning. bad NaN data: " << data[i] << std::endl;
	data[i] = T(0.0f);
	bad_data = true;
      }
      
      if(whiteice::math::isinf(data[i])){
	if(verbose)
	  std::cout << "nnetwork::importdata() warning. bad Inf data: " << data[i] << std::endl;
	data[i] = T(0.0f);
	bad_data = true;
      }

      for(unsigned int k=0;k<data[i].size();k++){
	if(data[i][k].real() > 10000.0f){
	  if(verbose)
	    std::cout << "nnetwork::importdata() warning. bad real data: " << data[i] << std::endl;
	  data[i][k].real(+10000.0f);
	  bad_data = true;
	}
	else if(data[i][k].real() < -10000.0f){
	  if(verbose)
	    std::cout << "nnetwork::importdata() warning. bad real data: " << data[i] << std::endl;
	  data[i][k].real(-10000.0f);
	  bad_data = true;
      }
	
	if(data[i][k].imag() > 10000.0f){
	  if(verbose)
	    std::cout << "nnetwork::importdata() warning. bad imag data: " << data[i] << std::endl;
	  data[i][k].imag(+10000.0f);
	  bad_data = true;
	}
	else if(data[i][k].imag() < -10000.0f){
	  if(verbose)
	    std::cout << "nnetwork::importdata() warning. bad imag data: " << data[i] << std::endl;
	  data[i][k].imag(-10000.0f);
	  bad_data = true;
	}
      }
    }

    if(bad_data){
      // re-import fixed data to internal data structures

      unsigned int index = 0;
      
      for(unsigned int l=0;l<getLayers();l++){
	if(W[l].load_from_vertex(data, index) == false)
	  return false;
	
	index += W[l].size();
	
	if(b[l].importData(&(data[index])) == false)
	  return false;
	
	index += b[l].size();
      }
      
    }
    
#endif
    
    return true;
  }
  
  
  // number of dimensions used by import/export
  template <typename T>
  unsigned int nnetwork<T>::exportdatasize() const {
    return size;
  }

  ////////////////////////////////////////////////////////////////////////////
  
  // exports and imports bathc norm parameters to/from vertex
  template <typename T>
  bool nnetwork<T>::exportBNdata(math::vertex<T>& v) const
  {
    if(batchnorm == false) return false;
    
    unsigned int bnsize = 0;
    
    for(unsigned int i=0;i<bn_mu.size();i++)
      bnsize += bn_mu[i].size();

    for(unsigned int i=0;i<bn_sigma.size();i++)
      bnsize += bn_sigma[i].size();
    
    v.resize(bnsize);

    unsigned int index = 0;

    for(unsigned int l=0;l<getLayers();l++){
      if(bn_mu[l].exportData(&v[index]) == false)
	return false;

      index += bn_mu[l].size();

      if(bn_sigma[l].exportData(&(v[index])) == false)
	return false;

      index += bn_sigma[l].size();
    }

    return true;
  }


  // exports and imports batch norm parameters to/from vertex
  template <typename T>
  bool nnetwork<T>::importBNdata(const math::vertex<T>& v)
  {
    if(batchnorm == false) return false;
    
    unsigned int bnsize = 0;
    
    for(unsigned int i=0;i<bn_mu.size();i++)
      bnsize += bn_mu[i].size();

    for(unsigned int i=0;i<bn_sigma.size();i++)
      bnsize += bn_sigma[i].size();

    if(bnsize != v.size()) return false;
    
    unsigned int index = 0;

    for(unsigned int l=0;l<getLayers();l++){
      if(bn_mu[l].importData(&v[index]) == false)
	return false;

      index += bn_mu[l].size();

      if(bn_sigma[l].importData(&(v[index])) == false)
	return false;

      index += bn_sigma[l].size();
    }

    return true;
  }

  
  

  ////////////////////////////////////////////////////////////////////////////

#if 0
  template <typename T>
  unsigned int nnetwork<T>::getLayers() const {
    return (arch.size()-1); 
  }

  // number of neurons per layer
  template <typename T>
  unsigned int nnetwork<T>::getNeurons(unsigned int layer) const 
  {
    if(layer+1 >= arch.size()) return 0;
    return arch[layer+1];
  }

  // number of inputs per layer
  template <typename T>
  unsigned int nnetwork<T>::getInputs(unsigned int layer) const 
  {
    if(layer >= arch.size()-1) return 0;
    return arch[layer];
  }
#endif

  template <typename T>
  bool nnetwork<T>::getBias(math::vertex<T>& b, unsigned int layer) const 
  {
    if(layer >= this->b.size()) return false;

    b = this->b[layer];

    return true;
  }

  
  template <typename T>
  bool nnetwork<T>::setBias(const math::vertex<T>& b, unsigned int layer) 
  {
    if(layer >= this->b.size()) return false;

    if(b.size() != this->b[layer].size()) return false;

    this->b[layer] = b;

    return true;
  }
  

  template <typename T>
  bool nnetwork<T>::getWeights(math::matrix<T>& W, unsigned int layer) const 
  {
    if(layer >= this->W.size()) return false;

    W = this->W[layer];

    return true;
  }

  
  template <typename T>
  bool nnetwork<T>::setWeights(const math::matrix<T>& W, unsigned int layer) 
  {
    if(layer >= this->W.size()) return false;

    if(W.ysize() != this->W[layer].ysize() || W.xsize() != this->W[layer].xsize())
      return false;

    this->W[layer] = W;
    
    return true;
  }
  

  template <typename T>
  bool nnetwork<T>::setNonlinearity(nnetwork<T>::nonLinearity nl)
  {
    for(unsigned int l=0;l<nonlinearity.size();l++)
      nonlinearity[l] = nl;

    // last layer is always linear!! (for now)
    // nonlinearity[nonlinearity.size()-1] = pureLinear; 
    
    return true;
  }

  template <typename T>
  typename nnetwork<T>::nonLinearity nnetwork<T>::getNonlinearity(unsigned int layer) const 
  {
    if(layer >= nonlinearity.size()) return pureLinear; // silent failure..

    return nonlinearity[layer];
  }

  template <typename T>
  bool nnetwork<T>::setNonlinearity(unsigned int layer, nonLinearity nl)
  {
    if(layer >= nonlinearity.size()) return false;
    
    nonlinearity[layer] = nl;
    
    return true;
  }
  
  template <typename T>
  void nnetwork<T>::getNonlinearity(std::vector<nonLinearity>& nls) const 
  {
    nls = nonlinearity;
  }

  template <typename T>
  bool nnetwork<T>::setNonlinearity(const std::vector<nonLinearity>& nls) 
  {
    if(nonlinearity.size() != nls.size()) return false;

    nonlinearity = nls;
    return true;
  }

  
  template <typename T>
  bool nnetwork<T>::setFrozen(unsigned int layer, bool f)
  {
    if(layer >= frozen.size())
      return false;

    frozen[layer] = f;
    return true;
  }

  
  template <typename T>
  bool nnetwork<T>::setFrozen(const std::vector<bool>& f)
  {
    if(this->frozen.size() != f.size()) return false;

    this->frozen = f;
    
    return true;
  }

  
  template <typename T>
  bool nnetwork<T>::getFrozen(unsigned int layer) const
  {
    if(layer < frozen.size()) return frozen[layer];
    else return false; // silent error
  }

  
  template <typename T>
  void nnetwork<T>::getFrozen(std::vector<bool>& frozen) const
  {
    frozen = this->frozen;
  }

  // creates subnet starting from fromLayer:th layer to the output
  template <typename T>
  nnetwork<T>* nnetwork<T>::createSubnet(const unsigned int fromLayer)
  {
    if(fromLayer >= getLayers()) return nullptr;

    std::vector<unsigned int> a;

    for(unsigned int i=fromLayer;i<arch.size();i++)
      a.push_back(arch[i]);

    nnetwork<T>* nn = new nnetwork<T>(a);

    math::matrix<T> W;
    math::vertex<T> b;

    // sets parameters of the network
    for(unsigned int i=fromLayer;i<(arch.size()-1);i++){
      nn->frozen[i-fromLayer] = this->frozen[i];
      nn->nonlinearity[i-fromLayer] = this->nonlinearity[i];

      nn->W[i-fromLayer] = this->W[i];
      nn->b[i-fromLayer] = this->b[i];
    }

    return nn;
  }

  // creates subnet starting from fromLayer:th layer to the toLayer
  template <typename T>
  nnetwork<T>* nnetwork<T>::createSubnet(const unsigned int fromLayer,
					 const unsigned int toLayer)
  {
    if(fromLayer >= getLayers()) return nullptr;
    if(toLayer >= arch.size() || fromLayer >= toLayer) return nullptr;

    std::vector<unsigned int> a;

    for(unsigned int i=fromLayer;i<=(toLayer+1);i++)
      a.push_back(arch[i]);

    nnetwork<T>* nn = new nnetwork<T>(a);

    math::matrix<T> W;
    math::vertex<T> b;

    // sets parameters of the network
    for(unsigned int i=fromLayer;i<=toLayer;i++){
      nn->frozen[i-fromLayer] = this->frozen[i];
      nn->nonlinearity[i-fromLayer] = this->nonlinearity[i];

      nn->W[i-fromLayer] = this->W[i];
      nn->b[i-fromLayer] = this->b[i];
    }

    return nn;
  }

  // injects (if possible) subnet into net starting from fromLayer:th layer
  template <typename T>
  bool nnetwork<T>::injectSubnet(const unsigned int fromLayer, nnetwork<T>* nn)
  {
    if(nn == nullptr) return false;

    // check if architecture matches exactly
    if(this->arch.size() != nn->arch.size() + fromLayer)
      return false;

    for(unsigned int i=0;i<nn->arch.size();i++){
      if(arch[fromLayer+i] != nn->arch[i])
	return false;
    }

    for(unsigned int l=0;l<nn->getLayers();l++){
      if(nonlinearity[fromLayer+l] != nn->nonlinearity[l])
	return false;
    }

    for(unsigned int i=0;i<nn->getLayers();i++){
      this->W[fromLayer+i] = nn->W[i];
      this->b[fromLayer+i] = nn->b[i];
    }

    return true;
  }
  
  
  template <typename T>
  unsigned int nnetwork<T>::getSamplesCollected() const 
  {
    if(samples.size() > 0)
      return samples[0].size();
    else
      return 0;
  }
  
  template <typename T>
  bool nnetwork<T>::getSamples(std::vector< math::vertex<T> >& samples,
			       unsigned int layer,
			       unsigned int MAXSAMPLES) const 
  {
    if(layer >= this->samples.size()) return false;
    
    if(MAXSAMPLES == 0 || MAXSAMPLES >= this->samples[layer].size()){
      samples = this->samples[layer];
      return true;
    }
    
    for(unsigned int i=0;i<MAXSAMPLES;i++){
      const unsigned int index = rng.rand() % this->samples[layer].size();
      samples.push_back(this->samples[layer][index]);
    }
    
    return true;
  }
  
  template <typename T>
  void nnetwork<T>::clearSamples() 
  {
    for(auto& s : samples)
      s.clear();
  }

  template <typename T>
  bool nnetwork<T>::setDropOut(const T probability) 
  {
    if(real(probability.first()) <= 0.0f || real(probability.first()) > 1.0f)
      return false; // we cannot set all neurons to be dropout neurons

    retain_probability = probability;
    dropout.resize(getLayers());
    
    for(unsigned int l=0;l<dropout.size();l++){
      dropout[l].resize(getNeurons(l));
      if(l != (dropout.size()-1)){
	unsigned int numdropped = 0;

	for(unsigned int i=0;i<dropout[l].size();i++){
	  if(real(T(rng.uniformf()).first()) > real(retain_probability.first())){
	    dropout[l][i] = true;
	    numdropped++;
	  }
	  else{
	    dropout[l][i] = false;
	  }
	}

	// if we are about to drop all nodes we always
	// randomly keep at least one node
	if(numdropped == dropout[l].size()){
	  const unsigned int index = rng.rand() % dropout[l].size();
	  dropout[l][index] = false;
	}
      }
      else{ // we always keep all last layer nodes
	for(unsigned int i=0;i<dropout[l].size();i++){
	  dropout[l][i] = false;
	}
      }
    }

    return true;
  }


  template <typename T>
  bool nnetwork<T>::setDropOut(std::vector< std::vector<bool> >& dropout,
			       const T probability) const
  {
    if(real(probability.first()) <= 0.0f || real(probability.first()) > 1.0f)
      return false; // we cannot set all neurons to be dropout neurons

    dropout.resize(getLayers());
    
    for(unsigned int l=0;l<dropout.size();l++){
      dropout[l].resize(getNeurons(l));
      if(l != (dropout.size()-1)){
	unsigned int numdropped = 0;

	for(unsigned int i=0;i<dropout[l].size();i++){
	  if(real(T(rng.uniformf()).first()) > real(probability.first())){
	    dropout[l][i] = true;
	    numdropped++;
	  }
	  else{
	    dropout[l][i] = false;
	  }
	}

	// if we are about to drop all nodes we always
	// randomly keep at least one node
	if(numdropped == dropout[l].size()){
	  const unsigned int index = rng.rand() % dropout[l].size();
	  dropout[l][index] = false;
	}
      }
      else{ // we always keep all last layer nodes
	for(unsigned int i=0;i<dropout[l].size();i++){
	  dropout[l][i] = false;
	}
      }
    }

    return true;
  }
  

  template <typename T>
  bool nnetwork<T>::removeDropOut(T probability) 
  {    
    // scales weights according to retain_probability
    // (except the first layer where we always keep all inputs)

    //for(unsigned int l=1;l<getLayers();l++){ // FIX BUG HERE
    for(unsigned int l=0;l<(getLayers()-1);l++){
      W[l] *= probability;
    }

    dropout.clear();

    retain_probability = T(1.0f);

    return true;
  }

  // remove all drop-out
  template <typename T>
  void nnetwork<T>::clearDropOut() 
  {
    retain_probability = T(1.0);
    dropout.clear();
  }


  /////////////////////////////////////////////////////////////////////////////

  // residual neural network support
  template <typename T>
  void nnetwork<T>::setResidual(const bool residual)
  {
    this->residual = residual;
  }

  template <typename T>
  bool nnetwork<T>::getResidual() const
  {
    return residual;
  }

  /////////////////////////////////////////////////////////////////////////////
  // batch normalization

  template <typename T> 
  void nnetwork<T>::setBatchNorm(const bool bn)
  {
    this->batchnorm = bn;

    if(bn){
      bn_sigma.resize(b.size());
      bn_mu.resize(b.size());

      for(unsigned int i=0;i<b.size();i++){
	bn_mu[i].resize(b[i].size());
	bn_sigma[i].resize(b[i].size());

	bn_mu[i].zero();
	bn_sigma[i].ones();
      }
    }
  }

  template <typename T>
  bool nnetwork<T>::getBatchNorm()
  {
    return batchnorm;
  }

  
  template <typename T>
  bool nnetwork<T>::calculateBatchNorm(const std::vector< math::vertex<T> >& data)
  {
    if(batchnorm == false) return false;
    if(data.size() <= 2) return false;

    // resets batch normalization constants
    {
      bn_sigma.resize(b.size());
      bn_mu.resize(b.size());
      
      for(unsigned int i=0;i<b.size();i++){
	bn_mu[i].resize(b[i].size());
	bn_sigma[i].resize(b[i].size());
	
	bn_mu[i].zero();
	bn_sigma[i].ones();
      }
    }
    

    // calculates batch normalization constants using data

    std::vector< math::vertex<T> > u;

    for(unsigned int j=1;j<W.size();j++){
      u.clear();

      // calculates outputs for the data
      for(const auto& v : data){

	auto state = v;

	math::vertex<T> skipValue;

	if(residual) skipValue = state;

	for(unsigned int l=0;l<j;l++){
	    
	  if(residual && (l % 2) == 0 && l != 0 && W[l].ysize() == skipValue.size())
	    state = W[l]*state + b[l] + skipValue;
	  else
	    state = W[l]*state + b[l];
	  
	  for(unsigned int i=0;i<state.size();i++){
	    state[i] = nonlin(state[i], l, i);
	  }
	  
	  if(residual && (l % 2) == 0 && l != 0)
	    skipValue = state;
	}

	u.push_back(state);
      }

      // calculates new mean and standard deviations
      auto& mu = bn_mu[j-1];
      auto& sigma = bn_sigma[j-1];
      
      mu.zero();
      sigma.zero();

      for(const auto& v : u){
	mu += v;

	for(unsigned int i=0;i<v.size();i++)
	  sigma[i] += v[i]*v[i];
      }

      mu /= T(u.size());
      sigma /= T(u.size());

      T epsilon;
      epsilon.ones();
      epsilon = epsilon*T(1e-3);

      for(unsigned int i=0;i<mu.size();i++){
	sigma[i] -= mu[i]*mu[i];
	sigma[i] = whiteice::math::abs(sigma[i]);
	sigma[i] += epsilon;
	sigma[i] = whiteice::math::sqrt(sigma[i]);
      }

      
    }

    return true; 
  }

  

  /////////////////////////////////////////////////////////////////////////////
  
  template <typename T>
  inline void nnetwork<T>::gemv_gvadd(unsigned int yd, unsigned int xd, 
				      const T* W, T* x, T* y,
				      unsigned int dim, T* s, const T* b,
				      T* temp) const
  {
    // calculates y = b + W*x (y == x)

#ifdef CUBLAS
    // memory areas W, x and y and temp must be in NVIDIA GPU device!!! [FIX ME]
    // temp must be maxwidth vertex in GPU memory

    if(typeid(T) == typeid(whiteice::math::blas_real<float>)){

      auto err = cudaMemcpy(temp, b, yd*sizeof(T), cudaMemcpyDeviceToDevice);

      if(err != cudaSuccess){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cudaMemcpy() failed.");
	throw CUDAException("CUBLAS cudaMemcpy() call failed.");
      }
      
      T alpha = T(1.0f);
      T beta  = T(1.0f);

      auto s = cublasSgemv(cublas_handle, CUBLAS_OP_N, yd, xd,
			   (const float*)&alpha,
			   (const float*)W, yd, (const float*)x, 1,
			   (const float*)&beta, (float*)temp, 1);

      if(s != CUBLAS_STATUS_SUCCESS){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cublasSgemv() failed.");
	throw CUDAException("CUBLAS cublasSgemv() call failed.");
      }

      
      err = cudaMemcpy((void*)y, temp, yd*sizeof(T), cudaMemcpyDeviceToDevice);
      
      if(err != cudaSuccess){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cudaMemcpy() failed.");
	throw CUDAException("CUBLAS cudaMemcpy() call failed.");
      }

      gpu_sync();
      
    }
    else if(typeid(T) == typeid(whiteice::math::blas_real<double>)){

      auto err = cudaMemcpy(temp, b, yd*sizeof(T), cudaMemcpyDeviceToDevice);
      
      if(err != cudaSuccess){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cudaMemcpy() failed.");
	throw CUDAException("CUBLAS cudaMemcpy() call failed.");
      }
      
      T alpha = T(1.0f);
      T beta  = T(1.0f);

      auto s = cublasDgemv(cublas_handle, CUBLAS_OP_N, yd, xd,
			   (const double*)&alpha,
			   (const double*)W, yd, (const double*)x, 1,
			   (const double*)&beta, temp, 1);

      if(s != CUBLAS_STATUS_SUCCESS){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cublasDgemv() failed.");
	throw CUDAException("CUBLAS cublasDgemv() call failed.");
      }

      err = cudaMemcpy((void*)y, temp, yd*sizeof(T), cudaMemcpyDeviceToDevice);
      
      if(err != cudaSuccess){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cudaMemcpy() failed.");
	throw CUDAException("CUBLAS cudaMemcpy() call failed.");
      }

      gpu_sync();
      
    }
    else if(typeid(T) == typeid(whiteice::math::blas_complex<float>)){

      auto err = cudaMemcpy(temp, b, yd*sizeof(T), cudaMemcpyDeviceToDevice);
      
      if(err != cudaSuccess){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cudaMemcpy() failed.");
	throw CUDAException("CUBLAS cudaMemcpy() call failed.");
      }
      
      T alpha = T(1.0f);
      T beta  = T(1.0f);

      auto s = cublasCgemv(cublas_handle, CUBLAS_OP_N, yd, xd,
			   (const cuComplex*)&alpha,
			   (const cuComplex*)W, yd, (const cuComplex*)x, 1,
			   (const cuComplex*)&beta, (cuComplex*)temp, 1);

      if(s != CUBLAS_STATUS_SUCCESS){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cublasCgemv() failed.");
	throw CUDAException("CUBLAS cublasCgemv() call failed.");
      }

      err = cudaMemcpy((void*)y, temp, yd*sizeof(T), cudaMemcpyDeviceToDevice);
      
      if(err != cudaSuccess){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cudaMemcpy() failed.");
	throw CUDAException("CUBLAS cudaMemcpy() call failed.");
      }

      gpu_sync();
      
    }
    else if(typeid(T) == typeid(whiteice::math::blas_complex<double>)){
      
      auto err = cudaMemcpy(temp, b, yd*sizeof(T), cudaMemcpyDeviceToDevice);

      if(err != cudaSuccess){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cudaMemcpy() failed.");
	throw CUDAException("CUBLAS cudaMemcpy() call failed.");
      }

      T alpha = T(1.0);
      T beta  = T(1.0);

      auto s = cublasZgemv(cublas_handle, CUBLAS_OP_N, yd, xd,
			   (const cuDoubleComplex*)&alpha,
			   (const cuDoubleComplex*)W, yd, (const cuDoubleComplex*)x, 1,
			   (const cuDoubleComplex*)&beta, (cuDoubleComplex*)temp, 1);

      if(s != CUBLAS_STATUS_SUCCESS){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cublasZgemv() failed.");
	throw CUDAException("CUBLAS cublasZgemv() call failed.");
      }

      err = cudaMemcpy((void*)y, temp, yd*sizeof(T), cudaMemcpyDeviceToDevice);
      
      if(err != cudaSuccess){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cudaMemcpy() failed.");
	throw CUDAException("CUBLAS cudaMemcpy() call failed.");
      }

      gpu_sync();
    }
    else{
      for(unsigned int j=0;j<yd;j++){
	T sum = b[j];
	for(unsigned int i=0;i<xd;i++)
	  sum += W[i*yd + j]*x[i]; // changed to COLUMN MAJOR matrix W
	
	temp[j] = sum;
      }

      auto err = cudaMemcpy((void*)y, temp, yd*sizeof(T), cudaMemcpyDeviceToDevice);
      
      if(err != cudaSuccess){
	whiteice::logging.error("nnetwork<>::gemv_gvadd(): cudaMemcpy() failed.");
	throw CUDAException("CUBLAS cudaMemcpy() call failed.");
      }
    }
#else
    // std::vector<T> temp;
    // temp.resize(maxwidth); // USE GLOBAL VARIABLE (ok??)
    
    if(typeid(T) == typeid(whiteice::math::blas_real<float>)){
      memcpy((T*)temp, (T*)b, yd*sizeof(T));
      
      cblas_sgemv(CblasRowMajor, CblasNoTrans, yd, xd,
		  1.0f, (float*)W, xd, (float*)x, 1, 
		  1.0f, (float*)temp, 1);
      
      memcpy((T*)y, (const T*)temp, yd*sizeof(T));
    }
    else if(typeid(T) == typeid(whiteice::math::blas_real<double>)){
      memcpy((T*)temp, (const T*)b, yd*sizeof(T));
      
      cblas_dgemv(CblasRowMajor, CblasNoTrans, yd, xd,
		  1.0f, (double*)W, xd, (double*)x, 1, 
		  1.0f, (double*)temp, 1);
      
      memcpy((T*)y, (const T*)temp, yd*sizeof(T));
    }
    else if(typeid(T) == typeid(whiteice::math::blas_complex<float>)){
      memcpy((T*)temp, (T*)b, yd*sizeof(T));
      
      whiteice::math::blas_complex<float> a, b;
      a = 1.0f; b = 1.0f;
      
      cblas_cgemv(CblasRowMajor, CblasNoTrans, yd, xd,
		  (float*)(&a), (float*)W, xd, (float*)x, 1, 
		  (float*)(&b), (float*)temp, 1);
      
      memcpy((T*)y, (const T*)temp, yd*sizeof(T));
    }
    else if(typeid(T) == typeid(whiteice::math::blas_complex<double>)){
      memcpy((T*)temp, (const T*)b, yd*sizeof(T));

      whiteice::math::blas_complex<double> a, b;
      a = 1.0; b = 1.0;
      
      cblas_zgemv(CblasRowMajor, CblasNoTrans, yd, xd,
		  (double*)(&a), (double*)W, xd, (double*)x, 1, 
		  (double*)(&b), (double*)temp, 1);
      
      memcpy((T*)y, (const T*)temp, yd*sizeof(T));      
    }
    else{
      for(unsigned int j=0;j<yd;j++){
	T sum = b[j];
	for(unsigned int i=0;i<xd;i++)
	  sum += W[i + j*xd]*x[i];
	
	temp[j] = sum;
      }
      
      memcpy((T*)y, (const T*)temp, yd*sizeof(T));
    }
#endif

  }
  
  
  ////////////////////////////////////////////////////////////
  // can be used to decrease memory usage
  
#if 0
  // changes NN to compressed form of operation or
  // back to normal non-compressed form
  template <typename T>  
  bool nnetwork<T>::compress() {
    if(compressed) return false;
    
    compressor = new MemoryCompressor();
    compressor->setMemory(data, sizeof(T)*size);
    
    if(compressor->compress()){
      free(data); data = 0;
      compressor->setMemory(data, 0);
      compressed = true;
      return true;
    }
    else{
      if(compressor->getTarget() != 0)
	free(compressor->getTarget());
      
      delete compressor;
      compressor = 0;
      return false;
    }
  }
  
  
  template <typename T>
  bool nnetwork<T>::decompress() {
    if(compressed == false) return false;
    
    if(compressor->decompress()){
      data = (T*)( compressor->getMemory() );
      
      free(compressor->getTarget());
      
      delete compressor;
      
      compressor = 0;
      compressed = false;
      
      return true;
    }
    else
      return false;
  }
  
  
  template <typename T>
  bool nnetwork<T>::iscompressed() const {
    return compressed;
  }
  
  
  // returns compression ratio: compressed/orig
  template <typename T>
  float nnetwork<T>::ratio() const {
    if(compressed) return compressor->ratio();
    else return 1.0f;
  }
  
#endif
  
  
  //////////////////////////////////////////////////////////////////////
  
  //template class nnetwork< float >;
  //template class nnetwork< double >;
  
  template class nnetwork< math::blas_real<float> >;
  template class nnetwork< math::blas_real<double> >;

  template class nnetwork< math::blas_complex<float> >;
  template class nnetwork< math::blas_complex<double> >;

  template class nnetwork< math::superresolution< math::blas_real<float>,
						  math::modular<unsigned int> > >;
  template class nnetwork< math::superresolution< math::blas_real<double>,
						  math::modular<unsigned int> > >;

  template class nnetwork< math::superresolution< math::blas_complex<float>,
						  math::modular<unsigned int> > >;
  template class nnetwork< math::superresolution< math::blas_complex<double>,
						  math::modular<unsigned int> > >;
  
};
