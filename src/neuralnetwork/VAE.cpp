/*
 * Variational Autoencoder code
 * Tomas Ukkonen <tomas.ukkonen@iki.fi> 2020
 * 
 * Implementation research paper:
 * "Auto-Encoding Variational Bayes. Diedrik Kingma and Max Welling."
 * Implementation extra math notes in docs directory:
 * [variational-bayes-encoder.tm]
 *
 * NOTE: the input data x ~ p(x) should be roughly distributed as Normal(0,I)
 * because it is assumed that p(x|z) don't estimate variance term but
 * fixes it to be Var[x_i] = 1 for each dimension.
 */

#include "VAE.h"

#include "linear_ETA.h"
#include <list>


namespace whiteice
{

  template <typename T>
  VAE<T>::VAE(const VAE<T>& vae)
  {
    this->encoder = vae.encoder;
    this->decoder = vae.decoder;

    this->minibatchMode = vae.minibatchMode;
  }
  
  template <typename T>
  VAE<T>::VAE(const nnetwork<T>& encoder, // x -> (z_mean, z_var)
	      const nnetwork<T>& decoder) // z -> (x_mean)
    
  {
    if(encoder.output_size() & 1) // odd value
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");
    if(encoder.input_size() != decoder.output_size()) // dim(x) != dim(d(e(x)))
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");
    if(encoder.output_size() != 2*decoder.input_size()) // dim(e(x)/2) != dim(z)
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");

    if(encoder.input_size() <= 0 || decoder.output_size() <= 0 ||
       encoder.output_size() <= 0 || decoder.input_size() <= 0)
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");

    this->encoder = encoder;
    this->decoder = decoder;

    this->minibatchMode = false;
  }

  
  template <typename T>
  VAE<T>::VAE(const std::vector<unsigned int> encoderArchitecture, // x -> (z_mean, z_var)
	      const std::vector<unsigned int> decoderArchitecture) // z -> (x_mean)
    
    
  {
    if(encoderArchitecture.size() < 2 || decoderArchitecture.size() < 2)
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");

    
    if(encoderArchitecture[encoderArchitecture.size()-1] & 1) // odd value
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");
    
    if(encoderArchitecture[0] !=
       decoderArchitecture[decoderArchitecture.size()-1]) // dim(x) != dim(d(e(x)))
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");

    if(2*decoderArchitecture[0] !=
       encoderArchitecture[encoderArchitecture.size()-1]) // dim(e(x)/2) != dim(z)
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");

    for(unsigned int i=0;i<encoderArchitecture.size();i++){
      if(encoderArchitecture[i] <= 0)
	throw std::invalid_argument("bad encoder/decoder dimensions/arch");
    }

    for(unsigned int i=0;i<decoderArchitecture.size();i++){
      if(decoderArchitecture[i] <= 0)
	throw std::invalid_argument("bad encoder/decoder dimensions/arch");
    }
    
    this->encoder.setArchitecture(encoderArchitecture, nnetwork<T>::rectifier);
    this->decoder.setArchitecture(decoderArchitecture, nnetwork<T>::rectifier);

    this->initializeParameters();
    this->minibatchMode = false;
  }
  
  
  template <typename T>
  void VAE<T>::getModel(nnetwork<T>& encoder,
			nnetwork<T>& decoder) const
  {
    encoder = this->encoder;
    decoder = this->decoder;
  }

  template <typename T>
  nnetwork<T>& VAE<T>::getEncoder()
  {
    return this->encoder;
  }

  template <typename T>
  nnetwork<T>& VAE<T>::getDecoder()
  {
    return this->decoder;
  }
  
  template <typename T>
  bool VAE<T>::setModel(const nnetwork<T>& encoder,
			const nnetwork<T>& decoder)
  {
    if(encoder.output_size() & 1) // odd value
      return false;
    if(encoder.input_size() != decoder.output_size()) // dim(x) != dim(d(e(x)))
      return false;
    if(encoder.output_size() != 2*decoder.input_size()) // dim(e(x)) != 2*dim(z)
      return false;

    if(encoder.input_size() <= 0 || decoder.output_size() <= 0 ||
       encoder.output_size() <= 0 || decoder.input_size() <= 0)
      return false;

    this->encoder = encoder;
    this->decoder = decoder;

    return true;
  }


  template <typename T>
  bool VAE<T>::setModel(const std::vector<unsigned int> encoderArchitecture,
			const std::vector<unsigned int> decoderArchitecture)
  {
    if(encoderArchitecture.size() < 2 || decoderArchitecture.size() < 2)
      return false;
    
    if(encoderArchitecture[encoderArchitecture.size()-1] & 1) // odd value
      return false;
    
    if(encoderArchitecture[0] !=
       decoderArchitecture[decoderArchitecture.size()-1]) // dim(x) != dim(d(e(x)))
      return false;

    if(2*decoderArchitecture[0] !=
       encoderArchitecture[encoderArchitecture.size()-1]) // dim(e(x)/2) != dim(z)
      return false;

    for(unsigned int i=0;i<encoderArchitecture.size();i++){
      if(encoderArchitecture[i] <= 0)
	return false;
    }

    for(unsigned int i=0;i<decoderArchitecture.size();i++){
      if(decoderArchitecture[i] <= 0)
	return false;
    }
    
    this->encoder.setArchitecture(encoderArchitecture, nnetwork<T>::rectifier);
    this->decoder.setArchitecture(decoderArchitecture, nnetwork<T>::rectifier);

    this->initializeParameters();

    return true;
  }
  
  
  // x -> z (hidden) [returns N(zm,zv) distribution parameters]
  template <typename T>
  bool VAE<T>::encode(const math::vertex<T>& x,
		      const nnetwork<T>& encoder,
		      math::vertex<T>& zmean,
		      math::vertex<T>& zstdev) const
  {
    math::vertex<T> result;
    if(encoder.calculate(x, result) == false)
      return false;

    zmean.resize(decoder.input_size());
    zstdev.resize(decoder.input_size());

    if(decoder.input_size()*2 != result.size())
      return false;
    
    for(unsigned int i=0;i<zmean.size();i++){
      zmean[i] = result[i];
      zstdev[i] = result[i+zmean.size()];

      T value = zstdev[i];

      if(value >= T(20.0)) // restrict values so we don't go to infinity
	value = T(20.0);
      else if(value <= T(-20.0))
	value = T(-20.0);
      
      zstdev[i] = ::exp(value.c[0]);
    }

    return true;
  }


  // x -> z (hidden) [returns N(zm,zv) distribution parameters]
  template <typename T>
  bool VAE<T>::encode(const math::vertex<T>& x,
		      const nnetwork<T>& encoder,
		      const std::vector< std::vector<bool> >& dropout,
		      math::vertex<T>& zmean,
		      math::vertex<T>& zstdev) const
  {
    math::vertex<T> result;
    if(encoder.calculate(x, result, dropout) == false)
      return false;

    zmean.resize(decoder.input_size());
    zstdev.resize(decoder.input_size());

    if(decoder.input_size()*2 != result.size())
      return false;
    
    for(unsigned int i=0;i<zmean.size();i++){
      zmean[i] = result[i];
      zstdev[i] = result[i+zmean.size()];

      T value = zstdev[i];

      if(value >= T(20.0)) // restrict values so we don't go to infinity
	value = T(20.0);
      else if(value <= T(-20.0))
	value = T(-20.0);
      
      zstdev[i] = ::exp(value.c[0]);
    }

    return true;
  }


  // x -> z (hidden) [returns sample from ~ Normal(zm(x),zv(x))]
  template <typename T>
  bool VAE<T>::encodeSample(const math::vertex<T>& x,
			    const nnetwork<T>& encoder,
			    math::vertex<T>& zsample) const
  {
    math::vertex<T> result;

    if(encoder.calculate(x, result) == false)
      return false;

    zsample.resize(decoder.input_size());

    if(decoder.input_size()*2 != result.size())
      return false;
    
    for(unsigned int i=0;i<zsample.size();i++){
      auto zmean = result[i];
      T value = result[i+zsample.size()];
      
      if(value >= T(20.0)) // restrict values so we don't go to infinity
	value = T(20.0);
      else if(value <= T(-20.0))
	value = T(-20.0);
      
      auto zstdev = ::exp(value.c[0]);
      
      zsample[i] = zmean + rng.normal()*zstdev;
    }

    return true;
  }

  
  // x -> z (hidden) [returns sample from ~ Normal(zm(x),zv(x))]
  template <typename T>
  bool VAE<T>::encodeSample(const math::vertex<T>& x,
			    const nnetwork<T>& encoder,
			    const std::vector< std::vector<bool> >& dropout,
			    math::vertex<T>& zsample) const
  {
    math::vertex<T> result;

    if(encoder.calculate(x, result, dropout) == false)
      return false;

    zsample.resize(decoder.input_size());

    if(decoder.input_size()*2 != result.size())
      return false;
    
    for(unsigned int i=0;i<zsample.size();i++){
      auto zmean = result[i];
      T value = result[i+zsample.size()];
      
      if(value >= T(20.0)) // restrict values so we don't go to infinity
	value = T(20.0);
      else if(value <= T(-20.0))
	value = T(-20.0);
      
      auto zstdev = ::exp(value.c[0]);
      
      zsample[i] = zmean + rng.normal()*zstdev;
    }

    return true;
  }
  
  
  // z (hidden) -> xmean (variance = I)
  template <typename T>
  bool VAE<T>::decode(const math::vertex<T>& z,
		      const nnetwork<T>& decoder,
		      math::vertex<T>& xmean) const
  {
    if(decoder.calculate(z, xmean) == false)
      return false;
    
    return true;
  }


  // z (hidden) -> xmean (variance = I)
  template <typename T>
  bool VAE<T>::decode(const math::vertex<T>& z,
		      const nnetwork<T>& decoder,
		      const std::vector< std::vector<bool> >& dropout,
		      math::vertex<T>& xmean) const
  {
    if(decoder.calculate(z, xmean, dropout) == false)
      return false;
    
    return true;
  }

  
  template <typename T>
  unsigned int VAE<T>::getDataDimension() const
  {
    return encoder.input_size();
  }

  
  template <typename T>
  unsigned int VAE<T>::getEncodedDimension() const
  {
    return decoder.input_size();
  }
  

  template <typename T>
  void VAE<T>::getParameters(math::vertex<T>& p) const
  {
    math::vertex<T> p_encoder, p_decoder;
    if(encoder.exportdata(p_encoder) == false || decoder.exportdata(p_decoder) == false)
      return;

    p.resize(p_encoder.size() + p_decoder.size());
    p.write_subvertex(p_encoder, 0);
    p.write_subvertex(p_decoder, p_encoder.size());
  }


  template <typename T>
  bool VAE<T>::setParameters(const math::vertex<T>& p,
			     const bool dropout)
  {
    math::vertex<T> p_encoder, p_decoder;
    if(encoder.exportdata(p_encoder) == false || decoder.exportdata(p_decoder) == false)
      return false;

    if(p.size() != p_encoder.size() + p_decoder.size())
      return false;

    if(p.subvertex(p_encoder, 0, p_encoder.size()) == false)
      return false;

    if(p.subvertex(p_decoder, p_encoder.size(), p_decoder.size()) == false)
      return false;

    if(encoder.importdata(p_encoder) == false ||
       decoder.importdata(p_decoder) == false)
      return false;

    if(dropout){
      encoder.removeDropOut();
      decoder.removeDropOut();
    }
    
    return true;
  }

  // to set minibatch mode in which we use only sample of 30 data points when calculating gradient
  template <typename T>
  void VAE<T>::setUseMinibatch(bool use_minibatch)
  {
    minibatchMode = use_minibatch;
  }

  template <typename T>
  bool VAE<T>::getUseMinibatch() const
  {
    return minibatchMode;
  }

  template <typename T>
  void VAE<T>::setDropout(bool dropout)
  {
    this->dropout = dropout;
  }

  template <typename T>
  bool VAE<T>::getDropout() const
  {
    return this->dropout;
  }

  template <typename T>
  void VAE<T>::initializeParameters()
  {
    encoder.randomize();
    decoder.randomize();
  }


  // calculates raw reconstruction error ||x - decoder(encoder(x))||^2
  template <typename T>
  T VAE<T>::getError(const std::vector< math::vertex<T> >& xsamples) const
  {
    if(xsamples.size() == 0) return T(0.0f);
    
    T error = T(0.0);
    const unsigned int K = 100; // the number of random samples used to estimate E[error]
    const unsigned int MINIBATCHSIZE = 30;

    unsigned int N = K*xsamples.size();

    if(minibatchMode) N = K*MINIBATCHSIZE;

#pragma omp parallel shared(error)
    {
      math::vertex<T> zmean, zstdev, xmean, epsilon;
      T e = T(0.0);

      zmean.resize(encoder.output_size()/2);
      zstdev.resize(encoder.output_size()/2);
      xmean.resize(decoder.output_size());
      epsilon.resize(encoder.output_size()/2);
      
      //nnetwork<T> encoder(this->encoder);
      //nnetwork<T> decoder(this->decoder);

      std::vector< std::vector<bool> > encoder_dropout, decoder_dropout;

#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<N;i++){
	unsigned int index = i/K;

	if(minibatchMode)
	  index = rng.rand() % xsamples.size();

	if(dropout){
	  encoder.setDropOut(encoder_dropout);
	  decoder.setDropOut(decoder_dropout);

	  encode(xsamples[index], encoder, encoder_dropout, zmean, zstdev);
	}
	else{
	  encode(xsamples[index], encoder, zmean, zstdev);
	}

	rng.normal(epsilon);

	auto zi = zmean;
	for(unsigned int l=0;l<zmean.size();l++){
	  zi[l] += zstdev[l]*epsilon[l];
	}

	if(dropout){
	  decode(zi, decoder, decoder_dropout, xmean);
	}
	else{
	  decode(zi, decoder, xmean);
	}
	
	auto delta = xsamples[index] - xmean;
	e += delta.norm();
      }

#pragma omp critical (rewioprwejfwaawq)
      {
	error += e;
      }
    }

    if(xsamples.size() > 0){
      error /= T(N);
      error /= T(xsamples[0].size()); // scales to per dimension E[|error|]
    }

    return error;
  }
    

  // log(P) should be maximized..
  template <typename T>
  T VAE<T>::getLoglikelihood(const std::vector< math::vertex<T> >& xsamples) const
  {
    T logp = T(0.0), DL = T(0.0);
    T error = T(0.0);
    
    // data variance s^2 term: c = Dz/Dx (was: c = Dx)
    // const float c = 0.05f*0.05f;
    // const float c = (0.05f*0.05f)*((float)decoder.input_size())/((float)decoder.output_size());
    
    const float c = 0.05f*0.05f*(3.0f/139.0f)*((float)decoder.output_size())/((float)decoder.input_size());
    // GIVE BIG IMPACT ON RECONSTRUCTION ERROR: High dimensional vector data dominates.
    // const float c = 0.05f*0.05f;
    
    // const float c = ((float)decoder.output_size())/((float)decoder.input_size());
    
    // constant term C:
    logp += T(0.5)*decoder.input_size() - T(0.5)*decoder.output_size()*(::log((float)(2.0*M_PI*c)));
    
    const unsigned int K = 100; // the number of random samples used to estimate E[error]
    
    const unsigned int N = K*xsamples.size();

#pragma omp parallel shared(error, logp)
    {
      math::vertex<T> zmean, zstdev, xmean, epsilon;
      T e = T(0.0);
      T l = T(0.0);      
      
      zmean.resize(encoder.output_size()/2);
      zstdev.resize(encoder.output_size()/2);
      xmean.resize(decoder.output_size());
      epsilon.resize(encoder.output_size()/2);

      //nnetwork<T> encoder(this->encoder);
      //nnetwork<T> decoder(this->decoder);

      std::vector< std::vector<bool> > encoder_dropout, decoder_dropout;
      
#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<N;i++){
	const unsigned int index = i/K;
	const unsigned int mod   = i % K;

	if(dropout){
	  encoder.setDropOut(encoder_dropout);
	  decoder.setDropOut(decoder_dropout);

	  encode(xsamples[index], encoder, encoder_dropout, zmean, zstdev);
	}
	else{
	  encode(xsamples[index], encoder, zmean, zstdev);
	}
	
	if(mod == 0){ // calculates D_KL term
	  T li = T(0.0);
	  
	  li -= (T(0.5)*((zmean*zmean)[0]) + T(0.5)*((zstdev*zstdev)[0]));
	  
	  for(unsigned int n=0;n<zstdev.size();n++)
	    li += log(zstdev[n]);
	  
	  l += li;
	}
	
	
	rng.normal(epsilon);
	
	auto zi = zmean;
	
	for(unsigned int n=0;n<zmean.size();n++){
	  zi[n] += zstdev[n]*epsilon[n];
	}

	if(dropout){
	  decode(zi, decoder, decoder_dropout, xmean);
	}
	else{
	  decode(zi, decoder, xmean);
	}
	
	auto delta = xsamples[index] - xmean;
	e += (delta*delta)[0];
      }
      
#pragma omp critical (fewqqwipofdsaa)
      {
	error += e;
	DL    += l;
      }
    }
    
    if(xsamples.size() > 0){
      error /= T(2*N*c);
      DL /= xsamples.size();
      
      logp += DL - error;
    }
    
    return logp;
  }

  
  
  // x = samples, learn x = decode(encode(x))
  // optimizes using gradient descent,
  // divides data to teaching and test
  // datasets (50%/50%)
  //
  // model_stdev_error(last N iters)/model_mean_error(last N iters) < convergence_ratio 
  //
  template <typename T>
  bool VAE<T>::learnParameters(const std::vector< math::vertex<T> >& xsamples,
			       T convergence_ratio, bool verbose,
			       LoggingInterface* messages,
			       VisualizationInterface* gui, 
			       bool* running)
  {
    // implements gradient descent
    if(convergence_ratio <= T(0.0) || convergence_ratio >= T(1.0))
      return false;
    
    if(running){
      if(*running == false){
	return false;
      }
    }
    
    //std::list<T> errors;
    T error = getError(xsamples);
    
    std::list<T> mlogprobs; // negative log propabilities (-log(P)) [values are [0,inf[ where smaller is better]
    T mloglikelihood = -getLoglikelihood(xsamples);
    
    T lrate = T(0.0001);
    
    unsigned int counter = 0;
    const unsigned int MAXITER = 1000;

    unsigned int noimprove_counter = 0;
    const unsigned int MAXNOIMPROVE = 50;

    // best solution found so far
    math::vertex<T> bestparams;
    T bestmloglikelihood = mloglikelihood;
    T besterror = error;
    getParameters(bestparams);
    
    const int BUFLEN = 1024;
    char buf[BUFLEN];
    
    whiteice::linear_ETA<double> eta;
    eta.start(0.0, MAXITER);
    eta.update(0.0);

    if(gui){
      gui->show();
      gui->clear();
    }
    
    while(counter < MAXITER && noimprove_counter < MAXNOIMPROVE){
      if(running){
	if(*running == false){
	  printf("VAE::learn() aborting computation\n");
	  if(messages) messages->printMessage("learn(): aborting computation\n");
	  break;
	}
      }
      
      // gradient search of better solution (uses dropout to calculate gradient)
      math::vertex<T> grad;
      if(calculateGradient(xsamples, grad, messages, running) == false){
	if(verbose){
	  std::cout << "calculateGradient() returns false!" << std::endl;
	  if(messages) messages->printMessage("calculateGradient() returns false!\n");
	  std::cout << std::flush;;
	}
	return false;
      }
      
      
      // FIXME write proper line search in grad direction
      math::vertex<T> params, p;
      getParameters(params);
      
      
      bool found = false;
      
      T eprev = error;
      T mlprev = mloglikelihood;
      
      //if(lrate <= T(10e-30)){
      //lrate = T(0.0001);
      //}
      
      // lrate *= T(4.0);
      lrate = T(0.50f);
      
      // search in gradient direction as long as error is hasn't become smaller
      while(mlprev <= mloglikelihood && lrate > T(10e-30) && found == false){
	if(running){
	  if(*running == false){
	    printf("VAE::learn() aborting computation\n");
	    if(messages) messages->printMessage("learn()/gradient-search: aborting computation\n");
	    break;
	  }
	}
		
	p = params;
	p += lrate*grad;
	setParameters(p, false);
	
	//error = getError(xsamples);
	mloglikelihood = -getLoglikelihood(xsamples);
	
	if(mloglikelihood < mlprev){
	  // TODO: remove this later after you have found good initial learning rate
	  std::cout << "GRAD DESCENT FOUND: lrate = " << lrate << std::endl; std::cout << std::flush;
	  lrate *= T(2.0);
	  found = true;
	}
	else if(mloglikelihood >= mlprev){
	  // std::cout << "GRAD DESCENT NOT FOUND: lrate = " << lrate << std::endl; std::cout << std::flush;
	  lrate *= T(0.50);
	}

	error = getError(xsamples);
	
	// leaky error reduction, we sometimes allow jump to worse
	// position in gradient direction
	{
	  const unsigned int rnd = (rng.rand() % 20);
	  
	  printf("RNG::RAND() == %d\n", (int)rnd);
	  fflush(stdout);
	  
	  //if(rnd == 0 && mloglikelihood < 1e6)
	  if(rnd == 0 && error <= 1.0f){
	    std::cout << "! JUMP TO WORSE: GO TO GRADIENT DIRECTION BUT ERROR INCREASES."
		      << std::endl;
	    found = true;
	  }
	}
	
      }

      
      // updates parameters based on the found solution
      if(found){
	params = p;
	setParameters(p, false);	
	
	if(mloglikelihood > bestmloglikelihood){ // no improvement but follow gradient
	  noimprove_counter++;
	  
	  std::cout << "! No improvement in loglikelihood ("
		    << noimprove_counter << "/" << MAXNOIMPROVE << ")"
		    << std::endl;
	}
	else{ // true improvement
	  noimprove_counter = 0;
	  bestmloglikelihood = mloglikelihood;
	  besterror = error;
	  bestparams = p;
	}
      }
      else{
	setParameters(params, false);
	//error = getError(xsamples);
	mloglikelihood = -getLoglikelihood(xsamples);
      }
      
      counter++;
      eta.update(counter);
      
      if(verbose){
	std::cout << "ERROR " << counter << "/" << MAXITER << ": " << error << " loglikelihood: " << -mloglikelihood
		  << " (ETA " << eta.estimate()/3600.0 << " hours)" << std::endl;
	
	float errf = 10e10;
	whiteice::math::convert(errf, besterror);
	
	float mloglikelihoodf = -10e10;
	whiteice::math::convert(mloglikelihoodf, bestmloglikelihood);
	
	//float lratef = 0.0f;
	//whiteice::math::convert(lratef, lrate);
	
	snprintf(buf, BUFLEN, "VAE: learn() iter %d/%d best error: %.2f (loglikelihood: %.2f) [ETA %.2f hours]\n",
		 counter, MAXITER, errf, -mloglikelihoodf, eta.estimate()/3600.0);
	if(messages) messages->printMessage(buf);
	
	std::cout << std::flush;
      }

      // prints Z-space statistics
      math::vertex<T> z;
      {
	math::vertex<T> zvar, model_zstdev; // removes mean away from the encoded vectors
	math::vertex<T> zmean, zstdev;

	z.resize(getEncodedDimension());
	z.zero();

	zvar.resize(getEncodedDimension());
	zvar.zero();

	model_zstdev.resize(getEncodedDimension());
	model_zstdev.zero();

	for(const auto& x : xsamples){
	  this->encode(x, this->encoder, zmean, zstdev);

	  z += zmean;
	  model_zstdev += zstdev;

	  for(unsigned int i=0;i<zvar.size();i++)
	    zvar[i] += zmean[i]*zmean[i];
	}

	z /= T(xsamples.size());
	zvar /= T(xsamples.size());
	model_zstdev /= T(xsamples.size());

	for(unsigned int i=0;i<zvar.size();i++){
	  zvar[i] -= z[i]*z[i];
	  zvar[i] = whiteice::math::sqrt(whiteice::math::abs(zvar[i]));
	}

	std::cout << "Z-SPACE MEAN  = " << z << std::endl;
	std::cout << "Z-SPACE STDEV = " << zvar << std::endl;
	std::cout << "Z-MODEL STDEV = " << model_zstdev << std::endl;
	std::cout << std::flush;
      }
      
      // visualizes current location of xsamples
      if(gui != 0 && getEncodedDimension() >= 2){
	
	math::vertex<T> zmean, zstdev;

	const unsigned int XMAX = gui->getScreenX();
	const unsigned int YMAX = gui->getScreenY();
	gui->clear();

	for(const auto& x : xsamples){
	  this->encode(x, encoder, zmean, zstdev);

	  zmean -= z; // removes mean from the encoded vectors but keeps variance

	  float x0 = 0.0f;
	  whiteice::math::convert(x0, zmean[0]);
	  x0 /= 2.0f;
	  if(x0 < -1.0f) x0 = -1.0f;
	  if(x0 > +1.0f) x0 = +1.0f;

	  x0 = (x0 + 1.0f)/2.0f;
	  x0 = x0 * (XMAX - 1);

	  float y0 = 0.0f;
	  whiteice::math::convert(y0, zmean[1]);
	  y0 /= 2.0f;
	  if(y0 < -1.0f) y0 = -1.0f;
	  if(y0 > +1.0f) y0 = +1.0f;

	  y0 = (y0 + 1.0f)/2.0f;
	  y0 = y0 * (YMAX - 1);

	  const unsigned int xx = (unsigned int)whiteice::math::round(x0);
	  const unsigned int yy = (unsigned int)whiteice::math::round(y0);

	  gui->plot(xx, yy);
	}

	gui->updateScreen();
      }
      
      // update convergence check (errors)
      mlogprobs.push_back(mloglikelihood);
      
      while(mlogprobs.size() > 20)
	mlogprobs.pop_front();
      
      // check for convergence
      if(mlogprobs.size() >= 20){
	T m = T(0.0), v = T(0.0);
	
	for(const auto& e : mlogprobs){
	  m += e;
	}
	
	m /= mlogprobs.size();
	
	for(const auto& e : mlogprobs){
	  v += (e - m)*(e - m);
	}
	
	v /= (mlogprobs.size() - 1);
	
	v = sqrt(abs(v));
	
	// FIXME: rewrite me
	if(verbose){
	  T ratio = T(100.0)*v/m;
	  
	  std::cout << "LOGLIKELIHOOD CONVERGENCE " << ratio << "%" << std::endl;
	  
	  float ratiof = 10e10;
	  whiteice::math::convert(ratiof, ratio);
	  
	  float convf  = 10e10;
	  T convp = T(100.0)*convergence_ratio;
	  whiteice::math::convert(convf, convp);
	  
	  snprintf(buf, BUFLEN, "LOGLIKELIHOOD CONVERGENCE: %.2f%% (> %.2f%%)\n", ratiof, convf);
	  if(messages) messages->printMessage(buf);
	  
	  std::cout << std::flush;;
	}

#if 0
	// DISABLED: WE STOP WHEN NUMBER OF NOIMPROVEMENT ITERATIONS IS LARGE ENOUGH
	if(v/m <= convergence_ratio)
	  break; // stdev is less than c% of mean (5%)
#endif
      }
      
    } // end of while(counter < MAXITER && noimprovement_counter < MAXNOIMPROVEROUNDS) loop

    {
      float besterrorf = 0.0f;
      whiteice::math::convert(besterrorf, besterror);

      float bestmloglikelihoodf = 0.0f;
      whiteice::math::convert(bestmloglikelihoodf, bestmloglikelihood);
      
      printf("Computation stopped. Best error: %f. Best loglikeligood %f.\n",
	     besterrorf, bestmloglikelihoodf);
    }
      
    setParameters(bestparams, dropout); // remove dropout only here
    
    return true;
  }

  
  // calculates gradient of parameter p using all samples
  template <typename T>
  bool VAE<T>::calculateGradient(const std::vector< math::vertex<T> >& xsamples,
				 math::vertex<T>& pgradient,
				 LoggingInterface* messages,
				 bool* running) const
  {
    pgradient.resize(encoder.gradient_size() + decoder.gradient_size());
    pgradient.zero();
    
    bool failure = false;
    const bool verbose = false;
    unsigned int MINIBATCHSIZE = 0; // number of samples used to estimate gradient

    if(xsamples.size() <= 0) return false;
    if(xsamples[0].size() != encoder.input_size()) return false;

    // data variance term c, this allows errors to be made in reconstruction (0.05: was c = Dx (x.size()), c = Dz/Dx
    // const float c = 0.05f*0.05f;
    // const float c = (0.05f*0.05f)*((float)decoder.input_size())/((float)decoder.output_size());
    const float c = 0.05f*0.05f*(3.0f/139.0f)*((float)decoder.output_size())/((float)decoder.input_size());
    // GIVE BIG IMPACT ON RECONSTRUCTION ERROR: high dimensional data dominates
    // const float c = 0.05f*0.05f;
    
    // const float c = ((float)decoder.output_size())/((float)decoder.input_size());

    const int BUFLEN = 1024;
    char buffer[BUFLEN];

    if(minibatchMode)
      MINIBATCHSIZE = 30;
    else
      MINIBATCHSIZE = xsamples.size(); // use all samples


#pragma omp parallel shared(pgradient)
    {
      math::vertex<T> pgrad;
      pgrad.resize(pgradient.size());
      pgrad.zero();

      //nnetwork<T> encoder(this->encoder);
      //nnetwork<T> decoder(this->decoder);
      std::vector< std::vector<bool> > encoder_dropout, decoder_dropout;

      // encoder values
      math::vertex<T> zmean, zstdev;
      math::vertex<T> output;
      math::vertex<T> ones;

      zmean.resize(encoder.output_size()/2);
      zstdev.resize(encoder.output_size()/2);
      ones.resize(encoder.output_size()/2);
      
      math::matrix<T> J; // Jacobian matrix
      math::matrix<T> Jmean, Jstdev;

      math::vertex<T> decoder_gradient, encoder_gradient;

      math::vertex<T> epsilon;
      
      math::vertex<T> zi;
      math::vertex<T> xmean;
      math::matrix<T> grad_meanx;
      math::matrix<T> J_meanx_value;
      math::vertex<T> gzx;
      math::matrix<T> Jstdev_epsilon;
      
#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<MINIBATCHSIZE;i++)
      {
	if(failure) continue; // do nothing after first failure
	if(running){
	  if(*running == false){
	    failure = true;
	    continue; // do nothing if computation has been stopped
	  }
	}

  	if(verbose){
	  snprintf(buffer, BUFLEN, "Calculating gradient sample %d/%d..\n", i, MINIBATCHSIZE);
  	  if(messages) messages->printMessage(buffer);
  	  printf("%s", buffer);
  	  fflush(stdout);
  	}

	if(dropout){
	  encoder.setDropOut(encoder_dropout);
	  decoder.setDropOut(decoder_dropout);
	}
	
	
	unsigned int index = 0;
	
	if(minibatchMode){
	  index = rng.rand() % ((unsigned int)xsamples.size());
	}
	else{
	  index = i;
	}
	
	
	const auto& x = xsamples[index];
	// 1. first calculate gradient of "-D_KL" term
	
	
	if(encoder.calculate(x, output, encoder_dropout) == false){
	  failure = true;
	  continue;
	}
	
	if(output.subvertex(zmean, 0, zmean.size()) == false){
	  failure = true;
	  continue;
	}
	
	if(output.subvertex(zstdev, zmean.size(), zstdev.size()) == false){
	  failure = true;
	  continue;
	}

	for(unsigned int j=0;j<zstdev.size();j++){
	  T value = zstdev[j];
	  
	  if(value >= T(20.0)) // restrict values so we don't go to infinity
	    value = T(20.0);
	  else if(value <= T(-20.0))
	    value = T(-20.0);
	  
	  zstdev[j] = ::exp(value.c[0]); // convert s = Log(stdev) to proper standard deviation
	  ones[j] = T(1.0);
	}
	
	// gradient of encoder
	
	if(encoder.jacobian(x, J, encoder_dropout) == false){
	  failure = true;
	  continue;
	}
	
	assert(J.ysize() == zmean.size()+zstdev.size());
	assert(J.xsize() == encoder.gradient_size());
	
	if(J.submatrix(Jmean,
		       0, 0,
		       encoder.gradient_size(), zmean.size()) == false)
	{
	  failure = true;
	  continue;
	}
	
	
	if(J.submatrix(Jstdev,
		       0, zmean.size(),
		       encoder.gradient_size(), zstdev.size()) == false)
	{
	  failure = true;
	  continue;
	}


	auto zvar = zstdev;
	for(unsigned int j=0;j<zvar.size();j++)
	  zvar[j] = zvar[j]*zvar[j];
	
	auto g1 = T(-1.0)*zmean*Jmean + (ones + T(-1.0)*zvar)*Jstdev;
	
	
	// 2. second calculate gradient	
	decoder_gradient.resize(decoder.gradient_size());
	encoder_gradient.resize(encoder.gradient_size());
	decoder_gradient.zero();
	encoder_gradient.zero();
	
	const unsigned int N = 15;
	
	epsilon.resize(zmean.size());
	
	if(verbose){
	  printf("%d/%d: CALCULATING 2ND GRADIENT\n", index, (int)xsamples.size());
	  fflush(stdout);
	}

	for(unsigned int j=0;j<N;j++){
	  // epsilon is ~ Normal(0,I)
	  rng.normal(epsilon);

	  zi = zmean;
	  for(unsigned int k=0;k<zmean.size();k++){
	    zi[k] += zstdev[k]*epsilon[k];
	  }

	  xmean = x;
	  if(decoder.calculate(zi, xmean, decoder_dropout) == false){
	    failure = true;
	    continue;
	  }
	  
	  if(decoder.jacobian(zi, grad_meanx, decoder_dropout) == false){
	    failure = true;
	    continue;
	  }
	  
	  decoder_gradient += (x - xmean)*grad_meanx;
	  
	  if(decoder.gradient_value(zi, J_meanx_value, decoder_dropout) == false){
	    failure = true;
	    continue;
	  }
	  
	  gzx = (x - xmean)*J_meanx_value; // first part of gradient

	  Jstdev_epsilon = Jstdev;
	  for(unsigned int y=0;y<Jstdev.ysize();y++)
	    for(unsigned int x=0;x<Jstdev.xsize();x++)
	      Jstdev_epsilon(y,x) = epsilon[y]*zstdev[y]*Jstdev(y,x);
	  
	  // encoder_gradient += gzx * (Jmean + epsilon*Jstdev); // second part of gradient
	  encoder_gradient += gzx * (Jmean + Jstdev_epsilon); // second part of gradient
	}

	encoder_gradient *= T(1.0/N);
	decoder_gradient *= T(1.0/N);

	if(verbose){
	  printf("%d/%d: CALCULATING 2ND GRADIENT: SUM CALCULATED\n", i, (int)xsamples.size());
	  fflush(stdout);
	}
	
	// calculates gradient
	auto pg = pgradient;
	pg.zero();
	
	if(pg.write_subvertex(g1, 0) == false){
	  failure = true;
	  continue;
	}
	
	pgrad += pg;

	// HEURISTICS: scales "dimensionality" of p(x) to be same as in p(z) which are both
	// guessed to be Normal(0,I) distributed.
	// DISABLED: (NOTE if you scale gradient by alpha, then normal noises st.dev/sigma should be sqrt(D/3))
	encoder_gradient *= T(1.0f/c);
	decoder_gradient *= T(1.0f/c);
	
	pg.zero();
	if(pg.write_subvertex(encoder_gradient, 0) == false){
	  failure = true;
	  continue;
	}
	
	if(pg.write_subvertex(decoder_gradient, encoder_gradient.size()) == false){
	  failure = true;
	  continue;
	}
	
	pgrad += pg;
      }
      

#pragma omp critical (rewkowrejowgra)
      if(!failure){
	pgradient += pgrad;
      }

    }

    if(failure) return false;

    if(xsamples.size() > 0)
      pgradient /= T(MINIBATCHSIZE);

    // ENABLED gradient norm calculation
    T nrm = pgradient.norm();

    if(nrm > T(0.0)){
      // is this really needed? (adaptive gradient descent don't work well without this)
      pgradient /= nrm; 
    }
    
    return true;
  }
  

  template <typename T>
  bool VAE<T>::load(const std::string& filename) 
  {
    if(filename.size() <= 0) return false;
    
    const std::string encoderfile = filename + ".vae-encoder";
    const std::string decoderfile = filename + ".vae-decoder";
    
    auto en = encoder;
    auto de = decoder;

    if(en.load(encoderfile) == false || de.load(decoderfile) == false)
      return false;

    if(en.input_size() != de.output_size() || en.output_size() != 2*de.input_size())
      return false;

    encoder = en;
    decoder = de;

    return true;
  }


  template <typename T>
  bool VAE<T>::save(const std::string& filename) const 
  {
    if(filename.size() <= 0) return false;
    
    const std::string encoderfile = filename + ".vae-encoder";
    const std::string decoderfile = filename + ".vae-decoder";
    
    if(encoder.save(encoderfile) == false || decoder.save(decoderfile) == false)
      return false;

    return true;
  }
  
  
  
  // template class VAE< float >;
  // template class VAE< double >;  
  template class VAE< math::blas_real<float> >;
  template class VAE< math::blas_real<double> >;
}
