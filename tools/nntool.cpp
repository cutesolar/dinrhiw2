/*
 * nntool (narya) - 
 * a feedforward neural network
 * optimizer command line tool.
 * 
 * (C) Copyright Tomas Ukkonen 2004, 2005, 2014-2022
 *
 *************************************************************
 * 
 * neural networks and other machine learning 
 * models try to build generic models which can
 * describe and predict essential features of data.
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>


//#include <dinrhiw/dinrhiw.h>
#include <dinrhiw.h>

#include <exception>

#include <vector>
#include <string>

#ifdef WINNT
#include <windows.h>
#endif

#include "argparser.tab.h"
#include "cpuid_threads.h"

#ifndef _WIN32
#undef __STRICT_ANSI__
#include <fenv.h>

// enables floating point exceptions, these are good for debugging 
// to notice BAD floating point values that come from software bugs..
#include <fenv.h>

extern "C" {

  // traps floating point exceptions..
#define _GNU_SOURCE 1
#ifdef __linux__
#include <fenv.h>
  static void __attribute__ ((constructor))
  trapfpe(){
    feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    //feenableexcept(FE_INVALID);
  }
#endif
  
}
#endif

void print_usage(bool all);

void sleepms(unsigned int ms);


// is set true if receives CRTL-C or some other signal
volatile bool stopsignal = false; 
void install_signal_handler();


using namespace whiteice;



int main(int argc, char** argv)
{
  try{
    std::string datafn, nnfn;
    std::string lmethod;
    std::vector<std::string> lmods;
    std::vector<unsigned int> arch;
    unsigned int cmdmode;
    bool no_init, verbose;
    bool load = false, help = false;
    
    bool overfit = false;
    bool adaptive = false;
    bool negfeedback = false;
    unsigned int deep = 0;
    bool residual = true;
    bool dropout = false;
    bool crossvalidation = false;
    bool batchnorm = false;
    
    // minimum norm error ||y-f(x)|| gradient instead of MSE ||y-f(x)||^2
    bool MNE = true;

    // TESTING PURPOSES: if linearnet is true [default false] construct linear neural network
    const bool linearnet = false;

    bool subnet = false;

    // should we use recurent neural network or not..
    unsigned int SIMULATION_DEPTH = 1;
    
    unsigned int samples = 0; // number of samples or iterations in learning process
    unsigned int secs = 0;    // how many seconds the learning process should take

    // number of threads used in optimization
    unsigned int threads = 0;

    // number of datapoints to be used in learning (taken randomly from the dataset)
    unsigned int dataSize = 0;

    
#ifdef _GLIBCXX_DEBUG    
    // enables FPU exceptions
    feenableexcept(FE_INVALID |
		   FE_DIVBYZERO);
#endif

    // special value to enable writing to console
    whiteice::logging.setOutputFile("nntool.log");
    
    parse_commandline(argc,
		      argv,
		      datafn,
		      nnfn, 
		      lmethod,
		      lmods,
		      arch, 
		      cmdmode,
		      secs,
		      samples,
		      threads,
		      dataSize,
		      SIMULATION_DEPTH,
		      no_init,
		      load,
		      overfit,
		      adaptive,
		      negfeedback,
		      deep,
		      residual,
		      dropout,
		      crossvalidation,
		      batchnorm,
		      help,
		      verbose);
    srand(time(0));

    if(secs <= 0 && samples <= 0) // no time limit
      samples = 2000; // we take 2000 samples/tries as the default

    if(batchnorm){
      printf("FIXME: batchnorm is not implemented in nntool/dinrhiw fully yet.\n");
      help = true;
    }

    if(help){ // prints command line usage information
      print_usage(true);
      return 0;
    }

    install_signal_handler();

    if(threads <= 0)
      threads = // for multithread-enabled code
	        // only uses half of the resources as the default
	(unsigned int)numberOfCPUThreads()/2;
    
    if(threads <= 0)
      threads = 1;
    
    
    if(cmdmode != 0){
      printf("Daemon and 'send command' modes aren't supported yet.\n");
      return 0;
    }

    // we load network data from disc if we want information
    if(lmethod == "info") load = true;
    
    // tries to open data and nnfile
    
    // loads data
    dataset< whiteice::math::blas_real<float> > data;
    dataset< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > > sdata;
    bool stdinout_io = false;
    
    if(datafn.size() > 0){ // file input
      if(!data.load(datafn)){
	fprintf(stderr, "error: couldn't open datafile: %s.\n", datafn.c_str());
	exit(-1);
      }
      
      if(data.getNumberOfClusters() < 1){
	fprintf(stderr, "error: datafile is empty.\n");
	exit(-1);
      }
      
      if(lmethod != "use" && lmethod != "minimize"){
	if(data.getNumberOfClusters() < 2){
	  fprintf(stderr, "error: datafile doesn't contain example pairs.\n");
	  exit(-1);
	}
      }
      
      
      if(arch[0] <= 0){
	if(SIMULATION_DEPTH > 1){
	  if(data.getNumberOfClusters() >= 2){
	    const int RDIM2 = ((int)arch[arch.size()-1]) - ((int)data.dimension(1));
	    if(RDIM2 > 0){
	      arch[0] = data.dimension(0)+RDIM2;
	    }
	    else{
	      fprintf(stderr, "error: cannot compute recurrent network input layer size.\n");
	      exit(-1);
	    }
	  }
	  else{
	    fprintf(stderr, "error: cannot compute recurrent network input layer size.\n");
	    exit(-1);
	  }
	}
	else{
	  arch[0] = data.dimension(0);
	}
      }
      else{
	if(SIMULATION_DEPTH > 1){
	  if(data.getNumberOfClusters() >= 2){
	    const int RDIM1 = ((int)arch[0]) - ((int)data.dimension(0));

	    if(RDIM1 <= 0){
	      fprintf(stderr, "error: bad recurrent network input layer size.\n");
	      exit(-1);
	    }
	  }
	}
	else{
	  if(arch[0] != data.dimension(0)){
	    fprintf(stderr, "error: bad network input layer size, input data dimension pair.\n");
	    exit(-1);
	  }
	}
      }
      
      if(arch[arch.size()-1] <= 0){
	if(data.getNumberOfClusters() >= 2){
	  const int RDIM1 = ((int)arch[0]) - ((int)data.dimension(0));

	  if(RDIM1 >= 0){
	    arch[arch.size()-1] = data.dimension(1) + RDIM1;
	  }
	  else{
	    fprintf(stderr, "error: bad recurrent network input/output layer size.\n");
	    exit(-1);
	  }
	}
	else{
	  fprintf(stderr, "error: neural network do not have proper output dimension.\n");
	  exit(-1);
	}
      }
      else{
	if(data.getNumberOfClusters() >= 2){
	  const int RDIM1 = ((int)arch[0]) - ((int)data.dimension(0));

	  if(RDIM1 <= 0 && SIMULATION_DEPTH > 1){
	    fprintf(stderr, "error: bad recurrent network input layer size.\n");
	    exit(-1);
	  }
	  
	  if(arch[arch.size()-1] != data.dimension(1)+RDIM1){
	    fprintf(stderr, "error: bad network output layer size, output data dimension pair.\n");
	    exit(-1);
	  }
	}
      }
      
      if(data.size(0) == 0 || (data.size(1) == 0 && (lmethod != "use" && lmethod != "minimize"))){
	fprintf(stderr, "error: empty datasets cannot be used for training.\n");
	exit(-1);
      }
      else if((lmethod != "use" && lmethod != "minimize") && data.size(0) != data.size(1)){
	if(data.size(0) < data.size(1)){
	  printf("warning: output dataset is larger than input dataset.\n");
	  printf("some data is discarded. pairing may be incorrect.\n");
	  
	  data.resize(1, data.size(0));
	}
	else if(data.size(0) > data.size(1)){
	  printf("warning. input dataset is larger than output dataset.\n");
	  printf("some data is discarded. pairing may be incorrect.\n");
	  
	  data.resize(0, data.size(1));
	}
      }
    }
    else{
      stdinout_io = true;
      fprintf(stderr, "stdin/stdout I/O isn't supported yet.\n");    
      exit(-1);
    }

    if(crossvalidation && lmethod != "grad"){
      fprintf(stderr, "error: crossvalidation currently only works with 'grad' method.\n");
      exit(-1);
    }
    
    if(SIMULATION_DEPTH > 1){
      if(verbose){
	printf("Simple recurrent neural network (leaky rectifier).\n");
      }
    }


    // copies data to sdata too
    {
      std::vector<dataset< math::blas_real<float> >::data_normalization> preprocessings;
      
      math::vertex< math::superresolution<math::blas_real<float>,
					  math::modular<unsigned int> > > w;

      // data to sdata
      sdata.clear();
      
      for(unsigned int c=0;c<data.getNumberOfClusters();c++){
	
	sdata.createCluster(data.getName(c), data.dimension(c));
	
	for(unsigned int i=0;i<data.size(c);i++){
	  
	  auto v = data.access(c, i);
	  // data.invpreprocess(c, v);
	  
	  w.resize(v.size());
	  
	  for(unsigned int k=0;k<v.size();k++)
	    whiteice::math::convert(w[k], v[k]);
	  
	  sdata.add(c, w, true);
	}

	// dont't use preprocessings in sdata for now [buggy]
#if 0
	// add preprocessings to sdata that were in the original dataset
	// [don't work very well currently]
	
	data.getPreprocessings(c, preprocessings);
	
	for(unsigned int i=0;i<preprocessings.size();i++){
	  sdata.preprocess(c, (dataset< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >::data_normalization)(preprocessings[i]));
	}
#endif
      }
      
    }
    
    
    nnetwork< whiteice::math::blas_real<float> >* nn = new nnetwork< whiteice::math::blas_real<float> >(arch);

    nnetwork< math::superresolution< math::blas_real<float>,
				     math::modular<unsigned int> > >* snn =
      new nnetwork< math::superresolution< math::blas_real<float>,
					   math::modular<unsigned int> > >(arch);
    
    bayesian_nnetwork< whiteice::math::blas_real<float> >* bnn = new bayesian_nnetwork< whiteice::math::blas_real<float> >();

    bayesian_nnetwork< math::superresolution< math::blas_real<float>,
					      math::modular<unsigned int> > >* sbnn =
      new bayesian_nnetwork< math::superresolution< math::blas_real<float>,
						    math::modular<unsigned int> > >();
   
    nn->setResidual(residual);
    snn->setResidual(residual);

    if(verbose){
      if(dropout && residual) printf("Using residual neural network with dropout heuristics.\n");
      else if(dropout) printf("Using normal neural network with dropout heuristics.\n");
      else if(residual) printf("Using residual neural network.\n");
      else printf("Using normal neural network.\n");
    }

    
    whiteice::nnetwork< whiteice::math::blas_real<float> >::nonLinearity nl =
      whiteice::nnetwork< whiteice::math::blas_real<float> >::rectifier;

    if(linearnet) // only make sense when testing optimization
      nl = whiteice::nnetwork< whiteice::math::blas_real<float> >::pureLinear;

    {
      nn->setNonlinearity(nl);
      nn->setNonlinearity(nn->getLayers()-1,
			  whiteice::nnetwork< whiteice::math::blas_real<float> >::pureLinear);
    }

    whiteice::nnetwork< math::superresolution<math::blas_real<float>, math::modular<unsigned int> > >::nonLinearity snl =
      whiteice::nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >::rectifier;

    if(linearnet) // only make sense when testing optimization
      snl = whiteice::nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >::pureLinear;

    {
      snn->setNonlinearity(snl);
      snn->setNonlinearity(nn->getLayers()-1,
			   whiteice::nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >::pureLinear);
    }

    
    if(verbose && !stdinout_io){
      math::vertex< whiteice::math::blas_real<float> > w;
      nn->exportdata(w);
      
      if(lmethod == "use"){
	printf("Processing %d data points (%d parameters in neural network).\n", data.size(0), w.size());
      }
      else{
	if(SIMULATION_DEPTH <= 1){
	  printf("%d data points for %d -> %d mapping (%d parameters in neural network).\n",
		 data.size(0), data.dimension(0), data.dimension(1),
		 w.size());
	}
	else{
	  printf("%d data points for %d -> %d mapping (%d parameters in neural network).\n",
		 data.size(0),
		 data.dimension(0),
		 data.dimension(1),
		 w.size());
	}
      }
    }
    
    
    fflush(stdout);

    
    if(lmethod != "use" && dataSize > 0 && dataSize < data.size(0)){
      printf("Resampling dataset down to %d datapoints.\n", dataSize);

      data.downsampleAll(dataSize);
      sdata.downsampleAll(dataSize);
    }

    if((lmethod != "use" && lmethod != "minimize") && deep > 0){
      printf("Deep pretraining (stacked RBMs) of neural network weights (slow).\n");

      bool binary = true;

      if(deep == 1) binary = true; // full RBM network
      else if(deep == 2) binary = false; // gaussian-bernoulli rbm input layer
      bool running = true;
      int v = 0;
      if(verbose) v = 1;

      nn->setNonlinearity(whiteice::nnetwork< whiteice::math::blas_real<float> >::sigmoid);
      nn->setNonlinearity(nn->getLayers()-1,
			  whiteice::nnetwork< whiteice::math::blas_real<float> >::pureLinear);

      if(snn){
	snn->setNonlinearity(whiteice::nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >::sigmoid);
	snn->setNonlinearity(nn->getLayers()-1,
			     whiteice::nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >::pureLinear);
      }
      
      if(nn){
	if(deep_pretrain_nnetwork(nn, data, binary, v, &running) == false){
	  printf("ERROR: deep pretraining of nnetwork failed.\n");
	  return -1;
	}
	
	if(linearnet) // only make sense when testing optimization
	  nn->setNonlinearity(whiteice::nnetwork< whiteice::math::blas_real<float> >::pureLinear);
	else
	  nn->setNonlinearity(whiteice::nnetwork< whiteice::math::blas_real<float> >::rectifier);
	nn->setNonlinearity(nn->getLayers()-1,
			    whiteice::nnetwork< whiteice::math::blas_real<float> >::pureLinear);
      }
      else if(snn){
	printf("FIXME: Superresolutional numbers don't support depp pretraining of neural network weights..\n");

	if(linearnet) // only make sense when testing optimization
	  snn->setNonlinearity(nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >::pureLinear);
	else
	  snn->setNonlinearity(nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >::rectifier);
	snn->setNonlinearity(snn->getLayers()-1,
			     nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >::pureLinear);
      }
      
    }
    /*
     * default: initializes nnetwork weight values using 
     * (simple) deep ica if possible
     */
    else if((lmethod != "use" && lmethod != "minimize") && no_init == false && load == false)
    {
      if(nn) nn->randomize();
      if(snn) snn->randomize();
      
#if 0
      if(verbose)
	std::cout << "Heuristics: NN weights normalization initialization."
		  << std::endl;

      if(normalize_weights_to_unity(*nn, true) == false){
	std::cout << "ERROR: NN weights normalization FAILED."
		  << std::endl;
	return -1;
      }

      // also sets initial weights to be "orthogonal" against each other
      if(negfeedback){
	math::blas_real<float> alpha = 0.5f;
	negative_feedback_between_neurons(*nn, data, alpha);
      }
#endif
      
    }
    else if(load == true || lmethod  == "info"){
      if(verbose)
	std::cout << "Loading the previous network data from the disk." << std::endl;

      if(sbnn->load(nnfn) == false){
	if(bnn->load(nnfn) == false){
	  std::cout << "ERROR: Loading neural network failed (bnn)." << std::endl;
	  if(nn) delete nn;
	  if(snn) delete snn;
	  if(bnn) delete bnn;
	  if(sbnn) delete sbnn;
	  nn = NULL;
	  snn = NULL;
	  bnn = NULL;
	  sbnn = NULL;
	  return -1;
	}
	else{
	  if(snn) delete snn;
	  if(sbnn) delete sbnn;
	  snn = NULL;
	  sbnn = NULL;
	}
      }
      else{ // we use sbnn (superresolutional numbers)
	if(bnn) delete bnn;
	bnn = NULL;
      }

      if(sbnn != NULL){

	std::vector< math::vertex< math::superresolution<
	  math::blas_real<float>,
	  math::modular<unsigned int> > > > weights;
	
	nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > > nnParams;
	
	if(sbnn->exportSamples(nnParams, weights) == false){
	  std::cout << "ERROR: Loading neural network failed." << std::endl;
	  if(nn) delete nn;
	  if(snn) delete snn;
	  if(bnn) delete bnn;
	  if(sbnn) delete sbnn;
	  nn = NULL;
	  bnn = NULL;
	  sbnn = NULL;
	  return -1;
	}
	
	if(lmethod != "info"){
	  std::vector<unsigned int> loadedArch;
	  nnParams.getArchitecture(loadedArch);
	  
	  if(arch.size() != loadedArch.size()){
	    std::cout << "ERROR: Mismatch between loaded network architecture and parameter architecture." << std::endl;
	    if(nn) delete nn;
	    if(snn) delete snn;
	    if(bnn) delete bnn;
	    if(sbnn) delete sbnn;
	    nn = NULL;
	    bnn = NULL;
	    sbnn = NULL;
	    return -1;
	  }
	
	  for(unsigned int i=0;i<arch.size();i++){
	    if(arch[i] != loadedArch[i]){
	      std::cout << "ERROR: Mismatch between loaded network architecture and parameter architecture." << std::endl;
	      if(nn) delete nn;
	      if(snn) delete snn;
	      if(bnn) delete bnn;
	      if(sbnn) delete sbnn;
	      nn = NULL;
	      bnn = NULL;
	      sbnn = NULL;
	      return -1;
	    }
	  }
	}
	

	*snn = nnParams;
	
	// just pick one randomly if there are multiple ones
	unsigned int index = 0;
	if(weights.size() > 1)
	  index = rand() % weights.size();
	
	if(snn->importdata(weights[index]) == false){
	  std::cout << "ERROR: Loading neural network failed (incorrect network architecture?)." << std::endl;
	  if(nn) delete nn;
	  if(snn) delete snn;
	  if(bnn) delete bnn;
	  if(sbnn) delete sbnn;
	  return -1;
	}
	
      }
      else{ // bnn != NULL

	std::vector< math::vertex< whiteice::math::blas_real<float> > > weights;
	nnetwork< whiteice::math::blas_real<float> > nnParams;
	
	if(bnn->exportSamples(nnParams, weights) == false){
	  std::cout << "ERROR: Loading neural network failed." << std::endl;
	  if(nn) delete nn;
	  if(bnn) delete bnn;
	  nn = NULL;
	  return -1;
	}
	
	if(lmethod != "info"){
	  std::vector<unsigned int> loadedArch;
	  nnParams.getArchitecture(loadedArch);
	  
	  if(arch.size() != loadedArch.size()){
	    std::cout << "ERROR: Mismatch between loaded network architecture and parameter architecture." << std::endl;
	    if(nn) delete nn;
	    if(bnn) delete bnn;
	    nn = NULL;
	    return -1;
	  }
	
	  for(unsigned int i=0;i<arch.size();i++){
	    if(arch[i] != loadedArch[i]){
	      std::cout << "ERROR: Mismatch between loaded network architecture and parameter architecture." << std::endl;
	      if(nn) delete nn;
	      if(snn) delete snn;
	      if(bnn) delete bnn;
	      if(sbnn) delete sbnn;
	      nn = NULL;
	      return -1;
	    }
	  }
	}
	

	*nn = nnParams;
	
	// just pick one randomly if there are multiple ones
	unsigned int index = 0;
	if(weights.size() > 1)
	  index = rand() % weights.size();
	
	if(nn->importdata(weights[index]) == false){
	  std::cout << "ERROR: Loading neural network failed (incorrect network architecture?)." << std::endl;
	  if(nn) delete nn;
	  if(snn) delete snn;
	  if(bnn) delete bnn;
	  if(sbnn) delete sbnn;
	  return -1;
	}
      
      }
    }

    // checks if we only need to calculate subnet (if there are frozen layers before the first non-frozen layer)
    
    nnetwork< whiteice::math::blas_real<float> >* parent_nn = NULL;
    bayesian_nnetwork< whiteice::math::blas_real<float> >* parent_bnn = NULL;
    dataset< whiteice::math::blas_real<float> >* parent_data = NULL;
    unsigned int initialFrozen = 0;

    nnetwork< math::superresolution<math::blas_real<float>, math::modular<unsigned int> > >* parent_snn = NULL;
    bayesian_nnetwork< math::superresolution<math::blas_real<float>, math::modular<unsigned int> > >* parent_sbnn = NULL;
    dataset< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >* parent_sdata = NULL;
    
    if(load == true && lmethod != "use" && lmethod != "minimize" && lmethod != "info")
    {
      std::vector<bool> frozen;
      nn->getFrozen(frozen);
      
      while(frozen[initialFrozen] == true) initialFrozen++;

      if(initialFrozen > 0){
	subnet = true;
	parent_nn = nn;
	parent_snn = snn; 
	parent_bnn = bnn;
	parent_sbnn = sbnn;

	nn = nn->createSubnet(initialFrozen); // create subnet by skipping the first N layers
	snn = snn->createSubnet(initialFrozen); // create subnet by skipping the first N layers
	if(bnn) bnn = bnn->createSubnet(initialFrozen); // create subnet by skipping the firsst N layers
	if(sbnn) sbnn = sbnn->createSubnet(initialFrozen); // create subnet by skipping the firsst N layers

	if(verbose)
	  printf("Optimizing subnet (%d parameters in neural network)..\n", nn->exportdatasize());

	parent_data = new dataset< whiteice::math::blas_real<float> >(data);
	parent_sdata = new dataset< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >(sdata);

	const unsigned int newInputDimension = nn->getInputs(0);
	std::vector< math::vertex< math::blas_real<float> > > samples; // new input samples
	std::vector< math::vertex< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > > > ssamples; // new input samples

	for(unsigned int i=0;i<data.size(0);i++){
	  parent_nn->input() = data.access(0, i);
	  parent_nn->calculate(false, true); // collect samples per each layer
	}

	for(unsigned int i=0;i<sdata.size(0);i++){
	  parent_snn->input() = sdata.access(0, i);
	  parent_snn->calculate(false, true); // collect samples per each layer
	}

	parent_nn->getSamples(samples, initialFrozen);
	parent_snn->getSamples(ssamples, initialFrozen);

	data.resetCluster(0, "input", newInputDimension);
	data.add(0, samples);
	
	sdata.resetCluster(0, "input", newInputDimension);
	sdata.add(0, ssamples);
      }
      
    }
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // learning or activation
    if(lmethod == "info"){

      if(sbnn)
	sbnn->printInfo();
      else if(bnn)
	bnn->printInfo();
      
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "mix"){
      // mixture of experts
      Mixture< whiteice::math::blas_real<float> > moe(2, SIMULATION_DEPTH, overfit, negfeedback); 
            

      time_t t0 = time(0);
      unsigned int counter = 0;
      unsigned int iterations = 0;
      whiteice::math::blas_real<float> error = 1000.0;
      whiteice::linear_ETA<float> eta;

      if(samples > 0)
	eta.start(0.0f, (double)samples);

      moe.minimize(*nn, data);
      
      while(error > math::blas_real<float>(0.001f) &&
	    (counter < secs || secs <= 0) && // compute max SECS seconds
	    (iterations < samples || samples <= 0) && // or max samples
	    moe.solutionConverged() == false && moe.isRunning() == true && // or until solution converged.. (or exit due error)
	    !stopsignal)
      {
	sleep(1);

	std::vector< whiteice::math::vertex< whiteice::math::blas_real<float> > > weights;
	std::vector< whiteice::math::blas_real<float> > errors;
	std::vector< whiteice::math::blas_real<float> > percent;
	unsigned int changes = 0;
	
	moe.getSolution(weights, errors, percent, iterations, changes);

	error = 0.0;
	double c = 0;

	for(unsigned int i=0;i<errors.size();i++){
	  if(isinf(errors[i]) == false){
	    error += errors[i];
	    c++;
	  }
	}

	if(c > 0.0) error /= c;
	else error = 1000.0;
	
	eta.update(iterations);
	
	time_t t1 = time(0);
	counter = (unsigned int)(t1 - t0); // time-elapsed
	
	if(secs > 0){
	  printf("\r                                                            \r");
	  printf("%d iters: %f [%.1f minutes]",
		 iterations, 
		 error.c[0], (secs - counter)/60.0f);
	}
	else{
	  printf("\r                                                            \r");
	  printf("%d/%d iters: %f [%.1f minutes]",
		 iterations, samples,  
		 error.c[0], eta.estimate()/60.0f);
	  
	}
	  
	fflush(stdout);
      }

      if(moe.isRunning())
	moe.stopComputation();

      // TODO: copy majority expert as the predicting expert (so we just work against noise here..)
      assert(0); // FIXME!
      
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "bbrbm" || lmethod == "gbrbm"){
      // greedely learns Bernoulli-Bernoulli RBM or Gaussian-Bernoulli RBM
      // where the last layer is linear predictor/regression layer
      nl = whiteice::nnetwork< whiteice::math::blas_real<float> >::sigmoid;
      nn->setNonlinearity(whiteice::nnetwork< whiteice::math::blas_real<float> >::sigmoid);
      nn->setNonlinearity(nn->getLayers()-1,
			  whiteice::nnetwork< whiteice::math::blas_real<float> >::pureLinear);

      // the idea here is that we train N-M-output neural network where M is much larger than input
      // RBM does non-linear transformation and calculates features which are then linearly combined.
      // For example, optimize 2-layer  1207-10000-1 neural network where 10000 calculates
      // 10.000 features from input space which are then linearly combined.

      printf("Starting RBM neural network optimization (ignoring parameters)..\n");

      bool binary = false; // as the default assumes the first layer is gaussian RBM

      if(lmethod == "bbrbm") binary = true;  // trains full binary RBM (last layer is linear)
      else binary = false;                   // trains gaussian-bernoully RBM (last layer is linear)
      
      if(deep_pretrain_nnetwork(nn, data, binary, verbose) == false){
	printf("ERROR: deep pretraining of nnetwork failed.\n");
	return -1;
      }

#if 0
      printf("RBM LEARNED NETWORK\n");
      nn->printInfo(); // prints general information about trained nnetwork
#endif
      
      if(bnn->importNetwork(*nn) == false){
	std::cout << "ERROR: internal error cannot import optimized RBM to data structure" << std::endl;
	return -1;
      }
      
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "lbfgs"){
      
      if(verbose){
	if(overfit == false){
	  if(secs > 0)
	    std::cout << "Starting neural network L-BFGS optimization with early stopping (T=" << secs << " seconds).."
		      << std::endl;
	  else
	    std::cout << "Starting neural network L-BFGS optimization with early stopping.."
		      << std::endl;
	}
	else{
	  if(secs > 0)
	    std::cout << "Starting neural network L-BFGS optimization (T=" << secs << " seconds threads).."
		      << std::endl;
	  else
	    std::cout << "Starting neural network L-BFGS optimization.."
		      << std::endl;
	}
      }

      if(secs <= 0 && samples <= 0){
	fprintf(stderr, "L-BFGS search requires --time or --samples command line switch.\n");
	return -1;
      }

      
      rLBFGS_nnetwork< whiteice::math::blas_real<float> > bfgs(*nn, data, SIMULATION_DEPTH, overfit, negfeedback);

      bfgs.setGradientOnly(true);
      
      {
	time_t t0 = time(0);
	unsigned int counter = 0;
	math::blas_real<float> error = 1000.0f;
	math::vertex< whiteice::math::blas_real<float> > w;
	unsigned int iterations = 0;
	whiteice::linear_ETA<float> eta;

	if(samples > 0)
	  eta.start(0.0f, (double)samples);

	// initial starting position
	nn->exportdata(w);
	
	bfgs.minimize(w);

	while(error > math::blas_real<float>(0.001f) &&
	      (counter < secs || secs <= 0) && // compute max SECS seconds
	      (iterations < samples || samples <= 0) && // or max samples
	      bfgs.solutionConverged() == false && bfgs.isRunning() == true && // or until solution converged.. (or exit due error)
	      !stopsignal) 
	{
	  sleep(1);
	  
	  bfgs.getSolution(w, error, iterations);
	  
	  eta.update(iterations);

	  time_t t1 = time(0);
	  counter = (unsigned int)(t1 - t0); // time-elapsed

	  if(secs > 0){
	    printf("\r                                                            \r");
	    printf("%d iters: %f [%.1f minutes]",
		   iterations, 
		   error.c[0], (secs - counter)/60.0f);
	  }
	  else{
	    printf("\r                                                            \r");
	    printf("%d/%d iters: %f [%.1f minutes]",
		   iterations, samples,  
		   error.c[0], eta.estimate()/60.0f);

	  }
	  
	  fflush(stdout);
	}
	      
	
	if(secs > 0){
	  printf("\r                                                            \r");
	  printf("%d iters: %f [%.1f minutes]\n",
		 iterations,
		 error.c[0], (secs - counter)/60.0f);
	}
	else{
	  printf("\r                                                            \r");
	  printf("%d/%d iters: %f [%.1f minutes]\n",
		 iterations, samples,  
		 error.c[0], eta.estimate()/60.0f);
	}
	
	if(bfgs.solutionConverged()){
	  printf("Optimizer solution converged and cannot improve the result further.\n");
	}
	else if(bfgs.isRunning() == false){
	  printf("Optimizer stopped running (early stopping/overfitting).\n");
	}
	  
	fflush(stdout);

	bfgs.stopComputation();
	
	// gets the final (optimum) solution
	bfgs.getSolution(w, error, iterations);
	
	if(nn->importdata(w) == false){
	  std::cout << "ERROR: internal error" << std::endl;
	  return -1;
	}
	
	if(bnn->importNetwork(*nn) == false){
	  std::cout << "ERROR: internal error" << std::endl;
	  return -1;
	}
      }
      
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "plbfgs"){

      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      if(verbose){
	if(overfit == false){
	  if(secs > 0)
	    std::cout << "Starting parallel neural network L-BFGS optimization with early stopping (T=" << secs << " seconds, " << threads << " threads).."
		      << std::endl;
	  else
	    std::cout << "Starting parallel neural network L-BFGS optimization with early stopping (" << threads << " threads).."
		      << std::endl;
	}
	else{
	  if(secs > 0)
	    std::cout << "Starting parallel neural network L-BFGS optimization (T=" << secs << " seconds, " << threads << " threads).."
		      << std::endl;
	  else
	    std::cout << "Starting parallel neural network L-BFGS optimization (" << threads << " threads).."
		      << std::endl;
	}
      }

      if(secs <= 0 && samples <= 0){
	fprintf(stderr, "L-BFGS search requires --time or --samples command line switch.\n");
	return -1;
      }

      // FIXME add support for recursive neural networks
      pLBFGS_nnetwork< whiteice::math::blas_real<float> > bfgs(*nn, data, overfit, negfeedback);
      
      {
	time_t t0 = time(0);
	unsigned int counter = 0;
	math::blas_real<float> error = 1000.0f;
	math::vertex< whiteice::math::blas_real<float> > w;
	unsigned int iterations = 0;
	whiteice::linear_ETA<float> eta;

	if(samples > 0)
	  eta.start(0.0f, (double)samples);

	// initial starting position
	// nn->exportdata(w);
	
	bfgs.minimize(threads);

	while(error > math::blas_real<float>(0.001f) &&
	      (counter < secs || secs <= 0) && // compute max SECS seconds
	      (iterations < samples || samples <= 0) && 
	      !stopsignal)
	{
	  sleep(5);

	  bfgs.getSolution(w, error, iterations);
	  
	  // error = bfgs.getError(w); // we already have the error
	  
	  eta.update(iterations);

	  time_t t1 = time(0);
	  counter = (unsigned int)(t1 - t0); // time-elapsed

	  if(secs > 0){
	    printf("\r                                                            \r");
	    printf("%d iters: %f [%.1f minutes]",
		   iterations, 
		   error.c[0], (secs - counter)/60.0f);
	  }
	  else{
	    printf("\r                                                            \r");
	    printf("%d/%d iters: %f [%.1f minutes]",
		   iterations, samples,  
		   error.c[0], eta.estimate()/60.0f);

	  }

	  fflush(stdout);
	}
	      
	
	if(secs > 0){
	  printf("\r                                                            \r");
	  printf("%d iters: %f [%.1f minutes]\n",
		 iterations,
		 error.c[0], (secs - counter)/60.0f);
	}
	else{
	  printf("\r                                                            \r");
	  printf("%d/%d iters: %f [%.1f minutes]\n",
		 iterations, samples,  
		 error.c[0], eta.estimate()/60.0f);
	}
	  
	fflush(stdout);
	
	bfgs.stopComputation();

	// gets the final (optimum) solution
	bfgs.getSolution(w, error, iterations);
	
	if(nn->importdata(w) == false){
	  std::cout << "ERROR: internal error" << std::endl;
	  return -1;
	}
	
	if(bnn->importNetwork(*nn) == false){
	  std::cout << "ERROR: internal error" << std::endl;
	  return -1;
	}
	
      }
      
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "random"){

      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      if(verbose)
	std::cout << "Starting neural network parallel random search (T=" << secs << " seconds, " << threads << " threads).."
		  << std::endl;

      if(secs <= 0){
	fprintf(stderr, "Random search requires --time TIME command line switch.\n");
	return -1;
      }

#if 0
      // hack to test ultradeep
      // NOTE: brute-forcing does not really work..
      {
	std::vector< math::vertex< whiteice::math::blas_real<float> > > input;
	std::vector< math::vertex< whiteice::math::blas_real<float> > > output;
	
	data.getData(0, input);
	data.getData(1, output);
	
	UltraDeep ud;
	
	while(1){
	  ud.calculate(input, output);
	}
	return 0;
      }
#endif
      
      math::NNRandomSearch< whiteice::math::blas_real<float> > search;
      search.startOptimize(data, arch, threads);

      
      {
	time_t t0 = time(0);
	unsigned int counter = 0;
	math::blas_real<float> error = 100.0f;
	unsigned int solutions = 0;
	
	
	while(error > math::blas_real<float>(0.001f) &&
	      counter < secs && // compute max SECS seconds
	      !stopsignal) 
	{
	  search.getSolution(*nn, error, solutions);

	  sleep(1);
	  
	  time_t t1 = time(0);
	  counter = (unsigned int)(t1 - t0); // time-elapsed

	  printf("\r                                                            \r");
	  printf("%d tries: %f [%.1f minutes]", solutions, error.c[0], (secs - counter)/60.0f);
	  fflush(stdout);
	}

	printf("\r                                                            \r");
	printf("%d tries: %f [%.1f minutes]\n", solutions, error.c[0], (secs - counter)/60.0f);
	fflush(stdout);

	search.stopComputation();
	
	// gets the final (optimum) solution
	if(search.getSolution(*nn, error, solutions) == false){
	  std::cout << "ERROR: Cannot get result from optimizer (internal error)." << std::endl;
	}

	if(bnn->importNetwork(*nn) == false){
	  std::cout << "ERROR: Cannot transfer neural network data (internal error)." << std::endl;
	}
      }
      

    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "pgrad"){
      
      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      if(verbose)
	std::cout << "Starting neural network parallel multistart gradient descent (T=" << secs << " seconds, " << threads << " threads).."
		  << std::endl;
      
      if(secs <= 0){
	fprintf(stderr, "Parallel gradient descent requires --time TIME command line switch.\n");
	return -1;
      }
      
      
      math::NNGradDescent< whiteice::math::blas_real<float> > grad(negfeedback);
      grad.setUseMinibatch(true);
      grad.setOverfit(overfit);
      
      if(samples > 0)
	grad.startOptimize(data, *nn, threads, samples, dropout);
      else
	grad.startOptimize(data, *nn, threads, 0xFFFFFFFF, dropout);

      
      {
	time_t t0 = time(0);
	unsigned int counter = 0;
	math::blas_real<float> error = 100.0f;
	unsigned int solutions = 0;
	
	
	while((counter < secs) && grad.isRunning() && !stopsignal) // compute max SECS seconds
	{
	  grad.getSolution(*nn, error, solutions);

	  sleepms(5000);
	  
	  time_t t1 = time(0);
	  counter = (unsigned int)(t1 - t0); // time-elapsed

	  printf("\r                                                            \r");
	  printf("%d tries: %f [%.1f minutes]", solutions, error.c[0], (secs - counter)/60.0f);
	  fflush(stdout);
	  
	}

	printf("\r                                                            \r");
	printf("%d tries: %f [%.1f minutes]\n", solutions, error.c[0], (secs - counter)/60.0f);
	fflush(stdout);

	grad.stopComputation();

	// gets the final (optimum) solution
	grad.getSolution(*nn, error, solutions);
	bnn->importNetwork(*nn);
      }

      
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "sgrad"){
      
      if(verbose){
	std::cout << "Starting neural network gradient descent optimizer [superresolutional/polynomial arithmetic numbers].."
		  << std::endl;
	if(overfit == false)
	  std::cout << "Early stopping (testing dataset)." << std::endl;
	else
	  std::cout << "Overfitting to whole dataset (recommended)." << std::endl;
      }

      if(snn == NULL || nn == NULL){
	std::cout << "ERROR: neural network data structure doesn't exist!" << std::endl;
	return -1;
      }


      std::vector< std::vector<bool> > dropout_neurons;
      
      // convert net to snet
      {
	for(unsigned int l=0;l<nn->getLayers();l++){
	  snn->setNonlinearity(l, (whiteice::nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >::nonLinearity)(nn->getNonlinearity(l)));
	  snn->setFrozen(l, nn->getFrozen(l));
	}

	snn->setResidual(nn->getResidual());
	snn->setBatchNorm(nn->getBatchNorm());
	
	if(dropout){
	  printf("FIXME: dropout is not currently supported by SGD!\n");
	  snn->setDropOut(dropout_neurons);
	  return -1;
	}
      }

      const bool use_minibatch = true;

      whiteice::SGD_snet< math::blas_real<float> > sgd(*snn, data, overfit, use_minibatch);
      
      math::superresolution<math::blas_real<float>,
			    math::modular<unsigned int> > lrate(0.01f); // WAS: 0.0001, 0.01

      math::superresolution<math::blas_real<float>,
			    math::modular<unsigned int> > error;
      
      math::vertex< math::superresolution<math::blas_real<float>,
					  math::modular<unsigned int> > > w0;
      
      snn->exportdata(w0);
      
      sgd.setAdaptiveLRate(true); // was: false [adaptive don't work]
      sgd.setSmartConvergenceCheck(false); // [too easy to stop for convergence]
      
      if(sgd.minimize(w0, lrate, 0, 1000) == false){ // was: 200
	printf("ERROR: Cannot start SGD optimizer.\n");
	return -1;
      }
      
      int old_iters = -1;
      
      while(sgd.isRunning() && !stopsignal){
	sleep(1);
	
	unsigned int iters = 0;
	
	sgd.getSolutionStatistics(error, iters);
	
	if(((int)iters) > old_iters){
	  std::cout << "iter: " << iters << " error: " << error[0] << std::endl;
	  old_iters = (int)iters;
	}
      }

      printf("SGD Optimizer stopped.\n");

      {
	std::vector< math::vertex< math::superresolution< math::blas_real<float>,
							  math::modular<unsigned int> > > > best_weights_list;

	unsigned int iters = 0;
	sgd.getSolution(w0, error, iters);
	best_weights_list.push_back(w0);
	
	// stores only the best weights found using gradient descent
	sbnn->importSamples(*snn, best_weights_list);
      }
      
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "grad"){

      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      if(verbose){
	std::cout << "Starting neural network gradient descent optimizer.."
		  << std::endl;
	if(overfit == false)
	  std::cout << "Early stopping (testing dataset)." << std::endl;
      }

      // number of crossvalidation datasets to compute
      unsigned int CROSSVALIDATION_K = 0;
      if(crossvalidation == false){
	CROSSVALIDATION_K = 1;
      }
      else{
	CROSSVALIDATION_K = 10;
      }
      
      std::vector< math::vertex< whiteice::math::blas_real<float> > > best_weights_list;
      auto initial_nn = *nn;
      
      
      math::vertex< whiteice::math::blas_real<float> > best_weights;
      whiteice::RNG< whiteice::math::blas_real<float> > rng(true); // hardware random number generator

      for(unsigned int cvdk=0;cvdk<CROSSVALIDATION_K && stopsignal == false;cvdk++){
	if(CROSSVALIDATION_K > 1)
	  printf("Crossvalidation dataset %d/%d.\n", cvdk+1, CROSSVALIDATION_K);

	if(load == false){
	  nn->randomize();
	}
	else{
	  *nn = initial_nn;
	}
	
	// divide data to training and testing sets
	dataset< whiteice::math::blas_real<float> > dtrain, dtest;
	
	dtrain  = data;
	dtest   = data;
	
	dtrain.clearData(0);
	dtrain.clearData(1);
	dtest.clearData(0);
	dtest.clearData(1);
	
	for(unsigned int i=0;i<data.size(0);i++){
	  const unsigned int r = (rng.rand() % 10); // was & 3
	  
	  if(r < 9){ // 90% go to training data (was: 80%, 75%, 50%)
	    math::vertex< whiteice::math::blas_real<float> > in  = data.access(0,i);
	    math::vertex< whiteice::math::blas_real<float> > out = data.access(1,i);
	    
	    dtrain.add(0, in,  true);
	    dtrain.add(1, out, true);
	  }
	  else if(r == 9){ // 10% go to testing data (was: 10%, 25%, 50%)
	    math::vertex< whiteice::math::blas_real<float> > in  = data.access(0,i);
	    math::vertex< whiteice::math::blas_real<float> > out = data.access(1,i);
	    
	    dtest.add(0, in,  true);
	    dtest.add(1, out, true);	    
	  }
	  else{
	    assert(0); // error this should never happen
	  }
	}

	
	if(overfit){ // we keep full data both in training and testing sets
	  dtrain  = data;
	  dtest   = data;
	}

	if(dtrain.size(0) == 0 || dtest.size(0) == 0){ // too little data to make division
	  dtrain  = data;
	  dtest   = data;
	}

	// 1. normal gradient descent optimization using dtrain dataset
	{
	  //math::vertex< whiteice::math::blas_real<float> > grad, err, weights;
	  math::vertex< whiteice::math::blas_real<float> > weights;	  
	  time_t t0 = time(0);
	  unsigned int counter = 0;
	  math::blas_real<float> error, mean_ratio;
	  math::blas_real<float> prev_error;
	  math::blas_real<float> lrate = math::blas_real<float>(0.05f);
	  math::blas_real<float> delta_error = 0.0f;	  

	  math::blas_real<float> minimum_error = 10000000000.0f;
	  
	  std::list< math::blas_real<float> > ratios;

	  // early stopping if error has not decreased within MAX_NOIMPROVE_ITERS iterations of training
	  int noimprovement_counter = 0; // used to diagnosize stuck to local minimum (no overfitting allowed)
	  const int MAX_NOIMPROVE_ITERS = 1000;

	  // drop out code do works (0.8) reasonably well
	  const double retain_probability = 0.8;

	  nn->exportdata(best_weights);
	  nn->exportdata(weights);
	  
	  error = 1000.0f;
	  prev_error = 1000.0f;
	  mean_ratio = 1.0f;

	  whiteice::linear_ETA<float> eta;
	  if(samples > 0)
	    eta.start(0.0f, (double)samples);
	  
	  const unsigned int SAMPLE_SIZE = 100; // was 500

	  
	  while(((counter < samples && samples > 0) ||
		 (counter < secs && secs > 0)) && !stopsignal)
	  {

	    while(ratios.size() > 10)
	      ratios.pop_front();
	    
	    math::blas_real<float> inv = 1.0f;

	    if(ratios.size() > 0) inv = 1.0f/ratios.size();
	    
	    mean_ratio = 1.0f;
	    
	    for(auto& r : ratios){ // mean ratio of the past 10 iters
	      mean_ratio *= r;
	    }
	    
	    mean_ratio = math::pow(mean_ratio, inv);
	    
	    if(overfit == false){
#if 0
	      if(mean_ratio > 3.0)
		if(counter > 10) break; // do not stop immediately
#endif
	      if(noimprovement_counter >= MAX_NOIMPROVE_ITERS){
		// stop if there has not been improvement for sometime
		break;
	      }
	    }

	    nn->importdata(weights);
	    prev_error = error;
	    error = 0.0;

	    // goes through data, calculates gradient
	    // exports weights, weights -= lrate*gradient
	    // imports weights back

	    math::vertex< whiteice::math::blas_real<float> > sumgrad;
	    math::blas_real<float> ninv = 1.0/SAMPLE_SIZE;

	    sumgrad.resize(nn->gradient_size());
	    sumgrad.zero();
	    

#pragma omp parallel shared(sumgrad)
	    {
	      nnetwork< math::blas_real<float> > net(*nn);
	      //nnetwork< math::blas_real<float> >& net = *nn;
	      math::vertex< whiteice::math::blas_real<float> > grad(nn->gradient_size()), err(nn->output_size());
	      math::vertex< whiteice::math::blas_real<float> > sgrad(nn->gradient_size());
	      sgrad.zero();
	      grad.zero();
	      err.zero();							     
	      
#pragma omp for nowait schedule(dynamic)
	      for(unsigned int i=0;i<SAMPLE_SIZE;i++){
		if(dropout) net.setDropOut(retain_probability);
		
		const unsigned index = rng.rand() % dtrain.size(0);
		
		net.input() = dtrain.access(0, index);
		net.calculate(true);
		err = net.output() - dtrain.access(1, index);

		if(MNE)
		  err.normalize();

		if(net.mse_gradient(err, grad) == false)
		  std::cout << "gradient failed." << std::endl;
		
		sgrad += ninv*grad;
	      }
	      
#pragma omp critical (jgjiwrejorefrgehAAfgge)
	      {
		sumgrad += sgrad;
	      }
	    }

	    
	    if(nn->exportdata(weights) == false){
	      std::cout << "FATAL: export failed." << std::endl;
	      exit(-1);
	    }
	    
	    
	    lrate = 1.0;
	    math::vertex< whiteice::math::blas_real<float> > w;

	    do{	      
	      lrate = 0.5f*lrate;
	      w = weights;
	      w -= lrate*sumgrad;

	      nn->importdata(w);

	      
	      if(negfeedback){
		// using negative feedback heuristic
		math::blas_real<float> alpha = 0.5f;
		negative_feedback_between_neurons(*nn, dtrain, alpha);	      
	      }
	      
	      error = 0.0;

	      // calculates error from the testing dataset (should use train?)
#pragma omp parallel shared(error)
	      {
		math::vertex< whiteice::math::blas_real<float> > err;
		math::blas_real<float> e = 0.0;
		nnetwork< math::blas_real<float> > net(*nn);
		
#pragma omp for nowait schedule(dynamic)		
		for(unsigned int i=0;i<dtrain.size(0);i++){
		  const unsigned int index = i; // rand() % dtrain.size(0);
		  auto input = dtrain.access(0, index);
		  auto output = dtrain.access(1, index);

		  if(dropout) net.setDropOut(retain_probability);
		  
		  //nn->calculate(input, output); // thread-safe
		  net.calculate(input, output); // thread-safe
		  err = dtrain.access(1, index) - output;
		  
		  e += (err*err)[0] / math::blas_real<float>((double)err.size());
		}

#pragma omp critical (mvowefnigihgRERE)
		{
		  error += e;
		}
	      }
	      
	      
	      error /= dtrain.size(0);
	      error *= math::blas_real<float>(0.5f); // missing scaling constant

	      // if the error is negative (error increases)
	      // we try again with smaller lrate
	      
	      delta_error = (prev_error - error);

	      // leaky error reduction, we sometimes allow jump to worse position in gradient direction
	      // if((rng.rand() % 5) == 0) delta_error = +1.0;
	      if((rng.rand() % 5) == 0 && error < 10.0)
		delta_error = +1.0;
	    }
	    while(delta_error < 0.0 && lrate > 10e-25);

	    weights = w;

	    // check if error has decreased in testing set (early stopping)
	    {
	      whiteice::math::blas_real<float> test_error = 0.0;
	      nn->importdata(weights);

	      // calculates error from the testing dataset (should use train?)
#pragma omp parallel shared(test_error)
	      {
		math::vertex< whiteice::math::blas_real<float> > err;
		math::blas_real<float> e = 0.0;
		nnetwork< math::blas_real<float> > net(*nn);
		
#pragma omp for nowait schedule(dynamic)		
		for(unsigned int i=0;i<dtest.size(0);i++){
		  const unsigned int index = i; // rand() % dtest.size(0);
		  auto input = dtest.access(0, index);
		  auto output = dtest.access(1, index);

		  if(dropout) net.setDropOut(retain_probability);
		  
		  //nn->calculate(input, output); // thread-safe
		  net.calculate(input, output); // thread-safe
		  err = dtest.access(1, index) - output;
		  
		  e += (err*err)[0] / math::blas_real<float>((double)err.size());
		}

#pragma omp critical (korwepojwogGEVET)
		{
		  test_error += e;
		}
	      }
	      
	      
	      test_error /= dtest.size(0);
	      test_error *= math::blas_real<float>(0.5f); // missing scaling constant

	      
	      // check if we have new best solution
	      if(test_error < minimum_error){
		best_weights = weights;
		minimum_error = test_error;
		noimprovement_counter = 0;
	      }
	      else{
		noimprovement_counter++;
	      }
	    }
	    
	    
	    math::blas_real<float> ratio = error / minimum_error;
	    ratios.push_back(ratio);

	    if(secs > 0){
	      time_t t1 = time(0);
	      counter = (unsigned int)(t1 - t0);
	    }
	    else{
	      counter++;
	      eta.update((double)counter);
	    }

	    printf("\r                                                                                   \r");
	    if(samples > 0){
	      printf("%d/%d iters: %f (%f) <%f> [%.1f minutes]",
		     counter, samples, error.c[0], mean_ratio.c[0],
		     minimum_error.c[0],
		     eta.estimate()/60.0);	      
	    }
	    else{ // secs
	      printf("%d secs: %f (%f) <%f> [%.1f minutes]",
		     counter, error.c[0], mean_ratio.c[0],
		     minimum_error.c[0],
		     (secs - counter)/60.0);
	    }
	    
	    fflush(stdout);
	  }

	  
	  if(nn->importdata(best_weights) == false){
	    printf("ERROR: importing best weights FAILED.\n");
	    return -1;
	  }
	  
	  if(dropout){
	    nn->removeDropOut(retain_probability);
	    nn->exportdata(best_weights);
	  }

	  printf("\r                                                                                   \r");
	  printf("%d/%d : %f (%f) <%f> [%.1f minutes]\n",
		 counter, samples, error.c[0], mean_ratio.c[0],
		 minimum_error.c[0],
		 eta.estimate()/60.0);
	  fflush(stdout);
	}

	best_weights_list.push_back(best_weights);
      }



      {
	// stores only the best weights found using gradient descent
	bnn->importSamples(*nn, best_weights_list);
      }
      
      
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "bayes"){
      
      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      threads = 1;

      if(verbose){
	if(secs > 0){
	  std::cout << "Starting neural network bayesian inference (T=" << secs << " secs, "
		    << threads << " threads)..."
		    << std::endl;
	}
	else{
	  std::cout << "Starting neural network bayesian inference (" << threads << " threads)..."
		    << std::endl;
	}
      }

      
      // whiteice::HMC_convergence_check< whiteice::math::blas_real<float> > hmc(*nn, data, adaptive);
      unsigned int ptlayers =
	(unsigned int)(math::log(data.size(0))/math::log(1.25));
      
      if(ptlayers <= 10) ptlayers = 10;
      else if(ptlayers > 100) ptlayers = 100;

      // std::cout << "Parallel Tempering depth: " << ptlayers << std::endl;

      // need for speed: (we downsample
      

      whiteice::HMC< whiteice::math::blas_real<float> > hmc(*nn, data, adaptive);
      // whiteice::UHMC< whiteice::math::blas_real<float> > hmc(*nn, data, adaptive);
      
      // whiteice::PTHMC< whiteice::math::blas_real<float> > hmc(ptlayers, *nn, data, adaptive);
      whiteice::linear_ETA<float> eta;
      
      time_t t0 = time(0);
      unsigned int counter = 0;
      
      hmc.startSampler();
      
      if(samples > 0)
	eta.start(0.0f, (double)samples);
      
      while(((hmc.getNumberOfSamples() < samples && samples > 0) || (counter < secs && secs > 0)) && !stopsignal){
      // while(!hmc.hasConverged() && !stopsignal){

	eta.update((double)hmc.getNumberOfSamples());
	
	if(hmc.getNumberOfSamples() > 0){
	  if(secs > 0){
	    printf("\r                                                            \r");
	    printf("%d samples: %f [%.1f minutes]",
		   hmc.getNumberOfSamples(),
		   hmc.getMeanError(1).c[0],		   
		   (secs - counter)/60.0);
	  }
	  else{
	    printf("\r                                                            \r");
	    printf("%d/%d samples : %f [%.1f minutes]",
		   hmc.getNumberOfSamples(),
		   samples,
		   hmc.getMeanError(1).c[0],
		   eta.estimate()/60.0);
	  }
	  fflush(stdout);
	}
	
	sleep(5);

	time_t t1 = time(0);
	counter = (unsigned int)(t1 - t0);
      }
      
      hmc.stopSampler();

      if(secs > 0){
	printf("\r                                                            \r");
	printf("%d samples : %f\n",
	       hmc.getNumberOfSamples(), hmc.getMeanError(100).c[0]);
      }
      else{
	printf("\r                                                            \r");
	printf("%d/%d samples : %f\n",
	       hmc.getNumberOfSamples(), samples, hmc.getMeanError(100).c[0]);
      }
      
      fflush(stdout);

      // nn->importdata(hmc.getMean());
      delete nn;
      nn = NULL;

      // saves samples to bayesian network data structure
      // (keeps only 50% latest samples) so we ignore initial tail
      // to high probability distribution
      {
	bnn = new bayesian_nnetwork< whiteice::math::blas_real<float> >();
	
	unsigned int savedSamples = 1;
	if(hmc.getNumberOfSamples() > 1){
	  savedSamples = hmc.getNumberOfSamples()/2;
	}

	assert(hmc.getNetwork(*bnn, savedSamples) == true);
      }

      // instead of using mean weight vector
      // we now use y = E[network(x,w)] in bayesian inference
      //
      // TODO: what we really want is
      //       the largest MODE of p(w|data) distribution as 
      //       this is then the global minima (assuming samples
      //       {w_i} converge to p(w|data)).
      
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "minimize"){
      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      if(verbose){
	if(secs > 0){
	  std::cout << "Finding neural network input (genetic algorithms) with minimum response (T=" 
		    << secs << " seconds)"
		    << std::endl;
	}
	else{
	  std::cout << "Finding neural network input (genetic algorithms) with minimum response.."
		    << std::endl;
	}
      }
      
            
      if(bnn->load(nnfn) == false){
	std::cout << "Loading neural network failed." << std::endl;
	delete nn;
	delete bnn;
	nn = NULL;
	bnn = NULL;
	return -1;
      }
      
      // loads nnetwork weights from BNN
      {
	std::vector< math::vertex< math::blas_real<float> > > weights;
	nnetwork< whiteice::math::blas_real<float> > nnParam;
	
	if(bnn->exportSamples(nnParam, weights) == false){
	  std::cout << "Loading neural network failed." << std::endl;
	  delete nn;
	  delete bnn;
	  nn = NULL;
	  return -1;
	}

	if(weights.size() == 0){
	  std::cout << "Loading neural network failed." << std::endl;
	  delete nn;
	  delete bnn;
	  nn = NULL;
	  return -1;
	}
	
	delete nn;
	nn = new nnetwork< whiteice::math::blas_real<float> >(nnParam);

	*nn = nnParam;
	nn->importdata(weights[(rand() % weights.size())]);;
      }
      
      nnetwork_function< whiteice::math::blas_real<float> > nf(*nn);
      GA3< whiteice::math::blas_real<float> > ga(&nf);

      time_t t0 = time(0);
      unsigned int counter = 0;
      
      ga.minimize();
      
      whiteice::math::vertex< whiteice::math::blas_real<float> > s;
      math::blas_real<float> r;
      
      while(((ga.getGenerations() < samples && samples > 0) || (counter < secs && secs > 0)) && !stopsignal){
	r = ga.getBestSolution(s);
	const unsigned int g = ga.getGenerations();
	
	if(secs > 0){
	  printf("\r                                                            \r");
	  printf("%d generations: %f [%.1f minutes]",
		 g, r.c[0],
		 (secs - counter)/60.0);
	}
	else{
	  printf("\r                                                            \r");
	  printf("%d/%d generations : %f",
		 g, samples, r.c[0]);
	}
	fflush(stdout);
	
	sleep(1);
	
	time_t t1 = time(0);
	counter = (unsigned int)(t1 - t0);
      }
      
      printf("\n");
      
      data.invpreprocess(0, s);
      std::cout << "Best solution found (" << r << "): " << s << std::endl;
      
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "edit"){
      if(verbose)
	std::cout << "Editing neural network architecture.." << std::endl;

      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }

      if(bnn->load(nnfn) == false){
	std::cout << "Loading neural network failed." << std::endl;
	if(nn) delete nn;
	if(bnn) delete bnn;
	nn = NULL;
	return -1;
      }

      std::vector<unsigned int> oldArch;
      bnn->getArchitecture(oldArch);

      if(arch[0] != oldArch[0] ||
	 arch[arch.size()-1] != oldArch[oldArch.size()-1])
      {
	std::cout << "ERROR: new architecture input/output mismatch\n"
		  << std::endl;
	if(nn) delete nn;
	if(bnn) delete bnn;
	return -1;
      }

      bool same = false;

      if(arch.size() == oldArch.size())
      {
	unsigned int counter = 0;
	
	for(unsigned int i=0;i<arch.size();i++){
	  if(arch[i] == oldArch[i]) counter++;
	}

	if(counter == arch.size()){
	  same = true;
	}
      }

      // transform architecture to given arch and add given
      // nonlinenarity as a new nonlinearity for the changed
      // layers (except the final linear one)
      
      if(same == false){
	if(bnn->editArchitecture(arch, nl) == false){
	  std::cout << "ERROR: Cannot transform network" << std::endl;
	  if(nn) delete nn;
	  if(bnn) delete bnn;
	  return -1;
	}
      }
      
      
    }
    //////////////////////////////////////////////////////////////
    else if(lmethod == "use"){
      if(verbose)
	std::cout << "Activating loaded neural network configuration.."
		  << std::endl;

      if(sbnn){
	
	if(sbnn->load(nnfn) == false){
	
	  if(snn) delete snn;
	  if(sbnn) delete sbnn;
	  snn = NULL;
	  sbnn = NULL;
	}
      }

      if(bnn && sbnn == NULL){
	
	if(bnn->load(nnfn) == false){
	  
	  if(nn) delete nn;
	  if(bnn) delete bnn;
	  nn = NULL;
	  bnn = NULL;
	}
      }

      if(bnn == NULL && sbnn == NULL){
	std::cout << "Loading neural network failed." << std::endl;
	return -1;
      }
      else{
      }
      

      if(sbnn){

	const int RDIM1 = ((int)sbnn->inputSize()) - ((int)sdata.dimension(0));
	const int RDIM2 = ((int)sbnn->outputSize()) - ((int)sdata.dimension(1));
	
	if(sbnn->inputSize() != data.dimension(0) && SIMULATION_DEPTH == 1){
	  std::cout << "Neural network input dimension mismatch for input dataset ("
		    << sbnn->inputSize() << " != " << sdata.dimension(0) << ")"
		    << std::endl;
	  delete sbnn;
	  delete snn;
	  snn = NULL;
	  return -1;
	}
	else if((RDIM1 != RDIM2 || (RDIM1 <= 0 || RDIM2 <= 0)) && SIMULATION_DEPTH > 1){
	  std::cout << "Recurrent neural network input dimension mismatch for input dataset ("
		    << sbnn->inputSize() << " != " << sdata.dimension(0)+sbnn->outputSize() << ")"
		    << std::endl;
	  delete sbnn;
	  delete snn;
	  snn = NULL;
	  return -1;
	}
	
	
	
	bool compare_clusters = false;
	
	if(sdata.getNumberOfClusters() == 2){
	  if(sdata.size(0) > 0 && sdata.size(1) > 0 && 
	     sdata.size(0) == sdata.size(1)){
	    compare_clusters = true;
	  }
	  
	  const int RDIM1 = ((int)sbnn->inputSize()) - ((int)sdata.dimension(0));
	  
	  if(sbnn->outputSize() != sdata.dimension(1)+RDIM1){
	    std::cout << "Neural network output dimension mismatch for dataset ("
		      << sbnn->outputSize() << " != " << sdata.dimension(1) << ")"
		      << std::endl;
	    delete sbnn;
	    delete snn;
	    return -1;	    
	  }
	}
	else if(sdata.getNumberOfClusters() == 3){
	  if(sdata.size(0) > 0 && sdata.size(1) > 0 && 
	     sdata.size(0) == data.size(1)){
	    compare_clusters = true;
	  }
	  
	  const int RDIM1 = ((int)sbnn->inputSize()) - ((int)sdata.dimension(0));
	  
	  if(sbnn->outputSize() != sdata.dimension(1)+RDIM1){
	    std::cout << "Neural network output dimension mismatch for dataset ("
		      << sbnn->outputSize() << " != " << sdata.dimension(1) << ")"
		      << std::endl;
	    delete sbnn;
	    delete snn;
	    return -1;	    
	  }
	  
	  if(sbnn->outputSize() != sdata.dimension(2)+RDIM1){
	    std::cout << "Neural network output dimension mismatch for dataset ("
		      << sbnn->outputSize() << " != " << sdata.dimension(2) << ")"
		      << std::endl;
	    delete sbnn;
	    delete snn;
	    return -1;	    
	  }
	}
	else{
	  std::cout << "Unsupported number of data clusters in dataset: "
		    << sdata.getNumberOfClusters() << std::endl;
	  delete sbnn;
	  delete snn;
	  return -1;	    
	}
	
	
	if(compare_clusters == true){
	  
	  math::superresolution< math::blas_real<float>, math::modular<unsigned int> > error1 = math::superresolution< math::blas_real<float>, math::modular<unsigned int> >(0.0f);
	  math::superresolution< math::blas_real<float>, math::modular<unsigned int> > error2 = math::superresolution< math::blas_real<float>, math::modular<unsigned int> >(0.0f);
	  math::superresolution< math::blas_real<float>, math::modular<unsigned int> > c = math::superresolution< math::blas_real<float>, math::modular<unsigned int> >(0.5f);
	  math::vertex< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > > err;
	  
	  whiteice::nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > > single_snn(*snn);
	  std::vector< math::vertex< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > > > sweights;
	  
	  sbnn->exportSamples(single_snn, sweights);
	  auto w = sweights[0];
	  w.zero();
	  
	  for(auto& wi : sweights)
	    w += wi;
	  
	  w /= math::superresolution< math::blas_real<float>, math::modular<unsigned int> >(sweights.size()); // E[w]
	  
	  
	  {
	    std::vector<unsigned int> arch2;
	    single_snn.getArchitecture(arch2);
	    
	    if(arch2.size() != arch.size()){
	      printf("ERROR: cannot import weights from bayesian nnetwork to a single network (mismatch network layout %d != %d).\n",
		     (int)arch2.size(), (int)arch.size());
	      delete sbnn;
	      delete snn;
	      exit(-1);
	    }
	    
	    for(unsigned int i=0;i<arch.size();i++){
	      if(arch2[i] != arch[i]){
		printf("ERROR: cannot import weights from bayesian nnetwork to a single network (mismatch network layout).\n");
		for(unsigned int i=0;i<arch.size();i++)
		  printf("%d ", arch[i]);
		printf("\n");
		
		for(unsigned int i=0;i<arch2.size();i++)
		  printf("%d ", arch2[i]);
		printf("\n");
		
		delete sbnn;
		delete snn;
		exit(-1);
	      }
	    }
	  }
	  
	  
	  if(single_snn.importdata(w) == false){
	    printf("ERROR: cannot import weights from bayesian nnetwork to a single network.\n");
	    delete sbnn;
	    delete snn;
	    exit(-1);
	  }
	  
	  whiteice::linear_ETA<float> eta;
	  
	  if(sdata.size(0) > 0)
	    eta.start(0.0f, (double)sdata.size(0));
	  
	  unsigned int counter = 0; // number of points calculated..
	  
	  for(unsigned int i=0;i<sdata.size(0) && !stopsignal;i++){
	    
	    math::vertex< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > > out1;
	    math::vertex< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > > rdim;
	    
	    rdim.resize(sbnn->inputSize() - sdata.dimension(0));
	    rdim.zero();
	    
	    sbnn->calculate(sdata.access(0, i), out1, SIMULATION_DEPTH, 0);
	    err = sdata.access(1,i) - out1;
	    
	    for(unsigned int i=0;i<err.size();i++)
	      error1 += c*(err[i][0]*err[i][0]) / math::superresolution< math::blas_real<float>, math::modular<unsigned int> >((double)err.size());
	    
	    single_snn.input().zero();
	    single_snn.output().zero();
	    single_snn.input().write_subvertex(sdata.access(0, i), 0);	  
	    
	    for(unsigned int d=0;d<SIMULATION_DEPTH;d++){
	      if(SIMULATION_DEPTH > 1){
		single_snn.input().write_subvertex(rdim, sdata.dimension(0));
	      }
	      
	      single_snn.calculate(false, false);
	      
	      if(SIMULATION_DEPTH > 1){
		single_snn.output().subvertex(rdim, sdata.dimension(1), rdim.size());
	      }
	    }
	    
	    if(SIMULATION_DEPTH > 1){
	      single_snn.output().subvertex(out1, 0, sdata.dimension(1));
	    }
	    else{
	      out1 = single_snn.output();
	    }
	    
	    err = sdata.access(1, i) - out1;
	    
	    for(unsigned int i=0;i<err.size();i++)
	      error2 += c*(err[i][0]*err[i][0]) / math::blas_real<float>((double)err.size());
	    
	    eta.update((double)(i+1));

	    double percent = 100.0f*((double)i+1)/((double)sdata.size(0));
	    double etamin  = eta.estimate()/60.0f;
	    
	    printf("\r                                                            \r");
	    printf("%d/%d (%.1f%%) [ETA: %.1f minutes]", i+1, sdata.size(0), percent, etamin);
	    fflush(stdout);
	    
	    counter++;
	  }
	  
	  printf("\n"); fflush(stdout);
	  
	  if(counter > 0){
	    error1 /= math::blas_real<float>((double)counter);
	    error2 /= math::blas_real<float>((double)counter);
	  }
	  
	  std::cout << "Average error in dataset (E[f(x|w)]): " << error1[0] << std::endl;
	  std::cout << "Average error in dataset (f(x|E[w])): " << error2[0] << std::endl;
	}
	
	else{
	  std::cout << "Predicting data points.." << std::endl;
	  
	  if(sdata.getNumberOfClusters() == 2 && sdata.size(0) > 0){
	    
	    sdata.clearData(1);
	    
	    sdata.setName(0, "input");
	    sdata.setName(1, "output");
	    
	    whiteice::linear_ETA<float> eta;
	    
	    if(sdata.size(0) > 0)
	      eta.start(0.0f, (double)sdata.size(0));
	    
	    for(unsigned int i=0;i<data.size(0) && !stopsignal;i++){
	      
	      math::vertex< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > > out;
	      math::vertex< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > > var;
	      
	      eta.update((double)i);
	      
	      double percent = 100.0 * ((double)(i+1))/((double)data.size(0));
	      double etamin  = eta.estimate()/60.0f;
	      
	      printf("\r                                                            \r");
	      printf("%d/%d (%.1f%%) [ETA %.1f minutes]", i+1, data.size(0), percent, etamin);
	      fflush(stdout);
	      
	      sbnn->calculate(sdata.access(0, i),  out, SIMULATION_DEPTH, 0);
	      
	      // we do NOT preprocess the output but inject it directly into dataset
	      sdata.add(1, out, true);
	    }
	    
	    printf("\n");
	    fflush(stdout);	  
	  }
	  else if(data.getNumberOfClusters() == 3 && data.size(0) > 0){
	    
	    sdata.clearData(1);
	    sdata.clearData(2);
	    
	    sdata.setName(0, "input");
	    sdata.setName(1, "output");
	    sdata.setName(2, "output_stddev");
	    
	    for(unsigned int i=0;i<sdata.size(0) && !stopsignal;i++){
	      math::vertex< math::superresolution< math::blas_real<float>,
						   math::modular<unsigned int> > > out;
	      math::vertex< math::superresolution< math::blas_real<float>,
						   math::modular<unsigned int> > > var;
	      math::matrix< math::superresolution< math::blas_real<float>,
						   math::modular<unsigned int> > > cov;
	      
	      sbnn->calculate(sdata.access(0, i), out, cov, SIMULATION_DEPTH, 0);
	      
	      // we do NOT preprocess the output but inject it directly into dataset
	      sdata.add(1, out, true);
	      
	      var.resize(cov.xsize());	    
	      for(unsigned int j=0;j<cov.xsize();j++)
		var[j] = math::sqrt(abs(cov(j,j))); // st.dev.
	      
	      sdata.add(2, var, true);
	    }
	  }
	  else{
	    printf("Bad dataset data (no data?)\n");
	    exit(-1);
	  }
	  
	  if(stopsignal){
	    if(sbnn) delete bnn;
	    if(snn)  delete nn;
	    
	    exit(-1);
	  }

	  // convert sdata to data
	  {
	    // data.clear();
	    
	    std::vector< dataset< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >::data_normalization > preprocessings;
	    
	    math::vertex< math::blas_real<float> > w;
	    
	    // sdata to data
	    
	    for(unsigned int c=0;c<sdata.getNumberOfClusters();c++){

	      data.clearData(c);

	      if(data.dimension(c) != sdata.dimension(c)){
		printf("ERROR: dataset dimensions mismatch.\n");
		return -1;
	      }
	      
	      for(unsigned int i=0;i<sdata.size(c);i++){
		
		auto v = sdata.access(c, i);
		//sdata.invpreprocess(c, v);
		
		w.resize(v.size());
	  
		for(unsigned int k=0;k<v.size();k++)
		  whiteice::math::convert(w[k], v[k]);
	  
		data.add(c, w, true);
	      }
	
	      // add preprocessings to data that were in the original dataset sdata
	      // [don't work very well currently]

	      data.repreprocess(c);
	      
#if 0
	      
	      sdata.getPreprocessings(c, preprocessings);
	
	      for(unsigned int i=0;i<preprocessings.size();i++){
		data.preprocess(c, (dataset< math::blas_real<float> >::data_normalization)(preprocessings[i]));
	      }
#endif
	    }
	  }

	  
	  if(data.save(datafn) == true){
	    std::cout << "Storing results to dataset file: " 
		      << datafn << std::endl;
	  }
	  else{
	    std::cout << "Storing results to dataset file FAILED." << std::endl;
	  }
	
	}
	
      }
      else if(bnn){
	
	const int RDIM1 = ((int)bnn->inputSize()) - ((int)data.dimension(0));
	const int RDIM2 = ((int)bnn->outputSize()) - ((int)data.dimension(1));
	
	if(bnn->inputSize() != data.dimension(0) && SIMULATION_DEPTH == 1){
	  std::cout << "Neural network input dimension mismatch for input dataset ("
		    << bnn->inputSize() << " != " << data.dimension(0) << ")"
		    << std::endl;
	  delete bnn;
	  delete nn;
	  nn = NULL;
	  return -1;
	}
	else if((RDIM1 != RDIM2 || (RDIM1 <= 0 || RDIM2 <= 0)) && SIMULATION_DEPTH > 1){
	  std::cout << "Recurrent neural network input dimension mismatch for input dataset ("
		    << bnn->inputSize() << " != " << data.dimension(0)+bnn->outputSize() << ")"
		    << std::endl;
	  delete bnn;
	  delete nn;
	  nn = NULL;
	  return -1;
	}
	
	
	
	bool compare_clusters = false;
	
	if(data.getNumberOfClusters() == 2){
	  if(data.size(0) > 0 && data.size(1) > 0 && 
	     data.size(0) == data.size(1)){
	    compare_clusters = true;
	  }
	  
	  const int RDIM1 = ((int)bnn->inputSize()) - ((int)data.dimension(0));
	  
	  if(bnn->outputSize() != data.dimension(1)+RDIM1){
	    std::cout << "Neural network output dimension mismatch for dataset ("
		      << bnn->outputSize() << " != " << data.dimension(1) << ")"
		      << std::endl;
	    delete bnn;
	    delete nn;
	    return -1;	    
	  }
	}
	else if(data.getNumberOfClusters() == 3){
	  if(data.size(0) > 0 && data.size(1) > 0 && 
	     data.size(0) == data.size(1)){
	    compare_clusters = true;
	  }
	  
	  const int RDIM1 = ((int)bnn->inputSize()) - ((int)data.dimension(0));
	  
	  if(bnn->outputSize() != data.dimension(1)+RDIM1){
	    std::cout << "Neural network output dimension mismatch for dataset ("
		      << bnn->outputSize() << " != " << data.dimension(1) << ")"
		      << std::endl;
	    delete bnn;
	    delete nn;
	    return -1;	    
	  }
	  
	  if(bnn->outputSize() != data.dimension(2)+RDIM1){
	    std::cout << "Neural network output dimension mismatch for dataset ("
		      << bnn->outputSize() << " != " << data.dimension(2) << ")"
		      << std::endl;
	    delete bnn;
	    delete nn;
	    return -1;	    
	  }
	}
	else{
	  std::cout << "Unsupported number of data clusters in dataset: "
		    << data.getNumberOfClusters() << std::endl;
	  delete bnn;
	  delete nn;
	  return -1;	    
	}
	
	
	if(compare_clusters == true){
	  math::blas_real<float> error1 = math::blas_real<float>(0.0f);
	  math::blas_real<float> error2 = math::blas_real<float>(0.0f);
	  math::blas_real<float> c = math::blas_real<float>(0.5f);
	  math::vertex< whiteice::math::blas_real<float> > err;
	  
	  whiteice::nnetwork< whiteice::math::blas_real<float> > single_nn(*nn);
	  std::vector< math::vertex< whiteice::math::blas_real<float> > > weights;
	  
	  bnn->exportSamples(single_nn, weights);
	  math::vertex< whiteice::math::blas_real<float> > w = weights[0];
	  w.zero();
	  
	  for(auto& wi : weights)
	    w += wi;
	  
	  w /= weights.size(); // E[w]
	  
	  
	  {
	    std::vector<unsigned int> arch2;
	    single_nn.getArchitecture(arch2);
	    
	    if(arch2.size() != arch.size()){
	      printf("ERROR: cannot import weights from bayesian nnetwork to a single network (mismatch network layout %d != %d).\n",
		     (int)arch2.size(), (int)arch.size());
	      delete bnn;
	      delete nn;
	      exit(-1);
	    }
	    
	    for(unsigned int i=0;i<arch.size();i++){
	      if(arch2[i] != arch[i]){
		printf("ERROR: cannot import weights from bayesian nnetwork to a single network (mismatch network layout).\n");
		for(unsigned int i=0;i<arch.size();i++)
		  printf("%d ", arch[i]);
		printf("\n");
		
		for(unsigned int i=0;i<arch2.size();i++)
		  printf("%d ", arch2[i]);
		printf("\n");
		
		delete bnn;
		delete nn;
		exit(-1);
	      }
	    }
	  }
	  
	  
	  if(single_nn.importdata(w) == false){
	    printf("ERROR: cannot import weights from bayesian nnetwork to a single network.\n");
	    delete bnn;
	    delete nn;
	    exit(-1);
	  }
	  
	  whiteice::linear_ETA<float> eta;
	  
	  if(data.size(0) > 0)
	    eta.start(0.0f, (double)data.size(0));
	  
	  unsigned int counter = 0; // number of points calculated..
	  
	  for(unsigned int i=0;i<data.size(0) && !stopsignal;i++){
	    math::vertex< whiteice::math::blas_real<float> > out1;
	    math::vertex< whiteice::math::blas_real<float> > rdim;
	    
	    rdim.resize(bnn->inputSize() - data.dimension(0));
	    rdim.zero();
	    
	    bnn->calculate(data.access(0, i), out1, SIMULATION_DEPTH, 0);
	    err = data.access(1,i) - out1;
	    
	    for(unsigned int i=0;i<err.size();i++)
	      error1 += c*(err[i]*err[i]) / math::blas_real<float>((double)err.size());
	    
	    single_nn.input().zero();
	    single_nn.output().zero();
	    single_nn.input().write_subvertex(data.access(0, i), 0);	  
	    
	    for(unsigned int d=0;d<SIMULATION_DEPTH;d++){
	      if(SIMULATION_DEPTH > 1){
		single_nn.input().write_subvertex(rdim, data.dimension(0));
	      }
	      
	      single_nn.calculate(false, false);
	      
	      if(SIMULATION_DEPTH > 1){
		single_nn.output().subvertex(rdim, data.dimension(1), rdim.size());
	      }
	    }
	    
	    if(SIMULATION_DEPTH > 1){
	      single_nn.output().subvertex(out1, 0, data.dimension(1));
	    }
	    else{
	      out1 = single_nn.output();
	    }
	    
	    err = data.access(1, i) - out1;
	    
	    for(unsigned int i=0;i<err.size();i++)
	      error2 += c*(err[i]*err[i]) / math::blas_real<float>((double)err.size());
	    
	    eta.update((double)(i+1));

	    double percent = 100.0f*((double)i+1)/((double)data.size(0));
	    double etamin  = eta.estimate()/60.0f;
	    
	    printf("\r                                                            \r");
	    printf("%d/%d (%.1f%%) [ETA: %.1f minutes]", i+1, data.size(0), percent, etamin);
	    fflush(stdout);
	    
	    counter++;
	  }
	  
	  printf("\n"); fflush(stdout);
	  
	  if(counter > 0){
	    error1 /= math::blas_real<float>((double)counter);
	    error2 /= math::blas_real<float>((double)counter);
	  }
	  
	  std::cout << "Average error in dataset (E[f(x|w)]): " << error1 << std::endl;
	  std::cout << "Average error in dataset (f(x|E[w])): " << error2 << std::endl;
	}
	
	else{
	  std::cout << "Predicting data points.." << std::endl;
	  
	  if(data.getNumberOfClusters() == 2 && data.size(0) > 0){
	    
	    data.clearData(1);
	    
	    data.setName(0, "input");
	    data.setName(1, "output");
	    
	    whiteice::linear_ETA<float> eta;
	    
	    if(data.size(0) > 0)
	      eta.start(0.0f, (double)data.size(0));
	    
	    for(unsigned int i=0;i<data.size(0) && !stopsignal;i++){
	      math::vertex< whiteice::math::blas_real<float> > out;
	      math::vertex< whiteice::math::blas_real<float> > var;
	      
	      eta.update((double)i);
	      
	      double percent = 100.0 * ((double)(i+1))/((double)data.size(0));
	      double etamin  = eta.estimate()/60.0f;
	      
	      printf("\r                                                            \r");
	      printf("%d/%d (%.1f%%) [ETA %.1f minutes]", i+1, data.size(0), percent, etamin);
	      fflush(stdout);
	      
	      bnn->calculate(data.access(0, i),  out, SIMULATION_DEPTH, 0);
	      
	      // we do NOT preprocess the output but inject it directly into dataset
	      data.add(1, out, true);
	    }
	    
	    printf("\n");
	    fflush(stdout);	  
	  }
	  else if(data.getNumberOfClusters() == 3 && data.size(0) > 0){
	    
	    data.clearData(1);
	    data.clearData(2);
	    
	    data.setName(0, "input");
	    data.setName(1, "output");
	    data.setName(2, "output_stddev");
	    
	    for(unsigned int i=0;i<data.size(0) && !stopsignal;i++){
	      math::vertex< whiteice::math::blas_real<float> > out;
	      math::vertex< whiteice::math::blas_real<float> > var;
	      math::matrix< whiteice::math::blas_real<float> > cov;
	      
	      bnn->calculate(data.access(0, i), out, cov, SIMULATION_DEPTH, 0);
	      
	      // we do NOT preprocess the output but inject it directly into dataset
	      data.add(1, out, true);
	      
	      var.resize(cov.xsize());	    
	      for(unsigned int j=0;j<cov.xsize();j++)
		var[j] = math::sqrt(abs(cov(j,j))); // st.dev.
	      
	      data.add(2, var, true);
	    }
	  }
	  else{
	    printf("Bad dataset data (no data?)\n");
	    exit(-1);
	  }
	  
	  if(stopsignal){
	    if(bnn) delete bnn;
	    if(nn)  delete nn;
	    
	    exit(-1);
	  }
	  
	  if(data.save(datafn) == true){
	    std::cout << "Storing results to dataset file: " 
		      << datafn << std::endl;
	  }
	  else{
	    std::cout << "Storing results to dataset file FAILED." << std::endl;
	  }
	
	}
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // we have processed subnet and not the real data, we inject subnet data back into the master data structures
    if(subnet)
    {
      if(bnn){
	// we attempt to inject subnet data structure starting from initialFrozen:th layer to parent net
	if(parent_nn->injectSubnet(initialFrozen, nn) == false){
	  printf("ERROR: injecting subnet into larger master network FAILED (1).\n");
	  
	  delete nn; delete bnn;
	  delete parent_nn; delete parent_bnn;
	  delete parent_data;
	  
	  return -1;
	}
	
	if(parent_bnn->injectSubnet(initialFrozen, bnn) == false){
	  printf("ERROR: injecting subnet into larger master network FAILED (2).\n");
	  
	  delete nn; delete bnn;
	  delete parent_nn; delete parent_bnn;
	  delete parent_data;
	  
	  return -1;
	}
      
	delete nn;
	delete bnn;
	nn = parent_nn;
	bnn = parent_bnn;
	
	parent_nn = nullptr;
	parent_bnn = nullptr;
	
	delete parent_data; // we do not need to keep parent data structure
	parent_data = nullptr;
      }

      if(sbnn){
	// superresolutional data structures
	
	// we attempt to inject subnet data structure starting from initialFrozen:th layer to parent net
	if(parent_snn->injectSubnet(initialFrozen, snn) == false){
	  printf("ERROR: injecting subnet into larger master network FAILED (1).\n");
	  
	  delete snn; delete sbnn;
	  delete parent_snn; delete parent_sbnn;
	  delete parent_sdata;
	  
	  return -1;
	}
	
	if(parent_sbnn->injectSubnet(initialFrozen, sbnn) == false){
	  printf("ERROR: injecting subnet into larger master network FAILED (2).\n");
	  
	  delete snn; delete sbnn;
	  delete parent_snn; delete parent_sbnn;
	  delete parent_sdata;
	  
	  return -1;
	}
	
	delete snn;
	delete sbnn;
	snn = parent_snn;
	sbnn = parent_sbnn;
	
	parent_snn = nullptr;
	parent_sbnn = nullptr;
	
	delete parent_sdata; // we do not need to keep parent data structure
	parent_sdata = nullptr;
      }
      
    }
    
        
    if(lmethod != "use" && lmethod != "minimize" && lmethod != "info"){

      bool use_superreso = false;

      if(sbnn){
	if(sbnn->getNumberOfSamples() > 0){
	  if(sbnn->save(nnfn) == false){
	    std::cout << "Saving neural network data failed (sbnn)." << std::endl;
	    delete sbnn;
	    return -1;
	  }
	  else{
	    use_superreso = true;
	    if(verbose)
	      std::cout << "Saving neural network data: " << nnfn << std::endl;
	  }
	}
      }

	
      if(bnn && use_superreso == false){
	if(bnn->save(nnfn) == false){
	  std::cout << "Saving neural network data failed (bnn)." << std::endl;
	  delete bnn;
	  return -1;
	}
	else{
	  if(verbose)
	    std::cout << "Saving neural network data: " << nnfn << std::endl;
	}
      }
    }
    
    
    if(bnn){ delete bnn; bnn = 0; }
    if(nn){ delete nn; nn = 0; }

    if(sbnn){ delete sbnn; sbnn = 0; }
    if(snn){ delete snn; snn  = 0; }
    
    
    return 0;
  }
  catch(std::exception& e){
    std::cout << "FATAL ERROR: unexpected exception. Reason: " 
	      << e.what() << std::endl;
    return -1;
  }
  
}


void sigint_signal_handler(int s)
{
  stopsignal = true;
}


void install_signal_handler()
{
#ifndef WINOS
  struct sigaction sih;
  
  sih.sa_handler = sigint_signal_handler;
  sigemptyset(&sih.sa_mask);
  sih.sa_flags = 0;

  sigaction(SIGINT, &sih, NULL);
#endif
}


void sleepms(unsigned int ms)
{
#ifndef WINNT
  struct timespec ts;
  ts.tv_sec  = 0;
  ts.tv_nsec = ms*1000000;
  nanosleep(&ts, 0);
#else
  Sleep(ms);
#endif 
}


void print_usage(bool all)
{
  printf("Usage: nntool [options] [data] [arch] <nnfile> [lmethod]\n");
  
  if(!all){
    printf("Try 'nntool --help' for more information.\n");
    return;
  }
  
  
  printf("Create, train and use neural nets.\n\n");
  printf("-v                shows ETA and other details\n");
  printf("--help            shows this help\n");
  printf("--version         displays version and exits\n");
  printf("--info            prints net architecture information\n");
  printf("--no-init         don't use heuristics when initializing net\n");
  printf("--overfit         do not use early stopping/use whole data (grad,pgrad,lbfgs)\n");
  printf("--deep=*          pretrains neural net as a RBM\n");
  printf("                  (* = binary or gaussian input layer)\n");
  printf("--dropout         enable dropout heuristics (grad,pgrad)\n");
  printf("--noresidual      disable residual neural net (grad,pgrad)\n");
  printf("--crossvalidation random crossvalidation (K=10) (grad)\n");
  printf("--batchnorm       batch normalization between layers [not implemented yet]\n");
  printf("--recurrent N     simple recurrent net (lbfgs, use)\n");
  printf("--adaptive        adaptive step in bayesian HMC (bayes)\n");
  printf("--negfb           use negative feedback between neurons\n");
  printf("--load            use previously computed weights (grad,lbfgs,bayes)\n");
  printf("--time TIME       sets time limit for computations\n");
  printf("--samples N       use N samples or optimize for N iterations\n");
  printf("--threads N       uses N parallel threads (pgrad, plbfgs)\n");
  printf("--data N          only use N random samples of data\n");
  printf("[data]            dstool file containing data (binary file)\n");
  printf("[arch]            architecture of net (Eg. 3-10-9)\n");
  printf("<nnfile>          used/loaded/saved neural net weights file\n");
  printf("[lmethod]         method: use, random, grad, sgrad, pgrad, bayes,\n"); 
  printf("                  lbfgs, plbfgs, edit, (gbrbm, bbrbm, mix)\n");
  printf("                  edit edits net to have new architecture\n");
  printf("                  previous weights are preserved if possible\n");
  printf("                  sgrad uses polynomial arithmetic numbers to optimize nn\n");
  printf("\n");
  printf("                  Ctrl-C shutdowns the program.\n");
  printf("\n");
  printf("This program is distributed under GPL license (the author keeps full rights to code).\n");
  printf("Copyright <tomas.ukkonen@iki.fi> (other licenses available).\n");
  
}


