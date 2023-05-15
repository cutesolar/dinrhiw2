/* 
 * superresolution test cases
 *
 * Tomas Ukkonen
 */

#include <iostream>
#include <vector>

#include "superresolution.h"
#include "dataset.h"
#include "nnetwork.h"
#include "NNGradDescent.h"
#include "RNG.h"

#include "SGD_snet.h" // superresolutional optimizer



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
    //feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    //feenableexcept(FE_INVALID);
  }
#endif
  
}


using namespace whiteice;

int main()
{
  std::cout << "superresolution test cases" << std::endl;

  whiteice::logging.setPrintOutput(false);
  
  // creates training dataset
  std::cout << "Creating training dataset.." << std::endl;
  
  dataset<
    math::superresolution< math::blas_real<double>,
			   math::modular<unsigned int> > > data;

  dataset< math::blas_real<double> > data2;

  data.createCluster("input", 4);
  data.createCluster("output", 4);
  data2.createCluster("input", 4);
  data2.createCluster("output", 4);  
    
  std::vector< math::vertex<
    math::superresolution< math::blas_real<double>,
			   math::modular<unsigned int> > > > inputs;

  std::vector< math::vertex<
    math::superresolution< math::blas_real<double>,
			   math::modular<unsigned int> > > > outputs;

  std::vector< math::vertex<math::blas_real<double> > > inputs2;
  std::vector< math::vertex<math::blas_real<double> > > outputs2;

  RNG< math::blas_real<double> > prng;

  nnetwork< math::blas_real<double> > net;
  nnetwork< math::superresolution< math::blas_real<double>,
				   math::modular<unsigned int> > > snet;

  math::vertex< math::blas_real<double> > weights;
  math::vertex< math::superresolution< math::blas_real<double>,
				       math::modular<unsigned int> > > sweights;

  // was: 4-10-10-10-4 network
  // now: 5-50-50-50-50-4 network
  std::vector<unsigned int> arch;  
  arch.push_back(10);
  
  const unsigned int LAYERS = 2; // was: 2, 10, 40 [TODO: test 100 dimensional neural network]
  
  for(unsigned int l=0;l<LAYERS;l++)
    arch.push_back(20);
  
  arch.push_back(10);

  
#if 0
  arch.clear();
  arch.push_back(4);
  arch.push_back(20); // was: 20,50
  arch.push_back(20); // was: 20,50
  arch.push_back(4);
#endif
  
  
#if 0  
  arch.clear();
  arch.push_back(4);
  arch.push_back(50);
  arch.push_back(50);
  arch.push_back(50);
  arch.push_back(50);
  arch.push_back(4);
#endif

  // pureLinear non-linearity (layers are all linear) [pureLinear or rectifier]
  net.setArchitecture(arch, nnetwork< math::blas_real<double> >::rectifier); // rectifier, hermite 
  snet.setArchitecture(arch, nnetwork< math::superresolution<
		       math::blas_real<double>,
		       math::modular<unsigned int> > >::rectifier); // rectifier, hermite

  net.randomize();
  snet.randomize();
  net.setResidual(false); // was: true
  snet.setResidual(false);

  const bool BN = false; // batch normalization (on/off)

  if(BN){
    net.setBatchNorm(true);
    snet.setBatchNorm(true);

    std::cout << "Enabling BatchNorm(alization) between neural network layers.." << std::endl;
  }
  

  std::cout << "net weights size: " << net.gradient_size() << std::endl;
  std::cout << "snet weights size: " << snet.gradient_size() << std::endl;

  net.exportdata(weights);
  snet.exportdata(sweights);

  //std::cout << sweights << std::endl;
  
  //math::convert(sweights, weights);

  // sweights = abs(sweights); // drop complex parts of initial weights

  //snet.importdata(sweights);
  //snet.exportdata(sweights);

  //std::cout << "weights = " << weights << std::endl;
  //std::cout << "sweights = " << sweights << std::endl;

  math::vertex<
    math::superresolution< math::blas_real<double>,
			   math::modular<unsigned int> > > sx, sy;
  
  math::vertex< math::blas_real<double> > x, y;
  
  
  const unsigned int NUMDATAPOINTS = 1000; // was: 1000, 100 for testing purposes

  for(unsigned int i=0;i<NUMDATAPOINTS;i++){

    math::blas_real<double> sigma = 4.0;
    math::blas_real<double> f = 10.0, a = 1.10, w = 10.0, one = 1.0;

    x.resize(4);
    prng.normal(x);
    x = sigma*x; // x ~ N(0,4^2*I)

    y.resize(4);

    y[0] = math::sin((f*x[0]*x[1]*x[2]*x[3]));
    if(x[3].c[0] >= 0.0f)
      y[1] =  math::pow(a, (x[0]/(math::abs(x[2])+one)) );
    else
      y[1] = -math::pow(a, (x[0]/(math::abs(x[2])+one)) );

    y[2] = 0.0f;
    if(x[1].c[0] > 0.0f) y[2] += one;
    else y[2] -= one;
    if(x[3].c[0] > 0.0f) y[2] += one;
    else y[2] -= one;
    auto temp = math::cos(w*x[0]);
    if(temp.c[0] > 0.0f) y[2] += one;
    else y[2] -= one;
    
    y[3] = x[1]/(math::abs(x[0])+one) + x[2]*math::sqrt(math::abs(x[3])) + math::abs(x[3] - x[0]);

    // std::cout << "x = " << x << std::endl;
    //std::cout << "y = " << y << std::endl;

    //net.calculate(x, y); // USE nnetwork as a target function (easier)

    inputs2.push_back(x);
    outputs2.push_back(y);

    whiteice::math::convert(sx, x);
    whiteice::math::convert(sy, y);

    // std::cout << "sx = " << sx << std::endl;
    //std::cout << "sy = " << sy << std::endl;

    inputs.push_back(sx);
    outputs.push_back(sy);
    
    // calculates nnetwork response to y=f(sx) with net and snet which should give same results
    
    //net.calculate(x, y);

    // std::cout << "y  =  f(x) = " << y << std::endl;

    //snet.calculate(sx, sy);

    // std::cout << "sy = f(sx) = " << sy << std::endl;
  }

  data.add(0, inputs);
  data.add(1, outputs);

  // do not do [now: enabled]
  //data.preprocess(0);
  //data.preprocess(1);

  data2.add(0, inputs2);
  data2.add(1, outputs2);

  // do not do [now: enabled]
  //data2.preprocess(0);
  //data2.preprocess(1);

  if(data2.save("simpleproblem.ds") == false){
    printf("ERROR: saving data to file failed!\n");
    exit(-1);
  }

  
#if 1
  // use SHA-256 HASH data instead
  {
    printf("USING SHA-256 HASH CRYPTO DATASET.\n");
    fflush(stdout);

    data.clear();
    data2.clear();
    data2.load("hash-data.ds");

    std::cout << "data.size(0): " << data.size(0) << std::endl;
    std::cout << "data.size(1): " << data.size(1) << std::endl;
    std::cout << "data2.size(0): " << data2.size(0) << std::endl;
    std::cout << "data2.size(1): " << data2.size(1) << std::endl;
    
    
    // remove preprocessings from data
    data2.convert(0);
    data2.convert(1);

    //data2.preprocess(0);
    //data2.preprocess(1);
    
    data.clear();
    data.createCluster("input", 10);
    data.createCluster("output", 10);

    x.resize(10);
    y.resize(10);
    sx.resize(10);
    sy.resize(10);
    
    for(unsigned int i=0;i<data2.size(0);i++){
      x = data2.access(0, i);
      y = data2.access(1, i);

      // std::cout << x << std::endl;
      
      //y = y/255.0; // convert to [-2,2]/255 valued data! [with preprocess] (small values work better!)

      // scales [0,1] data to [-0.5,0.5] data
      for(unsigned int i=0;i<y.size();i++)
	y[i] -= 0.50;
      
      whiteice::math::convert(sx, x);
      whiteice::math::convert(sy, y);

      // DON'T SWAP ANYMORE AS WE HAVE SWAPPED DATA ALREADY
      if(data.add(0, sx) == false) assert(0);
      if(data.add(1, sy) == false) assert(0);
    }


    data2.clear();
    data2.createCluster("input", 10);
    data2.createCluster("output", 10);

    for(unsigned int i=0;i<data.size(0);i++){
      sx = data.access(0, i);
      sy = data.access(1, i);

      whiteice::math::convert(x, sx);
      whiteice::math::convert(y, sy);

      data2.add(0, x);
      data2.add(1, y);
    }
    
    
    data.downsampleAll(50); // should be at least 1000
    data2.downsampleAll(50); // should be at least 1000
  }
#endif

  
  std::cout << "data.size(0): " << data.size(0) << std::endl;
  std::cout << "data.size(1): " << data.size(1) << std::endl;
  std::cout << "data2.size(0): " << data2.size(0) << std::endl;
  std::cout << "data2.size(1): " << data2.size(1) << std::endl;


  // SGD gradient descent code for superresolution..
  if(0){

    std::cout << "Stochastic Gradient Descent (SGD) optimizer for superreso neural networks."
	      << std::endl;

    const bool overfit = true;
    const bool use_minibatch = false;
    
    whiteice::SGD_snet< math::blas_real<double> > sgd(snet, data2, overfit, use_minibatch);

    math::superresolution<math::blas_real<double>,
			  math::modular<unsigned int> > lrate(0.0001f); // WAS: 0.0001, 0.01

    math::vertex< math::superresolution<math::blas_real<double>,
					math::modular<unsigned int> > > w0;

    snet.exportdata(w0);

    sgd.setAdaptiveLRate(true); // was: false [adaptive don't work]
    sgd.setSmartConvergenceCheck(false); // [too easy to stop for convergence]

    if(sgd.minimize(w0, lrate, 0, 1000) == false){ // was: 200
      printf("ERROR: Cannot start SGD optimizer.\n");
      return -1;
    }

    int old_iters = -1;

    while(sgd.isRunning()){
      sleep(1);

      unsigned int iters = 0;

      math::superresolution<math::blas_real<double>,
			    math::modular<unsigned int> > error;

      sgd.getSolutionStatistics(error, iters);

      if(((int)iters) > old_iters){
	std::cout << "iter: " << iters << " error: " << error[0] << std::endl;
	old_iters = (int)iters;
      }
    }

    printf("SGD optimizer stopped.\n"); 

    return 0;
  }
  
  

  //////////////////////////////////////////////////////////////////////
  // next DO NNGradDescent<> && nnetwork<> to actually learn the data.
  if(0)
  {
    whiteice::math::NNGradDescent<math::blas_real<double>> grad;

    grad.startOptimize(data2, net, 1);

    while(grad.isRunning()){
      sleep(1);
      
      unsigned int iters = 0;
      whiteice::math::blas_real<double> error;
      
      grad.getSolutionStatistics(error, iters);
      
      std::cout << "iter: " << iters << " error: " << error << std::endl;
    }
  }
  
  //////////////////////////////////////////////////////////////////////

  // pretrainer for superresolution code
  {
    whiteice::PretrainNN< math::superresolution< math::blas_real<double>,
						 math::modular<unsigned int> > > pretrainer;

    pretrainer.setMatrixFactorization(true);

    if(pretrainer.startTrain(snet, data) == false){
      printf("PRETRAINER FAILED\n");
      return -1;
    }
    else{
      printf("PretrainNN started..\n");
    }

    while(pretrainer.isRunning()){
      sleep(1);
      unsigned int iters = 0;
      math::superresolution< math::blas_real<double>, math::modular<unsigned int> > error;

      pretrainer.getStatistics(iters, error);

      std::cout << "iter " << iters << " : " << error[0] << std::endl;
    }

    pretrainer.stopTrain();
    printf("PretrainNN stop.\n");
    
    pretrainer.getResults(snet);
  }
    

  // gradient descent code
  {
    math::vertex< math::superresolution<math::blas_real<double>,
					math::modular<unsigned int> > > weights, w0;
    
    math::vertex< math::superresolution<math::blas_real<double>,
					math::modular<unsigned int> > > sumgrad;

    std::vector< math::vertex< math::superresolution<math::blas_real<double>,
						     math::modular<unsigned int> > > > gradients;
    
    unsigned int counter = 0;
    math::superresolution<math::blas_real<double>,
			  math::modular<unsigned int> > error(1000.0f), min_error(1000.0f), latest_error(1000.0f);
    math::superresolution<math::blas_real<double>,
			  math::modular<unsigned int> > lrate(0.01f); // WAS: 0.05
    
    double lratef = 0.01;
    double best_error = 10.0f;
    unsigned int grad_search_counter = 0;

    std::vector<double> errors; // history of errors (10 last errors)
    
    
    while(abs(error)[0].real() > math::blas_real<double>(0.001f) &&
	  grad_search_counter < 300 &&
	  lratef > 1e-100 && counter < 100000)
    {
      auto batchdata = data;
      //batchdata.downsampleAll(200);
      
      error = math::superresolution<math::blas_real<double>,
				    math::modular<unsigned int> >(0.0f);
      sumgrad.zero();
      
      // goes through data, calculates gradient
      // exports weights, weights -= 0.01*gradient
      // imports weights back

      math::superresolution<math::blas_real<double>,
			    math::modular<unsigned int> > ninv =
	math::superresolution<math::blas_real<double>,
			      math::modular<unsigned int> >
	(1.0f/(batchdata.size(0)*batchdata.access(1,0).size()));

      math::superresolution<math::blas_real<double>,
			    math::modular<unsigned int> > h, s0(1.0), epsilon(1e-30);

      h.ones(); // differential operation difference
      h = epsilon*h;

      snet.exportdata(weights);
      snet.exportdata(w0);


      if(BN){ // batch normalization code..
	std::vector< math::vertex< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > > > datav;
	batchdata.getData(0, datav);

	assert(snet.calculateBatchNorm(datav) == true);
      }

      sumgrad.resize(snet.exportdatasize());
      sumgrad.zero();

#pragma omp parallel
      {
	math::vertex< math::superresolution<math::blas_real<double>,
					    math::modular<unsigned int> > > grad, err, threaded_grad;

	math::superresolution<math::blas_real<double>,
			      math::modular<unsigned int> > threaded_error(0.0f);

	threaded_grad.resize(snet.exportdatasize());
	threaded_grad.zero();

	threaded_error.zero();

	//const auto& snet = snet; // marks snet to be const so there are no changes to snet
	//const auto& batchdata = batchdata; // marks batchdata to be const so there are no changes
	
#pragma omp for nowait
	for(unsigned int i=0;i<batchdata.size(0);i++){
	  
	  // selects K:th dimension in number and adjust weights according to it.
	  //const unsigned int K = prng.rand() % weights[0].size();
	  
	  {
	    const auto x = batchdata.access(0,i);
	    const auto y = batchdata.access(1,i);
	    
	    snet.calculate(x, err);
	    err -= y;
	    
	    
	    for(unsigned int j=0;j<err.size();j++){
	      const auto& ej = err[j];
	      //for(unsigned int k=0;k<ej.size();k++)
	      //  error += ninv*ej[k]*math::conj(ej[k]);
	      
	      threaded_error += ninv*math::sqrt(ej[0]*math::conj(ej[0])); // ABSOLUTE VALUE
	    }
	    
	    // this works with pureLinear non-linearity
	    math::matrix< math::superresolution<math::blas_real<double>,
						math::modular<unsigned int> > > DF;
	    
	    math::matrix< math::superresolution<math::blas_complex<double>,
						math::modular<unsigned int> > > cDF;
	    
	    snet.jacobian(x, DF);
	    cDF.resize(DF.ysize(), DF.xsize());
	    
	    // circular convolution in F-domain
	    
	    math::vertex<
	      math::superresolution< math::blas_complex<double>,
				     math::modular<unsigned int> > > ce, cerr;
	    
	    for(unsigned int j=0;j<DF.ysize();j++){
	      for(unsigned int i=0;i<DF.xsize();i++){
		whiteice::math::convert(cDF(j,i), DF(j,i));
		cDF(j,i).fft();
	      }
	    }
	    
	    ce.resize(err.size());
	    
	    for(unsigned int i=0;i<err.size();i++){
	      whiteice::math::convert(ce[i], err[i]);
	      ce[i].fft();
	    }
	    
	    cerr.resize(DF.xsize());
	    cerr.zero();
	    
	    for(unsigned int i=0;i<DF.xsize();i++){
	      auto ctmp = ce;
	      for(unsigned int j=0;j<DF.ysize();j++){
		cerr[i] += ctmp[j].circular_convolution(cDF(j,i));
	      }
	    }
	    
#if 1
	    // after we have FFT(gradient) which we convolve with FFT([1 0 ...]) dimensional number
	    
	    math::superresolution<math::blas_complex<double>,
				  math::modular<unsigned int> > one;
	    one.zero();
	    one[0] = whiteice::math::blas_complex<double>(1.0, 0.0);
	    one.fft();
	    
	    for(unsigned int i=0;i<cerr.size();i++)
	      cerr[i].circular_convolution(one);
#endif
	    
	    // finally we do inverse Fourier transform
	    
	    err.resize(cerr.size());
	    
	    for(unsigned int i=0;i<err.size();i++){
	      cerr[i].inverse_fft();
	      for(unsigned int k=0;k<err[i].size();k++)
		whiteice::math::convert(err[i][k], cerr[i][k]); // converts complex numbers to real
	    }
	    
	    grad = err;
	  }

	  threaded_grad += ninv*grad;
	}


#pragma omp critical
	{
	  sumgrad += threaded_grad;
	  error += threaded_error;
	}
	
      }

      // now we have gradient, sample from normal distribution of historical gradients
      // for better gradient search
      {
	const unsigned int GRAD_HISTORY = 10;
	
	gradients.push_back(sumgrad);
	
	while(gradients.size() > GRAD_HISTORY)
	  gradients.erase(gradients.begin());
	
	
	if(gradients.size() >= GRAD_HISTORY){ // sample from gradient distribution

	  const unsigned int dimensions = gradients.size()-1;
	  
	  math::matrix< math::superresolution<math::blas_real<double>,
					      math::modular<unsigned int> > > PCA;

	  std::vector< math::superresolution<math::blas_real<double>,
					      math::modular<unsigned int> > > eigenvalues;

	  auto mg = gradients[0];
	  mg.zero();

	  for(const auto & g : gradients)
	    mg += g;

	  mg /= math::superresolution< math::blas_real<double>, math::modular<unsigned int> >
	    (gradients.size());
	  

	  if(fastpca(gradients, dimensions, PCA, eigenvalues, false)){ // fast PCA is successful

	    // sample from N(0,I) [what is superresolutional normal distribution???] 
	    math::vertex< math::superresolution<math::blas_real<double>,
						math::modular<unsigned int> > > u, w;

	    u.resize(gradients[0].size());
	    w.resize(gradients[0].size());
	    w.zero();
	    

	    for(unsigned int j=0;j<u.size();j++)
	      for(unsigned int i=0;i<u[j].size();i++)
		u[j][i] = rng.normal().real();

	    for(unsigned int j=0;j<PCA.ysize();j++){

	      math::vertex< math::superresolution<math::blas_real<double>,
						  math::modular<unsigned int> > > v;

	      v.resize(PCA.xsize());

	      for(unsigned int i=0;i<v.size();i++){
		v[i] = PCA(j,i);
	      }

	      w += eigenvalues[j]*(u*v)*v;
	    }

	    
	    //w += mg;
	    // use mean value as the current gradient so only variance term is from gradient history..
	    w += sumgrad;
	    
	    // std::cout << "gradient sampled successfully from historical gradients (N=" << gradients.size() << ")" << std::endl;
	    
	    sumgrad = w;
	  }
	  
	}
	
      }

      // std::cout << "sumgrad = " << sumgrad << std::endl;

      auto abserror = error;
      abserror[0] = abs(abserror[0]);
      
      for(unsigned int i=1;i<abserror.size();i++){
	abserror[0] += abs(abserror[i]);
	//abserror[i] = math::blas_real<double>(0.0f);
	abserror[i] = 0.0f;
      }

      auto orig_error = abserror;
      auto abserror2 = abserror;

      grad_search_counter = 0;
      
      while(grad_search_counter < 300){ // until error becomes smaller

	auto delta_grad = sumgrad;
	
	for(unsigned int j=0;j<sumgrad.size();j++){
	  for(unsigned int k=0;k<sumgrad[0].size();k++){
	    delta_grad[j][k] *= lratef;
	  }
	}
	
	weights = w0 - delta_grad;

	snet.importdata(weights);

	// recalculates error in dataset

	error.zeros();

#pragma omp parallel
	{
	  math::superresolution<math::blas_real<double>,
				math::modular<unsigned int> > thread_error(0.0f);

	  math::vertex< math::superresolution<math::blas_real<double>,
					      math::modular<unsigned int> > > err;
	  
#pragma omp for nowait
	  for(unsigned int i=0;i<batchdata.size(0);i++){
	    
	    snet.calculate(batchdata.access(0,i), err);
	    err -= batchdata.access(1, i);
	    
	    for(unsigned int j=0;j<err.size();j++){
	      const auto& ej = err[j];
	      //for(unsigned int k=0;k<ej.size();k++)
	      //  error += ninv*ej[k]*math::conj(ej[k]);
	      
	      thread_error += ninv*math::sqrt(ej[0]*math::conj(ej[0])); // ABSOLUTE VALUE
	    }
	  }

#pragma omp critical
	  {
	    error += thread_error;
	  }
	}

	abserror2 = error;
	abserror2[0] = abs(abserror2[0]);
	
	for(unsigned int i=1;i<abserror2.size();i++){
	  abserror2[0] += abs(abserror2[i]);
	  //abserror2[i] = math::blas_real<double>(0.0f);
	  abserror2[i] = 0.0f;
	} 

	bool go_worse = false;
#if 1
	unsigned int r = rng.rand() % 50;
	
	if(errors.size() > 0){
	  double mean_error = 0.0;
	  for(const auto& e : errors)
	    mean_error += e;
	  mean_error /= errors.size();
	  
	  
	  if((r == 0 &&
	      abserror2[0].real() < 1.15*best_error && // was 1.50
	      abserror2[0].real() < 100.0) /*||
					     (abserror2-orig_error)[0] < 1e-5*/)
	    go_worse = true;
	}
#endif

	if(abserror2[0].real() < abserror[0].real() || go_worse){
	  // error becomes smaller => found new better solution
	  lratef *= 2.0; // bigger step length..
	  //abserror = abserror2;
	  break;
	}
	else{ // try shorter step length
	  lratef *= 1/2.0;
	  grad_search_counter++;
	}
	
      }

      abserror = abserror2;

      {
	double e = 0.0;
	whiteice::math::convert(e, abserror[0]);
	errors.push_back(e);

	while(errors.size() > 10)
	  errors.erase(errors.begin());

	if(e < best_error) best_error = e;
      }

      // weights -= sumgrad + regularizer;
      // weights -= lrate * sumgrad + regularizer; // (alpha*weights);
      
      std::cout << counter << " [" << grad_search_counter << "] : " << abserror
		<< " (delta: " << (abserror-orig_error)[0] << ")"
		<< " (lrate: " << lratef << ")" 
		<< std::endl;

      snet.save("inverse_hash_snet.dat");

      //if(lratef < 0.01) lratef = sqrt(lratef);
      //if(lratef < 0.01) lratef = 0.01f;
      //lratef = sqrt(lratef);
      if(lratef < 1.0) lratef = 1.0f;

      error = abserror;
      
      counter++;
    }
    
    std::cout << counter << " : " << abs(error) << std::endl;

    math::vertex< math::superresolution<math::blas_real<double>,
					math::modular<unsigned int> > > params;
    snet.exportdata(params);
    //std::cout << "nn solution weights = " << params << std::endl;
    
  }
  
  
    
  return 0;
}
