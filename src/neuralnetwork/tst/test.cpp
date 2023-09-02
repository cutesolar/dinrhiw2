/*
 * simple unit testcases
 *
 * Tomas Ukkonen
 */

#include "HC.h"

#include "activation_function.h"
#include "odd_sigmoid.h"

#include "nnetwork.h"
#include "lreg_nnetwork.h"
#include "GDALogic.h"

#include "Mixture.h"
#include "EnsembleMeans.h"
#include "KMeans.h"

#include "dataset.h"
#include "dinrhiw_blas.h"

#include "bayesian_nnetwork.h"
#include "HMC.h"
#include "HMC_gaussian.h"
#include "deep_ica_network_priming.h"

#include "pretrain.h"

#include "RBM.h"
#include "GBRBM.h"
#include "HMCGBRBM.h"
#include "PTHMCGBRBM.h"
#include "CRBM.h"
#include "LBFGS_GBRBM.h"
#include "LBFGS_BBRBM.h"
#include "BBRBM.h"

#include "LBFGS_nnetwork.h"
#include "rLBFGS_nnetwork.h"

#include "DBN.h"

#include "VAE.h"
#include "TSNE.h"

#include "globaloptimum.h"

#include "PSO.h"
#include "RBMvarianceerrorfunction.h"

#include "NNGradDescent.h"

#include "RNG.h"

#include "hermite.h"

#include "KMBoosting.h"

#include "rLBFGS_recurrent_nnetwork.h"
#include "SGD_recurrent_nnetwork.h"

#include "LinearKCluster.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <new>
#include <random>
#include <chrono>
#include <thread>

#include <assert.h>
#include <string.h>
#undef __STRICT_ANSI__
#include <float.h>

#include <fenv.h>

#if 0

extern "C" {

  // traps floating point exceptions..
#define _GNU_SOURCE 1
#ifdef __linux__
#include <fenv.h>
  static void __attribute__ ((constructor))
  trapfpe(){
    feenableexcept(FE_INVALID); // |FE_DIVBYZERO); // |FE_OVERFLOW);
  }
#endif
  
}
#endif


using namespace whiteice;


void activation_test();

/* OLD CODE: DISABLED
void neuron_test();
void neuronlayer_test();
void neuronlayer_test2();
void neuralnetwork_test();
*/

void recurrent_nnetwork_test();

void kmboosting_test();

void nnetwork_entropy_test();
void nnetwork_kl_divergence_test();

void nnetwork_test();
void nnetwork_complex_test();

void lreg_nnetwork_test();
void simple_recurrent_nnetwork_test();
void mixture_nnetwork_test();
void ensemble_means_test();

void nnetwork_gradient_test();
void nnetwork_residual_gradient_test();
void nnetwork_gradient_value_test();
void nnetwork_complex_gradient_test();

void nngraddescent_complex_test();
  
void rbm_test();
void simple_rbm_test();

void lbfgs_rbm_test();

void bbrbm_test();

void dbn_test();

void bayesian_nnetwork_test();
void backprop_test(const unsigned int size);
void neuralnetwork_saveload_test();
void neuralnetwork_pso_test();

void hmc_test();

void gda_clustering_test();

void simple_dataset_test();

void simple_vae_test();

void simple_tsne_test();

void simple_global_optimum_test();

void kmeans_test();

void pretrain_test(); // good pretraining, optimization idea test

void compressed_neuralnetwork_test();

void linear_kcluster_test(); // unit tests LinearKCluster class



void createHermiteCurve(std::vector< math::vertex< math::blas_real<double> > >& samples,
			const unsigned int NPOINTS,
			const unsigned int DIMENSION,
			const unsigned int NSAMPLES);

bool saveSamples(const std::string& filename, std::vector< math::vertex< math::blas_real<double> > >& samples);




int main()
{
  unsigned int seed = (unsigned int)time(0);
  // seed = 0x5f54fc68;
  printf("seed = 0x%x\n", seed);
  srand(seed);

  whiteice::logging.setOutputFile("testsuite1.log");
  
  try{

    linear_kcluster_test(); // unit tests LinearKCluster class
    
    return 0;
    
    // nnetwork_entropy_test();

    // nnetwork_kl_divergence_test();

    hmc_test();

    return 0;

    pretrain_test();

    bayesian_nnetwork_test();    
    
    // nnetwork_test();
    
    // simple_vae_test(); // FAILS FOR NOW??

    // nnetwork_gradient_value_test(); // gradient_value() calculation works

    // kmboosting_test();

    // recurrent_nnetwork_test();
    
    // simple_tsne_test(); // FIXME: has bugs

    return 0;

    // nngraddescent_complex_test();

    // nnetwork_complex_test(); // works about correctly

    kmeans_test();

    nnetwork_gradient_test(); // gradient calculation works

    nnetwork_gradient_value_test(); // gradient_value() calculation works
    
    nnetwork_residual_gradient_test(); // gradient calculation test for residual neural network
    
    //nnetwork_complex_gradient_test(); // gradient calculation works now for complex data

    return 0;

    // simple_recurrent_nnetwork_test(); // FIXME doesn't seem to work anymore.
    
    return 0;
	
    
    simple_global_optimum_test(); // DOES NOT WORK WELL

      
    // simple_vae_test();
    
    // simple_rbm_test();
    
    return 0;
    
    bbrbm_test();
    
    // mixture_nnetwork_test();

    

    
    // dbn_test();
#if 0
    lbfgs_rbm_test();

    ensemble_means_test();

    rbm_test();
#endif
    /////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////
    
    // bayesian_nnetwork_test();
    
    // lreg_nnetwork_test();
    
    // hmc_test();

#if 0    
    activation_test();  
    
    gda_clustering_test(); // DO NOT WORK
#endif


#if 0
    simple_dataset_test();
    backprop_test(500);
    
    neuron_test();
    neuronlayer_test();
    neuronlayer_test2();
    neuralnetwork_test();
#endif
    
#if 0
    neuralnetwork_pso_test();
    // neuralnetwork_saveload_test();
    // compressed_neuralnetwork_test();    
#endif
    
    return 0;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
    return -1;
  }    
}


/************************************************************/


class test_exception : public std::exception
{
public:
  
  test_exception() 
  {
    reason = 0;
  }
  
  test_exception(const std::exception& e) 
  {
    reason = 0;
    
    const char* ptr = e.what();
    
    if(ptr)
      reason = (char*)malloc(sizeof(char)*(strlen(ptr) + 1));
    
    if(reason) strcpy(reason, ptr);
  }
  
  
  test_exception(const char* ptr) 
  {
    reason = 0;
    
    if(ptr){
      reason = (char*)malloc(sizeof(char)*(strlen(ptr) + 1));
      
      if(reason) strcpy(reason, ptr);
    }
  }
  
  
  virtual ~test_exception() 
  {
    if(reason) free(reason);
    reason = 0;
  }
  
  virtual const char* what() const throw()
  {
    if(reason == 0) // null terminated value
      return ((const char*)&reason);

    return reason;
  }
  
private:
  
  char* reason;
  
};


/**********************************************************************/

void linear_kcluster_test() // unit tests LinearKCluster class
{
  // tests code with superresolution< blas_complex<> > class
  std::cout << "Linear K-Cluster test" << std::endl;

  LinearKCluster< math::superresolution< math::blas_complex<> > > model(20, 5);

  const unsigned int TIME_LIMIT = 3600; // 60 minutes

  // creates data using random neural network
  std::vector< math::vertex< math::superresolution< math::blas_complex<> > > > xdata, ydata;
  
  nnetwork< math::superresolution< math::blas_complex<> > > nn;
  std::vector<unsigned int> arch;

  arch.push_back(20);
  arch.push_back(10);
  arch.push_back(5);

  nn.setArchitecture(arch);
  nn.randomize();

  for(unsigned int n=0;n<500;n++){
    math::vertex< math::superresolution< math::blas_complex<> > > x, y;
    x.resize(arch[0]);
    y.resize(arch[arch.size()-1]);

    for(unsigned int i=0;i<x.size();i++)
      for(unsigned int j=0;j<x[i].size();j++)
	for(unsigned int k=0;k<x[i][j].size();k++)
	  x[i][j][k] = rng.normal();

    nn.calculate(x, y);

    xdata.push_back(x);
    ydata.push_back(y);
  }

  // normalize data [do not work with superresolution!]
#if 0
  {
    dataset< math::superresolution< math::blas_complex<> > > data;
    data.createCluster("x", xdata[0].size());
    data.createCluster("y", ydata[0].size());

    data.add(0, xdata);
    data.add(1, ydata);

    data.preprocess(0);
    data.preprocess(1);

    xdata.clear();
    ydata.clear();

    data.getData(0, xdata);
    data.getData(1, ydata);
  }
#endif

  //const unsigned int K = 50;
  //std::cout << "Starting optimization with K=" << K << " clusters.." << std::endl;
  
  if(model.startTrain(xdata, ydata) == false){
    std::cout << "ERROR: startTrain() FAILED!" << std::endl;
    return; 
  }

#if 0
  if(model.getNumberOfClusters() != K){
    std::cout << "WARN: number of clusters is not: " << K << std::endl;
  }
#endif

  unsigned long long start_time = (unsigned long long)time(0);
  unsigned int iters_shown = 0;

  while(model.isRunning()){
    sleep(1);
    unsigned long long cur_time = (unsigned long long)time(0);

    if((cur_time-start_time) > TIME_LIMIT && iters_shown >= 10){
      std::cout << "ERROR: solution does not converge in " << TIME_LIMIT << " seconds!" << std::endl;
      break;
    }

    double e = 0;
    unsigned int iters = 0;
    model.getSolutionError(iters, e);
    if(iters > iters_shown){
      std::cout << iters << ": current model error: " << e << std::endl;
      iters_shown = iters;
    }
  }
  
  if(model.stopTrain() == false){
    std::cout << "WARN: stopTrain() FAILED!" << std::endl;
  }
  else std::cout << "stopTrain() was OK" << std::endl;
  
  double e = 0.0;
  unsigned int iters = 0;
  model.getSolutionError(iters, e);
  std::cout << "Final model error: " << e << std::endl;

  if(model.save("linear-k-cluster-model.dat") == false){
    std::cout << "ERROR: save() FAILED!" << std::endl;
    return;
  }

  if(model.load("linear-k-cluster-model.dat") == false){
    std::cout << "ERROR: load() FAILED!" << std::endl;
  }

  double ee = 0.0;
  model.getSolutionError(iters, ee);

  if(whiteice::math::abs(e - ee) > 1e-2){
    std::cout << "ERROR: getSolutionError() FAILS AFTER LOAD!" << std::endl;
    std::cout << "orig e = " << e << std::endl;
    std::cout << "model e = " << ee << std::endl;
    return;
  }

#if 0
  if(model.getNumberOfClusters() != K){
    std::cout << "ERROR: getNumberOfClusters() FAILS AFTER LOAD!" << std::endl;
    return;
  }
#endif

  {
    double err = 0.0;

    for(unsigned int i=0;i<xdata.size();i++){
      math::vertex< math::superresolution< math::blas_complex<> > > x, y;
      
      x = xdata[i];
      if(model.predict(x, y) == false){
	std::cout << "ERROR: predict() FAILS!" << std::endl;
	return;
      }

      auto delta = y - ydata[i];
      auto n = delta.norm();


      // std::cout << "norm = " << n << std::endl; 
      
      double nd = 0.0f;
      whiteice::math::convert(nd, n);
      err += nd;
    }

    err /= xdata.size();
    err /= ydata[0].size();

    std::cout << "Experimental model error: " << err << std::endl;
    model.getSolutionError(iters, e);
    std::cout << "Model reported error: " << e << std::endl; 
  }

  std::cout << "LinearKCluster class UNIT tests DONE." << std::endl;
}




/************************************************************/

void pretrain_test() // good pretraining, optimization idea test
{
  std::cout << "smart neural network pretraining test." << std::endl;


  // generates test data, 10-layer 10->10 dimensional neural nnetwork
  // (residual rectifier neural network)

  const unsigned int LAYERS = 50; // was: 3, 10, 100
  const unsigned int INPUT_DIM = 4;
  const unsigned int HIDDEN_DIM = 50; // was: 100
  const unsigned int OUTPUT_DIM = INPUT_DIM;
  
  whiteice::nnetwork< math::blas_real<double> > nnet;
  whiteice::RNG< math::blas_real<double> > rng;

  std::vector<unsigned int> layers;

  layers.push_back(INPUT_DIM);
  
  for(unsigned int i=0;i<(LAYERS-1);i++) 
    layers.push_back(HIDDEN_DIM);
  
  layers.push_back(OUTPUT_DIM);

  nnet.setArchitecture(layers);
  nnet.setResidual(false);
  nnet.setNonlinearity(whiteice::nnetwork< math::blas_real<double> >::rectifier);

  nnet.randomize(2, 0.01);


  whiteice::nnetwork< math::blas_real<double> > gennet; // for generating dataset example problem

  layers.clear();
  layers.push_back(INPUT_DIM);
  layers.push_back(HIDDEN_DIM);
  layers.push_back(HIDDEN_DIM);
  layers.push_back(OUTPUT_DIM);

  gennet.setArchitecture(layers);
  gennet.setNonlinearity(whiteice::nnetwork< math::blas_real<double> >::rectifier);

  gennet.randomize();

  
  whiteice::dataset< math::blas_real<double> > data;
  
  data.createCluster("input", INPUT_DIM);
  data.createCluster("output", OUTPUT_DIM);

  for(unsigned int i=0;i<2000;i++){ // only 2000 elements for faster results..
    math::vertex< math::blas_real<double> > datum(INPUT_DIM), output;
    datum.resize(INPUT_DIM);
    output.resize(OUTPUT_DIM);
    rng.normal(datum);

    math::blas_real<double> sigma = 4.0;
    math::blas_real<double> f = 10.0, a = 1.10, w = 10.0, one = 1.0;

    datum = sigma*datum;

    data.add(0, datum);

    auto& x = datum;
    auto& y = output;

    
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
    
    // gennet.calculate(datum, output);

    data.add(1, output);
  }

  data.preprocess(0);
  data.preprocess(1);

  printf("DATA GENERATED %d\n", data.size(0));
  
  nnet.setBatchNorm(false);

  math::vertex< math::blas_real<double> > initial_weights;

  nnet.exportdata(initial_weights);

#if 0
  // trains similar network using pretrain and reports error in 20 first iterations

  PretrainNN< math::blas_real<double> > pretrainer;
  const unsigned int MAXITERS = 2000;

  if(pretrainer.startTrain(nnet, data, MAXITERS) == false){
    printf("ERROR: starting pretrainer FAILED.\n");
    return;
  }

  while(pretrainer.isRunning()){
    sleep(1);
    math::blas_real<double> error;
    unsigned int iters = 0;

    pretrainer.getStatistics(iters, error);

    std::cout << "Pretrainer " << iters << "/" << MAXITERS << " : " << error << std::endl;
  }

  pretrainer.stopTrain();

  if(pretrainer.getResults(nnet) == false){
    printf("ERROR: getting results from pretrainer FAILED.\n");
    return;
  }
#endif
  

#if 0
  
  std::vector< math::vertex< math::blas_real<double> > > vdata;

  data.getData(0, vdata);
  
  
  const unsigned int ITERS = 0; // was: 100,20, 5000 

  auto initial_mse = nnet.mse(data);
  auto best_mse = math::blas_real<double>(1e5);
  math::vertex< math::blas_real<double> > weights, w0, w1;
  nnet.exportdata(weights);

  math::blas_real<double> adaptive_step_length = (1e-5f);


  // convergence detection
  std::list< math::blas_real<double> > errors;
  const unsigned int ERROR_HISTORY_SIZE = 30;
  

  
  //for(unsigned int k=0;k<ITERS;k++){
  unsigned int k = 0;
  while(k<ITERS){

    nnet.setBatchNorm(false);
    // nnet.calculateBatchNorm(vdata);

#if 1
    if((whiteice::rng.rand() % 100) == 0){
      printf("LARGE ADAPTIVE STEPLENGTH\n");
      adaptive_step_length = whiteice::math::sqrt(whiteice::math::sqrt(adaptive_step_length));
      if(adaptive_step_length.c[0] >= 0.45f)
	adaptive_step_length = 0.45f;
    }
#endif


#if 0
    if((whiteice::rng.rand() % 200) == 0)
      whiten1d_nnetwork(nnet, data);
#endif

    nnet.exportdata(w1);
    
    if(whiteice::pretrain_nnetwork_matrix_factorization(nnet, data,
							math::blas_real<double>(0.5f)*adaptive_step_length) == false)
    {
      printf("pretrain_nnetwork() FAILED!\n");
      return; 
    }

    nnet.exportdata(w0);
#if 1
    for(unsigned int i=0;i<w0.size();i++){
      
      if(w0[i].c[0] < -0.75f) w0[i].c[0] = -0.75f;
      if(w0[i].c[0] > +0.75f) w0[i].c[0] = +0.75f;

      // printf("w0[%d] = %f\n", i, w0[i].c[0]);
    }
#endif
    
    
    nnet.importdata(w0);

    auto smaller_mse = nnet.mse(data);

    nnet.importdata(w1);

    if(whiteice::pretrain_nnetwork_matrix_factorization(nnet, data,
							math::blas_real<double>(2.0f)*adaptive_step_length) == false)
    {
      printf("pretrain_nnetwork() FAILED!\n");
      return; 
    }

    nnet.exportdata(w1);

#if 1
    for(unsigned int i=0;i<w1.size();i++){
      if(w1[i].c[0] < -0.75f) w1[i].c[0] = -0.75f;
      if(w1[i].c[0] > +0.75f) w1[i].c[0] = +0.75f;

      // printf("w1[%d] = %f\n", i, w1[i].c[0]);
    }
#endif
    
    nnet.importdata(w1);

    auto larger_mse = nnet.mse(data);
    auto mse = larger_mse;

    if(smaller_mse < larger_mse){
      adaptive_step_length *= (0.5f);
      if(adaptive_step_length < 1e-10)
	adaptive_step_length = 1e-10;
      mse = smaller_mse;

      nnet.importdata(w0);
    }
    else{
      adaptive_step_length *= (2.0f);
      if(adaptive_step_length.c[0] >= 0.45f)
	adaptive_step_length = 0.45f;
    }


    {
      errors.push_back(mse);
      
      while(errors.size() > ERROR_HISTORY_SIZE)
	errors.pop_front();

      if(errors.size() >= ERROR_HISTORY_SIZE){

	auto iter = errors.begin();
	
	auto mean = *iter;
	auto stdev  = (*iter)*(*iter);

	iter++;

	for(unsigned int i=1;i<errors.size();i++,iter++){
	  mean += *iter;
	  stdev += (*iter)*(*iter);
	};

	mean /= errors.size();
	stdev /= errors.size();

	stdev = stdev - mean*mean;
	stdev = whiteice::math::sqrt(whiteice::math::abs(stdev));

	auto convergence = (stdev/(whiteice::math::blas_real<double>(1e-5) + mean));
	
	std::cout << "convergence = " << convergence << std::endl;

	if(convergence < 0.1f){
	  printf("LARGE ADAPTIVE STEPLENGTH\n");
	  adaptive_step_length = whiteice::math::sqrt(whiteice::math::sqrt(adaptive_step_length));
	  if(adaptive_step_length.c[0] >= 0.45f)
	    adaptive_step_length = 0.45f;

	  errors.clear();
	}
	
      }
    }
    
    
    if(best_mse > mse){
      best_mse = mse;
      nnet.exportdata(weights);
    }

    // adaptive_step_length = 1e-5;

    //if(mse > 10000.0f){
    //  nnet.randomize(2, 0.01);
    //  printf("RANDOMIZE NEURAL NETWORK\n");
    //}

    printf("%d/%d: Neural network MSE for this problem: %f %f%% %f %f%% (%e)\n",
	   k, ITERS, mse.c[0],
	   (mse/initial_mse).c[0]*100.0f,
	   best_mse.c[0],
	   (best_mse/initial_mse).c[0]*100.0f,
	   adaptive_step_length.c[0]);
    
    
    

    k++;
  }


  nnet.importdata(weights);

#endif
  
  nnet.setBatchNorm(false);
  // nnet.calculateBatchNorm(vdata);
  
  nnet.setNonlinearity(whiteice::nnetwork< math::blas_real<double> >::rectifier);
  nnet.setResidual(true);
  
  auto mse = nnet.mse(data);
  printf("Neural network MSE for this problem: %f (per dimension)\n", mse.c[0]);


  {
    // trains neural network using SGD
    
    whiteice::math::NNGradDescent<  math::blas_real<double> > grad;
    
    grad.setUseMinibatch(true);
    grad.setOverfit(true);

    grad.setMatrixFactorizationPretrainer(true);
    
    grad.startOptimize(data, nnet, 3, 2500);
    
    while(grad.isRunning()){
      math::blas_real<double> error = 0.0f;
      unsigned int iters = 0;
      
      if(grad.getSolutionStatistics(error, iters)){
	std::cout << "MSE ERROR OF GRADIENT DESCENT: " << error
		  << " ITERS: " << iters << "/5000" << std::endl;
      }
      
      sleep(1);
    }
    
    grad.stopComputation();
  }


  //////////////////////////////////////////////////////////////////////

  printf("Optimizing neural network without pretrainer..\n");
  
  nnet.importdata(initial_weights);

  
  {
    // trains neural network using SGD
    
    whiteice::math::NNGradDescent<  math::blas_real<double> > grad;
    
    grad.setUseMinibatch(true);
    grad.setOverfit(true);
    
    grad.startOptimize(data, nnet, 3, 2500);
    
    while(grad.isRunning()){
      math::blas_real<double> error = 0.0f;
      unsigned int iters = 0;
      
      if(grad.getSolutionStatistics(error, iters)){
	std::cout << "MSE ERROR OF GRADIENT DESCENT: " << error
		  << " ITERS: " << iters << "/5000" << std::endl;
      }
      
      sleep(1);
    }
    
    grad.stopComputation();
  }
  

  printf("pretrain_nnetwork() tests OK (?)\n");
}


/************************************************************/


void nnetwork_kl_divergence_test()
{
  std::cout << "nnetwork KL divergence minimization test" << std::endl;

  // NOTE: it's better to use KL-divergence than REVERSE KL-divergence which matches
  // output to largest MODE of the target distribution [no multiple choices]
  
  // generates training data
  
  whiteice::nnetwork<> net;
  
  whiteice::dataset<> data;
  const unsigned int INPUT_DIM = 5;
  const unsigned int OUTPUT_DIM = INPUT_DIM;
  
  data.createCluster("input", INPUT_DIM);
  data.createCluster("output", OUTPUT_DIM);

  for(unsigned int i=0;i<200;i++){ // only 200 elements for faster results..
    math::vertex<> datum(INPUT_DIM);
    rng.normal(datum);
    data.add(0, datum);

    // gives 50% probability to first and second largest elements
    unsigned int selected1_index = 0, selected2_index = 1;
    whiteice::math::blas_real<float> selected1_value = datum[0];
    whiteice::math::blas_real<float> selected2_value = datum[1];

    if(selected1_value < selected2_value){
      std::swap(selected1_index, selected2_index);
      std::swap(selected1_value, selected2_value);
    }

    for(unsigned int a=2;a<datum.size();a++){
      if(datum[a] > selected1_value){
	selected2_value = selected1_value;
	selected2_index = selected1_index;

	selected1_value = datum[a];
	selected1_index = a;
      }
      else if(datum[a] > selected2_value){
	selected2_value = datum[a];
	selected2_index = a;
      }
    }

    datum.zero();
    
    datum[selected1_index] = 0.50f;
    datum[selected2_index] = 0.50f;

    data.add(1, datum);
  }

  const unsigned int NUMLAYERS = 5; // 5 layer neural network..
  std::vector<unsigned int> arch;
  arch.push_back(INPUT_DIM);
  for(unsigned int l=0;l<(NUMLAYERS-1);l++)
    arch.push_back(100);
  arch.push_back(OUTPUT_DIM);

  net.setArchitecture(arch);
  net.randomize();

  const whiteice::math::blas_real<float> min_divergence = 0.0f;

  unsigned int no_improvements_counter = 0;

  linear_ETA<> eta;
  eta.start(0, 1000);

  for(unsigned int counter=0;counter<1000 && no_improvements_counter < 10;counter++){
    eta.update(counter);

    // calculates average entropy of output and reports it
    math::vertex<> input, output, correct;
    whiteice::math::blas_real<float> H = 0.0f;

    for(unsigned int i=0;i<data.size(0);i++){
      input = data.access(0, i);
      correct = data.access(1, i);

      net.calculate(input, output);
      H += net.kl_divergence(output, 0, output.size(), correct);
    }

    H /= data.size(0);

    std::cout << "ITER " << counter
	      <<  ". Current output KL-divergence = " << H
	      << " (min KL divergence = " << min_divergence << ")"
	      << " [ETA: " << eta.estimate()/60.0f << " minute(s)]"
	      << std::endl;

    // calculates gradient
    std::vector< math::vertex<> > bpdata;
    math::vertex<> grad, tmp_grad;
    grad.resize(net.gradient_size());
    grad.zero();

    for(unsigned int i=0;i<data.size(0);i++){
      input = data.access(0, i);
      correct = data.access(1, i);

      net.calculate(input, output, bpdata);

      net.kl_divergence_gradient(output, 0 , output.size(), correct, bpdata, tmp_grad);

      grad += tmp_grad;
    }

    grad /= data.size(0);

    // updates weights
    whiteice::math::blas_real<float> LRATE = 1.00f;

    whiteice::math::vertex<> weights, w0;
    
    net.exportdata(w0);
    
    while(LRATE > 1e-10){
      weights = w0;
      
      weights -= LRATE*grad; // minimizes KL-divergence
      
      net.importdata(weights);

      {
	whiteice::math::blas_real<float> KL = 0.0f;
	
	for(unsigned int i=0;i<data.size(0);i++){
	  input = data.access(0, i);
	  correct = data.access(1, i);
	  
	  net.calculate(input, output);
	  KL += net.kl_divergence(output, 0, output.size(), correct);
	}
	
	KL /= data.size(0);

	if(KL < H){
	  no_improvements_counter = 0;
	  break;
	}
	else{
	  LRATE /= 2.0f;
	}
      }
      
    }

    if(LRATE <= 1e-10)
      no_improvements_counter++;

  }

  // print 5 first examples and their solutions vs. returned values
  for(unsigned int i=0;i<5;i++){
    math::vertex<> input, output, correct;
    input = data.access(0, i);
    correct = data.access(1, i);

    net.calculate(input, output);
    net.softmax_output(output, 0, output.size());

    std::cout << "CASE " << i << " CORRECT : " << correct << std::endl;
    std::cout << "CASE " << i << " PREDICT : " << output << std::endl;
    
  }
  
}

/************************************************************/

void nnetwork_entropy_test()
{
  std::cout << "nnetwork entropy maximization test" << std::endl;

  // generates training data

  whiteice::dataset<> data;
  const unsigned int INPUT_DIM = 5+(rng.rand() % 10);
  const unsigned int OUTPUT_DIM = 10+(rng.rand() % 5);
  
  data.createCluster("input", INPUT_DIM);

  for(unsigned int i=0;i<1000;i++){
    math::vertex<> datum(INPUT_DIM);
    rng.normal(datum);
    data.add(0, datum);
  }

  whiteice::nnetwork<> net;
  std::vector<unsigned int> arch;
  arch.push_back(INPUT_DIM);
  for(unsigned int l=0;l<(2-1);l++) // minimal two layer neural network..
    arch.push_back((INPUT_DIM+OUTPUT_DIM)/2);
  arch.push_back(OUTPUT_DIM);

  net.setArchitecture(arch);
  net.randomize();

  const whiteice::math::blas_real<float> max_entropy = math::log((float)OUTPUT_DIM);

  for(unsigned int counter=0;counter<1000;counter++){

    // calculates average entropy of output and reports it
    math::vertex<> input, output;
    whiteice::math::blas_real<float> H = 0.0f;

    for(unsigned int i=0;i<data.size(0);i++){
      input = data.access(0, i);

      net.calculate(input, output);
      H += net.entropy(output, 0, output.size());
    }

    H /= data.size(0);

    std::cout << "ITER " << counter
	      <<  ". Current output entropy H = " << H
	      << " (max entropy = " << max_entropy << ")"
	      << std::endl;

    // calculates gradient
    std::vector< math::vertex<> > bpdata;
    math::vertex<> grad, tmp_grad;
    grad.resize(net.gradient_size());
    grad.zero();

    for(unsigned int i=0;i<data.size(0);i++){
      input = data.access(0, i);

      net.calculate(input, output, bpdata);

      net.entropy_gradient(output, 0 , output.size(), bpdata, tmp_grad);

      grad += tmp_grad;
    }

    grad /= data.size(0);

    // updates weights
    const whiteice::math::blas_real<float> LRATE = 0.10f;

    whiteice::math::vertex<> weights;

    net.exportdata(weights);

    weights += LRATE*grad;

    net.importdata(weights);

  }
  
}


/************************************************************/

void kmeans_test()
{
  std::cout << "K-Means clustering test" << std::endl;

  // three cluster test
  {
    std::vector< math::vertex<> > data;
    whiteice::RNG<> rng;

    const unsigned int DIM = 2; // + rand() % 10;
    math::vertex<> mean[3], x;
    x.resize(DIM);

    
    for(unsigned int k=0;k<3;k++){
      mean[k].resize(DIM);
      rng.normal(mean[k]);
      mean[k] *= 10.0f;
      
      for(unsigned int i=0;i<100;i++){
	rng.normal(x);
	x += mean[k];
	data.push_back(x);
      }
      
    }

    whiteice::KMeans<> kmeans;

    assert(kmeans.startTrain(3, data));

    while(kmeans.isRunning()){
      std::cout << "K-Means clustering error: " << kmeans.getSolutionError() << std::endl;
      sleep(1);
    }

    kmeans.stopTrain();
    std::cout << "K-Means clustering error: " << kmeans.getSolutionError() << std::endl;

    std::cout << "Found means: " << std::endl;
    std::cout << "K = " << kmeans.size() << std::endl;
    for(unsigned int i=0;i<kmeans.size();i++){
      std::cout << kmeans[i] << std::endl;
    }

    std::cout << "Data means: " << std::endl;
    for(unsigned int i=0;i<kmeans.size();i++){
      std::cout << mean[i] << std::endl;
    }

    // save data for plotting clustering results
    std::ofstream outfile[3];
    outfile[0].open("cluster1.txt");
    outfile[1].open("cluster2.txt");
    outfile[2].open("cluster3.txt");

    for(unsigned int i=0;i<data.size();i++){
      const unsigned int index = kmeans.getClusterIndex(data[i]);

      if(index < 3){
	for(unsigned int n=0;n<data[i].size();n++)
	  outfile[index] << data[i][n] << " ";
	outfile[index] << std::endl;
      }
      
    }

    outfile[0].close();
    outfile[1].close();
    outfile[2].close();
  }

  // save()&load() TEST
  {
    std::cout << "K-Means save()&load() test." << std::endl;

    whiteice::KMeans<> kmeans;
    whiteice::KMeans<> kmeans2;

    // three cluster test
    std::vector< math::vertex<> > data;
    whiteice::RNG<> rng;

    const unsigned int DIM = 2; // + rand() % 10;
    math::vertex<> mean[3], x;
    x.resize(DIM);

    
    for(unsigned int k=0;k<3;k++){
      mean[k].resize(DIM);
      rng.normal(mean[k]);
      mean[k] *= 10.0f;
      
      for(unsigned int i=0;i<100;i++){
	rng.normal(x);
	x += mean[k];
	data.push_back(x);
      }

      // std::cout << "mean(" << k << ") = " << mean[k] << std::endl;
    }

    kmeans.learn(3, data);
    
    /*
    assert(kmeans.startTrain(3, data));

    while(kmeans.isRunning()){
      sleep(1);
    }

    kmeans.stopTrain();
    */

    assert(kmeans.save("kmeans.dat") == true);
    assert(kmeans2.load("kmeans.dat") == true);

    if(kmeans.size() != kmeans2.size()){
      std::cout << "ERROR: size of clusters mismatch after load()" << std::endl;
      return;
    }

    for(unsigned int i=0;i<kmeans.size();i++){
      auto delta = kmeans[i] - kmeans2[i];

      if(delta.norm() > 0.0){
	std::cout << "ERROR: cluster mean mismatch after load()" << std::endl;
	return;
      }
    }

    std::cout << "K-Means save()&load() test PASSED." << std::endl;
  }
  
}

/************************************************************/

void recurrent_nnetwork_test()
{
  std::cout << "Full recurrent neural network training test." << std::endl;

  whiteice::dataset<> data;

  const unsigned int INPUT_DIM = 2 + (rng.rand() % 10); // 2-11
  const unsigned int OUTPUT_DIM = 2 + (rng.rand() % 10); // 2-11
  const unsigned int RECURRENT_DIM = 1 + (rng.rand() % 5); // 1-5

  std::cout << "Input dim: " << INPUT_DIM
	    << ". Output dim: " << OUTPUT_DIM
	    << ". Recurrent dim: " << RECURRENT_DIM << "." << std::endl;

  data.createCluster("input", INPUT_DIM);
  data.createCluster("output", OUTPUT_DIM);
  data.createCluster("episodes info", 2);

  std::vector<unsigned int> arch, arch2;

  arch.push_back(INPUT_DIM);
  arch.push_back(10);
  arch.push_back(10);
  arch.push_back(OUTPUT_DIM);

  arch2.push_back(INPUT_DIM+RECURRENT_DIM);
  arch2.push_back(10);
  arch2.push_back(10);
  arch2.push_back(OUTPUT_DIM+RECURRENT_DIM);

  whiteice::nnetwork<> gen(arch), trained_nnetwork(arch2);

  //gen.setNonlinearity(gen.getLayers()-1, nnetwork<>::tanh);
  //trained_nnetwork.setNonlinearity(trained_nnetwork.getLayers()-1, nnetwork<>::tanh);
  
  gen.randomize();
  trained_nnetwork.randomize();

  math::vertex<> in, out, range;
  in.resize(arch[0]);
  range.resize(2);

  unsigned int counter=0;

  // generates training episodes
  // [no recurrency/state in training samples so even regular nnetwork can learn these]
  for(unsigned int episode=0;episode<50;episode++){
    const unsigned int START = counter;

    for(unsigned int i=0;i<20;i++,counter++){
      rng.normal(in);
      gen.calculate(in, out);

      data.add(0, in);
      data.add(1, out);
    }

    const unsigned int END = counter;

    range[0] = START;
    range[1] = END;

    data.add(2, range);
  }

  //whiteice::rLBFGS_recurrent_nnetwork<> trainer(trained_nnetwork, data);
  // // wolfe conditions are required to guaranteed to get good results
  // trainer.setUseWolfeConditions(true);
  // 
  // trainer.setGradientOnly(false);

  whiteice::SGD_recurrent_nnetwork<> trainer(trained_nnetwork, data);
  trainer.setKeepWorse(false);

  // const float SGD_LRATE = 1e-5;
  // const unsigned int MAX_NOPROGRESS_ITERATIONS = 50;
  // 
  // trainer.setSGD(1e-5,

  const unsigned int MAX_NO_IMPROVE_ITERS = 30;
  float lrate = 1e-2;
  
  whiteice::math::vertex<> x0;
  trained_nnetwork.exportdata(x0);

  trainer.minimize(x0, lrate, 0, MAX_NO_IMPROVE_ITERS);

  int solution_iteration_seen = -1;

  while(trainer.isRunning()){
    whiteice::math::vertex<> x;
    whiteice::math::blas_real<float> error;
    unsigned int iterations;

    if(trainer.getSolution(x, error, iterations)){
      if(((int)iterations) > solution_iteration_seen){
	solution_iteration_seen = (int)iterations;

	std::cout << "ITER " << iterations << ". Error = " << error << std::endl;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(333)); // 333ms sleep 
  }

  trainer.stopComputation();

  std::cout << "Trainer stopped." << std::endl;
  
}


/************************************************************/

void kmboosting_test()
{
  // THIS DOES NOT WORK: refactor KMBoosting to do gradient boosting instead.
  
  std::cout << "KMBoosting test (neural network boosting test)" << std::endl;

  whiteice::dataset<> data;
  const unsigned int M = 10;
  std::vector<unsigned int> arch;

  arch.push_back(10);
  arch.push_back(10);
  arch.push_back(10);
  arch.push_back(1);

  whiteice::KMBoosting<> nnboosting(M, arch);

  data.createCluster("input", arch[0]);
  data.createCluster("output", arch[arch.size()-1]);

  // generates data from a neural network
  whiteice::nnetwork<> gen(arch);
  gen.randomize();

  math::vertex<> in;
  math::vertex<> out;
  in.resize(arch[0]);

  
  for(unsigned int i=0;i<10000;i++){
    rng.normal(in);

    gen.calculate(in, out);
    
    data.add(0, in);
    data.add(1, out);
  }

  
  if(nnboosting.startOptimize(data) == false){
    printf("ERROR: KMBoosting::startOptimize() FAILED.\n");
    exit(-1);
  }

  while(nnboosting.isRunning()){
    sleep(1);
    // printf("Computing boosting results\n");
  }

  if(nnboosting.hasModel() == false){
    printf("ERROR: KMBoosting stopped without model. (FAILURE).\n");
    exit(-1);
  }
}

/************************************************************/

void simple_tsne_test()
{
  std::cout << "t-SNE dimension reducer test" << std::endl;

  // testcase 1: test that computations happen correctly
  {
    std::cout << "TESTCASE 1" << std::endl;
    std::cout << std::flush;

    // we generate HIGHDIM dimensional gaussian balls N(mean,I) in random locations
    // in hypercube [-10,10]^HIGHDIM. mean = random([-10,10]^HIGHDIM)

    const unsigned int HIGHDIM = 20;
    const unsigned int CLUSTERS = 10; // number of clusters
    const unsigned int N = 10; // number of samples per cluster (total of 1000 datapoints)

    whiteice::RNG<> rng;

    // generate training data
    std::vector< whiteice::math::vertex<> > data;
    whiteice::math::vertex<> x;
    whiteice::math::vertex<> mean;

    mean.resize(HIGHDIM);
    x.resize(HIGHDIM);

    for(unsigned int c=0;c<CLUSTERS;c++){
      rng.uniform(mean);
      for(unsigned int i=0;i<HIGHDIM;i++){
	whiteice::math::blas_real<float> v1 = 20.0f;
	whiteice::math::blas_real<float> v2 = 10.0f;
	mean[i] = v1*mean[i] - v2;
      }

      for(unsigned int n=0;n<N;n++){
	rng.normal(x);
	x += mean;
	data.push_back(x);
      }
    }

    // calculate dimension reduction
    whiteice::TSNE<> tsne(false);
    std::vector< whiteice::math::vertex<> > ydata;

    if(tsne.calculate(data, 2, ydata, true) == false){
      printf("ERROR: calculating t-SNE dimension reduction FAILED.\n");
    }
    else{
      printf("GOOD: t-SNE computation proceeded without errors.\n");
    }

    fflush(stdout);
  }
}

/*********************************************************************************/

void nngraddescent_complex_test()
{
  std::cout << "COMPLEX value NNGradDescent<> optimizer tests." << std::endl;

  nnetwork< math::blas_complex<double> >* nn;
  dataset< math::blas_complex<double> > data;
  std::vector<unsigned int> arch;

  for(unsigned int i=0;i<4;i++) // 3 layer neural network
    arch.push_back(rand()%10 + 10);

  //arch[0] = 2;
  //arch[arch.size()-1] = 2;

  printf("Neural network architecture: ");
  for(unsigned int i=0;i<arch.size();i++){
    if(i == 0) printf("%d", arch[i]);
    else printf("-%d", arch[i]);
  }
  printf("\n");

  // pureLinear optimization works..
  nn = new nnetwork< math::blas_complex<double> >
    (arch, nnetwork< math::blas_complex<double> >::rectifier);

  // set last layer to be linear layer
  nn->setNonlinearity(nn->getLayers()-1, nnetwork< math::blas_complex<double> >::pureLinear);
  
  nn->randomize();
  

  {
    std::vector< math::vertex< math::blas_complex<double> > > inputs;
    std::vector< math::vertex< math::blas_complex<double> > > outputs;

    RNG< math::blas_complex<double> > rng;

    math::vertex< math::blas_complex<double> > in, out;
    in.resize(arch[0]);
    out.resize(arch[arch.size()-1]);

    math::matrix< math::blas_complex<double> > A;
    A.resize(arch[arch.size()-1], arch[0]);
    rng.normal(A); // complex valued N(0,I) values
    // A.abs(); // takes absolute value (real valued matrix)
   
    for(unsigned int n=0;n<1000;n++){
      rng.normal(in);

      // nonlinear test function (should break Cauchy-Riemann conditions)
      for(unsigned int i=0;i<in.size();i++){
	in[i] = whiteice::math::sin(imag(in[i]))*real(in[i]);
      }
      
      out = A*in;

      inputs.push_back(in);
      outputs.push_back(out);
    }

    data.createCluster("input", arch[0]);
    data.createCluster("output", arch[arch.size()-1]);
    
    data.add(data.getCluster("input"), inputs);
    data.add(data.getCluster("output"), outputs);

    data.preprocess(0);
    data.preprocess(1);
  }

  printf("Starting neural network optimizer (non-linear problem).\n");
  fflush(stdout);

  // whiteice::logging.setPrintOutput(true); // for enabling internal logging

  // MNE and dropout heuristics don't work well with complex valued data
  // 1. MNE gradient maybe needed to be computed differently with complex values
  // 2. Dropout may modify network so that it doesn't satisfy Cauchy-Riemann conditions
  //    well anymore meaning that gradient calculation don't work..
  math::NNGradDescent< math::blas_complex<double> > grad;
  
  const unsigned int MAXITERS = 10000;
  const unsigned int THREADS = 2;
  const bool dropout = false; 
  
  grad.setUseMinibatch(false);
  grad.setOverfit(false);
  grad.setMNE(false); // MNE doesn't seem to work well with complex number (is gradient correct)
  grad.setRegularizer(0.01f); // set regularizer
  // grad.setRegularizer( 0.01*((double)nn->output_size())/nn->gradient_size() );

  linear_ETA<double> eta;
  unsigned int N = 0;
  
  eta.start(0.0, MAXITERS);
  
  
  if(grad.startOptimize(data, *nn, THREADS, MAXITERS, dropout, true) == false){
    printf("ERROR: NNGradDescent::startOptimize() failed.\n");
    delete nn;
    return;
  }

  printf("NNGradDescent Optimizer started.\n");

  while(grad.hasConverged() == false){
    math::blas_complex<double> error;

    if(grad.getSolutionStatistics(error, N) == false){
      printf("ERROR: NNGradDescent::getSolutionStatistics() failed.\n");
      delete nn;
      return;
    }
    else{
      eta.update(N);
      
      double errorf = 0.0;
      convert(errorf,error);
      
      printf("%d/%d NN model error: %f. [ETA %f hours] (%f minutes)\n",
	     N, MAXITERS, errorf, eta.estimate()/3600.0, eta.estimate()/60.0);
      fflush(stdout);
    }

    sleep(1);
  }

  printf("Optimization stopped.\n");
  
}


/*********************************************************************************/

void simple_global_optimum_test()
{
  std::cout << "Global Optimum pretraining tests." << std::endl;

  // 0. test case: create random neural network and generate 10.000 training samples.
  //    test how well neural network learns with and without global optimum pretraining.
  //

  {
    // not implemented (random neural network works better without global optimum prelearning)
  }

  // 2. test case: create XOR-like neural network and generate 10.000 training samples
  //    test how well neural network learns with and without global optimum pretraining
  //
  if(0){
    std::cout << "2. Test learning XOR-like neural network." << std::endl;

    const unsigned int N = 10000;

    whiteice::nnetwork<> rand_net;
    std::vector<unsigned int> arch;
    arch.push_back(10);
    //arch.push_back(50);
    //arch.push_back(50);
    arch.push_back(50);    
    arch.push_back(2);
    
    // rand_net.setArchitecture(arch, whiteice::nnetwork<>::tanh); // sigmoid
    // rand_net.setArchitecture(arch, whiteice::nnetwork<>::rectifier);
    rand_net.setArchitecture(arch, whiteice::nnetwork<>::halfLinear);
			     
    rand_net.randomize(0);

    std::vector< math::vertex<> > input;
    std::vector< math::vertex<> > output;

    whiteice::RNG<> rng;
    math::vertex<> v, w;
    v.resize(rand_net.input_size());
    w.resize(rand_net.output_size());

    std::vector< std::vector<unsigned int> > pairs;
    for(unsigned int i=0;i<rand_net.output_size();i++){
      std::vector<unsigned int> p;
      
      unsigned int pN = 1 + (rng.rand() % 3);
      
      for(unsigned int pi=0;pi<pN;pi++){
	int p1 = rng.rand() % rand_net.input_size();
	p.push_back(p1);
      }
      
      pairs.push_back(p);
    }

    for(unsigned int n=0;n<N;n++){
      rng.normal(v);

      //rand_net.calculate(v, w);
      for(unsigned int i=0;i<w.size();i++){
	math::blas_real<float> value = (1.0f);
	
	// is this XOR-like non-linearity
	for(unsigned int pi=0;pi<pairs[i].size();pi++)
	  value *= v[ pairs[i][pi] ];
	
	w[i] = whiteice::math::pow(whiteice::math::abs(value), (1.0f)/((float)pairs[i].size()));
	if(value < (0.0f)) w[i] = -w[i];
      }

      
      input.push_back(v);
      output.push_back(w);
    }


    
    float final_error1 = 0.0f, final_error2 = 0.0f;

    whiteice::dataset<> train_data;
    
    train_data.createCluster("input", rand_net.input_size());
    train_data.createCluster("output", rand_net.output_size());
    
    train_data.add(0, input);
    train_data.add(1, output);
    
    train_data.preprocess(0, dataset<>::dnMeanVarianceNormalization);
    train_data.preprocess(1, dataset<>::dnMeanVarianceNormalization);
    

    // learn without global optimum pretraining
    {
      whiteice::nnetwork<> net(rand_net);

      // initializes weights using data
      //net.presetWeightsFromData(train_data);
      net.randomize();

      whiteice::math::NNGradDescent<> grad;
      grad.setUseMinibatch(true);
      grad.startOptimize(train_data, net, 1);

      while(grad.hasConverged(0.01) == false){
	sleep(1);
	whiteice::math::blas_real<float> error = 0.0f;
	unsigned int Nconverged = 0;
	
	if(grad.getSolutionStatistics(error, Nconverged) == false){
	  printf("ERROR: grad.getSolutionStatistics() failed.\n");
	  return;
	}
	else{
	  float errorf = 0.0f;
	  whiteice::math::convert(errorf, error);
	  printf("Optimizing error %d: %f.\n", Nconverged, errorf);
	}
      }

      grad.stopComputation();

      {
	nnetwork<> nn;
	whiteice::math::blas_real<float> error;
	unsigned int Nconverged;
	
	if(grad.getSolution(nn, error, Nconverged) == false){
	  printf("ERROR: grad.getSolution() failed.\n");
	  return;
	}
	
	net = nn;

	float errorf = 0.0f;
	whiteice::math::convert(errorf, error);

	printf("Optimization converged. Error: %f. STOP.\n", errorf);
	final_error1 = errorf;
      }
      
    }

    
    // learn WITH global optimum pretraining
    {
      whiteice::nnetwork<> net(rand_net);

      const unsigned int N = 100; // only 100 random neural networks are used to estimate weights
      const unsigned int M = 10000; // only 1000 random training points per NN
      const unsigned int K = 16;    // discretizes each variables to 16 bins

      // initializes weights using data
      net.presetWeightsFromData(train_data);
      if(global_optimizer_pretraining(net, train_data, N, M, K) == false){
	printf("ERROR: global optimizer pretraining FAILED.\n");
	return;
      }

      whiteice::math::NNGradDescent<> grad;
      grad.setUseMinibatch(true);
      grad.startOptimize(train_data, net, 1);

      while(grad.hasConverged(0.01) == false){
	sleep(1);
	whiteice::math::blas_real<float> error = 0.0f;
	unsigned int Nconverged = 0;
	
	if(grad.getSolutionStatistics(error, Nconverged) == false){
	  printf("ERROR: grad.getSolutionStatistics() failed.\n");
	  return;
	}
	else{
	  float errorf = 0.0f;
	  whiteice::math::convert(errorf, error);
	  printf("Optimizing error %d: %f.\n", Nconverged, errorf);
	}
      }

      

      grad.stopComputation();

      {
	nnetwork<> nn;
	whiteice::math::blas_real<float> error;
	unsigned int Nconverged;
	
	if(grad.getSolution(nn, error, Nconverged) == false){
	  printf("ERROR: grad.getSolution() failed.\n");
	  return;
	}
	
	net = nn;

	float errorf = 0.0f;
	whiteice::math::convert(errorf, error);

	printf("Optimization converged (Global Optimum pretraining). Error: %f. STOP.\n", errorf);
	final_error2 = errorf;
      }
    }

    printf("Final optimization error without pretraining: %f.\n", final_error1);
    printf("Final optimization error with global optimum pretraining: %f.\n", final_error2);
  }
  
  // 3. test case: create testcase which uses XOR of digits of decimal as output,
  //    generates 10.000 training samples
  //    test how well neural network learns with and without global optimum pretraining
  //       x, y = (int)(10^k(i) * input(i)) % 10;  output(j) = x*y
  if(0){
    std::cout << "3. Test learning Kth digit XOR/multi neural network." << std::endl;

    const unsigned int N = 10000;

    whiteice::nnetwork<> rand_net;
    std::vector<unsigned int> arch;
    arch.push_back(10);
    //arch.push_back(50);
    //arch.push_back(50);
    arch.push_back(50);
    arch.push_back(2);
    
    // rand_net.setArchitecture(arch, whiteice::nnetwork<>::tanh); // sigmoid
    // rand_net.setArchitecture(arch, whiteice::nnetwork<>::sigmoid);
    rand_net.setArchitecture(arch, whiteice::nnetwork<>::halfLinear);
			     
    rand_net.randomize(0);

    std::vector< math::vertex<> > input;
    std::vector< math::vertex<> > output;

    whiteice::RNG<> rng;
    math::vertex<> v, w;
    v.resize(rand_net.input_size());
    w.resize(rand_net.output_size());

    std::vector< std::vector<unsigned int> > pair;
    std::vector< std::vector<unsigned int> > digit;
    for(unsigned int i=0;i<rand_net.output_size();i++){
      std::vector<unsigned int> e;
      const unsigned int d1 = 1 + (rng.rand() % 3); // digit = [1,3]:th decimal position
      const unsigned int d2 = 1 + (rng.rand() % 3); // digit = [1,3]:th decimal position
      e.push_back(d1);
      e.push_back(d2);
      digit.push_back(e);

      e.clear();
      const unsigned int k1 = rng.rand() % rand_net.input_size();
      const unsigned int k2 = rng.rand() % rand_net.input_size();
      e.push_back(k1);
      e.push_back(k2);
      pair.push_back(e);
    }

    for(unsigned int n=0;n<N;n++){
      rng.normal(v);
      w.zero();

      //rand_net.calculate(v, w);
      for(unsigned int i=0;i<w.size() && i<v.size();i++){
	math::blas_real<float> value = (0.0f);
	value = v[ pair[i][0] ]*whiteice::math::pow(10.0f, (float)digit[i][0]);
	int k1 = 0;
	whiteice::math::convert(k1, value);
	int t = whiteice::math::abs(k1) % 10;
	if(k1 >= 0) k1 = t;
	else k1 = -t;
	  
	value = v[ pair[i][1] ]*whiteice::math::pow(10.0f, (float)digit[i][1]);
	int k2 = 0;
	whiteice::math::convert(k2, value);
	t = whiteice::math::abs(k2) % 10;
	if(k2 >= 0) k2 = t;
	else k2 = -t;

	w[i] = ((float)(k1)) * ((float)k2);
      }
      
      
      input.push_back(v);
      output.push_back(w);
    }

    
    
    float final_error1 = 0.0f, final_error2 = 0.0f;

    whiteice::dataset<> train_data;
    
    train_data.createCluster("input", rand_net.input_size());
    train_data.createCluster("output", rand_net.output_size());
    
    train_data.add(0, input);
    train_data.add(1, output);
    
    train_data.preprocess(0, dataset<>::dnMeanVarianceNormalization);
    train_data.preprocess(1, dataset<>::dnMeanVarianceNormalization);
    

    // learn without global optimum pretraining
    {
      whiteice::nnetwork<> net(rand_net);

      // initializes weights using data
      // net.presetWeightsFromData(train_data);
      net.randomize();

      whiteice::math::NNGradDescent<> grad;
      grad.setUseMinibatch(true);
      grad.startOptimize(train_data, net, 1);

      while(grad.hasConverged(0.01) == false){
	sleep(1);
	whiteice::math::blas_real<float> error = 0.0f;
	unsigned int Nconverged = 0;
	
	if(grad.getSolutionStatistics(error, Nconverged) == false){
	  printf("ERROR: grad.getSolutionStatistics() failed.\n");
	  return;
	}
	else{
	  float errorf = 0.0f;
	  whiteice::math::convert(errorf, error);
	  printf("Optimizing error %d: %f.\n", Nconverged, errorf);
	}
      }

      grad.stopComputation();

      {
	nnetwork<> nn;
	whiteice::math::blas_real<float> error;
	unsigned int Nconverged;
	
	if(grad.getSolution(nn, error, Nconverged) == false){
	  printf("ERROR: grad.getSolution() failed.\n");
	  return;
	}
	
	net = nn;

	float errorf = 0.0f;
	whiteice::math::convert(errorf, error);

	printf("Optimization converged. Error: %f. STOP.\n", errorf);
	final_error1 = errorf;
      }
      
    }

    
    // learn WITH global optimum pretraining
    {
      whiteice::nnetwork<> net(rand_net);

      const unsigned int N = 100; // only 100 random neural networks are used to estimate weights
      const unsigned int M = 10000; // only 1000 random training points per NN
      const unsigned int K = 16;    // discretizes each variables to 16 bins

      // initializes weights using data
      net.presetWeightsFromData(train_data);
      if(global_optimizer_pretraining(net, train_data, N, M, K) == false){
	printf("ERROR: global optimizer pretraining FAILED.\n");
	return;
      }

      whiteice::math::NNGradDescent<> grad;
      grad.setUseMinibatch(true);
      grad.startOptimize(train_data, net, 1);

      while(grad.hasConverged(0.01) == false){
	sleep(1);
	whiteice::math::blas_real<float> error = 0.0f;
	unsigned int Nconverged = 0;
	
	if(grad.getSolutionStatistics(error, Nconverged) == false){
	  printf("ERROR: grad.getSolutionStatistics() failed.\n");
	  return;
	}
	else{
	  float errorf = 0.0f;
	  whiteice::math::convert(errorf, error);
	  printf("Optimizing error %d: %f.\n", Nconverged, errorf);
	}
      }

      

      grad.stopComputation();

      {
	nnetwork<> nn;
	whiteice::math::blas_real<float> error;
	unsigned int Nconverged;
	
	if(grad.getSolution(nn, error, Nconverged) == false){
	  printf("ERROR: grad.getSolution() failed.\n");
	  return;
	}
	
	net = nn;

	float errorf = 0.0f;
	whiteice::math::convert(errorf, error);

	printf("Optimization converged (Global Optimum pretraining). Error: %f. STOP.\n", errorf);
	final_error2 = errorf;
      }
    }

    printf("Final optimization error without pretraining: %f.\n", final_error1);
    printf("Final optimization error with global optimum pretraining: %f.\n", final_error2);
  }


  // 4. test case: input is 
  //    generates 10.000 training samples
  if(1){
    std::cout << "4. Test learning Two Rings problem." << std::endl;

    const unsigned int N = 10000;

    whiteice::nnetwork<> rand_net;
    std::vector<unsigned int> arch;
    arch.push_back(10);
    arch.push_back(50);
    arch.push_back(50);
    arch.push_back(50);
    arch.push_back(50);
    arch.push_back(2);
    
    // rand_net.setArchitecture(arch, whiteice::nnetwork<>::tanh); // sigmoid
    // rand_net.setArchitecture(arch, whiteice::nnetwork<>::sigmoid);
    // rand_net.setArchitecture(arch, whiteice::nnetwork<>::halfLinear);
    // rand_net.setArchitecture(arch, whiteice::nnetwork<>::pureLinear);
    rand_net.setArchitecture(arch, whiteice::nnetwork<>::rectifier);
    rand_net.randomize();

    std::vector< math::vertex<> > input;
    std::vector< math::vertex<> > output;

    whiteice::RNG<> rng;
    math::vertex<> v, w;
    v.resize(rand_net.input_size());
    w.resize(rand_net.output_size());

    
    for(unsigned int n=0;n<N;n++){
      whiteice::math::blas_real<float> r = (float)(rng.rand() & 1);
      if(r >= 1.0f) r = 20.0f;
      rng.normal(v);
      v = r*v;
      
      w.zero();

      //rand_net.calculate(v, w);
      for(unsigned int i=0;i<w.size() && i<v.size();i++){
	w[i] = r;
      }
      
      input.push_back(v);
      output.push_back(w);
    }

    
    
    float final_error1 = 0.0f, final_error2 = 0.0f;

    whiteice::dataset<> train_data;
    
    train_data.createCluster("input", rand_net.input_size());
    train_data.createCluster("output", rand_net.output_size());
    
    train_data.add(0, input);
    train_data.add(1, output);
    
    train_data.preprocess(0, dataset<>::dnMeanVarianceNormalization);
    train_data.preprocess(1, dataset<>::dnMeanVarianceNormalization);
    

    // learn without global optimum pretraining
    {
      whiteice::nnetwork<> net(rand_net);

      net.randomize();

      whiteice::math::NNGradDescent<> grad;
      grad.setUseMinibatch(true);
      grad.startOptimize(train_data, net, 1);

      while(grad.hasConverged(0.01) == false){
	sleep(1);
	whiteice::math::blas_real<float> error = 0.0f;
	unsigned int Nconverged = 0;
	
	if(grad.getSolutionStatistics(error, Nconverged) == false){
	  printf("ERROR: grad.getSolutionStatistics() failed.\n");
	  return;
	}
	else{
	  float errorf = 0.0f;
	  whiteice::math::convert(errorf, error);
	  printf("Optimizing error %d: %f.\n", Nconverged, errorf);
	}
      }

      grad.stopComputation();

      {
	nnetwork<> nn;
	whiteice::math::blas_real<float> error;
	unsigned int Nconverged;
	
	if(grad.getSolution(nn, error, Nconverged) == false){
	  printf("ERROR: grad.getSolution() failed.\n");
	  return;
	}
	
	net = nn;

	float errorf = 0.0f;
	whiteice::math::convert(errorf, error);

	printf("Optimization converged. Error: %f. STOP.\n", errorf);
	final_error1 = errorf;
      }
      
    }

    
    // learn WITH global optimum pretraining
    {
      whiteice::nnetwork<> net(rand_net);

      const unsigned int N = 100; // only 100 random neural networks are used to estimate weights
      const unsigned int M = 10000; // only 1000 random training points per NN
      const unsigned int K = 16;    // discretizes each variables to 16 bins

      // net.presetWeightsFromData(train_data);
      if(global_optimizer_pretraining(net, train_data, N, M, K) == false){
	printf("ERROR: global optimizer pretraining FAILED.\n");
	return;
      }

      whiteice::math::NNGradDescent<> grad;
      grad.setUseMinibatch(true);
      grad.startOptimize(train_data, net, 1);

      while(grad.hasConverged(0.01) == false){
	sleep(1);
	whiteice::math::blas_real<float> error = 0.0f;
	unsigned int Nconverged = 0;
	
	if(grad.getSolutionStatistics(error, Nconverged) == false){
	  printf("ERROR: grad.getSolutionStatistics() failed.\n");
	  return;
	}
	else{
	  float errorf = 0.0f;
	  whiteice::math::convert(errorf, error);
	  printf("Optimizing error %d: %f.\n", Nconverged, errorf);
	}
      }

      

      grad.stopComputation();

      {
	nnetwork<> nn;
	whiteice::math::blas_real<float> error;
	unsigned int Nconverged;
	
	if(grad.getSolution(nn, error, Nconverged) == false){
	  printf("ERROR: grad.getSolution() failed.\n");
	  return;
	}
	
	net = nn;

	float errorf = 0.0f;
	whiteice::math::convert(errorf, error);

	printf("Optimization converged (Global Optimum pretraining). Error: %f. STOP.\n", errorf);
	final_error2 = errorf;
      }
    }

    printf("Final optimization error without pretraining: %f.\n", final_error1);
    printf("Final optimization error with global optimum pretraining: %f.\n", final_error2);
  }
  
}

/************************************************************/

void simple_vae_test()
{
  try{
    std::cout << "VAE optimization test" << std::endl;
    
    whiteice::nnetwork<> encoder, decoder;
    
    std::vector<unsigned int> arch_e, arch_d;
    
    arch_e.push_back(10);
    arch_e.push_back(20);
    arch_e.push_back(2*2);
    
    arch_d.push_back(2);
    arch_d.push_back(20);
    arch_d.push_back(10);
    
    encoder.setArchitecture(arch_e, whiteice::nnetwork<>::rectifier);
    decoder.setArchitecture(arch_d, whiteice::nnetwork<>::rectifier); 
    
    whiteice::VAE<> vae(encoder, decoder);
    
    
    if(vae.setModel(encoder, decoder) == false){
      std::cout << "ERROR: setting models FAILED" << std::endl;
      exit(-1);
    }
    
    whiteice::RNG<> rng;
    std::vector< math::vertex<> > samples;
    
    for(unsigned int i=0;i<1000;i++){
      math::vertex<> v;
      v.resize(10);
      rng.uniform(v);
      samples.push_back(v);
    }
    
    vae.initializeParameters();
    
    if(vae.learnParameters(samples, 0.01, true) == false){
      std::cout << "ERROR: learning parameters FAILED" << std::endl;
      exit(-1);
    }

    std::cout << "VAE tests ended (success)." << std::endl;
    
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception " 
	      << e.what() << std::endl;
    return;
  }

}

/************************************************************/

void nnetwork_gradient_test()
{
  std::cout << "Real-valued nnetwork gradient() calculations test" << std::endl;

  whiteice::RNG< whiteice::math::blas_real<double> > rng;

  for(unsigned int e=0;e<10;e++) // number of tests
  {
    std::vector<unsigned int> arch;

    const unsigned int dimInput = rng.rand() % 10 + 3;
    const unsigned int dimOutput = rng.rand() % 10 + 3;
    const unsigned int layers = rng.rand() % 2 + 1;

    arch.push_back(dimInput);
    for(unsigned int i=0;i<layers;i++)
      arch.push_back(rng.rand() % 5 + 2);
    arch.push_back(dimOutput);

    whiteice::nnetwork< whiteice::math::blas_real<double> >::nonLinearity nl =
      whiteice::nnetwork< whiteice::math::blas_real<double> >::rectifier;
    
    unsigned int nli = rng.rand() % 7;
    // nli = 3; // force purelinear f(x)

    if(nli == 0){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::sigmoid;
      printf("Sigmoidal non-linearity\n");
    }
    else if(nli == 1){
      // do not calculate gradients for stochastic sigmoid..
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::sigmoid; 
      printf("Sigmoidal non-linearity\n");
    }
    else if(nli == 2){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::halfLinear;
      printf("halfLinear non-linearity\n");
    }
    else if(nli == 3){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::pureLinear;
      printf("pureLinear non-linearity\n");
    }
    else if(nli == 4){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::tanh;
      printf("tanh non-linearity\n");
    }
    else if(nli == 5){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::rectifier;
      printf("rectifier non-linearity\n");
    }
    else if(nli == 6){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::softmax;
      printf("softmax non-linearity\n");
    }

    whiteice::nnetwork< whiteice::math::blas_real<double> > nn(arch, nl);

    nn.randomize();

    whiteice::math::vertex< whiteice::math::blas_real<double> > x(dimInput);
    whiteice::math::vertex< whiteice::math::blas_real<double> > y(dimOutput);
    x.zero();
    y.zero();

    rng.normal(x);
    rng.exp(y);

    nn.input() = x;
    nn.calculate(true, false);

    auto error = nn.output() - y;

    whiteice::math::vertex< whiteice::math::blas_real<double> > grad;

    if(nn.mse_gradient(error, grad) == false){
      printf("ERROR: nn::gradient(1) FAILED.\n");
      continue;
    }
    
    whiteice::math::matrix< whiteice::math::blas_real<double> > grad2;

    if(nn.jacobian(x, grad2) == false){
      printf("ERROR: nn::gradient(2) FAILED.\n");
      continue;
    }

    whiteice::math::vertex< whiteice::math::blas_real<double> > g = error*grad2;

    if(grad.size() != g.size()){
      printf("ERROR: nn::gradient sizes mismatch!\n");
      continue;
    }

    whiteice::math::blas_real<double> err = 0.0;

    auto grad_delta = grad - g;

    err = grad_delta.norm();
    err /= ((double)g.size());

    if(err > 0.001){
      printf("ERROR: gradient difference is too large (%f)!\n", err.c[0]);
      std::cout << "grad_delta = " << grad_delta << std::endl;
    }
    else{
      printf("Backpropagation and error*nnetwork::gradient() return same value (error %f). Good.\n",
	     err.c[0]);
      fflush(stdout);
    }
    
  }
  
}

/**********************************************************************/

void nnetwork_gradient_value_test()
{
  std::cout << "nnetwork::gradient_value() optimization test." << std::endl;

  whiteice::RNG< whiteice::math::blas_real<double> > rng;

  for(unsigned int e=0;e<10;e++)
  {
    std::vector<unsigned int> arch;

    const unsigned int dimInput = rng.rand() % 10 + 3;
    const unsigned int dimOutput = 1;
    const unsigned int layers = rng.rand() % 3 + 4;
    const unsigned int width = rng.rand() % 10 + 10;

    arch.push_back(dimInput);
    for(unsigned int i=0;i<layers;i++)
      arch.push_back(width);
    arch.push_back(dimOutput);

    whiteice::nnetwork< whiteice::math::blas_real<double> >::nonLinearity nl =
      whiteice::nnetwork< whiteice::math::blas_real<double> >::rectifier;
    unsigned int nli = 5; // rng.rand() % 7;

    if(nli == 0){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::sigmoid;
    }
    else if(nli == 1){
      // do not calculate gradients for stochastic sigmoid..
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::sigmoid; 
    }
    else if(nli == 2){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::halfLinear;
    }
    else if(nli == 3){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::pureLinear;
    }
    else if(nli == 4){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::tanh;
    }
    else if(nli == 5){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::rectifier;
    }
    else if(nli == 6){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::softmax;
    }

    whiteice::nnetwork< whiteice::math::blas_real<double> > nn(arch, nl);
    
    nn.randomize();
    nn.setResidual(true);

    whiteice::math::vertex< whiteice::math::blas_real<double> > x(dimInput);
    whiteice::math::vertex< whiteice::math::blas_real<double> > y(dimOutput);
    whiteice::math::matrix< whiteice::math::blas_real<double> > GRAD_x;
    whiteice::math::vertex< whiteice::math::blas_real<double> > grad_x;
    whiteice::math::blas_real<double> alpha = 0.1f; // 0.0001f

    x.zero();
    rng.normal(x);

    nn.calculate(x, y);

    auto start_value = y;
    
    for(unsigned int i=0;i<1000;i++){
      //printf("ABOUT TO CALCULATE GRAD VALUE..\n"); fflush(stdout);
      nn.gradient_value(x, GRAD_x);
      //printf("ABOUT TO CALCULATE GRAD VALUE.. DONE\n"); fflush(stdout);
      //printf("GRAD_x: %d %d = %f\n", GRAD_x.ysize(), GRAD_x.xsize(), GRAD_x(0,0).c[0]);
      
      grad_x.resize(GRAD_x.xsize());
      for(unsigned int i=0;i<grad_x.size();i++)
	grad_x[i] = GRAD_x(0, i);
      
      x += alpha*grad_x;

      nn.calculate(x, y);

      // std::cout << i << "/100: " << y << std::endl;
    }

    auto end_value = y;

    if(end_value < start_value){
      std::cout << "ERROR: start value larger than end value. "
		<< start_value << " > " << end_value << std::endl;
      std::cout << "arch " << arch.size() << " : ";
#if 0
      for(unsigned int i=0;i<arch.size();i++)
	std::cout << arch[i]  << " ";
      std::cout << std::endl;
#endif
      
      return;
    }
    else{
      std::cout << "GOOD: start value smaller than end value. "
		<< start_value << " < " << end_value << std::endl;
#if 0
      std::cout << "arch " << arch.size() << " : ";
      for(unsigned int i=0;i<arch.size();i++)
	std::cout << arch[i]  << " ";
      std::cout << std::endl;
#endif
    }
  }
  
}

/**********************************************************************/


void nnetwork_residual_gradient_test()
{
  std::cout << "Real-valued RESIDUAL nnetwork gradient() calculations test" << std::endl;

  whiteice::RNG< whiteice::math::blas_real<double> > rng;

  for(unsigned int e=0;e<10;e++) // number of tests
  {
    std::vector<unsigned int> arch;

    const unsigned int dimInput = rng.rand() % 10 + 3;
    const unsigned int dimOutput = rng.rand() % 10 + 3;
    const unsigned int layers = rng.rand() % 10 + 4;
    const unsigned int width = rng.rand() % 10 + 10;

    arch.push_back(dimInput);
    for(unsigned int i=0;i<layers;i++)
      arch.push_back(width);
    arch.push_back(dimOutput);

    whiteice::nnetwork< whiteice::math::blas_real<double> >::nonLinearity nl =
      whiteice::nnetwork< whiteice::math::blas_real<double> >::rectifier;
    
    unsigned int nli = rng.rand() % 7;
    // nli = 3; // force purelinear f(x)

    if(nli == 0){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::sigmoid;
      printf("Sigmoidal non-linearity\n");
    }
    else if(nli == 1){
      // do not calculate gradients for stochastic sigmoid..
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::sigmoid; 
      printf("Sigmoidal non-linearity\n");
    }
    else if(nli == 2){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::halfLinear;
      printf("halfLinear non-linearity\n");
    }
    else if(nli == 3){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::pureLinear;
      printf("pureLinear non-linearity\n");
    }
    else if(nli == 4){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::tanh;
      printf("tanh non-linearity\n");
    }
    else if(nli == 5){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::rectifier;
      printf("rectifier non-linearity\n");
    }
    else if(nli == 6){
      nl = whiteice::nnetwork< whiteice::math::blas_real<double> >::softmax;
      printf("softmax non-linearity\n");
    }

    whiteice::nnetwork< whiteice::math::blas_real<double> > nn(arch, nl);

    nn.randomize();
    nn.setResidual(true);

    whiteice::math::vertex< whiteice::math::blas_real<double> > x(dimInput);
    whiteice::math::vertex< whiteice::math::blas_real<double> > y(dimOutput);
    x.zero();
    y.zero();

    rng.normal(x);
    rng.exp(y);

    nn.input() = x;
    nn.calculate(true, false);

    auto error = nn.output() - y;

    whiteice::math::vertex< whiteice::math::blas_real<double> > grad;

    if(nn.mse_gradient(error, grad) == false){
      printf("ERROR: nn::gradient(1) FAILED.\n");
      continue;
    }
    
    whiteice::math::matrix< whiteice::math::blas_real<double> > grad2;

    if(nn.jacobian(x, grad2) == false){
      printf("ERROR: nn::gradient(2) FAILED.\n");
      continue;
    }

    whiteice::math::vertex< whiteice::math::blas_real<double> > g = error*grad2;

    if(grad.size() != g.size()){
      printf("ERROR: nn::gradient sizes mismatch!\n");
      continue;
    }

    whiteice::math::blas_real<double> err = 0.0;

    auto grad_delta = grad - g;

    err = grad_delta.norm();
    err /= ((double)g.size());

    if(err > 0.001){
      printf("ERROR: gradient difference is too large (%f)!\n", err.c[0]);
      //std::cout << "grad_delta = " << grad_delta << std::endl;
      return;
    }
    else{
      printf("RESIDUAL Backpropagation and error*nnetwork::jacobian() return same value (error %f). Good.\n", err.c[0]);
      fflush(stdout);
    }
    
  }
  
}




/************************************************************/

void nnetwork_complex_gradient_test()
{
  std::cout << "COMPLEX nnetwork gradient() calculations test" << std::endl;

  whiteice::RNG< whiteice::math::blas_complex<double> > rng;

  const unsigned int NUMTESTS = 20; // was 10

  for(unsigned int e=0;e<NUMTESTS;e++) // number of tests
  {
    std::vector<unsigned int> arch;

    const unsigned int dimInput = rng.rand() % 10 + 3;
    const unsigned int dimOutput = rng.rand() % 10 + 3;
    const unsigned int layers = rng.rand() % 3 + 1;

    arch.push_back(dimInput);
    for(unsigned int i=0;i<layers;i++)
      arch.push_back(rng.rand() % 5 + 2);
    arch.push_back(dimOutput);

    whiteice::nnetwork< whiteice::math::blas_complex<double> >::nonLinearity nl;
    unsigned int nli = rng.rand() % 7;
    // nli = 3; // force purelinear f(x)
    if(nli == 1) nli = 0; // don't use stochastic sigmoid..
    nl = (whiteice::nnetwork< whiteice::math::blas_complex<double> >::nonLinearity)(nli);

    whiteice::nnetwork< whiteice::math::blas_complex<double> > nn(arch, nl);

    nn.randomize(); // could large data cause problems(??)

    whiteice::math::vertex< whiteice::math::blas_complex<double> > x(dimInput);
    whiteice::math::vertex< whiteice::math::blas_complex<double> > y(dimOutput);
    x.zero();
    y.zero();

    rng.normal(x);

    rng.exp(y);

    nn.input() = x;

    nn.calculate(true, true);

    auto error = nn.output() - y;
    
    whiteice::math::vertex< whiteice::math::blas_complex<double> > mse_grad;

    if(nn.mse_gradient(error, mse_grad) == false){ // returns delta*conj(Jnn)
      printf("ERROR: nn::gradient(1) FAILED.\n");
      continue;
    }

    // calculates correct positive gradient direction which is (f(z)-y)*conj(grad(fz))
    whiteice::math::matrix< whiteice::math::blas_complex<double> > Jnn;

    if(nn.jacobian(x, Jnn) == false){
      printf("ERROR: nn::gradient(2) FAILED.\n");
      continue;
    }

#if 0
    {
      printf("Neural network input/output size: %d->%d\n",
	     nn.input_size(), nn.output_size());
      printf("NN Jacobian size: y x x = %d x %d\n",
	     Jnn.ysize(), Jnn.xsize());
      
      printf("NN Architecture: ");
      for(unsigned int a=0;a<arch.size();a++)
	printf(" %d", arch[a]);
      printf("\n");

      if(nli == 0){	
	printf("Sigmoidal non-linearity\n");
      }
      else if(nli == 1){
	// do not calculate gradients for stochastic sigmoid..
	nl = whiteice::nnetwork< whiteice::math::blas_complex<double> >::sigmoid; 
	printf("Sigmoidal non-linearity\n");
      }
      else if(nli == 2){
	nl = whiteice::nnetwork< whiteice::math::blas_complex<double> >::halfLinear;
	printf("halfLinear non-linearity\n");
      }
      else if(nli == 3){
	nl = whiteice::nnetwork< whiteice::math::blas_complex<double> >::pureLinear;
	printf("pureLinear non-linearity\n");
      }
      else if(nli == 4){
	nl = whiteice::nnetwork< whiteice::math::blas_complex<double> >::tanh;
	printf("tanh non-linearity\n");
      }
      else if(nli == 5){
	nl = whiteice::nnetwork< whiteice::math::blas_complex<double> >::rectifier;
	printf("rectifier non-linearity\n");
      }
      else if(nli == 6){
	nl = whiteice::nnetwork< whiteice::math::blas_complex<double> >::softmax;
	printf("softmax non-linearity\n");
      }
      
      fflush(stdout);
    }
#endif
    
    
    auto delta = error; // we want want positive direction: (f(x)-y)
    Jnn.conj();

    
    whiteice::math::vertex< whiteice::math::blas_complex<double> > g = delta*Jnn;
    
    if(mse_grad.size() != g.size()){
      printf("ERROR: nn::gradient sizes mismatch!\n");
      fflush(stdout);
      continue;
    }

    whiteice::math::blas_real<double> err = 0.0;

    auto grad_delta = mse_grad - g;

    err = grad_delta.norm();
    err /= ((double)g.size());

    if(abs(err) > 0.01){
      printf("ERROR: complex gradient difference is too large!\n");
      printf("Norm(error) value: %f\n", err.c[0]);
      std::cout << "grad_delta = " << grad_delta << std::endl;
      fflush(stdout);
    }
    else{
      printf("Complex backpropagation and error*nnetwork::gradient() return same value (Error: %f). Good.\n", err.c[0]);
      fflush(stdout);
    }
    
  }

  std::cout << std::endl << std::endl;
  
}

/*********************************************************************************/

void ensemble_means_test()
{
  std::cout << "Ensemble means testing" << std::endl;

  std::vector< math::vertex< math::blas_real<double> > > data;
  whiteice::RNG< math::blas_real<double> > rng;

  {
    std::vector< math::vertex< math::blas_real<double> > > input;
    
    const unsigned int DIM=5;

    for(unsigned int i=0;i<10000;i++){ // 10000 examples
      math::vertex< math::blas_real<double> > in;

      in.resize(DIM);
      for(unsigned int j=0;j<in.size();j++){
	in[j] = rng.uniform();
      }

      data.push_back(in);
    }
  }

  whiteice::EnsembleMeans< math::blas_real<double> > em;

  assert(em.learn(3, data) == true);

  std::vector< math::vertex< math::blas_real<double> > > kmeans;
  std::vector< math::blas_real<double> > p;

  assert(em.getClustering(kmeans, p) == true);

  for(const auto& v : p)
    std::cout << 100.0*v << "% ";
  std::cout << std::endl;
  
}

/************************************************************/

void mixture_nnetwork_test()
{
  std::cout << "Mixture of experts learning" << std::endl;
  
  whiteice::RNG< math::blas_real<double> > rng;
  
  whiteice::dataset< math::blas_real<double> >  data;

  // generates dummy data for learning (5 values and takes min value) (so it is a clear function..)
  {
    std::vector< math::vertex< math::blas_real<double> > > input;
    std::vector< math::vertex< math::blas_real<double> > > output;

    const unsigned int DIM=5;

    for(unsigned int i=0;i<10000;i++){ // 10000 examples
      math::vertex< math::blas_real<double> > in;
      math::vertex< math::blas_real<double> > out;

      in.resize(DIM);
      for(unsigned int j=0;j<in.size();j++){
	in[j] = rng.uniform();
      }

      auto min = in[0];
      for(unsigned int j=0;j<in.size();j++){
	if(in[j] < min) min = in[j];
      }

      out.resize(1);
      out[0] = min;
      
      input.push_back(in);
      output.push_back(out);
    }

    data.createCluster("input", DIM);
    data.createCluster("output", 1);

    data.add(0, input);
    data.add(1, output);

    // no preprocessing of data
    data.preprocess(0, whiteice::dataset< math::blas_real<double> >::dnMeanVarianceNormalization);
    data.preprocess(1, whiteice::dataset< math::blas_real<double> >::dnMeanVarianceNormalization);
  }

  whiteice::nnetwork< math::blas_real<double> > * nn = nullptr;
  
  // creates nnetwork
  {
    std::vector< unsigned int > arch;

    {
      arch.push_back(data.dimension(0));
      arch.push_back(10*(data.dimension(0)+data.dimension(1)));
      arch.push_back(10*(data.dimension(0)+data.dimension(1)));
      arch.push_back(data.dimension(1));
    }
    
    
    nn = new whiteice::nnetwork< math::blas_real<double> >(arch);
    nn->setNonlinearity(nnetwork< math::blas_real<double> >::halfLinear);
    nn->randomize();
  }

  whiteice::Mixture< whiteice::math::blas_real<double> > mixture(5);

  assert(mixture.minimize(*nn, data) == true);

  unsigned int iters = 0;

  while(mixture.isRunning() &&
	mixture.solutionConverged() == false){

    std::vector< whiteice::math::blas_real<double> > error;
    std::vector< whiteice::math::vertex< whiteice::math::blas_real<double> > > w;
    std::vector< whiteice::math::blas_real<double> > p;
    
    unsigned int it = 0;
    unsigned int changes = 0;

    if(mixture.getSolution(w, error, p, it, changes)){
      if(iters != it){
	printf("%d ITERS: %d deltas ", iters, changes);
	for(unsigned int i=0;i<error.size();i++){
	  printf("%.4f(%.1f%%) ", error[i].c[0], 100.0*p[i].c[0]);
	}
	printf("\n");
	fflush(stdout);
	iters = it;
      }
    }

    sleep(1);
  }

  
  
}


/************************************************************/

void simple_recurrent_nnetwork_test()
{

  std::cout << "Simple recurrent neural network optimizer tests."
	    << std::endl;

  whiteice::RNG< math::blas_real<double> > rng;
  
  const unsigned int DEEPNESS = 10; // 10
  const unsigned int RDIM = 5; // recursion data dimension

  printf("DEEPNESS = %d\n", DEEPNESS);

  whiteice::nnetwork< math::blas_real<double> > * nn = nullptr;
  whiteice::dataset< math::blas_real<double> >  data;

  // generates dummy data for learning (5 values and takes min value)
  {
    std::vector< math::vertex< math::blas_real<double> > > input;
    std::vector< math::vertex< math::blas_real<double> > > output;

    const unsigned int DIM=5;

    for(unsigned int i=0;i<1000;i++){ // 1000 examples
      math::vertex< math::blas_real<double> > in;
      math::vertex< math::blas_real<double> > out;

      in.resize(DIM);
      for(unsigned int j=0;j<in.size();j++){
	in[j] = rng.uniform();
      }

      auto min = in[0];
      for(unsigned int j=0;j<in.size();j++){
	if(in[j] < min) min = in[j];
      }

      out.resize(1);
      out[0] = min;
      
      input.push_back(in);
      output.push_back(out);
    }

    data.createCluster("input", DIM);
    data.createCluster("output", 1);

    data.add(0, input);
    data.add(1, output);

    // no preprocessing of data
    data.preprocess(0, whiteice::dataset< math::blas_real<double> >::dnMeanVarianceNormalization);
    data.preprocess(1, whiteice::dataset< math::blas_real<double> >::dnMeanVarianceNormalization);
  }

  // creates nnetwork
  {
    std::vector< unsigned int > arch;

    if(DEEPNESS > 1){
      arch.push_back(data.dimension(0)+RDIM);
      arch.push_back(10*(data.dimension(0)+data.dimension(1)));
      arch.push_back(10*(data.dimension(0)+data.dimension(1)));
      arch.push_back(data.dimension(1)+RDIM);
    }
    else{
      arch.push_back(data.dimension(0));
      arch.push_back(10*(data.dimension(0)+data.dimension(1)));
      arch.push_back(10*(data.dimension(0)+data.dimension(1)));
      arch.push_back(data.dimension(1));
    }
    
    
    nn = new whiteice::nnetwork< math::blas_real<double> >(arch);
    //nn->randomize(2, true); // SETS SMALL INITIAL VALUES
    nn->randomize(2); // SETS LARGE VALUES
  }

  math::vertex< math::blas_real<double> > w;

  nn->exportdata(w);
  nn->setNonlinearity(nnetwork< math::blas_real<double> >::rectifier);

  // deepness of recursiveness
  whiteice::rLBFGS_nnetwork< math::blas_real<double> >
    optimizer(*nn, data, DEEPNESS, false, false);

  //optimizer.setGradientOnly(true); // follow only gradient

  optimizer.minimize(w); // starts optimization

  unsigned int iters = 0;

  // deepness 10 (best seen 0.009577)!!! (linear and half-linear rectifier..)

  while(optimizer.isRunning() &&
	optimizer.solutionConverged() == false){

    whiteice::math::blas_real<double> error = 0.0;

    unsigned int it = 0;

    if(optimizer.getSolution(w, error, it)){
      if(iters != it){
	printf("%d ITERS: %f (best non-recursive %f [rectifier])\n", iters, error.c[0], 0.012551);
	fflush(stdout);
	iters = it;
      }
    }

    sleep(1);
  }


  
}

/************************************************************/

void bbrbm_test()
{
  std::cout << "BBRBM UNIT test.." << std::endl;



  {
    std::cout << "Unit testing BBRBM::save() and BBRBM::load() functions.." << std::endl;
    
    whiteice::BBRBM< math::blas_real<double> > rbm1, rbm2;
    
    rbm1.resize(20,30);
    rbm1.initializeWeights();
    
    rbm2.resize(11,24);
    rbm2.initializeWeights();
    
    if(rbm1 != rbm1) std::cout << "BBRBM comparison ERROR." << std::endl;
    if(rbm2 == rbm1) std::cout << "BBRBM comparison ERROR." << std::endl;
    
    rbm2 = rbm1;
    if(rbm2 != rbm1) std::cout << "BBRBM comparison ERROR." << std::endl;

    rbm2.initializeWeights();
    rbm1.initializeWeights();
    
    if(rbm2.save("rbmparams.dat") == false)
      std::cout << "BBRBM::save() FAILED." << std::endl;
    if(rbm1.load("rbmparams.dat") == false)
      std::cout << "BBRBM::load() FAILED." << std::endl;
    
    if(rbm2 != rbm1){
      std::cout << "BBRBM load() FAILED to create identical BBRBM." << std::endl;
      
      auto Wdiff = rbm1.getWeights() - rbm2.getWeights();
      auto bdiff = rbm1.getBValue() - rbm2.getBValue();
      auto adiff = rbm1.getAValue() - rbm2.getAValue();
      
      if(math::frobenius_norm(Wdiff) < 10e-6 && bdiff.norm() < 10e-6 && adiff.norm() < 10e-6){
	std::cout << "But weights difference is less than 10e-6.. OK" << std::endl;
      }
    }
    
  }
  
  
  // use higher number of dimensions 16x16 = 256
  const unsigned int DIMENSION = 256;
  const unsigned int HIDDEN_DIMENSION = 10;
  const unsigned int SAMPLES = 10000;

  whiteice::BBRBM< math::blas_real<double> > bbrbm;
  std::vector< math::vertex< math::blas_real<double> > > samples;
  whiteice::RNG< math::blas_real<double> > rng;
  
  // generates training data for binary RBM machine
  {
    // we generate pictures of a circles with random center and radius of 4 with a wrap-a-around
    // around the borders and inspect how well RBM can learn them..

    math::vertex< math::blas_real<double> > image(DIMENSION);

    for(unsigned int n=0;n<SAMPLES;n++){
      image.zero();

      math::blas_real<double> radius = 8;
      math::blas_real<double> x0 = rng.rand() % 16;
      math::blas_real<double> y0 = rng.rand() % 16;
      
      for(unsigned int k=0;k<256;k++){
	double angle = 2*M_PI*((double)k/256.0);

	auto x = x0 + radius*sin(angle);
	auto y = y0 + radius*cos(angle);

	if(x<0.0) x += 16.0;
	if(y<0.0) y += 16.0;
	if(x>=16.0) x -= 16.0;
	if(y>=16.0) y -= 16.0;

	unsigned int xx = 0, yy = 0;
	math::convert(xx, x);
	math::convert(yy, y);

	unsigned int address = xx + yy*16;

	image[address] = 1.0;
      }

      samples.push_back(image);
    }
  }

#if 1
  // learns weights given training data (gradient descent)
  {

    bbrbm.resize(DIMENSION, HIDDEN_DIMENSION); // only 8x8=64 sized image as a hidden vector (NOW: check that we can learn identity)
    
    auto error = bbrbm.learnWeights(samples, 50, true);

    std::cout << "BBRBM final reconstruction error = "
	      << error << std::endl;
    
  }
  
#else

  // learns parameters using LBFGS second order optimizer
  {
    whiteice::dataset< math::blas_real<double> > ds;
    ds.createCluster("input", DIMENSION);
    ds.add(0, samples);

    bbrbm.resize(DIMENSION, HIDDEN_DIMENSION); // only 8x8=64 sized image as a hidden vector (NOW: check that we can learn identity)
    if(bbrbm.setUData(samples) == false)
      printf("Setting samples FAILED!\n");

    whiteice::LBFGS_BBRBM< math::blas_real<double> > optimizer(bbrbm, ds, false);

    math::vertex< math::blas_real<double> > x0;
    bbrbm.getParametersQ(x0);
    
    optimizer.minimize(x0);

    math::vertex< math::blas_real<double> > x;
    math::blas_real<double> error;
    int last_iter = -1;
    unsigned int iters = 0;

    while(true){
      if(!optimizer.isRunning() || optimizer.solutionConverged()){
	break;
      }
      
      optimizer.getSolution(x, error, iters);
      
      if((signed)iters > last_iter){
	printf("%d ITERATIONS. LBFGS RECONSTRUCTION ERROR: %f\n", iters, error.c[0]);
	if(bbrbm.setParametersQ(x) == false)
	  printf("setParametersQ() error\n");
	
	fflush(stdout);
	last_iter = iters;
      }
      
      sleep(1);
    }

    assert(optimizer.getSolution(x, error, iters) == true);

    bbrbm.setParametersQ(x0);
  }
#endif

  
  // after we have trained BBRBM we transform it to nnetwork manually and see we get the same results (hidden layer)!
  {
    whiteice::nnetwork< math::blas_real<double> > net;
    std::vector<unsigned int> arch;

    arch.push_back(DIMENSION);
    arch.push_back(HIDDEN_DIMENSION); // 1st layer is BBRBM non-linearity
    arch.push_back(HIDDEN_DIMENSION); // 2nd layer is identity layer because last layer of net is pureLinear

    if(net.setArchitecture
       (arch, nnetwork< math::blas_real<double> >::stochasticSigmoid) == false)
      std::cout << "ERROR: could not set architecture of nnetwork"
		<< std::endl;
    net.setNonlinearity(net.getLayers()-1,
			nnetwork< math::blas_real<double> >::pureLinear);
    
    math::matrix< math::blas_real<double> > W;
    math::vertex< math::blas_real<double> > b;

    W = bbrbm.getWeights();
    b = bbrbm.getBValue();

    if(net.setWeights(W, 0) == false){
      std::cout << "Nnetwork: Cannot set weights (0)" << std::endl;
    }
    
    if(net.setBias(b, 0) == false){
      std::cout << "Nnetwork: Cannot set bias (0)" << std::endl;
    }

    W.resize(HIDDEN_DIMENSION, HIDDEN_DIMENSION);
    W.identity();
    b.resize(HIDDEN_DIMENSION);
    b.zero();

    if(net.setWeights(W, 1) == false){
      std::cout << "Nnetwork: Cannot set weights (1)" << std::endl;
    }
    
    if(net.setBias(b, 1) == false){
      std::cout << "Nnetwork: Cannot set bias (1)" << std::endl;
    }

    // compares stimulation of BBRBM and nnetwork
    {
      auto& s = samples[0];
      math::vertex< math::blas_real<double> > out1, out2;

      bbrbm.setVisible(s);
      bbrbm.reconstructData(1);
      bbrbm.getHidden(out1);

      net.calculate(s, out2);

      std::cout << "THESE SHOULD BE MORE OR LESS SAME:" << std::endl;
      std::cout << "BBRBM    output: " << out1 << std::endl;
      std::cout << "nnetwork output: " << out2 << std::endl;

      for(unsigned int i=0;i<samples.size();i++){
	bbrbm.setVisible(samples[i]);
	bbrbm.reconstructData(1);
	bbrbm.getHidden(out1);

	std::cout << "BBRBM : " << out1 << std::endl;
      }
    }
    
    
  }
  
  
}

/************************************************************/

void dbn_test()
{
  std::cout << "Deep Belief Network (DBN) test.." << std::endl;

  whiteice::DBN< math::blas_real<double> > dbn;
  whiteice::dataset< math::blas_real<double> > ds;

  // generate training data
  std::vector< math::vertex< math::blas_real<double> > > input;
  std::vector< math::vertex< math::blas_real<double> > > output;

  const unsigned int DIMENSION = 2;
  const unsigned int NSAMPLES  = 10000;
  {
    // hermite curve and output values based on (sin(distance))
    // (this can be tricky problem when the curve is very long one,
    //  nnetwork should basically learn the curve and the concept of
    //  distance travelled)

    createHermiteCurve(input, 5, DIMENSION, NSAMPLES);

    // create output values
    math::blas_real<double> value, distance = 0;
      
    for(unsigned int i=0;i<input.size();i++){
      distance += 10*2*M_PI/NSAMPLES;
      value = sin(distance);

      math::vertex< math::blas_real<double> > o(1);
      o[0] = value;

      output.push_back(o);
    }

    ds.createCluster("input", DIMENSION);
    ds.createCluster("output", 1);

    ds.add(0, input);
    ds.add(1, output);
  }
  

  // learn DBN-network
  {
    // 
    // D x 50D x 10 neural network
    // 
    std::vector<unsigned int> arch;

    arch.push_back(input[0].size());
    arch.push_back(input[0].size()*50);
    arch.push_back(10);

    dbn.resize(arch);

#if 0
    math::blas_real<double> dW = 0.1;

    // optimizes DBN neural network
    if(dbn.learnWeights(input, dW, true) == false){
      std::cout << "DBN LEARNING FAILURE" << std::endl;
      exit(-1);
    }
#endif

    // reconstruct data
    auto reconstructedData = input;  
    
    if(dbn.reconstructData(reconstructedData) == false){
      std::cout << "ERROR: reconstructedData().." << std::endl;
      exit(-1);
    }

    if(!saveSamples("dbn_input.txt", input) ||
       !saveSamples("dbn_output.txt", output) || 
       !saveSamples("dbn_reconstructed.txt", reconstructedData)){
      std::cout << "ERROR: saveSamples() failed" << std::endl;
      exit(-1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    // converts DBN to lreg_nnetwork<T>

    printf("TESTING DBN CONVERSION TO NNETWORK..\n");

    whiteice::nnetwork< math::blas_real<double> >* nnet = NULL;

    if(dbn.convertToNNetwork(ds, nnet) == false){
      std::cout << "ERROR: conversion of DBN to neural network code failed." << std::endl;
      exit(-1);
    }
    else{
      printf("CONVERT TO NNETWORK: SUCCESS\n");
    }

    // normal optimization of pretrained DBN neural network
    LBFGS_nnetwork< math::blas_real<double> > optimizer(*nnet, ds,
							true, false);

    math::vertex< math::blas_real<double> > q;
    nnet->exportdata(q);

    optimizer.minimize(q);

    int latestIteration = -1;
    unsigned int currentIterator = 0;
    
    while(optimizer.isRunning() && !optimizer.solutionConverged()){
      math::blas_real<double> error;
      
      if(optimizer.getSolution(q, error, currentIterator)){
	if(latestIteration < (signed)currentIterator){
	  latestIteration = (signed)currentIterator;

	  std::cout << "FINETUNE " << currentIterator
		    << ": " << error << std::endl;
	  fflush(stdout);
	}
      }
      
      sleep(1);
    }

    // we have optimized DBN nnetwork and save results to disk
    {
      std::cout << "Saving deep neural network.." << std::endl;
      
      if(nnet->save("dbn_lreg_nnetwork.dat") == false){
	printf("Saving deep neural network FAILED\n");
      }
    }
    
  }
  
}



/************************************************************/

void hmc_test()
{

#ifdef __linux__
  // LINUX
  {
    feenableexcept(FE_INVALID |
		   FE_DIVBYZERO | 
		   FE_OVERFLOW |  FE_UNDERFLOW);
  }
  
#else
  // WINDOWS
  {
    _clearfp();
    unsigned unused_current_word = 0;
    // clearing the bits unmasks (throws) the exception
    _controlfp_s(&unused_current_word, 0,
		 _EM_INVALID | _EM_OVERFLOW | _EM_ZERODIVIDE);  // _controlfp_s is the secure version of _controlfp
  }
#endif



  std::cout << "HMC SAMPLING TEST (Neural network)" << std::endl;

  {
    // simple dummy test problem

    nnetwork<> example;
    dataset<>  ds;

    ds.createCluster("input (x)", 10);
    ds.createCluster("output (y)", 5);
    
    std::vector<unsigned int> arch;
    arch.push_back(ds.dimension(0));
    arch.push_back(10);
    arch.push_back(ds.dimension(1));
    
    example.setArchitecture(arch);
    example.randomize();

    for(unsigned int n=0;n<10000;n++){
      whiteice::math::vertex<> x, y;
      x.resize(ds.dimension(0));
      y.resize(ds.dimension(1));

      rng.uniform(x);
      
      example.calculate(x, y);

      ds.add(0, x);
      ds.add(1, y);
    }
    
    ds.preprocess(0);
    ds.preprocess(1);

    std::cout << "sizes: " << ds.size(0) << ", " << ds.size(1) << std::endl;

    example.randomize();

    HMC<> sampler(example, ds, true);

    sampler.startSampler();

    unsigned int counter = 0;

    while(/*counter < 1000*/1){
      std::cout << "sampling.. samples = "
		<< sampler.getNumberOfSamples()
		<< " error = "
		<< sampler.getMeanError(1000)
		<< std::endl;

      
      fflush(stdout);

      sleep(1);

      counter++;
    }

    
  }


#if 0
  std::cout << "HMC SAMPLING TEST (Normal distribution)" << std::endl;
  
  {
    HMC_gaussian<> sampler(2);
    
    time_t start_time = time(0);
    unsigned int counter = 0;

    sampler.startSampler();

    // samples for 5 seconds
    while(counter < 5){
      std::cout << "sampling.. samples = "
		<< sampler.getNumberOfSamples() << std::endl;
      fflush(stdout);
      
      sleep(1);
      counter = (unsigned int)(time(0) - start_time);
    }

    sampler.stopSampler();

    std::cout << sampler.getNumberOfSamples()
	      << " samples." << std::endl;

    std::cout << "mean = " << sampler.getMean() << std::endl;
    // std::cout << "cov  = " << sampler.getCovariance() << std::endl;

    std::cout << "Should be zero mean and unit I variance: N(0,I)"
	      << std::endl;

    std::cout << "Saving samples to CSV-file: gaussian.out" << std::endl;

    FILE* out = fopen("gaussian.out", "wt");

    std::vector< math::vertex< math::blas_real<float> > > samples;
    sampler.getSamples(samples);

    for(unsigned int i=0;i<samples.size();i++){
      fprintf(out, "%f, %f\n",
	      samples[i][0].c[0], samples[i][1].c[0]);
    }
    
    fclose(out);
    
  }
#endif

  std::cout << "HMC sampling test DONE." << std::endl;
  
  
}


/************************************************************/
/* restricted boltzmann machines tests */


void lbfgs_rbm_test()
{
#if 0
	std::cout << "LBFGS GB-RBM OPTIMIZATION TEST" << std::endl;

#ifdef __linux__
	// LINUX
	{
	  feenableexcept(FE_INVALID |
			 FE_DIVBYZERO);
	  //			 FE_OVERFLOW |  FE_UNDERFLOW);
	}
	
#else
	// WINDOWS
	{
		_clearfp();
		unsigned unused_current_word = 0;
		// clearing the bits unmasks (throws) the exception
		_controlfp_s(&unused_current_word, 0,
			     _EM_INVALID | _EM_OVERFLOW | _EM_ZERODIVIDE);  // _controlfp_s is the secure version of _controlfp
	}
#endif




	// generates test data: a D dimensional sphere with gaussian noise and tests that GB-RBM can correctly learn it
	{
		std::cout << "Generating spherical data.." << std::endl;

		unsigned int DIMENSION = 2; // mini-image size (16x16 = 256)

		std::vector< math::vertex< math::blas_real<double> > > samples;
		math::vertex< math::blas_real<double> > var;

		var.resize(DIMENSION);

		for(unsigned int i=0;i<DIMENSION;i++){
			var[i] = 1.0 + log((double)(i+1));
		}

		whiteice::RNG< math::blas_real<double> > rng;

		{
			// generates data for D (D-1 area) dimensional hypersphere

			math::vertex< math::blas_real<double> > m, v;
			m.resize(DIMENSION);
			v.resize(DIMENSION);

			for(unsigned int i=0;i<10000;i++){
				math::vertex< math::blas_real<double> > s;
				s.resize(DIMENSION);
				rng.normal(s);
				s.normalize(); // sample from unit hypersphere surface

				// 5x larger radius than the largest variance [to keep noise separated]
				s *= 5.0*math::sqrt(var[var.size()-1]);

				// adds variance
				for(unsigned int d=0;d<DIMENSION;d++)
					s[d] = s[d] + rng.normal()*math::sqrt(var[d]);

				samples.push_back(s);

				m += s;
				for(unsigned int j=0;j<s.size();j++)
					v[j] += s[j]*s[j];
			}

			m /= (double)samples.size();
			v /= (double)samples.size();

			for(unsigned int j=0;j<v.size();j++){
				v[j] -= m[j]*m[j];
				v[j] = sqrt(v[j]);
			}


			// normalizes mean and variance of data
			for(unsigned int s=0;s<samples.size();s++){
			  auto x = samples[s];
			  x -= m;
			  
			  for(unsigned int i=0;i<x.size();i++){
			    x[i] /= v[i];
			  }
			  
			  samples[s] = x;
			}

			for(unsigned int i=0;i<var.size();i++){
			  var[i] = math::sqrt(var[i])/v[i];
			  var[i] = var[i]*var[i];
			}
			
		}


		std::cout << "Learning GB-RBM parameters using LBFGS (2nd order optimization).." << std::endl;
		
		whiteice::GBRBM< math::blas_real<double> > gbrbm;
		whiteice::dataset< math::blas_real<double> > ds;
		
		gbrbm.resize(DIMENSION, 10); // 100 hidden nodes should be enough..
		gbrbm.initializeWeights();
		gbrbm.setUData(samples);

		ds.createCluster("input", DIMENSION);
		ds.add(0, samples);

		math::vertex< math::blas_real<double> > x0;
		//gbrbm.setVariance(var); // presets variance to correct values..
		gbrbm.getParametersQ(x0);

		const unsigned int NUMSOLVERS = 100;

		whiteice::LBFGS_GBRBM< math::blas_real<double> >* optimizer[NUMSOLVERS];

		math::vertex< math::blas_real<double> > x;
		math::blas_real<double> error;
		int last_iter = -1;
		unsigned int iters = 0;

		for(unsigned int i=0;i<NUMSOLVERS;i++){
		  auto temperature = 1.0;
		  // if(NUMSOLVERS > 1) temperature = i/((double)(NUMSOLVERS-1));
		  
		  if((i & 1) == 0){
		    std::cout << i << ": Optimization of variance." << std::endl;
		    gbrbm.setLearnVarianceMode();
		  }
		  else{
		    std::cout << i << ": Optimization of main variables." << std::endl;
		    gbrbm.setLearnParametersMode();
		  }

		  
		  std::cout << "Optimization at temperature: " << temperature << std::endl;
		  gbrbm.setUTemperature(temperature);
		  gbrbm.getParametersQ(x0);
		  
		  optimizer[i] = new whiteice::LBFGS_GBRBM< math::blas_real<double> >(gbrbm, ds, false);
		  optimizer[i]->minimize(x0);

		  last_iter = -1;
		  iters = 0;
		  
		  while(true){
		    if(!optimizer[i]->isRunning() || optimizer[i]->solutionConverged()){
		      break;
		    }
		    
		    optimizer[i]->getSolution(x, error, iters);
		      
		    if((signed)iters > last_iter){
		      printf("%d ITERATIONS. LBFGS RECONSTRUCTION ERROR: %f\n", iters, error.c[0]);
		      if(gbrbm.setParametersQ(x) == false)
			printf("setParametersQ() error\n");
		      
		      // math::vertex< math::blas_real<double> > v;
		      // gbrbm.getVariance(v);
		      // std::cout << "Variance: " << v << std::endl;
		      
		      fflush(stdout);
		      last_iter = iters;
		    }
		      
		    sleep(1);
		  }

		  
		}
		
		//printf("%d IS RUNNING. %d SOLUTION CONVERGED\n", optimizer.isRunning(), optimizer.solutionConverged());

		auto bestx = x;
		auto besterror = error;
		optimizer[0]->getSolution(x, error, iters);

		for(unsigned int i=0;i<NUMSOLVERS;i++){
		  optimizer[i]->getSolution(x, error, iters);
		  if(error < besterror){
		    besterror = error;
		    bestx = x;
		  }
		}
		
		gbrbm.setParametersQ(bestx);
		
		auto rdata = samples;
		rdata.clear();
		
		// samples directly from P(v) [using AIS - annealed importance sampling. This probably DO NOT WORK PROPERLY.]
		gbrbm.sample(1000, rdata, samples); 

		std::cout << "Storing reconstruct (circle) results to disk.." << std::endl;

		saveSamples("gbrbm_lbfgs_sphere_input.txt", samples);
		saveSamples("gbrbm_lbfgs_sphere_output.txt", rdata);

		rdata = samples;
		
		// calculates reconstruction v -> h -> v
		gbrbm.reconstructData(rdata);

		saveSamples("gbrbm_lbfgs_sphere_output2.txt", rdata);
	}

#endif
}



void rbm_test()
{
	std::cout << "GB-RBM UNIT TEST" << std::endl;

#ifdef __linux__
	// LINUX
	{
	  feenableexcept(FE_INVALID |
			 FE_DIVBYZERO);
	  //			 FE_OVERFLOW |  FE_UNDERFLOW);
	}
	
#else
	// WINDOWS
	{
		_clearfp();
		unsigned unused_current_word = 0;
		// clearing the bits unmasks (throws) the exception
		_controlfp_s(&unused_current_word, 0,
			     _EM_INVALID | _EM_OVERFLOW | _EM_ZERODIVIDE);  // _controlfp_s is the secure version of _controlfp
	}
#endif


	{
	  std::cout << "Unit testing GBRBM::save() and GBRBM::load() functions.." << std::endl;

	  whiteice::GBRBM< math::blas_real<double> > rbm1, rbm2;

	  rbm1.resize(20,30);
	  rbm1.initializeWeights();

	  rbm2.resize(11,24);
	  rbm2.initializeWeights();

	  if(rbm1 != rbm1) std::cout << "GBRBM comparison ERROR." << std::endl;
	  if(rbm2 == rbm1) std::cout << "GBRBM comparison ERROR." << std::endl;

	  rbm2 = rbm1;
	  if(rbm2 != rbm1) std::cout << "GBRBM comparison ERROR." << std::endl;

	  if(rbm2.save("rbmparams.dat") == false) std::cout << "GBRBM::save() FAILED." << std::endl;
	  if(rbm1.load("rbmparams.dat") == false) std::cout << "GBRBM::load() FAILED." << std::endl;

	  if(rbm2 != rbm1) std::cout << "GBRBM load() FAILED to create identical GBRBM." << std::endl;
          
	}


#if 0

	// generates test data: a D dimensional sphere with gaussian noise and tests that GB-RBM can correctly learn it
	{
		std::cout << "Generating spherical data.." << std::endl;

		unsigned int DIMENSION = 2; // mini-image size (16x16 = 256)

		std::vector< math::vertex< math::blas_real<double> > > samples;
		math::vertex< math::blas_real<double> > var;

		var.resize(DIMENSION);

		for(unsigned int i=0;i<DIMENSION;i++){
			var[i] = 1.0 + log((double)(i+1));
		}

		whiteice::RNG< math::blas_real<double> > rng;

		{
			// generates data for D (D-1 area) dimensional hypersphere

			math::vertex< math::blas_real<double> > m, v;
			m.resize(DIMENSION);
			v.resize(DIMENSION);

			for(unsigned int i=0;i<10000;i++){
				math::vertex< math::blas_real<double> > s;
				s.resize(DIMENSION);
				rng.normal(s);
				s.normalize(); // sample from unit hypersphere surface

				// 5x larger radius than the largest variance [to keep noise separated]
				s *= 5.0*math::sqrt(var[var.size()-1]);

				// adds variance
				for(unsigned int d=0;d<DIMENSION;d++)
					s[d] = s[d] + rng.normal()*math::sqrt(var[d]);

				samples.push_back(s);

				m += s;
				for(unsigned int j=0;j<s.size();j++)
					v[j] += s[j]*s[j];
			}

			m /= (double)samples.size();
			v /= (double)samples.size();

			for(unsigned int j=0;j<v.size();j++){
				v[j] -= m[j]*m[j];
				v[j] = sqrt(v[j]);
			}

			// normalizes mean and variance of data to zero
			for(unsigned int s=0;s<samples.size();s++){
			  auto x = samples[s];
			  x -= m;
			  
			  for(unsigned int i=0;i<x.size();i++)
			    x[i] /= v[i];
			  
			  samples[s] = x;
			}
			
		}

		std::cout << "Learning RBM parameters.." << std::endl;
		math::vertex< math::blas_real<double> > best_variance;
		math::vertex< math::blas_real<double> > best_q;
		std::vector< math::vertex< math::blas_real<double> > > qsamples;

#if 1
		// tests HMC sampling
		{
			// adaptive step length

#if 1
			whiteice::HMC_GBRBM< math::blas_real<double> > hmc(samples, 50, true, true);
			hmc.setTemperature(1.0);
#else
			whiteice::PTHMC_GBRBM< math::blas_real<double> > hmc(10, samples, 10*DIMENSION, true);
#endif
			auto start = std::chrono::system_clock::now();
			hmc.startSampler();

			std::vector< math::vertex< math::blas_real<double> > > starting_data;
			std::vector< math::vertex< math::blas_real<double> > > current_data;
			math::blas_real<double> min_error = INFINITY;
			int last_checked_index = 0;

			std::cout << "HMC-GBRBM sampling.." << std::endl;

			while(hmc.getNumberOfSamples() < 2000){
				sleep(1);				
				if(hmc.getNumberOfSamples() > 0){
					std::vector< math::vertex< math::blas_real<double> > > qsamples;

#if 1
					hmc.getSamples(qsamples);

					// calculate something using the samples..
					// [get the most recent sample and try to calculate log probability]

					if(qsamples.size()-last_checked_index > 5){
					        std::cout << "HMC-GBRBM number of samples: " << hmc.getNumberOfSamples() << std::endl;
					  
						// hmc.getRBM().sample(1000, current_data, samples);
						math::blas_real<double> mean = 0.0;

						last_checked_index = qsamples.size() - 10;
						if(last_checked_index < 0) last_checked_index = 0;

						GBRBM< math::blas_real<double > > rbm = hmc.getRBM();

						math::vertex< math::blas_real<double> > var;
						rbm.setParametersQ(qsamples[qsamples.size()-1]);
						rbm.getVariance(var);

						math::blas_real<double> meanvar = 0.0;

						for(unsigned int i=0;i<var.size();i++)
							meanvar += var[i];
						meanvar /= var.size();

						std::cout << "mean(var) = " << meanvar << std::endl;

						for(int s=last_checked_index;s < (signed)qsamples.size();s++){
							GBRBM< math::blas_real<double> > local_rbm = rbm;
							local_rbm.setParametersQ(qsamples[s]);
							auto e = local_rbm.reconstructError(samples);
							{
								if(e < min_error){
									min_error = e;
									best_q = qsamples[s];
								}

								mean += e;
							}
						}

						mean /= (qsamples.size() - last_checked_index);

						last_checked_index = qsamples.size();

						std::cout << "HMC-GBRBM E[error]:   " << mean << std::endl;
						std::cout << "HMC-GBRBM min(error): " << min_error << std::endl;
						//std::cout << "PTHMC-GBRBM temps: " << hmc.getNumberOfTemperatures() << std::endl;
						//std::cout << "PTHMC-GBRBM arate: " << 100.0*hmc.getAcceptRate() << "%" << std::endl;
					}
#endif
				}

				auto end = std::chrono::system_clock::now();
				auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			}

			hmc.stopSampler();

			qsamples.clear();

			hmc.getSamples(qsamples);
		}
#endif

		
		whiteice::GBRBM< math::blas_real<double> > rbm;
		rbm.resize(DIMENSION, 20); // 20 hidden units..
		// rbm.setVariance(best_variance);
		
		rbm.learnWeights(samples, 100, true);
		// rbm.setParametersQ(best_q);

		auto rdata = samples;
		rdata.clear();
		
		// samples directly from P(v) [using AIS - annealed importance sampling. This probably DO NOT WORK PROPERLY.]
		rbm.sample(1000, rdata, samples); 

		std::cout << "Storing reconstruct (circle) results to disk.." << std::endl;

		saveSamples("gbrbm_sphere_input.txt", samples);
		saveSamples("gbrbm_sphere_output.txt", rdata);

		rdata = samples;
		// rbm.reconstructDataBayesQ(rdata, qsamples);	 // calculates reconstruction v -> h -> v

		rbm.reconstructData(rdata);

		saveSamples("gbrbm_sphere_output2.txt", rdata);
	}
#endif

	
	// generates another test/learning example
	std::vector< math::vertex< math::blas_real<double> > > samples;

	const unsigned int NPOINTS = 5;
	const unsigned int DIMENSION = 2;
	const unsigned int NSAMPLES = 10000; // 10.000 samples

	{
		std::cout << "Generating d-dimensional curve dataset for learning.." << std::endl;

		// pick N random points from DIMENSION dimensional space [-10,10]^D (initially N=5, DIMENSION=2)
		// calculate hermite curve interpolation between N points and generate data
		// add random noise N(0,1) to the spline to generate distribution
		// target is then to learn spline curve distribution

		{
			whiteice::math::hermite< math::vertex< math::blas_real<double> >, math::blas_real<double> > curve;

		        whiteice::RNG< math::blas_real<double> > rng;

			std::vector< math::vertex< math::blas_real<double> > > points;
			points.resize(NPOINTS);

			for(auto& p : points){
				p.resize(DIMENSION);
				for(unsigned int d=0;d<DIMENSION;d++){
					p[d] = rng.uniform()*20.0f - 10.0f; // [-10,10]
				}

			}

			// hand-selected point data to create interesting looking "8" figure
			double pdata[5][2] = { {  4.132230, -8.69549 },
					       { -0.201683, -4.68169 },
					       { -1.406790, +3.38615 },
					       { +7.191170, +7.73804 },
					       { -1.802390, -7.46397 }
			};

			for(unsigned int p=0;p<5;p++){
				for(unsigned int d=0;d<2;d++){
					points[p][d] = pdata[p][d];
				}
			}

			curve.calculate(points, (int)NSAMPLES);

			for(unsigned s=0;s<NSAMPLES;s++){
				auto& m = curve[s];
				auto  n = m;
				rng.normal(n);
				math::blas_real<double> stdev = 0.5;
				n = m + n*stdev;

				samples.push_back(n);
			}
		}


		{
		  // normalizes mean and variance for each dimension
		  math::vertex< math::blas_real<double> > m, v;
		  m.resize(DIMENSION);
		  v.resize(DIMENSION);
		  m.zero();
		  v.zero();
		  
		  for(unsigned int s=0;s<NSAMPLES;s++){
		    auto x = samples[s];
		    m += x;

		    for(unsigned int i=0;i<x.size();i++)
		      v[i] += x[i]*x[i];
		  }

		  m /= NSAMPLES;
		  v /= NSAMPLES;

		  for(unsigned int i=0;i<m.size();i++){
		    v[i] -= m[i]*m[i];
		    v[i] = sqrt(v[i]); // st.dev.
		  }

		  // normalizes mean to be zero and st.dev. / variance to be one
		  for(unsigned int s=0;s<NSAMPLES;s++){
		    auto x = samples[s];
		    x -= m;

		    for(unsigned int i=0;i<x.size();i++)
		      x[i] /= v[i];

		    samples[s] = x;
		  }
		}
		
		// once data has been generated saves it to disk for inspection that is has been generated correctly
		// stores the results into file for inspection
		saveSamples("splinecurve.txt", samples);
	}


	// after generating samples train it using HMC GBRBM sampler ~ p(gbrbm_params|data)
	std::cout << "Learning RBM parameters.." << std::endl;
	std::vector< math::vertex< math::blas_real<double> > > qsamples;
	math::vertex< math::blas_real<double> > best_q; // minimum single reconstruction error case..

	const unsigned int HIDDEN_NODES = 50; // was 200
	
#if 0
	const unsigned int ITERLIMIT = 100;

	{
		math::vertex< math::blas_real<double> > best_variance;

		whiteice::HMC_GBRBM< math::blas_real<double> > hmc(samples, HIDDEN_NODES, true, true);
		hmc.setTemperature(1.0);

		hmc.startSampler();

		std::vector< math::vertex< math::blas_real<double> > > starting_data;
		std::vector< math::vertex< math::blas_real<double> > > current_data;
		math::blas_real<double> min_error = INFINITY;
		int last_checked_index = 0;

		while(hmc.getNumberOfSamples() < ITERLIMIT){
			sleep(1);
			std::cout << "HMC-GBRBM number of samples: " << hmc.getNumberOfSamples() << std::endl;
			if(hmc.getNumberOfSamples() > 0){
				std::vector< math::vertex< math::blas_real<double> > > qsamples;

#if 1
				hmc.getSamples(qsamples);

				// calculate something using the samples..

				if(qsamples.size()-last_checked_index > 5){
					math::blas_real<double> mean = 0.0;

					last_checked_index = qsamples.size() - 10;
					if(last_checked_index < 0) last_checked_index = 0;

					GBRBM< math::blas_real<double> > rbm = hmc.getRBM();

					math::vertex< math::blas_real<double> > var;
					rbm.setParametersQ(qsamples[qsamples.size()-1]);
					rbm.getVariance(var);

					math::blas_real<double> meanvar = 0.0;

					for(unsigned int i=0;i<var.size();i++)
						meanvar += var[i];
					meanvar /= var.size();

					std::cout << "mean(var) = " << meanvar << std::endl;

					for(int s=last_checked_index;s < (signed)qsamples.size();s++){
						GBRBM< math::blas_real<double> > local_rbm = rbm;
						local_rbm.setParametersQ(qsamples[s]);
						auto e = local_rbm.reconstructError(samples);
						{
							if(e < min_error){
								min_error = e;
								best_q = qsamples[s];
							}

							mean += e;
						}
					}

					mean /= (qsamples.size() - last_checked_index);

					last_checked_index = qsamples.size();

					std::cout << "HMC-GBRBM E[error]:   " << mean << std::endl;
					std::cout << "HMC-GBRBM min(error): " << min_error << std::endl;
				}
#endif
			}

		}

		hmc.stopSampler();

		qsamples.clear();

		auto temp_all_samples = qsamples;
		hmc.getSamples(temp_all_samples);

		for(unsigned int i=(temp_all_samples.size()/2);i<temp_all_samples.size();i++)
			qsamples.push_back(temp_all_samples[i]);
	}
#endif


	// now we have training data samples and parameters q from the RBM sampling learning
	// we keep ONLY variance from the sampling and then optimize (learn) parameters of basic RBM
	{
		whiteice::GBRBM< math::blas_real<double> > rbm;
		math::vertex< math::blas_real<double> > best_variance;
		best_variance.resize(DIMENSION);

		// sets initial variance to be one (according to our normalization)
		for(unsigned int i=0;i<best_variance.size();i++){
		  best_variance[i] = 1.00;
		}

		rbm.resize(DIMENSION, HIDDEN_NODES);
		rbm.setVariance(best_variance);

		auto rdata = samples;

		// rbm.setParametersQ(best_q);
		rbm.learnWeights(samples, 150, true);
		rdata.clear();

		// samples directly from P(v) [using AIS - annealed importance sampling. This probably DO NOT WORK PROPERLY.]
		rbm.sample(1000, rdata, samples);
		

		std::cout << "Storing reconstruct results to disk.." << std::endl;

		// stores the results into file for inspection

		saveSamples("gbrbm_input.txt", samples);
		saveSamples("gbrbm_output.txt", rdata);
		
		rdata = samples;
		// rbm.reconstructDataBayesQ(rdata, qsamples);	 // calculates reconstruction v -> h -> v
		rbm.reconstructData(rdata);

		saveSamples("gbrbm_output2.txt", rdata);
	}

	return;



  std::cout << "RBM TESTS" << std::endl;
  
  // saves and loads machine to and from disk and checks configuration is correct
  {
    whiteice::RBM< math::blas_real<double> > machine(100,50);
    whiteice::RBM< math::blas_real<double> > machine2 = machine;
    
    std::cout << "RBM LOAD/SAVE TEST" << std::endl;
      
    if(machine.save("rbmtest.cfg") == false){
      std::cout << "ERROR: saving RBM failed." << std::endl;
      return;
    }
    
    if(machine.load("rbmtest.cfg") == false){
      std::cout << "ERROR: loading RBM failed." << std::endl;
      return;
    }
    
    // compares machine and machine2
    math::matrix< math::blas_real<double> > W1, W2;
    W1 = machine.getWeights();
    W2 = machine2.getWeights();
    W1 = W2 - W1;
    
    math::blas_real<double> e = 0.0001;
    
    if(frobenius_norm(W1) > e){
      std::cout << "ERROR: loaded RBM machine have different weights." 
		<< std::endl;
      return;
    }
    
    std::cout << "RBM LOAD/SAVE TEST OK." << std::endl;
  }


}


void simple_rbm_test()
{
  // creates a toy problems and check that results seem to make sense
  {
    std::cout << "RBM TOY PROBLEM TEST" << std::endl;
    
    const unsigned int H = 2;
    const unsigned int V = 2*H;
    
    std::vector< math::vertex<> > samples;
    
    for(unsigned int i=0;i<1000;i++){
      math::vertex<> h;
      math::vertex<> v;
      h.resize(H);
      v.resize(V);
      
      for(unsigned int j=0;j<H;j++){
	float f = (float)(rand() & 1); 
	h[j] = f; // 0, 1 valued vector
      }
      
      // calculates visible vector v from h
      for(unsigned int j=0;j<V;j++){
	float f = (float)(rand() & 1);
	f = 1.0f; // no noise mask..
	v[j] = f * h[j/2]; // 0, 1 valued vector [noise masking]
      }
      
      samples.push_back(v);
      
      // std::cout << "h = " << h << std::endl;
      // std::cout << "v = " << v << std::endl;
    }
    
    std::cout << "RBM TOY PROBLEM GENERATION OK." << std::endl;
    
    std::cout << "RBM LEARNING TOY PROBLEM.." << std::endl;
    
    whiteice::RBM<> machine(V, H);
    
    math::blas_real<float> delta;
    math::blas_real<float> elimit = 0.005;
    unsigned int epochs = 0;
    
    do{
      delta = machine.learnWeights(samples);
      epochs++;
      
      std::cout << "RBM learning epoch " << epochs 
		<< " deltaW = " << delta << std::endl;
    }
    while(delta > elimit && epochs < 10000);

    std::cout << "RBM LEARNING TOY PROBLEM.. DONE." << std::endl;
    
    std::cout << "W  = " << machine.getWeights() << std::endl;
    std::cout << "Wt = " << machine.getWeights().transpose() << std::endl;


    std::cout << "BBRBM LEARNING TOY PROBLEM.." << std::endl;
    {
      whiteice::BBRBM< math::blas_real<float> > bbrbm;
      bbrbm.resize(V, H);

      auto error = bbrbm.learnWeights(samples, 50, true);

      std::cout << "BBRMB LEARNING TOY PROBLEM.. DONE" << std::endl;
      std::cout << "W = " << bbrbm.getWeights() << std::endl;
      std::cout << "b = " << bbrbm.getBValue() << std::endl;
      std::cout << "a = " << bbrbm.getAValue() << std::endl;
	
    }
    
    
#if 0
    std::cout << "DBN LEARNING TOY PROBLEM.." << std::endl;

    std::vector<unsigned int> arch;
    whiteice::DBN<> dbn;
    
    arch.push_back(V);
    arch.push_back(2*V);
    arch.push_back(4*V);
    arch.push_back(H);
    
    dbn.resize(arch);
    
    if(dbn.learnWeights(samples, elimit) == false)
      std::cout << "Training DBN failed." << std::endl;

    std::cout << "DBN LEARNING TOY PROBLEM.. DONE." << std::endl;
#endif

    
  }
  
  
  // test continuous RBM
  {
    std::cout << "CONTINUOUS RBM TOY PROBLEM TESTING" << std::endl;
    
    
    // creates a toy problem: creates randomly data from three 2d clusters with given 
    std::cout << "GENERATING CONTINUOUS DATA FOR C-RBM..." << std::endl;
    
    std::vector< math::vertex<> > samples;
    {
      math::vertex<> m1, m2, m3, m4;     // mean probabilities of data (same probability for each cluster)
      math::blas_real<float> dev = 0.2f; // "standard deviation of cluster data"
      
      m1.resize(2); m1[0] = 0.5f; m1[1] = 0.5f;
      m2.resize(2); m2[0] = 0.1f; m2[1] = 0.1f;
      m3.resize(2); m3[0] = 0.1f; m3[1] = 0.9f;
      m4.resize(2); m4[0] = 0.9f; m4[1] = 0.9f;
      
      for(unsigned int i=0;i<1000;){
	unsigned int r = rand()%4;
	
	math::vertex<> d;
	
	if(r == 0)      d = m1;
	else if(r == 1) d = m2;
	else if(r == 2) d = m3;
	else if(r == 3) d = m4;
	
	math::vertex<> var;
	var.resize(2);
	var[0] = dev*(((float)rand()/(float)RAND_MAX) - 0.5f);
	var[1] = dev*(((float)rand()/(float)RAND_MAX) - 0.5f);
	
	d += var;

	samples.push_back(d); // adds a single data point from cluster
	  
	i++;
      }
    }
    
    std::cout << "GENERATING CONTINUOUS DATA FOR C-RBM... OK." << std::endl;

    
    // now trains 
    std::cout << "CRBM TRAINING: TOY PROBLEM 2" << std::endl;
    
    whiteice::CRBM<> machine(2, 8);
    {
            
      math::blas_real<float> delta;
      math::blas_real<float> elimit = 0.0001;
      unsigned int epochs = 0;
      
      do{
	delta = machine.learnWeights(samples);
	epochs++;
	
	std::cout << "CRBM learning epoch " << epochs 
		  << " deltaW = " << delta << std::endl;
      }
      while(delta > elimit && epochs < 1000);
      
      std::cout << "CRBM LEARNING TOY PROBLEM 2.. DONE." << std::endl;
      
      std::cout << "W  = " << machine.getWeights() << std::endl;
      std::cout << "Wt = " << machine.getWeights().transpose() << std::endl;      
    }

    
    // TODO: test recontruction of random datapoints using CD-10 to the target
    std::cout << "CRBM RECONSTRUCT AND STORING RESULTS TO CSV FILES" << std::endl;
    
    std::vector< math::vertex<> > reconstruct;
    {
      // reconstruct data points
      for(unsigned int i=0;i<samples.size();i++){
	math::vertex<> v;
	v.resize(2);
	
	// randomly generated point [0,1]x[0,1] interval to be reconstructed
	v[0] = ((float)rand()/(float)RAND_MAX);
	v[1] = ((float)rand()/(float)RAND_MAX);
	
	if(machine.setVisible(v) == false)
	  std::cout << "WARN: setVisible() failed." << std::endl;
	
	if(machine.reconstructData(20) == false) // CD-10
	  std::cout << "WARN: reconstructData() failed." << std::endl;
	
	v = machine.getVisible();
	
	reconstruct.push_back(v);
      }
      
      // saves training data to CSV file for analysis and plotting purposes
      FILE* handle = fopen("rbm_inputdata.csv", "wt"); // no error checking here
      
      for(unsigned int i=0;i<samples.size();i++){
	const math::vertex<>& v = samples[i];
	
	fprintf(handle, "%f", v[0].c[0]);
	for(unsigned int j=1;j<v.size();j++)
	  fprintf(handle, ",%f", v[j].c[0]);
	fprintf(handle, "\n");
      }
      
      fclose(handle);

      
      // saves reconstructed C-RBM CD-10 data to CSV file for analysis and plotting purposes
      handle = fopen("rbm_reconstruct.csv", "wt"); // no error checking here
      
      for(unsigned int i=0;i<reconstruct.size();i++){
	math::vertex<>& v = reconstruct[i];
	
	fprintf(handle, "%f", v[0].c[0]);
	for(unsigned int j=1;j<v.size();j++)
	  fprintf(handle, ",%f", v[j].c[0]);
	fprintf(handle, "\n");
      }
      
      fclose(handle);
    }
    
  }
  std::cout << "CRBM RECONSTRUCT AND STORING RESULTS TO CSV FILES.. DONE." << std::endl;
  
  
  
  std::cout << "RBM TESTS DONE." << std::endl;
}

/************************************************************/


void createHermiteCurve(std::vector< math::vertex< math::blas_real<double> > >& samples,
			const unsigned int NPOINTS, const unsigned int DIMENSION,
			const unsigned int NSAMPLES)
{
  // const unsigned int NPOINTS = 5;
  // const unsigned int DIMENSION = 2;
  // const unsigned int NSAMPLES = 10000; // 10.000 samples
	
  {
    std::cout << "Generating d-dimensional curve dataset for learning.." << std::endl;
    
    // pick N random points from DIMENSION dimensional space [-10,10]^D (initially N=5, DIMENSION=2)
    // calculate hermite curve interpolation between N points and generate data
    // add random noise N(0,1) to the spline to generate distribution
    // target is then to learn spline curve distribution
    
    {
      whiteice::math::hermite< math::vertex< math::blas_real<double> >, math::blas_real<double> > curve;
      
      whiteice::RNG< math::blas_real<double> > rng;
      
      std::vector< math::vertex< math::blas_real<double> > > points;
      points.resize(NPOINTS);
      
      for(auto& p : points){
	p.resize(DIMENSION);
	for(unsigned int d=0;d<DIMENSION;d++){
	  p[d] = rng.uniform()*20.0f - 10.0f; // [-10,10]
	}
	
      }

#if 0
      // hand-selected point data to create interesting looking "8" figure
      double pdata[5][2] = { {  4.132230, -8.69549 },
			     { -0.201683, -4.68169 },
			     { -1.406790, +3.38615 },
			     { +7.191170, +7.73804 },
			     { -1.802390, -7.46397 }
      };
      
      for(unsigned int p=0;p<5;p++){
	for(unsigned int d=0;d<2;d++){
	  points[p][d] = pdata[p][d];
	}
      }
#endif
      
      curve.calculate(points, (int)NSAMPLES);
      
      for(unsigned s=0;s<NSAMPLES;s++){
	auto& m = curve[s];
	auto  n = m;
	rng.normal(n);
	math::blas_real<double> stdev = 0.5;
	n = m + n*stdev;
	
	samples.push_back(n);
      }
    }
    
    
    {
      // normalizes mean and variance for each dimension
      math::vertex< math::blas_real<double> > m, v;
      m.resize(DIMENSION);
      v.resize(DIMENSION);
      m.zero();
      v.zero();
      
      for(unsigned int s=0;s<NSAMPLES;s++){
	auto x = samples[s];
	m += x;
	
	for(unsigned int i=0;i<x.size();i++)
	  v[i] += x[i]*x[i];
      }
      
      m /= NSAMPLES;
      v /= NSAMPLES;
      
      for(unsigned int i=0;i<m.size();i++){
	v[i] -= m[i]*m[i];
	v[i] = sqrt(v[i]); // st.dev.
      }
      
      // normalizes mean to be zero and st.dev. / variance to be one
      for(unsigned int s=0;s<NSAMPLES;s++){
	auto x = samples[s];
	x -= m;
	
	for(unsigned int i=0;i<x.size();i++)
	  x[i] /= v[i];
	
	samples[s] = x;
      }
    }

    saveSamples("splinecurve.txt", samples);
  }
  
}

  
bool saveSamples(const std::string& filename, std::vector< math::vertex< math::blas_real<double> > >& samples)
{
  // once data has been generated saves it to disk for inspection that is has been generated correctly
  // stores the results into file for inspection
  FILE* fp = fopen(filename.c_str(), "wt");

  for(unsigned int i=0;i<samples.size();i++){
    for(unsigned int n=0;n<samples[i].size();n++)
      fprintf(fp, "%f ", samples[i][n].c[0]);
    fprintf(fp, "\n");
  }
  
  fclose(fp);

  return true;
}


/************************************************************/

void rechcprint_test(hcnode< GDAParams, math::blas_real<float> >* node,
		     unsigned int depth);


void gda_clustering_test()
{
  std::cout << "GDA CLUSTERING TEST" << std::endl;
  
  {
    whiteice::HC<GDAParams, math::blas_real<float> > hc;
    whiteice::GDALogic behaviour;
    
    std::vector< math::vertex< math::blas_real<float> > > data;
    
    // creates test data
    {
      std::vector< math::vertex< math::blas_real<float> > > means;
      std::vector< unsigned int > sizes; // variance of all clusters is same
      
      math::vertex< math::blas_real<float> > v(2);
      v[0] =  3.0f; v[1] =  7.5f; means.push_back(v); sizes.push_back(20);
      v[0] =  2.0f; v[1] =  5.0f; means.push_back(v); sizes.push_back(10);
      v[0] =  3.0f; v[1] =  5.0f; means.push_back(v); sizes.push_back(10);
      v[0] =  4.0f; v[1] =  5.0f; means.push_back(v); sizes.push_back(10);
      v[0] =  1.5f; v[1] =  1.5f; means.push_back(v); sizes.push_back(20);
      v[0] =  6.0f; v[1] =  5.0f; means.push_back(v); sizes.push_back(10);
      v[0] =  8.0f; v[1] =  6.0f; means.push_back(v); sizes.push_back(20);
      v[0] =  9.0f; v[1] =  6.0f; means.push_back(v); sizes.push_back(10);
      v[0] = 10.0f; v[1] =  4.0f; means.push_back(v); sizes.push_back(10);
      
      for(unsigned int j=0;j<sizes.size();j++){
	for(unsigned int i=0;i<sizes[j];i++){
	  v = means[j];
	  
	  v[0] += ((rand()/((float)RAND_MAX)) - 0.5f)*3.0f;
	  v[1] += ((rand()/((float)RAND_MAX)) - 0.5f)*3.0f;
	  
	  data.push_back(v);
	}
      }
      
    }
    
    
    hc.setProgramLogic(&behaviour);
    hc.setData(&data);
    
    if(hc.clusterize() == false){
      std::cout << "ERROR: CLUSTERING FAILED" << std::endl;
      return;
    }
    
    if(hc.getNodes().size() != 1){
      std::cout << "ERROR: INCORRECT NUMBER OF ROOT NODES" << std::endl;
      return;
    }
    
    std::cout << "FORM OF THE RESULTS IS CORRECT." << std::endl;
    std::cout << "PRINTING CLUSTERING RESULTS." << std::endl;
    
    rechcprint_test(hc.getNodes()[0], 1);
    
    std::cout << "CLUSTERING LIST PRINTED." << std::endl;
    std::cout << std::endl;
  }
}


void rechcprint_test(hcnode< GDAParams, math::blas_real<float> >* node,
		     unsigned int depth)
{
  std::cout << "NODE " << std::hex << node 
	    << std::dec << " AT DEPTH " << depth << std::endl;
  std::cout << node->p.n << " POINTS  MEAN: " << node->p.mean << std::endl;

  if(node->data)
    std::cout << "LEAF NODE." << std::endl;
  
  if(node->childs.size()){
    std::cout << "CHILD NODES: ";
    std::cout << std::hex;
    
    for(unsigned int i=0;i<node->childs.size();i++)
      std::cout << node->childs[i] << " ";
    
    std::cout << std::dec << std::endl;
  }
  else{
    std::cout << "NO CHILDS (BUG?)" << std::endl;
  }
  
  std::cout << std::endl;
  
  
  for(unsigned int i=0;i<node->childs.size();i++)
    rechcprint_test(node->childs[i], depth+1);
}


/************************************************************/
#if 0
void compressed_neuralnetwork_test()
{
  std::cout << "COMPRESSED NEURAL NETWORK TEST" << std::endl;
  
  using namespace whiteice::math;
  
  
  // first test - compare values are same with compressed and
  // uncompressed neural network
  {
    neuralnetwork< blas_real<float> >* nn;
    neuralnetwork< blas_real<float> >* cnn;
    
    std::vector<unsigned int> nn_arch;
    nn_arch.push_back(2);
    nn_arch.push_back(100);
    nn_arch.push_back(100);
    nn_arch.push_back(2);
    
    nn = new neuralnetwork< blas_real<float> >(nn_arch, false);
    nn->randomize();
    cnn = new neuralnetwork< blas_real<float> >(*nn);
    
    if(cnn->compress() == false){
      std::cout << "neural network compression failed"
		<< std::endl;
      
      delete nn;
      delete cnn;
      return;
    }
    
    
    // tests operation
    for(unsigned int i=0;i<10;i++){
      
      for(unsigned int d=0;d<nn->input().size();d++){
	nn->input()[d] = (rand() / ((float)RAND_MAX));
	cnn->input()[d] = nn->input()[d];
      }
      
      if(nn->calculate() == false ||
	 cnn->calculate() == false){
	std::cout << "neural network activation failed"
		  << std::endl;
	delete nn;
	delete cnn;
	return;
      }
      
      for(unsigned int d=0;d<nn->output().size();d++){
	if(nn->output()[d] != cnn->output()[d]){
	  std::cout << "nn and compressed nn outputs differ"
		    << "(dimension " << d << " )"
		    << nn->output()[d] << " != " 
		    << cnn->output()[d] << std::endl;
	  
	  delete nn;
	  delete cnn;
	  return;
	}
      }
    }
    
    
    delete nn;
    delete cnn;
    
  }
  
  
  // second test - compression rate of taught neural network
  // (compression ratio of random network >= 1)
  {
    neuralnetwork< blas_real<float> >* nn;
    neuralnetwork< blas_real<float> >* cnn;
    
    std::vector<unsigned int> nn_arch;
    nn_arch.push_back(2);
    nn_arch.push_back(100);
    nn_arch.push_back(100);
    nn_arch.push_back(2);
    
    nn = new neuralnetwork< blas_real<float> >(nn_arch, false);
    
    nn->randomize();
    
    //////////////////////////////////////////////////
    // teaches nn
    
    dataset< blas_real<float> >  in(2), out(2);
    
    // creates data
    {
      const unsigned int SIZE = 1000;
      std::vector< math::vertex<math::blas_real<float> > > input;
      std::vector< math::vertex<math::blas_real<float> > > output;
    
      input.resize(SIZE);
      output.resize(SIZE);
      
      for(unsigned int i=0;i<SIZE;i++){
	input[i].resize(2);
	output[i].resize(2);
	
	input[i][0] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
	input[i][1] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
      
	output[i][0] = input[i][0] - 0.2f * input[i][1];
	output[i][1] = -0.12f * input[i][0] + 0.1f * input[i][1];
      }
      
      if(in.add(input) == false || out.add(output) == false){
	std::cout << "data creation failed" << std::endl;
	return;
      }
      
      in.preprocess();
      out.preprocess();
    }
    
    
    // pso learner
    {
      nnPSO< math::blas_real<float> >* nnoptimizer;
      nnoptimizer = 
	new nnPSO<math::blas_real<float> >(nn, &in, &out, 20);

      nnoptimizer->improve(25);
      std::cout << "learnt nn error: " << nnoptimizer->getError()
		<< std::endl;
      
      std::cout << "random sample: "
		<< nnoptimizer->sample() << std::endl;
	
      
      delete nnoptimizer;
    }
    
    //////////////////////////////////////////////////
    
    cnn = new neuralnetwork< blas_real<float> >(*nn);
    
    if(cnn->compress() == false){
      std::cout << "neural network compression failed"
		<< std::endl;
      
      delete nn;
      delete cnn;
      return;
    }
    
    
    // shows compression ratio

    std::cout << "NN COMPRESSION RATIO:" 
	      <<  cnn->ratio() << std::endl;
    
    delete nn;
    delete cnn;
  }
  
  
}
#endif


void simple_dataset_test()
{
  try{
    std::cout << "DATASET PREPROCESS TEST" 
	      << std::endl;
    
    math::blas_real<float> a;
    
    math::vertex< math::blas_real<float> > v[2];
    std::vector< math::vertex<math::blas_real<float> > > data;
    dataset< math::blas_real<float> > set(2);
    
    data.resize(2);
    data[0].resize(2);
    data[1].resize(2);
    v[0].resize(2);
    v[1].resize(2);
    
    std::cout << "OLD A " << a << std::endl;    
    a = 1.1;
    std::cout << "    A " << a << std::endl;    
    
    std::cout << "OLD DATA1: " << data[0] << std::endl;
    std::cout << "OLD DATA2: " << data[1] << std::endl;
    std::cout << "DATA1 SIZE: " << data[0].size() << std::endl;
    std::cout << "DATA2 SIZE: " << data[1].size() << std::endl;
    
    std::cout << "OLD V1: " << v[0] << std::endl;
    std::cout << "OLD V2: " << v[1] << std::endl;
    
    v[0][0] = 1.1;
    v[0][1] = 2.1;
    v[1][0] = -1.3;
    v[1][1] = 2.7;
    
    std::cout << "V1: " << v[0] << std::endl;
    std::cout << "V2: " << v[1] << std::endl;
    
    std::cout << "v[0] =";
    for(unsigned int i=0;i<v[0].size();i++)
      std::cout << " " << v[0][i];
    std::cout << std::endl;
    
    data[0] = v[0];
    data[1] = v[1];
    
    std::cout << "DATA1: " << data[0] << std::endl;
    std::cout << "DATA2: " << data[1] << std::endl;
    std::cout << "DATA1.0: " << data[0][0] << std::endl;
    std::cout << "DATA2.1: " << data[1][1] << std::endl;
    

    set.add(data);
    set.preprocess();

    
    std::cout << "DATASET PREPROCESS TESTS SUCCESFULLY DONE"
	      << std::endl;
    
    
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception " 
	      << e.what() << std::endl;
    return;
  }
}



#if 0

void neuralnetwork_pso_test()
{
  try{
    std::cout << "NEURAL NETWORK PARTICLE SWARM OPTIMIZER" << std::endl;
    
    neuralnetwork< math::blas_real<float> >* nn;
    nnPSO< math::blas_real<float> >* nnoptimizer;
    dataset< math::blas_real<float> > I1(2), O1(1);
    
    const unsigned int SIZE = 10000;
    
    std::vector< math::vertex<math::blas_real<float> > > input;
    std::vector< math::vertex<math::blas_real<float> > > output;
    
    // creates data
    {
      input.resize(SIZE);
      output.resize(SIZE);
      
      for(unsigned int i=0;i<SIZE;i++){
	input[i].resize(2);
	output[i].resize(1);
	
	input[i][0] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
	input[i][1] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
	
	output[i][0] = input[i][0] - 0.2f*input[i][1];
	// output[i][1] = -0.12f*input[i][0] + 0.1f*input[i][1];
      }
      
      
      for(unsigned int i=0;i<SIZE;i++){
	if(I1.add(input[i]) == false){
	  std::cout << "dataset creation failed [1]\n";
	  return;
	}
	
	if(O1.add(output[i]) == false){
	  std::cout << "dataset creation failed [2]\n";
	  return;
	}
      }
    }
    
    
    I1.preprocess(dataset< math::blas_real<float> >::dnMeanVarianceNormalization);
    O1.preprocess(dataset< math::blas_real<float> >::dnMeanVarianceNormalization);
    
    
    std::vector<unsigned int> nn_arch;
    nn_arch.push_back(2);
    nn_arch.push_back(10);
    nn_arch.push_back(1);
    
    nn = new neuralnetwork< math::blas_real<float> >(nn_arch);
    nn->randomize();
    
    // nn arch
    {
      std::cout << "NN-ARCH = ";
      
      for(unsigned int i=0;i<nn_arch.size();i++)
	std::cout << nn_arch[i] << " ";
      
      std::cout << std::endl;
    }
    
    const unsigned int SWARM_SIZE = 20;
    const unsigned int STEP = 10;
    
    nnoptimizer = 
      new nnPSO<math::blas_real<float> >
      (nn, &I1, &O1, SWARM_SIZE);
    
    nnoptimizer->verbosity(true);
    
    std::cout << "PSO OPTIMIZATION START" << std::endl;
    std::cout << "SWARM SIZE: " << SWARM_SIZE << std::endl;
    
    for(unsigned int i=0;i<(1000/STEP);i++){
      nnoptimizer->improve(STEP);
      
      std::cout << (i+1)*10 << ": " 
		<< nnoptimizer->getError() << std::endl;
    }
    
    
    delete nn;
    delete nnoptimizer;
    
    return;
  }
  catch(std::exception& e){
    return;
  }
}



void neuralnetwork_saveload_test()
{
  try{
    std::cout << "NEURAL NETWORK SAVE & LOAD TEST" << std::endl;
    
    // NOTE: saving and loading tests may 'fail' due to lack of precision
    // in floating point presentation if T type uses more accurate
    // representation
    
    
    // create random networks - save & load them and check that
    // NN is same after load
    
    neuralnetwork< math::blas_real<float> >* nn[2];
    std::vector<unsigned int> s;
    
    std::string file = "nnarch.cfg";
    
    for(unsigned int i=0;i<10;i++){
      
      // makes neuro arch
      {
	s.clear();
	unsigned int k = (rand() % 10) + 2;
	
	do{
	  s.push_back((rand() % 4) + 2);
	  k--;
	}
	while(k > 0);
      }
      
      
      nn[0] = new neuralnetwork< math::blas_real<float> >(s);
      nn[1] = new neuralnetwork< math::blas_real<float> >(2, 2);
      
      // set random values of neural network
      nn[0]->randomize();
      
      for(unsigned int j=0;j<nn[0]->length();j++){
	(*nn[0])[j].moment() = 
	  math::blas_real<float>(((float)rand()) / ((float)RAND_MAX));
	
	(*nn[0])[j].learning_rate() = 
	  math::blas_real<float>(((float)rand()) / ((float)RAND_MAX));
	
	math::vertex<math::blas_real<float> >& bb =
	  (*nn[0])[j].bias();
	
	math::matrix<math::blas_real<float> >& MM =
	  (*nn[0])[j].weights();
	
	for(unsigned int k=0;k<bb.size();k++)
	  bb[k] = (rand() / ((float)RAND_MAX))*100.0;
	
	for(unsigned int k=0;k<MM.ysize();k++)
	  for(unsigned int l=0;l<MM.xsize();l++)
	    MM(k,l) = (rand() / ((float)RAND_MAX))*50.0;
      }
      
      std::cout << "saving ...\n";
      
      if(!(nn[0]->save(file))){
	std::cout << "neural network save() failed."
		  << std::endl;
	return;
      }
      
      std::cout << "loading ...\n";
      
      if(!(nn[1]->load(file))){
	std::cout << "neural network load() failed."
		  << std::endl;
	return;
      }
      

      // compares values between neural networks
      // (these should be same)
      
      std::cout << "compare: ";
      
      // compares global architecture      
      if(nn[0]->input().size() != nn[1]->input().size()){
	std::cout << "loaded nn: input size differ\n";
	return;
      }
      
      if(nn[0]->output().size() != nn[1]->output().size()){
	std::cout << "loaded nn: output size differ\n";
	return;
      }
      
      if(nn[0]->length() != nn[1]->length()){
	std::cout << "loaded nn: wrong number of layers\n";
	return;
      }
      else
	std::cout << "A";
      
      
      // compares layers
      for(unsigned int j=0;j<nn[1]->length();j++){
	
	if( (*nn[0])[j].input_size() != (*nn[1])[j].input_size() ){
	  std::cout << "loaded nn, layer " << j 
		    << " : input size is wrong" << std::endl;
	  return;
	}
	
	
	if( (*nn[0])[j].size() != (*nn[1])[j].size() ){
	  std::cout << "loaded nn, layer " << j 
		    << " : number of neurons is wrong" << std::endl;
	  return;
	}
	
	
	
	if( typeid( (*nn[0])[j].get_activation() ) != 
	    typeid( (*nn[1])[j].get_activation() ) ){
	  std::cout << "loaded nn, layer " << j 
		    << " : wrong activation function" << std::endl;
	  return;
	}
	
	
	// compares bias
	if( (*nn[0])[j].bias() != (*nn[1])[j].bias() ){
	  std::cout << "loaded nn, layer " << j 
		    << " : incorrect bias" << std::endl;
	  return;
	}
	
	
	// compares weights
	if( (*nn[0])[j].weights() != (*nn[1])[j].weights() ){
	  std::cout << "loaded nn, layer " << j 
		    << " : incorrect weight matrix" << std::endl;
	  return;
	}
	
	// compares moment & learning rate
	if( (*nn[0])[j].moment() != (*nn[1])[j].moment() ){
	  std::cout << "loaded nn, layer " << j 
		    << " : incorrect moment" << std::endl;
	  return;
	}
	
	if( (*nn[0])[j].learning_rate() != (*nn[1])[j].learning_rate() ){
	  std::cout << "loaded nn, layer " << j 
		    << " : incorrect lrate" << std::endl;
	  return;
	}
	else
	  std::cout << "L";
      }            
      
      // checks nn activation works correctly
      nn[0]->calculate();
      nn[1]->calculate();
      
      std::cout << "C" << std::endl;
      
      delete nn[0];
      delete nn[1];
    }
    

    
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
    assert(0);
  }
}
#endif

/************************************************************/

#if 0
void neuronlayer_test()
{
  // checks neuronlayer is ok
  
  std::cout << "NEURONLAYER TEST" << std::endl;
  
  // neuronlayer ctor (input, output, act. func) +
  // correct values + calculations are correct test
  try{    
    
    {
      math::vertex< math::blas_real<float> > input, output;
      neuronlayer< math::blas_real<float> >* l;
      
      odd_sigmoid< math::blas_real<float> > os;
      
      input.resize(10);
      output.resize(5);
      
      l = new neuronlayer< math::blas_real<float> >(&input, &output, os);
      
      if(l->input_size() != 10)
	throw test_exception("neuronlayer input_size() returns wrong value");
      
      if(l->size() != 5)
	throw test_exception("neuronlayer size() returns wrong value");
      
      if(l->input() != &input)
	throw test_exception("neuronlayer input() returns wrong value");
      
      if(l->output() != &output)
	throw test_exception("neuronlayer output() returns wrong value");
      
      // checks calculation with trivial case W = 0, b != 0.
      
      // sets bias and weights
      
      math::matrix< math::blas_real<float> >& W = l->weights();
      math::vertex< math::blas_real<float> >& b = l->bias();
      
      W = math::blas_real<float>(0.0f); // set W
      
      for(unsigned int i=0;i<b.size();i++)
	b[i] = ((float)rand())/((float)RAND_MAX);
      
      std::cout << "bias = " << b << std::endl;
      std::cout << "output 1 = " << output << std::endl;
      if(l->calculate() == false)
	std::cout << "layer->calculate failed." << std::endl;
      std::cout << "output 2 = " << output << std::endl;
      
      // calculates bias = output manually
      for(unsigned int i=0;i<b.size();i++)
	b[i] = os(b[i]);
      
      output -= b;
      
      // calculates mean squared error
      {
	math::blas_real<float> error = 0.0f;
	
	for(unsigned int i=0;i<output.size();i++)
	  error += output[i]*output[i];
	
	if(error > 0.01f){ 
	  std::cout << "cout: errors: " << output << std::endl;
	  throw test_exception("W = 0, b = x. test failed.");
	}
      }
      
      
      delete l;
    }
    
    
    {
      math::vertex< math::blas_real<float> > input, output;
      neuronlayer< math::blas_real<float> >* l;
      odd_sigmoid< math::blas_real<float> > os;
      
      input.resize(10);
      output.resize(10);
      
      l = new neuronlayer< math::blas_real<float> >(&input, &output, os);
      
      math::matrix< math::blas_real<float> >& W = l->weights();
      math::vertex< math::blas_real<float> >& b = l->bias();
    
      // W = I, b = 0. test
      
      for(unsigned int i=0;i<b.size();i++)
	b[i] = 0.0f;
      
      for(unsigned int j=0;j<W.ysize();j++){
	for(unsigned int i=0;i<W.xsize();i++){
	  if(i == j)
	    W(j,i) = 1.0f;
	  else
	    W(j,i) = 0.0f;
	}
      }
      
      
      for(unsigned int i=0;i<input.size();i++)
	input[i] = ((float)rand())/((float)RAND_MAX);
      
      l->calculate(); // output = g(W*x+b) = g(x)
      
      // calculates output itself
      for(unsigned int i=0;i<input.size();i++)
	input[i] = os(input[i]);
      
      output -= input;
      
      {
	math::blas_real<float> error = 0.0f;
	
	for(unsigned int j=0;j<output.size();j++)
	  error += output[j] * output[j];
	
	if(error > 0.01f)
	  throw test_exception("W=I, b = 0. test failed.");
      }
      
      delete l;
    }
    
    
    {
      math::vertex< math::blas_real<float> > input, output;
      neuronlayer< math::blas_real<float> >* l;
      odd_sigmoid< math::blas_real<float> > os;
      
      input.resize((rand() % 13) + 1);
      output.resize((rand() % 13) + 1);
      
      l = new neuronlayer< math::blas_real<float> >(&input, &output, os);
      
      math::matrix< math::blas_real<float> >& W = l->weights();
      math::vertex< math::blas_real<float> >& b = l->bias();
      
      // sets weights and biases
      
      for(unsigned int j=0;j<W.ysize();j++){
	for(unsigned int i=0;i<W.xsize();i++){
	  W(j,i) = ((float)rand())/((float)RAND_MAX);
	}
      }
      
      for(unsigned int i=0;i<b.size();i++)
	b[i] = ((float)rand())/((float)RAND_MAX);
      
      // sets input & output
      
      for(unsigned int i=0;i<input.size();i++)
	input[i] = ((float)rand())/((float)RAND_MAX);
      
      math::vertex< math::blas_real<float> > result(input);
      
      result = W*result;
      result += b;
      
      for(unsigned int i=0;i<result.size();i++)
	result[i] = os(result[i]);
      
      l->calculate();
      
      result -= output;
      
      {
	math::blas_real<float> error = 0.0f;
	
	for(unsigned int i=0;i<result.size();i++)
	  error += result[i] * result[i];
	
	if(error > 0.01f)
	  throw test_exception("W=rand, b = rand. test failed.");
      }
      
      delete l;
    }
    
    
  }
  catch(std::exception& e){
    std::cout << "Error - unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
  // neuronlayer ctor (isize, nsize) and
  // neuronlayer copy ctor tests
  try{

    {
      neuronlayer< math::blas_real<float> >* l[2];
      math::vertex< math::blas_real<float> > input;
      math::vertex< math::blas_real<float> > output;
      
      input.resize(10);
      output.resize(1);
      
      l[0] = new neuronlayer< math::blas_real<float> >(10, 1);
      l[0]->input() = &input;
      l[0]->output() = &output;
      
      if(l[0]->input_size() != 10 || l[0]->size() != 1)
	throw test_exception("neuronlayer (isize, nsize) test failed.");
      
      // sets up biases and weights randomly
      
      math::vertex< math::blas_real<float> >& b = l[0]->bias();
      math::matrix< math::blas_real<float> >& W = l[0]->weights();
      
      for(unsigned int i=0;i<b.size();i++)
	b[i] = ((float)rand())/((float)RAND_MAX);
      
      for(unsigned int j=0;j<W.ysize();j++)
	for(unsigned int i=0;i<W.xsize();i++){
	  W(j,i) = ((float)rand())/((float)RAND_MAX);
	}
      
      
      l[1] = new neuronlayer< math::blas_real<float> >(*(l[0]));
      
      if(l[0]->input() != l[1]->input()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [input()]");
      }
	
      if(l[0]->output() != l[1]->output()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [output()]");
      }
      
      if(l[0]->moment() != l[1]->moment()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [moment()]");
      }
      
      if(l[0]->learning_rate() != l[1]->learning_rate()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [learning_rate()]");
      }
      
      if(l[0]->input_size() != l[1]->input_size()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [input_size()]");
      }
      
      if(l[0]->size() != l[1]->size()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [size()]");
      }
      
      if(l[0]->bias() != l[1]->bias()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [bias()]");
      }
      
      if(l[0]->weights() != l[1]->weights()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [weights()]");
      }
      
      
      // tests calculation() gives same output
      
      // sets input and output randomly
      
      for(unsigned int i=0;i<input.size();i++)
	(*(l[0]->input()))[i] = ((float)rand())/((float)RAND_MAX);
      
      for(unsigned int i=0;i<output.size();i++)
	(*(l[0]->output()))[i] = ((float)rand())/((float)RAND_MAX);
      
      math::vertex< math::blas_real<float> > alternative_output(*(l[0]->output()));
      
      l[0]->output() = &alternative_output;
      
      l[0]->calculate();
      l[1]->calculate();
      
      output -= alternative_output;
      
      {
	// calculates error
	
	math::blas_real<float> error = 0.0f;
	
	for(unsigned int i=0;i<output.size();i++)
	  error += output[i]*output[i];
	
	if(error >= 0.01f)
	  throw test_exception("neuronlayer copy ctor test: calculation result mismatch.");
	
      }
      
      delete l[0];
      delete l[1];
    }

  }
  catch(std::exception& e){
    std::cout << "Error - unexpected exception: " 
	      << e.what() << std::endl;
  }
}
#endif

#if 0
void backprop_test(const unsigned int size)
{
  {
    cout << "SIMPLE BACKPROPAGATION TEST 1" << endl;
    
    /* tests linear separation ability of nn between two classes */
    
    std::vector< math::vertex<float > > input(size);
    std::vector< math::vertex<float > > output(size);
    
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);

      input[i][0] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
      input[i][1] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
      
      output[i][0] = input[i][0] - 0.2*input[i][1];
      output[i][1] = -0.12*input[i][0] + 0.1*input[i][1];
      
      
      // output[i][0] += rand()/(double)RAND_MAX;
    }
    
    dataset< float > I0(2);
    if(!I0.add(input)){
      std::cout << "dataset creation failed\n";
      return;
    }
    
    dataset< float > I1(2);
    if(!I1.add(output)){
      std::cout << "dataset creation failed\n";
      return;
    }
  
  
    I0.preprocess();
    I1.preprocess();
    
    neuralnetwork< float >* nn;
    backpropagation< float > bp;
    
    vector<unsigned int> nn_arch;
    nn_arch.push_back(2);
    nn_arch.push_back(2); // 8
    nn_arch.push_back(2);
  
    /* two outputs == bad - fix/make nn structure more dynamic,
       ignores second one */
    nn = new neuralnetwork< float >(nn_arch); 
    nn->randomize();
    
    math::vertex< float > correct_output(2);
    
    // FOR SOME REASON C++ BACKPROPAGATION CODE HAS ERRORS. FIX IT (OR GCC 3.4 HAS BUGS.., TEST WITH GCC 3.3)
    
    float sum_error;
  
  
    std::cout << "NN-ARCH = ";
    
    for(unsigned int i=0;i<nn_arch.size();i++)
      std::cout << nn_arch[i] << " ";
    
    std::cout << std::endl;
    std::cout << "====================================================" << std::endl;
    
    bp.setData(nn, &I0, &I1);
    
    for(unsigned int e=0;e<100;e++){
      
      if(bp.improve(1) == false)
	std::cout << "BP::improve() failed." << std::endl;
      
      sum_error = bp.getError();
      
      std::cout << e << "/100  MEAN SQUARED ERROR: "
		<< sum_error << std::endl;
    }
    
    
    std::cout << "EPOCH ERROR SHOULD HAVE CONVERGED CLOSE TO ZERO." << std::endl;
    
    delete nn;
  }





  
  
  {
    cout << "SIMPLE BACKPROPAGATION TEST 2" << endl;
    
    /* tests linear separation ability of nn between two classes */
    
    std::vector< math::vertex<math::blas_real<float> > > input(size);
    std::vector< math::vertex<math::blas_real<float> > > output(size);

    // creates data
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);

      input[i][0] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
      input[i][1] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
      
      output[i][0] = input[i][0] - 0.2f*input[i][1];
      output[i][1] = -0.12f*input[i][0] + 0.1f*input[i][1];      
    }
    
    dataset< math::blas_real<float> > I0(2);
    if(!I0.add(input)){
      std::cout << "dataset creation failed\n";
      return;
    }
    
    dataset< math::blas_real<float> > I1(2);
    if(!I1.add(output)){
      std::cout << "dataset creation failed\n";
      return;
    }
    
    std::cout << "pre abs maxs" << std::endl;
    math::blas_real<float> mm = 0.0, MM = 0.0;
    
    for(unsigned int i=0;i<I0.size(0);i++){
      if(mm < whiteice::math::abs(I0[i][0]))
	mm = whiteice::math::abs(I0[i][1]);	
      if(MM < whiteice::math::abs(I0[i][1]))
	MM = whiteice::math::abs(I0[i][1]);
    }
    std::cout << "mm = " << mm << std::endl;
    std::cout << "MM = " << MM << std::endl;
    
    I0.preprocess();
    I1.preprocess();
    
    
    std::cout << "post abs maxs" << std::endl;
    mm = 0, MM = 0;
    for(unsigned int i=0;i<I0.size(0);i++){
      if(mm < whiteice::math::abs(I0[i][0]))
	mm = whiteice::math::abs(I0[i][1]);	
      if(MM < whiteice::math::abs(I0[i][1]))
	MM = whiteice::math::abs(I0[i][1]);
    }
    std::cout << "mm = " << mm << std::endl;
    std::cout << "MM = " << MM << std::endl;
    
    
    neuralnetwork< math::blas_real<float> >* nn;
    backpropagation< math::blas_real<float> > bp;
    
    vector<unsigned int> nn_arch;
    nn_arch.push_back(2);
    nn_arch.push_back(8);
    nn_arch.push_back(2);
    
    /* two outputs == bad - fix/make nn structure more dynamic,
       ignores second one */
    nn = new neuralnetwork< math::blas_real<float> >(nn_arch); 
    nn->randomize();
    
    math::vertex< math::blas_real<float> >& nn_input  = nn->input();
    math::vertex< math::blas_real<float> >& nn_output = nn->output();
    
    math::vertex< math::blas_real<float> > correct_output(2);
    
    math::blas_real<float> sum_error;
    
    
    std::cout << "NN-ARCH = ";
    
    for(unsigned int i=0;i<nn_arch.size();i++)
      std::cout << nn_arch[i] << " ";
    
    std::cout << std::endl;
    
  
    for(unsigned int e=0;e<1000;e++){
      
      sum_error = 0;
      
      for(unsigned int i = 0;i<I0.size(0);i++){
	unsigned int index = rand() % I0.size(0);
	
	nn_input[0] = I0[index][0];
	nn_input[1] = I0[index][1];      
	
	nn->calculate();
	
	correct_output[0] = I1[index][0];
	correct_output[1] = I1[index][1];
	
	math::blas_real<float> sq_error = 
	  (nn_output[0] - correct_output[0])*(nn_output[0] - correct_output[0]) +
	  (nn_output[1] - correct_output[1])*(nn_output[1] - correct_output[1]);
	
	// std::cout << "neural network i/o" << std::endl;
	// std::cout << nn_output[0] << std::endl;
	// std::cout << correct_output[0] << std::endl;
	// std::cout << nn_output[1] << std::endl;
	// std::cout << correct_output[1] << std::endl;
	
	sum_error += sq_error;
	
	if(bp.calculate(*nn, correct_output) == false){
	  cout << "WARNING CALCULATION FAILED: " << i << endl;
	}
	else{
	  //cout << "SQ_ERROR: " << sq_error << endl;
	}
	
      }
      
      sum_error = sqrt(sum_error / ((math::blas_real<float>)I0.size(0)) );
      
      cout << "MEAN SQUARED ERROR: " << sum_error << endl;
    }
    
    
    cout << "EPOCH ERROR SHOULD HAVE CONVERGED CLOSE TO ZERO." << endl;
    
    delete nn;
  }
  
}



void neuralnetwork_test()
{
  cout << "SIMPLE NEURAL NETWORK TEST" << endl;
  
  neuralnetwork<double>* nn;
  
  {
    nn = new neuralnetwork<double>(100,10);
    
    math::vertex<double>& input  = nn->input();
    math::vertex<double>& output = nn->output();
    
    for(unsigned int i=0;i<input.size();i++)
      input[i] = 1.0;
    
    for(unsigned int i=0;i<output.size();i++)
      output[i] = 0.0;
    
    
    nn->randomize();
    nn->calculate();
    
    std::cout << "OUTPUTS " << std::endl;
    std::cout << output << std::endl;
    
    delete nn;
  }
  
  cout << "10-1024x10-4 NEURAL NETWORK TEST" << endl;
  
  {
    vector<unsigned int> nn_arch;
    nn_arch.push_back(10);
    
    for(unsigned int c=0;c<10;c++)
      nn_arch.push_back(1024);
    
    nn_arch.push_back(4);
    
    nn = new neuralnetwork<double>(nn_arch);
    
    math::vertex<double>& input  = nn->input();
    math::vertex<double>& output = nn->output();
    
    for(unsigned int i=0;i<input.size();i++)
      input[i] = 1.0;
    
    for(unsigned int i=0;i<output.size();i++)
      output[i] = 0.0;
    
    nn->randomize();
    nn->calculate();

    nn->randomize();
    nn->calculate();
    
    std::cout << "2ND NN OUTPUTS" << std::endl;
    std::cout << output << std::endl;
    
    delete nn;
  }
  
}


void neuronlayer_test2()
{
  cout << "NEURONLAYER TEST2" << endl;
  
  neuronlayer<double>* layer;
  
  layer = new neuronlayer<double>(10, 10); /* 10 inputs, 10 outputs 1-layer nn */
  
  math::vertex<double> input(10);
  math::vertex<double> output(10);

  layer->input()  = &input;
  layer->output() = &output;

  for(unsigned int i=0;i<input.size();i++) input[i] = 0;
  
  layer->bias()[5] = 10;

  layer->calculate();

  cout << "neural network activation results:" << endl;

  for(unsigned int i=0;i<10;i++){
    cout << output[i] << " ";
  }

  cout << endl;
    
  delete layer;
}



void neuron_test()
{
  cout << "NEURON TEST" << endl;
  
  neuron<double> n(10);
  double v = 0;
  
  n.bias() = -55*0.371;
  
  vector<double> input;
  
  input.resize(10);
  
  for(int i=1;i<=10;i++){
    input[i-1] = i*0.371;
    n[i-1] = 1;
  }
  
  v = n(input);
  
  std::cout << "neuron(" << n.local_field() << ") = " << v << std::endl; // should be zero
}
#endif


template <typename T>
void calculateF(activation_function<T>& F, T value)
{
  std::cout << "F(" << value << ") = " << F(value) << std::endl;
}


void activation_test()
{
  std::cout << "SIGMOID ACTIVATION FUNCTION TEST" << std::endl;
  
  activation_function<double> *F;
  F = new odd_sigmoid<double>;
  
  
  double value = -1;
  
  while(value <= 1.0){
    calculateF<double>(*F, value);
    value += 0.25;
  }
  
  delete F;
}


/************************************************************************/

void nnetwork_complex_test()
{
  printf("COMPLEX VALUED NNETWORK<> TESTS\n");
  fflush(stdout);
  
  try{
    std::cout << "NNETWORK TEST -1: GET/SET PARAMETER TESTS"
	      << std::endl;

    std::vector<unsigned int> arch;
    arch.push_back(4);
    arch.push_back(4);
    arch.push_back(4);
    arch.push_back(5);
    
    nnetwork< math::blas_complex<float> > nn(arch); // 4-4-4-5 network (3 layer network)

    math::vertex< math::blas_complex<float> > b;
    math::matrix< math::blas_complex<float> > W;

    nn.getBias(b, 0);
    nn.getWeights(W, 0);

    std::cout << "First layer W*x + b." << std::endl;
    std::cout << "W = " << W << std::endl;
    std::cout << "b = " << b << std::endl;

    math::vertex< math::blas_complex<float> > all;
    nn.exportdata(all);

    std::cout << "whole nn vector = " << all << std::endl;

    W(0,0) = 100.0f;

    if(nn.setWeights(W, 1) == false)
      std::cout << "ERROR: cannot set NN weights." << std::endl;

    b.resize(5);

    if(nn.setBias(b, 2) == false)
      std::cout << "ERROR: cannot set NN bias." << std::endl;
    
    math::vertex< math::blas_complex<float> > b2;

    if(nn.getBias(b2, 2) == false)
      std::cout << "ERROR: cannot get NN bias." << std::endl;

    if(b.size() != b2.size())
      std::cout << "ERROR: bias terms mismatch (size)." << std::endl;

    math::vertex< math::blas_complex<float> > e = b - b2;

    if(abs(e.norm()) > 0.01f)
      std::cout << "ERROR: bias terms mismatch." << std::endl;
    
      
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }  

  

  
  try{
    std::cout << "NNETWORK TEST 0: SAVE() AND LOAD() TEST" << std::endl;
    
    nnetwork< math::blas_complex<float> >* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(18);
    arch.push_back(10);
    arch.push_back(1);

    std::vector<unsigned int> arch2;
    arch2.push_back(8);
    arch2.push_back(10);
    arch2.push_back(3);
    
    nn = new nnetwork< math::blas_complex<float> >(arch);    
    nn->randomize();
    nn->setBatchNorm(true);

    nnetwork< math::blas_complex<float> >* copy = new nnetwork< math::blas_complex<float> >(*nn);
    
    if(nn->save("nntest.cfg") == false){
      std::cout << "nnetwork::save() failed." << std::endl;
      delete nn;
      delete copy;
      return;
    }

    nn->randomize();
    
    delete nn;
    
    nn = new nnetwork< math::blas_complex<float> >(arch2);    
    nn->randomize();

    if(nn->load("nntest.cfg") == false){
      std::cout << "nnetwork::load() failed." << std::endl;
      delete nn;
      delete copy;
      return;
    }

    math::vertex< math::blas_complex<float> > p1, p2;
    math::vertex< math::blas_complex<float> > b1, b2;
    
    if(nn->exportdata(p1) == false || copy->exportdata(p2) == false || 
       nn->exportBNdata(b1) == false || copy->exportBNdata(b2) == false){
      std::cout << "nnetwork exportdata failed." << std::endl;
      delete nn;
      delete copy;
      return;
    }
    
    math::vertex< math::blas_complex<float> > e1 = p1 - p2;
    math::vertex< math::blas_complex<float> > e2 = b1 - b2;
    
    std::cout << "p1 = " << p1 << std::endl;
    std::cout << "p2 = " << p2 << std::endl;
    std::cout << "e1 = " << e1 << std::endl;
    std::cout << "e2 = " << e2 << std::endl;

    if(abs(e1.norm()) > 0.001f || abs(e2.norm()) > 0.001f){
      std::cout << "ERROR: save() & load() failed in nnetwork!" << std::endl;
      delete nn;
      delete copy;
      return;
    }
    else{
      std::cout << "nnetwork::save() and nnetwork::load() work correctly." << std::endl;
    }

    delete nn;
    delete copy;
    nn = 0;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }  


  
#if 0
  try{
    std::cout << "NNETWORK TEST 2: SIMPLE PROBLEM + DIRECT GRADIENT DESCENT" << std::endl;
    
    nnetwork< math::blas_complex<float> >* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(2);
    arch.push_back(10);
    arch.push_back(10);
    arch.push_back(10);
    arch.push_back(2);
    
    nn = new nnetwork< math::blas_complex<float> >(arch);
    nn->randomize();
    
    const unsigned int size = 500;
    
    
    std::vector< math::vertex< math::blas_complex<float> > > input(size);
    std::vector< math::vertex< math::blas_complex<float> > > output(size);
    
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);
      input[i].zero();
      output[i].zero();
      
      input[i][0].real( (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f ); // [-1.0,+1.0]
      input[i][0].imag( (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f ); // [-1.0,+1.0]
      input[i][1].real( (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f ); // [-1.0,+1.0]
      input[i][1].imag( (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f ); // [-1.0,+1.0]      

      
      output[i][0] = input[i][0] - math::blas_complex<float>(0.2f)*input[i][1];
      output[i][1] = math::blas_complex<float>(-0.12f)*input[i][0] + math::blas_complex<float>(0.1f)*input[i][1];
      
    }
    
    
    dataset< math::blas_complex<float> > data;
    std::string inString, outString;
    inString = "input";
    outString = "output";
    data.createCluster(inString,  2);
    data.createCluster(outString, 2);
    
    data.add(0, input);
    data.add(1, output);
    
    //data.preprocess(0);
    //data.preprocess(1);
    
    math::vertex< math::blas_complex<float> > grad, err, weights;
    
    unsigned int counter = 0;
    math::blas_complex<float> error = math::blas_complex<float>(1000.0f);
    math::blas_complex<float> lrate = math::blas_complex<float>(0.01f);

    while(abs(error) > math::blas_real<float>(0.001f) && counter < 10000){
      error = math::blas_complex<float>(0.0f);
      
      // goes through data, calculates gradient
      // exports weights, weights -= 0.01*gradient
      // imports weights back

      for(unsigned int i=0;i<data.size(0);i++){
	nn->input() = data.access(0, i);

	nn->calculate(true);
	err = nn->output() - data.access(1,i);

	for(unsigned int i=0;i<err.size();i++)
	  error += err[i]*math::conj(err[i]);

	if(nn->mse_gradient(err, grad) == false)
	  std::cout << "gradient failed." << std::endl;

	if(nn->exportdata(weights) == false)
	  std::cout << "export failed." << std::endl;
	
	weights -= lrate * grad;
	
	if(nn->importdata(weights) == false)
	  std::cout << "import failed." << std::endl;
      }


      error /= math::blas_real<float>((float)data.size(0));
      
      std::cout << counter << " : " << abs(error) << std::endl;
      
      counter++;
    }
    
    std::cout << counter << " : " << error << std::endl;
    
    delete nn;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }
#endif


  try{
    std::cout << "NNETWORK TEST 3: SIMPLE PROBLEM + SUM OF DIRECT GRADIENT DESCENT" << std::endl;
    
    nnetwork< math::blas_complex<float> >* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(2);
    arch.push_back(20);
    arch.push_back(2);
    
    nn = new nnetwork< math::blas_complex<float> >();
    // tests linear network
    // non-linearies for complex valued neural networks:
    // * pureLinear (works with complex numbers)
    // * rectifier (don't work with complex numbers/gradient descent don't work
    //              as complex analytical assumptions are not fulfilled by rectifier)
    // * halfLinear (works with regularizer) [don't work with MSE gradient calls!]
    // * tanh (works with regularizer)
    // * sigmoid (crashes quickly with regularizer)
    // * softmax (works with regularizer)
    
    nn->setArchitecture(arch, nnetwork< math::blas_complex<float> >::softmax);
    nn->randomize();
    
    const unsigned int size = 500;
    
    
    std::vector< math::vertex< math::blas_complex<float> > > input(size);
    std::vector< math::vertex< math::blas_complex<float> > > output(size);
    
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);
      input[i].zero();
      output[i].zero();
      
      input[i][0].real( (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f ); // [-1.0,+1.0]
      input[i][0].imag( (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f ); // [-1.0,+1.0]
      input[i][1].real( (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f ); // [-1.0,+1.0]
      input[i][1].imag( (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f ); // [-1.0,+1.0]      
      
      output[i][0] = input[i][0] - math::blas_complex<float>(0.2f)*input[i][1];
      output[i][1] = math::blas_complex<float>(-0.12f)*input[i][0] + math::blas_complex<float>(0.1f)*input[i][1];
      
    }
    
    
    dataset< math::blas_complex<float> > data;
    std::string inString, outString;
    inString = "input";
    outString = "output";
    data.createCluster(inString,  2);
    data.createCluster(outString, 2);
    
    data.add(0, input);
    data.add(1, output);
    
    //data.preprocess(0);
    //data.preprocess(1);
    
    math::vertex< math::blas_complex<float> > grad, err, weights;
    math::vertex< math::blas_complex<float> > sumgrad;
    
    unsigned int counter = 0;
    math::blas_complex<float> error = math::blas_real<float>(1000.0f);
    math::blas_complex<float> lrate = math::blas_real<float>(0.01f);
    
    while(abs(error) > math::blas_real<float>(0.001f) && counter < 10000){
      error = math::blas_complex<float>(0.0f);
      sumgrad.zero();
      
      // goes through data, calculates gradient
      // exports weights, weights -= 0.01*gradient
      // imports weights back

      math::blas_complex<float> ninv =
	math::blas_complex<float>(1.0f/data.size(0));
      
      for(unsigned int i=0;i<data.size(0);i++){
	nn->input() = data.access(0, i);
	nn->calculate(true);
	err = nn->output() - data.access(1,i);
	
	for(unsigned int j=0;j<err.size();j++)
	  error += ninv*err[j]*math::conj(err[j]);

#if 1
	// this works with pureLinear non-linearity
	const auto delta = err; // delta = (f(z) - y)
	whiteice::math::matrix< whiteice::math::blas_complex<float> > DF;
	nn->jacobian(data.access(0, i), DF);

	auto cdelta = delta;
	//old: only calculates Re(f(w))/dw
	auto cDF = DF;
	cdelta.conj();
	cDF.conj();
	//grad = cdelta*DF + delta*cDF;

	// new: this should calculate f(w)/dw
	grad = delta*cDF;

	// grad = delta*DF; // [THIS DOESN'T WORK]
#else
	const auto delta = err;
	
	if(nn->mse_gradient(err, grad) == false) // returns: delta*conj(DF)
	  std::cout << "gradient failed." << std::endl;
	
#endif
	
	if(i == 0)
	  sumgrad = ninv*grad;
	else
	  sumgrad += ninv*grad;
      }

      sumgrad.normalize(); // normalizes gradient length

      if(nn->exportdata(weights) == false)
	std::cout << "export failed." << std::endl;

      const whiteice::math::blas_complex<float> alpha(1e-6f);
      
      weights -= lrate * (sumgrad + alpha*weights);
      
      if(nn->importdata(weights) == false)
	std::cout << "import failed." << std::endl;

      
      std::cout << counter << " : " << abs(error) << std::endl;
      
      counter++;
    }
    
    std::cout << counter << " : " << abs(error) << std::endl;

    math::vertex< math::blas_complex<float> > params;
    nn->exportdata(params);
    std::cout << "nn solution weights = " << params << std::endl;
    
    delete nn;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }
  
}


/************************************************************************/

void nnetwork_test()
{
#if 0
  try{
    std::cout << "NNETWORK TEST -3: JACOBIAN MATRIX TEST" << std::endl;

    std::vector<unsigned int> layers;
    layers.push_back(7);
    layers.push_back(45);
    layers.push_back(5);
    layers.push_back(5);
    layers.push_back(13);

    whiteice::nnetwork< math::blas_complex<float> > nn(layers);
    nn.setNonlinearity(nn.getLayers()-1,
		       whiteice::nnetwork< math::blas_complex<float> >::pureLinear);
    
    // calculates jacobian matrix

    for(unsigned int n=0;n<20;n++){
      whiteice::RNG< math::blas_complex<float> > rng;
      math::vertex< math::blas_complex<float> > input(7);
      
      rng.normal(input);
      
      math::matrix< math::blas_complex<float> > JNN1, JNN2;
      
      if(nn.jacobian(input, JNN1) == false){
	std::cout << "ERROR: Calculating Jacobian matrix failed." << std::endl;
	return;
      }
      
      if(nn.jacobian_optimized(input, JNN2) == false){
	std::cout << "ERROR: Calculating Jacobian matrix failed." << std::endl;
	return;
      }
    
      auto delta = whiteice::math::abs(JNN1 - JNN2);
      
      if(frobenius_norm(delta).abs() > 0.01){
	std::cout << "ERROR: Jacobian matrices differ when changing computation method."
		  << std::endl;
	return;
      }
      else{
	std::cout << "Jacobian matrices are same. Good." << std::endl;
      }
    }
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: Unexpected exception: " << e.what() << std::endl;
  }
#endif
  
  
  try{
    std::cout << "NNETWORK TEST -2: GET/SET DEEP ICA PARAMETERS"
	      << std::endl;

    std::vector< math::vertex<> > data;
    
    for(unsigned int i=0;i<1000;i++){
      double t = i/500.0f - 1.0f;
      math::vertex<> x;
      x.resize(4);
      x[0] = sin(2.0*t);
      x[1] = cos(t)*cos(t)*cos(t);
      x[2] = sin(1.0 + 2.0*cos(t));
      x[3] = ((float)rand())/((float)RAND_MAX);

      data.push_back(x);
    }

    std::vector<deep_ica_parameters> params;
    unsigned int deepness = 2;

    if(deep_nonlin_ica(data, params, deepness) == false)
      std::cout << "ERROR: deep_nonlin_ica FAILED." << std::endl;

    std::cout << "Parameter layers: " << params.size() << std::endl;

    std::vector<unsigned int> layers;

    layers.push_back(4);

    for(unsigned int i=0;i<params.size();i++){
      layers.push_back(4);
      layers.push_back(4);
    }
    
    layers.push_back(6);
    layers.push_back(2);
    
    nnetwork<> nn(layers);

    if(initialize_nnetwork(params, nn) == false){
      std::cout << "ERROR: initialize_nnetwork FAILED." << std::endl;
    }
    
    
    std::cout << "Deep ICA network init PASSED." << std::endl;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }


  try{
    std::cout << "NNETWORK TEST -1: GET/SET PARAMETER TESTS"
	      << std::endl;

    std::vector<unsigned int> arch;
    arch.push_back(4);
    arch.push_back(4);
    arch.push_back(4);
    arch.push_back(5);

    // 16+4 + 16+4 + 20+5 parameters = 65 parameters in nnetwork<>
    
    nnetwork<> nn(arch); // 4-4-4-5 network (3 layer network)

    math::vertex<> b;
    math::matrix<> W;

    for(unsigned int l=0;l<nn.getLayers();l++){
      nn.getBias(b, l);
      nn.getWeights(W, l);

      std::cout << "Layer " << l << " W = " << W << std::endl;
      std::cout << "Layer " << l << " b = " << b << std::endl;
    }
    
    math::vertex<> all;
    nn.exportdata(all);

    std::cout << "whole nn vector = " << all << std::endl;

    nn.getWeights(W, 0);
    W(0,0) = 100.0f;

    if(nn.setWeights(W, 1) == false)
      std::cout << "ERROR: cannot set NN weights." << std::endl;

    b.resize(5);

    if(nn.setBias(b, 2) == false)
      std::cout << "ERROR: cannot set NN bias." << std::endl;
    
    math::vertex<> b2;

    if(nn.getBias(b2, 2) == false)
      std::cout << "ERROR: cannot get NN bias." << std::endl;

    if(b.size() != b2.size())
      std::cout << "ERROR: bias terms mismatch (size)." << std::endl;

    math::vertex<> e = b - b2;

    if(e.norm() > 0.01)
      std::cout << "ERROR: bias terms mismatch." << std::endl;
    
      
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }  

  

  
  try{
    std::cout << "NNETWORK TEST 0: SAVE() AND LOAD() TEST" << std::endl;
    std::cout << std::flush;
    
    nnetwork<>* nn;
    math::vertex<> p1, p2, b1, b2;
    
    std::vector<unsigned int> arch;
    arch.push_back(18);
    arch.push_back(10);
    arch.push_back(1);

    std::vector<unsigned int> arch2;
    arch2.push_back(8);
    arch2.push_back(10);
    arch2.push_back(3);
    
    nn = new nnetwork<>(arch);    
    nn->randomize();
    nn->setBatchNorm(true);

    nn->exportdata(p1);
    std::cout << "nn weights after random init = " << p1 << std::endl;
    
    nnetwork<>* copy = new nnetwork<>(*nn);
    
    
    if(nn->save("nntest.cfg") == false){
      std::cout << "nnetwork::save() failed." << std::endl;
      delete nn;
      delete copy;
      return;
    }
    
    nn->randomize();
    
    delete nn;
    
    nn = new nnetwork<>(arch2);    
    nn->randomize();
    
    if(nn->load("nntest.cfg") == false){
      std::cout << "nnetwork::load() failed." << std::endl;
      delete nn;
      delete copy;
      return;
    }
    

    
    if(nn->exportdata(p1) == false || copy->exportdata(p2) == false ||
       nn->exportBNdata(b1) == false || copy->exportBNdata(b2) == false){
      std::cout << "nnetwork exportdata failed." << std::endl;
      delete nn;
      delete copy;
      return;
    }
    
    math::vertex<> e1 = p1 - p2;
    math::vertex<> e2 = b1 - b2;
    
    std::cout << "p1 = " << p1 << std::endl;
    std::cout << "p2 = " << p2 << std::endl;
    std::cout << "e1 = " << e1 << std::endl;
    std::cout << "e2 = " << e2 << std::endl;
    
    if(e1.norm() > 0.001f || e2.norm() > 0.001f){
      std::cout << "ERROR: save() & load() failed in nnetwork!" << std::endl;
      delete nn;
      delete copy;
      return;
    }
       
      
    delete nn;
    delete copy;
    nn = 0;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }  


  
  
  try{
    std::cout << "NNETWORK TEST 2: SIMPLE PROBLEM + DIRECT GRADIENT DESCENT" << std::endl;
    
    nnetwork<>* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(2);
    arch.push_back(10);
    arch.push_back(10);
    arch.push_back(10);
    arch.push_back(2);
    
    nn = new nnetwork<>(arch);
    nn->setNonlinearity(nn->getLayers()-1, nnetwork<>::pureLinear);

    nn->randomize();
    
    const unsigned int size = 500;
    
    
    std::vector< math::vertex< math::blas_real<float> > > input(size);
    std::vector< math::vertex< math::blas_real<float> > > output(size);
    
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);
      input[i].zero();
      output[i].zero();
      
      input[i][0] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      input[i][1] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      
      output[i][0] = input[i][0] - math::blas_real<float>(0.2f)*input[i][1];
      output[i][1] = math::blas_real<float>(-0.12f)*input[i][0] +
	math::blas_real<float>(0.1f)*input[i][1];
    }
    
    
    dataset<> data;
    std::string inString, outString;
    inString = "input";
    outString = "output";
    data.createCluster(inString,  2);
    data.createCluster(outString, 2);
    
    data.add(0, input);
    data.add(1, output);

    assert(data.size(0) == data.size(1));
    
    //data.preprocess(0);
    //data.preprocess(1);

    printf("Starting gradient descent..\n");
    fflush(stdout);
    
    math::vertex<> grad, err, weights;
    
    unsigned int counter = 0;
    math::blas_real<float> error = math::blas_real<float>(1000.0f);
    math::blas_real<float> lrate = math::blas_real<float>(0.01f);
    
    while(error > math::blas_real<float>(0.001f) && counter < 10000){
      
      error = math::blas_real<float>(0.0f);
      
      // goes through data, calculates gradient
      // exports weights, weights -= 0.01*gradient
      // imports weights back

      auto ninv = math::blas_real<float>(1.0f/(float)data.size(0));

      for(unsigned int i=0;i<data.size(0);i++){
	
	const auto x = data.access(0, i);
	const auto y = data.access(1, i);

	nn->input() = x;
	nn->calculate(true);
	err = nn->output() - y;
	
	for(unsigned int i=0;i<err.size();i++)
	  error += ninv*err[i]*err[i];

#if 0
	// seperate gradient calculation	
	const auto delta = err;
	whiteice::math::matrix< whiteice::math::blas_real<float> > DF;
	nn->jacobian(x, DF);
	DF.conj();
	grad = delta*DF;
#else	
	if(nn->mse_gradient(err, grad) == false)
	  std::cout << "gradient failed." << std::endl;
#endif
	
	//printf("||grad(neuralnetwork)|| = %f\n", grad.norm().c[0]);
	//fflush(stdout);
	
	if(nn->exportdata(weights) == false)
	  std::cout << "export failed." << std::endl;
	
	weights -= lrate * grad;
	
	if(nn->importdata(weights) == false)
	  std::cout << "import failed." << std::endl;
      }
      
      std::cout << counter << " : " << error << std::endl;
      std::cout << std::flush;
      
      counter++;
    }
    
    std::cout << counter << " : " << error << std::endl;

    math::vertex<> params;
    nn->exportdata(params);
    std::cout << "nn solution weights = " << params << std::endl;
    
    delete nn;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }


  try{
    std::cout << "NNETWORK TEST 3: SIMPLE PROBLEM + SUM OF DIRECT GRADIENT DESCENT" << std::endl;
    
    nnetwork<>* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(2);
    arch.push_back(10);
    arch.push_back(10);
    arch.push_back(10);
    arch.push_back(2);
    
    nn = new nnetwork<>(arch);
    nn->setNonlinearity(nn->getLayers()-1, nnetwork<>::pureLinear);

    nn->randomize();
    
    const unsigned int size = 500;
    
    
    std::vector< math::vertex< math::blas_real<float> > > input(size);
    std::vector< math::vertex< math::blas_real<float> > > output(size);
    
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);
      
      input[i][0] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      input[i][1] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      
      output[i][0] = input[i][0] - math::blas_real<float>(0.2f)*input[i][1];
      output[i][1] = math::blas_real<float>(-0.12f)*input[i][0] + math::blas_real<float>(0.1f)*input[i][1];
      
    }
    
    
    dataset<> data;
    std::string inString, outString;
    inString = "input";
    outString = "output";
    data.createCluster(inString,  2);
    data.createCluster(outString, 2);
    
    data.add(0, input);
    data.add(1, output);
    
    data.preprocess(0);
    data.preprocess(1);
    
    math::vertex<> grad, err, weights;
    math::vertex<> sumgrad;
    
    unsigned int counter = 0;
    math::blas_real<float> error = math::blas_real<float>(1000.0f);
    math::blas_real<float> lrate = math::blas_real<float>(0.01f);
    
    while(error > math::blas_real<float>(0.001f) && counter < 10000){
      error = math::blas_real<float>(0.0f);
      
      // goes through data, calculates gradient
      // exports weights, weights -= 0.01*gradient
      // imports weights back

      math::blas_real<float> ninv =
	math::blas_real<float>(1.0f/data.size(0));
      
      for(unsigned int i=0;i<data.size(0);i++){
	nn->input() = data.access(0, i);
	nn->calculate(true);
	err = nn->output() - data.access(1,i);
	
	for(unsigned int j=0;j<err.size();j++)
	  error += err[j]*err[j];
	
	if(nn->mse_gradient(err, grad) == false)
	  std::cout << "gradient failed." << std::endl;

	if(i == 0)
	  sumgrad = ninv*grad;
	else
	  sumgrad += ninv*grad;
      }
      
      
      if(nn->exportdata(weights) == false)
	std::cout << "export failed." << std::endl;

      
      weights -= lrate * sumgrad;
      
      if(nn->importdata(weights) == false)
	std::cout << "import failed." << std::endl;

      
      error /= math::blas_real<float>((float)data.size(0));
      
      std::cout << counter << " : " << error << std::endl;
      
      counter++;
    }
    
    std::cout << counter << " : " << error << std::endl;

    math::vertex<> params;
    nn->exportdata(params);
    std::cout << "nn solution weights = " << params << std::endl;
    
    delete nn;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }


#if 0
  try{
    std::cout << "NNETWORK TEST 4: SIMPLE PROBLEM + HMC sampler" << std::endl;
    
    nnetwork<>* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(2);
    arch.push_back(10);
    arch.push_back(2);
    
    nn = new nnetwork<>(arch);
    
    const unsigned int size = 500;
    
    
    std::vector< math::vertex< math::blas_real<float> > > input(size);
    std::vector< math::vertex< math::blas_real<float> > > output(size);
    
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);
      
      input[i][0] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      input[i][1] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      
      output[i][0] = input[i][0] - math::blas_real<float>(0.2f)*input[i][1];
      output[i][1] = math::blas_real<float>(-0.12f)*input[i][0] + math::blas_real<float>(0.1f)*input[i][1];
      
    }
    
    
    dataset<> data;
    std::string inString, outString;
    inString = "input";
    outString = "output";
    data.createCluster(inString,  2);
    data.createCluster(outString, 2);
    
    data.add(0, input);
    data.add(1, output);
    
    data.preprocess(0);
    data.preprocess(1);


    {
      whiteice::HMC<> hmc(*nn, data);

      hmc.startSampler(); // DISABLED: 20 threads (extreme testcase) [HMC currently only supports a single thread]

      while(hmc.getNumberOfSamples() < 200){
	if(hmc.getNumberOfSamples() > 0){
	  std::cout << "Number of samples: "
		    << hmc.getNumberOfSamples() << std::endl;
	  std::cout << "Mean error (100 latest samples): "
		    << hmc.getMeanError(100) << std::endl;
	}
	sleep(1);
      }

      hmc.stopSampler();
      nn->importdata(hmc.getMean());
    }

    math::blas_real<float> error = math::blas_real<float>(0.0f);
      
    for(unsigned int i=0;i<data.size(0);i++){
      nn->input() = data.access(0, i);
      nn->calculate(true);
      math::vertex<> err = data.access(1,i) - nn->output();
	
      for(unsigned int j=0;j<err.size();j++)
	error += err[j]*err[j];
    }
      
    
    error /= math::blas_real<float>((float)data.size());
    
    std::cout << "FINAL MEAN ERROR : " << error << std::endl;
    
    delete nn;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }  
#endif
  
  
}

/******************************************************************/

#if 0

// FIXME!!! we need logistic regression test for DBN

void lreg_nnetwork_test()
{
  try{
    std::cout << "SINH NNETWORK TEST -2: GET/SET DEEP ICA PARAMETERS"
	      << std::endl;

    std::vector< math::vertex<> > data;
    
    for(unsigned int i=0;i<1000;i++){
      double t = i/500.0f - 1.0f;
      math::vertex<> x;
      x.resize(4);
      x[0] = sin(2.0*t);
      x[1] = cos(t)*cos(t)*cos(t);
      x[2] = sin(1.0 + 2.0*cos(t));
      x[3] = ((float)rand())/((float)RAND_MAX);

      data.push_back(x);
    }

    std::vector<deep_ica_parameters> params;
    unsigned int deepness = 2;

    if(deep_nonlin_ica_sinh(data, params, deepness) == false)
      std::cout << "ERROR: deep_nonlin_ica FAILED." << std::endl;

    std::cout << "Parameter layers: " << params.size() << std::endl;

    std::vector<unsigned int> layers;

    layers.push_back(4);

    for(unsigned int i=0;i<params.size();i++){
      layers.push_back(4);
      layers.push_back(4);
    }
    
    layers.push_back(6);
    layers.push_back(2);
    
    sinh_nnetwork<> nn(layers);

    if(initialize_nnetwork(params, nn) == false){
      std::cout << "ERROR: initialize_nnetwork FAILED." << std::endl;
    }
    
    
    std::cout << "Deep ICA network init PASSED." << std::endl;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }


  try{
    std::cout << "SINH NNETWORK TEST -1: GET/SET PARAMETER TESTS"
	      << std::endl;

    std::vector<unsigned int> arch;
    arch.push_back(4);
    arch.push_back(4);
    arch.push_back(4);
    arch.push_back(5);
    
    sinh_nnetwork<> nn(arch); // 4-4-4-5 network (3 layer network)

    math::vertex<> b;
    math::matrix<> W;

    nn.getBias(b, 0);
    nn.getWeights(W, 0);

    std::cout << "First layer W*x + b." << std::endl;
    std::cout << "W = " << W << std::endl;
    std::cout << "b = " << b << std::endl;

    math::vertex<> all;
    nn.exportdata(all);

    std::cout << "whole nn vector = " << all << std::endl;

    W(0,0) = 100.0f;

    if(nn.setWeights(W, 1) == false)
      std::cout << "ERROR: cannot set NN weights." << std::endl;

    b.resize(5);

    if(nn.setBias(b, 2) == false)
      std::cout << "ERROR: cannot set NN bias." << std::endl;
    
    math::vertex<> b2;

    if(nn.getBias(b2, 2) == false)
      std::cout << "ERROR: cannot get NN bias." << std::endl;

    if(b.size() != b2.size())
      std::cout << "ERROR: bias terms mismatch (size)." << std::endl;

    math::vertex<> e = b - b2;

    if(e.norm() > 0.01)
      std::cout << "ERROR: bias terms mismatch." << std::endl;
    
      
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }  

  

  
  try{
    std::cout << "SINH NNETWORK TEST 0: SAVE() AND LOAD() TEST" << std::endl;
    
    sinh_nnetwork<>* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(18);
    arch.push_back(10);
    arch.push_back(1);

    std::vector<unsigned int> arch2;
    arch2.push_back(8);
    arch2.push_back(10);
    arch2.push_back(3);
    
    nn = new sinh_nnetwork<>(arch);    
    nn->randomize();
    
    sinh_nnetwork<>* copy = new sinh_nnetwork<>(*nn);
    
    if(nn->save("nntest.cfg") == false){
      std::cout << "nnetwork::save() failed." << std::endl;
      delete nn;
      delete copy;
      return;
    }
    
    nn->randomize();
    
    delete nn;
    
    nn = new sinh_nnetwork<>(arch2);    
    nn->randomize();
    
    if(nn->load("nntest.cfg") == false){
      std::cout << "nnetwork::load() failed." << std::endl;
      delete nn;
      delete copy;
      return;
    }
    
    math::vertex<> p1, p2;
    
    if(nn->exportdata(p1) == false || copy->exportdata(p2) == false){
      std::cout << "nnetwork exportdata failed." << std::endl;
      delete nn;
      delete copy;
      return;
    }
    
    math::vertex<> e = p1 - p2;
    
    std::cout << "p1 = " << p1 << std::endl;
    std::cout << "p2 = " << p2 << std::endl;
    std::cout << "e  = " << e << std::endl;
    
    if(e.norm() > 0.001f){
      std::cout << "ERROR: save() & load() failed in nnetwork!" << std::endl;
      delete nn;
      delete copy;
      return;
    }
       
    delete copy;
    delete nn;
    nn = 0;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }  


  
  
  try{
    std::cout << "SINH NNETWORK TEST 2: SIMPLE PROBLEM + DIRECT GRADIENT DESCENT" << std::endl;
    
    sinh_nnetwork<>* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(2);
    arch.push_back(20);
    arch.push_back(2);
    
    nn = new sinh_nnetwork<>(arch);
    
    const unsigned int size = 500;
    
    
    std::vector< math::vertex< math::blas_real<float> > > input(size);
    std::vector< math::vertex< math::blas_real<float> > > output(size);
    
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);
      
      input[i][0] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      input[i][1] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      
      output[i][0] = input[i][0] - math::blas_real<float>(0.2f)*input[i][1];
      output[i][1] = math::blas_real<float>(-0.12f)*input[i][0] + math::blas_real<float>(0.1f)*input[i][1];
      
    }
    
    
    dataset<> data;
    std::string inString, outString;
    inString = "input";
    outString = "output";
    data.createCluster(inString,  2);
    data.createCluster(outString, 2);
    
    data.add(0, input);
    data.add(1, output);
    
    data.preprocess(0);
    data.preprocess(1);
    
    math::vertex<> grad, err, weights;
    
    unsigned int counter = 0;
    math::blas_real<float> error = math::blas_real<float>(1000.0f);
    math::blas_real<float> lrate = math::blas_real<float>(0.01f);
    while(error > math::blas_real<float>(0.001f) && counter < 10000){
      error = math::blas_real<float>(0.0f);
      
      // goes through data, calculates gradient
      // exports weights, weights -= 0.01*gradient
      // imports weights back
      
      for(unsigned int i=0;i<data.size(0);i++){
	nn->input() = data.access(0, i);
	nn->calculate(true);
	err = nn->output() - data.access(1,i);
	
	for(unsigned int i=0;i<err.size();i++)
	  error += err[i]*err[i];
	
	if(nn->mse_gradient(err, grad) == false)
	  std::cout << "gradient failed." << std::endl;
	
	if(nn->exportdata(weights) == false)
	  std::cout << "export failed." << std::endl;
	
	weights -= lrate * grad;
	
	if(nn->importdata(weights) == false)
	  std::cout << "import failed." << std::endl;
      }
      
      error /= math::blas_real<float>((float)data.size());
      
      std::cout << counter << " : " << error << std::endl;
      
      counter++;
    }
    
    std::cout << counter << " : " << error << std::endl;
    
    delete nn;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }


  try{
    std::cout << "SINH NNETWORK TEST 3: SIMPLE PROBLEM + SUM OF DIRECT GRADIENT DESCENT" << std::endl;
    
    sinh_nnetwork<>* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(2);
    arch.push_back(20);
    arch.push_back(2);
    
    nn = new sinh_nnetwork<>(arch);
    
    const unsigned int size = 500;
    
    
    std::vector< math::vertex< math::blas_real<float> > > input(size);
    std::vector< math::vertex< math::blas_real<float> > > output(size);
    
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);
      
      input[i][0] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      input[i][1] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      
      output[i][0] = input[i][0] - math::blas_real<float>(0.2f)*input[i][1];
      output[i][1] = math::blas_real<float>(-0.12f)*input[i][0] + math::blas_real<float>(0.1f)*input[i][1];
      
    }
    
    
    dataset<> data;
    std::string inString, outString;
    inString = "input";
    outString = "output";
    data.createCluster(inString,  2);
    data.createCluster(outString, 2);
    
    data.add(0, input);
    data.add(1, output);
    
    data.preprocess(0);
    data.preprocess(1);
    
    math::vertex<> grad, err, weights;
    math::vertex<> sumgrad;
    
    unsigned int counter = 0;
    math::blas_real<float> error = math::blas_real<float>(1000.0f);
    math::blas_real<float> lrate = math::blas_real<float>(0.01f);
    
    while(error > math::blas_real<float>(0.001f) && counter < 10000){
      error = math::blas_real<float>(0.0f);
      
      // goes through data, calculates gradient
      // exports weights, weights -= 0.01*gradient
      // imports weights back

      math::blas_real<float> ninv =
	math::blas_real<float>(1.0f/data.size(0));
      
      for(unsigned int i=0;i<data.size(0);i++){
	nn->input() = data.access(0, i);
	nn->calculate(true);
	err = nn->output() - data.access(1,i);
	
	for(unsigned int j=0;j<err.size();j++)
	  error += err[j]*err[j];
	
	if(nn->mse_gradient(err, grad) == false)
	  std::cout << "gradient failed." << std::endl;

	if(i == 0)
	  sumgrad = ninv*grad;
	else
	  sumgrad += ninv*grad;
      }
      
      
      if(nn->exportdata(weights) == false)
	std::cout << "export failed." << std::endl;
      
      weights -= lrate * sumgrad;
      
      if(nn->importdata(weights) == false)
	std::cout << "import failed." << std::endl;

      
      error /= math::blas_real<float>((float)data.size());
      
      std::cout << counter << " : " << error << std::endl;
      
      counter++;
    }
    
    std::cout << counter << " : " << error << std::endl;
    
    delete nn;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }

}
#endif

/********************************************************************/


void bayesian_nnetwork_test()
{
  try{
    std::cout << "BAYES NNETWORK TEST 0: SAVE() AND LOAD() TEST"
	      << std::endl;
    
    nnetwork<>* nn;
    bayesian_nnetwork<> bnn;
    
    std::vector<unsigned int> arch;
    arch.push_back(4);
    arch.push_back(10);
    arch.push_back(1);

    std::vector< typename nnetwork<>::nonLinearity > nl;
    nl.resize(arch.size()-1);

    for(unsigned int l=0;l<nl.size();l++)
      nl[l] = nnetwork<>::sigmoid;

    std::vector<bool> frozenLayers;
    frozenLayers.resize(arch.size()-1);

    for(unsigned int l=0;l<frozenLayers.size();l++)
      frozenLayers[l] = (bool)(rand()&1);
    
    nn = new nnetwork<>(arch);
    nn->setNonlinearity(nl);
    nn->setFrozen(frozenLayers);
    nn->setBatchNorm(true);

    std::vector< math::vertex<> > weights;
    std::vector< math::vertex<> > bndatas;
    weights.resize(10);
    bndatas.resize(10);

    for(unsigned int i=0;i<weights.size();i++){
      nn->randomize();
      if(nn->exportdata(weights[i]) == false){
	std::cout << "ERROR: NN exportdata failed.\n";
	return;
      }

      if(nn->exportBNdata(bndatas[i]) == false){
	std::cout << "ERROR: NN exportdata failed.\n";
	return;
      }
    }
    
    if(bnn.importSamples(*nn, weights, bndatas) == false){
      std::cout << "ERROR: BNN importSamples() failed" << std::endl;
      return;
    }

    if(bnn.save("bnn_file.conf") == false){
      std::cout << "ERROR: BNN save() failed" << std::endl;
      return;
    }

    bayesian_nnetwork<> bnn2;

    if(bnn2.load("bnn_file.conf") == false){
      std::cout << "ERROR: BNN load() failed" << std::endl;
      return;
    }

    std::vector<unsigned int> loaded_arch;
    std::vector< math::vertex<> > loaded_weights;
    std::vector< math::vertex<> > loaded_bndatas;
    std::vector< nnetwork<>::nonLinearity > loaded_nl;
    std::vector<bool> loadedFrozenLayers;

    whiteice::nnetwork<> loaded_nn;

    if(bnn2.exportSamples(loaded_nn, loaded_weights, loaded_bndatas) == false){
      std::cout << "ERROR: BNN exportSamples() failed" << std::endl;
      return;
    }

    loaded_nn.getArchitecture(loaded_arch);
    loaded_nn.getNonlinearity(loaded_nl);
    loaded_nn.getFrozen(loadedFrozenLayers);

    if(loaded_arch.size() != arch.size()){
      std::cout << "ERROR: BNN save/load arch size mismatch"
		<< loaded_arch.size() << " != "
		<< arch.size() 
		<< std::endl;
      return;
    }
    
    if(loaded_weights.size() != weights.size())
      std::cout << "ERROR: BNN save/load weight size mismatch "
		<< loaded_weights.size() << " != "
		<< weights.size()
		<< std::endl;

    for(unsigned int i=0;i<arch.size();i++){
      if(loaded_arch[i] != arch[i]){
	std::cout << "ERROR: BNN arch values mismatch" << std::endl;
	return;
      }
    }

    if(loaded_nl.size() != nl.size()){
      std::cout << "ERROR: BNN nonlinearity settings mismatch (1)" << std::endl;
      return;
    }

    for(unsigned int l=0;l<nl.size();l++){
      if(loaded_nl[l] != nl[l]){
	std::cout << "ERROR: BNN nonlinearity settings mismatch (2)" << std::endl;
	return;
      }
    }

    if(frozenLayers.size() != loadedFrozenLayers.size()){
      std::cout << "ERROR: BNN frozen layers settings mismatch (1)." << std::endl;
      return;
    }

    for(unsigned int l=0;l<frozenLayers.size();l++){
      if(loadedFrozenLayers[l] != frozenLayers[l]){
	std::cout << "ERROR: BNN frozen settings mismatch (2)." << std::endl;
	return;
      }
    }

    for(unsigned int i=0;i<loaded_weights.size();i++){
      math::vertex<> e = loaded_weights[i] - weights[i];

      if(e.norm() > 0.01f){
	std::cout << "ERROR: BNN weights value mismatch." << std::endl;
	return;
      }
    }


    for(unsigned int i=0;i<loaded_bndatas.size();i++){
      math::vertex<> e = loaded_bndatas[i] - bndatas[i];

      if(e.norm() > 0.01f){
	std::cout << "ERROR: BNN BN data vector value mismatch." << std::endl;
	return;
      }
    }
    
    
    delete nn;
    nn = 0;


    std::cout << "BAYES NNETWORK SAVE/LOAD TEST OK." << std::endl;
    
    
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }  

  
  try{
    std::cout << "BAYES NNETWORK TEST 1: HMC BAYESIAN NEURAL NETWORK TEST"
	      << std::endl;
    
    std::cout << "ERROR: NOT DONE" << std::endl;
    
    
    
    
    
  }
  catch(std::exception& e){
    
  }
}

