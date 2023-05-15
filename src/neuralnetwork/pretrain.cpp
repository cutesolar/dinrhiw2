/*
 * Heuristics to pretrain neural network weights using data.
 *
 * Let A, B and C be neural network layer's operators with matrix multiplication and 
 * non-linearity (C => y = g(C*x) )  
 * 
 * We assume operators are invertible so there is inverse functions inv(C) and 
 * inv(C*B*A)=inv(A)*inv(B)*inv(C).
 *
 * We calculate weights using linear optimization and training data (x,y). 
 * Parameters are initialized randomly and set to have unit weights 
 * (data aprox in the range of -1..1 typically)
 *
 * First we solve last layer weights, x' = B*A*x and we optimize 
 * linearly x' -> y and operator C's parameters (g^-1(y) = M_c*x' + b_c)
 *
 * Next we solve each layer's parameters x'' = A*x, and 
 * we solve B's parameters, we solve y' = inv(C)*y and have 
 * training data x'' -> y' to solve for parameters of B.
 *
 * You can run pretrain_nnetwork() many times for the same network until aprox convergence.
 *
 * 
 * Copyright Tomas Ukkonen 2023 <tomas.ukkonen@iki.fi>
 * Novel Insight Research
 *
 */


#include "pretrain.h"
#include "deep_ica_network_priming.h"
#include "nnetwork.h"
#include "linear_equations.h"

#include "Log.h"


#include <functional>
#include <system_error>



namespace whiteice
{

  template <typename T>
  PretrainNN<T>::PretrainNN()
  {
    running = false;
    iterations = 0;
    MAXITERS = 0;

    worker_thread = nullptr;
  }

  
  template <typename T>
  PretrainNN<T>::~PretrainNN()
  {
    stopTrain();
  }

  
  template <typename T>
  bool PretrainNN<T>::startTrain(const whiteice::nnetwork<T>& nnet,
				 const whiteice::dataset<T>& data,
				 const unsigned int NUMITERATIONS)
  {
    std::lock_guard<std::mutex> lock(thread_mutex);

    if(running) return false;
    if(NUMITERATIONS == 0) return false;

    if(data.getNumberOfClusters() < 2) return false;
    if(data.size(0) != data.size(1)) return false;
    if(data.dimension(0) != nnet.input_size()) return false;
    if(data.dimension(1) != nnet.output_size()) return false;
    
    if(worker_thread){
      delete worker_thread;
      worker_thread = nullptr;
    }

    {
      std::lock_guard<std::mutex> lock(solution_mutex);
      iterations = 0;
      MAXITERS = NUMITERATIONS;
      this->nnet = nnet;
      this->nnet.setResidual(false);
      this->nnet.setNonlinearity(whiteice::nnetwork<T>::pureLinear);
      this->data = data;
      current_error = T(-1.0f);
    }

    try{
      running = true;
      worker_thread = new std::thread(std::bind(&PretrainNN<T>::worker_loop, this));

      return true;
    }
    catch(const std::system_error& e){
      running = false;
      worker_thread = nullptr;

      return false;
    }

    return true;
  }

  
  template <typename T>
  bool PretrainNN<T>::isRunning() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);

    if(running == true && worker_thread != nullptr) return true;
    else return false;
  }

  
  template <typename T>
  void PretrainNN<T>::getStatistics(unsigned int& iterations, T& error) const
  {
    std::lock_guard<std::mutex> lock(solution_mutex);

    iterations = this->iterations;
    error = this->current_error;
  }

  
  template <typename T>
  bool PretrainNN<T>::stopTrain()
  {
    std::lock_guard<std::mutex> lock(thread_mutex);

    if(running == false) return false;
    
    try{
      running = false;
      
      if(worker_thread) worker_thread->join();
    }
    catch(const std::system_error& e){
      if(worker_thread) running = true;
      return false;
    }

    delete worker_thread;
    worker_thread = nullptr;

    return true;
  }
  

  template <typename T>
  bool PretrainNN<T>::getResults(whiteice::nnetwork<T>& nnet) const
  {
    std::lock_guard<std::mutex> lock(solution_mutex);

    if(iterations > 0){
      math::vertex<T> weights;
      
      if(this->nnet.exportdata(weights) == false) return false;
      if(nnet.importdata(weights) == false) return false;
      
      return true;
    }

    return false;
  }
  
  
  template <typename T>
  void PretrainNN<T>::worker_loop()
  {
    {
      std::lock_guard<std::mutex> lock(thread_mutex);

      if(!running) return; 
    }

    whiteice::nnetwork<T> nnet;

    {
      std::lock_guard<std::mutex> lock(solution_mutex);

      nnet = this->nnet;
    }
    
    std::vector< math::vertex<T> > vdata;
    
    data.getData(0, vdata);

    
    auto initial_mse = nnet.mae(data);
    auto best_mse = initial_mse;
    this->current_error = initial_mse;
    math::vertex<T> weights, w0, w1;
    
    nnet.exportdata(weights);
    
    T adaptive_step_length = T(1e-5f);
    
    // convergence detection
    std::list< T > errors;
    const unsigned int ERROR_HISTORY_SIZE = 30;
    
    
    
    while(1){
      
      {
	std::lock_guard<std::mutex> lock(thread_mutex);
	
	if(!running)
	  break;
      }

      {
	std::lock_guard<std::mutex> lock(solution_mutex);
	
	if(iterations >= MAXITERS)
	  break;
      }


      if(nnet.getBatchNorm()){
	nnet.calculateBatchNorm(vdata);
      }

      // std::cout << "step length = " << adaptive_step_length << std::endl; 

      nnet.exportdata(w1);

      if(matrixFactorizationMode){
	if(whiteice::pretrain_nnetwork_matrix_factorization
	   (nnet, data,
	    T(0.5f)*adaptive_step_length) == false){
	  //printf("ERROR!\n");
	  //continue;
	  break;
	}
      }
      else{
	if(whiteice::pretrain_nnetwork
	   (nnet, data) == false){
	  printf("ERROR!\n");
	  
	  break;
	}
      }

      nnet.exportdata(w0);

      if(matrixFactorizationMode){
	for(unsigned int i=0;i<w0.size();i++){
	  for(unsigned int k=0;k<w0[i].size();k++){
	    if(w0[i][k].c[0] < -0.75f) w0[i][k].c[0] = -0.75f;
	    if(w0[i][k].c[0] > +0.75f) w0[i][k].c[0] = +0.75f;
	  }
	}
      }

      nnet.importdata(w0);
      
      auto smaller_mse = nnet.mae(data);
      
      nnet.importdata(w1);
      
      if(matrixFactorizationMode){
	if(whiteice::pretrain_nnetwork_matrix_factorization
	   (nnet, data,
	    T(2.0f)*adaptive_step_length) == false){
	  // continue;
	  //printf("ERROR!\n");
	  
	  break;
	}
      }
      else{
	if(whiteice::pretrain_nnetwork
	   (nnet, data) == false){
	  printf("ERROR!\n");
	  
	  break;
	}
      }
      
      nnet.exportdata(w1);

      
      if(matrixFactorizationMode){
	for(unsigned int i=0;i<w1.size();i++){
	  for(unsigned int k=0;k<w1[i].size();k++){
	    if(w1[i][k].c[0] < -0.75f) w1[i][k].c[0] = -0.75f;
	    if(w1[i][k].c[0] > +0.75f) w1[i][k].c[0] = +0.75f;
	  }
	}
      }
      
      nnet.importdata(w1);
      
      auto larger_mse = nnet.mae(data);
      auto mse = larger_mse;
      
      if(smaller_mse[0] <= larger_mse[0]){
	adaptive_step_length *= T(0.5f);
	if(adaptive_step_length[0] < (1e-20))
	  adaptive_step_length = T(1e-20);
	mse = smaller_mse;
	
	nnet.importdata(w0);
      }
      else{
	adaptive_step_length *= T(2.0f);
	if(adaptive_step_length[0].c[0] >= 1e-5f)
	  adaptive_step_length = T(1e-5f);
      }
      
      
      for(unsigned int k=1;k<mse.size();k++)
	mse[k] = 0.0f;
      
      // std::cout << "MAE = " << mse << std::endl;
      
      if(best_mse[0] > mse[0]){
	best_mse = mse;
	this->current_error = best_mse;
	nnet.exportdata(weights);
      }
      
      
      {
	errors.push_back(best_mse);
	
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
	  
	  auto convergence = (stdev/(T(1e-5) + mean));
	  
	  // std::cout << "convergence = " << convergence << std::endl;
	  
	  if(convergence[0] < 0.1f){
	    // printf("LARGE ADAPTIVE STEPLENGTH\n");
	    adaptive_step_length = whiteice::math::sqrt(whiteice::math::sqrt(adaptive_step_length));
	    if(adaptive_step_length[0].c[0] >= 1e-5f)
	      adaptive_step_length = T(1e-5f);
	    
	    errors.clear();
	  }
	  
	}
      }

      // adaptive_step_length = 1e-5;

      // debugging messages
      {
	std::lock_guard<std::mutex> lock(solution_mutex);
	
	char buffer[256]; 
	
	snprintf(buffer, 256,
		 "whiteice::Pretrain: %d/%d: Neural network MAE for problem: %f %f%% %f %f%% (%e)",
		 iterations, MAXITERS, mse[0].c[0],
		 (mse/initial_mse)[0].c[0]*100.0f,
		 best_mse[0].c[0],
		 (best_mse/initial_mse)[0].c[0]*100.0f,
		 adaptive_step_length[0].c[0]);

	whiteice::logging.info(buffer);
      }
	
      {
        std::lock_guard<std::mutex> lock(solution_mutex);
	
	iterations++;
	this->nnet.importdata(weights);
      }
    }

    running = false;
    
  }
  
  


  template class PretrainNN< math::blas_real<float> >;
  template class PretrainNN< math::blas_real<double> >;

  template class PretrainNN< math::blas_complex<float> >;
  template class PretrainNN< math::blas_complex<double> >;

  template class PretrainNN< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >;
  template class PretrainNN< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >;

  template class PretrainNN< math::superresolution< math::blas_complex<float>, math::modular<unsigned int> > >;
  template class PretrainNN< math::superresolution< math::blas_complex<double>, math::modular<unsigned int> > >;

  //////////////////////////////////////////////////////////////////////
  
  template <typename T>
  bool pretrain_nnetwork(nnetwork<T>& nnet, const dataset<T>& data)
  {
    if(data.getNumberOfClusters() < 2) return false;
    if(data.size(0) != data.size(1)) return false;
    if(data.size(0) < 10) return false; // needs at least 10 data points to calculate something usable
    
    if(data.dimension(0) != nnet.input_size()) return false;
    if(data.dimension(1) != nnet.output_size()) return false;

    // zero means,unit variances neural network weights/data in layers
    
    // if(whiten1d_nnetwork(nnet, data) == false) return false;
    
    //printf("WHITENING DONE\n");
       
    // optimizes each layers linear parameters, first last layer

    const unsigned int SAMPLES = ((data.size(0) > 500) ? 500 : data.size(0));
    std::vector< math::vertex<T> > samples;
    

    for(unsigned int l=0;l<nnet.getLayers();l++){
      // for(unsigned int l=(nnet.getLayers());l>0;l--){
      // const unsigned int L = l-1;
      const unsigned int L = l;

      // printf("LAYER: %d/%d\n", L+1, nnet.getLayers());

      ////////////////////////////////////////////////////////////////////////
      // first calculates x values for the layer

      for(unsigned int s=0;s<SAMPLES;s++){
	const unsigned int index = rand() % data.size(0);
	samples.push_back(data.access(0, index));
	nnet.input() = data.access(0, index);
	
	if(nnet.calculate(false, true) == false)
	  
	  return false;
      }

      std::vector< math::vertex<T> > xsamples;

      if(nnet.getSamples(xsamples, L, SAMPLES) == false)
	return false;

      nnet.clearSamples();

      // printf("X SAMPLES DONE\n");

      //////////////////////////////////////////////////////////////////////
      // next calculates y values for the layer
      
      std::vector< std::vector<bool> > dropout;
      nnet.setDropOut(dropout, T(1.0f));
      
      for(unsigned int s=0;s<SAMPLES;s++){
	math::vertex<T> x, y;
	x = samples[s];
	nnet.calculate(x, y);
	if(nnet.inv_calculate(x, y, dropout, true) == false) return false;
      }

      std::vector< math::vertex<T> > ysamples;

      if(nnet.getSamples(ysamples, L, SAMPLES) == false)
	return false;

      nnet.clearSamples();

      // printf("Y SAMPLES DONE\n");

      ////////////////////////////////////////////////////////////////////////
      // calculates y' = g^-1(y) for the linear problems y values
      
      for(unsigned int s=0;s<ysamples.size();s++){
	auto& v = ysamples[s];

	for(unsigned int i=0;i<v.size();i++)
	  v[i] = nnet.inv_nonlin_nodropout(v[i], L, i);
      }

      // printf("Y SAMPLES INVERSE DONE\n");

      ////////////////////////////////////////////////////////////////////////
      // optimizes linear problem y' = A*x + b, solves A and b and injects them into network

      math::matrix<T> W;
      math::vertex<T> b;

      if(nnet.getWeights(W, L) == false) return false;
      if(nnet.getBias(b, L) == false) return false;

      {
	auto& input = xsamples;
	auto& output = ysamples;

	math::matrix<T> Cxx, Cxy;
	math::vertex<T> mx, my;
	
	Cxx.resize(input[0].size(),input[0].size());
	Cxy.resize(input[0].size(),output[0].size());
	mx.resize(input[0].size());
	my.resize(output[0].size());
	
	Cxx.zero();
	Cxy.zero();
	mx.zero();
	my.zero();
	
	for(unsigned int i=0;i<SAMPLES;i++){
	  Cxx += input[i].outerproduct();
	  Cxy += input[i].outerproduct(output[i]);
	  mx  += input[i];
	  my  += output[i];
	}
	
	Cxx /= T((float)SAMPLES);
	Cxy /= T((float)SAMPLES);
	mx  /= T((float)SAMPLES);
	my  /= T((float)SAMPLES);
      
	Cxx -= mx.outerproduct();
	Cxy -= mx.outerproduct(my);
	
	math::matrix<T> INV;
	T l = T(10e-3);
	
	do{
	  INV = Cxx;
	  
	  T trace = T(0.0f);
	  
	  for(unsigned int i=0;(i<(Cxx.xsize()) && (i<Cxx.ysize()));i++){
	    trace += Cxx(i,i);
	    INV(i,i) += l; // regularizes Cxx (if needed)
	  }

	  if(Cxx.xsize() < Cxx.ysize())	  
	    trace /= Cxx.xsize();
	  else
	    trace /= Cxx.ysize();
	  
	  l += (T(0.1)*trace + T(2.0f)*l); // keeps "scale" of the matrix same
	}
	while(whiteice::math::symmetric_inverse(INV) == false);

	
	if((whiteice::rng.rand() % 200) == 0){

	  math::matrix<T> A(W);
	  math::vertex<T> c(b);

	  for(unsigned int i=0;i<A.size();i++){
	    for(unsigned int k=0;k<A[i].size();k++){
	      A[i][k] = whiteice::rng.normalf();
	    }
	  }

	  for(unsigned int i=0;i<c.size();i++){
	    for(unsigned int k=0;k<c[i].size();k++){
	      c[i][k] = whiteice::rng.normalf();
	    }
	  }

	  W = T(0.250f)*W + T(0.750f)*A;
	  b = T(0.250f)*b + T(0.750f)*c;
	}
	else{
	  W = T(0.950f)*W + T(0.05f)*(Cxy.transpose() * INV);
	  b = T(0.950f)*b + T(0.05f)*(my - W*mx);
	}
      }

      

      ////////////////////////////////////////////////////////////////////////
      // sets new weights for this layer
      
      if(nnet.setWeights(W, L) == false)
	return false;
      
      if(nnet.setBias(b, L) == false)
	return false;

      // printf("CALCULATE WEIGHTS DONE\n");
    }


    // network's weights are solved using part-wise optimization (pretraining)
    // you can run this algorithm multiple times to optimize for layers
    // next: use optimizes to find best local solution for the whole network optimization problem

    return true;
  }


  //////////////////////////////////////////////////////////////////////


  // assumes whole network is linear matrix operations y = M*x, M = A*B*C*D,
  // linear M is solves from data
  // solves changes D to matrix using equation A*(B+D)*C = M => D = A^-1*M*C^-1 - B
  // solves D for each matrix and then applies changes
  // [assumes linearity so this is not very good solution] 
  template <typename T>
  bool pretrain_nnetwork_matrix_factorization(nnetwork<T>& nnet,
					      const dataset<T>& data,
					      const T step_length) // step-lenght is small 1e-5 or so
  {
    if(data.getNumberOfClusters() < 2) return false;
    if(data.size(0) != data.size(1)) return false;
    if(data.size(0) < 10) return false; // needs at least 10 data points to calculate something usable
    
    if(data.dimension(0) != nnet.input_size()) return false;
    if(data.dimension(1) != nnet.output_size()) return false;

    if(step_length <= T(0.0f) || step_length >= T(1.0)) return false;

    std::vector< math::matrix<T> > operators;

    math::matrix<T> W, A;
    math::vertex<T> b;

    for(unsigned int l=0;l<nnet.getLayers();l++){
      nnet.getWeights(W, l);
      nnet.getBias(b, l);

      A.resize(W.ysize()+1, W.xsize()+1);

      A.zero();

      for(unsigned int j=0;j<W.ysize();j++){
	for(unsigned int i=0;i<W.xsize();i++){

	  // printf("W(%d,%d) = %f\n", j, i, W(j,i).c[0]);

	  for(unsigned int k=0;k<W(j,i).size();k++){
	    if(W(j,i)[k].c[0] > (1e2)){ W(j,i)[k].c[0] = (1e2); }
	    if(W(j,i)[k].c[0] < (-1e2)){ W(j,i)[k].c[0] = (-1e2); }
	  }
	  
	  A(j,i) = W(j,i);
	}
      }
      
      for(unsigned int i=0;i<b.size();i++){

	for(unsigned int k=0;k<b[i].size();k++){
	  if(b[i][k].c[0] > (1e2)){ b[i][k].c[0] = (1e2); }
	  if(b[i][k].c[0] < (-1e2)){ b[i][k].c[0] = (-1e2); }
	}
	
	A(i, W.xsize()) = b[i];
      }

      A(A.ysize()-1, A.xsize()-1) = T(1.0f);

      operators.push_back(A);
    }

    // solves matrix M from data
    math::matrix<T> M;

    math::matrix<T> V;

    if(nnet.getWeights(W, 0) == false) return false;
    if(nnet.getBias(b, 0) == false) return false;
    
    if(nnet.getWeights(V, nnet.getLayers()-1) == false) return false;
    if(nnet.getBias(b, nnet.getLayers()-1) == false) return false;

    M.resize(V.ysize()+1, W.xsize()+1);
    M.zero();
    M(V.ysize(), W.xsize()) = T(1.0f);
    
    {
      std::vector< math::vertex<T> > input;
      std::vector< math::vertex<T> > output;

      data.getData(0, input);
      data.getData(1, output);

      while(input.size() > 200){
	const unsigned int index = whiteice::rng.rand() % input.size();
	input.erase(input.begin()+index);
	output.erase(output.begin()+index);
      }

      const unsigned int SAMPLES = input.size(); 
      
      math::matrix<T> Cxx, Cxy;
      math::vertex<T> mx, my;
      
      Cxx.resize(input[0].size(),input[0].size());
      Cxy.resize(input[0].size(),output[0].size());
      mx.resize(input[0].size());
      my.resize(output[0].size());
      
      Cxx.zero();
      Cxy.zero();
      mx.zero();
      my.zero();
      
      for(unsigned int i=0;i<SAMPLES;i++){
	Cxx += input[i].outerproduct();
	Cxy += input[i].outerproduct(output[i]);
	mx  += input[i];
	my  += output[i];
      }
      
      Cxx /= T((float)SAMPLES);
      Cxy /= T((float)SAMPLES);
      mx  /= T((float)SAMPLES);
      my  /= T((float)SAMPLES);
      
      Cxx -= mx.outerproduct();
      Cxy -= mx.outerproduct(my);
      
      math::matrix<T> INV;
      T l = T(10e-3);
      
      do{
	INV = Cxx;
	
	T trace = T(0.0f);
	
	for(unsigned int i=0;(i<(Cxx.xsize()) && (i<Cxx.ysize()));i++){
	  trace += Cxx(i,i);
	  INV(i,i) += l; // regularizes Cxx (if needed)
	}
	
	if(Cxx.xsize() < Cxx.ysize())	  
	  trace /= Cxx.xsize();
	else
	  trace /= Cxx.ysize();
	
	l += (T(0.1)*trace + T(2.0f)*l); // keeps "scale" of the matrix same
      }
      while(whiteice::math::symmetric_inverse(INV) == false);
      
      
      W = (Cxy.transpose() * INV);
      b = (my - W*mx);

#if 0
      // calculates average error
      {
	T error = T(0.0f);

	for(unsigned int i=0;i<SAMPLES;i++){
	  auto e = (W*input[i] + b) - output[i];
	  auto enorm = e.norm();
	  error += enorm*enorm;
	}

	error /= SAMPLES;
	error /= output[0].size();
	error *= T(0.50f);

	std::cout << "Average error of linear fitting to data is: "  << error << std::endl;
      }
#endif


      for(unsigned int j=0;j<W.ysize();j++)
	for(unsigned int i=0;i<W.xsize();i++)
	  M(j,i) = W(j,i);
      
      for(unsigned int i=0;i<b.size();i++)
	M(i, W.xsize()) = b[i];
    }
    
    // now we have all operators in matrix format!

    std::vector< math::matrix<T> > deltas; // delta matrices to solve for each matrix operator

    for(unsigned int l=0;l<nnet.getLayers();l++){

      math::matrix<T> LEFT, RIGHT;
      
      RIGHT.resize(operators[0].xsize(), operators[0].xsize());
      LEFT.resize(operators[operators.size()-1].ysize(), operators[operators.size()-1].ysize());
      
      LEFT.identity(); // I matrix
      RIGHT.identity(); // I matrix

      //std::cout << "LAYER: " << l << std::endl;
      //std::cout << M.ysize() << "x" << M.xsize() << std::endl;


      for(unsigned int k=0;k<l;k++){
	//std::cout << "RIGHT LAYER: " << k << std::endl;
	//std::cout << RIGHT.ysize() << "x" << RIGHT.xsize() << std::endl;
	//std::cout << operators[k].ysize() << "x" << operators[k].xsize() << std::endl;
	RIGHT = operators[k]*RIGHT;
      }

      for(unsigned int k=nnet.getLayers()-1;k>l;k--){
	//std::cout << "LEFT LAYER: " << k << std::endl;
	//std::cout << LEFT.ysize() << "x" << LEFT.xsize() << std::endl;
	//std::cout << operators[k].ysize() << "x" << operators[k].xsize() << std::endl;
	LEFT = LEFT*operators[k];
      }

#if 1
      
      for(unsigned int j=0;j<RIGHT.ysize();j++){
	for(unsigned int i=0;i<RIGHT.xsize();i++){
	  for(unsigned int k=0;k<RIGHT(j,i).size();k++){
	    if(RIGHT(j,i)[k].c[0] > (+1.0f)) RIGHT(j,i)[k].c[0] = (+1.0f);
	    if(RIGHT(j,i)[k].c[0] < (-1.0f)) RIGHT(j,i)[k].c[0] = (-1.0f);
	  }
	}
      }

      for(unsigned int j=0;j<LEFT.ysize();j++){
	for(unsigned int i=0;i<LEFT.xsize();i++){
	  for(unsigned int k=0;k<LEFT(j,i).size();k++){
	    if(LEFT(j,i)[k].c[0] > (+1.0f)) LEFT(j,i)[k].c[0] = (+1.0f);
	    if(LEFT(j,i)[k].c[0] < (-1.0f)) LEFT(j,i)[k].c[0] = (-1.0f);
	  }
	}
      }
#endif

      // calculating pseudoinverse may require regularization.. 
      {
	math::matrix<T> INV;
	T l = T(1e-3);
	
	do{
	  INV = LEFT;

	  T trace = T(0.0f);

	  for(unsigned int i=0;(i<(INV.xsize()) && (i<INV.ysize()));i++){
	    trace += INV(i,i);
	    INV(i,i) += l; // regularizes matrix (if needed)
	  }

	  if(INV.xsize() < INV.ysize())	  
	    trace /= INV.xsize();
	  else
	    trace /= INV.ysize();

	  l += (T(0.1)*trace + T(2.0f)*l); // keeps "scale" of the matrix same
	}
	while(INV.pseudoinverse() == false);

	LEFT = INV;
      }
      
      
      // calculating pseudoinverse may require regularization.. 
      {
	math::matrix<T> INV;
	T l = T(1e-3);
	
	do{
	  INV = RIGHT;

	  T trace = T(0.0f);

	  for(unsigned int i=0;(i<(INV.xsize()) && (i<INV.ysize()));i++){
	    trace += INV(i,i);
	    INV(i,i) += l; // regularizes matrix (if needed)
	  }

	  if(INV.xsize() < INV.ysize())	  
	    trace /= INV.xsize();
	  else
	    trace /= INV.ysize();
	  
	  l += (T(0.1)*trace + T(2.0f)*l); // keeps "scale" of the matrix same
	}
	while(INV.pseudoinverse() == false);

	RIGHT = INV;
      }
      
      
      
      auto DELTA = LEFT*M*RIGHT;

      deltas.push_back(DELTA);
    }

    // does Ployak averaging/moving average and keeps only 10% of matrix weight changes
    
    for(unsigned int l=0;l<nnet.getLayers();l++)
      {
#if 0
	if((whiteice::rng.rand()%(31*nnet.getLayers()))==0){ // was 1000, which means 31 for two runs which must both happen..

	  // printf("RANDOM MATRIX\n");

	  // sets weights to random values! (jumps out of local minimum)
	  
	  nnet.getWeights(W, l);
	  nnet.getBias(b, l);
	  
	  A.resize(W.ysize()+1, W.xsize()+1);
	  
	  A = operators[l];
	  
	  if(A.ysize() != W.ysize()+1 || W.xsize()+1 != A.xsize())
	    return false; // extra check
	  
	  for(unsigned int j=0;j<W.ysize();j++){
	    for(unsigned int i=0;i<W.xsize();i++){
	      for(unsigned int k=0;k<A(j,i).size();k++){
		A(j,i)[k] = (0.5f) * whiteice::rng.normalf();
	      }
	    }
	  }
	  
	  for(unsigned int i=0;i<b.size();i++){
	    for(unsigned int k=0;k<b[i].size();k++){
	      A(i, W.xsize())[k] = (0.5f) * whiteice::rng.normalf();
	    }
	  }

	  operators[l] = A;
	}
	else
#endif
	{
	  //std::cout << "l = " << l << " op.size = " << operators.size() << " : delta.size = " << deltas.size() << std::endl;
	  operators[l] = (T(1.0)-step_length)*operators[l] + step_length*deltas[l];

#if 1
	  auto& W = operators[l];

	  for(unsigned int j=0;j<(W.ysize()-1);j++){
	    for(unsigned int i=0;i<W.xsize();i++){
	      for(unsigned int k=0;k<W(j,i).size();k++){
		if(W(j,i)[k] < -1.0f) W(j,i)[k] = -1.00f;
		else if(W(j,i)[k] > 1.00f) W(j,i)[k] = 1.00f;
	      }
	    }
	  }
#endif

	}
      }
    
    
    
    
    // finally solve parameters for W*x+b linear equation each per layer
    
    for(unsigned int l=0;l<nnet.getLayers();l++){
      nnet.getWeights(W, l);
      nnet.getBias(b, l);

      A.resize(W.ysize()+1, W.xsize()+1);

      A = operators[l];

      if(A.ysize() != W.ysize()+1 || W.xsize()+1 != A.xsize())
	return false; // extra check

      for(unsigned int j=0;j<W.ysize();j++){
	for(unsigned int i=0;i<W.xsize();i++){

	  for(unsigned int k=0;k<A(j,i).size();k++){
	    if(A(j,i)[k].c[0] > (+1e2f)) A(j,i)[k].c[0] = (+1e2f);
	    if(A(j,i)[k].c[0] < (-1e2f)) A(j,i)[k].c[0] = (-1e2f);
	  }
	  
	  W(j,i) = A(j,i);
	}
      }

      for(unsigned int i=0;i<b.size();i++){
	for(unsigned int k=0;k<A(i,W.xsize()).size();k++){
	  if(A(i, W.xsize())[k].c[0] > (+1e2)) A(i, W.xsize())[k].c[0] = (+1e2f);
	  if(A(i, W.xsize())[k].c[0] < (-1e2)) A(i, W.xsize())[k].c[0] = (-1e2f);
	}
	
	b[i] = A(i, W.xsize());
      }

      
      nnet.setWeights(W, l);
      nnet.setBias(b, l);
    }

    return true;
  }
  
  

  //////////////////////////////////////////////////////////////////////
  

  template bool pretrain_nnetwork< math::blas_real<float> >
  (nnetwork< math::blas_real<float> >& nnet, const dataset< math::blas_real<float> >& data);
  
  template bool pretrain_nnetwork< math::blas_real<double> >
  (nnetwork< math::blas_real<double> >& nnet, const dataset< math::blas_real<double> >& data);

  
  template bool pretrain_nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >
  (nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >& nnet,
   const dataset< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >& data);
  
  template bool pretrain_nnetwork< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >
  (nnetwork< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >& nnet,
   const dataset< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >& data);

  

  template bool pretrain_nnetwork_matrix_factorization< math::blas_real<float> >
  (nnetwork< math::blas_real<float> >& nnet, const dataset< math::blas_real<float> >& data,
   const math::blas_real<float> step_length);
  
  template bool pretrain_nnetwork_matrix_factorization< math::blas_real<double> >
  (nnetwork< math::blas_real<double> >& nnet, const dataset< math::blas_real<double> >& data,
   const math::blas_real<double> step_length);

  
  
  template bool pretrain_nnetwork_matrix_factorization< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >
  (nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >& nnet,
   const dataset< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >& data,
   const math::superresolution< math::blas_real<float>, math::modular<unsigned int> > step_length);
  
  template bool pretrain_nnetwork_matrix_factorization< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >
  (nnetwork< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >& nnet,
   const dataset< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >& data,
   const math::superresolution< math::blas_real<double>, math::modular<unsigned int> > step_length);
  
};
