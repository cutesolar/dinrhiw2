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



namespace whiteice
{
  
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

	  for(unsigned int i=0;i<A.size();i++)
	    A[i] = whiteice::rng.normal();

	  for(unsigned int i=0;i<c.size();i++)
	    c[i] = whiteice::rng.normal();

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

	  printf("W(%d,%d) = %f\n", j, i, W(j,i).c[0]);
	  
	  if(W(j,i) > T(1e1)){ W(j,i) = T(1e1); }
	  if(W(j,i) < T(-1e1)){ W(j,i) = T(-1e1); }
	  
	  A(j,i) = W(j,i);
	}
      }
      
      for(unsigned int i=0;i<b.size();i++){
	
	if(b[i] > T(1e1)){ b[i] = T(1e1); }
	if(b[i] < T(-1e1)){ b[i] = T(-1e1); }
	
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
      
      LEFT.resize(operators[0].xsize(), operators[0].xsize());
      RIGHT.resize(operators[operators.size()-1].ysize(), operators[operators.size()-1].ysize());
      
      LEFT.identity(); // I matrix
      RIGHT.identity(); // I matrix

      //std::cout << "LAYER: " << l << std::endl;
      //std::cout << LEFT.ysize() << "x" << LEFT.xsize() << std::endl;
      //std::cout << M.ysize() << "x" << M.xsize() << std::endl;
      //std::cout << RIGHT.ysize() << "x" << RIGHT.xsize() << std::endl;


      for(unsigned int k=0;k<l;k++){
	//std::cout << "LAYER: " << k << std::endl;
	//std::cout << operators[k].ysize() << "x" << operators[k].xsize() << std::endl;
	RIGHT = operators[k]*RIGHT;
      }

      for(unsigned int k=nnet.getLayers()-1;k>l;k--){
	//std::cout << "LAYER: " << k << std::endl;
	//std::cout << operators[k].ysize() << "x" << operators[k].xsize() << std::endl;
	LEFT *= operators[k];
      }

      for(unsigned int j=0;j<RIGHT.ysize();j++)
	for(unsigned int i=0;i<RIGHT.xsize();i++){
	  if(RIGHT(j,i) > T(+1e4f)) RIGHT(j,i) = T(+1e4f);
	  if(RIGHT(j,i) < T(-1e4f)) RIGHT(j,i) = T(-1e4f);
	}

      for(unsigned int j=0;j<LEFT.ysize();j++)
	for(unsigned int i=0;i<LEFT.xsize();i++){
	  if(LEFT(j,i) > T(+1e4f)) LEFT(j,i) = T(+1e4f);
	  if(LEFT(j,i) < T(-1e4f)) LEFT(j,i) = T(-1e4f);
	}

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

      // does Ployak averaging/moving average and keeps only 10% of matrix weight changes
      
      //for(unsigned int l=0;l<nnet.getLayers();l++)
      {
	if((whiteice::rng.rand()%1000)==0){

	  // sets weights to random values! (jumps out of local minimum)
	  
	  nnet.getWeights(W, l);
	  nnet.getBias(b, l);
	  
	  A.resize(W.ysize()+1, W.xsize()+1);
	  
	  A = operators[l];
	  
	  if(A.ysize() != W.ysize()+1 || W.xsize()+1 != A.xsize())
	    return false; // extra check
	  
	  for(unsigned int j=0;j<W.ysize();j++)
	    for(unsigned int i=0;i<W.xsize();i++)
	      A(j,i) = T(5.0f) * whiteice::rng.normal();
	  
	  for(unsigned int i=0;i<b.size();i++)
	    A(i, W.xsize()) = T(5.0f) * whiteice::rng.normal();

	  operators[l] = A;
	}
	else{
	  operators[l] = (T(1.0)-step_length)*operators[l] + step_length*deltas[l];
	}
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

	  if(A(j,i) > T(+1e1f)) A(j,i) = T(+1e1f);
	  if(A(j,i) < T(-1e1f)) A(j,i) = T(-1e1f);
	  
	  W(j,i) = A(j,i);
	}
      }

      for(unsigned int i=0;i<b.size();i++){
	if(A(i, W.xsize()) > T(+1e1f)) A(i, W.xsize()) = T(+1e1f);
	if(A(i, W.xsize()) < T(-1e1f)) A(i, W.xsize()) = T(-1e1f);
	
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


  template bool pretrain_nnetwork_matrix_factorization< math::blas_real<float> >
  (nnetwork< math::blas_real<float> >& nnet, const dataset< math::blas_real<float> >& data,
   const math::blas_real<float> step_length);
  
  template bool pretrain_nnetwork_matrix_factorization< math::blas_real<double> >
  (nnetwork< math::blas_real<double> >& nnet, const dataset< math::blas_real<double> >& data,
   const math::blas_real<double> step_length);

  
};
