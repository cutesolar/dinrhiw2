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
    
    if(whiten1d_nnetwork(nnet, data) == false) return false;
       
    // optimizes each layers linear parameters, first last layer

    const unsigned int SAMPLES = ((data.size() > 5000) ? 5000 : data.size());
    std::vector< math::vertex<T> > samples;
    

    for(unsigned int int l=(nnet.getLayers()),l>0;l--){
      const unsigned int L = l-1;

      // first calculates x values for the layer

      for(unsigned int s=0;SAMPLES;s++){
	const unsigned int index = rand() % data.size(0);
	samples.push_back(data.access(0, index));
	nnet.input() = data.access(0, index);
	if(nnet.calculate(false, true) == false) return false;
      }

      std::vector< math::vertex<T> > xsamples;

      if(nnet.getSamples(xsamples, L, SAMPLES) == false)
	return false;

      nnet.clearSamples();
      
      // next calculates y values for the layer

      for(unsigned int s=0;SAMPLES;s++){
	nnet.output() = samples[s];
	if(nnet.inv_calculate(true) == false) return false;
      }

      std::vector< math::vertex<T> > ysamples;

      if(nnet.getSamples(ysamples, L, SAMPLES) == false)
	return false;

      nnet.clearSamples();

      // calculates y' = g^-1(y) for the linear problems y values
      
      for(unsigned int s=0;s<ysamples.size();s++){
	auto& v = ysamples[s];

	for(unsigned int i=0;i<v.size();i++)
	  v[i] = nnet.inv_nonlin(v[i], L, i);
      }

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
	
	for(unsigned int i=0;i<N;i++){
	  Cxx += input[i].outerproduct();
	  Cxy += input[i].outerproduct(output[i]);
	  mx  += input[i];
	  my  += output[i];
	}
	
	Cxx /= T((float)N);
	Cxy /= T((float)N);
	mx  /= T((float)N);
	my  /= T((float)N);
      
	Cxx -= mx.outerproduct();
	Cxy -= mx.outerproduct(my);
	
	math::matrix<T> INV;
	T l = T(10e-6);
	
	do{
	  INV = Cxx;
	  
	  T trace = T(0.0f);
	  
	  for(unsigned int i=0;(i<Cxx.xsize()) && (i<Cxx.ysize())++){
	    trace += Cxx(i,i);
	    INV(i,i) += l; // regularizes Cxx (if needed)
	  }

	  if(Cxx.xsize() < Cxx.ysize())	  
	    trace /= Cxx.xsize();
	  else
	    trace /= Cxx.ysize();
	  
	  l += T(0.1)*(trace + T(2.0f)*l); // keeps "scale" of the matrix same
	}
	while(whiteice::math::symmetric_inverse(INV) == false);
	
	
	W = Cxy.transpose() * INV;
	b = my - W*mx;
      }

      // sets new weights for this layer
      
      if(nnet.setWeights(W, L) == false)
	return false;
      
      if(nnet.setBias(b, L) == false)
	return false;
      
    }


    // network's weights are solved using part-wise optimization (pretraining)
    // you can run this algorithm multiple times to optimize for layers
    // next: use optimizes to find best local solution for the whole network optimization problem

    return true;
  }
  

  //////////////////////////////////////////////////////////////////////
  

  template bool pretrain_nnetwork< math::blas_real<float> >
  (nnetwork< math::blas_real<float> >& nnet, const dataset< math::blas_real<float> >& data);
  
  template bool pretrain_nnetwork< math::blas_real<double> >
  (nnetwork< math::blas_real<double> >& nnet, const dataset< math::blas_real<double> >& data);

  
};
