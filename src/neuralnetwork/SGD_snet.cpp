
#include "SGD_snet.h"

using namespace whiteice::math;

// Batch Normalization dataset size
#define BN_SIZE 1000

namespace whiteice
{

  template <typename T>
  SGD_snet<T>::SGD_snet(const nnetwork< superresolution< T, modular<unsigned int> > >& nn,
			const dataset<T>& d,
			bool overfit):
    whiteice::math::SGD< superresolution<T, modular<unsigned int> > >(overfit),
    net(nn), data(d)
  {
    assert(data.getNumberOfClusters() == 2);
    assert(data.size(0) == data.size(1));
    assert(data.size(0) >= 1); // must have at least one data point

    // checks network has correct architecture
    {
      assert(net.input_size() == data.dimension(0));
      assert(net.output_size() == data.dimension(1));
    }
    

    // divides data in episoids to to training and testing sets
    ///////////////////////////////////////////////
    {
      dtrain = data;
      dtest  = data;
      
      dtrain.clearData(0);
      dtrain.clearData(1);
      
      dtest.clearData(0);
      dtest.clearData(1);
      
      
      for(unsigned int e=0;e<data.size(0);e++){
	
	const unsigned int r = (rng.rand() % 4);
	
	if(r != 0){ // training dataset 75% of cases go here (training set)

	  math::vertex<T> in  = data.access(0,e);
	  math::vertex<T> out = data.access(1,e);
	  
	  dtrain.add(0, in,  true);
	  dtrain.add(1, out, true);
	}
	else{ // testing dataset
	  math::vertex<T> in  = data.access(0,e);
	  math::vertex<T> out = data.access(1,e);
	  
	  dtest.add(0, in,  true);
	  dtest.add(1, out, true);
	}
      }
      
      // we cannot never have zero training or testing set size
      // in such a small cases (very little data) we just use
      // all the data both for training and testing and overfit
      if(dtrain.size(0) == 0 || dtest.size(0) == 0 || overfit){
	dtrain = data;
	dtest  = data;
      }
    }
    
  }

  
  template <typename T>
  SGD_snet<T>::~SGD_snet()
  {
  }


  template <typename T>
  superresolution<T, modular<unsigned int> > SGD_snet<T>::getError
  (const math::vertex< superresolution<T, modular<unsigned int> > >& x) const
  {
    superresolution< T, modular<unsigned int> > e(0.0f);

    { 

      whiteice::nnetwork< superresolution<T, modular<unsigned int> > > nnet(this->net);
      nnet.importdata(x);

      
      // batch normalization
      if(nnet.getBatchNorm() && dtest.size(0) > 0){
	// TODO: get only random 1000 data points and use only them
	std::vector< math::vertex< math::superresolution< T, math::modular<unsigned int> > > > sdata;
	std::vector< math::vertex<T> > data;
	math::vertex< math::superresolution< T, math::modular<unsigned int> > > s;

	// dtest.getData(0, data);

	for(unsigned int i=0;i<BN_SIZE;i++){
	  const unsigned int index = rng.rand() % dtest.size(0);
	  data.push_back(dtest.access(0,index));
	}
	
	s.resize(data[0].size());
	
	for(const auto& d : data){
	  for(unsigned int i=0;i<d.size();i++)
	    whiteice::math::convert(s[i],d[i]);
	  
	  sdata.push_back(s);
	}
	
	assert(nnet.calculateBatchNorm(sdata) == true);
      }
      
      
#pragma omp parallel shared(e)
      {
	superresolution< T, modular<unsigned int> > threaded_error(0.0f);
	
	vertex< superresolution< T, modular<unsigned int> > > err;

	math::vertex<
	  math::superresolution< T,
				 math::modular<unsigned int> > > sx, sy;
	
#pragma omp for nowait
	for(unsigned int i=0;i<dtest.size(0);i++){

	  const auto x = dtest.access(0,i);
	  const auto y = dtest.access(1,i);

	  whiteice::math::convert(sx, x);
	  whiteice::math::convert(sy, y);
	  
	  nnet.calculate(sx, err);
	  err -= sy;
	    
	  for(unsigned int j=0;j<err.size();j++){
	    const auto& ej = err[j];

	    //for(unsigned int k=0;k<ej.size();k++)
	    const unsigned int k = 0;
	    threaded_error += math::sqrt(ej[k]*math::conj(ej[k])); // ABSOLUTE VALUE
	  }
	}
	
#pragma omp critical
	{
	  e += threaded_error;
	}
	
      }
      
    }
    
    e /= T( (float)dtest.size(0)*dtest.dimension(1) ); // per N*dim(output)
    
    return e;
  }

  

  template <typename T>
  superresolution<T, modular<unsigned int> > SGD_snet<T>::U
  (const math::vertex< superresolution<T, modular<unsigned int> > >& x) const
  {
    superresolution< T, modular<unsigned int> > e(0.0f);
    
    { 
      whiteice::nnetwork< superresolution<T, modular<unsigned int> > > nnet(this->net);
      nnet.importdata(x);

      // batch normalization
      if(nnet.getBatchNorm() && dtrain.size(0) > 0){
	// TODO: get only random 1000 data points and use only them
	std::vector< math::vertex< math::superresolution< T, math::modular<unsigned int> > > > sdata;
	std::vector< math::vertex<T> > data;
	math::vertex< math::superresolution< T, math::modular<unsigned int> > > s;
	
	// dtrain.getData(0, data);

	for(unsigned int i=0;i<BN_SIZE;i++){
	  const unsigned int index = rng.rand() % dtrain.size(0);
	  data.push_back(dtrain.access(0,index));
	}
	
	s.resize(data[0].size());
	
	for(const auto& d : data){
	  for(unsigned int i=0;i<d.size();i++)
	    whiteice::math::convert(s[i],d[i]);
	  
	  sdata.push_back(s);
	}
	
	assert(nnet.calculateBatchNorm(sdata) == true);
      }
      
      
#pragma omp parallel shared(e)
      {
	superresolution< T, modular<unsigned int> > threaded_error(0.0f);
	
	vertex< superresolution< T, modular<unsigned int> > > err;

	math::vertex<
	  math::superresolution< T,
				 math::modular<unsigned int> > > sx, sy;

#pragma omp for nowait
	for(unsigned int i=0;i<dtrain.size(0);i++){

	  const auto x = dtrain.access(0,i);
	  const auto y = dtrain.access(1,i);

	  whiteice::math::convert(sx, x);
	  whiteice::math::convert(sy, y);
	  
	  nnet.calculate(sx, err);
	  err -= sy;
	    
	  for(unsigned int j=0;j<err.size();j++){
	    const auto& ej = err[j];

	    //for(unsigned int k=0;k<ej.size();k++)
	    const unsigned int k = 0;
	    threaded_error += math::sqrt(ej[k]*math::conj(ej[k])); // ABSOLUTE VALUE
	    
	  }
	}
	
#pragma omp critical
	{
	  e += threaded_error;
	}
	
      }
      
    }
    
    e /= T( (float)dtest.size(0)*dtrain.dimension(1) ); // per N*dim(output)
    

    // no REGULARIZER FOR
#if 0
    {
      // regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
      auto err = T(0.5)*alpha*(x*x)[0];
      e += err;
    }
#endif
    
    return (e);    
  }

  
  
  template <typename T>
  math::vertex< superresolution<T, modular<unsigned int> > >
  SGD_snet<T>::Ugrad
  (const math::vertex< superresolution<T, modular<unsigned int> > >& x) const
  {
    whiteice::nnetwork< superresolution<T, modular<unsigned int> > > snet(this->net);
    snet.importdata(x);

    // batch normalization
    if(snet.getBatchNorm() && dtrain.size(0) > 0){
      // TODO: get only random 1000 data points and use only them
      std::vector< math::vertex< math::superresolution< T, math::modular<unsigned int> > > > sdata;
      std::vector< math::vertex<T> > data;
      math::vertex< math::superresolution< T, math::modular<unsigned int> > > s;
      
      // dtrain.getData(0, data);
      
      for(unsigned int i=0;i<BN_SIZE;i++){
	const unsigned int index = rng.rand() % dtrain.size(0);
	data.push_back(dtrain.access(0,index));
      }
      
      
      s.resize(data[0].size());

      for(const auto& d : data){
	for(unsigned int i=0;i<d.size();i++)
	  whiteice::math::convert(s[i],d[i]);

	sdata.push_back(s);
      }
      
      assert(snet.calculateBatchNorm(sdata) == true);
    }

    
    math::vertex< math::superresolution<T,
					math::modular<unsigned int> > > sumgrad;

    sumgrad.resize(snet.exportdatasize());
    sumgrad.zero();
    
    math::superresolution<T,
			  math::modular<unsigned int> > ninv =
      math::superresolution<T,
			    math::modular<unsigned int> >
      (1.0f/(dtrain.size(0)*dtrain.access(1,0).size()));
    
    
#pragma omp parallel
    {
      math::vertex< math::superresolution<T,
					  math::modular<unsigned int> > > grad, err, threaded_grad;
      
      math::vertex<
	math::superresolution< T,
			       math::modular<unsigned int> > > sx, sy;
      
      threaded_grad.resize(snet.exportdatasize());
      threaded_grad.zero();

      math::matrix< math::superresolution<T,
					  math::modular<unsigned int> > > DF;
      
      math::matrix< math::superresolution<math::blas_complex<double>,
					  math::modular<unsigned int> > > cDF;

      math::vertex<
	math::superresolution< math::blas_complex<double>,
			       math::modular<unsigned int> > > ce, cerr;
      
#pragma omp for nowait
      for(unsigned int i=0;i<dtrain.size(0);i++){
	
	{
	  const auto x = dtrain.access(0,i);
	  const auto y = dtrain.access(1,i);

	  whiteice::math::convert(sx, x);
	  whiteice::math::convert(sy, y);

	  snet.calculate(sx, err);
	  err -= sy;
	  
	  snet.jacobian(sx, DF);
	  cDF.resize(DF.ysize(), DF.xsize());
	  
	  // circular convolution in F-domain
	  
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
      }
      
    }

    // NO REGULARIZER CODE HERE!
    {
      
    }

    return sumgrad;
  }
  
  
  template <typename T>
  bool SGD_snet<T>::heuristics
  (math::vertex< superresolution<T, modular<unsigned int> > >& x) const
  {
#if 0
    // box-values
    // don't allow values larger than 10^4
    
    for(unsigned int i=0;i<x.size();i++){
      for(unsigned int k=0;k<x[i].size();k++){
	if(x[i][k] > T(1e4)) x[i][k] = T(1e4);
	else if(x[i][k] < T(-1e4)) x[i][k] = T(-1e4);
      }
    }
#endif
    
    return true;
  }
  
  
  
  template class SGD_snet< math::blas_real<float> >;
  template class SGD_snet< math::blas_real<double> >;
  
};
