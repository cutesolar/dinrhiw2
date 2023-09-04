

#include "rUHMC.h"



namespace whiteice
{

  template <typename T>
  rUHMC<T>::rUHMC(const whiteice::nnetwork<T>& net, const whiteice::dataset<T>& ds,
		  bool adaptive, T alpha, bool store, bool restart_sampler) :
    UHMC<T>(net, ds, adaptive, alpha, store, restart_sampler)
  {
    // checks recurrent dimensions are correct and detects it

    assert(ds.getNumberOfClusters() >= 2);

    assert(net.input_size() >= ds.dimension(0));
    assert(net.output_size() >= ds.dimension(1));

    RDIM = net.input_size() - ds.dimension(0);

    assert(RDIM == net.output_size() - ds.dimension(1));
  }

  template <typename T>
  rUHMC<T>::~rUHMC()
  {
    
  }


  // probability functions for hamiltonian MC sampling
  template <typename T>
  T rUHMC<T>::U(const math::vertex<T>& q, bool useRegularizer) const
  {
    T e = T(0.0f);

    { // recurrent neural network

      whiteice::nnetwork<T> nnet(this->nnet);
      nnet.importdata(q);

      const bool real_error = true;

      if(this->use_minibatch){

	const unsigned int SAMPLES_MINIBATCH = (1000 > this->data.size(0)) ? this->data.size(0) : 1000;
	
#pragma omp parallel shared(e)
	{
	  //whiteice::nnetwork<T> nnet(this->net);
	  //nnet.importdata(x);
	  
	  math::vertex<T> err, correct;
	  T esum = T(0.0f);
	  
	  const unsigned int INPUT_DATA_DIM = this->data.dimension(0);
	  const unsigned int OUTPUT_DATA_DIM = this->data.dimension(1);
	  const unsigned int RDIM = this->nnet.output_size() - OUTPUT_DATA_DIM;
	  
	  math::vertex<T> input, output, output_r;
	  input.resize(this->data.dimension(0)+RDIM);
	  output.resize(this->data.dimension(1)+RDIM);
	  output_r.resize(RDIM);
	  err.resize(this->data.dimension(1));
	  
	  // E = SUM 0.5*e(i)^2
	  //for(unsigned int episode=0;episode<dtrain.size(2);episode++)
	  {
	    
	    //math::vertex<T> range = dtrain.access(2,episode);
	    
	    //unsigned int start = 0; 
	    //unsigned int length = drain.size(0);
	    
	    //whiteice::math::convert(start, range[0]);
	    //whiteice::math::convert(length, range[1]);
	    
	    input.zero();
	    unsigned int i = rng.rand() % (this->data.size(0));
	    
	    // recurrency: feebacks output back to inputs and
	    //             calculates error
#pragma omp for nowait schedule(auto)
	    for(unsigned int index = 0;index<SAMPLES_MINIBATCH;index++){
	      
	      input.write_subvertex(this->data.access(0, i), 0);
	      
	      //nnet.input() = input;
	      //nnet.calculate(false);
	      nnet.calculate(input, output);
	      
	      output.subvertex(output_r, this->data.dimension(1), RDIM);
	      assert(input.write_subvertex(output_r, INPUT_DATA_DIM));
	      
	      output.subvertex(err, 0, this->data.dimension(1));
	      
	      correct = this->data.access(1, i);
	      
	      if(real_error){
		this->data.invpreprocess(1, err);
		this->data.invpreprocess(1, correct);
	      }
	      
	      err -= correct;
	      
	      esum += T(0.5f)*(err*err)[0];

	      i++;
	      i = i % (this->data.size(0));
	    }
	    
	  }
	  
#pragma omp critical
	  {
	    e += esum; // per each recurrency
	  }
	  
	}

	e *= T(((float)this->data.size(0))/((float)SAMPLES_MINIBATCH));
	
	e /= this->sigma2;
	
	e /= this->temperature;
      }
      else{

#pragma omp parallel shared(e)
	{
	  //whiteice::nnetwork<T> nnet(this->net);
	  //nnet.importdata(x);
	  
	  math::vertex<T> err, correct;
	  T esum = T(0.0f);
	  
	  const unsigned int INPUT_DATA_DIM = this->data.dimension(0);
	  const unsigned int OUTPUT_DATA_DIM = this->data.dimension(1);
	  const unsigned int RDIM = nnet.output_size() - OUTPUT_DATA_DIM;
	  
	  math::vertex<T> input, output, output_r;
	  input.resize(this->data.dimension(0)+RDIM);
	  output.resize(this->data.dimension(1)+RDIM);
	  output_r.resize(RDIM);
	  err.resize(this->data.dimension(1));
	  
	  // E = SUM 0.5*e(i)^2
	  //for(unsigned int episode=0;episode<this->data.size(2);episode++)
	  {
	    
	    //math::vertex<T> range = this->data.access(2,episode);
	    
	    unsigned int start = 0; 
	    unsigned int length = this->data.size(0);
	    
	    //whiteice::math::convert(start, range[0]);
	    //whiteice::math::convert(length, range[1]);
	    
	    input.zero();
	    
	    // recurrency: feebacks output back to inputs and
	    //             calculates error
#pragma omp for nowait schedule(auto)
	    for(unsigned int i = start;i<length;i++){
	      input.write_subvertex(this->data.access(0, i), 0);
	      
	      //nnet.input() = input;
	      //nnet.calculate(false);
	      nnet.calculate(input, output);
	      
	      output.subvertex(output_r, this->data.dimension(1), RDIM);
	      assert(input.write_subvertex(output_r, INPUT_DATA_DIM));
	      
	      output.subvertex(err, 0, this->data.dimension(1));
	      
	      correct = this->data.access(1, i);
	      
	      if(real_error){
		this->data.invpreprocess(1, err);
		this->data.invpreprocess(1, correct);
	      }
	      
	      err -= correct;
	      
	      esum += T(0.5f)*(err*err)[0];
	    }
	    
	  }
	  
#pragma omp critical
	  {
	    e += esum; // per each recurrency
	  }
	  
	}

	e /= this->sigma2;
	
	e /= this->temperature;
	
      }
    }
    
    
#if 1
    if(useRegularizer){
      // regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
      auto err = T(0.5)*(this->alpha)*(q*q)[0];
      e += err;
    }
#endif
    
    return (e);    
  }

  

  template <typename T>
  math::vertex<T> rUHMC<T>::Ugrad(const math::vertex<T>& q, bool useRegularizer) const
  {

    if(this->use_minibatch)
    {
      const unsigned int SAMPLES_MINIBATCH = (1000 > this->data.size(0)) ? this->data.size(0) : 1000;

      // recurrent neural network!
      math::vertex<T> sumgrad;
      sumgrad = q;
      sumgrad.zero();

      const unsigned int INPUT_DATA_DIM = this->data.dimension(0);
      const unsigned int OUTPUT_DATA_DIM = this->data.dimension(1);
      const unsigned int RDIM = this->nnet.output_size() - OUTPUT_DATA_DIM;
      const unsigned int RDIM2 = this->nnet.input_size() - INPUT_DATA_DIM;
      assert(RDIM == RDIM2);

      whiteice::nnetwork<T> nnet(this->nnet);
      nnet.importdata(q);

      unsigned int counter = 0;
      
#pragma omp parallel shared(sumgrad) shared(counter)
      {
	// whiteice::nnetwork<T> nnet(this->net);
	// nnet.importdata(x);
	
	math::vertex<T> grad, err;
	math::vertex<T> sgrad;
	sgrad = q;
	sgrad.zero();
	grad = q;

	math::vertex<T> input, output, output_r;
	input.resize(this->data.dimension(0)+RDIM);
	output_r.resize(RDIM);

	math::matrix<T> UGRAD;
	UGRAD.resize(this->data.dimension(1)+RDIM, nnet.gradient_size());

	math::matrix<T> URGRAD;
	URGRAD.resize(RDIM, nnet.gradient_size());

	math::matrix<T> UYGRAD;
	UYGRAD.resize(this->data.dimension(1), nnet.gradient_size());

	math::matrix<T> FGRAD;
	FGRAD.resize(this->data.dimension(1)+RDIM, nnet.gradient_size());

	math::matrix<T> FRGRAD;
	FRGRAD.resize(RDIM, nnet.output_size());

	math::matrix<T> FGRADTMP;
	FGRADTMP.resize(this->data.dimension(1)+RDIM, RDIM);

	//#pragma omp for nowait schedule(auto)
	while(counter < SAMPLES_MINIBATCH){
	  //for(unsigned int episode=0;episode<this->data.size(2);episode++){
	  
	  //math::vertex<T> range = this->data.access(2,episode);

	  unsigned int start = rng.rand() % this->data.size(0); 
	  unsigned int length = start + 10;
	  if(length >= this->data.size(0)) length = this->data.size(0);

	  //whiteice::math::convert(start, range[0]);
	  //whiteice::math::convert(length, range[1]);
	  
	  UGRAD.zero();
	  grad.zero();
	  input.zero();
	  
	  for(unsigned int i=start;i<length;i++){
	    input.write_subvertex(this->data.access(0,i), 0);
	      
	    assert(nnet.jacobian(input, FGRAD) == true);
	    // df/dw (this->data.dimension(1)+RDIM, nnet.gradient_size())

	    {
	      assert(nnet.gradient_value(input, FGRADTMP) == true);
	      // df/dinput (this->data.dimension(1)+RDIM,this->data.dimension(0)+RDIM)

	      // df/dr
	      assert(FGRADTMP.submatrix(FRGRAD,
					this->data.dimension(0), 0,
					RDIM, nnet.output_size()) == true);

	      // KAPPA_r = I

	      // df/dr (this->data.dimension(1)+RDIM, RDIM)
	      // dU/dw (this->data.dimension(1)+RDIM, nnet.gradient_size())

	      // KAPPA_r operation to UGRAD to select only R terms
	      assert(UGRAD.submatrix(URGRAD,
				     0,this->data.dimension(1),
				     nnet.gradient_size(), RDIM) == true);
	    }

	    // dU(n+1)/dw = df/dw + df/dr * KAPPA_r * dU(n)/dw
	    UGRAD = FGRAD + FRGRAD*URGRAD;

	    //nnet.input() = input;
	    //nnet.calculate(false);
	    nnet.calculate(input, output);
	    
	    { // calculate error gradient value for E(i=0)..E(N) terms
	      
	      output.subvertex(err, 0, this->data.dimension(1));
	      err -= this->data.access(1,i);

	      // selects only Y terms from UGRAD
	      assert(UGRAD.submatrix
		     (UYGRAD,
		      0,0,
		      nnet.gradient_size(), this->data.dimension(1)));

	      grad = err*UYGRAD;

	      sgrad += grad;
	    }

	    assert(output.subvertex(output_r, this->data.dimension(1), RDIM));
	    assert(input.write_subvertex(output_r, INPUT_DATA_DIM));
	  }

#pragma omp critical
	  {
	    counter += (length-start);
	  }

	}
	
#pragma omp critical
	{
	  sumgrad += sgrad;
	}
	
      }

      sumgrad /= this->sigma2;
      sumgrad /= this->temperature; // scales gradient with temperature
      
#if 1
      if(useRegularizer){
	// regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
	
	sumgrad += this->alpha*q;
      }
#endif
      
      return sumgrad;
      
    }
    else
    {
      // recurrent neural network!
      math::vertex<T> sumgrad;
      sumgrad = q;
      sumgrad.zero();

      const unsigned int INPUT_DATA_DIM = this->data.dimension(0);
      const unsigned int OUTPUT_DATA_DIM = this->data.dimension(1);
      const unsigned int RDIM = this->nnet.output_size() - OUTPUT_DATA_DIM;
      const unsigned int RDIM2 = this->nnet.input_size() - INPUT_DATA_DIM;
      assert(RDIM == RDIM2);

      whiteice::nnetwork<T> nnet(this->nnet);
      nnet.importdata(q);
      
#pragma omp parallel shared(sumgrad)
      {
	// whiteice::nnetwork<T> nnet(this->net);
	// nnet.importdata(x);
	
	math::vertex<T> grad, err;
	math::vertex<T> sgrad;
	sgrad = q;
	sgrad.zero();
	grad = q;

	math::vertex<T> input, output, output_r;
	input.resize(this->data.dimension(0)+RDIM);
	output_r.resize(RDIM);

	math::matrix<T> UGRAD;
	UGRAD.resize(this->data.dimension(1)+RDIM, nnet.gradient_size());

	math::matrix<T> URGRAD;
	URGRAD.resize(RDIM, nnet.gradient_size());

	math::matrix<T> UYGRAD;
	UYGRAD.resize(this->data.dimension(1), nnet.gradient_size());

	math::matrix<T> FGRAD;
	FGRAD.resize(this->data.dimension(1)+RDIM, nnet.gradient_size());

	math::matrix<T> FRGRAD;
	FRGRAD.resize(RDIM, nnet.output_size());

	math::matrix<T> FGRADTMP;
	FGRADTMP.resize(this->data.dimension(1)+RDIM, RDIM);

	//for(unsigned int episode=0;episode<this->data.size(2);episode++)
	{
	  
	  //math::vertex<T> range = this->data.access(2,episode);

	  unsigned int start = 0; 
	  unsigned int length = this->data.size(0);

	  //whiteice::math::convert(start, range[0]);
	  //whiteice::math::convert(length, range[1]);
	  
	  UGRAD.zero();
	  grad.zero();
	  input.zero();

#pragma omp for nowait schedule(auto)
	  for(unsigned int i=start;i<length;i++){
	    input.write_subvertex(this->data.access(0,i), 0);
	      
	    assert(nnet.jacobian(input, FGRAD) == true);
	    // df/dw (this->data.dimension(1)+RDIM, nnet.gradient_size())

	    {
	      assert(nnet.gradient_value(input, FGRADTMP) == true);
	      // df/dinput (this->data.dimension(1)+RDIM,this->data.dimension(0)+RDIM)

	      // df/dr
	      assert(FGRADTMP.submatrix(FRGRAD,
					this->data.dimension(0), 0,
					RDIM, nnet.output_size()) == true);

	      // KAPPA_r = I

	      // df/dr (this->data.dimension(1)+RDIM, RDIM)
	      // dU/dw (this->data.dimension(1)+RDIM, nnet.gradient_size())

	      // KAPPA_r operation to UGRAD to select only R terms
	      assert(UGRAD.submatrix(URGRAD,
				     0,this->data.dimension(1),
				     nnet.gradient_size(), RDIM) == true);
	    }

	    // dU(n+1)/dw = df/dw + df/dr * KAPPA_r * dU(n)/dw
	    UGRAD = FGRAD + FRGRAD*URGRAD;

	    //nnet.input() = input;
	    //nnet.calculate(false);
	    nnet.calculate(input, output);
	    
	    { // calculate error gradient value for E(i=0)..E(N) terms
	      
	      output.subvertex(err, 0, this->data.dimension(1));
	      err -= this->data.access(1,i);

	      // selects only Y terms from UGRAD
	      assert(UGRAD.submatrix
		     (UYGRAD,
		      0,0,
		      nnet.gradient_size(), this->data.dimension(1)));

	      grad = err*UYGRAD;

	      sgrad += grad;
	    }

	    assert(output.subvertex(output_r, this->data.dimension(1), RDIM));
	    assert(input.write_subvertex(output_r, INPUT_DATA_DIM));
	  }

	}
	
#pragma omp critical
	{
	  sumgrad += sgrad;
	}
	
      }
      
      sumgrad /= this->sigma2;
      sumgrad /= this->temperature; // scales gradient with temperature
      
#if 1
      if(useRegularizer){
	// regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
	
	sumgrad += this->alpha*q;
      }
#endif
      
      return sumgrad;
    }

    
    
  }
  
  // calculates mean error for the latest N samples, 0 = all samples
  template <typename T>
  T rUHMC<T>::getMeanError(unsigned int latestN) const
  {
    std::lock_guard<std::mutex> lock(this->solution_lock);
    
    if(latestN == 0) latestN = this->samples.size();
    if(latestN > this->samples.size()) latestN = this->samples.size();

    T error = T(0.0);

#pragma omp parallel
    {
      T ei = T(0.0);

#pragma omp for nowait schedule(auto)
      for(unsigned int i = this->samples.size()-latestN;i<this->samples.size();i++){
	ei += this->U(this->samples[i], false);
      }

#pragma omp critical
      {
	error += ei;
      }
    }

    error = error/T(latestN);

    return error;
  }
  
  
};


namespace whiteice
{
  template class rUHMC< math::blas_real<float> >;
  template class rUHMC< math::blas_real<double> >;
};
