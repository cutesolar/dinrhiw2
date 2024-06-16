
#ifdef WINOS
#include <windows.h>
#endif

#include "maximizer.h"
#include "optimized_function.h"
#include "eig.h"
#include "correlation.h"
#include "RNG.h"

#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>

#include <ctime>

namespace whiteice
{
  namespace math
  {
    
    // symmetrizes near symmetric matrix
    template <typename T>
    void symmetrize(matrix<T>& M) 
    {
      const unsigned int DY = M.ysize();
      
      for(unsigned int j=0;j<DY;j++)
	for(unsigned int i=0;i<j;i++){
	  M(i,j) += M(j,i);
	  M(i,j) /= T(2.0f);
	  M(j,i) = M(i,j);
	}
    }
    
    
    
    
    template <typename T>
    StochasticOptimizer<T>::StochasticOptimizer()
    {
      f = 0;
#ifndef WINOS
      CLOCKS_SEC = sysconf(_SC_CLK_TCK);
#else
      CLOCKS_SEC = CLOCKS_PER_SEC;
#endif
    }
    
    
    template <typename T>
    StochasticOptimizer<T>::~StochasticOptimizer()
    {
    }
    
    
    template <typename T>
    T StochasticOptimizer<T>::getSolution(whiteice::math::vertex<T>& v) const
    {
      v = solution;
      return solutionValue;
    }
    
    
    template <typename T>
    T StochasticOptimizer<T>::calculate(vertex<T>& v) const 
    {
      return f->calculate(P*v + mean);
    }
    
    
    // does further optimization by using n secs of CPU time
    template <typename T>
    unsigned int StochasticOptimizer<T>::optimize(whiteice::optimized_function<T>* f,
					       float secs) 
    {
      if(f == 0) return 0;
      this->f   = f;
      
      // initialization
      
      dimensions = f->dimension();
      
      
      solution.resize(f->dimension());
      candidate.resize(dimensions);
      
      for(unsigned int i=0;i<dimensions;i++)
	solution[i] = T(2.0f*(whiteice::rng.uniformf()) - 1.0f);
      
      solutionValue = (f->calculate(solution));
      
      iteration = 0;
      framingInterval = 1000;
      nextFraming = iteration + framingInterval;
      
      // from search space to function space transform parameters
      // Px + mean = y, value = f(y)
      P.resize(dimensions, dimensions);
      R.resize(dimensions, dimensions);
      curR.resize(dimensions, dimensions);
      P.identity();
      R.identity();
      curR.zero();
      
      
      mean.resize(dimensions);
      curMean.resize(dimensions);
      mean.zero();
      curMean.zero();
      
      tr_solution = solution;
      
      
      for(unsigned int i=0;i<dimensions;i++)
	toFullSpace[i] = i;
      
      
      initialize();
      
      return optimizeMore(secs);
    }
    
    
    template <typename T>
    unsigned int StochasticOptimizer<T>::optimizeMore(float secs)
    {
      if(secs <= 0.0f) return 0;
      
      startTime = getTime();
      endTime   = startTime + secs;
      
      std::cout << dimensions << " DIMENSIONS" << std::endl;
      
      std::cout << "iteration: " << iteration << std::endl;
      std::cout << "framing: " << nextFraming << std::endl;
      
      do{
	// generates new candidate and updates *candValue*
	// so (candValue, candidate) pair is up to date
	generate();
	
	if(candValue > solutionValue){	  
	  solution = P*candidate + mean;
	  tr_solution = candidate;
	  solutionValue = candValue;
	}
	
	// saves locations of search process
	iteration++;
	

	curMean += candidate;
	curR += candidate.outerproduct(candidate);
	
	
	if(iteration == nextFraming && f->dimension() > 70){
	  // reoptimize active dimensions
	  // used in the search
	  
	  std::cout << "iteration: " << iteration << std::endl;
	  std::cout << "framing: " << nextFraming << std::endl;
	  std::cout << "solution:" << solutionValue << std::endl;
	  
	  // PCA
	  matrix<T> X, E;	  
	  
	  curR = curR / T(framingInterval);
	  curMean = curMean / T(framingInterval);
	  
	  // removes mean from autocorrelation
	  curR -= curMean.outerproduct(curMean);
	  
	  X.resize(dimensions, dimensions);
	  E.resize(dimensions, dimensions);
	  E = curR;
	  
	  symmetric_eig(E, X); // diag(E) = variances	  	  
	  
	  
	  // finds the 10 highest variance dimensions
	  std::vector<unsigned int> dims;
	  unsigned int smallest_index;
	  T smallest_var = T(0.0f);
	  dims.resize(10);
	  
	  dims[0] = 0;
	  smallest_var = E(0,0);
	  smallest_index = 0;
	  
	  for(unsigned int i=1;i<dims.size();i++){
	    dims[i] = i;
	    if(E(i,i) < smallest_var){
	      smallest_var = E(i,i);
	      smallest_index = i;
	    }
	  }
	  
	  for(unsigned int i=dims.size();i<dimensions;i++){
	    if(E(i,i) > smallest_var){
	      dims[smallest_index] = i;
	      smallest_var = E(i,i);
	      smallest_index = i;
	      
	      // updates smallest_* values
	      for(unsigned int j=0;j<dims.size();j++){
		if(E(dims[j],dims[j]) < smallest_var){
		  smallest_var = E(dims[j], dims[j]);
		  smallest_index = j;
		}
	      }
	    }
	  }
	  
	  // selected dimensions are in PCA-reduced space
	  // transforms dimensions to PCAed space
	  
	  std::vector<unsigned int> fulldims;
	  
	  for(unsigned int i=0;i<dims.size();i++){
	    fulldims.push_back(toFullSpace[dims[i]]);
	  }
	  
	  // selects extra random dimensions so that
	  // algorithm gets 50 dimensions
	  
	  while(fulldims.size() < 50){
	    unsigned index = whiteice::rng.rand() % (f->dimension());
	    
	    for(unsigned int i=0;i<fulldims.size();i++){
	      if(fulldims[i] == index) continue; // try again
	    }
	    
	    fulldims.push_back(index);
	  }
	  
	  

	  // updates full autocorrelation matrix with
	  // calculated small autocorrelation matrix
	  // (update rule: weighted mean)
	  
	  // transforms curR and curMean to be same size as full R
	  
	  curR = P * curR;
	  curMean = P * curMean + mean;
	  
	  P.transpose(); // destroys P matrix
	  curR = curR * P;
	  
	  T w1 = T(iteration-framingInterval)/T(iteration);
	  T w2 = T(framingInterval)/T(iteration);
	  
	  // weighted mean
	  for(unsigned int j=0;j<R.ysize();j++)
	    for(unsigned int i=0;i<R.xsize();i++)
	      R(j,i) = w1*R(j,i) + w2*curR(j,i);
	  
	  for(unsigned int j=0;j<mean.size();j++)
	    mean[j] = w1*mean[j] + w2*curMean[j];
	  
	  
	  
	  
	  // calculates new PCA and P from R
	  dimensions = fulldims.size();
	  
	  P = R;
	  
	  // because of computational errors P may not be
	  // symmetric enough and symmetric_eig() doesn't
	  // work. symmetrize() makes P symmetric.
	  // (better solution: write unsymmetric eigenvalue solver)
	  symmetrize(P);
	  
	  symmetric_eig(P, X); // diag(P) = variances
	  
	  P.resize(f->dimension(), fulldims.size());
	  
	  for(unsigned int j=0;j<fulldims.size();j++)
	    for(unsigned int i=0;i<f->dimension();i++)
	      P(i,j) = X(fulldims[j],i);
	  
	  for(unsigned int i=0;i<fulldims.size();i++)
	    toFullSpace[i] = fulldims[i];	  	  
	  
	  X = P; // X inverse of P
	  X.transpose();
	  
	  tr_solution.resize(dimensions);
	  tr_solution = X*(solution - mean);
	  
	  curR.resize(dimensions,dimensions);
	  curMean.resize(dimensions);
	  curR.zero();
	  curMean.zero();
	  
	  candidate.resize(dimensions);
	  
	  
	  // reinitializes search algorithm
	  initialize();
	  
	  // till we see again..
	  nextFraming = iteration + framingInterval;
	  
	}

	
      }
      while(getTime() < endTime);
      
      
      std::cout << iteration << " iterations evaluated." << std::endl;
      
      return iteration;
    }
    

    template <typename T>
    void StochasticOptimizer<T>::initialize() { assert(0); }
    
    template <typename T>
    void StochasticOptimizer<T>::generate() { assert(0); }
    
    template <typename T>
    const vertex<T>& StochasticOptimizer<T>::getInternalSolution() const 
    {
      return tr_solution;
    }
    
    
    template <typename T>
    float StochasticOptimizer<T>::getTime() const {
#ifndef WINOS
      struct tms t1;
      
      if(times(&t1) == -1)
	return -1.0f;
      
      return ( ((float)t1.tms_utime)/CLOCKS_SEC );
#else
      HANDLE handle;
      FILETIME creation_time, exit_time, kernel_time, user_time;

      handle = GetCurrentProcess();

      if(GetProcessTimes(handle, &creation_time, &exit_time, &kernel_time, &user_time)){
    	  uint64_t t, tlow, thigh;
    	  tlow  = user_time.dwLowDateTime;
    	  thigh = user_time.dwHighDateTime;

    	  t = tlow + (thigh<<32);

    	  return (float)( ((double)t)/ 10000000.0 ); // 100 nanosecond ticks in windows
      }
      else{
    	  return -1.0f;
      }
#endif
    }
    
    
    //////////////////////////////////////////////////
    
    
    template <typename T>
    IHRSearch<T>::IHRSearch(){
      meanSQDif = T(0.0001f);
    }
    
    template <typename T>
    IHRSearch<T>::~IHRSearch(){ }
    
    
    
    // initialization
    template <typename T>
    void IHRSearch<T>::initialize() 
    {
      dir.resize(this->dimensions);
    }
    
    
    template <typename T>
    void IHRSearch<T>::generate() 
    {
      T scaling, smin, smax;
      
      while(1){
	for(unsigned int i=0;i<this->dimensions;i++)
	  dir[i] = T(2.0f*(whiteice::rng.uniformf()) - 1.0f);
	
	dir.normalize();
	
	smax = T(1000000000000000000000000.0f);
	smin = -smax;
	
	for(unsigned int i=0;i<this->dimensions;i++){
	  if(dir[i] > T(0.0f)){
	    if(smax > (T(2.0f) - this->getInternalSolution()[i])/dir[i])
	      smax =  (T(2.0f) - this->getInternalSolution()[i])/dir[i];
	    if(smin < (T(-2.0f) - this->getInternalSolution()[i])/dir[i])
	      smin = (T(-2.0f) - this->getInternalSolution()[i])/dir[i];
	  }
	  else if(dir[i] < T(0.0f)){
	    if(smax > (T(-2.0f) - this->getInternalSolution()[i])/dir[i])
	      smax = (T(-2.0f) - this->getInternalSolution()[i])/dir[i];
	    if(smin < (T(2.0f) - this->getInternalSolution()[i])/dir[i])
	      smin = (T(2.0f) - this->getInternalSolution()[i])/dir[i];
	  }
	}
	
	// randomly selects scaling within a range [smin, smax]
	
	scaling = (smax-smin)*T(whiteice::rng.uniformf()) + smin;
	this->candidate = this->getInternalSolution() + scaling*dir;
      
	this->candValue = this->calculate(this->candidate);
	
	// acceptance
	if(this->solutionValue < this->candValue)
	  return;
	else{
	  meanSQDif = 0.8f*meanSQDif + 0.2f*(this->solutionValue - this->candValue)*(this->solutionValue - this->candValue);
	  
	  T p = exp(-2.0f*((this->solutionValue - this->candValue)*(this->solutionValue - this->candValue)/meanSQDif));
	  
	  if(whiteice::rng.uniformf() <= p)
	    return;
	}
      }
    }
    
    
    //////////////////////////////////////////////////
    
    
    template <typename T>
    GradientDescent<T>::GradientDescent()
    {
    }
    
    
    template <typename T>
    GradientDescent<T>::~GradientDescent()
    {
    }
    
    
    template <typename T>
    void GradientDescent<T>::initialize() 
    {
      step = T(0.01f);
      
      prevcandidate.resize(this->dimensions);
      prevcandidate = this->getInternalSolution();
      dir.resize(this->dimensions);
      goodDir.resize(this->dimensions);
    }
    
    
    template <typename T>
    void GradientDescent<T>::generate() 
    {
      // calculates smoothed aprox. gradient            
      T bestValue = T(-1000000000000000.0f);
      
      for(unsigned int p=0;p<25;p++){
	for(unsigned int i=0;i<this->dimensions;i++)
	  dir[i] = T(2.0f*(whiteice::rng.uniformf()) - 1.0f);
	
	dir.normalize();
	
	this->candidate = prevcandidate + step*dir;
	this->candValue = this->calculate(this->candidate);
	
	if(this->candValue > bestValue){
	  bestValue = this->candValue;
	  goodDir = dir;
	}
      }
      
      
      prevcandidate += T(10.0f)*step*goodDir;
      this->candidate = prevcandidate;
      
      this->candValue = this->calculate(this->candidate);
    }
    
    
    //////////////////////////////////////////////////////////////////////
    
    template class StochasticOptimizer< blas_real<float> >;
    template class IHRSearch< blas_real<float> >;
    template class GradientDescent< blas_real<float> >;
    
    
  };
};

