
#ifndef GeneticAlgorithm_cpp
#define GeneticAlgorithm_cpp

#include <iostream>
#include <exception>
#include <stdexcept>
#include <stdlib.h>
#include <math.h>

#include "GeneticAlgorithm.h"
#include "RNG.h"

namespace whiteice
{

  template <typename T>
  GeneticAlgorithm<T>::GeneticAlgorithm(const function<T,double>& f) :
    bits(sizeof(T)*8)
  {
    this->f = f.clone();
    maximization_task = true;
    
    p_crossover = 0.20;
    
    p_mutation  = 0.5/sqrt(((float)sizeof(T)*8.0f));
    
    verbose = false; // silence
    
    p[0] = 0;
    p[1] = 0;
    mean_value = 0;
  }

  
  template <typename T>
  GeneticAlgorithm<T>::~GeneticAlgorithm() 
  {
    delete f;
  }
  
  
  template <typename T>
  bool GeneticAlgorithm<T>::maximize(const unsigned int numIterations,
				     const unsigned int size) 
  {
    try{
      if(size == 0) return false;
      
      p[0] = &(q[0]);
      p[1] = &(q[1]);
      
      p[0]->resize(size);
      p[1]->resize(size);
      goodness.resize(size);
      maximization_task = true;
      mean_value = 0;    
      
      if(!create_initial_population()) return false;
      create_candidate((*p[0])[0], best_candidate);
      very_best_value = -10000000000000.0; // to infinity
      
      return continue_optimization(numIterations);
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  bool GeneticAlgorithm<T>::minimize(const unsigned int numIterations,
				     const unsigned int size) 
  {
    try{
      if(size == 0) return false;
      
      p[0] = &(q[0]);
      p[1] = &(q[1]);
      
      p[0]->resize(size);
      p[1]->resize(size);
      goodness.resize(size);
      maximization_task = false; // -> minimizes
      mean_value = 0;
      
      if(!create_initial_population()) return false;
      create_candidate((*p[0])[0], best_candidate);
      very_best_value = +10000000000000.0; // to infinity
      
      return continue_optimization(numIterations);  
    }
    catch(std::exception& e){ return false; }
  }
  
  
  /*
   * currently: slow but perfect
   */
  template <typename T>
  bool GeneticAlgorithm<T>::continue_optimization(const unsigned int numIterations) 
  {
    std::vector< std::bitset< sizeof(T)*8> >* tmp_ptr;
    
    unsigned int iter = 0;
    double total = 0;
    
    const unsigned int size = (unsigned int)(p[0]->size());
    T candidate;
    
    if(verbose){
      if(maximization_task)
	std::cout << "GA MAXIMIZATION.";
      else
	std::cout << "GA MINIMIZATION.";
      
      std::cout << "POP SIZE " << size << "  ";
      std::cout << numIterations << " ITERS" << std::endl;
    }
    
    while(iter < numIterations){
      /////////////////////////////////////////////////
      // evaluate
      
      total = 0;
      
      double best_value;
      
      if(maximization_task) // change to infinities
	best_value = -1000000.0f;
      else
	best_value = +1000000.0f;
      
      
      for(unsigned int j=0;j<size;j++){
	
	create_candidate( (*p[0])[j], candidate );
	
	goodness[j] = (*f)(candidate);
	
	// checks if this element is the biggest/smallest one
	// (in this iteration, in all iterations)
	
	if(maximization_task){
	  
	  if(goodness[j] > best_value){	  
	    best_value = goodness[j];
	    
	    if(best_value > very_best_value){
	      best_candidate = candidate;
	      very_best_value = best_value;
	    }
	  }
	}
	else{	
	  if(goodness[j] < best_value){
	    best_value = goodness[j];
	    
	    if(best_value < very_best_value){
	      best_candidate = candidate;
	      very_best_value = best_value;
	    }
	  }
	  
	  // reverses goodness for probabilistic
	  // next generation production (where larger is better)
	  goodness[j] = 1 / goodness[j];
	} 
	
	total += goodness[j];
      }
      
      if(maximization_task)
	mean_value = total/(double)size;
      else
	mean_value = 1.0f / (total / (double)size);
      
      for(unsigned int j=0;j<size;j++) // normalizes
	goodness[j] /= total;
      
      // selects genes for new population
      for(unsigned int i=0;i<size;i++){
	
	double r = whiteice::rng.uniformf();
	double sum = 0;
	unsigned int l = 0;
	
	do{
	  sum += goodness[l];
	  l++;
	}
	while(sum < r && l < size);
	l = l - 1;
	
	(*p[1])[i] = (*p[0])[l];
      }
      
      // crossovers
      for(unsigned int i=0;i<size;i++){
	double r = whiteice::rng.uniformf();
	
	if(r < p_crossover){
	  // partner
	  unsigned int j = whiteice::rng.rand() % size;
	  
	  // shifting invariant:
	  // crossover starting position and length
	  unsigned int index = whiteice::rng.rand() % (sizeof(T)*8);
	  unsigned int clen  = whiteice::rng.rand() % (sizeof(T)*8);
	  
	  for(unsigned int k=0;k<clen;k++){
	    int K = (k + index) % (sizeof(T) * 8);
	    
	    int tmp = (*p[1])[i][K];  // swaps bits
	    (*p[1])[i][K] = (*p[1])[j][K];
	    (*p[1])[j][K] = tmp;
	  }
	}
      }
      
      // mutates
      for(unsigned int i=0;i<size;i++){
	double r = whiteice::rng.uniformf();
	
	if(r < p_mutation){
	  unsigned int index = whiteice::rng.rand() % (sizeof(T)*8);
	  
	  (*p[1])[i].flip(index); // mutation
	}
      }
      
      
      if(verbose){
	if(iter % 50 == 0){
	  std::cout << "ITER " << iter << " / "
		    << numIterations;
	  std::cout << " BestValue: " << very_best_value
		    << " BogoMeanValue: " << mean_value
		    << std::endl;
	}
    }
      
      
      // swaps populations
      tmp_ptr = p[0];
      p[0] = p[1];
      p[1] = tmp_ptr;
      iter++;
    }
    
    return true;
  }
  
  
  /*
   * creates equally distributed random variables
   * assumes rand() has been initialized with srand().
   */
  template <typename T>
  bool GeneticAlgorithm<T>::create_initial_population() 
  {
    try{
      typename std::vector< std::bitset< sizeof(T)*8> >::iterator i;
      
      i = q[0].begin();
      
      while(i != q[0].end()){
	
	for(unsigned int j=0;j<sizeof(T)*8;j++){
	  
	  if(whiteice::rng.rand() & 1)
	    (*i).set(j, 0); // bit is zero
	  else
	    (*i).set(j, 1); // bit is one
	}
	
	i++;
      }
      
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  bool GeneticAlgorithm<T>::create_candidate(const std::bitset< sizeof(T)*8 >& bits,
					     T& candidate) const 
  {
    unsigned char* ptr = (unsigned char*)(&candidate);
    unsigned char value;
    
    for(unsigned int k = 0, m = 0;k<sizeof(T);k++,m+=8){
      value = 0;
      
      for(unsigned int r=0;r<8;r++)
	value += bits[m+r] << r;
    
      ptr[k] = value;
    }
    
    return true;
  }
  
  
  // returns value of the best candidate
  // saves it to best
  template <typename T>
  double GeneticAlgorithm<T>::getBest(T& best) const 
  {
    best = best_candidate;
    return very_best_value;
  }
  
  // returns mean value
  template <typename T>
  double GeneticAlgorithm<T>::getMean() const 
  {
    return mean_value;
  }
  
};
  
#endif
  

