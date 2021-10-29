
//#ifndef GeneticAlgorithm_cpp
//#define GeneticAlgorithm_cpp

#include <iostream>
#include <exception>
#include <stdexcept>
#include <stdlib.h>
#include <math.h>

#include "GeneticAlgorithm2.h"

namespace whiteice
{
  
  GA::GA(const unsigned int nbits,
	 const function<dynamic_bitset, float>& f) : bits(nbits)
  {
    this->f = f.clone();
    maximization_task = true;
    
    p_crossover = 0.80;
    
    // half of the maximum tolerable mutation rate
    // the value based on results presented in book.
    // MacKay. Information Theory, Inference and Learning Algorithms
    // this doesn't maximize rate of learning in evolution
    // (TODO: calculate and use exact value (wednesday))
    p_mutation  = 1.0/sqrt(((float)nbits));
    
    verbose = false; // silence
    
    p[0] = 0;
    p[1] = 0;
    mean_value = 0;
  }

  
  
  GA::~GA() { if(f) delete f; }
  
  
  bool GA::maximize(const unsigned int numIterations,
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
      best_candidate = (*p[0])[0];
      very_best_value = -10000000000000.0; // to infinity
      
      return continue_optimization(numIterations);
    }
    catch(std::exception& e){ return false; }
  }
  
  
  
  bool GA::minimize(const unsigned int numIterations,
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
      best_candidate = (*p[0])[0];
      very_best_value = +10000000000000.0; // to infinity
      
      return continue_optimization(numIterations);  
    }
    catch(std::exception& e){ return false; }
  }
  
  
  /*
   * currently: slow but perfect
   */
  bool GA::continue_optimization(const unsigned int numIterations) 
  {
    unsigned int iter = 0;
    float total = 0;
    
    const unsigned int size = p[0]->size();
    
    
    if(verbose){
      if(maximization_task)
	std::cout << "GA MAXIMIZATION.";
      else
	std::cout << "GA MINIMIZATION.";
      
      std::cout << " POP SIZE " << size << "  ";
      std::cout << numIterations << " ITERS" << std::endl;
    }
    
    
    while(iter < numIterations){
      // evaluate
      
      total = 0;
      float best_value;
      
      if(maximization_task) // change to infinities
	best_value = -1000000.0f;
      else
	best_value = +1000000.0f;
      
      
      for(unsigned int j=0;j<size;j++){
	goodness[j] = (*f)((*p)[0][j]);
	
	// checks if this element is the biggest/smallest one
	// (in this iteration, in all iterations)
	
	if(maximization_task){
	  if(goodness[j] > best_value){	  
	    best_value = goodness[j];
	    
	    if(best_value > very_best_value){
	      best_candidate = (*p)[0][j];
	      very_best_value = best_value;
	    }
	  }
	}
	else{	
	  if(goodness[j] < best_value){
	    best_value = goodness[j];
	    
	    if(best_value < very_best_value){
	      best_candidate = (*p)[0][j];
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
	mean_value = total/((float)size);
      else
	mean_value = 1.0f / (total / (float)size);
      
      for(unsigned int j=0;j<size;j++) // normalizes
	goodness[j] /= total;
      
      // selects genes for new population
      for(unsigned int i=0;i<size;i++){
	float sum = 0.0f;
        float r = rand()/((float)RAND_MAX);
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
        float r = rand()/((float)RAND_MAX);
	
	if(r < p_crossover){	  
	  // partner
	  unsigned int j = rand() % size;
	  
	  // shifting invariant:
	  // crossover starting position and length
	  unsigned int index = rand() % bits;
	  unsigned int clen  = rand() % bits;
	  
	  for(unsigned int k=0;k<clen;k++){
	    unsigned int K = (k + index) % bits;
	    
	    bool tmp = (*p[1])[i][K];  // swaps bits
	    (*p[1])[i].set((*p[1])[j][K], K);
	    (*p[1])[j].set(tmp, K);
	  }
	}
      }
      
      
      // mutates
      for(unsigned int i=0;i<size;i++){
	float r = rand()/((float)RAND_MAX);
	
	if(r < p_mutation){
	  unsigned int index = rand() % bits;
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
      std::swap(p[0], p[1]);
      iter++;
    }
    
    return true;
  }
  
  
  /*
   * creates equally distributed random variables
   * assumes rand() has been initialized with srand().
   */
  bool GA::create_initial_population() 
  {
    try{
      std::vector<dynamic_bitset>::iterator i;
      i = q[0].begin();
      
      unsigned int r = rand();
      unsigned int rc = 0;
      
      while(i != q[0].end()){
	i->resize(bits);
	
	for(unsigned int j=0;j<bits;j++){
	  i->set(j, r & 1);
	  r >>= 1;
	  
	  rc++;
	  if(rc == 32){
	    r = rand();
	    rc = 0;
	  }
	}
	
	i++;
      }
      
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  // returns value of the best candidate
  // saves it to best
  
  float GA::getBest(dynamic_bitset& best) const {
    best = best_candidate;
    return very_best_value;
  }
  
  // returns mean value
  float GA::getMean() const { return mean_value; }
  
};
  
//#endif
  

