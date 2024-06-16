

#include <vector>
#include "PSO.h"
#include "negative_function.h"
#include "stretched_function.h"
#include "RNG.h"

namespace whiteice
{
	template <typename T>
	PSO<T>::PSO(const optimized_function<T>& f)
	{
		this->f = dynamic_cast<optimized_function<T>*>(f.clone());
		maximization_task = true;
		first_time = true;
		verbose = false;

		c1 = T(0.8); // cognitive parameter
		c2 = T(0.5); // social parameter

		// default range parameters for data
		datarange.resize(f.dimension());

		for(unsigned int i=0;i<datarange.size();i++){
			datarange[i].min = T(0.0);
			datarange[i].max = T(1.0);
		}


		global_best.best.resize(f.dimension());
		global_best.velocity.resize(f.dimension());
		global_best.value.resize(f.dimension());

		total_best.best.resize(f.dimension());
		total_best.velocity.resize(f.dimension());
		total_best.value.resize(f.dimension());

		old_best.best.resize(f.dimension());
	}
  
  
	// r = range of input values
	template <typename T>
	PSO<T>::PSO(const optimized_function<T>& f,
			std::vector<PSO<T>::range> r)
	{
		this->f = dynamic_cast<optimized_function<T>*>(f.clone());
		maximization_task = true;
		first_time = true;
		verbose = false;

		c1 = T(0.8); // cognitive parameter
		c2 = T(0.5); // social parameter
    
		if(r.size() != f.dimension())
			throw std::invalid_argument("range data has incorrect dimensionality");

		// default range parameters for data
		datarange.resize(f.dimension());

		for(unsigned int i=0;i<datarange.size();i++){
			datarange[i].min = r[i].min;
			datarange[i].max = r[i].max;
		}

		global_best.best.resize(f.dimension());
		global_best.velocity.resize(f.dimension());
		global_best.value.resize(f.dimension());

		total_best.best.resize(f.dimension());
		total_best.velocity.resize(f.dimension());
		total_best.value.resize(f.dimension());

		old_best.best.resize(f.dimension());
	}
  
  
	template <typename T>
	PSO<T>::~PSO() 
	{
		if(f) delete f;
	}
  
  
	template <typename T>
	bool PSO<T>::maximize(const unsigned int numIterations,
				const unsigned int size) 
	{
		// transforms task into minimization task
		optimized_function<T>* old_f = this->f;
		this->f = new negative_function<T>(*(this->f));
		first_time = true;

		bool ok = create_initial_population(size);
		maximization_task = true;
		global_iter = 0;

		if(!ok){
			delete f;
			this->f = old_f;
			return false;
		}
        

		ok = continue_optimization(numIterations);

		// back to original problem
		delete f;
		this->f = old_f;

		return ok;
	}


	template <typename T>
	bool PSO<T>::minimize(const unsigned int numIterations,
			const unsigned int size) 
	{
		global_iter = 0;
		maximization_task = false;
		first_time = true;

		if(!create_initial_population(size))
			return false;

		return continue_optimization(numIterations);
	}
  
  
	// with custom starting population
	template <typename T>
	bool PSO<T>::maximize(const unsigned int numIterations,
				const std::vector< math::vertex<T> >& data) 
	{
		// transforms task into minimization task
		optimized_function<T>* old_f = this->f;
		this->f = new negative_function<T>(*(this->f));
		global_iter = 0;
		first_time = true;

		bool ok = setup_population(data);
		maximization_task = true; // task is now minimization

		if(!ok){
			delete f;
			this->f = old_f;
			return false;
		}

		ok = continue_optimization(numIterations);

		// back to original problem
		delete f;
		this->f = old_f;

		return ok;
	}
  

	// with custom starting population
	template <typename T>
	bool PSO<T>::minimize(const unsigned int numIterations,
			const std::vector< math::vertex<T> >& data) 
	{
		global_iter = 0;
		maximization_task = false;
		first_time = true;

		if(!setup_population(data))
			return false;

		return continue_optimization(numIterations);
	}
	
	
	template <typename T>
	bool PSO<T>::improve(const unsigned int numIterations) {
		return continue_optimization(numIterations);
	}


	template <typename T>
	bool PSO<T>::continue_optimization(const unsigned int numIterations) 
	{

		// invalidates possible cumulative distribution
		cumdistrib.clear();

		typename PSO<T>::particle gbest; // generation best particle

		// convergence/local minima detection

		if(maximization_task){
			if(verbose) std::cout << "PSO MAXIMIZATION\n";
		}
		else{
			if(verbose) std::cout << "PSO MINIMIZATION\n";
		}


		/* actual task is always presented as a minimization task */
		global_iter++;

		for(unsigned int iter=0;iter<numIterations;iter++,global_iter++){
			gbest = swarm[0]; // 'almost' correct initialization

			// calculates current fitness values
#pragma omp parallel for
			for(unsigned int j=0;j<swarm.size();j++){
				auto& i = swarm[j];

				i.fitness = f->calculate(i.value);

#pragma omp critical (global_best_update)
				{
					if(i.fitness < i.best_fitness){
						i.best_fitness = i.fitness;
						i.best = i.value;
					}

					if(i.fitness < gbest.best_fitness){
						gbest.best_fitness = i.fitness;
						gbest.best = i.value;
					}
				}
			}


			if(verbose){
				if(global_iter % 10 == 0){
					if(maximization_task){
						std::cout << "ITER " << global_iter << "/" << numIterations
								<< " GBEST " << -gbest.best_fitness
								<< std::endl;
					}
					else{
						std::cout << "ITER " << global_iter << "/" << numIterations
								<< " GBEST " << gbest.best_fitness
								<< std::endl;
					}
				}
			}


			// updates velocity and location of particles
			for(auto i = swarm.begin();i!=swarm.end();i++){

				// updates velocity and location

			  T r1 = T(whiteice::rng.uniformf());
			  T r2 = T(whiteice::rng.uniformf());

				i->velocity += (c1*r1) * ( (i->best) - (i->value) );
				i->velocity += (c2*r2) * ( (gbest.best) - (i->value) );
				i->value += i->velocity;

				clamp_values_to_range(*i);
			}
      

			// updates global best
			if(gbest.best_fitness < global_best.best_fitness){
				global_best = gbest;
				if(gbest.best_fitness < total_best.best_fitness){
					total_best = gbest;
				}
			}


			// convergence detection
			if(global_iter % 17 == 0){
				// less than 1%
				if(percentage_change(old_best.best, global_best.best) < 0.001){
					std::cout << "CONVERGENCE DETECTED" << std::endl;
					std::cout << "RESTARTING" << std::endl;

					if(create_initial_population(swarm.size()) == false){
						return false;
					}
#if 0
	  
					if(verbose){
						if(maximization_task)
							std::cout << "MAXIMUM DETECTED. FUNCTION STRETCHING ACTIVATED\n";
						else
							std::cout << "MINIMUM DETECTED. FUNCTION STRETCHING ACTIVATED\n";
					}
	  

					// convergence -> stretches function
					f = new stretched_function<T>(f, global_best.best);
#endif
				}

				for(unsigned int i=0;i<old_best.best.size();i++)
					old_best.best[i] = global_best.best[i];
			}

		}


		return true;
	}
  
  
  // returns best candidate found so far
  template <typename T>
  void PSO<T>::getCurrentBest(math::vertex<T>& best) const 
  {
    best.resize(f->dimension());
    best = global_best.best;
  }
  
  
  // returns best candidate found so far
  template <typename T>
  void PSO<T>::getBest(math::vertex<T>& best) const 
  {
    best.resize(f->dimension());
    best = total_best.best;
  }
  
  
  template <typename T>
  const typename PSO<T>::particle& PSO<T>::sample()
  {
    // finds smallest value of fitness +
    // calculates cumulative goodness
    // distribution
    
    if(swarm.size() <= 0)
      throw illegal_operation("trying to sample from empty swarm");
    
    
    if(cumdistrib.size() <= 0){ // -> recalculates distrib.
      typename std::vector<struct whiteice::PSO<T>::particle>::iterator i =
	swarm.begin();
      
      T minvalue  = i->fitness;
      T prevvalue = T(0.0f);
      cumdistrib.push_back(prevvalue);
      
      while(i != swarm.end()){
	prevvalue += i->fitness;
	cumdistrib.push_back(prevvalue);
	
	if(i->fitness < minvalue)
	  minvalue = i->fitness;	
	
	i++;
      }
      
      
      if(minvalue < T(0.0f)){ // fixes distribution
	T index = T(1.0f);
	typename std::vector<T>::iterator j = 
	  cumdistrib.begin();
	
	minvalue = -minvalue;
	
	// makes all value to be non-zero
	
	while(j != cumdistrib.end()){
	  *j = *j + index*minvalue;
	  index = index + T(1.0f);
	  j++;
	}
      }
      
      
      cumdistrib_maxvalue = T(0.0f);
      
      {
	typename std::vector<T>::iterator j = 
	  cumdistrib.begin();
	
	while(j != cumdistrib.end()){
	  if(*j > cumdistrib_maxvalue)
	    cumdistrib_maxvalue = *j;
	  
	  j++;
	}
      }
      
    }
    
    
    // now there's unscaled distribution based on goodness
    // that starts from zero
    
    // sampling:
    {
      T value = T(whiteice::rng.uniformf())*cumdistrib_maxvalue;
      
      unsigned int index = 0;
      
      while(cumdistrib[index] < value){
	index++;
	if(index >= cumdistrib.size())
	  break;
      }
      
      if(index > 0)
	index--;
      
      return swarm[index];
    }
    
  }
  
  
  // swarm size
  template <typename T>
  unsigned int PSO<T>::size() const {
    return swarm.size();
  }
  
  
  
  
    template <typename T>
    bool PSO<T>::create_initial_population(const unsigned int size)
	{
    	try{
    		// creates random swarm of particles
      
    		if(datarange.size() != f->dimension())
    			return false;

    		swarm.resize(size);
    		unsigned int i, j;

    		for(unsigned int j=0;j<swarm.size();j++){
    			swarm[j].value.resize(f->dimension());
    			swarm[j].velocity.resize(f->dimension());
    			swarm[j].best.resize(f->dimension());


    			// creates random location
    			for(i=0;i<f->dimension();i++){ // [0,1]
    				T temp = T(whiteice::rng.uniformf()) *
    						(datarange[i].max - datarange[i].min) + datarange[i].min;
    				swarm[j].value[i] = temp;
    			}

    			// create random velocity
    			for(i=0;i<f->dimension();i++){ // [-0.25, 0.25]*range
			  T temp = (T(whiteice::rng.uniformf())/T(2.0) - T(0.25)) *
			    (datarange[i].max - datarange[i].min);
			  swarm[j].velocity[i] = temp;
    			}
    		}


#pragma omp parallel for
    		for(unsigned int j=0;j<swarm.size();j++){
    			swarm[j].fitness = f->calculate(swarm[j].value);
    			swarm[j].best_fitness = swarm[j].fitness;
    			swarm[j].best = swarm[j].value;
    		}
      
      
    		// finds best (minimum) particle

    		unsigned int best_index = 0, index = 0;
    		T best_fitness = swarm[0].best_fitness;

    		for(j=0;j<swarm.size();j++,index++){
    			if(best_fitness > swarm[j].fitness){
    				best_fitness = swarm[j].fitness;
    				best_index = index;
    			}
    		}


    		// sets best partice and fitness values
    		global_best = swarm[best_index];

    		if(first_time){
    			total_best = global_best;
    			first_time = false;
    		}

    		return true;
    	}
    	catch(std::exception& e){
    		std::cout << "PSO-GBRBM: unexpected exception when creating population." << std::endl;
    		return false;
    	}
	}
  
  

  // creates initial particles from given data
  template <typename T>
  bool PSO<T>::setup_population(const std::vector< math::vertex<T> >& data) 
  {
    try{
      // creates random swarm of particles
      
      if(data.size() == 0)
	return false;
      
      swarm.resize(data.size());
      typename std::vector<math::vertex<T> >::const_iterator l;
      unsigned int i, j;
      
      for(j=0,l=data.begin();j<swarm.size();j++,l++){
	swarm[j].value.resize(f->dimension());
	swarm[j].velocity.resize(f->dimension());
	swarm[j].best.resize(f->dimension());
	
	if(l == data.end())
	  l = data.begin();
	
	
	swarm[j].value = *l; // copies data to particles
	
	// create random velocity
	for(i=0;i<f->dimension();i++) // [-0.25, 0.25]
	  swarm[j].velocity[i] = 
	    (T(whiteice::rng.uniformf())*T(0.5) - T(0.25)) * 
	    (datarange[i].max - datarange[i].min);

	
	// sets best value
	swarm[j].best_fitness = f->calculate(swarm[j].value);
	swarm[j].best = swarm[j].value;
      }
      
      // finds best (minimum) particle
      
      unsigned int best_index = 0, index = 0;    
      T best_fitness = f->calculate(swarm[0].value);
      
      for(j=0;j<swarm.size();j++,index++){
	swarm[j].fitness = f->calculate(swarm[j].value);
	if(best_fitness > swarm[j].fitness){
	  best_fitness = swarm[j].fitness;
	  best_index = index;
	}
      }
      
      // sets best partice and fitness values
      global_best = swarm[best_index];    
      
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  // calculates difference between |global_best - old_best|/|old_best|
  // in percentages/100
  template <typename T>
  T PSO<T>::percentage_change(const math::vertex<T>& old_best,
			      const math::vertex<T>& new_best) const
  {
    T oldlen = T(0.0); // squared
    T diflen = T(0.0); // squared
    
    for(unsigned int i=0;i<old_best.size();i++){
      oldlen += old_best[i] * old_best[i];
      diflen += (new_best[i] - old_best[i]) * (new_best[i] - old_best[i]);
    }
    
    
    if(oldlen == T(0.0)) return T(1.0);
    else return whiteice::math::sqrt(diflen/oldlen);
  }
  
  
  template <typename T>
  void PSO<T>::clamp_values_to_range(PSO<T>::particle& p) const 
  {
	  // clamps particles values to selected range
	  auto& r = this->datarange;

	  for(unsigned int i=0;i<p.value.size();i++){
		  if(p.value[i] < r[i].min) p.value[i] = r[i].min;
		  if(p.value[i] > r[i].max) p.value[i] = r[i].max;
	  }
  }

  //////////////////////////////////////////////////////////////////////
  
  template class PSO<float>;
  template class PSO<double>;
  template class PSO< math::blas_real<float> >;
  template class PSO< math::blas_real<double> >;
  
}






