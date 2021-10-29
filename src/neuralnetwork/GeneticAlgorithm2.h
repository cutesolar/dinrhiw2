/*
 * very basic
 * genetic algorithm for function optimization
 *
 * TODO: optimize, make more robust, faster, improve.
 *
 * idea: if / when lots of example values of x are
 * available, turn them to bits easily (direct binary string)
 * bits(x) and then use ica/pca to gaussianice/minimize information
 * between bits. x_ = ica(bits(x)). this should remove same information between
 * bits - bits cannot 'conflict' if b1 != b2. However making correlation matrix Rx = I
 * would be probably better/easier idea.
 *
 * If bits cannot conflict then feature which bits code is either on or off.
 * and not in-between. good/bad thing?
 *
 * for example changing representation of bits *during* optimization
 * determinized by:  ica(bits(x_good_ones)). -> coding is optimized to
 * represent information for good candidates very compactly ->
 * more bits left for coding extra features for good ones to improve.
 *
 * however: this makes mutation to be more disasterious (if it's not very good - common).
 *
 * ---
 * Some soft of histogram equalization could be also be a good method for keeping
 * distribution of genes 'flat' and without peaks. -> this kind of 'zooms' in to
 * peak so that the problem space is again flat: -> well used number of variables.
 * (uses whole problem space/dimensionality well for solving the problem).
 * ---
 *
 * above:
 * kind of feedback between 'preprocessing' stage (choosing of representation)
 * and optimization stage
 * or better (view point):  integrates choose of representation and actual optimization
 * into one task, actually kind of EM method when you think it:
 *    representation is constantly optimized for optimization task (needed parameters
 *       are constantly changing: optimizes for current instance (expected value) )
 *    and optimization optimizes normally.
 * 
 */

#ifndef GeneticAlgorithm2_h
#define GeneticAlgorithm2_h

#include <vector>
#include <bitset>
#include "function.h"
#include "dynamic_bitset.h"

namespace whiteice
{

  class GA
  {
  public:
    
    // function must be positive
    // f(x) > 0  with all x !
    GA(const unsigned int nbits,
       const function<dynamic_bitset, float>& f);
    ~GA() ;
    
    float& getCrossover() { return p_crossover; }
    float& getMutation()  { return p_mutation; }
    
    // optimizes with given
    // population size
    bool maximize(const unsigned int numIterations,
		  const unsigned int size) ;
    bool minimize(const unsigned int numIterations,
		  const unsigned int size) ;
    
      // continues optimization
    bool continue_optimization(const unsigned int numIterations) ;
      
    // returns value of the best candidate
    // saves it to best
    float getBest(dynamic_bitset& best) const ;
    
    // returns mean value
    float getMean() const ;
    
    // bad english...
    bool verbosity(bool v) { verbose = v; return verbose; }
    
  private:
    
    bool create_initial_population() ;
    
    const unsigned int bits;
    function<dynamic_bitset, float>* f;
    bool maximization_task;
      
    // probabilities
    float p_crossover;
    float p_mutation;
    
    // q[]   = two populations, current and the previous one
    // *p[0] = current population
    // *p[1] = previous population
    
    std::vector< dynamic_bitset >* p[2];
    std::vector< dynamic_bitset >  q[2];
    std::vector< float > goodness;
    
    bool verbose; // be talkative?
    
    // best candidate: (*p[1])[best_index]
    dynamic_bitset best_candidate;
    float very_best_value;
    
    float mean_value; // current state of the learning process
  };
}
  
//#include "GeneticAlgorithm2.cpp"

#endif




