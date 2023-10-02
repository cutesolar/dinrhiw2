/*
 * Finds frequent item sets from data source data.
 * 
 */

#ifndef FrequentSetsFinder_h
#define FrequentSetsFinder_h

#include <vector>
#include "dynamic_bitset.h"
#include "data_source.h"


namespace whiteice
{
  namespace datamining
  {
    
    class FrequentSetsFinder
    {
    public:
      FrequentSetsFinder(const whiteice::data_source<whiteice::dynamic_bitset>& data,
			 std::vector< whiteice::dynamic_bitset >& freqset,
			 float freq_limit);
      
      ~FrequentSetsFinder();
      
      // finds (more) associative rules that can be find in TIME_LIMIT seconds
      // if TIME_LIMIT <= 0 amount of time used is unlimited.
      // returns true if any new rule could be found
      bool find(float TIME_LIMIT = -1.0f);
      
      // returns true if last call of find() finished before time limit
      // and managed to find all possible rules
      bool finished() const ;
      
      // frees all data structures not needed
      // after all rules have been found
      // (clean everything except rules)
      void clean();
      
    private:
      
      unsigned int calculate_counts(const dynamic_bitset& set) const ;
      bool internal_isfinished() const ;
      
      // input data
      
      const whiteice::data_source<whiteice::dynamic_bitset>& data;
      std::vector<whiteice::dynamic_bitset>& freqset;
      
      const float freq_limit;
      
      
    private:
      // data structures used by search
      
      // frequent sets
      std::vector< std::vector<dynamic_bitset> > fset;
      
      // frequencies of frequent relations in fset
      std::vector< std::vector<unsigned int> > fcset; 
      
      
      // where the search is going: fset[level][elem]
      unsigned int level;
      unsigned int elem;
      
      bool search_finished;
    };
    
    
    
  }
}

#endif
