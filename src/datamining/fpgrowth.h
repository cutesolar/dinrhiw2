/*
 * Own implementation of FP-Growth algorithm
 *
 * Datamines frequent patterns from data.
 *
 */


#ifndef __whiteice__fpgrowth_h
#define __whiteice__fpgrowth_h

#include <set>
#include <vector>


namespace whiteice
{

  /*
   * datamines frequent itemsets using FP-Growth algorithm
   */ 
  bool frequent_items(const std::vector< std::set<long long> >& data,
		      std::set< std::set<long long> >& freq_sets,
		      double min_support = 0.0);
  
};


#endif
