
#ifndef _whiteice_discretize_h
#define _whiteice_discretize_h

#include <vector>
#include <string>
#include <set>

#include "dynamic_bitset.h"


namespace whiteice
{
  struct discretization
  {
    unsigned int TYPE;

    // TYPE == 0
    std::vector<float> bins; // BIN+1 bins between: bin[i-1] < x < bin[i]

    // TYPE == 1
    std::vector<std::string> elem; // i element is elem[i] string

    // TYPE == 2
    // ignore-value
    
  };


  // calculates discretization of data
  bool calculate_discretize(const std::vector< std::vector<std::string> >& data,
			    std::vector<struct discretization>& disc);

  // discretizes data and creates one-hot-encoding of discrete value in binary
  bool binarize(const std::vector< std::vector<std::string> >& data,
		const std::vector<struct discretization>& disc,
		std::vector< std::vector<double> >& result);


  // creates dataset with frequent sets added as extra-variables
  bool enrich_data(const std::vector< std::vector<double> >& binary_data,
		   std::set<whiteice::dynamic_bitset>& f, // frequent sets
		   std::vector< std::vector<double> >& result,
		   double freq_limit = 0.00); // 0 = automatic freq limit
  
  // creates dataset with frequent sets added as extra-variables
  bool enrich_data_again(const std::vector< std::vector<double> >& binary_data,
			 const std::set<whiteice::dynamic_bitset>& f, // frequent sets
			 std::vector< std::vector<double> >& result);


  
  
};

#endif
