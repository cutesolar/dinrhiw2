
#ifndef _whiteice_discretize_h
#define _whiteice_discretize_h

#include <vector>
#include <string>


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

  
  bool calculate_discretize(const std::vector< std::vector<std::string> >& data,
			    std::vector<struct discretization>& disc);

  // discretizes data and creates one-hot-encoding of discrete value in binary
  bool binarize(const std::vector< std::vector<std::string> >& data,
		const std::vector<struct discretization>& disc,
		std::vector< std::vector<double> >& result); 
  
};

#endif
