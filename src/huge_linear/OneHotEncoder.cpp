/*
 * 0/1 data binarizes of continuous data elements and discrete values
 */

#include "OneHotEncoder.h"
#include "vertex.h"

#include <stdio.h>
#include <vector>
#include <set>
#include <string>
#include <algorithm>

#include "fptree.hpp"



namespace whiteice
{

  bool calculateOneHotEncoding(const BinaryVectorsFile& input,
			       std::set<unsigned long>& ignoreVariables,
			       BinaryVectorsFile& output,
			       struct oneHotEncodingInfo& info)
  {
    if(input.hasFile() == false || output.hasFile() == false)
      return false;

    if(input.getVectorLength() == 0 || input.getNumberOfVectors() == 0)
      return false;

    info.discretized.clear();
    info.dmap.clear();
    info.numeric.clear();
    info.nbins.clear();

    // look for discrete variables with max 30 different values
    // + look for numerical values for discretization
    const unsigned long MAX_NDISCRETE = 30;

    info.NUM_ORIGINAL_VARIABLES = input.getVectorLength();
    info.NUM_DISCRETIZED_VARIABLES = 0;
    
    std::set< float > unique; // for a single column
    std::vector< float > values; // for a single column
    math::vertex< math::blas_real<float> > v, u;
    
    for(unsigned long c=0;c<input.getVectorLength();c++){

      if(ignoreVariables.find(c) != ignoreVariables.end()){
	continue; // don't process ignored variables
      }

      for(unsigned long i=0;i<input.getNumberOfVectors();i++){
	if(input.getVector(i, v) == false) return false;

	unique.insert(v[c].c[0]);
	values.push_back(v[c].c[0]);
      }

      if(unique.size() <= MAX_NDISCRETE){
	// calculates discretization for column c values
	info.discretized.insert(c);

	unsigned long index = 0;
	for(auto d : unique){
	  info.dmap[c][d] = index;
	  index++;
	}

	info.NUM_DISCRETIZED_VARIABLES += unique.size();
      }
      else{ // numerical value (rest of the fields)
	// sort vector values
	std::sort(values.begin(), values.end());

	unsigned long bins = values.size()/30;

	if(bins <= 1) bins = 2;
	else if(bins > 20) bins=20;

	const unsigned long NUMITEMSPERBIN = values.size()/(bins+1);

	info.numeric.insert(c);

	for(unsigned long b = 0;b<(bins-1);b++)
	  info.nbins[c].push_back(values[(b+1)*NUMITEMSPERBIN]);

	info.NUM_DISCRETIZED_VARIABLES += bins;
      }

      unique.clear();
      values.clear();
    }

    // now we have discretization information setup so we discretize out data

    output.clear();
    output.setVectorLength(info.NUM_DISCRETIZED_VARIABLES);

    for(unsigned long i=0;i<input.getNumberOfVectors();i++){
      if(input.getVector(i, v) == false) return false;

      if(oneHotEncoding(v, u, info) == false) return false;

      if(output.addVector(u) == false) return false;
    }
	
    return true;
  }

  
  bool oneHotEncoding(const math::vertex< math::blas_real<float> >& input,
		      math::vertex< math::blas_real<float> >& encoded,
		      const struct oneHotEncodingInfo& info)
  {

    encoded.resize(info.NUM_DISCRETIZED_VARIABLES);
    encoded.zero();

    unsigned int index=0;

    for(const unsigned long& column : info.discretized){
      const auto& m = info.dmap.find(column);

      if(m == info.dmap.end())
	continue; // bad data is ignoted
      
      auto value = m->second.find(input[column].c[0]);
    
      if(value != m->second.end()){
	encoded[index+(value->second)] = 1.0f;
      }

      index += m->second.size();
    }

    for(const unsigned long& column : info.numeric){
      const auto& bins = info.nbins.find(column);

      if(bins == info.nbins.end())
	continue; // bad data is ignored

      unsigned long bin = 0;

      if(input[column].c[0] < bins->second[bins->second.size()-1]){

	for(unsigned long k=0;k<bins->second.size();k++){
	  if(input[column].c[0] <= bins->second[k]){
	    bin = k;
	    break;
	  }
	}
	
      }
      else{ // last bin
	bin = bins->second.size();
      }

      encoded[index + bin] = 1.0f;
      
      index += (bins->second.size() + 1);
    }

    assert(index == info.NUM_DISCRETIZED_VARIABLES);

    return true;
  }

  
  class BinaryFileTransactionSource : public TransactionSourceInterface
  {
  public:
    BinaryFileTransactionSource(const BinaryVectorsFile& input) : data(&input)
    {
    }
    
    virtual const unsigned long getNumberOf() const // number of data vector pairs (x,y)
    {
      return data->getNumberOfVectors();
    }
    
    // gets index:th data points or return false (bad index or unknown error)
    virtual const bool getData(const unsigned long index,
			       Transaction& dataset) const
    {
      if(index >= data->getNumberOfVectors()) return false;

      math::vertex< math::blas_real<float> > v;

      if(data->getVector(index, v) == false)
	return false;

      dataset.clear();

      for(unsigned long i=0;i<v.size();i++){
	if(v[i] > 0.5) dataset.push_back(i); // assumes data is 0,1 valued binary data
      }

      return true;
    }

  private:

    const BinaryVectorsFile* const data;
    
  };


  // initial implementation, TODO: keep transactions on DISK and NOT loaded to memory
  bool calculateFrequentPatterns
  (const BinaryVectorsFile& input, // binary valued 0/1 vectors
   const float minimum_support, // 0.00-1.00 percentage of lines has pattern
   std::multimap< unsigned long, std::set<unsigned long> >& fpatterns)
  {
    if(input.hasFile() == false || input.getNumberOfVectors() == 0)
      return false;

    if(minimum_support < 0.0f || minimum_support > 1.0f)
      return false;

    fpatterns.clear();

    BinaryFileTransactionSource source(input);

    unsigned long min_support = minimum_support*input.getNumberOfVectors();

    if(min_support < 1000) min_support = 1000;

    const FPTree fptree(&source, min_support);

    printf("FP tree constructed (%ld %f support). Now generating frequent patterns...\n",
	   min_support, min_support/(float)input.getNumberOfVectors());
    fflush(stdout);
    
    std::set<Pattern> patterns;
    
    fptree_growth(fptree, patterns);

    printf("Number of patterns: %d\n", (int)patterns.size());
    
    // NOT DONE: convert results to multimap format

    return true;
  }
  
};

