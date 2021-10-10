/*
 * Data discretizer to 0/1 values, One Hot Encoder for BinaryVectorsFile(s)
 * 
 * (C) Copyright Tomas Ukkonen 2021
 */

#ifndef __whiteice_OneHotEncoder_h
#define __whiteice_OneHotEncoder_h

#include "CSVToBinaryFile.h"
#include "HugeLinear.h"
#include "vertex.h"

#include <vector>
#include <set>
#include <map>

namespace whiteice
{
  struct oneHotEncodingInfo
  {
    std::set<unsigned long> discretized; // discretized column variables
    std::map< unsigned long, std::map<float, unsigned long> > dmap; // maps discretized variables to indicator vars

    std::set<unsigned long> numeric; // numerical column variables
    std::map< unsigned long, std::vector<float> > nbins; // divisions to divide numerical value to bin and 0/1 value

    unsigned long NUM_ORIGINAL_VARIABLES;
    unsigned long NUM_DISCRETIZED_VARIABLES;
    
  };

  
  bool calculateOneHotEncoding(const BinaryVectorsFile& input,
			       std::set<unsigned long>& ignoreVariables,
			       BinaryVectorsFile& output,
			       struct oneHotEncodingInfo& info);

  bool oneHotEncoding(const math::vertex< math::blas_real<float> >& input,
		      math::vertex< math::blas_real<float> >& encoded,
		      const struct oneHotEncodingInfo& info);


  //////////////////////////////////////////////////////////////////////

  // calculates frequent patterns from data using FP-Growth algorithm
  // returns multimap of support in rows, and set of data vector column values (unsigned long)
  // NOW: loads data to memory, modify to keep data on disk!!
  bool calculateFrequentPatterns
  (const BinaryVectorsFile& input, // binary valued 0/1 vectors
   const float minimum_support, // 0.00-1.00 percentage of lines has pattern
   std::multimap< unsigned long, std::set<unsigned long> >& fpatterns);


  //////////////////////////////////////////////////////////////////////

  class BinaryFileFPDataSource : public DataSourceInterface
  {
  public:
    
    BinaryFileFPDataSource(const whiteice::BinaryVectorsFile& x_,
			 const std::multimap< unsigned long, std::set<unsigned long> >& fpatterns_,
			 const whiteice::BinaryVectorsFile& y_);
  
    virtual const unsigned long getNumber() const; // number of data vector pairs (x,y)
  
    virtual const unsigned long getInputDimension() const;  // x input vector dimension
  
    virtual const unsigned long getOutputDimension() const; // y output vector dimension
  
    // gets index:th data points or return false (bad index or unknown error)
    virtual const bool getData(const unsigned long index,
			       math::vertex< math::blas_real<float> > & x,
			       math::vertex< math::blas_real<float> >& y) const;
  
  private:
    const whiteice::BinaryVectorsFile* x;
    const whiteice::BinaryVectorsFile* y;

    const std::multimap< unsigned long, std::set<unsigned long> >& fpatterns;
  };
  
  
};

#endif
