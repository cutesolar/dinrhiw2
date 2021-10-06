/*
 * Data discretizer to 0/1 values, One Hot Encoder for BinaryVectorsFile(s)
 * 
 * (C) Copyright Tomas Ukkonen 2021
 */

#ifndef __whiteice_OneHotEncoder_h
#define __whiteice_OneHotEncoder_h

#include "CSVToBinaryFile.h"
#include "vertex.h"

#include <vector>
#include <map>

namespace whiteice
{
  struct oneHotEncodingInfo
  {
    std::vector<unsigned long> discretized; // discretized column variables
    std::vector< std::map<float, unsigned long> > dmap; // maps discretized variables to indicator vars

    std::vector<unsigned long> numeric; // numerical column variables
    std::vector< std::vector<float> > nbins; // divisions to divide numerical value to bin and 0/1 value
    
  };

  bool calculateOneHotEncoding(const BinaryVectorsFile& input,
			       BinaryVectorsFile& output,
			       struct oneHotEncodingInfo& info);

  bool oneHotEncoding(const math::vertex< math::blas_real<float> >& input,
		      math::vertex< math::blas_real<float> >& encoded,
		      const struct oneHotEncodingInfo& info);
};

#endif
