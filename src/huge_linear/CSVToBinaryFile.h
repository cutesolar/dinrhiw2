/*
 * CSV to BinaryVectorsFile converter (floats)
 *
 * don't load data into memory so we can process HUGE files/datasets.
 */

#ifndef __CSVToBinaryFile_h
#define __CSVToBinaryFile_h

#include <stdio.h>
#include <string>
#include "vertex.h"

namespace whiteice
{
  // numVectors is saved to filename. If binfile="filename.dat" the filename
  // saved is "filename.dat.<numVectors>"
  class BinaryVectorsFile
  {
  public:
    BinaryVectorsFile();
    BinaryVectorsFile(const std::string& binfile);
    ~BinaryVectorsFile();

    bool setVectorLength(const unsigned long veclen);
    unsigned long getVectorLength() const;

    bool setNumberOfVectors(const unsigned int numvec);
    unsigned long getNumberOfVectors() const;

    bool getVector(const unsigned long index, math::vertex< math::blas_real<float> >& v) const;
    bool setVector(const unsigned long index, const math::vertex< math::blas_real<float> >& v);

    bool addVector(const math::vertex< math::blas_real<float> >& v);
    bool removeVector(const unsigned long index);

    bool load(const std::string& binfile, const unsigned long numVectors=0);
    bool save(const std::string& binfile) const;

    // sets all values to zero (slow for large files)
    bool zero();
    
  private:

			 
    unsigned long numVectors;
    unsigned long vectorLength;
    FILE* binaryFile;
  };
  
  // csv: comma separeted file of floats without labels
  bool CSVToBinaryFile(const std::string& csvfile,
		       BinaryVectorsFile& binout);

  bool BinaryFileToCSV(const BinaryVectorsFile& binout,
		       const std::string& csvfile);
  
};

#endif
