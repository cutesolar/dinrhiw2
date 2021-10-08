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

    // sets and opens file for the data, tries to load() binfile but if it fails
    // creates a new file
    bool setFile(const std::string& binfile);

    // tells if we have file open
    bool hasFile() const;

    // closes binary file and deletes it, if it is open.
    bool deleteFile();

    bool load(const std::string& binfile, const unsigned long numVectors=0);
    bool save(const std::string& binfile) const;

    // sets all values to zero (slow for large files)
    bool zero();

    // empties vectors file
    bool clear();
    
  private:
			 
    unsigned long numVectors;
    unsigned long vectorLength;
    
    FILE* binaryFile;
    std::string binaryFilename;
  };

  // helper functions

  // removes spaces and control characters from the start and end of the string
  void trim(std::string& line);

  // tokenizes string by using delimiters.
  bool tokenize(const std::string& line, const std::string& separators,
		std::vector<std::string>& tokens);

  // parses single CSV file line to vertex values
  bool parseCSVFloatsLine(const std::string& line, const std::string& SEPARATORS,
			  math::vertex< math::blas_real<float> >& data);
  
  // csv: comma separeted file of floats without labels
  bool CSVToBinaryFile(const std::string& csvfile,
		       BinaryVectorsFile& binout);

  // FIXME: not implemented
  bool BinaryFileToCSV(const BinaryVectorsFile& binout,
		       const std::string& csvfile);
  
};

#endif
