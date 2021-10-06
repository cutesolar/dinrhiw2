
#include "CSVToBinaryFile.h"
#include <unistd.h>
#include <string.h>
#include <sys/types.h>

namespace whiteice
{

  BinaryVectorsFile::BinaryVectorsFile()
  {
    numVectors = 0;
    vectorLength = 0;
    binaryFile = 0;
  }
  
  BinaryVectorsFile::BinaryVectorsFile(const std::string& binfile)
  {
    this->load(binfile);
  }
  
  BinaryVectorsFile::~BinaryVectorsFile()
  {
    if(binaryFile) fclose(binaryFile);
    binaryFile = NULL;
  }
  
  bool BinaryVectorsFile::setVectorLength(const unsigned long veclen)
  {
    if(veclen == 0) return false;
    if(vectorLength == veclen) return true;
    if(binaryFile == NULL) return false;

    // resize file to have size (veclen*numVectors*sizeof(float))

    if(ftruncate(fileno(binaryFile), (off_t)(veclen*numVectors*sizeof(float))) == 0){
      fseek(binaryFile, 0, SEEK_SET);
      vectorLength = veclen;
      return true;
    }
    else{
      return false;
    }
  }
  
  unsigned long BinaryVectorsFile::getVectorLength() const
  {
    return vectorLength;
  }

  bool BinaryVectorsFile::setNumberOfVectors(const unsigned int numvec)
  {
    if(numVectors == numvec) return true;
    if(binaryFile == NULL) return false;

    // resize file to have size (veclen*numVectors*sizeof(float))

    if(ftruncate(fileno(binaryFile), (off_t)(vectorLength*numvec*sizeof(float))) == 0){
      fseek(binaryFile, 0, SEEK_SET);
      numVectors = numvec;
      return true;
    }
    else{
      return false;
    }
  }
  
  unsigned long BinaryVectorsFile::getNumberOfVectors() const
  {
    return numVectors;
  }
  
  bool BinaryVectorsFile::getVector(const unsigned long index,
				    math::vertex< math::blas_real<float> >& v) const
  {
    if(binaryFile == NULL) return false;
    if(index >= numVectors || vectorLength == 0) return false;

    v.resize(vectorLength);

    if(fseek(binaryFile, index*vectorLength*sizeof(float), SEEK_SET) == 0){

      if(fread(&(v[0]), sizeof(float), vectorLength, binaryFile) == vectorLength)
	return true;
      else
	return false;
      
    }
    else return false;
  }
  
  bool BinaryVectorsFile::setVector(const unsigned long index,
				    const math::vertex< math::blas_real<float> >& v)
  {
    if(binaryFile == NULL) return false;
    if(index >= numVectors || vectorLength == 0) return false;
    if(v.size() != vectorLength) return false;

    if(fseek(binaryFile, index*vectorLength*sizeof(float), SEEK_SET) == 0){

      if(fwrite(&(v[0]), sizeof(float), vectorLength, binaryFile) == vectorLength)
	return true;
      else
	return false;
      
    }
    else return false;
  }
  
  
  bool BinaryVectorsFile::addVector(const math::vertex< math::blas_real<float> >& v)
  {
    if(binaryFile == NULL) return false;
    if(v.size() != vectorLength) return false;

    if(fseek(binaryFile, numVectors*vectorLength*sizeof(float), SEEK_SET) == 0){

      if(fwrite(&(v[0]), sizeof(float), vectorLength, binaryFile) == vectorLength){
	numVectors++;
	return true;
      }
      else{
	return false;
      }
    }
    else return false;
  }
  
  bool BinaryVectorsFile::removeVector(const unsigned long index)
  {
    if(binaryFile == NULL) return false;
    if(index >= numVectors || vectorLength == 0) return false;

    if(numVectors >= 2){
      math::vertex< math::blas_real<float> > v;
      if(getVector(numVectors-1, v) == false) return false;
      if(setVector(index, v) == false) return false;
      if(setNumberOfVectors(numVectors-1) == false) return false;

      return true;
    }
    else{
      if(setNumberOfVectors(numVectors-1) == false) return false;
      return true;
    }
  }
  
  bool BinaryVectorsFile::load(const std::string& binfile,
			       const unsigned long numVectors)
  {
    FILE* handle = fopen(binfile.c_str(), "w+");
    if(handle == NULL) return false;

    fseek(handle, 0L, SEEK_END);
    const unsigned long size = ftell(handle);
    fseek(handle, 0L, SEEK_SET);
    
    if(numVectors != 0){
      
      if((size % numVectors) != 0){
	fclose(handle);
	return false;
      }

      this->numVectors = numVectors;
      this->vectorLength = size / numVectors;

      if(binaryFile) fclose(binaryFile);
      binaryFile = handle;

      return true;
    }
    else{
      // tries to detect numVectors from a filename

      const char* start = binfile.c_str();
      unsigned int npos = binfile.find_last_of('.');
      char* start_pos = ((char*)start) + npos;
      char* endptr = start_pos;

      const long numvec = strtol(start_pos, &endptr, 10);

      if(endptr == start_pos){
	fclose(handle);
	return false;
      }

      if((size % ((unsigned long)numvec)) != 0){
	fclose(handle);
	return false;
      }

      this->numVectors = (unsigned long)numvec;
      this->vectorLength = size / numVectors;

      if(binaryFile) fclose(binaryFile);
      binaryFile = handle;

      return true;
    }

    
  }
  
  bool BinaryVectorsFile::save(const std::string& binfile) const
  {
    if(binaryFile == NULL) return false;
    if(this->numVectors == 0 || this->vectorLength == 0)
      return false;
    
    std::string newfilename = binfile + std::string(".") + std::to_string(numVectors);

    FILE *handle = fopen(newfilename.c_str(), "w");

    if(handle == NULL) return false;

    float* v = (float*)malloc(sizeof(float)*vectorLength);
    
    if(v == NULL){
      fclose(handle);
      return false;
    }

    if(fseek(binaryFile, 0, SEEK_SET) != 0){
      fclose(handle);
      free(v);
      return false;
    }

    for(unsigned long i=0;i<numVectors;i++){
      if(fread(v, sizeof(float), vectorLength, binaryFile) != vectorLength)
	return false;
    
      if(fwrite(v, sizeof(float), vectorLength, binaryFile) != vectorLength)
	return false;
    }

    free(v);
    fclose(handle);

    return true;
  }

  
  // sets all values to zero (slow for large files)
  bool BinaryVectorsFile::zero()
  {
    if(binaryFile == NULL) return false;
    if(this->numVectors == 0 || this->vectorLength == 0)
      return false;

    float* v = (float*)malloc(sizeof(float)*vectorLength);
    
    if(v == NULL){
      return false;
    }

    if(fseek(binaryFile, 0, SEEK_SET) != 0){
      free(v);
      return false;
    }

    memset(v, 0, vectorLength);

    for(unsigned long i=0;i<numVectors;i++){
      if(fwrite(v, sizeof(float), vectorLength, binaryFile) != vectorLength)
	return false;
    }

    free(v);
    
    return true;
  }

  //////////////////////////////////////////////////////////////////////
  
  // csv: comma separeted file of floats without labels
  bool CSVToBinaryFile(const std::string& csvfile,
		       BinaryVectorsFile& binout)
  {
    return false;
  }

  bool BinaryFileToCSV(const BinaryVectorsFile& binout,
		       const std::string& csvfile)
  {
    return false;
  }

  
};
