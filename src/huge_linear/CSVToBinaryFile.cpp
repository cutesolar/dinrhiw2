
#include "CSVToBinaryFile.h"

#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <assert.h>

#include <bits/stdc++.h>


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
    this->setFile(binfile);
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

  // sets and opens file for the data, tries to load() binfile but if it fails
  // creates a new file, contents of data is unspecified if new file is created
  bool BinaryVectorsFile::setFile(const std::string& binfile)
  {
    if(load(binfile) == true) return true;

    // creates a new empty file

    FILE* handle = fopen(binfile.c_str(), "w+");
    if(handle == NULL) return false;

    // resize file to data size
    if(ftruncate(fileno(handle), (off_t)(vectorLength*numVectors*sizeof(float))) == 0){
      fseek(handle, 0, SEEK_SET);
      if(binaryFile) fclose(binaryFile);
      binaryFile = handle;
      
      return true;
    }
    else{
      fclose(handle);
      return false;
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


  bool BinaryVectorsFile::clear()
  {
    return this->setNumberOfVectors(0);
  }

  //////////////////////////////////////////////////////////////////////

  
  void trim(std::string& line)
  {
    // removes spaces and control characters from start end of tok
    
    const std::string WHITESPACE = " \n\r\t\f\v";
    
    size_t start = line.find_first_not_of(WHITESPACE);
    
    if(start == std::string::npos) line = "";
    else line = line.substr(start);
    
    size_t end = line.find_last_not_of(WHITESPACE);
    
    if(end == std::string::npos) line = "";
    else line = line.substr(0, end + 1);
  }
  
  
  // tokenizes string by using delimiters.
  bool tokenize(const std::string& line, const std::string& separators,
		std::vector<std::string>& tokens)
  {
    size_t tokenStart = 0;

    // don't currently allow spaces as separators
    for(unsigned long i=0;i<separators.length();i++)
      if(separators[i] == ' ') return false;

    tokens.clear();

    size_t delimPos = line.find_first_of(separators);    

    while(delimPos != std::string::npos){

      std::string tok = line.substr(tokenStart, delimPos - tokenStart);

      trim(tok);
      
      tokens.push_back(tok);
      
      delimPos++;
      
      tokenStart = delimPos;
      
      delimPos = line.find_first_of(separators, delimPos);
    }
    
    // (delimPos == string::npos)
    
    std::string tok = line.substr(tokenStart, line.length() - tokenStart);

    trim(tok);
    
    tokens.push_back(tok);
    
    
    return (tokens.size() > 0);
  }

  // parses single CSV file line to vertex values
  bool parseCSVFloatsLine(const std::string& line, const std::string& SEPARATORS,
			  math::vertex< math::blas_real<float> >& data)
  {
    std::vector<std::string> tokens;

    if(tokenize(line, SEPARATORS, tokens) == false)
      return false;

    if(tokens.size() == 0) return false;

    data.resize(tokens.size());

    for(unsigned long i=0;i<tokens.size();i++){

      std::string::size_type sz = 0;

      try{
	float value = std::stof(tokens[i], &sz);

	if(sz == 0){
	  return false;
	}

	data[i] = value;
      }
      catch(const std::out_of_range& e){
	return false;
      }
      catch(const std::invalid_argument& e){
	return false;
      }
      
    }

    return (data.size() > 0);
  }
    
  
  // csv: comma separeted file of floats without labels
  bool CSVToBinaryFile(const std::string& csvfile,
		       BinaryVectorsFile& binout)
  {
    // parse CSV file

    // assume file to be text file with separators = "," or ";" or "space".
    // assume floating point numbers are have format 3.2323 or -4324.323 or +232.2e10

    const std::string SEPARATORS = ",;";

    // lue rivi, poista control characters ja splittaa separaattoreilla.
    // tarkista että kahdella ensimmäisellä rivillä on sama määrä kenttiä
    // tämän jälkeen käsittele muuta binout muotoon riveittäin (vähän dataa muistissa kerralla)

    FILE* handle = fopen(csvfile.c_str(), "rt");
    const unsigned long BUFLEN = 512*1024*1024; // 0.5 GB of buffer size
    char* buffer = (char*)malloc(BUFLEN);

    if(handle == NULL){
      if(buffer) free(buffer);
      return false;
    }

    if(buffer == NULL){
      fclose(handle);
      return false;
    }

    {
      std::string line1, line2;
      
      if(fgets(buffer, BUFLEN, handle) == NULL){
	fclose(handle);
	free(buffer);
	return false;
      }
      
      line1 = buffer;

      if(fgets(buffer, BUFLEN, handle) == NULL){
	fclose(handle);
	free(buffer);
	return false;
      }
      
      line2 = buffer;

      math::vertex< math::blas_real<float> > vector1, vector2;

      if(parseCSVFloatsLine(line1, SEPARATORS, vector1) == false){
	fclose(handle);
	free(buffer);
	return false;
      }

      if(parseCSVFloatsLine(line2, SEPARATORS, vector2) == false){
	fclose(handle);
	free(buffer);
	return false;
      }

      if(vector1.size() != vector2.size()){
	fclose(handle);
	free(buffer);
	return false;
      }

      // configures binout file

      if(binout.clear() == false){
	fclose(handle);
	free(buffer);
	return false;
      }

      
      if(binout.setVectorLength(vector1.size()) == false){
	fclose(handle);
	free(buffer);
	return false;
      }
    }

    if(fseek(handle, 0, SEEK_SET) != 0){
      fclose(handle);
      free(buffer);
      return false;
    }

    // NOW parse the file and store results to binout
    // jos virhe parsetuksessa niin nollaa binout tiedoston koko ja palauta false

    char* readvalue = NULL;
    readvalue = fgets(buffer, BUFLEN, handle);

    std::string line;
    math::vertex< math::blas_real<float> > data;

    while(readvalue != NULL){

      // parses file
      
      line = buffer;

      trim(line);

      if(line.length() > 0){

	if(parseCSVFloatsLine(line, SEPARATORS, data) == false){
	  binout.clear();
	  fclose(handle);
	  free(buffer);
	  return false;
	}

	if(binout.addVector(data) == false){
	  binout.clear();
	  fclose(handle);
	  free(buffer);
	  return false;
	}
      }
      
      readvalue = fgets(buffer, BUFLEN, handle);
    }

    return (binout.getNumberOfVectors() > 0);
  }

  
  bool BinaryFileToCSV(const BinaryVectorsFile& binout,
		       const std::string& csvfile)
  {
    assert(0); // not implemented
    return false;
  }

  
};
