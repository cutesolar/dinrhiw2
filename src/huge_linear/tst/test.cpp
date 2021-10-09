/*
 * HugeLinear optimizer testcases
 */

#include <stdio.h>
#include <unistd.h>
#include <vector>

#include "HugeLinear.h"
#include "CSVToBinaryFile.h"
#include "OneHotEncoder.h"

using namespace whiteice;

//////////////////////////////////////////////////////////////////////

bool simple_test();
bool helper_functions_test();
bool binaryfile_test();
bool csv_load2bin_test();
bool one_hot_encoding_test();

int main(void)
{
  // simple_test();

  // helper_functions_test();

  // binaryfile_test();

  // csv_load2bin_test();

  one_hot_encoding_test();

  return 0;;
}

//////////////////////////////////////////////////////////////////////

class MemoryDataSource : public whiteice::DataSourceInterface
{
public:
  MemoryDataSource(const std::vector< math::vertex< math::blas_real<float> > >& x,
		   const std::vector< math::vertex< math::blas_real<float> > >& y){
    if(x.size() == y.size()){
      this->x = x;
      this->y = y;
    }
  }
  
  ~MemoryDataSource(){
    x.clear();
    y.clear();
  }
  
  virtual const unsigned long getNumber() const // number of data vector pairs (x,y)
  {
    return x.size();
  }
  
  virtual const unsigned long getInputDimension() const  // x input vector dimension
  {
    if(x.size() <= 0) return 0;
    else return x[0].size();
  }
  
  virtual const unsigned long getOutputDimension() const // y output vector dimension
  {
    if(y.size() <= 0) return 0;
    else return y[0].size();
  }
  
  // gets index:th data points or return false (bad index or unknown error)
  virtual const bool getData(const unsigned long index,
			     math::vertex< math::blas_real<float> > & x,
			     math::vertex< math::blas_real<float> >& y) const
  {
    if(index >= this->x.size()) return false;

    x = this->x[index];
    y = this->y[index];
    
    return true;
  }
  
private:
  std::vector< math::vertex< math::blas_real<float> > > x;
  std::vector< math::vertex< math::blas_real<float> > > y;
  
};

//////////////////////////////////////////////////////////////////////

bool simple_test()
{
  printf("HugeLinear first test: tests code runs without erros.\n");

  // generates simple test problem that fits perfectly to a linear problem

  std::vector< math::vertex< math::blas_real<float> > > datax;
  std::vector< math::vertex< math::blas_real<float> > > datay;

  {
    whiteice::RNG< math::blas_real<float> > rng;
    
    math::matrix< math::blas_real<float> > A;
    math::vertex< math::blas_real<float> > b;

    unsigned int inputDim = 1 + (rng.rand() % 100);
    unsigned int outputDim = 1 + (rng.rand() % 10);

    A.resize(outputDim, inputDim);
    b.resize(outputDim);

    rng.normal(A);
    rng.normal(b);

    const unsigned long N = 20000; // number of datapoints

    for(unsigned long i=0;i<N;i++){
      math::vertex< math::blas_real<float> > x, y;
      x.resize(inputDim);
      rng.normal(x);

      y = A*x + b;

      datax.push_back(x);
      datay.push_back(y);
    }
  }

  whiteice::HugeLinear solver;
  MemoryDataSource source(datax, datay);
  
  if(solver.startOptimize(&source) == false){
    printf("Starting solver FAILED.\n");
    return false;
    
  }
  else{
    printf("Succesfully started solver.\n");
  }
  
  unsigned int iteration_seen = 0;

  while(solver.isRunning()){
    sleep(1);

    const unsigned int iter = solver.getIterations();

    if(iter > iteration_seen){
      iteration_seen = iter;

      printf("Solver error after %d iterations. MSE: %f.\n",
	     iter,
	     solver.estimateSolutionMSE());
    }
  }

  solver.stopOptimize();

  printf("Final solver error after %d iterations. MSE: %f.\n",
	 solver.getIterations(),
	 solver.estimateSolutionMSE());

  return true;
}

//////////////////////////////////////////////////////////////////////

bool helper_functions_test()
{
  // test 1: test tokenize function

  printf("Testing whiteice::tokenize() function\n");

  const std::string line = "4.2, -10.5, 1e2,0.31415927,\r\n";
  const std::string separators = ",;";

  std::vector<std::string> tokens;

  if(whiteice::tokenize(line, separators, tokens) == false){
    printf("ERROR: tokenize() function FAILED.\n");
    return false;
  }

  if(tokens.size() != 5){
    printf("ERROR: incorrect number of tokens: %d\n", (int)tokens.size());

    printf("TOKENS:\n");
    for(unsigned int i=0;i<tokens.size();i++)
      printf("Token %d: '%s'\n", i, tokens[i].c_str());

    return false;
  }

  if(tokens[0] != "4.2" || tokens[1] != "-10.5" || tokens[2] != "1e2" ||
     tokens[3] != "0.31415927" || tokens[4] != ""){

    printf("ERROR: Found incorrect tokens from the test example.\n");
    return false;
  }

  printf("whiteice::tokenize() function appear to work correctly\n");


  //////////////////////////////////////////////////////////////////////

  // 2. test csv file line parsing

  printf("Testing CSV line parsing..\n");

  const std::string badline = "4.2, -10.5, 1e2,0.314159,\r\n";
  const std::string goodline = "4.2, -10.5, 1e2,0.314159 \r\n";

  math::vertex< math::blas_real<float> > vec;

  if(whiteice::parseCSVFloatsLine(badline, separators, vec) == true){
    printf("ERROR: parseCSVFloatsLine() returns true with bad line data.\n");
    return false;
  }

  if(whiteice::parseCSVFloatsLine(goodline, separators, vec) == false){
    printf("ERROR: parseCSVFloatsLine() returns false with good line data.\n");
    return false;
  }

  if(vec.size() != 4){
    printf("ERROR: returned data vector has bad dimensions (%d is not 4)\n", (int)vec.size());
    return false;
  }

  if(vec[0] != 4.2f || vec[1] != -10.5f || vec[2] != 1e2 || vec[3] != 0.314159){
    printf("ERROR: parsing CSV line gave wrong data values.\n");
    return false;
  }

  printf("CSV line parsing seem to work ok.\n");
  
  return true;
}

//////////////////////////////////////////////////////////////////////

bool binaryfile_test()
{
  printf("class BinaryVectorsFile testing..\n");

  whiteice::BinaryVectorsFile binfile;

  if(binfile.setVectorLength(100) == true){
    printf("ERROR: BinaryVectorsFile::setVectorLength() successfully without file.\n");
    return false;
  }

  if(binfile.setFile("vectors.dat") == false){
    printf("ERROR: BinaryVectorsFile::setFilename() gives error with correct filename.\n");
    return false;
  }

  if(binfile.setVectorLength(100) == false){
    printf("ERROR: BinaryVectorsFile::setVectorLength() returns error with good file.\n");
    return false;
  }

  if(binfile.setNumberOfVectors(1000) == false){
    printf("ERROR: BinaryVectorsFile::setNumberOfVectors() returns error with good file.\n");
    return false;
  }

  if(binfile.zero() == false){
    printf("ERROR: BinaryVectorsFile::zero() returns error with good file.\n");
    return false;
  }

  for(unsigned int n=0;n<10;n++){
    const unsigned long index = rand() % binfile.getNumberOfVectors();
    math::vertex< math::blas_real<float> > vec;

    if(binfile.getVector(index, vec) == false){
      printf("ERROR: Accessing %d:th (max %d) vector from BinaryVectorsFile failed.\n",
	     (int)index, (int)binfile.getNumberOfVectors());
      return false;
    }
    
  }

  // checks that adding a new vector works
  math::vertex< math::blas_real<float> > vec;
  vec.resize(binfile.getVectorLength());
  vec.zero();

  if(binfile.addVector(vec) == false){
    printf("ERROR: Adding vector to BinaryVectorsFile failed.\n");
    return false;
  }

  if(binfile.getNumberOfVectors() != 1001){
    printf("ERROR: Adding vector to BinaryVectorsFile didn't increase binfile array size correctly.\n");
    return false;
  }

  printf("Simple class whiteice::BinaryVectorsFile tests worked OK.\n");
  
  return true;
}

//////////////////////////////////////////////////////////////////////

bool csv_load2bin_test()
{
  printf("CSVToBinaryFile() function tests..\n");

  // test if loading generated CSV file (3 data lines + one empty line) works OK
  {
    const std::string line1 = " 1.0, 1.0, 1.0, 0.10,3.1415927\n";
    const std::string line2 = " 0.4,-1.1, 1.1, 1e1, 6.6  \n";
    const std::string line3 = "+4.4, 7.1, 100,5.1 , 8.8\r\n";
    const std::string line4 = "\n";

    const std::string filename = "datalines.csv";

    FILE* handle = fopen(filename.c_str(), "w");

    fputs(line1.c_str(), handle);
    fputs(line2.c_str(), handle);
    fputs(line3.c_str(), handle);
    fputs(line4.c_str(), handle);

    fclose(handle);

    BinaryVectorsFile bin("storage.bin");
    bin.setVectorLength(5);

    if(CSVToBinaryFile(filename, bin) == false){
      printf("ERROR: loading CSV file data to binary file FAILED!\n");
      return false;
    }

    // check if binary file is good

    if(bin.getVectorLength() != 5){
      printf("ERROR: After CSVToBinaryFile() call binary file vector length is not 5 anymore.\n");
      return false;
    }

    if(bin.getNumberOfVectors() != 3){
      printf("ERROR: After CSVToBinaryFile() call binary file number of vectors is not 3 (from CSV file).\n");
      return false;
    }
    
  }

  
  printf("CSVToBinaryFile() function worked OK..\n");
  
  return true;
}

//////////////////////////////////////////////////////////////////////


bool one_hot_encoding_test()
{
  printf("Testing OneHotEncoder tests..\n");

  // generates test CSV file
  std::string filename = "one_hot_encoding_data.csv";
  const unsigned int DIM = 10;
  const unsigned int NSAMPLES=1000;
    
  {
    std::string line;

    RNG< math::blas_real<float> > rng;
    
    FILE* handle = fopen(filename.c_str(), "w");

    char buffer[80];

    for(unsigned int i=0;i<NSAMPLES;i++){
      snprintf(buffer, 80, "%f", rng.uniform().c[0]);
      line = buffer;
      
      for(unsigned int d=1;d<DIM;d++){
	snprintf(buffer, 80, ",%f", rng.uniform().c[0]);
	line += buffer;
      }

      line += "\r\n";

      fputs(line.c_str(), handle);
    }
    
    fclose(handle);
  }

  BinaryVectorsFile in, out;
  struct oneHotEncodingInfo info;
  std::set<unsigned long> ignored; // empty set

  in.setFile("one_hot_encoding_data.bin");

  if(CSVToBinaryFile(filename, in) == false){
    printf("ERROR: loading data using CSVToBinaryFile FAILED.\n");
    return false;
  }

  out.setFile("one_hot_encoding_data.bin2");

  if(calculateOneHotEncoding(in, ignored, out, info) == false){
    printf("ERROR: one-hot-encoding data to binary file FAILED.\n");
    return false;
  }

  
  printf("One Hot Encoding code seems to function OK.\n");
  
  return true;
}
