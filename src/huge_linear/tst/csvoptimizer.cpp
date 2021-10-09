/*
 * Regression optimizer from CSV loaded data (y=f(x|params), params are optimized to minimize MSE)xs
 * Supports HUGE data files which are mostly kept on DISK and not in RAM memory.
 *
 * Usage: csvoptimizer modelfile.cfg data_x.csv data_y.csv OR
 *        csvoptimizer modelfile.cfg -p data_x.csv new_y.csv
 *
 * If -p is used modelfile.cfg must exist and contain model to predict y values.
 * without -p parametr pseudolinear model is calculated and stored to modelfile.cfg,
 * in this case both data_x.csv and data_y.csv files must exist.
 *
 * CSV files are comma separated values file where only numbers are accepted as values.
 * 
 *
 * (C) Copyright Tomas Ukkonen 2021 
 * (part of Dinrhiw2 which is also distributed under open source license)
 * 
 */

#ifndef __whiteice_csvoptimizer_h
#define __whiteice_csvoptimizer_h

#include <stdio.h>
#include <string>
#include <string.h>
#include <unistd.h>

#include "CSVToBinaryFile.h"
#include "OneHotEncoder.h"
#include "HugeLinear.h"
#include "vertex.h"

using namespace whiteice;

//////////////////////////////////////////////////////////////////////

bool parseParameters(int argc, char** argv,
		     bool& predictOnly,
		     std::string& modelfilename,
		     std::string& inputfilename,
		     std::string& outputfilename);

//////////////////////////////////////////////////////////////////////

class BinaryFileDataSource : public whiteice::DataSourceInterface
{
public:

  BinaryFileDataSource(whiteice::BinaryVectorsFile& x_,
		       whiteice::BinaryVectorsFile& y_)
  {
    x = NULL;
    y = NULL;

    x = &x_;
    y = &y_;
  }
  
  ~BinaryFileDataSource(){
  }
  
  virtual const unsigned long getNumber() const // number of data vector pairs (x,y)
  {
    if(x->getNumberOfVectors() == y->getNumberOfVectors())
      return x->getNumberOfVectors();
    else 
      return 0;
  }
  
  virtual const unsigned long getInputDimension() const  // x input vector dimension
  {
    return x->getVectorLength();
  }
  
  virtual const unsigned long getOutputDimension() const // y output vector dimension
  {
    return y->getVectorLength();
  }
  
  // gets index:th data points or return false (bad index or unknown error)
  virtual const bool getData(const unsigned long index,
			     math::vertex< math::blas_real<float> > & x,
			     math::vertex< math::blas_real<float> >& y) const
  {
    if(index >= this->x->getNumberOfVectors()) return false;
    if(index >= this->y->getNumberOfVectors()) return false;

    if(this->x->getVector(index, x) == false) return false;
    if(this->y->getVector(index, y) == false) return false;
    
    return true;
  }
  
private:
  whiteice::BinaryVectorsFile* x;
  whiteice::BinaryVectorsFile* y;
};

//////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
  printf("CSVOPTIMIZER\n");
  printf("(C) Copyright Tomas Ukkonen 2021\n");

  bool predict = false;
  std::string modelFilename;
  std::string inputFilename;
  std::string outputFilename;

  if(parseParameters(argc, argv,
		     predict,
		     modelFilename,
		     inputFilename,
		     outputFilename) == false){
    printf("ERROR: parsing command line parameters failed.\n");
    exit(-1);
  }

  printf("INPUT:\n");
  printf("Predict: %d\n", (int)predict);
  printf("Model file: %s\n", modelFilename.c_str());
  printf("Input file: %s\n", inputFilename.c_str());
  printf("Output file: %s\n", outputFilename.c_str());

  if(predict == false){
    // input and output exists and we compute model file
    
    BinaryVectorsFile in(inputFilename + ".bin");
    BinaryVectorsFile out(outputFilename + ".bin");

    printf("Preprocessing datafiles..\n");

    if(CSVToBinaryFile(inputFilename, in) == false){
      printf("Error parsing input CSV file: %s\n", inputFilename.c_str());
      exit(-1);
    }

    if(CSVToBinaryFile(outputFilename, out) == false){
      printf("Error parsing output CSV file: %s\n", outputFilename.c_str());
      exit(-1);
    }

    printf("Input binary file dimensions: %d\n", (int)in.getVectorLength());
    printf("Input binary file samples: %d\n", (int)in.getNumberOfVectors());
    printf("Output binary file dimensions: %d\n", (int)out.getVectorLength());
    printf("Output binary file samples: %d\n", (int)out.getNumberOfVectors());

    printf("Calculating one hot encoding..\n");

    std::set<unsigned long> ignored; // empty set, all variables are used
    oneHotEncodingInfo info;

    BinaryVectorsFile intmp(inputFilename + ".bin2");

    if(calculateOneHotEncoding(in, ignored, intmp, info) == false){
      printf("Error in one hot encoding input variables.\n");
      exit(-1);
    }

    printf("Input OHE binary file dimensions: %d\n", (int)intmp.getVectorLength());
    printf("Input OHE binary file samples: %d\n", (int)intmp.getNumberOfVectors());

    bool overfit = true; // enable overfit to find a global optimum point (hopefully)
    
    HugeLinear solver(overfit);
    BinaryFileDataSource source(intmp, out);

    printf("Starting optimization solver..\n");
    
    if(solver.startOptimize(&source) == false){
      printf("Error in HugeLinear::startOptimize(). Cannot start optimiztion/FAILED.\n");
      exit(-1);
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


    printf("FIXME: Save model data: A,b and one-hot-encoding info data.\n");

    return true;
  }

  else{
  }

  return 0;
}

//////////////////////////////////////////////////////////////////////

bool parseParameters(int argc, char** argv,
		     bool& predictOnly,
		     std::string& modelfilename,
		     std::string& inputfilename,
		     std::string& outputfilename)
{
  if(argc != 4 && argc != 5) return false;

  if(argc == 5){
    if(strcmp(argv[1], "-p") != 0){
      return false;
    }
    else predictOnly = true;

    modelfilename = argv[2];
    inputfilename = argv[3];
    outputfilename = argv[4];

    // modelfilename and inputfilename must exist

    FILE* handle = fopen(modelfilename.c_str(), "r");
    if(handle == NULL) return false;
    else fclose(handle);

    handle = fopen(inputfilename.c_str(), "r");
    if(handle == NULL) return false;
    else fclose(handle);

    return true;
  }
  else{
    predictOnly = false;

    modelfilename = argv[1];
    inputfilename = argv[2];
    outputfilename = argv[3];

    // inputfilename and outputfilename must exist

    FILE* handle = fopen(inputfilename.c_str(), "r");
    if(handle == NULL) return false;
    else fclose(handle);

    handle = fopen(outputfilename.c_str(), "r");
    if(handle == NULL) return false;
    else fclose(handle);

    return true;
  }

  return false; // should not never be here
}


#endif
