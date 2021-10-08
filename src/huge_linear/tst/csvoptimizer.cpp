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

//////////////////////////////////////////////////////////////////////

bool parseParameters(int argc, char** argv,
		     bool& predictOnly,
		     std::string& modelfilename,
		     std::string& inputfilename,
		     std::string& outputfilename);

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

  return 0;
}

//////////////////////////////////////////////////////////////////////

bool parseParameters(int argc, char** argv,
		     bool& predictOnly,
		     std::string& modelfilename,
		     std::string& inputfilename,
		     std::string& outputfilename)
{
  if(argc != 4 || argc != 5) return false;

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
