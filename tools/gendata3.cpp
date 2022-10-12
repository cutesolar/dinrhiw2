// simple program creating machine learning example data that is DIFFICULT to learn
// this is cryptographic function so its inverse should be VERY hard to predict
// by neural network
// [tensorflow neural network 10-140-140-10 gives 12.77 error when using
//  mean absolute error per character and scaling error back to 0-255 interval]
// 
// D = 5 (default)
// y = D_first_bits_of(sha256(Random_D_Letter_Ascii_String));


#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <string>
#include <iostream>

#include <dinrhiw/dinrhiw.h>
#include "SHA.h"


// generates examples used for machine learning
void generate(std::vector<float>& x)
{
  using namespace whiteice::crypto;
  using namespace whiteice;
  
  assert(x.size() != 0);
  assert((x.size() & 1) == 0); // even number

  if(x.size()>512) x.resize(512);

  std::vector<float> v;
  v.resize(x.size()/2);

  whiteice::crypto::SHA SHA256(256);
  
  unsigned char* message = NULL;
  char hash256[32];

  memset(hash256, 0, 32*sizeof(char));

  message = (unsigned char*)malloc(sizeof(char)*v.size());

  for(unsigned int i=0;i<v.size();i++){
    message[i] = rand() % 256;
    v[i] = (float)message[i]; // => NO!: transforms 0-255 values to [0,1] interval
    x[i] = v[i];
  }

  if(SHA256.hash(&message, v.size()*8, (unsigned char*)hash256) == false){
    std::cout << "ERROR: SHA-256 hash calculations FAILED. ABORT.\n";
    exit(-1);
  }

  free(message);

  for(unsigned int i=0;i<v.size();i++){

    unsigned int ch = i / 8;
    unsigned int bit = i % 8;
    
    x[i+v.size()] = (float)((hash256[ch] >> bit) & 0x01); // extracts first bits of SHA-256 hash 
  }
}



int main(int argc, char** argv)
{
  if(argc != 2 && argc != 3){
    printf("Usage: gendata <dimension_number> [-h]\n");
    return -1;
  }

  const unsigned int NUMDATA = 50000;
  const int dimension = atoi(argv[1]);

  bool useHeaders = false; 

  if(argc == 3)
    if(strcmp("-h", argv[2]) == 0)
      useHeaders = true;
  

  if(dimension <= 0){
    printf("Usage: gendata <dimension_number>\n");
    return -1;
  }

  // generates data sets
  srand(whiteice::rng.rand() + time(0));
  
  FILE* handle1 = fopen("hash_test.csv", "wt");
  FILE* handle2 = fopen("hash_train_input.csv", "wt");
  FILE* handle3 = fopen("hash_train_output.csv", "wt");

  printf("Generating files (%d data points)..\n", NUMDATA);
  printf("(hash_test.csv, hash_train_input.csv, hash_train_output.csv)\n");

  if(useHeaders){
    std::string str;
    char buffer[20];

    str = "Id";

    for(int i=0;i<dimension;i++){
      sprintf(buffer, ",Char%d", i+1);
      str += std::string(buffer);
    }

    for(int i=0;i<dimension;i++){
      sprintf(buffer, ",Bit%d", i+1);
      str += std::string(buffer);
    }

    str += std::string("\n");

    fprintf(handle1, "%s", str.c_str());

    str = "Id";

    for(int i=0;i<dimension;i++){
      sprintf(buffer, ",Char%d", i+1);
      str += std::string(buffer);
    }

    str += std::string("\n");
    
    fprintf(handle2, "%s", str.c_str());
    
    str = "Id";

    for(int i=0;i<dimension;i++){
      sprintf(buffer, ",Bit%d", i+1);
      str += std::string(buffer);
    }

    str += std::string("\n");
    
    fprintf(handle3, "%s", str.c_str());
  }

  for(unsigned int i=0;i<NUMDATA;i++){
    std::vector<float> example;
    example.resize(2*dimension);
    generate(example);

    if(useHeaders){
      fprintf(handle1, "%d,%f", i+1, example[0]);
    }
    else{
      fprintf(handle1, "%f", example[0]);
    }

    for(unsigned int j=0;j<example.size();j++){
      fprintf(handle1, ",%f", example[j]);
    }
    fprintf(handle1, "\n");

    
    example.resize(2*dimension);
    generate(example);

    for(unsigned int j=0;j<example.size();j++){
      if(j < (example.size()/2)){
	if(j == 0){
	  if(useHeaders){
	    fprintf(handle2, "%d,%f", i+1, example[j]);
	  }
	  else{
	    fprintf(handle2, "%f", example[j]);
	  }
	}
	else{
	  fprintf(handle2, ",%f", example[j]);
	}
      }
      else{
	if(j == (example.size()/2)){
	  if(useHeaders){
	    fprintf(handle3, "%d,%f", i+1, example[j]);
	  }
	  else{
	    fprintf(handle3, "%f", example[j]);
	  }
	}
	else{
	  fprintf(handle3, ",%f", example[j]);
	}
	
      }
    }
    
    fprintf(handle2, "\n");
    fprintf(handle3, "\n");
  }

  fclose(handle1);
  fclose(handle2);
  fclose(handle3);
  
  return 0;
}

