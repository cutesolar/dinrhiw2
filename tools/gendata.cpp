// simple program creating machine learning example data that is difficult to learn
// y = max(x). The function is simple but learning it requires sorting input vectors
//             numbers and selecting the biggest one. Feedforward neural network cannot
//             learn this easily so recurrent neural networks should be maybe used.


#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <time.h>
#include <math.h>

// generates examples used on machine learning
void generate(std::vector<float>& x){

  float max = -INFINITY;

  for(unsigned int i=0;i<x.size();i++){
    x[i] = ((float)rand())/((float)RAND_MAX);
    if(i != x.size()-1) if(x[i] > max) max = x[i];
  }

  x[x.size()-1] = max;
}



int main(int argc, char** argv)
{
  if(argc != 2){
    printf("Usage: gendata <dimension_number>\n");
    return -1;
  }

  const unsigned int NUMDATA = 50000;
  const int dimension = atoi(argv[1]);

  if(dimension <= 0){
    printf("Usage: gendata <dimension_number>\n");
    return -1;
  }

  // generates data sets
  srand(time(0));
  
  FILE* handle1 = fopen("gendata_training.csv", "wt");
  FILE* handle2 = fopen("gendata_scoring.csv", "wt");
  FILE* handle3 = fopen("gendata_scoring_correct.csv", "wt");

  printf("Generating files (%d data points)..\n", NUMDATA);
  printf("(gendata_training.csv, gendata_scoring.csv, gendata_scoring_correct.csv)\n");
  

  for(unsigned int i=0;i<NUMDATA;i++){
    std::vector<float> example;
    example.resize(dimension+1);
    generate(example);

    for(unsigned int j=0;j<example.size();j++){
      fprintf(handle1, "%f ", example[j]);
    }
    fprintf(handle1, "\n");

    example.resize(dimension+1);
    generate(example);

    for(unsigned int j=0;j<example.size();j++){
      if(j != example.size()-1)
	fprintf(handle2, "%f ", example[j]);
      else
	fprintf(handle3, "%f ", example[j]);
    }
    fprintf(handle2, "\n");
    fprintf(handle3, "\n");
  }

  fclose(handle1);
  fclose(handle2);
  fclose(handle3);
  
  return 0;
}

