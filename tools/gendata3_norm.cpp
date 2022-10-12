// simple program creating machine learning example data that is difficult to learn
// this involves matrix norms that are difficult to learn(??)
// z = f(w=x,y) = max(5-||x+y||, 0)
//
// IN PRATICE grad descent learns this problem rather well even with small neural network
// (so Q-learning in reinforcement code doesn't work for some reason)


#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <time.h>
#include <math.h>
#include <assert.h>

//#include <dinrhiw/dinrhiw.h>
#include <dinrhiw.h>

#include <random>


float norm(const std::vector<float>& x)
{
  float sum = 0.0f;
  for(unsigned int i=0;i<x.size();i++){
    sum += x[i]*x[i];
  }

  return sqrt(sum);
}

std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0,1.0);


// generates examples used for machine learning
void generate(std::vector<float>& w, std::vector<float>& z)
{
  assert(w.size() != 0);
  assert((w.size() & 1) == 0); // even number

  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> sum;
  
  x.resize(w.size()/2);
  y.resize(w.size()/2);
  sum.resize(w.size()/2);

  for(unsigned int i=0;i<x.size();i++){
    x[i] = distribution(generator);
    y[i] = distribution(generator);
  }

  for(unsigned int i=0;i<x.size();i++){
    w[i] = x[i];
    w[i+x.size()] = y[i];

    sum[i] = x[i] + y[i];
  }

  z.resize(1);

  z[0] = 5.0f - norm(sum);
  if(z[0] < 0.0f) z[0] = 0.0f;
}



int main(int argc, char** argv)
{
  if(argc != 2){
    printf("Usage: gendata3 <dimension_number>\n");
    return -1;
  }

  const unsigned int NUMDATA = 50000;
  const int dimension = atoi(argv[1]);

  if(dimension <= 0){
    printf("Usage: gendata3 <dimension_number>\n");
    return -1;
  }

  // generates data sets
  srand(time(0));
  generator.seed(time(0));
  
  FILE* handle1 = fopen("norm_test.csv", "wt");
  FILE* handle2 = fopen("norm_train_input.csv", "wt");
  FILE* handle3 = fopen("norm_train_output.csv", "wt");

  printf("Generating files (%d data points)..\n", NUMDATA);
  printf("(norm_test.csv, norm_train_input.csv, norm_train_output.csv)\n");
  

  for(unsigned int i=0;i<NUMDATA;i++){
    std::vector<float> example, result;
    example.resize(2*dimension);
    result.resize(1);
    generate(example, result);

    for(unsigned int j=0;j<example.size();j++){
      fprintf(handle1, "%f,", example[j]);
    }
    fprintf(handle1, "%f", result[0]);
    fprintf(handle1, "\n");

    example.resize(2*dimension);
    result.resize(1);
    generate(example,result);

    for(unsigned int j=0;j<example.size();j++){
      if(j!= 0) fprintf(handle2, ",%f", example[j]);
      else fprintf(handle2, "%f", example[j]);
    }
    fprintf(handle3, "%f", result[0]);

    fprintf(handle2, "\n");
    fprintf(handle3, "\n");
  }

  fclose(handle1);
  fclose(handle2);
  fclose(handle3);
  
  return 0;
}

