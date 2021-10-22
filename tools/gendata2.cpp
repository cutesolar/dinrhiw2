// simple program creating machine learning example data that is difficult to learn
// y = sort(x). 


#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <time.h>
#include <math.h>
#include <assert.h>

int qcompare(const void* a, const void* b){
  float f = (*(float*)a - *(float*)b);

  if(f < 0.0f) return -1.0f;
  else if(f == 0.0f) return 0.0f;
  else return 1.0f;
}

// generates examples used for machine learning
void generate(std::vector<float>& x)
{
  assert(x.size() != 0);
  assert((x.size() & 1) == 0); // even number

  std::vector<float> v;
  v.resize(x.size()/2);

  for(unsigned int i=0;i<v.size();i++){
    v[i] = 2.0f*((float)rand())/((float)RAND_MAX) - 1.0f; // [-1,1]
    x[i] = v[i];
  }

  qsort(v.data(), v.size(), sizeof(float), qcompare);

  for(unsigned int i=0;i<v.size();i++){
    x[i+v.size()] = v[i];
  }
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
  
  FILE* handle1 = fopen("sort_test.csv", "wt");
  FILE* handle2 = fopen("sort_train_input.csv", "wt");
  FILE* handle3 = fopen("sort_train_output.csv", "wt");

  printf("Generating files (%d data points)..\n", NUMDATA);
  printf("(sort_test.csv, sort_train_input.csv, sort_train_output.csv)\n");
  

  for(unsigned int i=0;i<NUMDATA;i++){
    std::vector<float> example;
    example.resize(2*dimension);
    generate(example);

    for(unsigned int j=0;j<example.size();j++){
      fprintf(handle1, "%f ", example[j]);
    }
    fprintf(handle1, "\n");

    example.resize(2*dimension);
    generate(example);

    for(unsigned int j=0;j<example.size();j++){
      if(j < (example.size()/2))
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

