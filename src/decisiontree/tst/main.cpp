
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <iostream>

#include "decisiontree.h"


int main(void)
{
  srand(time(0));
  
  whiteice::DecisionTree dt;

  printf("DECISION TREE TESTING CODE\n");

  std::vector< std::vector<bool> > input;
  std::vector< std::vector<bool> > output;

  input.resize(100);
  output.resize(100);

  for(unsigned int i=0;i<input.size();i++){
    input[i].resize(20);
    output[i].resize(10);

    for(unsigned int k=0;k<input[i].size();k++){
      input[i][k] = (bool)(rand()&1);
    }

    for(unsigned int k=0;k<output[i].size();k++){
      output[i][k] = (bool)(rand()&1);
    }
      
  }

  if(dt.startTrain(input, output) == false){
    printf("ERROR: CANNOT START TRAINING\n");
    return -1;
  }

  while(dt.isRunning()){
    printf("Running decision tree algorithm..\n");
    sleep(1);
  }

  dt.stopTrain();

  printf("Saving decision tree..\n"); 

  if(dt.save("dt.dat") == false){
    printf("ERROR: CANNOT SAVE DECISION TREE\n");
    return -1;
  }

  printf("Loading decision tree..\n"); 

  if(dt.load("dt.dat") == false){
    printf("ERROR: CANNOT LOAD DECISION TREE\n");
    return -1;
  }

  unsigned int outcome = dt.classify(input[0]);

  printf("OUTCOME IS %d. FOR FIRST INPUT DATA ELEMENT.\n", outcome);

  printf("ALL TESTS DONE.\n");
  

  return 0;
}
