
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <iostream>

#include "decisiontree.h"


int main(void)
{
  srand(time(0));

  // TESTCASE 1
  {
    whiteice::DecisionTree dt;
    
    printf("DECISION TREE TESTING CODE 1\n");
    
    std::vector< std::vector<bool> > input;
    std::vector< std::vector<bool> > output;

    input.resize(50);
    output.resize(50);


    for(unsigned int i=0;i<input.size();i++){
      input[i].resize(10);
      output[i].resize(2);
      
      for(unsigned int k=0;k<input[i].size();k++){
	input[i][k] = (bool)(rand()&1);
      }
      
      for(unsigned int k=0;k<output[i].size();k++){
	if(k == 0){
	  if(input[i][0]){
	    if((rand()%10) != 0) // 90% cases are true if label is true
	      output[i][k] = true;
	    else
	      output[i][k] = false;
	  }
	  else{
	    if((rand()%10) != 0) // 90% cases are false if label is false
	      output[i][k] = false;
	    else
	      output[i][k] = true;
	  }
	}
	else{
	  output[i][k] = !(input[i][0]); 
	}

	
      }
      
      std::cout << input[i][0] << " => " << output[i][0] << std::endl;
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
    
    {
      std::vector<int> outcomes;
      
      unsigned int correct=0, wrong=0;
      
      for(unsigned int i=0;i<input.size();i++){
	
	const int outcome = dt.classify(input[i]);

	if(outcome < 0){
	  printf("wrong outcome!\n");
	}
	else{
	
	  if(output[i][outcome]){
	    correct++;
	  }
	  else{
	    wrong++;
	  }
	  
	}
	  
	outcomes.push_back(outcome);
      }
      
      printf("OUTCOME IS %d. FOR FIRST INPUT DATA ELEMENT.\n", outcomes[0]);
      
      printf("PERCENT CLASSIFICATIONS CORRECT: %f\n", (float)correct/((float)(correct+wrong)));
    }
    
  }

  return 0;

  // TESTCASE 2
  {
    whiteice::DecisionTree dt;
    
    printf("DECISION TREE TESTING CODE 2\n");
    
    std::vector< std::vector<bool> > input;
    std::vector< std::vector<bool> > output;
    
    input.resize(100);
    output.resize(100);
    
    for(unsigned int i=0;i<input.size();i++){
      input[i].resize(20);
      output[i].resize(2);
      
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
    
    {
      std::vector<int> outcomes;
      
      unsigned int correct=0, wrong=0;
      
      for(unsigned int i=0;i<input.size();i++){
	
	const int outcome = dt.classify(input[i]);
	
	if(output[i][outcome]){
	  correct++;
	}
	else{
	  wrong++;
	}
	
	outcomes.push_back(outcome);
      }
      
      printf("OUTCOME IS %d. FOR FIRST INPUT DATA ELEMENT.\n", outcomes[0]);
      
      printf("PERCENT CLASSIFICATIONS CORRECT: %f\n", (float)correct/((float)(correct+wrong)));
    }
    
    
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
    
    
    {
      std::vector<int> outcomes;
      
      unsigned int correct=0, wrong=0;
      
      for(unsigned int i=0;i<input.size();i++){
	
	const int outcome = dt.classify(input[i]);
	
	if(output[i][outcome]){
	  correct++;
	}
	else{
	  wrong++;
	}
	
	outcomes.push_back(outcome);
      }
      
      printf("OUTCOME IS %d. FOR FIRST INPUT DATA ELEMENT.\n", outcomes[0]);
      
      printf("PERCENT CLASSIFICATIONS CORRECT: %f\n", (float)correct/((float)(correct+wrong)));
    }

  }

  printf("ALL TESTS DONE.\n");
  

  return 0;
}
