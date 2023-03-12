
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <iostream>

#include "decisiontree.h"
#include "discretization.h"


#include "RNG.h"


int main(void)
{
  srand(time(0));

  // TESTCASE 0: continuous variables 
  {
    whiteice::DecisionTree dt;

    printf("DECISION TREE TESTING CODE 0: CONTINUOUS DATA\n");

    fflush(stdout);

    std::vector< whiteice::math::vertex<> > cinput, coutput;
    
    std::vector< whiteice::math::vertex<> > cinput2, coutput2;
    
    std::vector< std::vector<bool> > input;
    std::vector< std::vector<bool> > output;

    std::vector< std::vector<bool> > input2;
    std::vector< std::vector<bool> > output2;

    std::vector< whiteice::math::vertex<> > conversion;
    
    whiteice::math::vertex<> in, out;
    whiteice::math::matrix<> A;
    whiteice::math::vertex<> b;

    const unsigned int DIM = 2; // was: 1, 2, 4, 10
    // const unsigned int SAMPLES = whiteice::math::pow(2.0f,(float)DIM)*1000; // was: 2000
    const unsigned int SAMPLES = 2000;


    in.resize(DIM);
    out.resize(1);

    A.resize(1, DIM);
    b.resize(1);

    for(unsigned int y=0;y<A.ysize();y++)
      for(unsigned int x=0;x<A.xsize();x++)
	A(y,x) = whiteice::rng.normal();

    for(unsigned int i=0;i<b.size();i++)
      b[i]= whiteice::rng.normal();

    for(unsigned int i=0;i<SAMPLES;i++){
      whiteice::rng.normal(in);
      out = A*in + b;

      cinput.push_back(in);
      coutput.push_back(out);
    }

    std::cout << "INPUT  DATA SIZE: " << cinput.size() << std::endl;
    std::cout << "OUTPUT DATA SIZE: " << coutput.size() << std::endl;
    
    std::cout << "Discretization.. start." << std::endl;

    fflush(stdout);
    
    discretization(cinput, coutput,
		   input, output,
		   conversion);

    std::cout << "Discretization.. DONE." << std::endl;

    while(input.size() > SAMPLES/2){
      auto iter1 = input.end();
      auto iter2 = output.end();

      iter1--;
      iter2--;

      input2.push_back(*iter1);
      output2.push_back(*iter2);

      input.erase(iter1);
      output.erase(iter2);
    }

    while(cinput.size() > SAMPLES/2){
      auto iter1 = cinput.end();
      auto iter2 = coutput.end();

      iter1--;
      iter2--;

      cinput2.push_back(*iter1);
      coutput2.push_back(*iter2);

      cinput.erase(iter1);
      coutput.erase(iter2);
    }

    
    std::cout << "INPUT  DATA SIZE: " << input.size() << std::endl;
    std::cout << "OUTPUT DATA SIZE: " << output.size() << std::endl;

    std::cout << "INPUT  DIM: " << input[0].size() << std::endl;
    std::cout << "OUTPUT DIM: " << output[0].size() << std::endl;

    fflush(stdout);

    
    if(dt.startTrain(input, output) == false){
      printf("ERROR: CANNOT START TRAINING\n");
      return -1;
    }
    
    while(dt.isRunning()){
      printf("Running decision tree algorithm..\n");
      sleep(1);
    }
    
    dt.stopTrain();

    
    // print outcomes
    {
      std::vector<int> outcomes;

      whiteice::math::blas_real<float> error = 0.0f;
      
      unsigned int correct=0, wrong=0;
      
      for(unsigned int i=0;i<input2.size();i++){
	
	const int outcome = dt.classify(input2[i]);

	if(outcome < 0){
	  printf("wrong outcome!\n");
	}
	else{
	
	  if(output2[i][outcome]){
	    correct++;
	  }
	  else{
	    wrong++;
	  }

	  auto nrm = (conversion[outcome]-coutput2[i]).norm();
	  error += nrm*nrm;
	  
	}
	
	outcomes.push_back(outcome);
      }

      error /= outcomes.size();
      
      printf("OUTCOME IS %d. FOR FIRST INPUT DATA ELEMENT.\n", outcomes[0]);
      
      printf("PERCENT CLASSIFICATIONS CORRECT: %f\n", (float)correct/((float)(correct+wrong)));

      std::cout << "Classification MSE error: " << error << std::endl;
    }
    
  }


  return 0;
  

  // TESTCASE 1: hand-selected binary variables (90% prediction result with simple dataset)
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
