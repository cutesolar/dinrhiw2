
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#ifndef WINOS
#include <fenv.h>
#endif

#include "MinihackRIFL.h"
#include "RIFL_abstract.h"
#include "Log.h"



int main(int argc, char** argv)
{
  printf("MINIHACK REINFORCEMENT LEARNING. (DISCRETE ACTIONS)\n");
  fflush(stdout);

  const std::string scriptFile = "minihack_env.py";

  printf("Using python script: %s\n", scriptFile.c_str());
  fflush(stdout);

  srand(time(0));

  whiteice::logging.setOutputFile("debug.log");

#ifndef WINOS
#if 1
  // enable floating point exceptions (for debugging)
  {
    // FE_UNDERFLOW | FE_OVERFLOW | FE_INEXACT
    feenableexcept(FE_DIVBYZERO | FE_INVALID);
  }
#endif
#endif
  
  bool useFlag = false;
  bool loadFlag = false;

  for(int i=1;i<argc;i++){
    if(strcmp(argv[i], "--use") == 0) useFlag = true;
    else if(strcmp(argv[i], "--load") == 0) loadFlag = true;
    else{
      printf("ERROR: Unknown command-line option.\n");
      return -1;
    }
  }
  
  
  if(useFlag == false){
    whiteice::MinihackRIFL< whiteice::math::blas_real<float> > system(scriptFile);

    system.setEpsilon(0.95); // 5% of control choices are random
    system.setLearningMode(true);

    if(loadFlag){
      printf("Loading existing model from disk..\n");
      if(system.load("minihack-rifl2.dat") == false){
	printf("ERROR: Loading model from disk FAILED.");
	return -1;
      }

      system.setHasModel(1);
    }
    
    if(system.start() == false){
      printf("ERROR: Starting reinforcement learning FAILED.\n");
      return -1;
    }

    sleep(1);

    unsigned int counter = 1;

    while(system.isRunning()){
      
      if(system.getHasModel() >= 2){
	// 95% are selected according to model
	// system.setEpsilon(0.95);
      }
      
      sleep(1); 
      if((counter % 180) == 0){ // saved model file every 3 minutes
	if(system.save("minihack2-rifl.dat"))
	  printf("MODEL FILE SAVED\n");
      }
      
      counter++;
    }

    if(system.save("minihack2-rifl.dat"))
      printf("MODEL FILE SAVED\n");

    system.stop();
    
  }
  else{

    whiteice::MinihackRIFL< whiteice::math::blas_real<float> > system(scriptFile);

    system.setEpsilon(1.00); // 100% of examples are selected accoring to model
    system.setLearningMode(false);    
    
    if(system.load("minihack2-rifl.dat") == false){
      printf("ERROR: loading model file failed.\n");
      return -1;
    }
    else{
      std::cout << "Reinforcement model SUCCESSFULLY loaded." << std::endl;
      system.setHasModel(1);
    }
    
    system.start();

    while(system.isRunning())
      sleep(1);
    
    system.stop();
  }
  

  return 0;
}
