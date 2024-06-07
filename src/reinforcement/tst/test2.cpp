
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#include "CartPole.h"
#include "CartPole2.h"
#include "RIFL_abstract.h"
#include "Log.h"

#ifndef USE_SDL
#include <fenv.h>
#endif



int main(int argc, char** argv)
{
  printf("REINFORCEMENT (CONTINUOS) LEARNING TESTCASES 2. (CartPole problem)\n");
  fflush(stdout);

  use_gpu_sync = false; 

  srand(time(0));

  whiteice::logging.setOutputFile("debug.log");

#ifndef WINOS
#ifndef USE_SDL
  // enable floating point exceptions (for debugging)
  {
    // FE_UNDERFLOW | FE_OVERFLOW | FE_INEXACT
    feenableexcept(FE_DIVBYZERO | FE_INVALID);
  }
#endif
#endif

#ifndef USE_SDL
  whiteice::logging.setOutputFile("cartpole2.log");
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
    whiteice::CartPole2< whiteice::math::blas_real<double> > system;

    system.setEpsilon(0.50); // 50% of control choices are random
    system.setLearningMode(true);
    system.setVerbose(false);

    if(loadFlag){
      system.setEpsilon(0.75); // 25% of control choices are random
      printf("Loading existing model from disk..\n");
      if(system.load("rifl.dat") == false){
	printf("ERROR: Loading model from disk FAILED.");
	return -1;
      }

      system.setHasModel(1);
    }
    
    system.start();

    sleep(1);

    unsigned int counter = 1;

    while(system.physicsIsRunning()){
      
      if(system.getHasModel() >= 2){
	// 95% are selected according to model
	// system.setEpsilon(0.95);
      }
      
      sleep(1); 
      if((counter % 180) == 0){ // saved model file every 3 minutes
	if(system.save("rifl.dat"))
	  printf("MODEL FILE SAVED\n");
      }
      
      counter++;
    }

    if(system.save("rifl.dat"))
      printf("MODEL FILE SAVED\n");

    system.stop();
    
  }
  else{

    whiteice::CartPole2< whiteice::math::blas_real<double> > system;

    system.setEpsilon(1.00); // 100% of examples are selected accoring to model
    system.setLearningMode(false);
    system.setHasModel(1);
    system.setVerbose(true);
    
    if(system.load("rifl.dat") == false){
      printf("ERROR: loading model file failed.\n");
      return -1;
    }
    else{
      std::cout << "Reinforcement model SUCCESSFULLY loaded." << std::endl;
    }
    
    system.start();

    while(system.physicsIsRunning())
      sleep(1);
    
    system.stop();
  }
  

  return 0;
}
