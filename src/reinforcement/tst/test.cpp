
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
  printf("REINFORCEMENT LEARNING TESTCASES 1 (DISCRETE ACTIONS)\n");
  fflush(stdout);

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

  
  if(argc <= 1){
    whiteice::CartPole< whiteice::math::blas_real<double> > system;

    system.setEpsilon(0.50); // 50% of examples are selected according to model
    system.setLearningMode(true);
    
    // system.load("rifl.dat");
    
    system.start();

    sleep(1);

    unsigned int counter = 1;

    while(system.physicsIsRunning()){

      // 20-25 degrees average theta happens with random control
      // if you get smaller than that then you have learnt something
      
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
  else if(strcmp(argv[1], "use") == 0){

    whiteice::CartPole< whiteice::math::blas_real<double> > system;

    system.setEpsilon(1.00); // 100% of examples are selected according to model
    system.setLearningMode(false);
    system.setHasModel(1);
    
    if(system.load("rifl.dat") == false){
      printf("ERROR: loading model file failed.\n");
      return -1;
    }
    
    system.start();

    while(system.physicsIsRunning())
      sleep(1);
    
    system.stop();
  }
  

  return 0;
}
