
#include "diffeqs.h"

#include <dinrhiw/dinrhiw.h>

#ifdef USE_SDL
#include <SDL.h>
#include <mutex>
#endif

int main(void)
{
  printf("Differential equation solver. Copyright Tomas Ukkonen.\n");
  printf("dx/dt = neuralnetwork(x|w) for differential equation model.\n");

  whiteice::RNG<> rng;

  srand(rng.rand64());

  // logging.setPrintOutput(true);
  
  // plots random lines to graphical window (1/2 of the whole screen size)
  
  int W, H;
  
  SDL_Window* window;
  SDL_Renderer* renderer;
  
  std::mutex sdl_mutex; // for sync access to SDL
  
  bool log_verbose = false; // used to control if we print internal log comments.

  bool running = true;
  
  window = NULL;
  renderer = NULL;
  
  W = 800;
  H = 600;
  

  while(running)
  {
    std::lock_guard<std::mutex> lock(sdl_mutex);
    
    if(renderer == NULL){
      // create SDL display
      
      if(renderer == NULL){
	SDL_Init(SDL_INIT_VIDEO);
	
	window = NULL;
	renderer = NULL;
	
	W = 800;
	H = 600;
	
	SDL_DisplayMode mode;
	
	if(SDL_GetCurrentDisplayMode(0, &mode) == 0){
	  W = (4*mode.w)/5;
	  H = (3*mode.h)/4;
	}
	
	
	SDL_CreateWindowAndRenderer(W, H, 0, &window, &renderer);

	SDL_SetWindowTitle(window, "Runge-Kutta neural network differential eq. solver");
      }
    }
    
    
    if(renderer != NULL){
      
      SDL_Event event;
      while(SDL_PollEvent(&event)){
	if(event.type == SDL_QUIT){
	  running = false;
	}
	
	if(event.type == SDL_KEYDOWN){
	  if(event.key.keysym.sym == SDLK_ESCAPE){
	    running = false;
	  }
	  else if(event.key.keysym.sym == SDLK_PLUS){
	  }
	  else if(event.key.keysym.sym == SDLK_MINUS){
	  }
	  else if(event.key.keysym.sym == SDLK_v){
	    log_verbose = !log_verbose;
	    whiteice::logging.setPrintOutput(log_verbose);
	    if(log_verbose)
	      whiteice::logging.setOutputFile("cartpole2.log");
	  }
	}
      }

      // generate training data (random model)
      std::vector< whiteice::math::vertex<> > tdata;
      std::vector< whiteice::math::blas_real<float> > ttimes;

      whiteice::math::vertex<> u, v;
      u.resize(2);
      v.resize(2);

      // generates data
      {
	whiteice::nnetwork<> diffeq;
	
	if(create_random_diffeq_model(diffeq, 2) == false) throw 1;

	// random starting point
	u[0] = rng.uniformf()*2.0f - 1.0f; // x
	u[1] = rng.uniformf()*2.0f - 1.0f; // y

#if 0
	// simulate training data
	if(simulate_diffeq_model(diffeq, u, 20.0, tdata, ttimes) == false) // 20 seconds long simulation
	  throw 2;
#endif

	// generates sin() function instead
	tdata.clear();
	ttimes.clear();
	

	const double A = 1.0;
	const double freq = 0.1*2.0*M_PI;

	for(double t = 0.0;t<10.0;t+=0.05){

	  v[0] = t/5.0;
	  v[1] = t/10.0; // A*sin(freq*t);

	  tdata.push_back(v);
	  ttimes.push_back(t);
	}
	
	printf("TRAINING simulation data points: %d\n", (int)tdata.size());
      }
      
      // drawing
      // black background
      SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
      SDL_RenderFillRect(renderer, NULL);
      SDL_RenderClear(renderer);

      // draws 1000 trajectories from random differential equation neural network model
      try{
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);

	whiteice::nnetwork<> diffeq;

	if(create_random_diffeq_model(diffeq, 2) == false) throw 1;

	// random starting point (zero same as in data)
	u[0] = 0.0f; // rng.uniformf()*2.0f - 1.0f; // x
	u[1] = 0.0f; // rng.uniformf()*2.0f - 1.0f; // y

	const unsigned int NUM_SAMPLES = 80000; // 20.000 makes error go to 3/4, need 80.000 samples

	// FIXME: disabled for now!!
	if(fit_diffeq_to_data_hmc2(diffeq, tdata, ttimes, NUM_SAMPLES) == false) throw 2;

	std::vector< std::vector< whiteice::math::vertex<> > > datas;
	datas.resize(200);

	std::vector< whiteice::math::blas_real<float> > times;

	for(unsigned int i=0;i<datas.size();i++){
	  
	  // random starting point (almost zero now, same as in training data)
	  // TODO: add multiple starting positions in training data
	  u[0] = 0.01f*(rng.uniformf()*2.0f - 1.0f); // x
	  u[1] = 0.01f*(rng.uniformf()*2.0f - 1.0f); // y

	  if(simulate_diffeq_model(diffeq, u, 5.0, datas[i], times) == false) // 5 seconds long simulation
	    throw 3;

	  // printf("Simulation data points: %d\n", (int)datas[i].size());

	  // uses/shows training data: 
	  // datas[i] = tdata;
	  // times = ttimes;
	}

	// origo
	v[0] = W/2.0;
	v[1] = H/2.0;
	
	whiteice::math::vertex<> mean, var, stdev; // to zero mean, unit variance
	
	mean = v;
	var  = v;
	stdev = v;
	mean.zero();
	var.zero();
	stdev.zero();

	unsigned int N = 0;
	
	for(unsigned int i=0;i<datas.size();i++){
	  
	  for(unsigned int j=0;j<datas[i].size();j++){
	    u = datas[i][j];

	    // std::cout << data[i] << std::endl;

	    if(u[0] < -10.0f) u[0] = -10.0f;
	    if(u[0] > +10.0f) u[0] = +10.0f;

	    if(u[1] < -10.0f) u[1] = -10.0f;
	    if(u[1] > +10.0f) u[1] = +10.0f;

	    mean += u;
	    var[0] += u[0]*u[0];
	    var[1] += u[1]*u[1];

	    datas[i][j] = u;
	  }

	  N += datas[i].size();
	}

	// calculates stdev of data
	mean /= (float)N;
	var /= (float)N;
	
	var[0] -= mean[0]*mean[0];
	var[1] -= mean[1]*mean[1];
	
	stdev[0] = whiteice::math::sqrt(whiteice::math::abs(var[0])) + 1e-6f;
	stdev[1] = whiteice::math::sqrt(whiteice::math::abs(var[1])) + 1e-6f;
	
	// scales data to be N(origo, (W/4,H/4)^2)
	// data = ((data-mean)/stdev)*W/4 + origo

	for(unsigned int i=0;i<datas.size();i++){
	  for(unsigned int j=0;j<datas[i].size();j++){
	    u = datas[i][j];
	    
	    // scaling
	    u -= mean;
	    u[0] *= ((float)(W/4))/(3.0f*stdev[0]);
	    u[1] *= ((float)(H/4))/(3.0f*stdev[1]);

	    // origo
	    u += v;
	    
	    datas[i][j] = u;
	  }
	}

	for(unsigned int i=0;i<datas.size();i++){
	  
	  whiteice::math::hermite
	    < whiteice::math::vertex<>, whiteice::math::blas_real<float> > spline;
	  
	  spline.calculate(datas[i], (int)(4*datas[i].size()));
	  
	  for(unsigned int j=1;j<spline.size();j++){
	    u = spline[j-1];
	    v = spline[j];
	    
	    SDL_RenderDrawLine(renderer,
			       (int)u[0].c[0], (int)u[1].c[0],
			       (int)v[0].c[0], (int)v[1].c[0]);
	  }
	  
	}

      }
      catch(int error){
	printf("Error in differential equation code: %d\n", error);
	return -1;
      }
      
      SDL_RenderPresent(renderer);
    }
    
  }
  
  
  {
    std::lock_guard<std::mutex> lock(sdl_mutex);
    
    if(renderer){
      auto temp = renderer;
      renderer = NULL;
      SDL_DestroyRenderer(temp);
    }
    
    if(window){
      auto temp = window;
      window = NULL;
      SDL_DestroyWindow(temp);
    }
    
    SDL_Quit();
  }

  
  return 0;
}
