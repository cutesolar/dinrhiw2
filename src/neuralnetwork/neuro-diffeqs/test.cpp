
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
      
      {
	whiteice::nnetwork<> diffeq;
	
	if(create_random_diffeq_model(diffeq, 2) == false) throw 1;

	// random starting point
	u[0] = rng.uniformf()*2.0f - 1.0f; // x
	u[1] = rng.uniformf()*2.0f - 1.0f; // y

	// simulate training data
	if(simulate_diffeq_model(diffeq, u, 20.0, tdata, ttimes) == false) // 20 seconds long simulation
	  throw 2;
	
	printf("TRAINING simulation data points: %d\n", (int)tdata.size());

#if 0
	for(unsigned int i=0;i<tdata.size();i++)
	  std::cout << "tdata: " << tdata[i] << std::endl;
#endif
      }
      
      // drawing
      // black background
      SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
      SDL_RenderFillRect(renderer, NULL);
      SDL_RenderClear(renderer);

#if 0
      // draws random hermite splines
      for(unsigned int i=0;i<2;i++)
      {
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);

	std::vector< whiteice::math::vertex<> > data;
	whiteice::math::vertex<> u, v;
	u.resize(2);
	v.resize(2);
	
	whiteice::math::hermite< whiteice::math::vertex<>, whiteice::math::blas_real<float> > spline;

	for(unsigned int i=0;i<10;i++){
	  double y0 = rng.uniformf()*H;
	  double x0 = rng.uniformf()*W;

	  v[0] = x0;
	  v[1] = y0;

	  data.push_back(v);
	}

	spline.calculate(data, 500);

	for(unsigned int i=1;i<spline.size();i++){
	  u = spline[i-1];
	  v = spline[i];

	  SDL_RenderDrawLine(renderer, (int)u[0].c[0], (int)u[1].c[0], (int)v[0].c[0], (int)v[1].c[0]);
	}
      }
#endif

      // draws 1000 trajectories from random differential equation neural network model
      try{
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);

	whiteice::nnetwork<> diffeq;

	if(create_random_diffeq_model(diffeq, 2) == false) throw 1;

	// random starting point
	u[0] = rng.uniformf()*2.0f - 1.0f; // x
	u[1] = rng.uniformf()*2.0f - 1.0f; // y
	
	if(fit_diffeq_to_data_hmc(diffeq, tdata, ttimes, u, 100) == false) throw 2;

	std::vector< std::vector< whiteice::math::vertex<> > > datas;
	datas.resize(1000);

	std::vector< whiteice::math::blas_real<float> > times;

	for(unsigned int i=0;i<datas.size();i++){
	  
	  // random starting point
	  u[0] = rng.uniformf()*2.0f - 1.0f; // x
	  u[1] = rng.uniformf()*2.0f - 1.0f; // y

	  if(simulate_diffeq_model(diffeq, u, 5.0, datas[i], times) == false) // 5 seconds long simulation
	    throw 3;

	  printf("Simulation data points: %d\n", (int)datas[i].size());
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
	    u[0] *= ((float)(W/4))/stdev[0];
	    u[1] *= ((float)(H/4))/stdev[1];

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
