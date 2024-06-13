

#include "CartPole2.h"
#include "Log.h"

#include <stdio.h>
#include <math.h>

#include <chrono>
#include <functional>

#include <unistd.h>

#ifdef USE_SDL
#include <SDL.h>
#endif

using namespace std::chrono_literals;


namespace whiteice
{

  template <typename T>
  CartPole2<T>::CartPole2() : RIFL_abstract2<T>(1, 6)
  {
    {
      g = T(9.81); // gravity
      l = T(1.00);  // 1 meter long pole [was 0.10]
      
      mc = T(1.000); // cart weight (2.0 kg)
      mp = T(0.200); // pole weight (200g) [was 1.0]
      
      up = T(0.01);  // friction forces [pole]
      uc = T(0.1);   // friction between track and a cart
      
      Nc = T(0.0);
      
      reset();
      resetLastStep = false;
      
      // simulatiom timestep
      dt = T(0.020); // 20ms
      
      iteration = 0;
    }

    
#ifdef USE_SDL
    window = NULL;
    renderer = NULL;
    no_init_sdl = false;
    
    W = 800;
    H = 600;
#endif
    


    // starts physics thread
    {
      std::lock_guard<std::mutex> lock(physics_mutex);
      
      running = true;
      physics_thread = new std::thread(std::bind(&CartPole2<T>::physicsLoop, this));
    }
    
  }


  template <typename T>
  CartPole2<T>::~CartPole2()
  {
#ifdef USE_SDL
    {
      std::lock_guard<std::mutex> lock(sdl_mutex);

      no_init_sdl = true;
      
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
#endif
    
    // stops physics thread
    {
      std::lock_guard<std::mutex> lock(physics_mutex);

      running = false;
      physics_thread->join();
      delete physics_thread;
      physics_thread = nullptr;
    }
  }


  template <typename T>
  void CartPole2<T>::reset()
  {
    // cart-pole system system state
    //theta = this->rng.uniform()*T(2.0*M_PI) - T(M_PI); // angle is [-PI, PI]
    //theta = theta / T(100.0);

    theta = T(0.0);
    theta_dot = T(0.0);
    theta_dotdot = T(0.0);
    
    x = T(0.0);
    x_dot = T(0.0);
    x_dotdot = T(0.0);

    // external force
    F = T(0.0);
    F_processed = false;

    // reset after this time step
    resetLastStep = true;
  }


  template <typename T>
  bool CartPole2<T>::getState(whiteice::math::vertex<T>& state)
  {
    // whiteice::logging.info("CartPole2: entering getState()");
    
    state.resize(this->numStates);

    state[0] = atan2(sin(theta).c[0],cos(theta).c[0]); // theta
    state[1] = sin(theta);
    state[2] = cos(theta);
    state[3] = theta_dot;
    state[4] = x;
    state[5] = x_dot;

#ifdef USE_SDL
    {
      std::lock_guard<std::mutex> lock(sdl_mutex);
      
      if(renderer == NULL && no_init_sdl == false){
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
	    H = H/2;
	  }
	  
	  
	  SDL_CreateWindowAndRenderer(W, H, 0, &window, &renderer);
	}
      }
      
      
      if(renderer != NULL){
	auto theta = this->theta;	
	auto x     = state[4]; // WAS: auto x     = state[2];
	
	SDL_Event event;
	while(SDL_PollEvent(&event)){
	  if(event.type == SDL_QUIT){
	    this->running = false;
	  }

	  if(event.type == SDL_KEYDOWN){
	    if(event.key.keysym.sym == SDLK_ESCAPE){
	      this->running = false;
	    }
	    else if(event.key.keysym.sym == SDLK_PLUS){
	      T e = this->getEpsilon();
	      e += T(0.1);
	      if(e > T(1.0)) e = T(1.0);
	      std::cout << "Follow model percentage: " << e << std::endl;
	      this->setEpsilon(e);
	    }
	    else if(event.key.keysym.sym == SDLK_MINUS){
	      T e = this->getEpsilon();
	      e -= T(0.1);
	      if(e < T(0.0)) e = T(0.0);
	      std::cout << "Follow model percentage: " << e << std::endl;
	      this->setEpsilon(e);
	    }
	    else if(event.key.keysym.sym == SDLK_v){
	      log_verbose = !log_verbose;
	      whiteice::logging.setPrintOutput(log_verbose);
	      if(log_verbose)
		whiteice::logging.setOutputFile("cartpole2.log");
	    }
	  }
	}
	  
	// drawing
	{
	  // black background
	  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
	  // SDL_RenderFillRect(renderer, NULL);
	  SDL_RenderClear(renderer);
	  
	  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
	  
	  double y0 = H/2.0;
	  double x0 = W/2.0 + x.c[0]/2;
	  
	  double l = 100.0; // line length
	  
	  double y1 = y0 - l*cos(theta.c[0]);
	  double x1 = x0 + l*sin(theta.c[0]);
	  
	  SDL_RenderDrawLine(renderer, (int)x0, (int)y0, (int)x1, (int)y1);
	  SDL_RenderPresent(renderer);
	}
      }
      
    }
#endif

    // whiteice::logging.info("CartPole2: exiting getState()");

    return true;
  }


  template <typename T>
  T CartPole2<T>::normalizeTheta(const T t) const 
  {
    // sets range between [-2*pi, +2*pi]
    T a = t/T(2.0*M_PI);

    // printf("N: %f\n", a.c[0]);

    // take the fractional part 2.3 -> 0.3, -2.3 -> -0.3
    if(a >= T(0.0)){
      a = a - floor(a);
    }
    else{
      a = -(abs(a) - floor(abs(a)));
    }

    // printf("A: %f\n", a.c[0]); // debugging..

    a = T(2.0*M_PI)*a; // [-2*pi, 2*pi]

    // converts range between [-pi, pi]
    if(a > T(M_PI)){
      a = T(-1.0)*(T(2.0*M_PI) - a);
    }
    else if(a < T(-M_PI)){
      a = T(2.0*M_PI) + a;
    }

    // auto temp = a / T(2.0*M_PI);
    // printf("T: %f\n", temp.c[0]);

    return a;
  }


  template <typename T>
  bool CartPole2<T>::performAction(const whiteice::math::vertex<T>& action,
				   whiteice::math::vertex<T>& newstate,
				   T& reinforcement, bool& endFlag)
  {
    // whiteice::logging.info("CartPole2: entering performAction()");
    
    // converts action to control in newtons
    double Fstep = 0.0;

    endFlag = false;

#if 1
    assert(action.size() > 0);
    
    {
      // action values are [0,0.2] from model. [really now: [0.0,1.0]
      
      const T value = action[0]*T(2.0) - T(1.0); // converts action to [-1,+1] valued value 
      Fstep = 100.0*value.c[0];
      
      // should cause action values to be larger [0,1] to give meaningful results
      // Fstep = 4.0*action[0].c[0]; 

      // keeps Fsteps within "SANE" values which is needed for numerical stability
      if(Fstep >= 100.0) Fstep = 100.0;
      else if(Fstep <= -100.0) Fstep = -100.0;

      // to test if Fstep has effect on results
      // Fstep = (T(100.0)*normalizeTheta(theta)/T(M_PI)).c[0]; // [-1,1]
    }
#endif


    {
      char buffer[128];
      snprintf(buffer, 128, "%d %d FORCE: %f", iteration, this->getHasModel(), Fstep);
      if(verbose)
	printf("%s\n", buffer);

      //whiteice::logging.info(buffer);
    }
    
    {
      T a0 = 0.0;
      
      {
	a0 = theta/T(2.0*M_PI);
	
	if(a0 > T(0.0))
	  a0 = a0 - floor(a0);
	else
	  a0 = a0 - (floor(a0)+T(1.0f));
	
	a0 *= T(360.0);
      }
      
      {
	std::unique_lock<std::mutex> lock(F_change);
	F = Fstep;
	F_processed = false;
	F_processed_cond.notify_all();

	while(F_processed == false){
	  auto now = std::chrono::system_clock::now();
	  F_processed_cond.wait_until(lock, now + 100ms);
	  if(running == false)
	    return false;
	}
	
	{
	  newstate.resize(this->numStates);
	  newstate[0] = atan2(sin(theta).c[0],cos(theta).c[0]); // theta
	  newstate[1] = sin(theta);
	  newstate[2] = cos(theta);
	  newstate[3] = theta_dot;
	  newstate[4] = x;
	  newstate[5] = x_dot;

	  if(resetLastStep){
	    endFlag = true;
	    resetLastStep = false;
	  }
	  else{
	    endFlag = false;
	  }
	}
	
	// our target is to keep theta at zero (the largest reinforcement value)
	{
	  // converts range between [-180.0, +180.0]

	  T a1 = theta/T(2.0*M_PI);

	  const T epsilon = T(0.1);
	  reinforcement = T(1.0)/(abs(a1)+epsilon);
	  if(reinforcement >= T(10.0f)) reinforcement = T(10.0f);
	  if(reinforcement <= T(0.0)) reinforcement = T(0.0f);
	  reinforcement /= T(10.0f);

	  reinforcement *= T(0.1f); // reinforcement is in the range of [0, 0.1]

	  //std::cout << "theta = " << theta << " reinforcement = " << reinforcement << std::endl;
	  
#if 0
	  T a1 = theta/T(2.0*M_PI);
		  
	  if(a1 > T(0.0))
	    a1 = a1 - floor(a1);
	  else
	    a1 = a1 - (floor(a1)+T(1.0f));

	  if(a1 > +0.5) a1 = a1-1.0;
	  if(a1 < -0.5) a1 = a1+1.0;

	  a1 *= T(360.0);
	  // a1 *= T(2.0*M_PI);

	  //std::cout << "x = " << x
	  //	    << " theta = " << T(360.0)*theta/T(2.0*M_PI)
	  //	    << " theta_dot = " << T(360.0)*theta_dot/T(2.0*M_PI) << std::endl;
#endif
	  

#if 0
	  //if(a1 > T(+12.0) || a1 < T(-12.0)) reinforcement = 0.0;
	  //else reinforcement = T(3.00);

	  reinforcement = T(1.0)*((T(180.0) - abs(a1))/T(180.0));
	  reinforcement = pow(reinforcement, T(10.0));
	  
	  if(x > T(+400.0f) || x < T(-400.0f))
	    reinforcement = T(0.0f);

	  reinforcement = reinforcement - pow(abs(T(Fstep))/T(100.0f), T(2.0f));

	  if(abs(a1) > T(90.0f)) reinforcement = reinforcement - T(0.5f);
#endif
	  // reinforcement = T(-1.0)*(T(0.1)*pow(a1, T(2.0)) /*+ T(5.0)*pow(T(Fstep), T(2.0))*/);

	  // if(abs(a1) < abs(a0)) reinforcement++;


	  {
	    char buffer[128];
	    snprintf(buffer, 128, "REINFORCEMENT: %f", reinforcement.c[0]);
	    if(verbose)
	      printf("%s\n", buffer);
	    
	    //whiteice::logging.info(buffer);
	  }
	}
      }
	
    }

    return true;
  }

  
  template <typename T>
  T CartPole2<T>::sign(T value)
  {
    if(value >= T(0.0)) return T(+1.0);
    else return T(-1.0);
  }

  
  template <typename T>
  void CartPole2<T>::physicsLoop()
  {    
    double t = 0.0;

    std::vector<T> thetas;
    T mth = T(0.0);
    T sth = T(0.0);

    resetLastStep = false;
    
    
    while(running){

      double degrees = 0.0;
      
      {
	std::unique_lock<std::mutex> lock(F_change);

	while(F_processed == true){
	  auto now = std::chrono::system_clock::now();
	  F_processed_cond.wait_until(lock, now + 100ms);
	  if(running == false) return;
	}

	theta_dotdot =
	  g*sin(theta) +
	  cos(theta) * ( (-F - mp*l*theta_dot*theta_dot*
			  (sin(theta) + uc*sign(Nc*x_dot)*cos(theta)))/(mc + mp) +
			 uc*g*sign(Nc*x_dot)) - up*theta_dot/(mp*l);
	theta_dotdot /= l*(T(4.0/3.0) -
			   (mp*cos(theta)/(mc+mp))*(cos(theta) - uc*sign(Nc*x_dot)));
	
	auto oldNc = Nc;
	
	Nc = (mc + mp)*g -
	  mp*l*(theta_dotdot*sin(theta) + theta_dot*theta_dot*cos(theta));
	
	if(sign(oldNc) != sign(Nc)){
	  theta_dotdot =
	    g*sin(theta) +
	    cos(theta) * ( (-F - mp*l*theta_dot*theta_dot*
			    (sin(theta) + uc*sign(Nc*x_dot)*cos(theta)))/(mc + mp) +
			   uc*g*sign(Nc*x_dot)) - up*theta_dot/(mp*l);
	  theta_dotdot /= l*(T(4.0/3.0) -
			     (mp*cos(theta)/(mc+mp))*(cos(theta) - uc*sign(Nc*x_dot)));
	}
	
	x_dotdot = F + mp*l*(theta_dot*theta_dot*sin(theta) -
			     theta_dotdot*cos(theta)) - uc*Nc*sign(Nc*x_dot);
	x_dotdot /= mc + mp;

	////////////////////////////////////////////////////////////////////////
	// we stimulate system for a single timestep
	
	x = x + x_dot*dt + T(0.5)*x_dotdot*dt*dt;
	theta = theta + theta_dot*dt + T(0.5)*theta_dotdot*dt*dt;
	
	x_dot = x_dot + x_dotdot*dt;
	theta_dot = theta_dot + theta_dotdot*dt;

	t += dt.c[0];
	
	// prints theta in degrees
	{
	  // keeps range between [-2*pi, +2*pi]
	  T a = theta/T(2.0*M_PI);
	  a = a - floor(a);
	  a = T(2.0*M_PI)*a;

	  if(a > T(M_PI)){
	    a = T(-1.0)*(T(2.0*M_PI) - a);
	  }

	  degrees = 360.0*(a.c[0]/(2.0*M_PI));

	  thetas.push_back(abs(degrees));

	  auto temp = abs(mth); //  + sth; [DO NOT ADD STANDARD DEVIATION]
	  auto eps = this->getEpsilon();
	  eps = T(100.0f)*eps;

	  {
	    char buffer[128];
	    snprintf(buffer, 128, "TIME %f theta = %f deg [%f] [%.1f%% FOLLOW MODEL] [%d MODEL]",
		     t, degrees, temp.c[0], eps.c[0], this->getHasModel());
	    if(verbose){
	      printf("%s\n", buffer);
	    }

	    //whiteice::logging.info(buffer);
	  }
	  
	  fflush(stdout);
	}
	
	F_processed = true;
	F_processed_cond.notify_all();
      }

      if(t >= 20.0 || abs(x) > T(1000.0f)){
	iteration++;
	t = 0.0;

	mth = T(0.0);
	sth = T(0.0);

	for(auto& d : thetas){
	  mth += d;
	  sth += d*d;
	}

	mth /= T((double)thetas.size());
	sth /= T((double)thetas.size());

	sth = sth - mth*mth;
	sth = sqrt(abs(sth));
	
	thetas.clear();
	
	reset(); // reset parameters of cart-pole
      }
#if 0
      else if(degrees > 90 || degrees < -90){
	reset(); // reset parameters of cart-pole if we get worse than horizontal
      }
#endif

      // 1x faster than time
      // usleep((unsigned int)(dt.c[0]*1000000.0));  // waits for a single timestep
      
    }
    
    
  }
  


  template class CartPole2< math::blas_real<float> >;
  template class CartPole2< math::blas_real<double> >;
};
