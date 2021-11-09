

#include "CartPole.h"

#include <stdio.h>
#include <math.h>

#include <functional>
#include <chrono>

#include <unistd.h>

#ifdef USE_SDL
#include <SDL.h>
#endif

using namespace std::chrono_literals;

namespace whiteice
{

  template <typename T>
  CartPole<T>::CartPole() : RIFL_abstract<T>(5, 6) // 2 possible actions
  {
    {
      g = T(9.81); // gravity
      l = T(0.10);  // 1 meter long pole
      
      mc = T(1.000); // cart weight (2.0 kg)
      mp = T(1.000); // pole weight (200g)
      
      up = T(0.01);  // friction forces [pole]
      uc = T(0.1);   // friction between track and a cart
      
      Nc = T(0.0);
      
      reset();
      resetLastStep = false;
      
      // simulatiom timestep
      dt = T(0.020); // was 10ms (0.010)
      
      iteration = 0;
    }
    
#ifdef USE_SDL
    // SDL variables
    window = NULL;
    renderer = NULL;
    W = 800;
    H = 600;
    no_init_sdl = false;
#endif
    
    // starts physics thread
    {
      std::lock_guard<std::mutex> lock(physics_mutex);
      
      running = true;
      physics_thread = new std::thread(std::bind(&CartPole<T>::physicsLoop, this));
    }
    
  }


  template <typename T>
  CartPole<T>::~CartPole()
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
    }
  }


  template <typename T>
  void CartPole<T>::reset()
  {
    // cart-pole system system state
    theta = T(rng.uniformf())*T(2.0*M_PI) - T(M_PI); // angle is [-PI, PI]
    theta = theta / T(100.0); 
    theta_dot = T(0.0);
    theta_dotdot = T(0.0);
    
    x = T(0.0);
    x_dot = T(0.0);
    x_dotdot = T(0.0);

    // external force
    F = T(0.0);
    F_processed = false;

    resetLastStep = true;
  }


  template <typename T>
  bool CartPole<T>::getState(whiteice::math::vertex<T>& state)
  {
    state.resize(6);

    state[0] = atan2(sin(theta).c[0],cos(theta).c[0]); // theta
    state[1] = sin(theta);
    state[2] = cos(theta);
    state[3] = theta_dot;
    state[4] = x;
    state[5] = x_dot;

#ifdef USE_SDL
    {
      std::lock_guard<std::mutex> lock(sdl_mutex);
      
      // create SDL display if not initialized already
      if(renderer == NULL && no_init_sdl == false){
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
#endif

#ifdef USE_SDL
      if(renderer != NULL){
	auto theta = this->theta;
	auto x     = state[2];
	
	SDL_Event event;
	while(SDL_PollEvent(&event)){
	  if(event.type == SDL_QUIT){
	    this->running = false;
	  }
	  
	  if(event.type == SDL_KEYDOWN){
	    if(event.key.keysym.sym == SDLK_ESCAPE){
	      this->running = false;
	    }
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

    return true;
  }


  template <typename T>
  T CartPole<T>::normalizeTheta(const T t) const 
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
  bool CartPole<T>::getActionFeature(const unsigned int action,
				     whiteice::math::vertex<T>& feature) const
  {
    feature.resize(1);
    feature.zero();

    if(action >= this->numActions) return false;

    // force value F

    double a = ((double)action)/((double)(this->numActions - 1)); // [0,1]
    a = 2.0*a - 1.0; // [-1.0,+1.0]
    double Fstep = 25.0*a; // [-25, +25]

    feature[0] = Fstep;

    return true;
  }

  
  template <typename T>
  bool CartPole<T>::performAction(const unsigned int action,
				  whiteice::math::vertex<T>& newstate,
				  T& reinforcement,
				  bool& endFlag)
  {

    auto old_theta = theta;
    
    // converts action to control in newtons
    // printf("%d action\n", action);
    double Fstep = 0.0;

    double a = ((double)action)/((double)(this->numActions - 1)); // [0,1]
    a = 2.0*a - 1.0; // [-1.0,+1.0]
    Fstep = 20.0*a; // [-20, +20]

    printf("%d FORCE: %f\n", iteration, Fstep);
    
    {
      {
	// converts range between [-180.0, +180.0]
	T a0 = theta / T(2.0*M_PI);
	a0 = a0 - floor(a0);
	a0 = T(1.0) - a0;
	
	// range is [0.0, 1.0]; // bigger is better (closer to one)
	
	std::cout << "A0 = " << a0 << std::endl;
	std::cout << "theta0 = " << theta << std::endl;
      }
      
      
      {
	std::unique_lock<std::mutex> lock(F_change);
	F = Fstep;
	F_processed = false;
	F_processed_cond.notify_all();

	while(F_processed == false){
	  auto now = std::chrono::system_clock::now();
	  F_processed_cond.wait_until(lock, now + 10*100ms);
	  if(running == false)
	    return false;
	}
	
	
	{
	  newstate.resize(this->numStates);
	  newstate[0] = atan2(sin(theta).c[0],cos(theta).c[0]);
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

	  auto old_angle = atan2(sin(old_theta).c[0],cos(old_theta).c[0]);

	  
	  std::cout << "angle_difference: " << newstate[0].c[0] - old_angle << std::endl;
	}
	
	// our target is to keep theta at zero (the largest reinforcement value)
	{
	  // converts range between [-180.0, +180.0]
	  T a1 = T(180.0)* normalizeTheta(theta) / T(M_PI);
	  // a1 = (T(180.0)-abs(a))/T(180.0);

	  a1 = theta/T(2.0*M_PI);
	  
	  if(a1 > T(0.0))
	    a1 = a1 - floor(a1);
	  else
	    a1 = a1 - (floor(a1)+T(1.0f));

	  if(a1 > +0.5) a1 = a1-1.0;
	  if(a1 < -0.5) a1 = a1+1.0;

	  // a1 *= T(360.0);
	  a1 *= T(2.0*M_PI);

	  //if(a1 > T(+12.0) || a1 < T(-12.0)) a1 = 0.0;
	  //else a1 = T(1.00);
	  
	  // range is [0.0, 1.0]; // bigger is better (closer to one)
	  
	  std::cout << "A1 = " << a1 << std::endl;
	  std::cout << "theta1 = " << theta << std::endl;
	  std::cout << "angle = " << 360.0*atan2(newstate[1].c[0], newstate[2].c[0])/(2*M_PI) << std::endl;
	  
	  reinforcement = T(-0.01)*(T(0.1)*pow(a1, T(2.0)) + T(5.0)*pow(T(Fstep), T(2.0)));

	  std::cout << "REINFORCEMENT: " << reinforcement << std::endl;
	  fflush(stdout);
	}
      }
	
    }

    return true;
  }

  
  template <typename T>
  T CartPole<T>::sign(T value)
  {
    if(value >= T(0.0)) return T(+1.0);
    else return T(-1.0);
  }

  
  template <typename T>
  void CartPole<T>::physicsLoop()
  {    
    double t = 0.0;

    resetLastStep = false;
    
    std::vector<T> thetas;
    T mth = T(0.0);
    T sth = T(0.0);

    double degrees;
    
    while(running){
      
      {
	std::unique_lock<std::mutex> lock(F_change);

	while(F_processed == true){
	  auto now = std::chrono::system_clock::now();
	  F_processed_cond.wait_until(lock, now + 10*100ms);
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
	  if(a > 0)
	    a = a - floor(a);
	  else
	    a = a - (floor(a)+1);
	  
	  a = T(2.0*M_PI)*a;

	  if(a > T(M_PI)){
	    a = T(-1.0)*(T(2.0*M_PI) - a);
	  }

	  degrees = 360.0*(a.c[0]/(2.0*M_PI));

	  thetas.push_back(abs(degrees));

	  auto temp = abs(mth); //  + sth; [DO NOT ADD STANDARD DEVIATION]
	  
	  printf("TIME %f theta = %f deg [%f] [model %d]\n",
		 t, degrees, temp.c[0], this->getHasModel());
	  
	  fflush(stdout);
	}
	
	F_processed = true;
	F_processed_cond.notify_all();
      }

      if(t >= 20.0){
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
      else if(degrees > 45 || degrees < -45){
	// resets if pole is more than 45 degrees off the center
	reset();
      }


      // usleep((unsigned int)(dt.c[0]*1000000.0));  // waits for a single timestep
      
    }
    
    
  }
  


  template class CartPole< math::blas_real<float> >;
  template class CartPole< math::blas_real<double> >;
};
