/*
 * Cart Pole (Inverted Pendulum) problem
 * 
 * Equations are from
 * 
 * "Correct equations for the dynamics of the cart-pole system"
 * Razvan V. Florian 2007
 * 
 * Applied force is purely CONTINUOUS in this implementation (RIFL_abstract2)
 */

#ifndef __whiteice__cartpole2_h
#define __whiteice__cartpole2_h

#include "RIFL_abstract2.h"

#ifdef USE_SDL
#include <SDL.h>
#include <mutex>
#endif

#include <condition_variable>

namespace whiteice
{

  template <typename T>
    class CartPole2 : public RIFL_abstract2<T>
    {
    public:
      CartPole2();
      ~CartPole2();

      bool physicsIsRunning(){ return this->running; }

      bool getVerbose(){ return verbose; }

      void setVerbose(bool flag){ verbose = flag; }
      
    protected:

      virtual bool getState(whiteice::math::vertex<T>& state);
      
      virtual bool performAction(const whiteice::math::vertex<T>& action,
				 whiteice::math::vertex<T>& newstate,
				 T& reinforcement, bool& endFlag);

      // helper function: normalizes theta values back into [-pi, pi] range
      T normalizeTheta(const T t) const ;

    protected:

#ifdef USE_SDL
      int W, H;
      
      SDL_Window* window;
      SDL_Renderer* renderer;

      std::mutex sdl_mutex; // for sync access to SDL
      bool no_init_sdl = false;

      bool log_verbose = false; // used to control if we print internal log comments.
#endif

      bool verbose = false;

      

      // resets cart-pole variables
      void reset();

      T sign(T value);

      // parameters
      T mc, mp, g, l;
      T uc, up;
      
      T theta, theta_dot, theta_dotdot;
      T x, x_dot, x_dotdot;
      T Nc;

      bool resetLastStep;

      T F;
      std::mutex F_change;
      bool F_processed;
      std::condition_variable F_processed_cond;

      T dt;
      int iteration;

      volatile bool running;
      std::thread* physics_thread;
      std::mutex physics_mutex;

      whiteice::RNG<T> rng;

      void physicsLoop();
      
    };

  
  extern template class CartPole2< math::blas_real<float> >;
  extern template class CartPole2< math::blas_real<double> >;
};


#endif
