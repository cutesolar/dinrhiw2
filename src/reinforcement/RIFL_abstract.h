/*
 * Reinforcement learning for 
 * discrete actions and continuous states
 * uses neural network (nnetwork) to learn 
 * utility function which is used to select 
 * the next action.
 * 
 * Implementation is mostly based on the following paper:
 * 
 * Self-Improving Reactive Agents Based On 
 * Reinforcement Learning, Planning and Teaching
 * LONG-JI, LIN
 * Machine Learning, 8, 293-321 (1992)
 *
 */

#ifndef whiteice_RIFL_abstract_h
#define whiteice_RIFL_abstract_h

#include <string>
#include <mutex>
#include <thread>
#include <vector>

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "bayesian_nnetwork.h"
#include "dataset.h"
#include "RNG.h"


namespace whiteice
{

  template <typename T>
    class CreateRIFLdataset;
  
  
  template <typename T = math::blas_real<float> >
    class RIFL_abstract
    {
    public:
      
      /*
       * numActions        - the number of discrete different actions
       * numStates         - the number of dimensions in state vectors
       */
      RIFL_abstract(const unsigned int numActions,
		    const unsigned int numStates);

      /*
       * numActions         - the number of discerete different actions
       * numStates          - the number of dimensions in state vectors
       * arch               - Q neural network architecture "numStates-*-*-numActions"
       */
      RIFL_abstract(const unsigned int numActions,
		    const unsigned int numStates,
		    std::vector<unsigned int> arch);
      
      ~RIFL_abstract() ;
      
      // starts Reinforcement Learning thread
      bool start();

      // stops Reinforcement Learning thread
      bool stop();
      
      bool isRunning() const;
      
      // epsilon E [0,1] percentage of actions are chosen according to model
      //                 1-e percentage of actions are random (exploration)
      bool setEpsilon(T epsilon) ;
      
      T getEpsilon() const ;
      
      /*
       * sets/gets learning mode 
       * (do we do just control or also try to learn from data)
       */
      void setLearningMode(bool learn) ;
      bool getLearningMode() const ;

      // do we sample episodes and not samples, needed for recurrent neural network learning
      void setSmartEpisodes(bool use_episodes){ useEpisodes = use_episodes; }
      bool getSmartEpisodes() const{ return useEpisodes; }
      
      
      /*
       * hasModel is number of current optimization model (starting from zero)
       * (from init or from load)
       *
       * as long as we don't have a proper model (hasModel == 0)
       * we make random actions (initially) 
       */
      void setHasModel(unsigned int hasModel) ;
      unsigned int getHasModel() ;
      
      unsigned int getNumActions() const { return numActions; }
      unsigned int getNumStates() const { return numStates; }
      
    // saves learnt Reinforcement Learning Model to file
      bool save(const std::string& filename) const;
      
      // loads learnt Reinforcement Learning Model from file
      bool load(const std::string& filename);
      
    protected:
      
      unsigned int numActions, numStates;
      
      virtual bool getState(whiteice::math::vertex<T>& state) = 0;
      
      virtual bool performAction(const unsigned int action,
				 whiteice::math::vertex<T>& newstate,
				 T& reinforcement,
				 bool& endFlag) = 0;
            
    protected:
      
      // helper function, returns minimum value in vec
      unsigned int min(const std::vector<unsigned int>& vec) const ;
      
      // separate network for each action
      whiteice::bayesian_nnetwork<T> model;
      whiteice::dataset<T> preprocess;
      mutable std::mutex model_mutex;
      
      unsigned int hasModel;
      bool learningMode;
      bool useEpisodes;
      
      T epsilon;
      T gamma;
      
      // whiteice::RNG<T> rng; [don't use own rng)]
      
      volatile int thread_is_running;
      std::thread* rifl_thread;
      std::mutex thread_mutex;
      
      void loop();
      
      // friend thread class to do heavy computations in background
      // out of main loop 
      friend class CreateRIFLdataset<T>;
      
  };

  template <typename T>
    struct rifl_datapoint
    {
      whiteice::math::vertex<T> state, newstate;
      unsigned int action;
      T reinforcement;
      
      bool lastStep;
    };


  extern template class RIFL_abstract< math::blas_real<float> >;
  extern template class RIFL_abstract< math::blas_real<double> >;
};

#include "CreateRIFLdataset.h"

#endif
