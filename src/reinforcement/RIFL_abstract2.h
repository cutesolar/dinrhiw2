/*
 * Reinforcement learning for 
 * continuous actions and continuous states
 * uses neural network (nnetwork) to learn 
 * utility (Q) and policy functions
 * 
 * Implementation is mostly based on the following paper:
 * 
 * Continuous Control With Deep Reinforcement Learning.
 * Timothy P. Lillicrap*, Jonathan J. Hunt*, 
 * Alexander Pritzel, Nicolas Heess, Tom Erez, 
 * Yuval Tassa, David Silver & Daan Wierstra
 * Google DeepMind, London, UK.
 * Conference paper at ICLR 2016
 *
 * NOTE 2021: Added L2 regularization 0.02*0.5*||w||^2 term 
 *            to optimization of Q and policy so that 
 *            neural network weights cannot explode.
 *
 */

#ifndef whiteice_RIFL_abstract2_h
#define whiteice_RIFL_abstract2_h

#include <string>
#include <mutex>
#include <thread>
#include <vector>

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "bayesian_nnetwork.h"
#include "RNG.h"
#include "dataset.h"


namespace whiteice
{
  template <typename T>
    class CreateRIFL2dataset;

  template <typename T>
    class CreatePolicyDataset;
  
  template <typename T>
  struct rifl2_datapoint
  {
    whiteice::math::vertex<T> state, newstate;
    whiteice::math::vertex<T> action;
    T reinforcement;
    
    bool lastStep; // true if was the last step of the simulation
  };
  

  template <typename T = math::blas_real<float> >
  class RIFL_abstract2
  {
  public:
    
    // parameters are dimensions of vectors dimActions and dimStates: R^d
    RIFL_abstract2(unsigned int numActions, unsigned int numStates);
    
    RIFL_abstract2(unsigned int numActions, unsigned int numStates,
		   std::vector<unsigned int> Q_arch,
		   std::vector<unsigned int> policy_arch);

    ~RIFL_abstract2() ;

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
     * sets/gets learning mode (default: on)
     * (do we do just control or also try to learn from data)
     */
    void setLearningMode(bool learn) ;
    bool getLearningMode() const ;

    /*
     * sets/gets sleeping mode (default: on)
     * if we sleep RIFL is inactive and does nothing [no learning, no actions]
     */
    void setSleepingMode(bool sleep) ;
    bool getSleepingMode() const ;

    /*
     * hasModel flag means we have a proper model
     * (from optimization or from load)
     *
     * as long as we don't have a proper model
     * we make random actions (initially) 
     */
    void setHasModel(unsigned int hasModel) ;
    unsigned int getHasModel() ;

    float getLatestEpisodeError() const;

    unsigned int getDatabaseSize() const;

    unsigned int getNumActions() const { return numActions; }
    unsigned int getNumStates() const { return numStates; }

    // tells policy() returns value one hot encoded unscaled probability values (log(p_i))
    // from which proper one-hot action vector is sampled.
    void setOneHotAction(bool isOneHotAction){ oneHotEncodedAction = isOneHotAction; }
    bool getOneHotAction() const{ return oneHotEncodedAction; }

    // do we sample episodes and not samples, needed for recurrent neural network learning
    // FIXME: don't happen properly now
    void setSmartEpisodes(bool use_episodes){ useEpisodes = use_episodes; }
    bool getSmartEpisodes() const{ return useEpisodes; }

    // saves learnt Reinforcement Learning Model to file
    bool save(const std::string& filename) const;
    
    // loads learnt Reinforcement Learning Model from file
    bool load(const std::string& filename);

  protected:
    
    unsigned int numActions, numStates; // dimensions of R^d vectors
    
    virtual bool getState(whiteice::math::vertex<T>& state) = 0;

    // action vector is [0,1]^d (output of sigmoid non-linearity)
    virtual bool performAction(const whiteice::math::vertex<T>& action,
			       whiteice::math::vertex<T>& newstate,
			       T& reinforcement,
			       bool& endFlag) = 0;

    void onehot_prob_select(const whiteice::math::vertex<T>& action,
			    whiteice::math::vertex<T>& new_action,
			    const T temperature = T(1.0f));

    // reinforcement Q model: Q(state, action) ~ discounted future cost
    whiteice::bayesian_nnetwork<T> Q, lagged_Q;
    whiteice::dataset<T> Q_preprocess;
    mutable std::mutex Q_mutex;

    // f(state) = action
    whiteice::bayesian_nnetwork<T> policy, lagged_policy;;
    whiteice::dataset<T> policy_preprocess;
    mutable std::mutex policy_mutex;

    // database
    std::vector< rifl2_datapoint<T> > database;
    mutable std::mutex database_mutex;
    


    
    std::vector<unsigned int> hasModel;
    mutable std::mutex has_model_mutex;
    
    float latestError;
    bool learningMode, sleepMode;
    
    T epsilon;
    T gamma;
    bool oneHotEncodedAction = false;
    bool useEpisodes = false;
    
    class whiteice::RNG<T> rng;
    
    volatile int thread_is_running;
    std::thread* rifl_thread;
    std::mutex thread_mutex;
    
    void loop();
    
    // friend thread class to do heavy computations in background
    // out of main loop 
    friend class CreateRIFL2dataset<T>;

    friend class CreatePolicyDataset<T>;
    
    };



  extern template class RIFL_abstract2< math::blas_real<float> >;
  extern template class RIFL_abstract2< math::blas_real<double> >;
};

#include "CreateRIFL2dataset.h"
#include "CreatePolicyDataset.h"

#endif
