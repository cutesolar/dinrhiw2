// helper class for RIFL2_abstract to generate
// (in the background) dataset for training reinforcement model

#ifndef whiteice_CreateRIFL2dataset_h
#define whiteice_CreateRIFL2dataset_h

#include <thread>
#include <mutex>
#include <vector>

#include "dataset.h"
#include "dinrhiw_blas.h"
#include "RIFL_abstract2.h"

namespace whiteice
{

  template <typename T = math::blas_real<float> >
    class CreateRIFL2dataset
    {
    public:
      
    // calculates reinforcement learning training dataset from database
    // using database_lock
    CreateRIFL2dataset(RIFL_abstract2<T> const & rifl, 
		       std::vector< rifl2_datapoint<T> > const & database,
		       std::vector< std::vector< rifl2_datapoint<T> > > const & episodes,
		       std::mutex & database_mutex,
		       unsigned int const& epoch);

      
    virtual ~CreateRIFL2dataset();

    // starts thread that creates NUMDATAPOINTS samples to dataset
    bool start(const unsigned int NUMDATAPOINTS, const bool smartEpisodes = false);

    // returns true when computation is completed
    bool isCompleted() const;

    // returns true if computation is running
    bool isRunning() const;
    
    bool stop();

    // returns reference to dataset
    // (warning: if calculations are running then dataset can change during use)
    const whiteice::dataset<T>& getDataset() const;

    private:

    whiteice::dataset<T> policy_preprocess;
    whiteice::bayesian_nnetwork<T> lagged_policy;

    whiteice::dataset<T> Q_preprocess;
    whiteice::bayesian_nnetwork<T> lagged_Q;

    whiteice::RNG<T> rng;
    
    RIFL_abstract2<T> const & rifl;

    const std::vector< rifl2_datapoint<T> >& database;
    const std::vector< std::vector< rifl2_datapoint<T> > >& episodes;
    std::mutex & database_mutex;

    bool smartEpisodes = false;

    unsigned int const& epoch;

    unsigned int NUMDATA; // number of datapoints to create
    whiteice::dataset<T> data;
    bool completed = false;;

    std::thread* worker_thread = NULL;
    mutable std::mutex   thread_mutex;
    bool running = false;

    // worker thread loop
    void loop();
    
      
    };


  extern template class CreateRIFL2dataset< math::blas_real<float> >;
  extern template class CreateRIFL2dataset< math::blas_real<double> >;
};

#endif
