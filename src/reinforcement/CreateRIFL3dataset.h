// helper class for RIFL_abstract to generate
// (in the background) dataset for training reinforcement model

#ifndef whiteice_CreateRIFL3dataset_h
#define whiteice_CreateRIFL3dataset_h

#include <thread>
#include <mutex>
#include <vector>

#include "dataset.h"
#include "dinrhiw_blas.h"
#include "RIFL_abstract3.h"

namespace whiteice
{

  template <typename T = math::blas_real<float> >
  class CreateRIFL3dataset
  {
  public:
    
    // calculates reinforcement learning training dataset from database
    // using database_lock
    CreateRIFL3dataset(const RIFL_abstract3<T> & rifl, 
		       //const std::vector< std::vector< rifl_datapoint<T> > >& episodes,
		       const std::multimap< T, std::vector< rifl_datapoint<T> > >& episodes,
		       std::mutex & database_mutex,
		       std::mutex & model_mutex,
		       const unsigned int & epoch, 
		       whiteice::dataset<T>& data);
    
    virtual ~CreateRIFL3dataset();
    
    // starts thread that creates NUMDATAPOINTS samples to dataset
    bool start(const unsigned int NUM_EPISODES);

    // returns true when computation is completed
    bool isCompleted() const;

    // returns true if computation is running
    bool isRunning() const;
    
    bool stop();

    // returns reference to dataset
    // (warning: if calculations are running then dataset can change during use)
    whiteice::dataset<T> const & getDataset() const;

  private:
    
    RIFL_abstract3<T> const & rifl;
    
    // const std::vector< std::vector< rifl_datapoint<T> > >& episodes;
    const std::multimap< T, std::vector< rifl_datapoint<T> > >& episodes;
    std::mutex & database_mutex;
    std::mutex & model_mutex;

    unsigned int const& epoch;

    unsigned int NUM_EPISODES; // number of datapoints to create
    whiteice::dataset<T>& data;
    bool completed;

    std::thread* worker_thread;
    std::mutex   thread_mutex;
    bool running;

    // worker thread loop
    void loop();
    
      
    };


  extern template class CreateRIFL3dataset< math::blas_real<float> >;
  extern template class CreateRIFL3dataset< math::blas_real<double> >;
};

#endif
