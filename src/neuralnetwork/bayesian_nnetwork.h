/*
 * MCMC bayesian neural network (using samples from p(w|data))
 *
 * supports use of samples of weights p(w) which
 * will be used to store and load network state and 
 * calculate responses.
 */

#ifndef bayesian_nnetwork_h
#define bayesian_nnetwork_h

#include "nnetwork.h"

namespace whiteice
{
  template < typename T = math::blas_real<float> >
    class bayesian_nnetwork
    {
    public:

    bayesian_nnetwork();
    bayesian_nnetwork(const bayesian_nnetwork<T>& bnet);
    virtual ~bayesian_nnetwork();

    bayesian_nnetwork<T>& operator=(const bayesian_nnetwork<T>& bnet);

    void printInfo() const; // mostly for debugging.. prints NN information/data.

    void diagnosticsInfo() const ;

    /*
     * imports and exports samples of p(w) to and from nnetwork
     */
    unsigned int getNumberOfSamples() const ; // number of samples in BNN

    bool importSamples(const whiteice::nnetwork<T>& nn,
		       const std::vector< math::vertex<T> >& weights);

    bool importSamples(const whiteice::nnetwork<T>& nn,
		       const std::vector< math::vertex<T> >& weights,
		       const std::vector< math::vertex<T> >& bndatas);
      
    bool importNetwork(const nnetwork<T>& net);

    bool exportSamples(whiteice::nnetwork<T>& nn, 
		       std::vector< math::vertex<T> >& weights,
		       int latestN = 0) const;

    bool exportSamples(whiteice::nnetwork<T>& nn, 
		       std::vector< math::vertex<T> >& weights,
		       std::vector< math::vertex<T> >& bndatas,
		       int latestN = 0) const;

    bool exportNetworks(std::vector< whiteice::nnetwork<T>* >& nnlist,
			int latestN = 0) const;

    const whiteice::nnetwork<T>& getNetwork() const {
      assert(nnets.size()>0); return (*(nnets[0]));
    }

    const whiteice::nnetwork<T>& getNetwork(unsigned int index) const {
      assert(index<nnets.size());
      return (*nnets[index]);
    }
    
    bool getArchitecture(std::vector<unsigned int>& arch) const;

    // alters architecture to target and keeps initial unchanged layers (unaltered arch) if possible
    bool editArchitecture(std::vector<unsigned int>& arch,
			  typename nnetwork<T>::nonLinearity nl);

    // creates and injects subnets starting from n:th layer
    bayesian_nnetwork<T>* createSubnet(const unsigned int fromLayer);
    bool injectSubnet(const unsigned int fromLayer, bayesian_nnetwork<T>* nn);
    
    bool setNonlinearity(typename nnetwork<T>::nonLinearity nl);
    void getNonlinearity(std::vector< typename nnetwork<T>::nonLinearity >& nl);
    
    /*
     * downsamples number of neural networks down to N neural networks
     * or if N > number of neural networks/samples does nothing
     */
    bool downsample(unsigned int N);

#if 1
    // calculates E[f(input,w)] = E[y|x] and Var[f(x,w)] = Var[y|x] for given input
    bool calculate(const math::vertex<T>& input,
		   math::vertex<T>& mean,
		   math::matrix<T>& covariance,
		   unsigned int SIMULATION_DEPTH /* = 1 */, // for recurrent use of nnetworks..
		   int latestN /*= 0 */) const;
#endif

    // don't calculate large covariance matrix (faster)
    bool calculate(const math::vertex<T>& input,
		   math::vertex<T>& mean,
		   unsigned int SIMULATION_DEPTH,
		   int latestN) const;

    unsigned int outputSize() const ;
    unsigned int inputSize() const ;

    // stores and loads bayesian nnetwork to a dataset file
    // (saves all samples into files)
    bool load(const std::string& filename) ;
    bool save(const std::string& filename) const ;

    private:

    std::vector< nnetwork<T>* > nnets;
      
      
    };

  extern template class bayesian_nnetwork< math::blas_real<float> >;
  extern template class bayesian_nnetwork< math::blas_real<double> >;

  extern template class bayesian_nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >;
  extern template class bayesian_nnetwork< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >;

  
};



#endif


