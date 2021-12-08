/*
 * Variational Autoencoder code
 * Tomas Ukkonen <tomas.ukkonen@iki.fi> 2020
 * 
 * Implementation research paper:
 * "Auto-Encoding Variational Bayes. Diedrik Kingma and Max Welling."
 * Implementation extra math notes in docs directory:
 * [variational-bayes-encoder.tm]
 * 
 * NOTE: the input data x ~ p(x) should be roughly distributed as Normal(0,I)
 * because it is assumed that p(x|z) don't estimate variance term but
 * fixes error to be Var[x_i] ~ 1 for each dimension.
 */

#ifndef __nnetwork_VAE_h
#define __nnetwork_VAE_h

#include "nnetwork.h"
#include "dinrhiw_blas.h"
#include "vertex.h"
#include "conffile.h"
//#include "compressable.h"
#include "dataset.h"
#include "LoggingInterface.h"
#include "VisualizationInterface.h"
#include "RNG.h"

#include <vector>

namespace whiteice
{
  template <typename T = math::blas_real<float> >
    class VAE
    {
    public:

      VAE(const VAE<T>& vae);
      VAE(const nnetwork<T>& encoder,  // x -> (z_mean, z_var)
	  const nnetwork<T>& decoder); // z -> (x_mean)
	
      VAE(const std::vector<unsigned int> encoderArchitecture,
	  const std::vector<unsigned int> decoderArchitecture);
      
      void getModel(nnetwork<T>& encoder,
		    nnetwork<T>& decoder) const;

      nnetwork<T>& getEncoder();
      nnetwork<T>& getDecoder();
      
      bool setModel(const nnetwork<T>& encoder,
		    const nnetwork<T>& decoder);
      
      bool setModel(const std::vector<unsigned int> encoderArchitecture,
		    const std::vector<unsigned int> decoderArchitecture);
      
      
      // x -> z (hidden)
      bool encode(const math::vertex<T>& x,
		  const nnetwork<T>& encoder,
		  math::vertex<T>& zmean,
		  math::vertex<T>& zstdev) const;

      // x -> z (hidden) [with dropout table]
      bool encode(const math::vertex<T>& x,
		  const nnetwork<T>& encoder,
		  const std::vector< std::vector<bool> >& dropout,
		  math::vertex<T>& zmean,
		  math::vertex<T>& zstdev) const;
      
      bool encodeSample(const math::vertex<T>& x,
			const nnetwork<T>& encoder,
			math::vertex<T>& zsample) const;

      bool encodeSample(const math::vertex<T>& x,
			const nnetwork<T>& encoder,
			const std::vector< std::vector<bool> >& dropout,
			math::vertex<T>& zsample) const;
      
      // z (hidden) -> x
      bool decode(const math::vertex<T>& z,
		  const nnetwork<T>& decoder,
		  math::vertex<T>& xmean) const;

      // z (hidden) -> x [with dropout table]
      bool decode(const math::vertex<T>& z,
		  const nnetwork<T>& decoder,
		  const std::vector< std::vector<bool> >& dropout,
		  math::vertex<T>& xmean) const;
      
      unsigned int getDataDimension() const;
      unsigned int getEncodedDimension() const;
      
      void getParameters(math::vertex<T>& p) const;
      bool setParameters(const math::vertex<T>& p, const bool dropout);
      
      // to set minibatch mode in which we use only sample of 30 data points when calculating gradient
      void setUseMinibatch(bool use_minibatch);
      bool getUseMinibatch() const;

      // set dropout mode
      void setDropout(bool dropout = true);
      bool getDropout() const;
      
      void initializeParameters();
      
      T getError(const std::vector< math::vertex<T> >& xsamples) const;
      
      T getLoglikelihood(const std::vector< math::vertex<T> >& xsamples) const;
      
      // x = samples, learn x = decode(encode(x))
      // optimizes using gradient descent,
      // TODO: divide data to teaching and test datasets(75%/25%)
      //
      // model_stdev_error(last N iters)/model_mean_error(last N iters) < convergence_ratio
      //
      // stops computation loop if *running becomes false
      // 
      bool learnParameters(const std::vector< math::vertex<T> >& xsamples,
			   T convergence_ratio = T(0.01f),
			   bool verbose = false,
			   LoggingInterface* messages = NULL,
			   VisualizationInterface* gui = NULL,
			   bool* running = NULL);
      
      // calculates gradient of parameter p using all samples
      bool calculateGradient(const std::vector< math::vertex<T> >& xsamples,
			     math::vertex<T>& pgradient,
			     LoggingInterface* messages = NULL,
			     bool* running = NULL) const;
      
      
      bool load(const std::string& filename) ;
      bool save(const std::string& filename) const ;
      
    private:
    
      nnetwork<T> encoder; // x -> (z mean ,z stdev)
      nnetwork<T> decoder; // z -> (x mean)

      whiteice::RNG<T> rng;
      
      bool minibatchMode = true;
      bool dropout = false; // drop out heuristic for optimization [DISABLED]
    
    };
  
  
  // extern template class VAE< float >;
  // extern template class VAE< double >;  
  extern template class VAE< math::blas_real<float> >;
  extern template class VAE< math::blas_real<double> >;
}

#endif
