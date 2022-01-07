/*
 * L-BFGS optimizer for full *recurrent* neural networks
 * 
 * dataset must have input x, output y clusters and episodes cluster
 * where we define i and end j for which training episodes end (input(i)..input(j-1))
 *
 */

#ifndef rLBFGS_recurrent_nnetwork_softmax_actions_h
#define rLBFGS_recurrent_nnetwork_softmax_actions_h

#include "LBFGS.h"
#include "nnetwork.h"
#include "dataset.h"
#include "vertex.h"

#include "RNG.h"

namespace whiteice
{
  template <typename T=math::blas_real<float> >
    class rLBFGS_recurrent_nnetwork_softmax_actions : public whiteice::math::LBFGS<T>
    {
    public:
      rLBFGS_recurrent_nnetwork_softmax_actions(const nnetwork<T>& net,
				const dataset<T>& d,
				bool overfit=false);
      
      virtual ~rLBFGS_recurrent_nnetwork_softmax_actions();
    
    protected:

      // optimized function
      virtual T U(const math::vertex<T>& x) const;
      virtual math::vertex<T> Ugrad(const math::vertex<T>& x) const;
    
      virtual bool heuristics(math::vertex<T>& x) const;

    public:
    
      // calculates the current solution's "real" error
      // (we keep iterating until U(x) converges or getError()
      //  increases instead of decreasing)
      T getError(const math::vertex<T>& x) const;

    private:
      // clipped gradient values which otherwise cause exploding gradients in LBFGS optimization
      void box_values(math::matrix<T>& GRAD) const;
      
      
      const nnetwork<T> net;
      const dataset<T>& data;

      const bool real_error = true; // convert error terms to real unprocessed error in data..
    
      bool negativefeedback;
    
      dataset<T> dtrain;
      dataset<T> dtest;

      // regularizers
      // [minimizes large w values if they happen]
      const T alpha = T(1e-10f); // log(exp(-0.5*||w||^2)) regularizer term to minimize (was: 1e-6)

      // was: 0.001 => now: 0.00001 (1e-5) (entropy value dominates when values are small??)
      const T entropy_regularizer = T(1e-6f); // negative entropy term to maximize entropy of selection
      
    };


  extern template class rLBFGS_recurrent_nnetwork_softmax_actions< math::blas_real<float> >;
  extern template class rLBFGS_recurrent_nnetwork_softmax_actions< math::blas_real<double> >;
  
}


#endif
