/*
 * Stochastic Gradient Descent optimizer for full *recurrent* neural networks
 * 
 * dataset must have input x, output y clusters and episodes cluster
 * where we define i and end j for which training episodes end (input(i)..input(j-1))
 *
 */

#ifndef SGD_recurrent_nnetwork_h
#define SGD_recurrent_nnetwork_h

#include "SGD.h"
#include "nnetwork.h"
#include "dataset.h"
#include "vertex.h"

#include "RNG.h"

namespace whiteice
{
  template <typename T=math::blas_real<float> >
    class SGD_recurrent_nnetwork : public whiteice::math::SGD<T>
    {
    public:
      SGD_recurrent_nnetwork(const nnetwork<T>& net,
			     const dataset<T>& d,
			     bool overfit=false);
      
      virtual ~SGD_recurrent_nnetwork();
    
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
      const nnetwork<T> net;
      const dataset<T>& data;    

      const bool real_error = true; // do we report unprocessed error..
      bool negativefeedback;
    
      dataset<T> dtrain;
      dataset<T> dtest;

      // regularizer
      const T alpha = T(1e-4);
      
    };


  extern template class SGD_recurrent_nnetwork< math::blas_real<float> >;
  extern template class SGD_recurrent_nnetwork< math::blas_real<double> >;
  
}


#endif
