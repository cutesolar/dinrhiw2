/*
 * L-BFGS optimizer for simple *recurrent* neural networks
 * 
 *
 */

#ifndef rLBFGS_nnetwork_h
#define rLBFGS_nnetwork_h

#include "LBFGS.h"
#include "nnetwork.h"
#include "dataset.h"
#include "vertex.h"

#include "RNG.h"


namespace whiteice
{
  template <typename T=math::blas_real<float> >
    class rLBFGS_nnetwork : public whiteice::math::LBFGS<T>
    {
    public:
      rLBFGS_nnetwork(const nnetwork<T>& net,
		      const dataset<T>& d,
		      const unsigned int deepness=1,
		      bool overfit=false,
		      bool negativefeedback=false);
    
      virtual ~rLBFGS_nnetwork();
    
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
      const unsigned int deepness;
    
      const nnetwork<T> net;
      const dataset<T>& data;    
    
      bool negativefeedback;
    
      dataset<T> dtrain;
      dataset<T> dtest;      
      
    };


  extern template class rLBFGS_nnetwork< math::blas_real<float> >;
  extern template class rLBFGS_nnetwork< math::blas_real<double> >;
  
}


#endif
