/*
 * LBFGS second order optimizer for GB-RBM
 * 
 * hopefully this will optimize GB-RBM better than first order gradient descent 
 * which seem to get stuck into local minimas or something..
 */

#ifndef LBFGS_GBRBM_h
#define LBFGS_GBRBM_h

#include "LBFGS.h"
#include "GBRBM.h"
#include "dataset.h"
#include "vertex.h"

namespace whiteice
{
  template <typename T=math::blas_real<float> >
    class LBFGS_GBRBM : public whiteice::math::LBFGS<T>
    {
    public:
      LBFGS_GBRBM(const GBRBM<T>& net,
		  const dataset<T>& d, bool overfit=false);
    
      virtual ~LBFGS_GBRBM();
    
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
      const GBRBM<T> net;
      const dataset<T>& data;
      
    };


  //extern template class LBFGS_GBRBM< float >;
  //extern template class LBFGS_GBRBM< double >;
  extern template class LBFGS_GBRBM< math::blas_real<float> >;
  extern template class LBFGS_GBRBM< math::blas_real<double> >;
  
}


#endif

