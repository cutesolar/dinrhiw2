/*
 * LBFGS second order optimizer for BB-RBM
 * 
 * hopefully this will optimize BB-RBM better than first order gradient descent 
 * which seem to get stuck into local minimas or something..
 */

#ifndef LBFGS_BBRBM_h
#define LBFGS_BBRBM_h

#include "LBFGS.h"
#include "BBRBM.h"
#include "dataset.h"
#include "vertex.h"


namespace whiteice
{
  template <typename T=math::blas_real<float> >
    class LBFGS_BBRBM : public whiteice::math::LBFGS<T>
    {
    public:
      LBFGS_BBRBM(const BBRBM<T>& net,
		  const dataset<T>& d, bool overfit=false);
    
      virtual ~LBFGS_BBRBM();
    
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
      const BBRBM<T> net;
      const dataset<T>& data;
      
    };


  //extern template class LBFGS_BBRBM< float >;
  //extern template class LBFGS_BBRBM< double >;
  extern template class LBFGS_BBRBM< math::blas_real<float> >;
  extern template class LBFGS_BBRBM< math::blas_real<double> >;
  
}


#endif

