/*
 * Stochastic Gradient Descent optimizer for superresolutional neural nets
 * 
 * Tomas Ukkonen 2022 
 */

#ifndef SGD_snet_h
#define SGD_snet_h

#include "SGD.h"
#include "nnetwork.h"
#include "dataset.h"
#include "vertex.h"


namespace whiteice
{

  template <typename T=math::blas_real<float> >
    class SGD_snet :
    public whiteice::math::SGD< whiteice::math::superresolution< T, whiteice::math::modular<unsigned int> > >
    {
    public:
    
    SGD_snet
    (const nnetwork< whiteice::math::superresolution< T, whiteice::math::modular<unsigned int> > >& net,
     const dataset< T >& d,
     bool overfit=false);
    
    virtual ~SGD_snet();
    
    protected:
    
    // optimized function
    virtual whiteice::math::superresolution<T, whiteice::math::modular<unsigned int> > 
    U
    (const math::vertex< whiteice::math::superresolution< T,
     whiteice::math::modular<unsigned int> > >& x) const;
    
    virtual math::vertex< whiteice::math::superresolution< T, whiteice::math::modular<unsigned int> > >
    Ugrad(const math::vertex< whiteice::math::superresolution< T,
	  whiteice::math::modular<unsigned int> > >& x) const;
    
    virtual bool heuristics
    (math::vertex< whiteice::math::superresolution< T,
     whiteice::math::modular<unsigned int> > >& x) const;
    
    public:
    
    // calculates the current solution's "real" error
    // (we keep iterating until U(x) converges or getError()
    //  increases instead of decreasing)
    virtual whiteice::math::superresolution<T, whiteice::math::modular<unsigned int> > 
    getError
    (const math::vertex<
     whiteice::math::superresolution< T,
     whiteice::math::modular<unsigned int> > >& x) const;
    
    private:
    
    const nnetwork< whiteice::math::superresolution< T, whiteice::math::modular<unsigned int> > > net;
    const dataset< T >& data;
    
    const bool real_error = true; // do we report unprocessed error..
    bool negativefeedback;
    
    dataset< T > dtrain;
    dataset< T > dtest;
    
    // regularizer
    const T alpha = T(1e-4);
    
  };
  
  
  extern template class SGD_snet< math::blas_real<float> >;
  extern template class SGD_snet< math::blas_real<double> >;
  
  
};

#endif
