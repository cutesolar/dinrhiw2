/*
 * cluster similarity and merging logic
 */

#include "dinrhiw.h"
#include "hctree.h"

#ifndef HCLogic_h
#define HCLogic_h


namespace whiteice
{
  
  template < typename Parameters, typename T = whiteice::math::blas_real<float> >
    class HCLogic
    {
      public:
      
      virtual ~HCLogic(){ }
      
      // calculates parameters of a leaf node
      virtual bool initialize(whiteice::hcnode<Parameters, T>* p)  = 0;
      
      // calculates similarity between clusters (parameters)
      virtual T similarity(std::vector< hcnode<Parameters, T>* > clusters,
			   unsigned int i, unsigned int j)  = 0;
      
      // calculates merged node
      virtual bool merge(whiteice::hcnode<Parameters, T>* p1,
			 whiteice::hcnode<Parameters, T>* p2,
			 whiteice::hcnode<Parameters, T>& result) const  = 0;
      
    };
};


#endif
