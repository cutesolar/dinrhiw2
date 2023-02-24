/*
 * discretizes decision tree data to binary variables used by binary decision tree
 *
 * Tomas Ukkonen 2023
 */


#ifndef __whiteice__discretization_h
#define __whiteice__discretization_h

#include "vertex.h"


namespace whiteice
{

  // discretizes real-valued or discrete data to binary data
  template <typename T=math::blas_real<float> >
    bool discretization(const std::vector< math::vertex<T> >& input,
			const std::vector< math::vertex<T> >& output,
			std::vector< std::vector<bool> >& inputResults,
			std::vector< std::vector<bool> >& outputResults);

  
  
  
  extern template bool discretization< math::blas_real<float> >
    (const std::vector< math::vertex< math::blas_real<float> > >& input,
     const std::vector< math::vertex< math::blas_real<float> > >& output,
     std::vector< std::vector<bool> >& inputResults,
     std::vector< std::vector<bool> >& outputResults);

  extern template bool discretization< math::blas_real<double> >
    (const std::vector< math::vertex< math::blas_real<double> > >& input,
     const std::vector< math::vertex< math::blas_real<double> > >& output,
     std::vector< std::vector<bool> >& inputResults,
     std::vector< std::vector<bool> >& outputResults);
  
};


#endif
