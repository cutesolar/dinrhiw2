/*
 * implements Fast PCA algorithm as described in:
 *
 * "Fast principal component analysis using fixed-point algorithm."
 * Alok Sharma *, Kuldip K. Paliwal (2007)
 *
 * FIXME: This apparently only works with real valued data but
 *        extensions to complex valued shouldn't be too hard.
 *
 *        Algorithm now don't calculate Cxx matrix but uses data vectors 
 *        to estimate g_next = Cxx*g_prev matrix multiplication step and when estimating
 *        eigenvalues.
 *
 *        TODO: Write improved code which works with large number of datapoints 
 *        (need only sample smaller numbers to make statistical estimate).
 *
 * Copyright Tomas Ukkonen 2014, 2022
 *
 */

#ifndef __fastpca_h
#define __fastpca_h

#include "matrix.h"
#include "vertex.h"

#include <vector>


namespace whiteice
{
  namespace math
  {
    
    /*
     * Extracts first "dimensions" PCA vectors from data
     * PCA = X^t when Cxx = E{(x-m)(x-m)^t} = X*D*X^t
     */
    template <typename T>
    bool fastpca(const std::vector< vertex<T> >& data, 
		 const unsigned int dimensions,
		 math::matrix<T>& PCA,
		 std::vector<T>& eigenvalues);
    
    /*
     * Extracts PCA vectors having top p% E (0,1] of the total
     * variance in data. (Something like 90% could be
     * good for preprocessing while keeping most of variation
     * in data.
     */
    template <typename T>
    bool fastpca_p(const std::vector <vertex<T> >& data,
		   const float percent_total_variance,
		   math::matrix<T>& PCA,
		   std::vector<T>& eigenvalues);
		 

    
    extern template bool fastpca< blas_real<float> >
    (const std::vector< vertex< blas_real<float> > >& data, 
     const unsigned int dimensions,
     math::matrix< blas_real<float> >& PCA,
     std::vector< blas_real<float> >& eigenvalues);
    
    extern template bool fastpca< blas_real<double> >
    (const std::vector< vertex< blas_real<double> > >& data, 
     const unsigned int dimensions,
     math::matrix< blas_real<double> >& PCA,
     std::vector< blas_real<double> >& eigenvalues);


    extern template bool fastpca< superresolution< blas_real<float>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution< blas_real<float>, modular<unsigned int> > > >& data, 
     const unsigned int dimensions,
     math::matrix< superresolution< blas_real<float>, modular<unsigned int> > >& PCA,
     std::vector< superresolution< blas_real<float>, modular<unsigned int> > >& eigenvalues);
    
    extern template bool fastpca< superresolution< blas_real<double>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution< blas_real<double>, modular<unsigned int> > > >& data, 
     const unsigned int dimensions,
     math::matrix< superresolution< blas_real<double>, modular<unsigned int> > >& PCA,
     std::vector< superresolution< blas_real<double>, modular<unsigned int> > >& eigenvalues);


    
    
    extern template bool fastpca_p< blas_real<float> >
    (const std::vector <vertex< blas_real<float> > >& data,
     const float percent_total_variance,
     math::matrix< blas_real<float> >& PCA,
     std::vector< blas_real<float> >& eigenvalues);

    extern template bool fastpca_p< blas_real<double> >
    (const std::vector <vertex< blas_real<double> > >& data,
     const float percent_total_variance,
     math::matrix< blas_real<double> >& PCA,
     std::vector< blas_real<double> >& eigenvalues);
    
  };
};


#endif

