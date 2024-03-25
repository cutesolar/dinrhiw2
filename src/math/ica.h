/*
 * independent components analysis code
 */

#ifndef ica_h
#define ica_h

#include "vertex.h"
#include "matrix.h"

namespace whiteice
{
  namespace math
  {
    
    // solves independent components from the data and saves
    // dependacy removal matrix to W. Uses deflate method.
    // data D MUST be already white (PCA preprocessed)
    // matrix W is transforms data to independent components
    // 
    // if not all ICA components are not found (no convergence),
    // uses non-converged ICA vectors with some kind of solution in PCA subspace
    //
    // adjust tolerance and MAXITERS to reduce quality of the results but get faster real-time results
    template <typename T>
    bool ica(const matrix<T>& D, matrix<T>& W, bool verbose = false,
	     const T tolerance = T(0.0001), const unsigned int MAXITERS=1000);

    // data MUST be already white (PCA preprocessed)
    // 
    // if not all ICA components are not found (no convergence),
    // uses non-conveged ICA vectors with some kind of solution in PCA subspace
    //  
    // adjust tolerance and MAXITERS to reduce quality of the results but get faster real-time results
    template <typename T>
      bool ica(const std::vector< math::vertex<T> >& data,
	       matrix<T>& W, bool verbose = false,
	       const T tolerance = T(0.0001), const unsigned int MAXITERS=1000);


    
    // ICA TODO: reordering and recalculating of ICs which have been computed
    // after first non-covergent IC. This way all ICs which converge will be reliable.
    // non-converged ICs should be checked for gassianity and gaussian ones should
    // grouped and PCAed so that the gaussian subspace is also solved as well as possible
    
    extern template bool ica< blas_real<float> >
    (const matrix< blas_real<float> >& D, matrix< blas_real<float> >& W, bool verbose, const blas_real<float> tolerance, const unsigned int MAXITERS) ;
    extern template bool ica< blas_real<double> >
    (const matrix< blas_real<double> >& D, matrix< blas_real<double> >& W, bool verbose, const blas_real<double> tolerance, const unsigned int MAXITERS) ;

    
    extern template bool ica< blas_complex<float> >
      (const matrix< blas_complex<float> >& D, matrix< blas_complex<float> >& W, bool verbose, const blas_complex<float> tolerance, const unsigned int MAXITERS) ;
    extern template bool ica< blas_complex<double> >
    (const matrix< blas_complex<double> >& D, matrix< blas_complex<double> >& W, bool verbose, const blas_complex<double> tolerance, const unsigned int MAXITERS) ;

    extern template bool ica< superresolution<blas_real<float>, modular<unsigned int> > >
    (const matrix< superresolution<blas_real<float>, modular<unsigned int> > >& D,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution<blas_real<float>, modular<unsigned int> > tolerance, const unsigned int MAXITERS) ;
    
    extern template bool ica< superresolution<blas_real<double>, modular<unsigned int> > >
    (const matrix< superresolution<blas_real<double>, modular<unsigned int> > >& D,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution<blas_real<double>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);
    
    extern template bool ica< superresolution<blas_complex<float>, modular<unsigned int> > >
    (const matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& D,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution<blas_complex<float>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);

     
    extern template bool ica< superresolution<blas_complex<double>, modular<unsigned int> > >
    (const matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& D,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution<blas_complex<double>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);

    
    extern template bool ica< blas_real<float> >
    (const std::vector< math::vertex< blas_real<float> > >& data, matrix< blas_real<float> >& W, bool verbose,
     const blas_real<float> tolerance, const unsigned int MAXITERS);
    
    extern template bool ica< blas_real<double> >
    (const std::vector< math::vertex< blas_real<double> > >& data, matrix< blas_real<double> >& W, bool verbose,
     const blas_real<double> tolerance, const unsigned int MAXITERS);
    
    // extern template bool ica< float >
    //  (const std::vector< math::vertex<float> >& data, matrix<float>& W, bool verbose) ;
    //extern template bool ica< double >
    //  (const std::vector< math::vertex<double> >& data, matrix<double>& W, bool verbose, const unsigned int MAXITERS) ;

    extern template bool ica< blas_complex<float> >
    (const std::vector< math::vertex< blas_complex<float> > >& data, matrix< blas_complex<float> >& W, bool verbose,
     const blas_complex<float> tolerance, const unsigned int MAXITERS);
    
    extern template bool ica< blas_complex<double> >
    (const std::vector< math::vertex< blas_complex<double> > >& data, matrix< blas_complex<double> >& W, bool verbose,
     const blas_complex<double> tolerance, const unsigned int MAXITERS);


    extern template bool ica< superresolution<blas_real<float>, modular<unsigned int> > >
    (const std::vector< math::vertex< superresolution<blas_real<float>, modular<unsigned int> > > >& data,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution< blas_real<float>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);
    
    extern template bool ica< superresolution<blas_real<double>, modular<unsigned int> > >
    (const std::vector< math::vertex< superresolution<blas_real<double>, modular<unsigned int> > > >& data,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution< blas_real<double>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);
    
    
    extern template bool ica< superresolution<blas_complex<float>, modular<unsigned int> > >
    (const std::vector< math::vertex< superresolution<blas_complex<float>, modular<unsigned int> > > >& data,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution< blas_complex<float>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);
    
    extern template bool ica< superresolution<blas_complex<double>, modular<unsigned int> > >
    (const std::vector< math::vertex< superresolution<blas_complex<double>, modular<unsigned int> > > >& data,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution< blas_complex<double>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);

    
  };
};


#endif
