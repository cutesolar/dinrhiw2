/*
 * independent component analysis
 */

#include "ica.h"
#include "matrix.h"
#include "vertex.h"
#include "eig.h"
#include "correlation.h"

#include <stdlib.h>


namespace whiteice
{
  namespace math
  {
    
    template bool ica< blas_real<float> >
      (const matrix< blas_real<float> >& D, matrix< blas_real<float> >& W, bool verbose) ;
    template bool ica< blas_real<double> >
      (const matrix< blas_real<double> >& D, matrix< blas_real<double> >& W, bool verbose) ;
    //template bool ica< float >
    //(const matrix<float>& D, matrix<float>& W, bool verbose) ;
    //template bool ica< double >
    //(const matrix<double>& D, matrix<double>& W, bool verbose) ;
    template bool ica< blas_complex<float> >
    (const matrix< blas_complex<float> >& D, matrix< blas_complex<float> >& W, bool verbose) ;
    template bool ica< blas_complex<double> >
    (const matrix< blas_complex<double> >& D, matrix< blas_complex<double> >& W, bool verbose) ;

    template bool ica< superresolution<blas_real<float>, modular<unsigned int> > >
    (const matrix< superresolution<blas_real<float>, modular<unsigned int> > >& D,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& W,
     bool verbose) ;
    
    template bool ica< superresolution<blas_real<double>, modular<unsigned int> > >
    (const matrix< superresolution<blas_real<double>, modular<unsigned int> > >& D,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& W,
     bool verbose) ;

    template bool ica< superresolution<blas_complex<float>, modular<unsigned int> > >
    (const matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& D,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& W,
     bool verbose) ;
    
    template bool ica< superresolution<blas_complex<double>, modular<unsigned int> > >
    (const matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& D,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& W,
     bool verbose) ;
    
    

    template bool ica< blas_real<float> >
      (const std::vector< math::vertex< blas_real<float> > >& data, matrix< blas_real<float> >& W, bool verbose) ;
    template bool ica< blas_real<double> >
      (const std::vector< math::vertex< blas_real<double> > >& data, matrix< blas_real<double> >& W, bool verbose) ;
    //template bool ica< float >
    //  (const std::vector< math::vertex<float> >& data, matrix<float>& W, bool verbose) ;
    //template bool ica< double >
    //  (const std::vector< math::vertex<double> >& data, matrix<double>& W, bool verbose) ;
    template bool ica< blas_complex<float> >
    (const std::vector< math::vertex< blas_complex<float> > >& data, matrix< blas_complex<float> >& W, bool verbose) ;
    template bool ica< blas_complex<double> >
    (const std::vector< math::vertex< blas_complex<double> > >& data, matrix< blas_complex<double> >& W, bool verbose) ;


    template bool ica< superresolution<blas_real<float>, modular<unsigned int> > >
    (const std::vector< math::vertex< superresolution<blas_real<float>, modular<unsigned int> > > >& data,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& W,
     bool verbose) ;
    
    template bool ica< superresolution<blas_real<double>, modular<unsigned int> > >
    (const std::vector< math::vertex< superresolution<blas_real<double>, modular<unsigned int> > > >& data,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& W,
     bool verbose) ;
    
    template bool ica< superresolution<blas_complex<float>, modular<unsigned int> > >
    (const std::vector< math::vertex< superresolution<blas_complex<float>, modular<unsigned int> > > >& data,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& W,
     bool verbose) ;
    
    template bool ica< superresolution<blas_complex<double>, modular<unsigned int> > >
    (const std::vector< math::vertex< superresolution<blas_complex<double>, modular<unsigned int> > > >& data,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& W,
     bool verbose) ;
    
    
    
    template <typename T>
    void __ica_project(vertex<T>& w, const unsigned int n, const matrix<T>& W);

    template void __ica_project< blas_real<float> >
    (vertex< blas_real<float> >& w, const unsigned int n, const matrix< blas_real<float> >& W);
    template void __ica_project< blas_real<double> >
    (vertex< blas_real<double> >& w, const unsigned int n, const matrix< blas_real<double> >& W);
    template void __ica_project<float>
    (vertex<float>& w, const unsigned int n, const matrix<float>& W);
    template void __ica_project<double>
    (vertex<double>& w, const unsigned int n, const matrix<double>& W);
    template void __ica_project< blas_complex<float> >
    (vertex< blas_complex<float> >& w, const unsigned int n, const matrix< blas_complex<float> >& W);
    template void __ica_project< blas_complex<double> >
    (vertex< blas_complex<double> >& w, const unsigned int n, const matrix< blas_complex<double> >& W);

    template void __ica_project< superresolution<blas_real<float>, modular<unsigned int> > >
    (vertex< superresolution<blas_real<float>, modular<unsigned int> > >& w,
     const unsigned int n,
     const matrix< superresolution<blas_real<float>, modular<unsigned int> > >& W);
    
    template void __ica_project< superresolution<blas_real<double>, modular<unsigned int> > >
    (vertex< superresolution<blas_real<double>, modular<unsigned int> > >& w,
     const unsigned int n,
     const matrix< superresolution<blas_real<double>, modular<unsigned int> > >& W);
    
    template void __ica_project< superresolution<blas_complex<float>, modular<unsigned int> > >
    (vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& w,
     const unsigned int n,
     const matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& W);
    
    template void __ica_project< superresolution<blas_complex<double>, modular<unsigned int> > >
    (vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& w,
     const unsigned int n,
     const matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& W);
    
    
    
    template <typename T>
    bool ica(const std::vector< math::vertex<T> >& data, matrix<T>& W, bool verbose) 
    {
      // this is interface class that simply makes dataset<> calls work, does an costly data format transformation here
      // as the data types are not compatible (TODO: write code that uses DIRECTLY std::vector<> datasets

      if(data.size() <= 0) return false;

      matrix<T> DATA;

      DATA.resize(data.size(), data[0].size());

      for(unsigned int j=0;j<DATA.ysize();j++)
	for(unsigned int i=0;i<DATA.xsize();i++)
	  DATA(j, i) = data[j][i];

      return ica(DATA, W, verbose);
    }

    
    template <typename T>
    bool ica(const matrix<T>& D, matrix<T>& W, bool verbose) 
    {
      try{
	
	const unsigned int num = D.ysize();
	const unsigned int dim = D.xsize();
	
	// data MUST be already white (PCA preprocessed)
	const matrix<T>& X = D; // X = (V * D')'
	
	const T TOLERANCE = T(0.0001);
	const unsigned int MAXITERS = 1000;
	
	// matrix<T> W;
	W.resize(dim,dim);
	
	for(unsigned int j=0;j<dim;j++){
	  for(unsigned int i=0;i<dim;i++){
	    for(unsigned int k=0;k<W(j,i).size();k++){
	      float r = ((float)rand())/ ((float)RAND_MAX);
	      r = 2.0f*r -1.0f; // [-1,+1]
	      W(j,i)[k] = (r);
	    }
	  }
	}
	
	
	vertex<T> w;
	w.resize(dim);            
	
	// solves each IC separatedly (deflate method)
	for(unsigned int n=0;n<dim;n++){
	  
	  // initialization
	  W.rowcopyto(w, n);
	  
	  w.normalize();
	  __ica_project(w, n, W);
	  
	  bool convergence = 0;
	  unsigned int iter = 0;
	  vertex<T> w_old, y;
	  T scaling = T(1.0) / T(num);
	  
	  vertex<T> x; x.resize(dim);
	  vertex<T> xgy; xgy.resize(dim); // E[x * g(w * x)]
	  T dgy; // E[g'(w * x)]
	  
	  while(convergence == false){
	    
	    // updates w vector
	    w_old = w;
	    
	    y = X * w;

	    // FIXME add tanh non-linearity

	    if(0/*(iter % 2) == 0*/){ // g(u) = u^3 non-linearity
	      
	      for(unsigned int i=0;i<dim;i++) xgy[i] = T(0.0f);
	      dgy = T(0.0f);
	      
	      for(unsigned int i=0;i<num;i++){
		X.rowcopyto(x, i);

#if 1
		xgy += scaling * (y[i]*y[i]*y[i])*x;
		dgy += scaling * T(3.0f)*y[i]*y[i];
#endif

#if 0
		for(unsigned int k=0;k<y[i].size();k++){
		  for(unsigned int d=0;d<dim;d++)
		    xgy[d][k] += scaling[0] * (y[i][k]*y[i][k]*y[i][k])*x[d][k];
		  
		  dgy[k] += scaling[0] * T(3.0f)[0]*y[i][k]*y[i][k];
		}
#endif
	      }
	    }
	    else{ // g(u) u*exp(-u**2/2) non-linearity	    
	      
	      for(unsigned int i=0;i<dim;i++) xgy[i] = T(0.0);
	      dgy = T(0.0);
	      
	      for(unsigned int i=0;i<num;i++){
		X.rowcopyto(x, i);

#if 0
		for(unsigned int k=0;k<y[i].size();k++){
		  auto temp = whiteice::math::exp(-(y[i][k]*y[i][k])/(2.0));
		  
		  for(unsigned int d=0;d<dim;d++)
		    xgy[d][k] += scaling[0] * ((y[i][k] * temp) * x[d][k]);
		  
		  dgy[k] += scaling[0] * (temp - (y[i][k]*y[i][k])*temp);
		}
#endif

#if 1
		T temp = whiteice::math::exp(-(y[i]*y[i])/T(2.0));
		
		xgy += scaling * ((y[i] * temp) * x);
		dgy += scaling * (temp - (y[i]*y[i])*temp);
#endif
	      }
	    }

	    
	    w = xgy - dgy*w;

#if 0
	    for(unsigned int k=0;k<dgy.size();k++)
	      for(unsigned int d=0;d<dim;d++)
		w[d][k] = dgy[k]*w[d][k] - xgy[d][k];
#endif
	    
	    w.normalize();
	    
	    __ica_project(w, n, W); // projection
	    
 	    
	    // checks for convergence / stopping critearias
	    T dotprod = (w_old * w)[0];

	    if(verbose){

	      auto dot = dotprod[0];
	      T rest = T(0.0f);

	      for(unsigned int k=1;k<dotprod.size();k++)
		rest[0] += whiteice::math::abs(dotprod[k]);
	      
	      std::cout << "iter: " << iter << " dot: " << dot << " + " << rest[0] << std::endl;
	    }
	    
	    
	    if(iter >= 10){

	      auto value = T(+1.0) - dotprod;
	      unsigned int counter = 0;

	      for(unsigned int k=0;k<value.size();k++)
		if(whiteice::math::abs(value[k]) < TOLERANCE[0])
		  counter++;

	      if(counter >= value.size())
		convergence = true;

	      
	      //if((T(1.0) - dotprod) < TOLERANCE && iter > 10){
	      //convergence = true;
	    }
	    else if(iter >= MAXITERS){
	      break;
	    }

	    
	    
	    iter++;
	  }
	  
	  W.rowcopyfrom(w, n);
	  
	  if(convergence == 0){
	    if(verbose)
	      std::cout << "Warning: IC " << (n+1) << " didn't converge"
			<< std::endl;
	  }
	  else{
	    if(verbose)
	      std::cout << "IC " << (n+1) << " converged after " 
			<< iter << " iterations." << std::endl;
	  }
	}
	
	return true;
      }
      catch(std::exception& e){
	if(verbose)
	  std::cout << "uncaught exception: " << e.what() << std::endl;
	return false;
      }
    }
    
    
    // projects w to the remaining free subspace
    template <typename T>
    void __ica_project(vertex<T>& w, const unsigned int n, const matrix<T>& W)
    {
      vertex<T> s(w.size()); // s = (initialized to be) zero vector
      vertex<T> t(w.size());

      s.zero();
      
      for(unsigned int i=0;i<n;i++){
	W.rowcopyto(t, i); // t = W(i,:)
	s += (t * w) * t;  // amount t basis vector w has
      }
      
      w -= s;
      w.normalize();
    }
    
    
  };
};
