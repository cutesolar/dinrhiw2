/*
 * independent component analysis
 */

#include "ica.h"
#include "matrix.h"
#include "vertex.h"
#include "eig.h"
#include "correlation.h"
#include "RNG.h"
#include "linear_ETA.h"

#include <stdlib.h>


namespace whiteice
{
  namespace math
  {
    
    template bool ica< blas_real<float> >
    (const matrix< blas_real<float> >& D, matrix< blas_real<float> >& W, bool verbose,
     const blas_real<float> tolerance, const unsigned int MAXITERS);
    template bool ica< blas_real<double> >
    (const matrix< blas_real<double> >& D, matrix< blas_real<double> >& W, bool verbose,
     const blas_real<double> tolerance, const unsigned int MAXITERS);

    template bool ica< blas_complex<float> >
    (const matrix< blas_complex<float> >& D, matrix< blas_complex<float> >& W, bool verbose,
     const blas_complex<float> tolerance, const unsigned int MAXITERS);
    
    template bool ica< blas_complex<double> >
    (const matrix< blas_complex<double> >& D, matrix< blas_complex<double> >& W, bool verbose,
     const blas_complex<double> tolerance, const unsigned int MAXITERS);

    template bool ica< superresolution<blas_real<float>, modular<unsigned int> > >
    (const matrix< superresolution<blas_real<float>, modular<unsigned int> > >& D,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution<blas_real<float>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);
    
    template bool ica< superresolution<blas_real<double>, modular<unsigned int> > >
    (const matrix< superresolution<blas_real<double>, modular<unsigned int> > >& D,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution<blas_real<double>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);

    template bool ica< superresolution<blas_complex<float>, modular<unsigned int> > >
    (const matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& D,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution<blas_complex<float>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);
    
    template bool ica< superresolution<blas_complex<double>, modular<unsigned int> > >
    (const matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& D,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution<blas_complex<double>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);
    
    

    template bool ica< blas_real<float> >
    (const std::vector< math::vertex< blas_real<float> > >& data, matrix< blas_real<float> >& W, bool verbose,
     const blas_real<float> tolerance, const unsigned int MAXITERS);
    template bool ica< blas_real<double> >
    (const std::vector< math::vertex< blas_real<double> > >& data, matrix< blas_real<double> >& W, bool verbose,
     const blas_real<double> tolerance, const unsigned int MAXITERS);
    
    template bool ica< blas_complex<float> >
    (const std::vector< math::vertex< blas_complex<float> > >& data, matrix< blas_complex<float> >& W, bool verbose,
     const blas_complex<float> tolerance, const unsigned int MAXITERS);
    template bool ica< blas_complex<double> >
    (const std::vector< math::vertex< blas_complex<double> > >& data, matrix< blas_complex<double> >& W, bool verbose,
     const blas_complex<double> tolerance, const unsigned int MAXITERS);


    template bool ica< superresolution<blas_real<float>, modular<unsigned int> > >
    (const std::vector< math::vertex< superresolution<blas_real<float>, modular<unsigned int> > > >& data,
     matrix< superresolution<blas_real<float>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution<blas_real<float>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);
    
    template bool ica< superresolution<blas_real<double>, modular<unsigned int> > >
    (const std::vector< math::vertex< superresolution<blas_real<double>, modular<unsigned int> > > >& data,
     matrix< superresolution<blas_real<double>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution<blas_real<double>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);
    
    template bool ica< superresolution<blas_complex<float>, modular<unsigned int> > >
    (const std::vector< math::vertex< superresolution<blas_complex<float>, modular<unsigned int> > > >& data,
     matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution<blas_complex<float>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);
    
    template bool ica< superresolution<blas_complex<double>, modular<unsigned int> > >
    (const std::vector< math::vertex< superresolution<blas_complex<double>, modular<unsigned int> > > >& data,
     matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& W,
     bool verbose,
     const superresolution<blas_complex<double>, modular<unsigned int> > tolerance, const unsigned int MAXITERS);
    
    
    
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
    bool ica(const std::vector< math::vertex<T> >& data, matrix<T>& W, bool verbose, const T tolerance, const unsigned int MAXITERS)
    {
      // this is interface class that simply makes dataset<> calls work, does an costly data format transformation here
      // as the data types are not compatible (TODO: write code that uses DIRECTLY std::vector<> datasets

      if(data.size() <= 0) return false;

      matrix<T> DATA;

      DATA.resize(data.size(), data[0].size());

      for(unsigned int j=0;j<DATA.ysize();j++)
	for(unsigned int i=0;i<DATA.xsize();i++)
	  DATA(j, i) = data[j][i];

      return ica(DATA, W, verbose, tolerance, MAXITERS);
    }

    
    template <typename T>
    bool ica(const matrix<T>& D, matrix<T>& W, bool verbose, const T tolerance, const unsigned int MAXITERS) 
   {
      try{

	if(MAXITERS == 0) return false;
	
	const unsigned int num = D.ysize();
	const unsigned int dim = D.xsize();
	
	// data MUST be already white (PCA preprocessed)
	const matrix<T>& X = D; // X = (V * D')'

	T TOLERANCE= T(0.0001);

	if(tolerance > T(0.0) && tolerance < T(1.0)){
	  TOLERANCE = tolerance;
	}
	else{
	  TOLERANCE = T(0.0001);
	}
	
	// const unsigned int MAXITERS = 1000;
	
	// matrix<T> W;
	W.resize(dim,dim);
	
	for(unsigned int j=0;j<dim;j++){
	  for(unsigned int i=0;i<dim;i++){
	    for(unsigned int k=0;k<W(j,i).size();k++){
	      float r = whiteice::rng.uniform().real();
	      r = 2.0f*r - 1.0f; // [-1,+1]
	      W(j,i)[k] = r;
	    }
	  }
	}

	
	whiteice::linear_ETA<float> eta;
	eta.start(0, dim);
	
	vertex<T> w;
	w.resize(dim);            
	
	// solves each IC separatedly (deflate method)
	for(unsigned int nn=0;nn<dim;nn++){
	  
	  // initialization
	  W.rowcopyto(w, nn);
	  
	  w.normalize();
	  __ica_project(w, nn, W);
	  
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

	    // restricts very big values causing havoc with bad data..
	    // [NOTE: does this break ICA algorithm sometimes?]
	    if(typeid(T) == typeid(whiteice::math::blas_real<float>)){
	      for(unsigned int i=0;i<y.size();i++){
		if(y[i] > T(1e10f)) y[i] = T(1e10f);
		else if(y[i] < T(-1e10f)) y[i] = T(-1e10f);
	      }
	    }
	    else if(typeid(T) == typeid(whiteice::math::blas_real<double>)){
	      for(unsigned int i=0;i<y.size();i++){
		if(y[i] > T(1e100)) y[i] = T(1e100);
		else if(y[i] < T(-1e100)) y[i] = T(-1e100);
	      }
	    }


	    if(typeid(T) == typeid(superresolution< blas_real<float>, modular<unsigned int> >) ||
	       typeid(T) == typeid(superresolution< blas_real<double>, modular<unsigned int> >)){

	      // needs special code to calculate Newton's iteration for superresolution numbers

	      // calculates gradient
	      vertex<T> grad; grad.resize(dim);
	      grad.zero();

	      // T exps2 = T(0.0f);

	      // calculates Hessian
	      matrix<T> H; H.resize(dim,dim);
	      H.zero();
	      

	      for(unsigned int i=0;i<num;i++){
		X.rowcopyto(x, i);

		T ss = T(-0.5);
		T frobenius_norm2 = T(0.0f);
		
		for(unsigned int k=0;k<y[i].size();k++)
		  frobenius_norm2 += y[i][k]*y[i][k];

		auto xy = x;
		xy.zero();
		
		for(unsigned int d=0;d<xy.size();d++)
		  for(unsigned int k=0;k<xy[d].size();k++)
		    xy[d] += x[d][k]*y[i][k];

		matrix<T> xx; xx.resize(dim,dim);
		xx.zero();
		for(unsigned int b=0;b<dim;b++)
		  for(unsigned int a=0;a<dim;a++)
		    for(unsigned int k=0;k<x[0].size();k++)
		      xx(b,a) += x[b][k]*x[a][k];

		grad += scaling*whiteice::math::exp(ss[0]*frobenius_norm2[0])*xy;

		H += scaling*whiteice::math::exp(ss[0]*frobenius_norm2[0])*
		  (-xy.outerproduct() + xx);
		//(T(1.0f) - y[i]*y[i])*x.outerproduct();

#if 0
		matrix<T> Z;
		Z.resize(x.size(), x[0].size());
		Z.zero();

		for(unsigned int d=0;d<dim;d++)
		  for(unsigned int k=0;k<x[d].size();k++)
		    Z(d,k)[0] = x[d][k];

		auto Zt = Z;
		Zt.transpose();

		T frobenius_norm2 = T(0.0f);

		for(unsigned int k=0;k<y[i].size();k++)
		  frobenius_norm2 += y[i][k]*y[i][k];

		auto ZZ = Z*Zt;
		auto vv = ZZ*w;

		grad += scaling*whiteice::math::exp(ss[0]*frobenius_norm2[0])*vv;

		// hessian

		H += scaling*whiteice::math::exp(ss[0]*frobenius_norm2[0])*(ZZ - vv.outerproduct());
#endif
	      }

#if 1
	      auto INV = H;
	      const float epsilon = (1e-9f);
	      unsigned int tries = 0;

	      // regularize matrix by adding to diagonal and try to invert matrix..
	      while(INV.inv() == false && tries < 60){
		INV = H;

		for(unsigned int i=0;i<H.xsize()&&i<H.ysize();i++){
		  INV(i,i) += T( epsilon * math::pow(2.0f, (float)tries) );
		}

		tries++;
	      }

	      if(tries >= 60)
		return false; // ICA can fail..

	      H = INV;
	      
	      // assert(H.inv() == true); // TODO: regularize if matrix is singular
	      // assert(H.pseudoinverse() == true); // TODO: regularize if matrix is singular

	      T alpha = T(1.0); // was: 1.0

	      w = w - alpha*(H*grad);
#endif

	      // w = w - grad/exps2;
	      
	    }
	    else{

	      // auto w_orig = w;

	      // calculates mean E[w] over different ICA non-linearities [seem to work better]
	      //const float p = 1.0f; // whiteice::rng.uniform().real();

	      if((iter % 2) == 0)
	      { // g(u) = u^3 non-linearity
		
		for(unsigned int i=0;i<dim;i++) xgy[i] = T(0.0f);
		dgy = T(0.0f);
		
		for(unsigned int i=0;i<num;i++){
		  X.rowcopyto(x, i);
		  
		  xgy += scaling * (y[i]*y[i]*y[i])*x;
		  dgy += scaling * T(3.0f)*y[i]*y[i];
		}

		// w += (xgy - dgy*w_orig)*T(p);
	      }
	      else
	      { // g(u) u*exp(-u**2/2) non-linearity	    
		
		for(unsigned int i=0;i<dim;i++) xgy[i] = T(0.0);
		dgy = T(0.0);
		
		for(unsigned int i=0;i<num;i++){
		  X.rowcopyto(x, i);
		  
		  T temp = whiteice::math::exp(-(y[i]*y[i])/T(2.0));
		  
		  xgy += scaling * ((y[i] * temp) * x);
		  dgy += scaling * (temp - (y[i]*y[i])*temp);
		}

		// w += (xgy - dgy*w_orig)*T(1-p);
	      }

	      // update w
	      w = xgy - dgy*w;
	      
	    }

	    
	    if(w.normalize() == false){ // handles zero vector..
	      const T len = T(sqrt(w.size()));
	      
	      for(unsigned int i=0;i<w.size();i++){
		w[i] = T(1.0f)/len;
	      }
	    }

	    // std::cout << "ica w = " << w << std::endl;
	    
	    __ica_project(w, nn, W); // projection
	    
 	    
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
	      // WE HAVE CONVERGED IF DOT PRODUCT IS +1 OR -1 (direction of w does not change)

	      auto value = T(+1.0) - dotprod;
	      unsigned int counter = 0;

	      for(unsigned int k=0;k<value.size();k++)
		if(whiteice::math::abs(value[k]) < TOLERANCE[0])
		  counter++;

	      if(counter >= value.size())
		convergence = true;

	      
	      value = T(-1.0) - dotprod;
	      counter = 0;

	      for(unsigned int k=0;k<value.size();k++)
		if(whiteice::math::abs(value[k]) < TOLERANCE[0])
		  counter++;

	      if(counter >= value.size())
		convergence = true;

	      
	      //if((T(1.0) - dotprod) < TOLERANCE && iter > 10){
	      //convergence = true;
	    }
	    if(iter >= MAXITERS){
	      break;
	    }

	    
	    
	    iter++;
	  }
	  
	  W.rowcopyfrom(w, nn);

	  eta.update(nn+1);
	  
	  if(convergence == 0){
	    if(verbose)
	      std::cout << "Warning: IC " << (nn+1) << " didn't converge"
			<< std::endl;
	  }
	  else{
	    if(verbose)
	      std::cout << "IC " << (nn+1) << " converged after " 
			<< iter << " iterations." << std::endl;
	  }

	  if(verbose){
	    printf("ETA: %f hour(s)\n", eta.estimate()/3600.0f);
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
