
#include "fastpca.h"
#include "correlation.h"
#include "linear_ETA.h"
#include "RNG.h"

using namespace whiteice;


namespace whiteice
{
  namespace math
  {

    /*
     * Extracts first "dimensions" PCA vectors from data
     * PCA = X^t when Cxx = E{(x-m)(x-m)^t} = X*L*X^t
     */
    template <typename T>
    bool fastpca(const std::vector< vertex<T> >& data, 
		 const unsigned int dimensions,
		 math::matrix<T>& PCA,
		 std::vector<T>& eigenvalues,
		 const bool verbose)
    {
      if(data.size() == 0) return false;
      if(data[0].size() < dimensions) return false;
      if(dimensions == 0) return false;

      // TODO: compute eigenvectors directly into PCA matrix

      math::vertex<T> m;
      math::matrix<T> Cxx;
      
      m = data[0];
      m.zero();
      
      if(m.size() <= 1000 && data.size() > 30){

	if(verbose){
	  printf("Estimating Cxx matrix directly..\n");
	  fflush(stdout);
	}
	
	if(mean_covariance_estimate(m, Cxx, data) == false)
	  return false;
      }
      else{
	// calculates Cxx live and only precalculates mean value m

	if(verbose){
	  printf("Using statistical inner product with Cxx matrix/data (Cxx could be HUGE)..\n");
	  fflush(stdout);
	}

	for(const auto& di : data)
	  m += di;
	
	m /= T(data.size());
      }

      linear_ETA<float> eta;
      eta.start(0, dimensions);
      
      
      std::vector< math::vertex<T> > pca; // pca vectors
      
      while(pca.size() < dimensions){
	math::vertex<T> gprev;
	math::vertex<T> g;
	g.resize(m.size());
	gprev.resize(m.size());

	
	if(typeid(T) == typeid(superresolution< blas_complex<float> >) ||
	   typeid(T) == typeid(superresolution< blas_complex<double> >)){

	  for(unsigned int i=0;i<g.size();i++){
	    for(unsigned int j=0;j<g[i].size();j++){
	      for(unsigned int k=0;k<g[i][j].size();k++){
		gprev[i][j][k] = ((2.0f*rng.uniform().c[0]) - 1.0f); // [-1,1]
		g[i][j][k] = ((2.0f*rng.uniform().c[0]) - 1.0f); // [-1,1]
	      }
	    }
	  }
	  
	}
	else if(typeid(T) == typeid(superresolution< blas_real<float> >) ||
		typeid(T) == typeid(superresolution< blas_real<double> >)){

	  for(unsigned int i=0;i<g.size();i++){
	    for(unsigned int j=0;j<g[i].size();j++){
	      gprev[i][j] = ((2.0f*rng.uniform().c[0]) - 1.0f); // [-1,1]
	      g[i][j] = ((2.0f*rng.uniform().c[0]) - 1.0f); // [-1,1]
	    }
	  }
	  
	}
	else{
	  for(unsigned int i=0;i<g.size();i++){
	    gprev[i] = T((2.0f*rng.uniform().c[0]) - 1.0f); // [-1,1]
	    g[i] = T((2.0f*rng.uniform().c[0]) - 1.0f); // [-1,1]
	  }
	}
	
	g.normalize();
	gprev.normalize();
	
	T convergence = T(1.0);
	T epsilon = T(1e-2);
	
	if(typeid(T) == typeid(superresolution< blas_real<float>, modular<unsigned int> >) ||
	   typeid(T) == typeid(superresolution< blas_real<double>, modular<unsigned int> >) ||
	   typeid(T) == typeid(superresolution< blas_complex<float>, modular<unsigned int> >) ||
	   typeid(T) == typeid(superresolution< blas_complex<double>, modular<unsigned int> >))
	{
	  for(unsigned int i=1;i<epsilon.size();i++){
	    epsilon[i] = epsilon[0];
	  }
	}
	
	unsigned int iters = 0;

	while(true){
	  
	  if(Cxx.xsize() == m.size()){ // has Cxx
	    g = Cxx*g;
	  }
	  else{
	    // calculates product without calculating Cxx matrix

	    const auto tmp = g;
	    g.zero();

	    // parallel threaded code for calculating matrix product with large amount of data
#pragma omp parallel shared(g)
	    {
	      auto delta = m;
	      auto gcopy = g;

#pragma omp for nowait schedule(auto)
	      for(unsigned int i=0;i<data.size();i++){
		delta = (data[i]-m);
		auto cdelta = delta;
		cdelta.conj();
		gcopy += delta*(cdelta*tmp);
	      }

#pragma omp critical (fast_pca_fhwuifhwu)
	      {
		g += gcopy;
	      }
	    }
	    
	    //for(const auto& di : data){
	    //  delta = (di - m);
	    //  g += delta*(delta*tmp);
	    //}

	    g /= T(data.size());
	  }
	  
	  // orthonormalizes g
	  {
	    auto t = g;
	    
	    for(auto& p : pca){
	      auto cp = p;
	      cp.conj();
	      T s = (t*cp)[0];
	      g -= p*s;
	    }
	    
	    g.normalize();
	  }

	  gprev.conj();
	  auto dot = (g*gprev)[0];

	  if(typeid(T) == typeid(superresolution< blas_real<float>, modular<unsigned int> >) ||
	     typeid(T) == typeid(superresolution< blas_real<double>, modular<unsigned int> >) ||
	     typeid(T) == typeid(superresolution< blas_complex<float>, modular<unsigned int> >) ||
	     typeid(T) == typeid(superresolution< blas_complex<double>, modular<unsigned int> >))
	  {
	    // need to still take vector inner product of the superresolution number elements
	    // this is equal to 1 in convergence!! (so inner product space makes sense too with
	    // superresolutional numbers!)
	    auto p = dot;
	    p.zero();
	    
	    for(unsigned int i=0;i<dot.size();i++){
	      p[0] += dot[i]*whiteice::math::conj(dot[i]);
	    }

	    p[0] = whiteice::math::sqrt(p[0]);
	    dot = p;

	    convergence = whiteice::math::abs(T(1.0f) - dot);
	    
	    // std::cout << "convergence = " << p << std::endl;
	    if(verbose){
	      std::cout << "Iteration " << iters << " convergence: " << convergence[0] << std::endl;
	    }
	    
	    
	    gprev = g;
	    
	    iters++;
	    
	    if(iters > 10){
	      if(convergence[0].real() < epsilon[0].real() || iters >= 250)
		break;
	    }

	  }
	  else{
	  
	    convergence = whiteice::math::abs(T(1.0f) - dot);
	    
	    if(verbose){
	      std::cout << "Iteration " << iters << " convergence: " << convergence << std::endl;
	    }
	    
	    gprev = g;
	    
	    iters++;
	    
	    if(iters > 10){
	      if(convergence < epsilon || iters >= 250)
		break;
	    }
	    
	  }
	}

	
	if(iters >= 250)
	  std::cout << "WARN: fastpca maximum number of iterations reached without convergence." << std::endl;
	
	pca.push_back(g);

	eta.update(pca.size());

	if(verbose){
	  printf("PCA %d/%d: ETA %f hour(s)\n", (int)pca.size(), dimensions,
		 eta.estimate()/3600.0f);
	  fflush(stdout);
	}
	
      }
      
      PCA.resize(pca.size(), data[0].size());
      
      auto j = 0;
      for(auto& p : pca){
	PCA.rowcopyfrom(p, j);
	j++;
      }

      eigenvalues.clear();

      // computes eigenvalues
      if(Cxx.xsize() == m.size()){ // has Cxx
	
	for(auto& p : pca){
	  auto cp = p;
	  cp.conj();
	  eigenvalues.push_back((cp*Cxx*p)[0]);
	}
	
      }
      else{ // no Cxx, need to estimate from the data

	eigenvalues.resize(pca.size());
	
	for(unsigned int i=0;i<pca.size();i++)
	  eigenvalues[i] = T(0.0f);
	
#pragma omp parallel
	{
	  auto delta = m;
	  
	  std::vector<T> e;
	  e.resize(eigenvalues.size());
	  for(auto& ei : e) ei = T(0.0f);

#pragma omp for schedule(auto) nowait
	  for(unsigned int d=0;d<data.size();d++){
	    const auto& di = data[d];
	    delta = (di - m);
	    
	    for(unsigned int i=0;i<pca.size();i++){
	      const auto& p = pca[i];
	      auto cp = p;
	      auto cdelta = delta;
	      cp.conj();
	      cdelta.conj();
	      
	      //auto squared = p*delta;
	      //eigenvalues[i] += (squared*squared)[0];
	      
	      e[i] += ((cp*delta)*(cdelta*p))[0];
	    }
	  }

#pragma omp critical
	  {
	    for(unsigned int i=0;i<eigenvalues.size();i++)
	      eigenvalues[i] += e[i];
	  }
	}

	
	for(unsigned int i=0;i<pca.size();i++)
	  eigenvalues[i] /= T(data.size());
	
      }
      
      
      return (pca.size() > 0);
    }


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
		   std::vector<T>& eigenvalues,
		   const bool verbose)
    {
      if(percent_total_variance <= 0.0f ||
	 percent_total_variance > 1.0f)
	return false;
      
      if(data.size() == 0) return false;
      if(data[0].size() == 0) return false;
      const unsigned int dimensions = data[0].size();
      
      // TODO: compute eigenvectors directly into PCA matrix


      math::vertex<T> m;
      math::matrix<T> Cxx;

      m = data[0];
      m.zero();
      
      // trace(Cxx) is total variance of eigenvectors
      T total_variance = T(0.0f);
      
      if(m.size() <= 1000 && data.size() > 30){

	if(verbose){
	  printf("Estimating Cxx matrix directly..\n");
	  fflush(stdout);
	}
	
	if(mean_covariance_estimate(m, Cxx, data) == false)
	  return false;

	for(unsigned int i=0;i<Cxx.xsize();i++)
	  total_variance += Cxx(i,i);
      }
      else{
	// calculates Cxx live and only precalculates mean value m

	if(verbose){
	  printf("Using statistical inner product with Cxx matrix/data (Cxx would be HUGE)..\n");
	  fflush(stdout);
	}

	for(const auto& d : data)
	  m += d;
	
	m /= T(data.size());

	// calculates total variance

	for(const auto& d : data){
	  auto delta = d - m;
	  auto cdelta = delta;
	  cdelta.conj();
	  total_variance += (delta*cdelta)[0];
	}

	total_variance /= T(data.size());
      }

      

      const T target_variance = T(percent_total_variance)*total_variance;
      T variance_found = T(0.0f);
      
      std::vector< math::vertex<T> > pca; // pca vectors
      
      while(pca.size() < dimensions && variance_found < target_variance){
	math::vertex<T> gprev;
	math::vertex<T> g;
	g.resize(m.size());
	gprev.resize(m.size());

	if(typeid(T) == typeid(superresolution< blas_complex<float> >) ||
	   typeid(T) == typeid(superresolution< blas_complex<double> >)){
	  
	  for(unsigned int i=0;i<g.size();i++){
	    for(unsigned int j=0;j<g[i].size();j++){
	      gprev[i][j] = ((2.0f*rng.uniform().c[0]) - 1.0f); // [-1,1]
	      g[i][j] = ((2.0f*rng.uniform().c[0]) - 1.0f); // [-1,1]
	    }
	  }
	  
	}
	else{
	  for(unsigned int i=0;i<g.size();i++){
	    gprev[i] = T((2.0f*rng.uniform().c[0]) - 1.0f); // [-1,1]
	    g[i] = T((2.0f*rng.uniform().c[0]) - 1.0f); // [-1,1]
	  }
	}
	
	g.normalize();
	gprev.normalize();
	
	T convergence = T(1.0);
	T epsilon = T(1e-2);
	
	if(typeid(T) == typeid(superresolution< blas_real<float>, modular<unsigned int> >) ||
	   typeid(T) == typeid(superresolution< blas_real<double>, modular<unsigned int> >))
	{
	  for(unsigned int i=1;i<epsilon.size();i++){
	    epsilon[i] = epsilon[0];
	  }
	}
	
	unsigned int iters = 0;
	

	while(1){
	  
	  if(Cxx.xsize() == m.size()){ // has Cxx
	    g = Cxx*g;
	  }
	  else{
	    // calculates product without calculating Cxx matrix

	    const auto tmp = g;
	    auto delta = m;
	    g.zero();

	    for(const auto& di : data){
	      delta = (di - m);
	      auto cdelta = delta;
	      cdelta.conj();
	      g += delta*(cdelta*tmp);
	    }

	    g /= T(data.size());
	  }
	  
	  
	  // orthonormalizes g against already found components
	  {
	    auto t = g;
	    
	    for(auto& p : pca){
	      auto cp = p;
	      cp.conj();
	      T s = (t*cp)[0];
	      g -= s*p;
	    }
	    
	    g.normalize();
	  }

	  gprev.conj();
	  auto dot = (g*gprev)[0];

	  if(typeid(T) == typeid(superresolution< blas_real<float>, modular<unsigned int> >) ||
	     typeid(T) == typeid(superresolution< blas_real<double>, modular<unsigned int> >) ||
	     typeid(T) == typeid(superresolution< blas_complex<float>, modular<unsigned int> >) ||
	     typeid(T) == typeid(superresolution< blas_complex<double>, modular<unsigned int> >))
	  {
	    // need to still take vector inner product of the superresolution number elements
	    // this is equal to 1 in convergence!! (so inner product space makes sense too with
	    // superresolutional numbers!)
	    auto p = dot;
	    p.zero();
	    
	    for(unsigned int i=0;i<dot.size();i++){
	      p[0] += dot[i]*whiteice::math::conj(dot[i]);
	    }
	    
	    p[0] = whiteice::math::sqrt(p[0]);
	    dot = p;
	  }
	  
	  convergence = whiteice::math::abs(T(1.0f) - dot);
	  
	  gprev = g;
	  
	  iters++;

	  if(iters > 10){
	    if(convergence[0].real() < epsilon[0].real() || iters >= 250)
	      break;
	  }
	}
	
	
	if(iters >= 250)
	  std::cout << "WARN: fastpca maximum number of iterations reached without convergence." << std::endl;

	// calculate variance of the found component [
	{
	  auto cg = g;
	  cg.conj();
	  
	  T mean = (cg*m)[0];
	  T var  = T(0.0f);

	  for(const auto& d : data){
	    const auto x = (cg*d)[0];
	    auto cdelta = x-mean;
	    cdelta.conj();
	    var += (x - mean)*cdelta;
	  }

	  var /= T(data.size());

	  variance_found += var;
	}
	
	pca.push_back(g);

	if(verbose){
	  printf("PCA %d/%d dimension calculated.\n", (int)pca.size(), dimensions);
	  fflush(stdout);
	}
      }
      
      PCA.resize(pca.size(), data[0].size());
      
      auto j = 0;
      for(auto& p : pca){
	PCA.rowcopyfrom(p, j);
	j++;
      }

      
      eigenvalues.clear();
      
      
      // computes eigenvalues
      if(Cxx.xsize() == m.size()){ // has Cxx
	
	for(const auto& p : pca){
	  auto cp = p;
	  cp.conj();
	  eigenvalues.push_back((cp*Cxx*p)[0]);
	}
	
      }
      else{ // no Cxx, need to estimate from the data

	auto delta = m;
	
	eigenvalues.resize(pca.size());
	
	for(unsigned int i=0;i<pca.size();i++)
	  eigenvalues[i] = T(0.0f);
	
	
	for(const auto& di : data){
	  delta = (di - m);

	  for(unsigned int i=0;i<pca.size();i++){
	    const auto& p = pca[i];
	    auto cp = p;
	    auto cdelta = delta;
	    cp.conj();
	    cdelta.conj();
	    
	    //auto squared = p*delta;
	    //eigenvalues[i] += (squared*squared)[0];

	    eigenvalues[i] += ((cp*delta)*(cdelta*p))[0];
	  }
	}

	
	for(unsigned int i=0;i<pca.size();i++)
	  eigenvalues[i] /= T(data.size());
	
      }

      
      return (pca.size() > 0);
    }


    //////////////////////////////////////////////////////////////////////
    
    
    template bool fastpca< blas_real<float> >
    (const std::vector< vertex< blas_real<float> > >& data, 
     const unsigned int dimensions,
     math::matrix< blas_real<float> >& PCA,
     std::vector< blas_real<float> >& eigenvalues,
     const bool verbose);
    
    template bool fastpca< blas_real<double> >
    (const std::vector< vertex< blas_real<double> > >& data, 
     const unsigned int dimensions,
     math::matrix< blas_real<double> >& PCA,
     std::vector< blas_real<double> >& eigenvalues,
     const bool verbose);


    template bool fastpca< superresolution< blas_real<float>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution< blas_real<float>, modular<unsigned int> > > >& data, 
     const unsigned int dimensions,
     math::matrix< superresolution< blas_real<float>, modular<unsigned int> > >& PCA,
     std::vector< superresolution< blas_real<float>, modular<unsigned int> > >& eigenvalues,
     const bool verbose);
    
    template bool fastpca< superresolution< blas_real<double>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution< blas_real<double>, modular<unsigned int> > > >& data, 
     const unsigned int dimensions,
     math::matrix< superresolution< blas_real<double>, modular<unsigned int> > >& PCA,
     std::vector< superresolution< blas_real<double>, modular<unsigned int> > >& eigenvalues,
     const bool verbose);

    
    template bool fastpca< superresolution< blas_complex<float>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution< blas_complex<float>, modular<unsigned int> > > >& data, 
     const unsigned int dimensions,
     math::matrix< superresolution< blas_complex<float>, modular<unsigned int> > >& PCA,
     std::vector< superresolution< blas_complex<float>, modular<unsigned int> > >& eigenvalues,
     const bool verbose);
    
    template bool fastpca< superresolution< blas_complex<double>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution< blas_complex<double>, modular<unsigned int> > > >& data, 
     const unsigned int dimensions,
     math::matrix< superresolution< blas_complex<double>, modular<unsigned int> > >& PCA,
     std::vector< superresolution< blas_complex<double>, modular<unsigned int> > >& eigenvalues,
     const bool verbose);
    
    
    
    
    template bool fastpca_p< blas_real<float> >
    (const std::vector <vertex< blas_real<float> > >& data,
     const float percent_total_variance,
     math::matrix< blas_real<float> >& PCA,
     std::vector< blas_real<float> >& eigenvalues,
     const bool verbose);
    
    template bool fastpca_p< blas_real<double> >
    (const std::vector <vertex< blas_real<double> > >& data,
     const float percent_total_variance,
     math::matrix< blas_real<double> >& PCA,
     std::vector< blas_real<double> >& eigenvalues,
     const bool verbose);
    
  };
  
};
