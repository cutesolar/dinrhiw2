
#include "fastpca.h"
#include "correlation.h"


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
		 std::vector<T>& eigenvalues)
    {
      if(data.size() == 0) return false;
      if(data[0].size() < dimensions) return false;
      if(dimensions == 0) return false;
      
      // TODO: compute eigenvectors directly into PCA matrix

      math::vertex<T> m;
      math::matrix<T> Cxx;
      
      m = data[0];
      m.zero();
      
      if(m.size() <= 100){
	if(mean_covariance_estimate(m, Cxx, data) == false)
	  return false;
      }
      else{
	// calculates Cxx live and only precalculates mean value m

	for(unsigned int i=0;i<data.size();i++)
	  m += data[i];
	
	m /= T(data.size());
      }
      
      
      std::vector< math::vertex<T> > pca; // pca vectors
      
      while(pca.size() < dimensions){
	math::vertex<T> gprev;
	math::vertex<T> g;
	g.resize(m.size());
	gprev.resize(m.size());
	
	for(unsigned int i=0;i<g.size();i++){
	  gprev[i] = T(2.0f*(float)rand()/((float)RAND_MAX) - 1.0f); // [-1,1]
	  g[i] = T(2.0f*(float)rand()/((float)RAND_MAX) - 1.0f); // [-1,1]
	}
	
	g.normalize();
	gprev.normalize();
	
	T convergence = T(1.0);
	T epsilon = T(10e-5);
	
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
	      g += delta*(delta*tmp);
	    }

	    g /= T(data.size());
	  }
	  
	  // orthonormalizes g
	  {
	    auto t = g;
	    
	    for(auto& p : pca){
	      T s = (t*p)[0];
	      g -= p*s;
	    }
	    
	    g.normalize();
	  }
	  
	  convergence = whiteice::math::abs(T(1.0f) - (g*gprev)[0]);
	  
	  gprev = g;
	  
	  iters++;

	  if(iters > 50){
	    if(convergence > epsilon || iters >= 200)
	      break;
	  }
	}

	
	if(iters >= 200)
	  std::cout << "WARN: fastpca maximum number of iterations reached without convergence." << std::endl;
	
	pca.push_back(g);
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
	  eigenvalues.push_back((p*Cxx*p)[0]);
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
	    auto squared = p*delta;
	    
	    eigenvalues[i] += (squared*squared)[0];
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
		   std::vector<T>& eigenvalues)
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
      
      if(m.size() <= 100){
	if(mean_covariance_estimate(m, Cxx, data) == false)
	  return false;

	for(unsigned int i=0;i<Cxx.xsize();i++)
	  total_variance += Cxx(i,i);
      }
      else{
	// calculates Cxx live and only precalculates mean value m

	for(const auto& d : data)
	  m += d;
	
	m /= T(data.size());

	// calculates total variance

	for(const auto& d : data){
	  auto delta = d - m;
	  total_variance += (delta*delta)[0];
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
	
	for(unsigned int i=0;i<g.size();i++){
	  gprev[i] = T(2.0f*(float)rand()/((float)RAND_MAX) - 1.0f); // [-1,1]
	  g[i] = T(2.0f*(float)rand()/((float)RAND_MAX) - 1.0f); // [-1,1]
	}
	
	g.normalize();
	gprev.normalize();
	
	T convergence = T(1.0);
	T epsilon = T(10e-5);
	
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
	      g += delta*(delta*tmp);
	    }

	    g /= T(data.size());
	  }
	  
	  
	  // orthonormalizes g against already found components
	  {
	    auto t = g;
	    
	    for(auto& p : pca){
	      T s = (t*p)[0];
	      g -= s*p;
	    }
	    
	    g.normalize();
	  }
	  
	  convergence = whiteice::math::abs(T(1.0f) - (g*gprev)[0]);
	  
	  gprev = g;
	  
	  iters++;

	  if(iters > 50){
	    if(convergence > epsilon || iters >= 200)
	      break;
	  }
	}
	
	
	if(iters >= 200)
	  std::cout << "WARN: fastpca maximum number of iterations reached without convergence." << std::endl;

	// calculate variance of the found component
	{
	  T mean = (g*m)[0];
	  T var  = T(0.0f);

	  for(const auto& d : data){
	    const auto x = (g*d)[0];
	    var += (x - mean)*(x-mean);
	  }

	  var /= T(data.size());

	  variance_found += var;
	}
	
	pca.push_back(g);
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
	  eigenvalues.push_back((p*Cxx*p)[0]);
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
	    auto squared = p*delta;
	    
	    eigenvalues[i] += (squared*squared)[0];
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
     std::vector< blas_real<float> >& eigenvalues);
    
    template bool fastpca< blas_real<double> >
    (const std::vector< vertex< blas_real<double> > >& data, 
     const unsigned int dimensions,
     math::matrix< blas_real<double> >& PCA,
     std::vector< blas_real<double> >& eigenvalues);


    template bool fastpca< superresolution< blas_real<float>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution< blas_real<float>, modular<unsigned int> > > >& data, 
     const unsigned int dimensions,
     math::matrix< superresolution< blas_real<float>, modular<unsigned int> > >& PCA,
     std::vector< superresolution< blas_real<float>, modular<unsigned int> > >& eigenvalues);
    
    template bool fastpca< superresolution< blas_real<double>, modular<unsigned int> > >
    (const std::vector< vertex< superresolution< blas_real<double>, modular<unsigned int> > > >& data, 
     const unsigned int dimensions,
     math::matrix< superresolution< blas_real<double>, modular<unsigned int> > >& PCA,
     std::vector< superresolution< blas_real<double>, modular<unsigned int> > >& eigenvalues);
    
    
    
    
    template bool fastpca_p< blas_real<float> >
    (const std::vector <vertex< blas_real<float> > >& data,
     const float percent_total_variance,
     math::matrix< blas_real<float> >& PCA,
     std::vector< blas_real<float> >& eigenvalues);
    
    template bool fastpca_p< blas_real<double> >
    (const std::vector <vertex< blas_real<double> > >& data,
     const float percent_total_variance,
     math::matrix< blas_real<double> >& PCA,
     std::vector< blas_real<double> >& eigenvalues);
    
  };
  
};
