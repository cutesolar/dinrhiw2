
#include "LBFGS_GBRBM.h"
#include "deep_ica_network_priming.h"

namespace whiteice
{

  template <typename T>
  LBFGS_GBRBM<T>::LBFGS_GBRBM(const whiteice::GBRBM<T>& nn, const dataset<T>& d, bool overfit) :
    whiteice::math::LBFGS<T>(overfit), net(nn), data(d)
  {
    // IMPORTANT incoming net must have setUData() set to proper samples!
    
    /*
    if(data.getNumberOfClusters() > 0){
      std::vector< math::vertex<T> > data;
      data.getData(0, data);
      net.setUData(data);
    }
    */

    this->setGradientOnly(true); // DEBUG: only follow gradient!
  }

  
  template <typename T>
  LBFGS_GBRBM<T>::~LBFGS_GBRBM()
  {
  }


  template <typename T>
  T LBFGS_GBRBM<T>::getError(const math::vertex<T>& x) const
  {
    // return -this->net.U(x);
    
    T e = T(0.0f);
    
#pragma omp parallel shared(e)
    {
      whiteice::GBRBM<T> nnet(this->net);
      nnet.setParametersQ(x);
      
      math::vertex<T> err;
      T esum = T(0.0f);
      
      
      // E = SUM ||e(i)||/dim(e)
      #pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<1000;i++){
	// std::cout << "data in  = " << dtest.access(0, i) << std::endl;
	// std::cout << "data out = " << dtest.access(1, i) << std::endl;
	unsigned int index = rand()%data.size(0);

	auto v = data.access(0, index);

	nnet.setVisible(v);
	nnet.reconstructData(1);
	nnet.getVisible(v);
	err = data.access(0, index) - v;
	
	err = err.norm()/T(err.size());
	esum += err[0];
      }

#pragma omp critical (rewowjogrjoajrqoecd)
      {
	e += esum;
      }

    }
    
    e /= T( (double)1000 ); // per N
    
    return e;
  }
  

  template <typename T>
  T LBFGS_GBRBM<T>::U(const math::vertex<T>& x) const
  {
    return this->getError(x);
    // return -this->net.U(x);
  }

  
  template <typename T>
  math::vertex<T> LBFGS_GBRBM<T>::Ugrad(const math::vertex<T>& x) const
  {
    whiteice::GBRBM<T> gbrbm(this->net);
    
    return -gbrbm.Ugrad(x);
  }
  
  
  template <typename T>
  bool LBFGS_GBRBM<T>::heuristics(math::vertex<T>& x) const
  {
    return true; // does nothing..
    
  }
  
  
  template class LBFGS_GBRBM< float >;
  template class LBFGS_GBRBM< double >;
  template class LBFGS_GBRBM< math::blas_real<float> >;
  template class LBFGS_GBRBM< math::blas_real<double> >;

  
};
