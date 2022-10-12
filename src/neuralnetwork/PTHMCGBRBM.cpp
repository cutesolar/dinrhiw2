/*
 * PTHMCGBRBM.cpp
 *
 *  Created on: 25.6.2015
 *      Author: Tomas
 */

#include "PTHMCGBRBM.h"
#include "vertex.h"

#include <exception>

namespace whiteice {

template <typename T>
PTHMC_GBRBM<T>::PTHMC_GBRBM(unsigned int deepness_, const std::vector< math::vertex<T> >& data_,
		unsigned int numHiddenNodes_, bool adaptive_) :
	PTHMC_abstract<T>(deepness_, adaptive_), data(data_), numHiddenNodes(numHiddenNodes_)
{

}

template <typename T>
PTHMC_GBRBM<T>::~PTHMC_GBRBM() {
	// nothing to do in this simple class..
}


template <typename T>
GBRBM<T> PTHMC_GBRBM<T>::getRBM()
{
	std::lock_guard<std::mutex> lock(this->sampler_lock);

	if(this->hmc.size() > 0){
		// TODO do C++ style smart casting
		HMC_abstract<T>* ptr = (this->hmc.front().get());
		HMC_GBRBM<T>* ptr2   = (HMC_GBRBM<T>*)ptr;
		return ptr2->getRBM();
	}
	else{
		throw std::logic_error("getRBM(): Illegal operation");
	}
}


// creates HMC_abstract<T> class with a new (PTHMC_abstract deletes it)
// adaptive : should HMC use adaptive step length
// storeSamples: should created HMC store old samples
template <typename T>
std::shared_ptr< HMC_abstract<T> > PTHMC_GBRBM<T>::newHMC(bool storeSamples, bool adaptive)
{
	// HMC_GBRBM(const std::vector< math::vertex<T> >& samples, unsigned int numHiddenNodes, bool adaptive=false);
	return std::make_shared< HMC_GBRBM<T> >(data, numHiddenNodes, storeSamples, adaptive);
}

  //template class PTHMC_GBRBM< float >;
  //template class PTHMC_GBRBM< double >;
  
  template class PTHMC_GBRBM< math::blas_real<float> >;
  template class PTHMC_GBRBM< math::blas_real<double> >;


} /* namespace whiteice */
