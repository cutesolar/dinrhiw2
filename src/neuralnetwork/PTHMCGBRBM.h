/*
 * PTHMCGBRBM.h
 *
 *  Created on: 25.6.2015
 *      Author: Tomas
 */

#ifndef NEURALNETWORK_PTHMCGBRBM_H_
#define NEURALNETWORK_PTHMCGBRBM_H_

#include "PTHMCabstract.h"
#include "HMCGBRBM.h"

#include <memory>


namespace whiteice {

template <typename T>
class PTHMC_GBRBM: public PTHMC_abstract<T> {
public:
	PTHMC_GBRBM(unsigned int deepness, const std::vector< math::vertex<T> >& data,
			unsigned int numHiddenNodes, bool adaptive);

	virtual ~PTHMC_GBRBM();

	GBRBM<T> getRBM();

	// creates HMC_abstract<T> class with a new (PTHMC_abstract deletes it)
	// adaptive : should HMC use adaptive step length
	// storeSamples: should created HMC store old samples
	virtual std::shared_ptr< HMC_abstract<T> > newHMC(bool storeSamples, bool adaptive);

protected:
	const std::vector< math::vertex<T> >& data;
	const unsigned int numHiddenNodes;

};

  //extern template class PTHMC_GBRBM< float >;
  //extern template class PTHMC_GBRBM< double >;
  
  extern template class PTHMC_GBRBM< math::blas_real<float> >;
  extern template class PTHMC_GBRBM< math::blas_real<double> >;

} /* namespace whiteice */

#endif /* NEURALNETWORK_PTHMCGBRBM_H_ */
