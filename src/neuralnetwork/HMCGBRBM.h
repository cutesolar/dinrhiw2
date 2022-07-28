/*
 * HMCGBRBM.h
 *
 *  Created on: 25.6.2015
 *      Author: Tomas
 */

#ifndef NEURALNETWORK_HMCGBRBM_H_
#define NEURALNETWORK_HMCGBRBM_H_

#include "HMC_abstract.h"
#include "GBRBM.h"
#include "vertex.h"

namespace whiteice {

template <typename T = whiteice::math::blas_real<float> >
class HMC_GBRBM: public whiteice::HMC_abstract<T> {
public:
	HMC_GBRBM(const std::vector< math::vertex<T> >& data_, unsigned int numHiddenNodes, bool storeSamples, bool adaptive=false);
	virtual ~HMC_GBRBM();

	GBRBM<T> getRBM() ;

	// temperature must be in [0,1] interval
	// 0 = means most freely changing parameters and RBM is just N(m, S) machine
	// 1 = means to fit into data as well as possible and RBM is non-linear P(v) estimator
	bool setTemperature(T temperature);

	T getTemperature();

    // probability functions for hamiltonian MC sampling of
    // P ~ exp(-U(q)) distribution
    virtual T U(const math::vertex<T>& q) const;

    virtual T Udiff(const math::vertex<T>& q1, const math::vertex<T>& q2) const;

    virtual math::vertex<T> Ugrad(const math::vertex<T>& q) const;

    // a starting point q for the sampler (may not be random)
    virtual void starting_position(math::vertex<T>& q) const;

protected:
    const std::vector< math::vertex<T> >& data;

    GBRBM<T> rbm;

};


extern template class HMC_GBRBM< float >;
extern template class HMC_GBRBM< double >;
extern template class HMC_GBRBM< math::blas_real<float> >;
extern template class HMC_GBRBM< math::blas_real<double> >;

} /* namespace whiteice */

#endif /* NEURALNETWORK_HMCGBRBM_H_ */
