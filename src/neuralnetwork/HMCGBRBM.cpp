/*
 * HMCGBRBM.cpp
 *
 *  Created on: 25.6.2015
 *      Author: Tomas
 */

#include <HMCGBRBM.h>

namespace whiteice {

template <typename T>
HMC_GBRBM<T>::HMC_GBRBM(const std::vector< math::vertex<T> >& data_, unsigned int numHiddenNodes, bool storeSamples, bool adaptive) :
	HMC_abstract<T>(storeSamples, adaptive), data(data_)
{
	if(data.size() > 0 && numHiddenNodes > 0){
		rbm.resize(data[0].size(), numHiddenNodes);
	}

	rbm.setUData(data);
	this->setTemperature(T(1.0));
}

template <typename T>
HMC_GBRBM<T>::~HMC_GBRBM()
{

}


template <typename T>
GBRBM<T> HMC_GBRBM<T>::getRBM() {
	return this->rbm;
}


template <typename T>
bool HMC_GBRBM<T>::setTemperature(T temperature) // temperature must be in [0,1] interval
{
	if(temperature >= T(0.0) && temperature <= T(1.0)){
		rbm.setUTemperature(temperature);
		return true;
	}
	else{
		return false;
	}
}


template <typename T>
T HMC_GBRBM<T>::getTemperature()
{
	return rbm.getUTemperature();
}


// probability functions for hamiltonian MC sampling of
// P ~ exp(-U(q)) distribution
template <typename T>
T HMC_GBRBM<T>::U(const math::vertex<T>& q) const
{
	if(data.size() > 0){
		return rbm.U(q);
	}
	else{
		std::cout << "HMC_GBRBM:U() error: data.size() == 0" << std::endl;
		return T(INFINITY); // P = 0
	}
}

template <typename T>
T HMC_GBRBM<T>::Udiff(const math::vertex<T>& q1, const math::vertex<T>& q2) const
{
	if(data.size() > 0){
		return rbm.Udiff(q1, q2);
	}
	else{
		std::cout << "HMC_GBRBM:U() error: data.size() == 0" << std::endl;
		return T(0.0); // zero difference
	}
}


template <typename T>
math::vertex<T> HMC_GBRBM<T>::Ugrad(const math::vertex<T>& q) const
{
	if(data.size() > 0){
		return rbm.Ugrad(q);
	}
	else{
		std::cout << "HMC_GBRBM:U() error: data.size() == 0" << std::endl;

		math::vertex<T> g(rbm.qsize());
		g.zero();
		return g;
	}

}

// a starting point q for the sampler (may not be random)
template <typename T>
void HMC_GBRBM<T>::starting_position(math::vertex<T>& q) const
{
	GBRBM<T> rbm;

	rbm.resize(this->rbm.getVisibleNodes(), this->rbm.getHiddenNodes());

	rbm.initializeWeights();

	math::matrix<T> W;
	math::vertex<T> a;
	math::vertex<T> b;
	math::vertex<T> z;

	math::vertex<T> var;

	rbm.getParameters(W, a, b, var);
	rbm.getLogVariance(z);

	rbm.convertParametersToQ(W, a, b, z, q);
}



template class HMC_GBRBM< float >;
template class HMC_GBRBM< double >;
template class HMC_GBRBM< math::blas_real<float> >;
template class HMC_GBRBM< math::blas_real<double> >;

} /* namespace whiteice */
