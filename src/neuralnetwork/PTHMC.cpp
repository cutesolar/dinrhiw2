/*
 * PTHMC.cpp
 *
 *  Created on: 18.6.2015
 *      Author: Tomas
 */

#include "PTHMC.h"
#include <mutex>

#include <exception>
#include <stdexcept>

#include <thread>
#include <chrono>

#include "RNG.h"

namespace whiteice {

template <typename T>
PTHMC<T>::PTHMC(unsigned int deepness, const whiteice::nnetwork<T>& net_, const whiteice::dataset<T>& ds_,
		bool adaptive_, T alpha_) : net(net_), ds(ds_), adaptive(adaptive_), alpha(alpha_)
{
	// sets used temperatures according to deepness T(k) = 2**k

	for(unsigned int k=0;k<deepness;k++)
		temperature.push_back(math::pow(T(1.25), T(k)));

	// 1.10 gives accept rate of 70% with a testcase
	// 1.25 gives accept rate of 45% with a testcase
	// 1.50 gives accept rate of <30%

	running = false;
	paused  = false;
	parallel_tempering_thread = nullptr;
}

template <typename T>
PTHMC<T>::~PTHMC(){
	stopSampler(); // if needed
}

template <typename T>
bool PTHMC<T>::startSampler()
{
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(running) return false; // already running

	for(auto h : hmc)
		delete h;
	hmc.clear();

	try{

		for(unsigned int i=0;i<temperature.size();i++){
			HMC<T>* h = new HMC<T>(net, ds, adaptive, alpha, i == 0); // only store samples of the "bottom" sampler [no temperature]
			h->setTemperature(temperature[i]);
			hmc.push_back(h);
		}

		for(auto h : hmc)
			if(h->startSampler() == false)
				throw std::runtime_error("Cannot start sampler");
	}
	catch(std::exception& e){
		for(auto h : hmc){
			h->stopSampler();
			delete h;
		}

		hmc.clear();

		return false;
	}

	// starts parallel tempering thread which swaps samples between different temperature samplers
	try{
		running = true;
		paused = false;
		parallel_tempering_thread = new std::thread(parallel_tempering, this);
	}
	catch(std::exception& e){
		for(auto h : hmc){
			h->stopSampler();
			delete h;
		}

		hmc.clear();
	}

	return true;
}

template <typename T>
bool PTHMC<T>::pauseSampler(){
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0) return false; // not running

	bool ok = true;

	for(auto h : hmc){
		ok = ok && h->pauseSampler();
	}

	paused = true;

	return ok;
}

template <typename T>
bool PTHMC<T>::continueSampler(){
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0) return false; // not running

	bool ok = true;

	for(auto h : hmc){
		ok = ok && h->continueSampler();
	}

	paused = false;

	return ok;
}

template <typename T>
bool PTHMC<T>::stopSampler(){
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0) return false; // not running

	running = false;
	paused = false;

	for(auto h : hmc){
		h->stopSampler();
		delete h;
	}

	hmc.clear();

	parallel_tempering_thread->join();
	delete parallel_tempering_thread;
	parallel_tempering_thread = nullptr;

	return true;
}

template <typename T>
unsigned int PTHMC<T>::getSamples(std::vector< math::vertex<T> >& samples) const{
	return hmc[0]->getSamples(samples);
}

template <typename T>
unsigned int PTHMC<T>::getNumberOfSamples() const{
	return hmc[0]->getNumberOfSamples();
}

template <typename T>
bool PTHMC<T>::getNetwork(bayesian_nnetwork<T>& bnn){
	return hmc[0]->getNetwork(bnn);
}

template <typename T>
math::vertex<T> PTHMC<T>::getMean() const{
	return hmc[0]->getMean();
}


template <typename T>
T PTHMC<T>::getMeanError(unsigned int latestN) const
{
	return hmc[0]->getMeanError(latestN);
}


template <typename T>
void PTHMC<T>::parallel_tempering()
{
	double hz = 1.0;
	unsigned int ms = (unsigned int)(1000.0/hz);

	accepts = 0;
	total_tries = 0;

	// on average we try to jump between every 10 samples so we measure
	// how long it seems take to get 10 samples
	unsigned int delay = 0;
	while(hmc[0]->getNumberOfSamples() < 10 && running){
		std::this_thread::sleep_for(std::chrono::milliseconds(100)); // waits for 100ms
		delay += 100;
	}

	ms = delay;
	hz = 1000.0/ms;


	while(running){
		// pauses sampling for each thread
		// for(auto h : hmc) h->pauseSampler();

		for(int i=hmc.size()-1;i>=1;i--){
			// tries to do MCMC swap between samples, starts from higher T samplers and moves to lower ones
			math::vertex<T> w1, w2;

			hmc[i]->getCurrentSample(w1);
			hmc[i-1]->getCurrentSample(w2);

			const T T1 = temperature[i];
			const T T2 = temperature[i-1];

			T E1 = T1 * (hmc[i]->U(w1, false));
			T E2 = T2 * (hmc[i-1]->U(w2, false));

			T p = math::exp( (E1 - E2)*(T(1.0)/T1 - T(1.0)/T2) );

			if(T(whiteice::rng.uniformf()) < p){
				accepts++;
/*
				if(i == 1){
					std::cout << "SWAP p = " << p << std::endl;
					fflush(stdout);
				}
*/
				hmc[i-1]->setCurrentSample(w2);
				hmc[i]->setCurrentSample(w1);
			}

			total_tries++;
		}

		// continue sampling in each thread
		// for(auto h : hmc) h->continueSampler();

		if(paused){
			while(paused && running){
				std::this_thread::sleep_for(std::chrono::milliseconds(ms));
			}
		}
		else{
			std::this_thread::sleep_for(std::chrono::milliseconds(ms));
		}
	}
}


} /* namespace whiteice */


namespace whiteice
{
	template class PTHMC< float >;
	template class PTHMC< double >;
	template class PTHMC< math::blas_real<float> >;
	template class PTHMC< math::blas_real<double> >;
};
