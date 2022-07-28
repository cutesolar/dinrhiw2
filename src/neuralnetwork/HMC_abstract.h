/*
 * Hamiltonian Monte Carlo Markov Chain sampler 
 */

// TODO: adaptive step length DO NOT work very well and
//       is disabled as the default

#ifndef HMC_abstract_h
#define HMC_abstract_h

#include <vector>
#include <unistd.h>

#include "vertex.h"
#include "matrix.h"
#include "dinrhiw_blas.h"

#include <thread>
#include <mutex>
#include <chrono>


namespace whiteice
{
	template <typename T = math::blas_real<float> >
	class HMC_abstract
	{
    	public:
		HMC_abstract(bool storeSamples, bool adaptive=false);
		~HMC_abstract();

		// set "temperature" for probability distribution [used in sampling/training]
		// [t = 1 => no (low) temperature]
		// [t = 0 => high temperature (high degrees of freedom)]
		virtual bool setTemperature(const T t) = 0;

		// get "temperature" of probability distribution
		virtual T getTemperature() = 0;

		// probability functions for hamiltonian MC sampling of
		// P ~ exp(-U(q)) distribution
		virtual T U(const math::vertex<T>& q) const = 0;

		// calculates difference: U(q1) - U(q2)
		// [inheritor may overload this operation as calculation of difference
		// maybe faster/possible instead of calculation of U(q) separatedly
		virtual T Udiff(const math::vertex<T>& q1, const math::vertex<T>& q2) const
		{
			return (U(q1) - U(q2));
		}

		virtual math::vertex<T> Ugrad(const math::vertex<T>& q) const = 0;

		// a starting point q for the sampler (may not be random)
		virtual void starting_position(math::vertex<T>& q) const = 0;

		bool startSampler();
		bool pauseSampler();
		bool continueSampler();
		bool stopSampler();

		unsigned int getSamples(std::vector< math::vertex<T> >& samples) const;
		unsigned int getNumberOfSamples() const;

		// gets the latest sample or sets the next sample of the sampling process [sampling point]
		bool getCurrentSample(math::vertex<T>& q);
		bool setCurrentSample(const math::vertex<T>& q);

		// set that we have seen that latest sample
		void setUpdated(bool updated);

		bool getUpdated(); // are there new sample since last call to setUpdated(false) ??

		math::vertex<T> getMean() const;
		// math::matrix<T> getCovariance() const;

		// calculates mean error for the latest N samples, 0 = all samples
		T getMeanError(unsigned int latestN = 0) const;

		bool getAdaptive() const { return adaptive; }

    	private:

		bool adaptive;

		bool storeSamples;
		std::vector< math::vertex<T> > samples;

		volatile bool q_overwritten;
		volatile bool q_updated;
		math::vertex<T> q;
		mutable std::mutex updating_sample;
    
		// used to calculate statistics when needed
		math::vertex<T> sum_mean;
		// math::matrix<T> sum_covariance;
		unsigned int sum_N;

		bool running, paused;

		mutable std::vector<std::thread*> sampling_thread;
		mutable std::mutex solution_lock, start_lock;

		void sampler_loop(); // worker thread loop
	};


};


namespace whiteice
{
	extern template class HMC_abstract< float >;
	extern template class HMC_abstract< double >;
	extern template class HMC_abstract< math::blas_real<float> >;
	extern template class HMC_abstract< math::blas_real<double> >;
};


#endif
