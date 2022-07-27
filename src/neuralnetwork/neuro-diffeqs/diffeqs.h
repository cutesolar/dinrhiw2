
#ifndef __43_diffeqs_h
#define __43_diffeqs_h

#include <dinrhiw/dinrhiw.h>

using namespace whiteice;

bool create_random_diffeq_model(whiteice::nnetwork<>& diffeq, const unsigned int DIMENSIONS);

template <typename T = math::blas_real<float> > 
  bool simulate_diffeq_model(const whiteice::nnetwork<T>& diffeq,
			     const whiteice::math::vertex<T>& start,
			     const float TIME_LENGTH,
			     std::vector< whiteice::math::vertex<T> >& data,
			     std::vector<T>& times);

// fits simulated data points to correct_times values
template <typename T = math::blas_real<float> > 
  bool simulate_diffeq_model2(const whiteice::nnetwork<T>& diffeq,
			      const whiteice::math::vertex<T>& start,
			      const float TIME_LENGTH,
			      std::vector< whiteice::math::vertex<T> >& data,
			      const std::vector<T>& correct_times);


template <typename T = math::blas_real<float> > 
bool simulate_diffeq_model_nn_gradient(const whiteice::nnetwork<T>& diffeq,
				       const whiteice::math::vertex<T>& start,
				       const std::vector< whiteice::math::vertex<T> >& deltas,
				       const std::vector<T>& delta_times,
				       std::vector< whiteice::math::vertex<T> >& data,
				       std::vector<T>& times);

// assumes times are are ordered from smallest to biggest
template <typename T = math::blas_real<float> >
bool simulate_diffeq_model_nn_gradient2(const whiteice::nnetwork<T>& diffeq,
					const whiteice::math::vertex<T>& start,
					const std::vector< whiteice::math::vertex<T> >& deltas,
					const std::vector<T>& delta_times,
					std::vector< whiteice::math::vertex<T> >& data,
					const std::vector<T>& correct_times);


// uses hamiltonian monte carlo to fit diffeq parameters to (data, times)
// Samples HMC_SAMPLES samples and selects the best parameter w solution from sampled values (max probability)
template <typename T = math::blas_real<float> > 
  bool fit_diffeq_to_data_hmc(whiteice::nnetwork<T>& diffeq,
			      const std::vector< whiteice::math::vertex<T> >& data,
			      const std::vector< T >& times,
			      const whiteice::math::vertex<T>& start_point,
			      const unsigned int HMC_SAMPLES);


// template <typename T = math::blas_real<float> >

extern template bool simulate_diffeq_model< math::blas_real<float> >
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const float TIME_LENGTH,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 std::vector< math::blas_real<float> >& times);


extern template bool simulate_diffeq_model< math::blas_real<double> >
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const float TIME_LENGTH,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 std::vector< math::blas_real<double> >& times);


// fits simulated data points to correct_times values
// template <typename T = math::blas_real<float> >
extern template bool simulate_diffeq_model2< math::blas_real<float> >
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const float TIME_LENGTH,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& correct_times);

extern template bool simulate_diffeq_model2< math::blas_real<double> >
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const float TIME_LENGTH,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& correct_times);


extern template bool simulate_diffeq_model_nn_gradient
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& deltas,
 const std::vector< math::blas_real<float> >& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 std::vector< math::blas_real<float> >& times);

extern template bool simulate_diffeq_model_nn_gradient
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& deltas,
 const std::vector< math::blas_real<double> >& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 std::vector< math::blas_real<double> >& times);


// assumes times are are ordered from smallest to biggest
extern template bool simulate_diffeq_model_nn_gradient2
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& deltas,
 const std::vector< math::blas_real<float> >& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& correct_times);

extern template bool simulate_diffeq_model_nn_gradient2
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& deltas,
 const std::vector< math::blas_real<double> >& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& correct_times);



extern template bool fit_diffeq_to_data_hmc< math::blas_real<float> >
(whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& times,
 const whiteice::math::vertex< math::blas_real<float> >& start_point,
 const unsigned int HMC_SAMPLES);


extern template bool fit_diffeq_to_data_hmc< math::blas_real<double> >
(whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& times,
 const whiteice::math::vertex< math::blas_real<double> >& start_point,
 const unsigned int HMC_SAMPLES);




#endif
