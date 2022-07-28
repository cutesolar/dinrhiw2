

#include "diffeqs.h"
#include "HMC_diffeq.h"

#include "DiffEQ_HMC.h"

#include "vertex.h"
#include "matrix.h"
#include "nnetwork.h"
#include "RungeKutta.h"

#include <vector>


using namespace whiteice;


bool create_random_diffeq_model(whiteice::nnetwork<>& diffeq,
				const unsigned int DIMENSIONS)
{
  if(DIMENSIONS <= 0) return false;

  std::vector<unsigned int> arch; // 11 layers, NOW: 2 layers

  const unsigned int width = 20;

  arch.push_back(DIMENSIONS);
  arch.push_back(width);
  // arch.push_back(width);
  /*
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  */
  arch.push_back(DIMENSIONS); 
  if(diffeq.setArchitecture(arch) == false) return false;

  diffeq.randomize();

  return true;
}


template <typename T>
class nnet_ode : public whiteice::math::odefunction<T>
{
public:

  nnet_ode(const whiteice::nnetwork<T>& nnet)
  {
    this->nnet = nnet;
  }

  // returns number of input dimensions
  unsigned int dimensions() const PURE_FUNCTION
  {
    return nnet.getInputs(0);
  }

  

  // calculates value of function
  whiteice::math::vertex<T> operator()
  (const whiteice::math::odeparam<T>& x)
    const PURE_FUNCTION
  {
    whiteice::math::vertex<T> output;

    if(nnet.calculate(x.y, output) == false) return output;

    return output;
  }
  
  // calculates value
  whiteice::math::vertex<T> calculate
  (const whiteice::math::odeparam<T>& x)
    const PURE_FUNCTION
  {
    whiteice::math::vertex<T> output;

    if(nnet.calculate(x.y, output) == false) return output;
    
    return output;
  }
  
  // calculates value 
  // (optimized version, this is faster because output value isn't copied)
  void calculate
  (const whiteice::math::odeparam<T>& x,
   whiteice::math::vertex<T>& y) const
  {
    whiteice::math::vertex<T> output;
    
    nnet.calculate(x.y, output);
    
    y = output;
  }
  
  // creates copy of object
  function< whiteice::math::odeparam<T>,
	    whiteice::math::vertex<T> >* clone() const
  {
    return new nnet_ode<T>(nnet);
  }

private:

  whiteice::nnetwork<T> nnet; 
};


template <typename T>
bool simulate_diffeq_model(const whiteice::nnetwork<T>& diffeq,
			   const whiteice::math::vertex<T>& start,
			   const float TIME_LENGTH,
			   std::vector< whiteice::math::vertex<T> >& data,
			   std::vector<T>& times)
{
  if(start.size() != diffeq.getInputs(0)) return false;
  if(start.size() != diffeq.getNeurons(diffeq.getLayers()-1)) return false;
  if(TIME_LENGTH < 0.0f) return false;
  if(TIME_LENGTH > 1e6f) return false; // too long simulation length [sanity check]

  data.clear();
  times.clear();

  // now uses Runge-Kutta to simulate/integrate diff.eq. dx/dt = diffeq(x)

  const float start_time = 0.0f;
  
  nnet_ode< T > ode(diffeq);

  whiteice::math::RungeKutta< T > rk(&ode);

  rk.calculate(start_time, TIME_LENGTH,
	       start,
	       data,
	       times);
  

  if(data.size() > 0 && times.size() > 0 && data.size() == times.size())
    return true;
  else
    return false;
}


// assumes times are are ordered from smallest to biggest
template <typename T>
bool simulate_diffeq_model2(const whiteice::nnetwork<T>& diffeq,
			    const whiteice::math::vertex<T>& start,
			    const float TIME_LENGTH,
			    std::vector< whiteice::math::vertex<T> >& data,
			    const std::vector<T>& correct_times)
{
  std::vector< whiteice::math::vertex<T> > data2;
  std::vector< T > times;
  
  if(simulate_diffeq_model(diffeq, start, TIME_LENGTH, data2, times) == false) return false;

  data.clear();

  unsigned int kbest = 0;
  
  for(unsigned int i=0;i<correct_times.size();i++){
    
    whiteice::math::blas_real<float> best_error = (float)INFINITY;

    for(unsigned int k=kbest;k<times.size();k++){
      auto error = whiteice::math::abs(times[k]-correct_times[i]);
      if(error <= best_error){
	kbest = k;
	best_error = error;
      }
      else break;
    }

    data.push_back(data2[kbest]);
  }

  return true; 
}


//////////////////////////////////////////////////////////////////////


template <typename T>
class nnet_gradient_ode : public whiteice::math::odefunction<T>
{public:

  nnet_gradient_ode(const whiteice::nnetwork<T>& nnet_,
		    const std::vector< whiteice::math::vertex<T> >& xdata_,
		    const std::vector< whiteice::math::vertex<T> >& deltas_,
		    const std::vector<T>& delta_times_) :
    nnet(nnet_), xdata(xdata_), deltas(deltas_), delta_times(delta_times_)
  {
    assert(xdata.size() == deltas.size());
    assert(delta_times.size() == xdata.size()); 
  }

  // returns number of input dimensions
  unsigned int dimensions() const PURE_FUNCTION
  {
    return nnet.gradient_size();
  }

  

  // calculates value of function
  whiteice::math::vertex<T> operator()
  (const whiteice::math::odeparam<T>& x)
    const PURE_FUNCTION
  {
    whiteice::math::vertex<T> output;

    // calculates delta term by linear interpolation

    whiteice::math::vertex<T> delta; // (nnet.output_size());

    // finds closest t value
    unsigned int index;
    for(index=0;index<delta_times.size();index++){
      if(x.t >= delta_times[index]) break;
    }
    if(index == delta_times.size()) index--;

    if(index+1<delta_times.size()){
      delta = deltas[index] +
	((x.t-delta_times[index])/(delta_times[index+1]-delta_times[index]))*deltas[index+1];
    }
    else delta = deltas[delta_times.size()-1];

#if 1
    math::vertex<T> xvalue;

    if(index+1<delta_times.size()){
      xvalue = xdata[index] +
	((x.t-delta_times[index])/(delta_times[index+1]-delta_times[index]))*xdata[index+1];
    }
    else xvalue = xdata[delta_times.size()-1];
#endif

    // now we have delta term, calculate MSE term (no jacobian explicitely computed)
    // we return delta^T*Jacobian(f(x))

    std::vector< math::vertex<T> > bpdata;

    if(nnet.calculate(xvalue, output, bpdata) == false)
      assert(0);
    
    if(nnet.mse_gradient(delta, bpdata, output) == false)
      assert(0);

    return output;
  }
  
  // calculates value
  whiteice::math::vertex<T> calculate
  (const whiteice::math::odeparam<T>& x)
    const PURE_FUNCTION
  {
    return (*this)(x);
  }
  
  // calculates value 
  // (optimized version, this is faster because output value isn't copied)
  void calculate
  (const whiteice::math::odeparam<T>& x,
   whiteice::math::vertex<T>& y) const
  {
    y = (*this)(x);
  }
  
  // creates copy of object
  function< whiteice::math::odeparam<T>,
	    whiteice::math::vertex<T> >* clone() const
  {
    return new nnet_gradient_ode<T>(nnet, xdata, deltas, delta_times);
  }

private:

  const whiteice::nnetwork<T>& nnet;
  const std::vector< whiteice::math::vertex<T> >& xdata;
  const std::vector< whiteice::math::vertex<T> >& deltas;
  const std::vector<T>& delta_times;
};


template <typename T>
bool simulate_diffeq_model_nn_gradient(const whiteice::nnetwork<T>& diffeq,
				       const whiteice::math::vertex<T>& start,
				       const std::vector< whiteice::math::vertex<T> >& xdata,
				       const std::vector< whiteice::math::vertex<T> >& deltas,
				       const std::vector<T>& delta_times,
				       std::vector< whiteice::math::vertex<T> >& data,
				       std::vector<T>& times)
{
  if(start.size() != diffeq.gradient_size()){
    printf("start size=%d, diffeq.gradient_size()=%d\n",
	   (int)start.size(), (int)diffeq.gradient_size());
    
    assert(0);
    
    return false;
  }
  

  data.clear();
  times.clear();

  // now uses Runge-Kutta to simulate/integrate diff.eq. d(delta(t)^T*grad(x))/dt = delta(t)^T*grad(diffeq(x))

  float START_TIME = 0.0f;
  float TIME_LENGTH = 0.0f;

  whiteice::math::convert(START_TIME, delta_times[0]);
  whiteice::math::convert(TIME_LENGTH, delta_times[delta_times.size()-1]);

  if(TIME_LENGTH < 0.0f){
    assert(0);
    return false;
  }
  
  if(TIME_LENGTH > 1e6f){
    assert(0);
    return false; // too long simulation length [sanity check]
  }

  
  nnet_gradient_ode< T > ode(diffeq, xdata, deltas, delta_times);

  whiteice::math::RungeKutta< T > rk(&ode);

  //std::cout << "RungeKutta START" << std::endl;

  rk.calculate(START_TIME, TIME_LENGTH,
	       start,
	       data,
	       times);

  //std::cout << "RungeKutta END" << std::endl;

  if(data.size() > 0 && times.size() > 0 && data.size() == times.size()){
    return true;
  }
  else{
    assert(0);
    return false;
  }
}


// assumes times are are ordered from smallest to biggest
template <typename T>
bool simulate_diffeq_model_nn_gradient2(const whiteice::nnetwork<T>& diffeq,
					const whiteice::math::vertex<T>& start,
					const std::vector< whiteice::math::vertex<T> >& xdata,
					const std::vector< whiteice::math::vertex<T> >& deltas,
					const std::vector<T>& delta_times,
					std::vector< whiteice::math::vertex<T> >& data,
					const std::vector<T>& correct_times)
{
  std::vector< whiteice::math::vertex<T> > data2;
  std::vector< T > times;

  //std::cout << "gradient2: simulate start" << std::endl;
  
  if(simulate_diffeq_model_nn_gradient(diffeq, start, xdata, deltas, delta_times, data2, times) == false)
    return false;

  //std::cout << "gradient2: simulate ok" << std::endl;
  
  data.clear();

  unsigned int kbest = 0;
  
  for(unsigned int i=0;i<correct_times.size();i++){
    
    whiteice::math::blas_real<float> best_error = (float)INFINITY;

    for(unsigned int k=kbest;k<times.size();k++){
      auto error = whiteice::math::abs(times[k]-correct_times[i]);
      if(error <= best_error){
	kbest = k;
	best_error = error;
      }
      else break;
    }

    data.push_back(data2[kbest]);
  }

  return true; 
}




// uses hamiltonian monte carlo sampler (HMC) to fit diffeq parameters to (data, times)
// Samples HMC_SAMPLES samples and selects the best parameter w solution from sampled values (max probability)
// assumes time starts from zero.
template <typename T>
bool fit_diffeq_to_data_hmc(whiteice::nnetwork<T>& diffeq,
			    const std::vector< whiteice::math::vertex<T> >& data,
			    const std::vector<T>& times,
			    const whiteice::math::vertex<T>& start_point,
			    const unsigned int HMC_SAMPLES)
{
  if(data.size() != times.size()) return false;
  if(HMC_SAMPLES <= 1) return false;
  if(diffeq.getInputs(0) != diffeq.getNeurons(diffeq.getLayers()-1)) return false;
  if(start_point.size() != diffeq.getInputs(0)) return false;
  if(data.size() <= 5) return false; // must have some data


  // TEST: samples initial x observations with given neural network to fit to data times t_i time values
  std::vector< whiteice::math::vertex<T> > xdata;

  auto delta_time = (times[times.size()-1] - times[0]).c[0];
  
  if(simulate_diffeq_model2(diffeq,
			    start_point,
			    delta_time,
			    xdata,
			    times) == false)
    return false;

  // setup HMC sampler and samples target number of points

  // TODO: extend HMC to be HMC_diffeq and calculate squared error term using diffeq simulation to get output values.. + insert correct times to HMC for sampler + starting point
  whiteice::dataset<T> ds;

  // create dataset
  ds.createCluster("input from diff.eq. 't-1'", diffeq.getInputs(0));
  ds.createCluster("correct output (to diff.eq.) 't'", diffeq.getInputs(0));

  for(unsigned int i=0;i<xdata.size();i++){
    ds.add(0, xdata[i]);
    ds.add(1, data[i]);
  }

  // whiteice::HMC<T> hmc(diffeq, ds);
  HMC_diffeq<T> hmc(diffeq, ds, start_point, times, true); // DON'T USE ADAPTIVE STEP LENGTH!

  hmc.setTemperature(0.001); // no wandering around in problem surface

  whiteice::linear_ETA<double> eta;

  eta.start(0.0, (double)HMC_SAMPLES);

  hmc.startSampler();

  while(hmc.getNumberOfSamples() <= HMC_SAMPLES){
    sleep(1);
    auto error = hmc.getMeanError(10);

    eta.update((double)hmc.getNumberOfSamples());

    std::cout << "HMC sampler error: " << error << ". ";
    std::cout << "HMC sampler samples (0 means no samples yet): " << hmc.getNumberOfSamples() << ". ";
    std::cout << "ETA: " << eta.estimate() << " secs." <<  std::endl;
  }

  hmc.stopSampler();

  auto wbest = hmc.getMean(); // FIXME: select minimum error weight

  if(diffeq.importdata(wbest) == false) return false;

  return true;
}


// uses hamiltonian monte carlo sampler (HMC) to fit diffeq parameters to (data, times)
// Samples HMC_SAMPLES samples and selects the best parameter w solution from sampled values (max probability)
// assumes time starts from zero.
template <typename T>
bool fit_diffeq_to_data_hmc2(whiteice::nnetwork<T>& diffeq,
			     const std::vector< whiteice::math::vertex<T> >& data,
			     const std::vector<T>& times,
			     const unsigned int HMC_SAMPLES)
{
  if(data.size() != times.size()) return false;
  if(HMC_SAMPLES <= 1) return false;
  if(diffeq.getInputs(0) != diffeq.getNeurons(diffeq.getLayers()-1)) return false;
  if(data.size() <= 5) return false; // must have some data
  if(data[0].size() != diffeq.getInputs(0)) return false;

  // construct dataset for HMC which used for training
  // we use samples from time-series data (data, times)
  
  whiteice::dataset<T> ds;

  // create dataset time-series data points
  const unsigned int TIME_SERIES_LENGTH = 10;
  
  ds.createCluster("time-series for differential equations", TIME_SERIES_LENGTH*data[0].size());
  
  // creates dataset
  for(unsigned int i=0;i<data.size();i+=TIME_SERIES_LENGTH){
    
    math::vertex<T> in;
    in.resize(ds.dimension(0));
    
    for(unsigned int j=0;j<TIME_SERIES_LENGTH;j++){
      in.write_subvertex(data[i+j], j*data[0].size());
    }

    ds.add(0, in);
  }

  // creates times for dataset
  auto delta_time = (times[times.size()-1] - times[0]).c[0];
  auto delta_t = delta_time/times.size();

  std::vector<T> dstimes;
  
  for(unsigned int i=0;i<TIME_SERIES_LENGTH;i++)
    dstimes.push_back(i*delta_t);

  //HMC_diffeq<T> hmc(diffeq, ds, start_point, times, true);
  
  whiteice::DiffEq_HMC<T> hmc(diffeq, ds, dstimes, true, true); // use adaptive step length

  whiteice::linear_ETA<double> eta;

  eta.start(0.0, (double)HMC_SAMPLES);

  hmc.startSampler();

  while(hmc.getNumberOfSamples() <= HMC_SAMPLES){
    sleep(5);
    auto error = hmc.getMeanError(10);

    eta.update((double)hmc.getNumberOfSamples());

    std::cout << "HMC sampler error: " << error << ". ";
    std::cout << "HMC sampler samples (0 means no samples yet): " << hmc.getNumberOfSamples() << ". ";
    std::cout << "ETA: " << eta.estimate() << " secs." <<  std::endl;
  }

  hmc.stopSampler();

  auto wbest = hmc.getMean(); // FIXME: select minimum error weight

  if(diffeq.importdata(wbest) == false) return false;

  return true;
}



template bool simulate_diffeq_model< math::blas_real<float> >
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const float TIME_LENGTH,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 std::vector< whiteice::math::blas_real<float> >& times);


template bool simulate_diffeq_model< math::blas_real<double> >
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const float TIME_LENGTH,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 std::vector< whiteice::math::blas_real<double> >& times);


// fits simulated data points to correct_times values
// template <typename T = math::blas_real<float> >
template bool simulate_diffeq_model2< math::blas_real<float> >
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const float TIME_LENGTH,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& correct_times);

template bool simulate_diffeq_model2< math::blas_real<double> >
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const float TIME_LENGTH,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& correct_times);


template bool simulate_diffeq_model_nn_gradient
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& xdata,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& deltas,
 const std::vector< math::blas_real<float> >& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 std::vector< math::blas_real<float> >& times);

template bool simulate_diffeq_model_nn_gradient
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& xdata,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& deltas,
 const std::vector< math::blas_real<double> >& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 std::vector< math::blas_real<double> >& times);


// assumes times are are ordered from smallest to biggest
template bool simulate_diffeq_model_nn_gradient2
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& xdata,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& deltas,
 const std::vector< math::blas_real<float> >& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& correct_times);

template bool simulate_diffeq_model_nn_gradient2
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& xdata,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& deltas,
 const std::vector< math::blas_real<double> >& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& correct_times);




template bool fit_diffeq_to_data_hmc< math::blas_real<float> >
(whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& times,
 const whiteice::math::vertex< math::blas_real<float> >& start_point,
 const unsigned int HMC_SAMPLES);


template bool fit_diffeq_to_data_hmc< math::blas_real<double> >
(whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& times,
 const whiteice::math::vertex< math::blas_real<double> >& start_point,
 const unsigned int HMC_SAMPLES);


template bool fit_diffeq_to_data_hmc2< math::blas_real<float> >
(whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& times,
 const unsigned int HMC_SAMPLES);

template bool fit_diffeq_to_data_hmc2< math::blas_real<double> >
(whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& times,
 const unsigned int HMC_SAMPLES);
