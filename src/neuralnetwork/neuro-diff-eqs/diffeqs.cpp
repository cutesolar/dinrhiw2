

#include "diffeqs.h"

#include <dinrhiw/RungeKutta.h>


bool create_random_diffeq_model(whiteice::nnetwork<>& diffeq,
				const unsigned int DIMENSIONS)
{
  if(DIMENSIONS <= 0) return false;

  std::vector<unsigned int> arch; // 4 layers

  unsigned int width = 10;

  arch.push_back(DIMENSIONS);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(DIMENSIONS);

  if(diffeq.setArchitecture(arch) == false) return false;

  diffeq.randomize();

  return true;
}


class nnet_ode : public whiteice::math::odefunction< whiteice::math::blas_real<float> >
{
public:

  nnet_ode(whiteice::nnetwork<>* nnet){
    if(nnet == NULL) throw std::string("ERROR: nnet_ode ctor: input parameter null!");
    
    this->nnet = nnet;
  }

  // returns number of input dimensions
  unsigned int dimensions() const PURE_FUNCTION
  {
    if(nnet) return nnet->getInputs(0);
    else return 0;
  }

  

  // calculates value of function
  whiteice::math::vertex<> operator()
  (const whiteice::math::odeparam< whiteice::math::blas_real<float> >& x)
    const PURE_FUNCTION
  {
    whiteice::math::vertex<> input, output;
    input = x.y;

    if(nnet)
      if(nnet->calculate(input, output) == false) return output;

    return output;
  }
  
  // calculates value
  whiteice::math::vertex<> calculate
  (const whiteice::math::odeparam< whiteice::math::blas_real<float> >& x)
    const PURE_FUNCTION
  {
    whiteice::math::vertex<> input, output;
    input = x.y;

    if(nnet)
      if(nnet->calculate(input, output) == false) return output;
    
    return output;
  }
  
  // calculates value 
  // (optimized version, this is faster because output value isn't copied)
  void calculate
  (const whiteice::math::odeparam< whiteice::math::blas_real<float> >& x,
   whiteice::math::vertex<>& y) const
  {
    whiteice::math::vertex<> input, output;
    input = x.y;

    if(nnet)
      nnet->calculate(input, output);
    
    y = output;
  }
  
  // creates copy of object
  function< whiteice::math::odeparam< whiteice::math::blas_real<float> >,
	    whiteice::math::vertex<> >* clone() const
  {
    return new nnet_ode(nnet);
  }

private:

  whiteice::nnetwork<>* nnet = NULL;
};


bool simulate_diffeq_model(whiteice::nnetwork<>& diffeq,
			   const whiteice::math::vertex<>& start,
			   const float TIME_LENGTH,
			   std::vector< whiteice::math::vertex<> >& data)
{
  if(start.size() != diffeq.getInputs(0)) return false;
  if(start.size() != diffeq.getNeurons(diffeq.getLayers()-1)) return false;
  if(TIME_LENGTH < 0.0f) return false;
  if(TIME_LENGTH > 1e4f) return false; // too long simulation length [sanity check]

  data.clear();

  // now uses Runge-Kutta to simulate/integrate diff.eq. dx/dt = diffeq(x)

  const float start_time = 0.0f;
  std::vector< whiteice::math::blas_real<float> > times;

  whiteice::math::odefunction< whiteice::math::blas_real<float> >* ode = new nnet_ode(&diffeq);

  whiteice::math::RungeKutta< whiteice::math::blas_real<float> > rk(ode);

  rk.calculate(start_time, TIME_LENGTH,
	       start,
	       data,
	       times);

  delete ode;
  
  if(data.size() > 0) return true;
  else return false;
}

