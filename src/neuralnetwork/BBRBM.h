/*
 * BBRBM.h
 *
 *  Created on: 22.6.2015
 *      Author: Tomas Ukkonen
 */

#ifndef NEURALNETWORK_BBRBM_H_
#define NEURALNETWORK_BBRBM_H_

#include "vertex.h"
#include "matrix.h"
#include "RNG.h"


namespace whiteice {

/**
 * Standard Bernoulli-Bernoulli RBM
 */
template <typename T = math::blas_real<float> >
class BBRBM {
  public:
  
  BBRBM();
  BBRBM(const BBRBM<T>& rbm);
  
  // creates 2-layer: V * H network
  BBRBM(unsigned int visible, unsigned int hidden) ;
  
  virtual ~BBRBM();
  
  
  BBRBM<T>& operator=(const BBRBM<T>& rbm);
  bool operator==(const BBRBM<T>& rbm) const;
  bool operator!=(const BBRBM<T>& rbm) const;
  
  bool resize(unsigned int visible, unsigned int hidden);
  
  ////////////////////////////////////////////////////////////

  unsigned int getVisibleNodes() const;
  unsigned int getHiddenNodes() const;
  
  void getVisible(math::vertex<T>& v) const;
  bool setVisible(const math::vertex<T>& v);
  
  void getHidden(math::vertex<T>& h) const;
  bool setHidden(const math::vertex<T>& h);

  // h = s(W*v + b), v = s(h*W + a)
  math::vertex<T> getBValue() const;
  math::vertex<T> getAValue() const;
  math::matrix<T> getWeights() const;

  bool setBValue(const math::vertex<T>& b);
  bool setAValue(const math::vertex<T>& a);
  bool setWeights(const math::matrix<T>& W);

  // v->h (but no discretization of h) useful when calculating gradients..
  bool getHiddenResponseField(const math::vertex<T>& v, math::vertex<T>& h) const;
  
  bool reconstructData(unsigned int iters = 2); // 2 to v->h->v
  bool reconstructData(std::vector< math::vertex<T> >& samples,
		       unsigned int iters = 1);
  bool reconstructDataHidden(unsigned int iters = 2); // 2 to h->v->h

  // calculates h = sigmoid(W*v + b) without disretization step
  bool calculateHiddenMeanField(const math::vertex<T>& v, math::vertex<T>& h) const;

  // calculates v = sigmoid(h*W + a) without discretization step
  bool calculateVisibleMeanField(const math::vertex<T>& h, math::vertex<T>& v) const;
  
  void getParameters(math::matrix<T>& W, math::vertex<T>& a, math::vertex<T>& b) const;
  
  bool initializeWeights(); // initialize weights to small values
  
  // calculates single epoch for updating weights using CD-1 and
  // returns reconstruction error
  T learnWeights(const std::vector< math::vertex<T> >& samples,
		 const unsigned int EPOCHS=1,
		 const int verbose = 0,
		 const bool* running = NULL);

  // calculates parameters using LBFGS 2nd order optimization and
  // CD-3 to estimate gradient
  T learnWeights2(const std::vector< math::vertex<T> >& samples,
		  const unsigned int EPOCHS=1,
		  const int verbose = 0,
		  const bool* running = NULL);
  

  T reconstructionError(const std::vector< math::vertex<T> >& samples,
			unsigned int N, // number of samples to use from samples to estimate reconstruction error
			const math::vertex<T>& a,
			const math::vertex<T>& b,			
			const math::matrix<T>& W) const ; // weight matrix (parameters) to use

  T reconstructionError(const std::vector< math::vertex<T> >& samples,
			unsigned int N) const  // number of samples to use from samples to estimate reconstruction error
  
  { return reconstructionError(samples, N, this->a, this->b, this->W); }

  T reconstructionError(const math::vertex<T>& s,
			const math::vertex<T>& a,
			const math::vertex<T>& b,
			const math::matrix<T>& W) const ; // weight matrix (parameters) to use

  ////////////////////////////////////////////////////////////
  // U(q) functions used to maximize P(v|data, q) ~ exp(-U(q))
  //       rbm parameters q

  bool setUData(const std::vector< math::vertex<T> >& samples);

  unsigned int qsize() const ; // size of q vector q = [vec(W)]
  
  // converts q vector into parameters (W, a, b)
  bool convertParametersToQ(const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b,
			    math::vertex<T>& q) const;

  // converts q vector into parameters (W, a, b)
  bool convertQToParameters(const math::vertex<T>& q, math::matrix<T>& W, math::vertex<T>& a, math::vertex<T>& b) const;
  
  // sets (W) parameters according to q vector
  bool setParametersQ(const math::vertex<T>& q);
  bool getParametersQ(math::vertex<T>& q) const;

  // keeps parameters within sane levels (clips overly large parameters and NaNs)
  void safebox(math::vertex<T>& a, math::vertex<T>& b, math::matrix<T>& W) const;
  
  T U(const math::vertex<T>& q) const ;

  // uses CD-3 to estimate gradient
  math::vertex<T> Ugrad(const math::vertex<T>& q) ;

  // prints min/max values of parameters to log
  bool diagnostics() const;

  ////////////////////////////////////////////////////////////
  
  // load & saves RBM data from/to file
  
  bool load(const std::string& filename) ;
  bool save(const std::string& filename) const ;

 protected:

  void sigmoid(const math::vertex<T>& input, math::vertex<T>& output) const;

  void sigmoid(math::vertex<T>& x) const;
  
 private:
  math::vertex<T> h, v;
  
  math::matrix<T> W;
  math::vertex<T> a;
  math::vertex<T> b;

  std::vector< math::vertex<T> > Usamples;

 public:

  RNG<T> rng;
};

 
  // extern template class BBRBM< float >;
  // extern template class BBRBM< double >;
 extern template class BBRBM< math::blas_real<float> >;
 extern template class BBRBM< math::blas_real<double> >;
 
} /* namespace whiteice */

#endif /* NEURALNETWORK_BBRBM_H_ */
