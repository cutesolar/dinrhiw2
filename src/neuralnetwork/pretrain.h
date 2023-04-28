/*
 * Heuristics to pretrain neural network weights using data.
 *
 * Let A, B and C be neural network layer's operators with matrix multiplication and 
 * non-linearity (C => y = g(C*x) )  
 * 
 * We assume operators are invertible so there is inverse functions inv(C) and 
 * inv(C*B*A)=inv(A)*inv(B)*inv(C).
 *
 * We calculate weights using linear optimization and training data (x,y). 
 * Parameters are initialized randomly and set to have unit weights 
 * (data aprox in the range of -1..1 typically)
 *
 * First we solve last layer weights, x' = B*A*x and we optimize 
 * linearly x' -> y and operator C's parameters (g^-1(y) = M_c*x' + b_c)
 *
 * Next we solve each layer's parameters x'' = A*x, and 
 * we solve B's parameters, we solve y' = inv(C)*y and have 
 * training data x'' -> y' to solve for parameters of B.
 *
 * You can run pretrain_nnetwork() many times for the same network until aprox convergence.
 *
 * 
 * Copyright Tomas Ukkonen 2023 <tomas.ukkonen@iki.fi>
 * Novel Insight Research
 *
 */


#ifndef __whiteice_good_pretrain_h
#define __whiteice_good_pretrain_h

#include "nnetwork.h"
#include "dataset.h"

#include <mutex>
#include <thread>


namespace whiteice
{

  // class to run pretraining of neural network in the background (matrix factorization method)
  // 
  // sets nnet to be linear neural network for training and disables residual neural network,
  // later actual optimization/training should switch to rectifier non-linearity for quite close to
  // linear operation where found parameter weights using linear pretrainer are still quite good
  // starting points.
  //
  // NOTE: It seems you get only 2% better results (N=20, nearly statistically significant) when
  // optimizing neural network with this pretrainer algorithm..
  // 
  template <typename T>
  class PretrainNN
  {
  public:
    
    PretrainNN();
    virtual ~PretrainNN();

    bool startTrain(const whiteice::nnetwork<T>& nnet,
		    const whiteice::dataset<T>& data,
		    const unsigned int NUMITERATIONS = 2000);

    bool isRunning() const;

    void getStatistics(unsigned int& iterations, T& error) const;

    bool stopTrain();

    bool getResults(whiteice::nnetwork<T>& nnet) const;
    
  private:

    void worker_loop();

    mutable std::mutex solution_mutex;
    whiteice::nnetwork<T> nnet;
    whiteice::dataset<T> data;
    
    unsigned int iterations;
    T current_error;

    unsigned int MAXITERS;
    
    
    mutable std::mutex thread_mutex;
    std::thread* worker_thread;

    bool running;
  };

  
  extern template class PretrainNN< math::blas_real<float> >;
  extern template class PretrainNN< math::blas_real<double> >;

  extern template class PretrainNN< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >;
  extern template class PretrainNN< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >;

  
  //////////////////////////////////////////////////////////////////////
  
  
  template <typename T>
  bool pretrain_nnetwork(nnetwork<T>& nnet, const dataset<T>& data);
  
  
  // assumes whole network is linear matrix operations y = M*x,
  // M = A*B*C*D, linear M is solves from data
  // solves changes D to matrix using equation A*(B+D)*C = M => D = A^-1*M*C^-1 - B
  // solves D for each matrix and then applies changes
  // [assumes linearity so this is not very good solution] 
  template <typename T>
  bool pretrain_nnetwork_matrix_factorization(nnetwork<T>& nnet, const dataset<T>& data,
					      const T step_length = T(1e-5f));
  

  //////////////////////////////////////////////////////////////////////
  

  extern template bool pretrain_nnetwork< math::blas_real<float> >
  (nnetwork< math::blas_real<float> >& nnet, const dataset< math::blas_real<float> >& data);
  
  extern template bool pretrain_nnetwork< math::blas_real<double> >
  (nnetwork< math::blas_real<double> >& nnet, const dataset< math::blas_real<double> >& data);


  
  extern template bool pretrain_nnetwork_matrix_factorization< math::blas_real<float> >
  (nnetwork< math::blas_real<float> >& nnet, const dataset< math::blas_real<float> >& data,
   const math::blas_real<float> step_length);

  extern template bool pretrain_nnetwork_matrix_factorization< math::blas_real<double> >
  (nnetwork< math::blas_real<double> >& nnet, const dataset< math::blas_real<double> >& data,
   const math::blas_real<double> step_length);

  
  extern template bool pretrain_nnetwork_matrix_factorization< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >
  (nnetwork< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >& nnet,
   const dataset< math::superresolution< math::blas_real<float>, math::modular<unsigned int> > >& data,
   const math::superresolution< math::blas_real<float>, math::modular<unsigned int> > step_length);
  
  extern template bool pretrain_nnetwork_matrix_factorization< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >
  (nnetwork< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >& nnet,
   const dataset< math::superresolution< math::blas_real<double>, math::modular<unsigned int> > >& data,
   const math::superresolution< math::blas_real<double>, math::modular<unsigned int> > step_length);

  
};


#endif
