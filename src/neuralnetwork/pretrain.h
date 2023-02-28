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



namespace whiteice
{
  
  template <typename T>
    bool pretrain_nnetwork(nnetwork<T>& nnet, const dataset<T>& data);


  //////////////////////////////////////////////////////////////////////
  

  extern template bool pretrain_nnetwork< math::blas_real<float> >
  (nnetwork< math::blas_real<float> >& nnet, const dataset< math::blas_real<float> >& data);
  
  extern template bool pretrain_nnetwork< math::blas_real<double> >
  (nnetwork< math::blas_real<double> >& nnet, const dataset< math::blas_real<double> >& data);
  
};


#endif
