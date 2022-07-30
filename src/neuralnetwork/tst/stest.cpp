/* 
 * superresolution test cases
 *
 * Tomas Ukkonen
 */

#include <iostream>
#include <vector>

#include "superresolution.h"
#include "dataset.h"
#include "nnetwork.h"
#include "NNGradDescent.h"
#include "RNG.h"

#undef __STRICT_ANSI__
#include <fenv.h>

// enables floating point exceptions, these are good for debugging 
// to notice BAD floating point values that come from software bugs..
#include <fenv.h>

extern "C" {

  // traps floating point exceptions..
#define _GNU_SOURCE 1
#ifdef __linux__
#include <fenv.h>
  static void __attribute__ ((constructor))
  trapfpe(){
    feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    //feenableexcept(FE_INVALID);
  }
#endif
  
}

using namespace whiteice;

int main()
{
  std::cout << "superresolution test cases" << std::endl;

  whiteice::logging.setPrintOutput(true);
  
  // creates training dataset
  std::cout << "Creating training dataset.." << std::endl;
  
  dataset<
    math::superresolution< math::blas_real<double>,
			   math::modular<unsigned int> > > data;

  dataset< math::blas_real<double> > data2;

  data.createCluster("input", 4);
  data.createCluster("output", 4);
  data2.createCluster("input", 4);
  data2.createCluster("output", 4);  
    
  std::vector< math::vertex<
    math::superresolution< math::blas_real<double>,
			   math::modular<unsigned int> > > > inputs;

  std::vector< math::vertex<
    math::superresolution< math::blas_real<double>,
			   math::modular<unsigned int> > > > outputs;

  std::vector< math::vertex<math::blas_real<double> > > inputs2;
  std::vector< math::vertex<math::blas_real<double> > > outputs2;

  RNG< math::blas_real<double> > prng;

  nnetwork< math::blas_real<double> > net;
  nnetwork< math::superresolution< math::blas_real<double>,
				   math::modular<unsigned int> > > snet;

  math::vertex< math::blas_real<double> > weights;
  math::vertex< math::superresolution< math::blas_real<double>,
				       math::modular<unsigned int> > > sweights;

  // was: 4-10-10-10-4 network
  // now: 5-50-4 network
  std::vector<unsigned int> arch;  
  arch.push_back(4);
  arch.push_back(50);
  //arch.push_back(50);
  //arch.push_back(50);
  arch.push_back(4);


  // pureLinear non-linearity (layers are all linear) [pureLinear or rectifier]
  // rectifier don't work!!!
  net.setArchitecture(arch, nnetwork< math::blas_real<double> >::rectifier);
  snet.setArchitecture(arch, nnetwork< math::superresolution<
		       math::blas_real<double>,
		       math::modular<unsigned int> > >::rectifier);

  net.randomize();
  snet.randomize();
  net.setResidual(false);
  snet.setResidual(false);
  

  std::cout << "net weights size: " << net.gradient_size() << std::endl;
  std::cout << "snet weights size: " << snet.gradient_size() << std::endl;

  net.exportdata(weights);
  snet.exportdata(sweights);
  
  math::convert(sweights, weights);

  // sweights = abs(sweights); // drop complex parts of initial weights

  // randomize sweights [also superresolution parts]
  {
    RNG< math::blas_real<double> > rng;
    
    for(unsigned int i=0;i<sweights.size();i++)
    {
      for(unsigned int k=0;k<sweights[i].size();k++){
	sweights[i][k] = whiteice::math::blas_real<double>(0.01f)*rng.normal();
      }
      
    }
    
  }
  
  snet.importdata(sweights);
  snet.exportdata(sweights);

  std::cout << "weights = " << weights << std::endl;
  std::cout << "sweights = " << sweights << std::endl;

  const unsigned int NUMDATAPOINTS = 10;
  
  for(unsigned int i=0;i<NUMDATAPOINTS;i++){
    math::vertex<
      math::superresolution< math::blas_real<double>,
			     math::modular<unsigned int> > > sx, sy;

    math::vertex< math::blas_real<double> > x, y;

    math::blas_real<double> sigma = 4.0;
    math::blas_real<double> f = 3.14157, a = 1.1132, w = 7.342, one = 1.0;

    x.resize(4);
    prng.normal(x);
    x = sigma*x; // x ~ N(0,4^2*I)

    y.resize(4);

    y[0] = math::sin((f*x[0]*x[1]*x[2]*x[3]));
    if(x[3].c[0] >= 0.0f)
      y[1] =  math::pow(a, (x[0]/(math::abs(x[2])+one)) );
    else
      y[1] = -math::pow(a, (x[0]/(math::abs(x[2])+one)) );

    y[2] = 0.0f;
    if(x[1].c[0] >= 0.0f) y[2] += one;
    else y[2] -= one;
    if(x[3].c[0] >= 0.0f) y[2] += one;
    else y[2] -= one;
    auto temp = math::cos(w*x[0]);
    if(temp.c[0] >= 0.0f) y[2] += one;
    else y[2] -= one;
    
    y[3] = x[1]/x[0] + x[2]*math::sqrt(math::abs(x[3])) + math::abs(x[3] - x[0]);

    // std::cout << "x = " << x << std::endl;
    //std::cout << "y = " << y << std::endl;

    //net.calculate(x, y); // USE nnetwork as a target function (easier)

    inputs2.push_back(x);
    outputs2.push_back(y);

    whiteice::math::convert(sx, x);
    whiteice::math::convert(sy, y);

    // std::cout << "sx = " << sx << std::endl;
    //std::cout << "sy = " << sy << std::endl;

    inputs.push_back(sx);
    outputs.push_back(sy);
    
    // calculates nnetwork response to y=f(sx) with net and snet which should give same results
    
    net.calculate(x, y);

    // std::cout << "y  =  f(x) = " << y << std::endl;

    snet.calculate(sx, sy);

    // std::cout << "sy = f(sx) = " << sy << std::endl;
  }

  data.add(0, inputs);
  data.add(1, outputs);
  
  //data.preprocess(0);
  //data.preprocess(1);

  data2.add(0, inputs2);
  data2.add(1, outputs2);

  data2.preprocess(0);
  data2.preprocess(1);

  if(data2.save("simpleproblem.ds") == false){
    printf("ERROR: saving data to file failed!\n");
    exit(-1);
  }

  // next DO NNGradDescent<> && nnetwork<> to actually learn the data.

  // gradient descent code
  {
    
    math::vertex< math::superresolution<math::blas_real<double>,
					math::modular<unsigned int> > > grad, err, weights, w0;
    math::vertex< math::superresolution<math::blas_real<double>,
					math::modular<unsigned int> > > sumgrad;
    
    unsigned int counter = 0;
    math::superresolution<math::blas_real<double>,
			  math::modular<unsigned int> > error(1000.0f), min_error(1000.0f), latest_error(1000.0f);
    math::superresolution<math::blas_real<double>,
			  math::modular<unsigned int> > lrate(0.01f); // WAS: 0.05
    
    double lratef = 0.01;
    
    while(abs(error)[0].real() > math::blas_real<double>(0.001f) && lratef > 1e-100 && counter < 100000)
    {
      error = math::superresolution<math::blas_real<double>,
				    math::modular<unsigned int> >(0.0f);
      sumgrad.zero();
      
      // goes through data, calculates gradient
      // exports weights, weights -= 0.01*gradient
      // imports weights back

      math::superresolution<math::blas_real<double>,
			    math::modular<unsigned int> > ninv =
	math::superresolution<math::blas_real<double>,
			      math::modular<unsigned int> >
	(1.0f/(data.size(0)*data.access(1,0).size()));

      math::superresolution<math::blas_real<double>,
			    math::modular<unsigned int> > h, s0(1.0), epsilon(1e-30);

      h.ones(); // differential operation difference
      h = epsilon*h;

      snet.exportdata(weights);
      snet.exportdata(w0);

      for(unsigned int i=0;i<data.size(0);i++){

	// selects K:th dimension in number and adjust weights according to it.
	//const unsigned int K = prng.rand() % weights[0].size();
	
	{
	  const auto x = data.access(0,i);
	  const auto y = data.access(1,i);
	  
	  snet.calculate(x, err);
	  err -= y;
	  
	  
	  for(unsigned int j=0;j<err.size();j++){
	    const auto& ej = err[j];
	    for(unsigned int k=0;k<ej.size();k++)
	      error += ninv*ej[k]*math::conj(ej[k]);

	    // error += ninv*math::sqrt(ej[0]*math::conj(ej[0]))/err.size();
	  }
	  
	  // this works with pureLinear non-linearity
	  auto delta = err; // delta = (f(z) - y)
	  math::matrix< math::superresolution<math::blas_real<double>,
					      math::modular<unsigned int> > > DF;

	  math::matrix< math::superresolution<math::blas_complex<double>,
					      math::modular<unsigned int> > > cDF;
		  
	  snet.jacobian(x, DF);
	  cDF.resize(DF.ysize(), DF.xsize());

	  // circular convolution in F-domain
	  
	  math::vertex<
	    math::superresolution< math::blas_complex<double>,
				   math::modular<unsigned int> > > ce, cerr;
	  
	  for(unsigned int j=0;j<DF.ysize();j++){
	    for(unsigned int i=0;i<DF.xsize();i++){
	      whiteice::math::convert(cDF(j,i), DF(j,i));
	      cDF(j,i).fft();
	    }
	  }

	  ce.resize(err.size());
	  
	  for(unsigned int i=0;i<err.size();i++){
	    whiteice::math::convert(ce[i], err[i]);
	    ce[i].fft();
	  }
	  
	  cerr.resize(DF.xsize());
	  cerr.zero();

	  for(unsigned int i=0;i<DF.xsize();i++){
	    auto ctmp = ce;
	    for(unsigned int j=0;j<DF.ysize();j++){
	      cerr[i] += ctmp[j].circular_convolution(cDF(j,i));
	    }
	  }
	  
	  err.resize(cerr.size());
	    
	  for(unsigned int i=0;i<err.size();i++){
	    cerr[i].inverse_fft();
	    for(unsigned int k=0;k<err[i].size();k++)
	      whiteice::math::convert(err[i][k], cerr[i][k]); // converts complex numbers to real
	  }

	  grad = err;
	}
	
	if(i == 0)
	  sumgrad = ninv*grad;
	else
	  sumgrad += ninv*grad;
      }

#if 0
      const math::superresolution<math::blas_real<double>,
				  math::modular<unsigned int> > alpha(0.50f); // was: 1e-3f
      const double alphaf = 0.50f;    

      math::vertex< math::superresolution<math::blas_real<double>,
					  math::modular<unsigned int> > > regularizer;

      regularizer = weights;

      for(unsigned int j=0;j<regularizer.size();j++){
	regularizer[j][0] = 0.0f;
	for(unsigned int k=1;k<regularizer[0].size();k++){
	  regularizer[j][k] *= alphaf;
	}
      }
#endif

      auto abserror = error;
      abserror[0] = abs(abserror[0]);
      
      for(unsigned int i=1;i<abserror.size();i++){
	abserror[0] += abs(abserror[i]);
	//abserror[i] = math::blas_real<double>(0.0f);
	abserror[i] = 0.0f;
      }

      auto orig_error = abserror;

      unsigned int grad_search_counter = 0;
      
      while(grad_search_counter < 500){ // until error becomes smaller

	auto delta_grad = sumgrad;
	
	for(unsigned int j=0;j<sumgrad.size();j++){
	  for(unsigned int k=0;k<sumgrad[0].size();k++){
	    delta_grad[j][k] *= lratef;
	  }
	}
	
	weights = w0 - delta_grad;

	snet.importdata(weights);

	// recalculates error in dataset

	error.zeros();

	for(unsigned int i=0;i<data.size(0);i++){
	  
	  snet.calculate(data.access(0,i), err);
	  err -= data.access(1,i);
	  
	  for(unsigned int j=0;j<err.size();j++){
	    const auto& ej = err[j];
	    for(unsigned int k=0;k<ej.size();k++)
	      error += ninv*ej[k]*math::conj(ej[k]);

	    //error += ninv*math::sqrt(ej[0]*math::conj(ej[0]))/err.size();
	  }
	}

	auto abserror2 = error;
	abserror2[0] = abs(abserror2[0]);
	
	for(unsigned int i=1;i<abserror2.size();i++){
	  abserror2[0] += abs(abserror2[i]);
	  //abserror2[i] = math::blas_real<double>(0.0f);
	  abserror2[i] = 0.0f;
	}

	if(abserror2[0].real() < abserror[0].real()){
	  // error becomes smaller => found new better solution
	  lratef *= 1.10; // bigger step length..
	  abserror = abserror2;
	  break;
	}
	else{ // try shorter step length
	  lratef *= 1/1.10;
	  grad_search_counter++;
	}
	
      }

      // weights -= sumgrad + regularizer;
      // weights -= lrate * sumgrad + regularizer; // (alpha*weights);
      
      std::cout << counter << " [" << grad_search_counter << "] : " << abserror
		<< " (delta: " << (abserror-orig_error)[0] << ")"
		<< " (lrate: " << lratef << ")" 
		<< std::endl;

      error = abserror;
      
      counter++;
    }
    
    std::cout << counter << " : " << abs(error) << std::endl;

    math::vertex< math::superresolution<math::blas_real<double>,
					math::modular<unsigned int> > > params;
    snet.exportdata(params);
    std::cout << "nn solution weights = " << params << std::endl;
    
  }
  
  
    
  return 0;
}
