/* 
 * global optimizer test cases
 *
 * Tomas Ukkonen, 2023
 */

#include <iostream>
#include <vector>

#include "GlobalOptimizer.h"
#include "dataset.h"
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
    //feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    //feenableexcept(FE_INVALID);
  }
#endif
  
}


using namespace whiteice;


int main()
{
  std::cout << "global optimizer test cases" << std::endl;
  
  whiteice::logging.setPrintOutput(false);
  
  // creates training dataset
  std::cout << "Creating training dataset.." << std::endl;
  
  dataset< math::blas_real<double> > data2;

  data2.createCluster("input", 4);
  data2.createCluster("output", 4);  
    
  std::vector< math::vertex<math::blas_real<double> > > inputs2;
  std::vector< math::vertex<math::blas_real<double> > > outputs2;

  RNG< math::blas_real<double> > prng;
  
  math::vertex< math::blas_real<double> > x, y;
  
  
  const unsigned int NUMDATAPOINTS = 5000; // was: 1000, 100 for testing purposes

  for(unsigned int i=0;i<NUMDATAPOINTS;i++){

    math::blas_real<double> sigma = 4.0;
    math::blas_real<double> f = 10.0, a = 1.10, w = 10.0, one = 1.0;

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
    if(x[1].c[0] > 0.0f) y[2] += one;
    else y[2] -= one;
    if(x[3].c[0] > 0.0f) y[2] += one;
    else y[2] -= one;
    auto temp = math::cos(w*x[0]);
    if(temp.c[0] > 0.0f) y[2] += one;
    else y[2] -= one;
    
    y[3] = x[1]/(math::abs(x[0])+one) + x[2]*math::sqrt(math::abs(x[3])) + math::abs(x[3] - x[0]);

    inputs2.push_back(x);
    outputs2.push_back(y);

  }

  data2.add(0, inputs2);
  data2.add(1, outputs2);

  // do not do [now: enabled]
  data2.preprocess(0);
  data2.preprocess(1);

  if(data2.save("simpleproblem.ds") == false){
    printf("ERROR: saving data to file failed!\n");
    exit(-1); 
  }

  
#if 0
  // use SHA-256 HASH data instead
  {
    printf("USING SHA-256 HASH CRYPTO DATASET.\n");
    fflush(stdout);

    data.clear();
    data2.clear();
    data2.load("hash-data.ds");

    std::cout << "data.size(0): " << data.size(0) << std::endl;
    std::cout << "data.size(1): " << data.size(1) << std::endl;
    std::cout << "data2.size(0): " << data2.size(0) << std::endl;
    std::cout << "data2.size(1): " << data2.size(1) << std::endl;
    
    
    // remove preprocessings from data
    data2.convert(0);
    data2.convert(1);

    //data2.preprocess(0);
    //data2.preprocess(1);
    
    data.clear();
    data.createCluster("input", 10);
    data.createCluster("output", 10);

    x.resize(10);
    y.resize(10);
    sx.resize(10);
    sy.resize(10);
    
    for(unsigned int i=0;i<data2.size(0);i++){
      x = data2.access(0, i);
      y = data2.access(1, i);

      // std::cout << x << std::endl;
      
      //y = y/255.0; // convert to [-2,2]/255 valued data! [with preprocess] (small values work better!)

      // scales [0,1] data to [-0.5,0.5] data
      for(unsigned int i=0;i<y.size();i++){
	y[i] -= math::blas_real<double>(0.50);
      }
      
      whiteice::math::convert(sx, x);
      whiteice::math::convert(sy, y);

      // DON'T SWAP ANYMORE AS WE HAVE SWAPPED DATA ALREADY
      if(data.add(0, sx) == false) assert(0);
      if(data.add(1, sy) == false) assert(0);
    }


    data2.clear();
    data2.createCluster("input", 10);
    data2.createCluster("output", 10);

    for(unsigned int i=0;i<data.size(0);i++){
      sx = data.access(0, i);
      sy = data.access(1, i);

      whiteice::math::convert(x, sx);
      whiteice::math::convert(y, sy);

      data2.add(0, x);
      data2.add(1, y);
    }
    
    
    data.downsampleAll(1000); // should be at least 1000, was: 50
    data2.downsampleAll(1000); // should be at least 1000, was: 50
  }
#endif

  
  std::cout << "data2.size(0): " << data2.size(0) << std::endl;
  std::cout << "data2.size(1): " << data2.size(1) << std::endl;

  GlobalOptimizer< math::blas_real<double> > optimizer;

  std::vector< math::vertex< math::blas_real<double> > > xxdata;
  std::vector< math::vertex< math::blas_real<double> > > yydata;

  data2.getData(0, xxdata);
  data2.getData(1, yydata);

  double minrows = 50.0;

  math::blas_real<double> min_support = minrows/xxdata.size();

  std::cout << "min rows: " << minrows << std::endl;

  if(optimizer.startTrain(xxdata, yydata, min_support) == false){
    std::cout << "optimizer failed!" << std::endl;
    return -1;
  }

  while(optimizer.isRunning()){
    sleep(1);
    printf(".");
    fflush(stdout);
  }
  printf("\n");

  double error = -1.0;
  
  optimizer.getSolutionError(error);

  std::cout << "MODEL ERROR: " << error << std::endl;
  

  //////////////////////////////////////////////////////////////////////
  // report linear fitting absolute error

#if 0
  {
    std::vector< math::vertex< math::superresolution<
      math::blas_real<double>, math::modular<unsigned int> > > > xsamples;

    std::vector< math::vertex< math::superresolution<
      math::blas_real<double>, math::modular<unsigned int> > > > ysamples;

    data.getData(0, xsamples);
    data.getData(1, ysamples);

    const unsigned int SAMPLES = xsamples.size();

    math::matrix< math::superresolution< math::blas_real<double>,
		  math::modular<unsigned int> > > Cxx, Cxy;
    math::vertex< math::superresolution< math::blas_real<double>,
		  math::modular<unsigned int> > > mx, my;

    const auto& input = xsamples;
    const auto& output = ysamples;
    
    Cxx.resize(input[0].size(),input[0].size());
    Cxy.resize(input[0].size(),output[0].size());
    mx.resize(input[0].size());
    my.resize(output[0].size());
    
    Cxx.zero();
    Cxy.zero();
    mx.zero();
    my.zero();

    std::cout << "Calculate Matrixes Cxx, Cxy." << std::endl << std::flush;
    
    for(unsigned int i=0;i<SAMPLES;i++){
      Cxx += input[i].outerproduct();
      Cxy += input[i].outerproduct(output[i]);
      mx  += input[i];
      my  += output[i];
    }
    
    Cxx /= math::superresolution< math::blas_real<double>, math::modular<unsigned int> >((float)SAMPLES);
    Cxy /= math::superresolution< math::blas_real<double>, math::modular<unsigned int> >((float)SAMPLES);
    mx  /= math::superresolution< math::blas_real<double>, math::modular<unsigned int> >((float)SAMPLES);
    my  /= math::superresolution< math::blas_real<double>, math::modular<unsigned int> >((float)SAMPLES);
    
    Cxx -= mx.outerproduct();
    Cxy -= mx.outerproduct(my);

    math::matrix< math::superresolution< math::blas_real<double>,
					 math::modular<unsigned int> > > INV;
    
    math::superresolution< math::blas_real<double>, math::modular<unsigned int> > l =
      math::superresolution< math::blas_real<double>, math::modular<unsigned int> >(10e-20);

    std::cout << "Calculate Matrix Inverse." << std::endl << std::flush;
    
    do{
      INV = Cxx;
      
      math::superresolution< math::blas_real<double>, math::modular<unsigned int> > trace =
	math::superresolution< math::blas_real<double>, math::modular<unsigned int> >(0.0f);
      
      for(unsigned int i=0;(i<(Cxx.xsize()) && (i<Cxx.ysize()));i++){
	trace += Cxx(i,i);
	INV(i,i) += l; // regularizes Cxx (if needed)
      }
      
      if(Cxx.xsize() < Cxx.ysize())	  
	trace /= Cxx.xsize();
      else
	trace /= Cxx.ysize();
      
      l += (math::superresolution< math::blas_real<double>, math::modular<unsigned int> >(0.1)*trace +
	    math::superresolution< math::blas_real<double>, math::modular<unsigned int> >(2.0f)*l); // keeps "scale" of the matrix same
    }
    while(whiteice::math::symmetric_inverse(INV) == false);

    std::cout << "Calculate Matrix Inverse DONE." << std::endl << std::flush;

    math::matrix< math::superresolution< math::blas_real<double>,
					 math::modular<unsigned int> > > W;
    math::vertex< math::superresolution< math::blas_real<double>,
					 math::modular<unsigned int> > > b;
    
    W = (Cxy.transpose() * INV);
    b = (my - W*mx);

    // y = W*x + b

    math::superresolution< math::blas_real<double>, math::modular<unsigned int> > err, e;
    err.zero();

    for(unsigned int i=0;i<SAMPLES;i++){
      auto delta = W*input[i] + b - output[i];

      e.zero();

      for(unsigned int d=0;d<delta.size();d++)
	e += math::superresolution< math::blas_real<double>,
				    math:: modular<unsigned int> > (delta[d][0].abs());

      e /= delta.size();
      err += e;
    }

    err /= math::superresolution< math::blas_real<double>, math::modular<unsigned int> >((float)SAMPLES);

    std::cout << "Linear Fit Absolute Error: " << err[0] << std::endl;
  }
#endif

  
    
  return 0;
}
