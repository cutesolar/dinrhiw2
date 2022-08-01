
/*
 * simple tests: quaternion rotation of point p around 'q-axis'
 * with complex numbers, vertex, matrix, modular numbers, pdftree, hermite 
 * and bezier splines etc. everything in blade_math.
 */

#include <iostream>
#include <exception>
#include <stdexcept>
#include <complex>
#include <memory>
#include <string>
#include <cmath>
#include <new>

#include <chrono>
#include <random>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>

#ifdef WINOS
#include <windows.h>
#endif

#include "dinrhiw_blas.h"
#include "number.h"
#include "vertex.h"
#include "quaternion.h"
#include "matrix.h"
#include "hermite.h"
#include "bezier.h"
#include "pdftree.h"
#include "simplex.h"
#include "linear_equations.h"
#include "correlation.h"
#include "integer.h"
#include "modular.h"
#include "norms.h"
#include "real.h"
#include "eig.h"

#include "gmatrix.h"
#include "gvertex.h"

#include "RNG.h"
#include "Log.h"

using namespace whiteice;
using namespace whiteice::math;

void own_terminate();
void own_unexpected();

void rng_test();

void vertex_test();
void outerproduct_test();

number <quaternion<double>, double, double, unsigned int> * quaternion_test();

void matrix_test();
void inter_test();
void hermite_test();
void bezier_test();
void pdftree_test();
void simplex_test();
void real_test();
void fft_test();
void linear_equations_test();
void statistics_test();
void test_integer();
void modular_test();
//void compression_test();
void correlation_test();

void blas_compile_tests();
void blas_correctness_tests();

//////////////////////////////////////////////////////////////////////

/* uniform distribution */

int uniformi(int a, int b)
{
  int c = (int)((((float)rand()) / ((float)RAND_MAX)) * (b - a));
  c = c + a;
  
  return c;
}


float uniformf(float a, float b)
{
  float c = (((float)rand()) / ((float)RAND_MAX)) * (b - a);
  c = c + a;
  
  return c;
}



/* gaussian distribution: calculates mean of N numbers */

/*
 * should be quite well distributed
 */
float gaussianf(float mean, float var)
{
  const unsigned int N = 50;
  float sum = 0;
  
  // calculates necessary [a, -a] spread of
  // uniform variable in order to meet
  // target gaussian variance
  // (only in limit!)
  
  float a = whiteice::math::sqrt(3*N) * var;
  
  for(unsigned int i=0;i<N;i++){
    sum += uniformf(-a, a);
  }

  sum /= N;
  sum += mean; // adds mean
  
  return sum;  
}


int gaussiani(float mean, float var)
{
  float f = gaussianf(mean, var);
  return (int)(f + 0.5);
}


//////////////////////////////////////////////////////////////////////

void rng_test()
{
        RNG<> rng(true);
	
	srand(rng.rand());
	
	printf("RAND_MAX: %d\n", RAND_MAX);

	for(unsigned int i=0;i<10;i++)
		std::cout << rng.uniform() << std::endl;

	const unsigned int SAMPLES=10000000; // 10.000.000 random numbers
	
	math::vertex<> v;
	v.resize(SAMPLES);
	
	auto t0=std::chrono::high_resolution_clock::now();
	for(unsigned int i=0;i<v.size();i++){
	  v[i] = ((float)rand()/((float)RAND_MAX));
	}
	auto t1=std::chrono::high_resolution_clock::now();
	auto ccc_uni_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
	
	t0=std::chrono::high_resolution_clock::now();
	rng.uniform(v);
	t1=std::chrono::high_resolution_clock::now();
	auto own_uni_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

	math::blas_real<float> m, s;
	m = 0.0;
	s = 0.0;


	for(unsigned int i=0;i<v.size();i++){
		m += v[i];
		s += v[i]*v[i];
	}

	m /= v.size();
	s /= v.size();

	s -= m*m;
	s *= v.size()/((double)(v.size() - 1)); // sample variance..


	std::cout << "uniform distributin mean (should be 0.50000): " << m << std::endl;
	std::cout << "uniform distribution var (should be 0.08333): " << s << std::endl;

	t0=std::chrono::high_resolution_clock::now();
	rng.normal(v);
	t1=std::chrono::high_resolution_clock::now();
	auto own_nrm_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

	m = 0.0;
	s = 0.0;

	for(unsigned int i=0;i<v.size();i++){
	  m += v[i];
	  s += v[i]*v[i];
	}

	m /= v.size();
	s /= v.size();

	s -= m*m;
	s *= v.size()/((double)(v.size() - 1)); // sample variance..

	std::cout << "normal distributin mean: " << m << std::endl;
	std::cout << "normal distribution var: " << s << std::endl;

	v.resize(100000); // only saves 100.000 samples instead of 10.000.000
	v.saveAscii("normal_rnd.txt");
	v.resize(SAMPLES);

	t0=std::chrono::high_resolution_clock::now();
	rng.exp(v);
	t1=std::chrono::high_resolution_clock::now();
	auto own_exp_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

	m = 0.0;
	s = 0.0;

	for(unsigned int i=0;i<v.size();i++){
		m += v[i];
		s += v[i]*v[i];
	}

	m /= v.size();
	s /= v.size();

	s -= m*m;
	s *= v.size()/((double)(v.size() - 1)); // sample variance..

	std::cout << "exponential distributin mean: " << m << std::endl;
	std::cout << "exponential distribution var: " << s << std::endl;

	v.resize(100000); // only saves 100.000 samples instead of 10.000.000
	v.saveAscii("exp_rnd.txt");
	v.resize(SAMPLES);

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0,1.0);

	t0=std::chrono::high_resolution_clock::now();

	for(unsigned int i=0;i<v.size();i++)
		v[i] = distribution(generator);

	t1=std::chrono::high_resolution_clock::now();
	auto cpp_nrm_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

	std::cout << "Uniform RNG     [samples/ms]: " << ((double)SAMPLES)/own_uni_time << std::endl;
	std::cout << "Normal RNG      [samples/ms]: " << ((double)SAMPLES)/own_nrm_time << std::endl;
	std::cout << "Exponential RNG [samples/ms]: " << ((double)SAMPLES)/own_exp_time << std::endl;
	std::cout << "C Uniform RNG   [samples/ms]: " << ((double)SAMPLES)/ccc_uni_time << std::endl;
	std::cout << "C++ Normal RNG  [samples/ms]: " << ((double)SAMPLES)/cpp_nrm_time << std::endl;

}


//////////////////////////////////////////////////////////////////////

int main()
{
  std::set_terminate(own_terminate);
  std::set_unexpected(own_unexpected);
  
  srand(time(0));

  // whiteice::logging.setPrintOutput(true); // print logging messages to stdout
  whiteice::logging.setOutputFile("testcase.log");

  try{

    std::cout << "FFT TEST" << std::endl;
    fft_test();
    
    
    /*  
	std::cout << "STATISTICS CODE TESTS" << std::endl;
	statistics_test();
    */
    
    std::cout << "VERTEX TEST" << std::endl;
    vertex_test();

    std::cout << "OUTERPRODUCT TEST" << std::endl;
    outerproduct_test();
    
    /*
      std::cout << "QUATERNION TEST" << std::endl;  
  
      // delete through base class pointer: should be ok.
      number<quaternion<double>,double, double,unsigned int> * ptr;
      
      ptr = quaternion_test();
      delete ptr; // should call ~quaternion and then ~number
    */
    
    std::cout << "RNG TEST" << std::endl;
    rng_test();

    // return 0; // temp disable rest of the tests

    std::cout << "MATRIX TEST" << std::endl;
    std::cout << std::flush;
    matrix_test();
    
    std::cout << "INTERACTION TEST" << std::endl;
    inter_test();
    
    std::cout << "HERMITE INTERPOLATION TEST" << std::endl;
    hermite_test();
    
    std::cout << "BEZIER INTERPOLATION TEST" << std::endl; // NOT WORKING?
    bezier_test();
    
    std::cout << "PDFTREE TEST" << std::endl;
    pdftree_test();
    
    std::cout << "SIMPLEX TEST" << std::endl;
    simplex_test();
    
    std::cout << "LINEAR ALGEBRA TESTS" << std::endl;
    linear_equations_test();
    
    std::cout << "FREE LENGTH INTEGER TESTS" << std::endl;
    test_integer();
    
    std::cout << "MODULAR ARITHMETIC TESTS" << std::endl;
    modular_test();
    
    std::cout << "REAL TESTS" << std::endl;
    real_test();
    
    //std::cout << "COMPRESSION TESTS" << std::endl;
    //compression_test();
    
    std::cout << "ATLAS COMPILE TESTS" << std::endl;
    blas_compile_tests();
  
    std::cout << "ATLAS CORRECTNESS TESTS" << std::endl;
    blas_correctness_tests();
    
    std::cout << "(AUTO)CORRELATION CALCULATION TESTS" << std::endl;  
    correlation_test();
  }
  catch(std::exception& e){
    std::cout << "Exception: " << e.what() << std::endl;
  }
  
  return true;
}




void modular_test()
{
  
  // TEST CASE 1
  // basic +,-,*,/ operator tests
  try
  {
    modular<> a(7), b(7);
    modular<>* c;
    
    c = new modular<>;
    (*c)[0] = 1; (*c)[1] = 7;
    
    a = 1;
    b = 1;
    
    // printing test
    std::cout << "printing test:" << std::endl;
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "c = " << (*c) << std::endl;
    
    a += ((unsigned int)7)*b; // adds 7 = zero
    
    if(a != modular<>(1,7)){
      std::cout << "ERROR: combined '+=' and '* scalar' operator error"
		<< std::endl;
    }
    
    a *= 2;
    
    if(a != modular<>(2,7)){
      std::cout << "ERROR: '*= scalar' operator error"
		<< std::endl;
    }
    
    a /= b;
    
    if(a != modular<>(2,7)){
      std::cout << "ERROR: '!=' operator error"
		<< std::endl;
    }
    
    a = 1;
    b = 2;
    a /= b;
    
    if(a != modular<>(4,7)){
      std::cout << "ERROR: calculation of inverse failed"
		<< std::endl;
    }
    
    *c = a;
    
    
    b = 6;
    a -= b;
    
    if(a != modular<>(5, 7)){
      std::cout << "ERROR: '-=' operator error"
		<< std::endl;
    }
    
    delete c;
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception was throw: "
	      << e.what() << std::endl;
  }
  
  
  // TEST CASE 2, same as test case 2 but with bigger prime
  // basic +,-,*,/ operator tests
  try
  {
    modular<> a(10007), b(10007);
    modular<>* c;
    
    c = new modular<>;
    (*c)[0] = 1; (*c)[1] = 10007;
    
    a = 1;
    b = 1;
    
    a += ((unsigned int)10007)*b; // adds 10007 = zero
    
    if(a != modular<>(1,10007)){
      std::cout << "ERROR: combined '+=' and '* scalar' operator error"
		<< std::endl;
    }
    
    a *= 2;
    
    if(a != modular<>(2,10007)){
      std::cout << "ERROR: '*= scalar' operator error"
		<< std::endl;
    }
    
    a /= b;
    
    if(a != modular<>(2,10007)){
      std::cout << "ERROR: '!=' operator error"
	   << std::endl;
    }
    
    a = 1;
    b = 2;
    a /= b;
    
    if(a != modular<>(5004,10007)){
      std::cout << "ERROR: calculation of inverse failed"
	   << std::endl;
    }
    
    *c = a;
    
    
    b = 6004;    
    a -= b;
    
    if(a != modular<>(9007, 10007)){
      std::cout << "ERROR: '-=' operator error"
	   << std::endl;
    }
    
    
    delete c;
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception was throw: "
	 << e.what() << std::endl;
  }
  
  
}




void statistics_test()
{
  vertex<float> u,g;
  const unsigned int N = 1000;
  
  u.resize(N);
  g.resize(N);

  for(unsigned int i=0;i<u.size();i++){
    u[i] = uniformf(-4.0, 4.0);
    g[i] = gaussianf(0, 50);
  }
  
  float um, uv, gm, gv; // means, vars
  
  um = 0.0f;
  for(unsigned int i=0;i<u.size();i++) um += u[i];
  um /= (float)u.size();
  
  uv = 0.0f;
  for(unsigned int i=0;i<u.size();i++) uv += (u[i] - um)*(u[i] - um);
  uv /= ((float)u.size() - 1);
  
  gm = 0.0f;
  for(unsigned int i=0;i<g.size();i++) gm += g[i];
  gm /= (float)g.size();
  
  gv = 0.0f;
  for(unsigned int i=0;i<g.size();i++) gv += (g[i] - gm)*(g[i] - gm);
  gv /= ((float)g.size() - 1);

  // calculates mean that mean and variances are close enough
  // (TODO: in order to do this really well should calculate
  // some kind of 99% probability interval for calculated mean and var)
  
  if(whiteice::math::abs(um) >= 0.05)
    std::cout << "ERROR: uniform mean is too big\n";
  
  if(whiteice::math::abs(uv - ((8.0f*8.0f) / 12.0f)) >= 0.1f) // Var = (b-a)^2 / 12
    std::cout << "ERROR: uniform variance has incorrect value\n";
  
  if(whiteice::math::abs(gm) >= 0.05)
    std::cout << "ERROR: gaussian mean is too big\n";
  
  if(whiteice::math::abs(gv - 50.0f) >= 0.1f)
    std::cout << "ERROR: gaussian variance has incorrect value\n";
}




void linear_equations_test()
{
  matrix<float> A, B;
  vertex<float> x, b, y;
  
  //////////////////////////////////////////////////
  // LINEAR EQUATION SOLUTION TEST (Ax = b)
  
  std::cout << "LINEAR EQUATIONS TEST\n";
  
  // CASE 1. trivial Ix = b
  A.resize(4,4);
  B.resize(4,4);
  x.resize(4);
  b.resize(4);
  y.resize(4);
  
  A.identity();
  for(unsigned int i=0;i<x.size();i++)
    x[i] = rand() / (double)RAND_MAX;
  
  b = A*x;
  
  // tries to solve Ay = b
  if(!linsolve(A, y, b))
    std::cout << "ERROR: CASE1: linsolve() failed.\n";
  
  x -= y;
  if(x.norm() >= 0.1){
    std::cout << "ERROR: CASE1 bad solution.\n";
    std::cout << "solution was " << y << std::endl;
  }
  
  // CASE 2. random A
  // (correct code might fail because of rank defiency)

  for(unsigned int i=0;i<x.size();i++)
    x[i] = rand() / (double)RAND_MAX;
  
  for(unsigned int j=0;j<A.ysize();j++)
    for(unsigned int i=0;i<A.xsize();i++)
      A(j,i) = rand() / (double)RAND_MAX;
  
  b = A*x;
  y = b;
  
  std::cout << "Attempting to solve Ax = b." << std::endl;
  
  if(!linsolve(A, y, b))
    std::cout << "ERROR: CASE2: linsolve() failed.\n";
  
  x -= y;
  if(x.norm() >= 0.1){
    std::cout << "ERROR: CASE2 bad solution.\n";
    std::cout << "solution was " << y << std::endl;
  }
  else{
    std::cout << "OK: correct solution was found. error: " 
	      << x << std::endl;
  }
  
  // CASE3. Singular A
  // (should fail)

  for(unsigned int j=0;j<A.ysize();j++)
    for(unsigned int i=0;i<A.xsize();i++)
      A(j,i) = rand() / (double)RAND_MAX;

  for(unsigned int j=0;j<A.xsize();j++)
    A(2,j) = 0.0;

  for(unsigned int i=0;i<x.size();i++)
    x[i] = rand() / (double)RAND_MAX;
  
  b = A*x;
  y = b;
  
  if(linsolve(A, y, b))
    std::cout << "ERROR: CASE2: linsolve() succeeded with singular matrix.\n";
  
  //////////////////////////////////////////////////
  // LEAST SQUARES SOLUTION TESTS
  
  // CASE0. Symmetric matrix inverse test
  for(unsigned iters=0;iters<100;iters++){
    const unsigned int DIM = 20;
    
    A.resize(DIM,DIM);
    
    for(unsigned int j=0;j<A.ysize();j++)
      for(unsigned int i=0;i<A.xsize();i++)
	A(j,i) = rand() / (double)RAND_MAX;
    
    B = A;
    B.transpose();
    B = B * A; // B = A^t * A
    
    {
      A = B;
      
      if(symmetric_inverse(B) == false)
	std::cout << "ERROR: symmetric_inverse failed for symmetric (positive definite) matrix" << std::endl;
      
      auto I = A*B; // B should be inverse of A
      
      float error = 0.0f;
      
      for(unsigned int j=0;j<I.ysize();j++){
	for(unsigned int i=0;i<I.xsize();i++){
	  auto e = I(j,i);
	  if(i == j) e -= 1.0f;
	  error +=  whiteice::math::sqrt(e*e);
	}
      }
      
      error /= (I.ysize()*I.xsize());
      
      if(error > 10e-3){
	std::cout << "ERROR: symmetric_inverse is incorrect." << std::endl;
	std::cout << "ERROR: error = " << error << std::endl;
	
	matrix<float> D, X;
	D = A;
	
	if(whiteice::math::symmetric_eig(D, X) == false)
	  std::cout << "Cannot compute eigenvalues" << std::endl;
	else{
	  auto min = D(0,0);
	  
	  for(unsigned int i=1;i<D.xsize();i++){
	    if(D(i,i) < min) min = D(i,i);
	  }
	  
	  std::cout << "Minimum eigenvalue: " << min << std::endl;
	  
	}
	
      }
    }
  }
  
  
  A.resize(4,3);
  B.resize(4,3);
  x.resize(3);
  b.resize(4);
  y.resize(3);
  
  for(unsigned int i=0;i<x.size();i++)
    x[i] = rand() / (double)RAND_MAX;

  for(unsigned int j=0;j<A.ysize();j++)
    for(unsigned int i=0;i<A.xsize();i++)
      A(j,i) = rand() / (double)RAND_MAX;
    
  B = A;
  
  std::cout << "LEAST MEAN SQUARES FITTING TEST\n";
  
  // CASE1. perfectly solvable problem
  
  b = A*x;

  std::cout << "Attempting to solve *overdetermined* Ax = b." << std::endl;
  
  if(!linlsqsolve(A, b, y))
    std::cout << "ERROR: CASE1: linlsqsolve() failed.\n";
  
  x -= y;
  
  if(x.norm() >= 0.1){
    std::cout << "ERROR: CASE1: bad/wrong solution.\n";
    std::cout << "solution was " << y << std::endl;
  }
  else{
    std::cout << "CASE1: linear equation solution was ok.\n";
  }
  
  // CASE2. hand crafted not perfectly solvable problem
  // A*x = y
  
  A(0,0) = 0.54818; A(0,1) = 0.96315; A(0,2) = 0.26132;
  A(1,0) = 0.43843; A(1,1) = 0.20022; A(1,2) = 0.61354;
  A(2,0) = 0.24534; A(2,1) = 0.75681; A(2,2) = 0.37010;
  A(3,0) = 0.23282; A(3,1) = 0.37384; A(3,2) = 0.26581;
  
  y.resize(4);
  y[0] = 0.11744; y[1] = 0.35004;
  y[2] = 0.29385; y[3] = 0.38112;
  
  b.resize(3);
  x.resize(3); // correct solution
  x[0] = -0.44782; x[1] = 0.15708; x[2] = 0.90452;
  
  if(!linlsqsolve(A, y, b))
    std::cout << "ERROR: CASE2: linlsqlsolve() failed.\n";
  
  x -= b;
  
  if(x.norm() > 0.1){
    std::cout << "ERROR: CASE2: bad/wrong solution.\n";
    std::cout << "solution was " << b << std::endl;
  }
  else{
    std::cout << "CASE2: linear equation solution was ok.\n";
  }
  
}




void simplex_test()
{
  simplex<float> *s = 0;

#if 0
  // unbounded objective values
  // (this should return infinite
  //  as an optimum value)  
  
  s = new simplex<float>(2,2);
  std::vector<float> v;
  v.resize(2);
  v[0] = 2; v[1] = 1;
  s->setTarget(v);
  v.resize(3);
  v[0] = +1; v[1] = -1; v[2] = 10;
  s->setConstraint(0,v,0);
  
  v[0] = +2; v[1] = +0; v[2] = 40;
  s->setConstraint(1,v,0);
#endif

#if 0  
  // ???
  
  s = new simplex<float>(2,2);
  std::vector<float> v;
  v.resize(2);
  v[0] = 3; v[1] = 2;
  s->setTarget(v);
  v.resize(3);
  v[0] = 2; v[1] = 1; v[2] = 2;
  s->setConstraint(0,v,0);
  
  v[0] = 3; v[1] = 4; v[2] = 12;
  s->setConstraint(1,v,2);
#endif

  //#if 0
  // problem with alternative
  // optimas (infinite) (optimum value is 10)
  
  s = new simplex<float>(2,2);
  std::vector<float> v;
  v.resize(2);
  v[0] = 2; v[1] = 4;
  s->setTarget(v);
  v.resize(3);
  v[0] = 1; v[1] = 2; v[2] = 5;
  s->setConstraint(0,v,0);
  
  v[0] = 1; v[1] = 1; v[2] = 4;
  s->setConstraint(1,v,0);
  //#endif
  
#if 0
  // problem with multiple solutions
  // because of degeneracy
  
  s = new simplex<float>(2,2);
  std::vector<float> v;
  v.resize(2);
  v[0] = 3; v[1] = 9;
  s->setTarget(v);
  v.resize(3);
  v[0] = 1; v[1] = 4; v[2] = 8;
  s->setConstraint(0,v,0);
  
  v[0] = 1; v[1] = 2; v[2] = 4;
  s->setConstraint(1,v,0);
#endif
  

#if 0  
  // problem with basic pseudovars after
  // phase I
  
  s = new simplex<float>(3,2);
  std::vector<float> v;
  v.resize(3);
  v[0] = 3; v[1] = 2; v[2] = 3;
  s->setTarget(v);
  
  v.resize(4);
  
  v[0] = 2; v[1] = 1; v[2] = 1; v[3] = 2;
  s->setConstraint(0, v, 0);
  
  v[0] = 3; v[1] = 4; v[2] = 2; v[3] = 8;
  s->setConstraint(1, v, 2);
#endif

#if 0
  s = new simplex<float>(2,3);
  std::vector<float> v;
  v.resize(2);
  v[0] = -4; v[1] = -1; // negative coefficients: we want to minimize
  s->setTarget(v);
  v.resize(3);
  v[0] = 3; v[1] = 1; v[2] = 3;
  s->setConstraint(0, v, 1);
  v[0] = 4; v[1] = 3; v[2] = 6;
  s->setConstraint(1, v, 2);
  v[0] = 1; v[1] = 2; v[2] = 4;
  s->setConstraint(2, v, 0);
#endif
  
#if 0
  // 2 variables, 4 constraints
  s = new simplex<float>(2,4);

  std::vector<float> v;  
  v.resize(2);
  v[0] = 5; v[1] = 4;
  s->setTarget(v);
  
  v.resize(3);
  v[0] = 6; v[1] = 4;  v[2] = 24;
  s->setConstraint(0, v);
    
  v[0] = 1; v[1] = 2;  v[2] = 6;
  s->setConstraint(1, v);
  
  v[0] = -1; v[1] = 1;  v[2] = 1;
  s->setConstraint(2, v);
  
  v[0] = 0; v[1] = 1; v[2] = 2;
  s->setConstraint(3, v);  
#endif
  
  s->show_simplex();
  
  if(s->maximize()){
    
    // waits for result (1sec)
    {
      for(unsigned int i=0;i<100;i++){
#ifndef WINOS
    struct timespec ts;
	ts.tv_sec  = 0;
	ts.tv_nsec = 10000000; // 10ms
	nanosleep(&ts, 0);
#else
	Sleep(10);
#endif
	
	if(s->hasResult())
	  break;
      }
    }
    
    if(s->hasResult()){
      std::cout << "simplex optimization results" << std::endl;
      s->show_simplex();
    }
    else{
      std::cout << "OPTIMIZATION FAILED (BECAUSE BADLY FORMED PROBLEM/NO SOLUTION)" << std::endl;
    }
  }
  else{
    std::cout << "OPTIMIZATION TASK INITIALIZATION FAILURE" << std::endl;
  }

  delete s;
}





void fft_test()
{
  {
    std::cout << "FFT TEST 1 (whiteice::math::complex<float>)" << std::endl;
    
    std::vector< vertex<whiteice::math::complex<float> > > samples;  
    samples.resize(3);
    samples[0].resize(16); // input data
    samples[1].resize(16); // fft data
    samples[2].resize(16); // result of iff(fft(samples[0]))
    
    whiteice::RNG<> prng;
    
  // initialization, s[i] = i
    for(unsigned int i=0;i<samples[0].size();i++){
      whiteice::math::convert(samples[0][i], prng.uniform());
    }
    
    for(unsigned int j=1;j<samples.size();j++){
      for(unsigned int i=0;i<samples[j].size();i++){
	samples[j][i] = samples[0][i];
      }
    }
    
    if(!fft<4>(samples[1])) std::cout << "ERROR: FFT of samples[1] failed" << std::endl;
    if(!fft<4>(samples[2])) std::cout << "ERROR: FFT of samples[2] failed" << std::endl;  
    if(!ifft<4>(samples[2])) std::cout << "ERROR: IFFT of samples[2] failed" << std::endl;
    
    
    //std::cout << "original sample: " << std::endl;
    //std::cout << samples[0] << std::endl;
    
    //std::cout << "fft of sample: " << std::endl;
    //std::cout << samples[1] << std::endl;
    
    //std::cout << "inverse fft of sample's fft: " << std::endl;
    //std::cout << samples[2] << std::endl;
    
    
    // calculates distance between vertexes
    // should be close to zero
    samples[0] -= samples[2]; 
    
    if(std::real(whiteice::math::abs(samples[0].norm())) > 0.01){
      std::cout << "ifft(fft(X)) operation error: " << samples[0].norm() << std::endl;
      std::cout << "WARNING: IFFT(FFT(X)) error is suspiciosly large" << std::endl;
    }
    else
      std::cout << "GOOD: FFT and IFFT seems to work correctly. [IFFT(FFT(x)) == x]" << std::endl;
  }

  
  {
    std::cout << "FFT TEST 2 (blas_complex<float>)" << std::endl;
    
    std::vector< vertex<whiteice::math::blas_complex<float> > > samples;  
    samples.resize(3);
    samples[0].resize(16); // input data
    samples[1].resize(16); // fft data
    samples[2].resize(16); // result of iff(fft(samples[0]))
    
    whiteice::RNG<> prng;
    
  // initialization, s[i] = i
    for(unsigned int i=0;i<samples[0].size();i++){
      whiteice::math::convert(samples[0][i], prng.uniform());
    }
    
    for(unsigned int j=1;j<samples.size();j++){
      for(unsigned int i=0;i<samples[j].size();i++){
	samples[j][i] = samples[0][i];
      }
    }
    
    if(!fft<4>(samples[1])) std::cout << "ERROR: FFT of samples[1] failed" << std::endl;
    if(!fft<4>(samples[2])) std::cout << "ERROR: FFT of samples[2] failed" << std::endl;  
    if(!ifft<4>(samples[2])) std::cout << "ERROR: IFFT of samples[2] failed" << std::endl;
    
    
    //std::cout << "original sample: " << std::endl;
    //std::cout << samples[0] << std::endl;
    
    //std::cout << "fft of sample: " << std::endl;
    //std::cout << samples[1] << std::endl;
    
    //std::cout << "inverse fft of sample's fft: " << std::endl;
    //std::cout << samples[2] << std::endl;
    
    
    // calculates distance between vertexes
    // should be close to zero
    samples[0] -= samples[2]; 
    
    if(whiteice::math::real(whiteice::math::abs(samples[0].norm())) > 0.01){
      std::cout << "ifft(fft(X)) operation error: " << samples[0].norm() << std::endl;
      std::cout << "WARNING: IFFT(FFT(X)) error is suspiciosly large" << std::endl;
    }
    else
      std::cout << "GOOD: FFT and IFFT seems to work correctly. [IFFT(FFT(x)) == x]" << std::endl;
  }


  {
    std::cout << "CIRCULAR CONVOLUTION TEST" << std::endl;

    std::vector< vertex<whiteice::math::blas_complex<float> > > samples;  
    samples.resize(4);
    samples[0].resize(16); // input1 data
    samples[1].resize(16); // input2 data
    samples[2].resize(16); // result of ifft(fft(samples[0]).*fft(samples[1]))
    samples[3].resize(16);
    
    // initialization, s[i] = i
    for(unsigned int j=1;j<samples.size();j++){
      for(unsigned int i=0;i<samples[j].size();i++){
	samples[j][i] = 0.0f;
      }
    }

    samples[0][0] = +1.320;
    samples[0][1] = -0.3841;
    samples[0][2] = +0.1934;
    
    samples[1][0] = +0.7320;
    samples[1][1] = +1.4384;
    samples[1][2] = -1.9342;

    // results
    samples[3][0] = 0.9662;
    samples[3][1] = 1.6175;
    samples[3][2] = -2.9641;
    samples[3][3] = 1.0211;
    samples[3][4] = -0.3741;

    if(fft<4>(samples[0]) == false) std::cout << "FFT FAILED ERROR" << std::endl;
    if(fft<4>(samples[1]) == false) std::cout << "FFT FAILED ERROR" << std::endl;

    for(unsigned int i=0;i<samples[0].size();i++){
      samples[0][i] *= samples[1][i];
    }

    if(ifft<4>(samples[0]) == false) std::cout << "InvFFT FAILED ERROR" << std::endl;

    // std::cout << "circular convolution = " << samples[0] << std::endl;

    samples[0] -= samples[3];
    
    if(whiteice::math::real(whiteice::math::abs(samples[0].norm())) > 0.01){
      std::cout << "ifft(fft(X)) operation error: " << samples[0].norm() << std::endl;
      std::cout << "WARNING: CircularConvolution error is suspiciosly large" << std::endl;
    }
    else{
      std::cout << "GOOD: FFT/IFFT CircularConvolution seems to work correctly.]" << std::endl;
    }
    
  }


  {
    std::cout << "GENERAL PURPOSE BASIC DFT FOURIER TEST" << std::endl;

    std::vector< vertex<whiteice::math::blas_complex<float> > > samples;

    samples.resize(4);

    for(unsigned int i=0;i<samples.size();i++){
      samples[i].resize(1 + rng.rand()%64);

      for(unsigned int k=0;k<samples[i].size();k++){
	samples[i][k] = rng.uniformf();
      }

      auto orig = samples[i];

      basic_fft(samples[i]);

      auto delta = orig - samples[i];

      if(delta.norm().c[0] < 0.01){
	std::cout << "WARN: fft(x) don't change the x signal." << std::endl; 
      }
      
      basic_ifft(samples[i]);

      auto err = orig - samples[i];

      if(err.norm().c[0] > 0.01){
	std::cout << "ERROR: ifft(fft(x)) != x (|x|=" << samples[i].size() << ")." << std::endl;
	exit(-1);
      }
      else{
	std::cout << "GOOD: ifft(fft(x)) == x (|x|=" << samples[i].size() << ")." << std::endl; 
      }
      
    }
    
  }
}




void pdftree_test()
{
  pdftree<double, double> p;

  std::vector<double> min, max, v;
  min.resize(2);
  max.resize(2);
  v.resize(2);

  min[0] = 0;
  min[1] = 0;
  max[0] = 1;
  max[1] = 1;  

  p.reset(min, max);
  
  v[0] = 1; v[1] = 2;
  p.add(v);

  v[0] = 0; v[1] = 2;
  p.add(v);

  v[0] = 1; v[1] = 0;
  p.add(v);

  v[0] = 0; v[1] = 1;
  p.add(v);

  std::cout << "p(" << vertex<double>(v) << ") = "
       << p.pdf(v) << std::endl;  
}





void bezier_test()
{
  std::vector< vertex<float> > data;
  bezier<float> fcurve;
  
  vertex<float> v(4);
  v[0] = 1; v[1] = 2; v[2] = 3; v[3] = 4;
  data.push_back(v);
  v[2] = -10;
  data.push_back(v);
  v[1] = 0;
  v[2] = 0;
  v[3] = 0;
  data.push_back(v);
  v[0] = 1; v[1] = 2; v[2] = 3; v[3] = 4;
  data.push_back(v);
  

  fcurve(data);
  
  bezier<float>::iterator i;
  i = fcurve.begin();

  // TODO: better test: - assume that hermite interpolation is correct
  // and test that result is qualitatively correct (calculate generic
  // comparision code between curves)
  
  std::cout << "TODO: WRITE BETTER INTERPOLATION TEST\n";
  
#if 0
  std::cout << "interpolation results start" << std::endl;
  std::cout << "data:  ";
  for(unsigned int i=0;i<data.size();i++)
    std::cout << reinterpret_cast< vertex<float> >(data[i]) << "  ";
  std::cout << std::endl;

  std::cout << "inte:  ";
  while(i != fcurve.end()){
    std::cout << *i << "  ";
    i++;
  }
  std::cout << std::endl;
  
  std::cout << "interpolation results end" << std::endl; 
#endif
}




void hermite_test()
{
  std::vector< vertex<float> > data;
  class hermite< whiteice::math::vertex<float>, float> fcurve;
  
  vertex<float> v(4);
  v[0] = 1; v[1] = 2; v[2] = 3; v[3] = 4;
  data.push_back(v);
  v[2] = -10;
  data.push_back(v);
  v[1] = 0;
  v[2] = 0;
  v[3] = 0;
  data.push_back(v);
  v[0] = 1; v[1] = 2; v[2] = 3; v[3] = 4;
  data.push_back(v);
  

  fcurve(data);
  
  whiteice::math::hermite< whiteice::math::vertex<float>, float>::iterator i;
  i = fcurve.begin();

  // TODO: better test
  std::cout << "TODO: WRITE BETTER INTERPOLATION TEST\n";

#if 0
  std::cout << "interpolation results start" << std::endl;
  std::cout << "data:  ";
  for(unsigned int i=0;i<data.size();i++)
    std::cout << reinterpret_cast< vertex<float> >(data[i]) << "  ";
  std::cout << std::endl;

  std::cout << "inte:  ";
  while(i != fcurve.end()){
    std::cout << *i << "  ";
    i++;
  }
  std::cout << std::endl;
  
  std::cout << "interpolation results end" << std::endl;  
#endif

}


void real_test()
{
  // tests that the default number of bits is enough
  
  {
    realnumber r;
    std::cout << "Default GMP real precision: " 
	      << r.getPrecision() << " bits" << std::endl;
    std::cout << "IEEE double precision: "
	      << 52 << " bits" << std::endl;
    
    if(r.getPrecision() < 52){
      std::cout << "ERROR: " 
		<< "default GMP float's precision is lower than IEEE double's"
		<< std::endl;
    }
  }
  
  // to handle divide by zero problems 
  signal(SIGFPE, SIG_IGN);
  
  
  // random operations test
  // we calculate randomly generated sequence of
  // floating point operations
  
  // OP0:'+', OP1:'-', OP2:'*', OP3:'/'
  // OP4:'+=' OP5:'-=',OP6:'*=',OP7:'/='
  
  std::vector<unsigned char> opers;
  std::vector<unsigned int> indexes;
  std::vector<double> vd;
  std::vector<realnumber> rd;
  
  opers.resize(100);
  for(unsigned int i=0;i<100;i++)
    opers[i] = (rand() % 8);
  
  vd.resize(64);
  rd.resize(vd.size());
  for(unsigned int i=0;i<vd.size();i++){
    vd[i] = (double)(rand()/((double)RAND_MAX));
    rd[i] = vd[i];
  }
  
  
  indexes.resize(3*opers.size());
  for(unsigned int i=0;i<indexes.size();i++){
    indexes[i] = rand() % vd.size();
  }
  
  /////////////////////////////////////////////////
  
  for(unsigned int i=0;i<opers.size();i++){
    double a = vd[ indexes[3*i + 0] ];
    double b = vd[ indexes[3*i + 1] ];
    
    realnumber ra = rd[ indexes[3*i + 0] ];
    realnumber rb = rd[ indexes[3*i + 1] ];
    
    unsigned char op = opers[i];
    
    if(op == 0){ // PLUS
      vd[ indexes[3*i + 2] ] = a + b;
      rd[ indexes[3*i + 2] ] = ra + rb;
    }
    else if(op == 1){ // MINUS
      vd[ indexes[3*i + 2] ] = a - b;
      rd[ indexes[3*i + 2] ] = ra - rb;
    }
    else if(op == 2){ // MULTI
      vd[ indexes[3*i + 2] ] = a * b;
      rd[ indexes[3*i + 2] ] = ra * rb;
    }
    else if(op == 3){ // DIV
      if(b != 0.0){
	vd[ indexes[3*i + 2] ] = a / b;
	rd[ indexes[3*i + 2] ] = ra / rb;
      }
    }
    else if(op == 4){ // OP4:'+=' 
      vd[ indexes[3*i + 2] ] += a;
      rd[ indexes[3*i + 2] ] += ra;
    }
    else if(op == 5){ // OP5:'-=',
      vd[ indexes[3*i + 2] ] -= a;
      rd[ indexes[3*i + 2] ] -= ra;
    }
    else if(op == 6){ // OP6:'*=',
      vd[ indexes[3*i + 2] ] *= a;
      rd[ indexes[3*i + 2] ] *= ra;
    }
    else if(op == 7){ // OP7:'/='
      if(a != 0){
	vd[ indexes[3*i + 2] ] /= a;
	rd[ indexes[3*i + 2] ] /= ra;
      }
    }
    
  }
  
  
  for(unsigned int j=0;j<vd.size();j++){

    double tmp = fabs(rd[j].getDouble() - vd[j]);
#if 0
    realnumber tmp = rd[j];
    tmp -= vd[j];
      tmp.abs();
#endif
      
      if(tmp > 0.001)
	std::cout << "error too big: " << tmp << std::endl;
  }
  
  ////////////////////////////////////////////////////////////
  
  
  
}


void inter_test()
{
  vertex<> u, v, w;
  matrix<> A, B, M;
  quaternion<> q1, q2;

  /* determinant test */
  
  M.resize(4,4);
  v.resize(4);

  M(0,0) = 1; M(0,1) = 2; M(0,2) = 1; M(0,3) = 1;
  M(1,0) = 4; M(1,1) = 1; M(1,2) = 2; M(1,3) = 1;
  M(2,0) = 2; M(2,1) = 5; M(2,2) = 4; M(2,3) = 2;
  M(3,0) = 7; M(3,1) = 8; M(3,2) = 1; M(3,3) = 6;
  
  std::cout << "r = det(M) should be -6" << std::endl;
  std::cout << "M = " << M << std::endl;
  std::cout << "r = " << M.det() << std::endl;

  /* matrix * vertex */

  v[0] = 2; v[1] = 1; v[2] = 3; v[3] = 5;
  M(0,0) = 1; M(0,1) = 2; M(0,2) = 1; M(0,3) = 1;
  M(1,0) = 4; M(1,1) = 1; M(1,2) = 2; M(1,3) = 1;
  M(2,0) = 2; M(2,1) = 5; M(2,2) = 4; M(2,3) = 2;
  M(3,0) = 7; M(3,1) = 8; M(3,2) = 1; M(3,3) = 6;

  u = M*v;

  std::cout << "u = M*v , where" << std::endl;
  std::cout << "M = " << M << std::endl;
  std::cout << "v = " << v << std::endl;
  std::cout << "u = " << u << std::endl;
  std::cout << "u should be [ 12 20 31 55 ]" << std::endl;

  /* vertex * matrix */

  v[0] = 2; v[1] = 1; v[2] = 3; v[3] = 5;
  M(0,0) = 1; M(0,1) = 2; M(0,2) = 1; M(0,3) = 1;
  M(1,0) = 4; M(1,1) = 1; M(1,2) = 2; M(1,3) = 1;
  M(2,0) = 2; M(2,1) = 5; M(2,2) = 4; M(2,3) = 2;
  M(3,0) = 7; M(3,1) = 8; M(3,2) = 1; M(3,3) = 6;

  u = v*M;

  std::cout << "u = v*M, where" << std::endl;
  std::cout << "M = " << M << std::endl;
  std::cout << "v = " << v << std::endl;
  std::cout << "u = " << u << std::endl;
  std::cout << "u should be [ 47 60 21 39 ]" << std::endl;
  

  /* quaternion = vertex */

  q1 = v;
  std::cout << "q = v" << std::endl;
  std::cout << "v  = " << v << std::endl;
  std::cout << "q1 = " << q1 << std::endl;
  std::cout << std::endl;

  /* vertex = quaternion */

  u = q1;
  std::cout << "u = q" << std::endl;
  std::cout << "q = " << q1 << std::endl;
  std::cout << "u = " << u << std::endl;
  std::cout << std::endl;
}




void matrix_test()
{
  matrix<blas_real<double> > A, B, *C, D(3,3);  
  C = new matrix< blas_real<double> >; // 4x4 matrix
  blas_real<double> e;
  
  
  // MINIMAL MATRIX MULTIPLICATION + SUBTRACTION TESTS
  try {
    C->identity();
    
    B(2,2) = M_PI/2;
    
    B *= blas_real<double>(2.0);
        
    A = (*C) * B;
    
    *C = A - B; // should be zero
    
    e = 0.0;
    for(unsigned int j=0;j<C->ysize();j++)
      for(unsigned int i=0;i<C->xsize();i++)
	e += (*C)(j,i);
    
    e /= (double)(C->ysize() * C->xsize());
    e = whiteice::math::sqrt(e);
    
    if(e > 0.01)
      std::cout << "ERROR: matrix multiplication + subtract test failed.\n";
    
    if(whiteice::math::sqrt((A(2,2) - M_PI)*(A(2,2) - M_PI)) > 0.01)
      std::cout << "ERROR: matrix multiplication test failed.\n";
    
    delete C;
  }
  catch(std::exception& e){
    std::cout << "ERROR: matrix multiplication + substraction tests: ";
    std::cout << "uncaught exception: " << e.what() << std::endl;
  }
  
  
  // MATRIX INVERSE CODE TEST
  try {

    D.resize(4,4);
    A.resize(4,4);
    A(0,0) = 1; A(0,1) =-1; A(0,2) = 5; A(0,3) = 3;
    A(1,0) = 1; A(1,1) = 2; A(1,2) = 3; A(1,3) = 8;
    A(2,0) =-9; A(2,1) = 6; A(2,2) =-6; A(2,3) = 7;
    A(3,0) = 9; A(3,1) = 1; A(3,2) = 2; A(3,3) = 3;
    D = A;
    
    // CORRECT INVERSE (FROM OCTAVE)
    B.resize(4,4);
    B(0,0) = -0.1787; B(0,1) =  0.1179; B(0,2) = -0.0760; B(0,3) =  0.0418;
    B(1,0) =  2.0000; B(1,1) = -2.0000; B(1,2) =  1.0000; B(1,3) =  1.0000;
    B(2,0) =  1.1901; B(2,1) = -1.0190; B(2,2) =  0.4639; B(2,3) =  0.4449;
    B(3,0) = -0.9240; B(3,1) =  0.9924; B(3,2) = -0.4144; B(3,3) = -0.4221;

#ifdef CUBLAS
    // gpu_sync(); // must call sync operation after direct RAM access?
#endif
    
    if(A.inv() == false){
      std::cout << "ERROR: computation of matrix inverse FAILED." << std::endl;
      assert(0);
    }
    
    e = 0.0;
    for(unsigned j=0;j<A.ysize();j++)
      for(unsigned int i=0;i<A.xsize();i++)
	e += (A(j,i) - B(j,i)) * (A(j,i) - B(j,i));
    
    e /= (double)(A.ysize() * A.xsize());
    e = whiteice::math::sqrt(e);
    
    if(e > 0.01){
      std::cout << "ERROR: matrix inverse failed / is incorrect\n";
      assert(0);
    }
    
    B = D*A; // B should be identity
    
    e = 0.0;
    for(unsigned j=0;j<A.ysize();j++){
      for(unsigned int i=0;i<A.xsize();i++){
	if(j == i) e += (B(j,i) - 1) * (B(j,i) - 1);
	else e += B(j,i) * B(j,i);
      }
    }
    
    e /= (double)(A.ysize() * A.xsize());
    e = whiteice::math::sqrt(e);
    
    if(e > 0.01){
      std::cout << "ERROR: matrix inverse is incorrect\n";
      assert(0);
    }
    
    B = A*D; // B should be identity
    
    e = 0.0;
    for(unsigned j=0;j<A.ysize();j++){
      for(unsigned int i=0;i<A.xsize();i++){
	if(j == i) e += (B(j,i) - 1) * (B(j,i) - 1);
	else e += B(j,i) * B(j,i);
      }
    }
    
    e /= (double)(A.ysize() * A.xsize());
    e = whiteice::math::sqrt(e);
    
    if(e > 0.01){
      std::cout << "ERROR: matrix inverse is incorrect\n";
      assert(0);
    }

    std::cout << "MATRIX INVERSE CHECKS OK." << std::endl;
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: matrix inverse tests: ";
    std::cout << "uncaught exception: " << e.what() << std::endl;
  }
  
  
  // MATRIX TRANSPOSE TEST
  try {
    A.resize(rand() % 4 + 2, 2 + rand() % 6);
    int Ay = A.ysize();
    int Ax = A.xsize();
    B.resize(Ax, Ay);
    
    for(unsigned int j=0;j<A.ysize();j++){
      for(unsigned int i=0;i<A.xsize();i++){
	A(j,i) = rand() / (double)RAND_MAX;
      }
    }

    for(unsigned int j=0;j<A.ysize();j++){
      for(unsigned int i=0;i<A.xsize();i++){
	B(i,j) = A(j,i);
      }
    }

    std::cout << "original A = " << A << std::endl;
    
    A.transpose();
    
    if(A.xsize() != B.xsize() || 
       A.ysize() != B.ysize()){
      
      std::cout << "ERROR: matrix transpose failed:";
      std::cout << "wrong matrix size\n";
    }
    else{
      bool ok = true;
    
      for(unsigned int j=0;j<A.ysize() && ok;j++){
	for(unsigned int i=0;i<A.xsize() && ok;i++){
	  if(A(j,i) != B(j,i)){
	    std::cout << "ERROR: matrix transpose failed:";
	    std::cout << "Element (" 
		      << j << "," << i << ") differ\n";
	    std::cout << "A = " << A << std::endl;
	    std::cout << "B = " << B << std::endl;
	    ok = false;
	    break;
	  }
	}
      }
    }
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: matrix transpose tests: ";
    std::cout << "uncaught exception: " << e.what() << std::endl;
  }
}




number <quaternion<double>, double, double, unsigned int>* 
quaternion_test()
{
  quaternion<double> q, p, r, *axis = 0;
  number<quaternion<double>,double, double, unsigned int> *ptr = 0;
  double e;
  
  axis = new quaternion<double>;
  ptr = 
    dynamic_cast< number< quaternion<double>,double, double, unsigned int>* >(axis);
    
  
  try{
    r[1] = 1.5;
    r *= 2;
    
    // r should be [0 3 0 0]
    e = 0;
    e += r[0]*r[0]; e += (r[1] - 3)*(r[1] - 3);
    e += r[2]*r[2]; e += r[3]*r[3];
    e /= 4.0; e = whiteice::math::sqrt(e);
    if(e > 0.01) std::cout << "ERROR: quaterion '=' and/or '*' with scalar failed.\n";
    
    
    r[3] = 3;
    r = r + r;
    
    // r should be [0 6 0 6]
    e += r[0]*r[0]; e += (r[1] - 6)*(r[1] - 6);
    e += r[2]*r[2]; e += (r[3] - 6)*(r[3] - 6);
    e /= 4.0; e = whiteice::math::sqrt(e);
    if(e > 0.01) std::cout << "ERROR: quaterion '=' and/or '+' failed.\n";
    
    r -= r;
    
    // r should be [0 0 0 0]
    e += r[0]*r[0]; e += r[1]*r[1];
    e += r[2]*r[2]; e += r[3]*r[3];
    e /= 4.0; e = whiteice::math::sqrt(e);
    if(e > 0.01) std::cout << "ERROR: quaterion '-='  failed.\n";
  }
  catch(std::exception& e){
    std::cout << "ERROR: quaterion basic tests: exception was thrown: ";
    std::cout << e.what() << std::endl;
  }
  
  
  try{
    r.inv();
    std::cout << "ERROR: exception should have been throwed." << std::endl;
  }
  catch(std::exception& e){
  }
  catch(...){
    std::cout << "ERROR: unknown exception/error. trying to ignore." << std::endl;
  }
  
  // bogus data
  
  try{
    p[0] = 0;     // obsolette when rotating / rotation happens in '3d complex space'
    p[1] = +1.1;  // x coordinate
    p[2] = -0.2;  // y
    p[3] = +12.1; // z
    
    (*axis)[0] = 0;
    (*axis)[1] = +1;
    (*axis)[2] = +3.1415927;
    (*axis)[3] = -1.141;
    
    axis->normalize(); double alpha = M_PI;
    
    std::cout << "axis = " << (*axis) << std::endl;
    
    q.setup_rotation( alpha, *axis ); // alpha radians around axis
    
    r = q * p * q.inv(); // rotation
    
    std::cout << "result of rotation: " << r << std::endl;
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception: ";
    std::cout << e.what() << std::endl;
  }
  
  std::cout << "Q TESTS DONE: " << ptr << std::endl;
  std::cout.flush();
  
  return ptr;
}





void vertex_test()
{
  try{
    std::cout << "vertex<double> test" << std::endl;
    
    vertex<double> v(3), w(3), a(3);
    
    v[0] = 7;
    w[1] = 7;

    std::cout << "initial v = " << v << std::endl;
    std::cout << "initial w = " << w << std::endl;
    
    v /= w[1];
    w[1] /= 7;
    
    std::cout << "v and w should be [1 0 0] and [0 1 0], respectively." << std::endl;
    std::cout << "v = " << v << std::endl;
    std::cout << "w = " << w << std::endl;

    a = v ^ w;

    std::cout << v << " x " << w;
    std::cout << " = " << a << std::endl;

    a = v + w;

    std::cout << v << " + " << v;
    std::cout << " = " << a << std::endl;
    w = a;

    v.normalize();
    w.normalize();

    a.resize(1);
    a = v * w;

    std::cout << v << " * " << w;
    std::cout << " = " << a << std::endl;

    v.resize(4); w.resize(4);

    v[0] = 1; v[1] = 2; v[2] = 1; v[3] = 1;
    w[0] = 2; w[1] = 1; w[2] = 3; w[3] = 5;

    a = v * w;
    std::cout << v << " * " << w << " = " << a << std::endl;
    a = w * v;
    std::cout << w << " * " << v << " = " << a << std::endl;
    
  }
  catch(whiteice::exception& e){
    std::cout << "exception: " << e.what() << std::endl;
  }

  try{
    std::cout << "vertex< blas_complex<double> > test" << std::endl;
    
    vertex< blas_complex<double> > v(3), w(3), a(3);
    
    v[0] = 7;
    w[1] = 7;

    v /= w[1];
    w[1] /= 7.0;
    
    std::cout << "v and w should be [1 0 0] and [0 1 0], respectively." << std::endl;
    std::cout << "v = " << v << std::endl;
    std::cout << "w = " << w << std::endl;

    a = v ^ w;

    std::cout << v << " x " << w;
    std::cout << " = " << a << std::endl;

    a = v + w;

    std::cout << v << " + " << v;
    std::cout << " = " << a << std::endl;
    w = a;

    v.normalize();
    w.normalize();

    a.resize(1);
    a = v * w;

    std::cout << v << " * " << w;
    std::cout << " = " << a << std::endl;

    v.resize(4); w.resize(4);

    v[0] = 1; v[1] = 2; v[2] = 1; v[3] = 1;
    w[0] = 2; w[1] = 1; w[2] = 3; w[3] = 5;

    a = v * w;
    std::cout << v << " * " << w << " = " << a << std::endl;
    a = w * v;
    std::cout << w << " * " << v << " = " << a << std::endl;
    
  }
  catch(whiteice::exception& e){
    std::cout << "exception: " << e.what() << std::endl;
  }
}

////////////////////////////////////////////////////////////

void outerproduct_test()
{
  {
    std::cout << "FAST OUTERPRODUCT TEST" << std::endl;
    
    matrix< blas_real<double> > R1, R2;
    vertex< blas_real<double> > a, b;
    blas_real<double> alpha = 1.0;
    
    const unsigned int rows = rand() % 100;
    const unsigned int cols = rand() % 100;

    a.resize(rows);
    b.resize(cols);

    R1.resize(rows, cols);
    R2.resize(rows, cols);
    R1.zero();
    R2.zero();

    for(unsigned int r=0;r<rows;r++) a[r] = rand() / ((double)RAND_MAX);
    for(unsigned int c=0;c<cols;c++) b[c] = rand() / ((double)RAND_MAX);

    R1 = a.outerproduct(b);
    addouterproduct(R2, alpha, a, b);

    R1 -= R2;

    auto error = frobenius_norm(R1);

    if(error > 0.01){
      std::cout << "ERROR: in fast outerproduct computation" << std::endl;
    }
    else{
      std::cout << "Error is within safe limits: " << error 
		<< ". Good." << std::endl;
    }

    error = frobenius_norm(R2);

    if(error <= 0.001){
      std::cout << "ERROR: in fast outerproduct computation" << std::endl;
    }
    else{
      std::cout << "Outer product produces non-zero matrix: " << error 
		<< ". Good." << std::endl;
    }
    
  }
  
}

////////////////////////////////////////////////////////////



void test_integer()
{
  
  // TEST 1
  // performs random additions and substractions
  try{
    
    std::cout << "INTEGER ADDITION AND SUBSTRACTION TESTS" 
	      << std::endl;
    std::cout << std::dec << std::noshowbase << std::endl;
    
    whiteice::math::integer i, old_i;
    whiteice::math::integer j, old_j;
    
    int ii, jj, old_ii, old_jj;
    
    unsigned int t = 0;
    
    i = 1;
    j = 1;
    ii = 1;
    jj = 1;
    
    const unsigned int T = 1000;

    std::string opname;
    char buf[100];
    
    
    while(t < T){
      old_ii = ii;
      old_jj = jj;
      old_i = i;
      old_j = j;
      
      opname = "<no operator>";
      
      unsigned int k = rand() % 16;
      double r1 = (2.0*(( ((double)rand())/((double)RAND_MAX) ) - 0.5));
      double r2 = (2.0*(( ((double)rand())/((double)RAND_MAX) ) - 0.5));      
      
      if(k == 0){ // + random
	int ri = (int)(r1*1000);
	int rj = (int)(r2*1000);	
	
	sprintf(buf,"<plus random: %d, %d>", ri, rj);
	opname = buf;
		
	
	i += ri;
	ii += ri;
	j += rj;
	jj += rj;
      }
      else if(k == 1){ // - random
	int ri = (int)(r1*1000);
	int rj = (int)(r2*1000);
	
	sprintf(buf,"<minus random: %d, %d>", ri, rj);
	opname = buf;
	
	i -= ri;
	ii -= ri;
	j -= rj;
	jj -= rj;
      }
      else if(k == 2){ // i += j
	opname = "<i += j>";
	
	i += j;
	ii += jj;
      }
      else if(k == 3){ // i -= j;
	opname = "<i -= j>";
	
	i -= j;
	ii -= jj;
      }
      else if(k == 4){ // j += i;
	opname = "<j += i>";
	
	j += i;
	jj += ii;
      }
      else if(k == 5){ // j -= i;
	opname = "<j -= i>";
	
	j -= i;
	jj -= ii;
      }
      else if(k == 6){ // i = i + j;
	opname = "<i = i + j>";
	
	i = i + j;
	ii = ii + jj;
      }
      else if(k == 7){ // i = j + i;
	opname = "<i = j + i>";
	
	i = j + i;
	ii = jj + ii;
      }
      else if(k == 8){ // i = i - j;
	opname = "<i = i - j>";
	
	i = i - j;
	ii = ii - jj;
      }
      else if(k == 9){ // i = j - i;
	opname = "<i = j - i>";
	
	i = j - i;
	ii = jj - ii;
      }
      else if(k == 10){ // j = j + i;
	opname = "<j = j + i>";
	
	j = j + i;
	jj = jj + ii;
      }
      else if(k == 11){ // j = i + j;
	opname = "<j = i + j>";
	
	j = i + j;
	jj = ii + jj;
      }
      else if(k == 12){ // j = i - j;
	opname = "<j = i - j>";
	
	j = i - j;
	jj = ii - jj;
      }
      else if(k == 13){ // j = j - i;
	opname = "<j = j - i>";
	
	j = j - i;
	jj = jj - ii;
      }
      else if(k == 14 && (t & 3) == 0){ // 25 % chance
	// i = random
	int ri = (int)(r1*1000);	
	
	sprintf(buf,"<reset i: %d", ri);
	opname = buf;
	
	i = ri;
	ii = ri;
      }
      else if(k == 15 && (t & 3) == 0){ // 25 % chance
	// j = random
	int rj = (int)(r2*1000);

	sprintf(buf,"<reset j: %d", rj);
	opname = buf;
	
	j = rj;
	jj = rj;
      }
      
      
      // checks results

      int i_conv = i.to_int();
      int j_conv = j.to_int();
      
      
      if(ii != i_conv || jj != j_conv){	
	std::cout 
	  << "iter(" << t << "): results mismatch" << std::endl;
	
	std::cout 
	  << "(reasons: bug *or* native C integer overflow where as arbitrary precision still works)"
	  << std::endl;
	
	std::cout << opname << std::endl;
	std::cout << "orig_i:  " << old_i  << " orig_j:  " << old_j << std::endl;
	std::cout << "orig_ii: " << old_ii << " orig_jj: " << old_jj << std::endl;
	std::cout << "new_i: " << i << " new_j: " << j << std::endl;
	std::cout << "correct_i: " << ii << " correct_j: " << jj << std::endl;
	std::cout << std::endl;
	  
	
	t = T;
      }
      
      t++;
    }
    
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception. " 
	      << e.what() << std::endl;
  }


  // TEST 2
  // creates random numbers and compares them
  try{
    std::cout << "INTEGER RANDOM NUMBER COMPARISION TESTS"
	      << std::endl;
    

    whiteice::math::integer i, old_i;
    whiteice::math::integer j, old_j;
    
    int ii, jj, old_ii, old_jj;
    
    unsigned int t = 0;
    
    i = 1;
    j = 1;
    ii = 1;
    jj = 1;
    
    const unsigned int T = 1000;

    
    while(t < T){
      old_ii = ii;
      old_jj = jj;
      old_i = i;
      old_j = j;
      
      double r1 = (2.0*(( ((double)rand())/((double)RAND_MAX) ) - 0.5));
      double r2 = (2.0*(( ((double)rand())/((double)RAND_MAX) ) - 0.5));      
      
      int ri = (int)(r1*1000);
      int rj = (int)(r2*1000);	
	
      i = ri;
      ii = ri;
      j = rj;
      jj = rj;
     
      // checks comparision results
      
      if((old_ii <= ii) != (old_i <= i)){
	
	std::cout << "iter(" << t << "): comparision mismatch" << std::endl;
	std::cout << "=<" << std::endl;
	std::cout << "orig_i:  " << old_i  << " orig_j:  " << old_j << std::endl;
	std::cout << "new_i: " << i << " new_j: " << j << std::endl;
	
	std::cout << "orig_ii: " << old_ii << " orig_jj: " << old_jj << std::endl;	
	std::cout << "new_ii: " << ii << " orig_j: " << jj << std::endl;
	std::cout << std::endl;
	
	t = T;
	
      }
      else if((old_jj >= jj) != (old_j >= j)){
	
	std::cout << "iter(" << t << "): comparision mismatch" << std::endl;
	std::cout << ">=" << std::endl;
	std::cout << "orig_i:  " << old_i  << " orig_j:  " << old_j << std::endl;
	std::cout << "new_i: " << i << " new_j: " << j << std::endl;
	
	std::cout << "orig_ii: " << old_ii << " orig_jj: " << old_jj << std::endl;	
	std::cout << "new_ii: " << ii << " orig_j: " << jj << std::endl;
	std::cout << std::endl;
	
	t = T;
	
      }
      
      t++;
    }
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception. " 
	      << e.what() << std::endl;
  }


  // TEST3
  // creates random multiplications and divisions
  try{
    
    std::cout << "INTEGER MULTIPLICATION AND DIVISION TESTS"
	      << std::endl;
    
    
    // multiplication test
    {
      whiteice::math::integer i, old_i;
      whiteice::math::integer j, old_j;
    
      int ii, jj; // , old_ii, old_jj;
      
      unsigned int t = 0;
      
      i = 1;
      j = 1;
      ii = 1;
      jj = 1;
      
      const unsigned int T = 1000;
      
      
      while(t < T){
	//old_ii = ii;
	//old_jj = jj;
	old_i = i;
	old_j = j;
	
	double r1 = (2.0*(( ((double)rand())/((double)RAND_MAX) ) - 0.5));
	double r2 = (2.0*(( ((double)rand())/((double)RAND_MAX) ) - 0.5));
	
	int ri = (int)(r1*10000);
	int rj = (int)(r2*10000);
	
	i = ri;   j = rj;
	ii = ri; jj = rj;
	
	i  *= j;
	ii *= jj;
	
	
	// converts i and j to integers
	
	int i_conv = i.to_int();
	
	
	// checks comparision results
	
	if(i_conv != ii){
	  
	  std::cout << "iter(" << t << "): multiplication mismatch" << std::endl;
	  std::cout << "ri *= rj" << std::endl;
	  std::cout << "orig ri: " << ri << " orig rj: " << rj << std::endl;
	  std::cout << "i: " << i << " j: " << j << std::endl;	  
	  std::cout << "ii: " << ii << " jj: " << jj << std::endl;
	  std::cout << std::endl;
	  
	  t = T;
	  
	}
	
	t++;
      }
    }
    
    
    // integer division test
    {
      whiteice::math::integer i, old_i;
      whiteice::math::integer j, old_j;
    
      int ii, jj; // , old_ii, old_jj;
      
      unsigned int t = 0;
      
      i = 1;
      j = 1;
      ii = 1;
      jj = 1;
      
      const unsigned int T = 1000;
      
      
      while(t < T){
	// old_ii = ii;
	// old_jj = jj;
	old_i = i;
	old_j = j;
	
	double r1 = (2.0*(( ((double)rand())/((double)RAND_MAX) ) - 0.5));
	double r2 = (2.0*(( ((double)rand())/((double)RAND_MAX) ) - 0.5));
	
	int ri = (int)(r1*10000);
	int rj = (int)(r2*10000);
	
	i = ri;   j = rj;
	ii = ri; jj = rj;
	
	i  /= j;
	ii /= jj;
	
	
	// converts i and j to integers
	
	int i_conv = i.to_int();
	
	// checks comparision results
	
	if(i_conv != ii){
	  
	  std::cout << "iter(" << t << "): division mismatch" << std::endl;
	  std::cout << "ri /= rj" << std::endl;
	  std::cout << "orig ri: " << ri << " orig rj: " << rj << std::endl;
	  std::cout << "i: " << i << " j: " << j << std::endl;	  
	  std::cout << "ii: " << ii << " jj: " << jj << std::endl;
	  std::cout << std::endl;
	  
	  t = T;
	  
	}
	
	t++;
      }
    }
    
    
    
    // integer modulo test
    {
      whiteice::math::integer i, old_i;
      whiteice::math::integer j, old_j;
    
      int ii, jj; // , old_ii, old_jj;
      
      unsigned int t = 0;
      
      i = 1;
      j = 1;
      ii = 1;
      jj = 1;
      
      const unsigned int T = 1000;
      
      
      while(t < T){
	//old_ii = ii;
	//old_jj = jj;
	old_i = i;
	old_j = j;
	
	double r1 = (2.0*(( ((double)rand())/((double)RAND_MAX) ) - 0.5));
	double r2 = (2.0*(( ((double)rand())/((double)RAND_MAX) ) - 0.5));
	
	int ri = (int)(r1*10000);
	int rj = (int)(r2*10000);
	
	i = ri;   j = rj;
	ii = ri; jj = rj;
	
	i  %= j;
	ii %= jj;
	
	
	// converts i and j to integers
	
	int i_conv = i.to_int();
	
	// checks comparision results
	
	if(i_conv != ii){
	  
	  std::cout << "iter(" << t << "): modulo mismatch" << std::endl;
	  std::cout << "ri %= rj" << std::endl;
	  std::cout << "orig ri: " << ri << " orig rj: " << rj << std::endl;
	  std::cout << "i: " << i << " j: " << j << std::endl;	  
	  std::cout << "ii: " << ii << " jj: " << jj << std::endl;
	  std::cout << std::endl;
	  
	  t = T;
	  
	}
	
	t++;
      }
    }
    
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception. " 
	      << e.what() << std::endl;
  }
  
  
}


#if 0

void compression_test()
{
  try{
    bool tests_ok = true;
    
    // vertex compression tests
    
    for(unsigned int j=0;j<10;j++)
    {
      vertex< blas_real<float> > a, b;
      a.resize( 1 + rand() % 7461 );
      b.resize( a.size() );
      
      for(unsigned int i=0;i<a.size();i++){
	a[i] = rand() / ((float)RAND_MAX);
	b[i] = a[i];
      }
      
      if(a.iscompressed() == true ||
	 b.iscompressed() == true){
	std::cout << "vertex::iscompressed() returned wrong value"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      
      
      if(a.compress() == false){
	std::cout << "vertex::compress() failed"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      if(a.iscompressed() == false){
	std::cout << "vertex::iscompressed() returned wrong value"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      if(a.decompress() == false){
	std::cout << "vertex::decompress() failed"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      if(a.iscompressed() == true){
	std::cout << "vertex::iscompressed() returned wrong value"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      for(unsigned int i=0;i<b.size();i++){
	if(a[i] != b[i]){
	  std::cout << "vertex comparision test 1 failed"
		    << std::endl;
	  tests_ok = false;
	  break;
	}
      }
      
      
      
      
      if(b.compress() == false){
	std::cout << "vertex::compress() failed"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      if(b.iscompressed() == false){
	std::cout << "vertex::iscompressed() returned wrong value"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      if(b.decompress() == false){
	std::cout << "vertex::decompress() failed"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      if(b.iscompressed() == true){
	std::cout << "vertex::iscompressed() returned wrong value"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      
      for(unsigned int i=0;i<b.size();i++){
	if(a[i] != b[i]){
	  std::cout << "vertex comparision test 2 failed"
		    << std::endl;
	  tests_ok = false;
	  break;
	}
      }
      
    }
    
    if(tests_ok == true)
      std::cout << "VERTEX CLASS COMPRESSION TESTS: PASSED" << std::endl;
    
    tests_ok = true;
    
    // matrix compression tests

    for(unsigned int j=0;j<10;j++)
    {
      matrix< blas_real<float> > a, b;
      a.resize( 1 + rand() % 34, 1 + rand() % 34);
      b.resize( a.ysize(), a.xsize() );
      

      for(unsigned int y=0;y<a.ysize();y++){
	for(unsigned int x=0;x<a.xsize();x++){
	  a(y,x) = rand() / ((float)RAND_MAX);
	  b(y,x) = a(y,x);
	}
      }
      
      
      if(a.iscompressed() == true ||
	 b.iscompressed() == true){
	std::cout << "matrix::iscompressed() returned wrong value"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      
      
      if(a.compress() == false){
	std::cout << "matrix::compress() failed"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      if(a.iscompressed() == false){
	std::cout << "matrix::iscompressed() returned wrong value"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      if(a.decompress() == false){
	std::cout << "matrix::decompress() failed"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      if(a.iscompressed() == true){
	std::cout << "matrix::iscompressed() returned wrong value"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      
      
      for(unsigned int y=0;y<a.ysize();y++){
	for(unsigned int x=0;x<a.xsize();x++){
	  if(a(y,x) != b(y,x)){
	    std::cout << "matrix comparision test 1 failed"
		      << std::endl;
	    tests_ok = false;
	  }
	}
      }
      
      if(tests_ok == false)
	break;
      
      
      if(b.compress() == false){
	std::cout << "matrix::compress() failed"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      if(b.iscompressed() == false){
	std::cout << "matrix::iscompressed() returned wrong value"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      if(b.decompress() == false){
	std::cout << "matrix::decompress() failed"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      if(b.iscompressed() == true){
	std::cout << "matrix::iscompressed() returned wrong value"
		  << std::endl;
	tests_ok = false;
	break;
      }
      
      
      
      for(unsigned int y=0;y<a.ysize();y++){
	for(unsigned int x=0;x<a.xsize();x++){
	  if(a(y,x) != b(y,x)){
	    std::cout << "matrix comparision test 1 failed"
		      << std::endl;
	    tests_ok = false;
	  }
	}
      }
      
      if(tests_ok == false)
	break;
      
    }
    
    
    if(tests_ok == true)
      std::cout << "MATRIX CLASS COMPRESSION TESTS: PASSED" << std::endl;
    
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception. " 
	      << e.what() << std::endl;
  }

}
#endif


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


class test_exception : public std::exception
{
public:
  
  test_exception() 
  {
    reason = 0;
  }
  
  test_exception(const std::exception& e) 
  {
    reason = 0;
    
    const char* ptr = e.what();
    
    if(ptr)
      reason = (char*)malloc(sizeof(char)*(strlen(ptr) + 1));
    
    if(reason) strcpy(reason, ptr);
  }
  
  
  test_exception(const char* ptr) 
  {
    reason = 0;
    
    if(ptr){
      reason = (char*)malloc(sizeof(char)*(strlen(ptr) + 1));
      
      if(reason) strcpy(reason, ptr);
    }
  }
  
  
  virtual ~test_exception() 
  {
    if(reason) free(reason);
    reason = 0;
  }
  
  virtual const char* what() const throw()
  {
    if(reason == 0) // null terminated value
      return ((const char*)&reason);

    return reason;
  }
  
private:
  
  char* reason;
  
};


//////////////////////////////////////////////////////////////////////


void blas_correctness_tests()
{
  std::cout << "ATLAS DATA TYPE FUNCTIONALITY TESTS" << std::endl;
  
  // known bug in the implementation (don't care/ not critical):
  // test code leaks memory
  
  try{
    
    std::cout << "ATLAS SQRT() AND ABS() TESTS" << std::endl;
    {
      float a[10];
      blas_real<float> b[10];
      
      // sets up data
      for(unsigned int i=0;i<10;i++){
	a[i] = ((float)rand())/((float)RAND_MAX);
	b[i] = a[i];
      }
      
      for(unsigned int i=0;i<10;i++){
	a[i] = whiteice::math::sqrt(a[i]);
	b[i] = whiteice::math::sqrt(b[i]);
	
	float delta = a[i] - b[i].c[0];
	if(fabsf(delta) > 0.001f){
	  std::cout << "ERROR: blas_real square root returned wrong result" << std::endl;
	  std::cout << "A[i] = " << a[i] << "  B[i] = " << b[i].c[0] << std::endl;
	}
	  
	a[i] -= 0.5f;
	b[i] -= 0.5f;
      }
      
      
      for(unsigned int i=0;i<10;i++){
	float orig = a[i];
	blas_real<float> origb = b[i];
	a[i] = whiteice::math::abs(a[i]);
	b[i] = whiteice::math::abs(b[i]);
	
	float delta = a[i] - b[i].c[0];
	if(fabsf(delta) > 0.001f){
	  std::cout << "ERROR: blas_real abs() returned wrong result" << std::endl;
	  std::cout << "Input: " << orig << " , " << origb.c[0] << std::endl;
	  blas_real<float> bb = b[i];
	  std::cout << "A[i] = " << a[i] << "  B[i] = " << bb.c[0] << std::endl;
	}
      }
      
    }
    
    
    std::cout << "ATLAS GVERTEX TESTS" << std::endl;
    
    std::cout << "GVERTEX GENERATION TESTS\n";
    
    // only tests floats and real number implementation
    
    // gvertex generation and "=",[] operator, comparision, resize() tests
    {
      // float tests
      
      whiteice::math::gvertex<float>* fver;
      whiteice::math::vertex< blas_real<float> >* aver;
      
      // ctor, [] operator, some comparision and resize tests
      for(unsigned int i=0;i<100;i++){
	const unsigned int N = (rand() % 20) + 1;
	
	fver = new gvertex<float>(N);
	aver = new vertex< blas_real<float> >(N);
	
	if(fver->size() != aver->size())
	  throw test_exception("gvertex size mismatch");
	
	for(unsigned int j=0;j<N;j++){
	  float f = (rand() / ((float)RAND_MAX));
	  
	  (*fver)[j] = f;
	  (*aver)[j] = f;
	}
	
	// compares individual values
	for(unsigned int j=0;j<N;j++){
	  if( (*fver)[j] != (*aver)[j] )
	    throw test_exception("gvertex values mismatch");
	}
	
	// alternative ctor
	whiteice::math::gvertex<float> g2(*fver);
	whiteice::math::vertex< blas_real<float> > a2(*aver);
	
	if(g2 != *fver)
	  throw test_exception("generic gvertex: comparision error when used copy ctor");
	
	if(a2 != *aver)
	  throw test_exception("atlas   gvertex: comparision error when used copy ctor");
	
	if(g2.size() != a2.size())
	  throw test_exception("gvertex size mismatch (2)");	
	
	// compares individual values
	for(unsigned int j=0;j<N;j++){
	  if( g2[j] != a2[j] )
	    throw test_exception("gvertex values mismatch (2)");
	}
	
	// keeps smallening the size of the gvertexes to zero dimensions and
	// runs all the comparision tests after each resize()
	
	for(int s = N - 1; s >= 0; s--){
	  fver->resize(s);
	  aver->resize(s);
	  g2.resize(s);
	  a2.resize(s);
	  
	  if(fver->size() != a2.size())
	    throw test_exception("gvertex size mismatch (3)");
	  
	  if(aver->size() != g2.size())
	    throw test_exception("gvertex size mismatch (4)");
	  
	  // compares values
	  //   within implementations
	  for(unsigned int j=0;j<a2.size();j++){
	    if((*aver)[j] != a2[j])
	      throw test_exception("atlas gvertex values mismatch after resize()");
	  }
	  for(unsigned int j=0;j<g2.size();j++){
	    if((*fver)[j] != g2[j])
	      throw test_exception("generic gvertex values mismatch after resize()");
	  }

#if 0
	  //   between implementations
	  for(unsigned int j=0;j<a2.size();j++){
	    if(g2[j] != a2[j])
	      throw test_exception("in-between gvertex impl. comparision: values mismatch after resize()");
	  }
	  
	  // tests "==" operator
	  
	  if( ( (*aver) != a2 ) == true )
	    throw test_exception("atlas gvertex != operator error after resize()");
	  
	  if( ( (*fver) != g2 ) == true )
	    throw test_exception("generic gvertex != operator error after resize()");
#endif
	  
	}
	
	
	delete fver;
	delete aver;
      }
      
      std::cout << "COMPARISION TESTS\n";
      
      // "=" and "==" and "!=" operator tests
      {
	whiteice::math::vertex< blas_real<float> >* a;
	
	const unsigned int N = (rand() % 20) + 1;
	const unsigned int M = (rand() % 20) + 1;
	
	a = new vertex< blas_real<float> >[N];
	
	for(unsigned int i=0;i<N;i++)
	  a[i].resize(M);
	
	for(unsigned int i=0;i<M;i++)
	  a[0][i] = (rand()/((float)RAND_MAX));
	
	for(unsigned int i=1;i<N;i++)
	  a[i] = a[i-1];
	
	// checks for equality
	for(unsigned int i=0;i<N;i++){
	  if( (a[0] != a[i]) == true )
	    throw test_exception("atlas gvertex mismatch after '=' operator (1)");
	  
	  if( (a[0] == a[i]) == false )
	    throw test_exception("atlas gvertex mismatch after '=' operator (2)");
	}
	
	// makes random change in the all odd gvertexes
	for(unsigned i=0;i<N;i++){
	  if((i % 2) == 1){
	    unsigned int k = rand() % M;
	    
	    a[i][k] = (rand()/((float)RAND_MAX));
	  }
	}
	
	// checks that comparision with odd indeces are ok
	for(unsigned int i=0;i<N;i++){
	  
	  if((i % 2) == 1){
	    if( (a[0] != a[i]) == false )
	      throw test_exception("atlas gvertex comparision error after changes (1)");
	    
	    if( (a[0] == a[i]) == true )
	      throw test_exception("atlas gvertex comparision errro after changes (2)");
	  }
	  else{
	    if( (a[0] != a[i]) == true )
	      throw test_exception("atlas gvertex mismatch after '=' operator (3)");
	    
	    if( (a[0] == a[i]) == false )
	      throw test_exception("atlas gvertex mismatch after '=' operator (4)");
	  }
	}
	
	
	delete[] a;
      }
      
    }
    
    
    
    std::cout << "GVERTEX ADD/SUB TESTS\n";
    
    // gvertex addition and substraction tests
    {
      const unsigned int N = (rand() % 20) + 1;
      
      whiteice::math::gvertex<float> a(N), b(N), c(N);
      whiteice::math::vertex< blas_real<float> > d(N), e(N), f(N);
      std::string str;
      
      for(unsigned int i=0;i<1000;i++){ // number of iterations
      
	// initialize with random data
	for(unsigned int j=0;j<N;j++){
	  a[j] = rand() / ((float)RAND_MAX);
	  b[j] = rand() / ((float)RAND_MAX);
	  c[j] = rand() / ((float)RAND_MAX);
	  
	  d[j] = a[j];
	  e[j] = b[j];
	  f[j] = c[j];
	}
	
	
	// randomly selects on of the possible operations
	{
	  unsigned int optype = rand() % 4;
	  
	  if(optype == 0){      // '+' operator
	    str = "'+'-operator";
	    
	    c = a + b;
	    f = d + e;
	  }
	  else if(optype == 1){ // '-' operator
	    str = "'-'-operator";
	    
	    c = a - b;
	    f = d - e;
	  }
	  else if(optype == 2){ // '+=' operator
	    str = "'+='-operator";
	    
	    a += b;
	    d += e;
	  }
	  else if(optype == 3){ // '-=' operator
	    str = "'-='-operator";
	    
	    a -= b;
	    d -= e;
	  }
	  
	  // compares a <-> d, b <-> e, c <-> f
	  
	  for(unsigned int j=0;j<N;j++){
	    if(a[j] != d[j].c[0] || b[j] != e[j].c[0] || c[j] != f[j].c[0]){
	      std::cout << str << " FAILED\n";
	      std::cout << "a = " << a << std::endl;
	      std::cout << "b = " << b << std::endl;
	      std::cout << "c = " << c << std::endl;
	      std::cout << "d = " << d << std::endl;
	      std::cout << "e = " << e << std::endl;
	      std::cout << "f = " << f << std::endl;
	      throw test_exception("addition/substraction tests: comparision error");
	    }
	  }
	}
      }
    }

    
    
    std::cout << "GVERTEX SCALAR MULTIPLICATION  TESTS\n";
    
    // gvertex scalar multiplication tests
    {
      const unsigned int N = (rand() % 20) + 1;
      
      whiteice::math::gvertex<float> a(N), b(N);
      float c;
      
      whiteice::math::vertex< blas_real<float> > d(N), e(N);
      blas_real<float> f;
      
      std::string str;
      
      for(unsigned int i=0;i<1000;i++){ // number of iterations
      
	// initialize with random data
	for(unsigned int j=0;j<N;j++){
	  a[j] = rand() / ((float)RAND_MAX);
	  b[j] = rand() / ((float)RAND_MAX);	  
	  
	  d[j] = a[j];
	  e[j] = b[j];	  
	}
	
	c = rand() / ((float)RAND_MAX);
	f = c;
	
	
	// randomly selects on of the possible operations
	{
	  unsigned int optype = rand() % 5;
	  
	  if(optype == 0){      // 's * v' operator
	    str = "'s*v'-operator";
	    
	    b = c * a;
	    e = f * d;
	  }
	  else if(optype == 1){ // 'v * s' operator
	    str = "'v*s'-operator";
	    
	    b = a * c;
	    e = d * f;
	  }
	  else if(optype == 2){ // 'v *= s' operator
	    str = "'v*=s'-operator";
	    
	    a *= c;
	    d *= f;
	  }
	  else if(optype == 3){ // 'v / s' operator
	    str = "'v/s'-operator";
	    
	    b = a / c;
	    e = d / f;
	  }
	  else if(optype == 4){ // 'v /= s' operator
	    str = "'v/=s'-operator";
	    
	    a /= c;
	    d /= f;
	  }
	  
	  // compares a <-> d, b <-> e
	  
	  for(unsigned int j=0;j<N;j++){
	    float _f1 = a[j];
	    float _f2 = d[j].c[0];
	    float _e1 = b[j];
	    float _e2 = e[j].c[0];
	    
	    if(whiteice::math::abs(_f1 - _f2) > 0.001 ||
	       whiteice::math::abs(_e1 - _e2) > 0.001){
	      
	      std::cout << str << " FAILED\n";
	      std::cout << "index = " << j << std::endl;
	      std::cout << "a = " << a << std::endl;
	      std::cout << "b = " << b << std::endl;
	      std::cout << "c = " << c << std::endl;
	      std::cout << "d = " << d << std::endl;
	      std::cout << "e = " << e << std::endl;
	      std::cout << "f = " << f << std::endl;
	      throw test_exception("multiplication/divide tests: comparision error");
	    }
	  }
	  
	}
      }
    }
    
    
    
    std::cout << "INNER/CROSS PRODUCT TESTS\n";
    
    // gvertex inner/cross product tests
    {
      const unsigned int N = (rand() % 10) + 1;
      
      whiteice::math::gvertex<float> a(N), b(N), c(N);
      whiteice::math::vertex< blas_real<float> > d(N), e(N), f(N);
      std::string str;
      
      for(unsigned int i=0;i<1000;i++){ // number of iterations
	
	a.resize(N);
	b.resize(N);
	c.resize(N);
	d.resize(N);
	e.resize(N);
	f.resize(N);
	
	// initialize with random data
	for(unsigned int j=0;j<N;j++){
	  a[j] = rand() / ((float)RAND_MAX);
	  b[j] = rand() / ((float)RAND_MAX);
	  c[j] = rand() / ((float)RAND_MAX);
	  
	  d[j] = a[j];
	  e[j] = b[j];
	  f[j] = c[j];
	}
	
	
	// randomly selects on of the possible operations
	{	  
	  
	  unsigned int optype = rand() % 3;
	  
	  if(optype == 0){      // '*' operator
	    str = "'*'-operator";
	    
	    c = a * b;
	    f = d * e;
	  }
	  else if(optype == 1){ // '^' operator (if possible)
	    
	    if(N == 3){
	      str = "'^'-operator";
	      
	      c = a ^ b;
	      f = d ^ e;
	    }
	    else{
	      str = "'*'-operator";
	      
	      c = a * b;
	      f = d * e;
	    }
	  }
	  else if(optype == 2){ // '*=' operator
	    str = "'*='-operator";
	    
	    a *= b;
	    d *= e;
	  }
	  
	  
	  // compares a <-> d, b <-> e, c <-> f
	  
	  {
	    bool ok = true;
	    
	    float f1, f2;
	    
	    for(unsigned int j=0;j<a.size();j++){
	      f1 = a[j]; f2 = d[j].c[0];
	      if(whiteice::math::abs(f1 - f2) > 0.001)
		ok = false;
	    }
	    
	    for(unsigned int j=0;j<b.size();j++){
	      f1 = b[j]; f2 = e[j].c[0];
	      
	      if(whiteice::math::abs(f1 - f2) > 0.001)
		ok = false;
	    }  
	  
	    for(unsigned int j=0;j<c.size();j++){
	      f1 = c[j]; f2 = f[j].c[0];
	      
	      if(whiteice::math::abs(f1 - f2) > 0.001)
		ok = false;
	    }
	    
	    if(ok == false){
	      std::cout << str << " FAILED\n";
	      std::cout << "(may be caused by differences in simd and ieee floating point arithmetic)\n";
	      std::cout << "a = " << a << std::endl;
	      std::cout << "b = " << b << std::endl;
	      std::cout << "c = " << c << std::endl;
	      std::cout << "d = " << d << std::endl;
	      std::cout << "e = " << e << std::endl;
	      std::cout << "f = " << f << std::endl;	      
	      throw test_exception("inner/cross product tests: comparision error");
	    }
	  }
	  
	  
	}
	
      }    
    }
    
  }
  catch(test_exception& e){
    std::cout << "TESTCASE FAILED: " << e.what() << std::endl;
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception: " << e.what() << std::endl;
  }
  


  
  try{
    std::cout << "ATLAS GMATRIX TESTS" << std::endl;
    
    // only tests floats and real number implementation
    
    // gmatrix generation and "=",value(), comparision, resize() tests
    {
      // float tests
      
      whiteice::math::gmatrix<float>* fmat;
      whiteice::math::matrix< blas_real<float> >* amat;
      
      // ctor, [] operator, some comparision and resize tests
      for(unsigned int i=0;i<100;i++){
	const unsigned int N1 = (rand() % 20) + 1;
	const unsigned int N2 = (rand() % 20) + 1;
	
	fmat = new gmatrix<float>(N1, N2);
	amat = new matrix< blas_real<float> >(N1, N2);
	
	if(fmat->xsize() != amat->xsize() ||
	   fmat->ysize() != amat->ysize() )	  
	  throw test_exception("gmatrix size mismatch");
	
	
	for(unsigned int j1=0;j1<N1;j1++){
	  for(unsigned int j2=0;j2<N2;j2++){
	    float f = (rand() / ((float)RAND_MAX));
	    
	    (*fmat)(j1,j2) = f;
	    (*amat)(j1,j2) = f;
	  }
	}
	
	// compares individual values
	for(unsigned int j1=0;j1<N1;j1++){
	  for(unsigned int j2=0;j2<N2;j2++){
	    if( (*fmat)(j1,j2) != (*amat)(j1,j2) ){
	      throw test_exception("gmatrix values mismatch");
	    }
	  }
	}
	
	// alternative ctor
	whiteice::math::gmatrix<float> g2(*fmat);
	whiteice::math::matrix< blas_real<float> > a2(*amat);
	
	if(g2 != *fmat)
	  throw test_exception("generic gmatrix: comparision error when used copy ctor");
	
	if(a2 != *amat)
	  throw test_exception("atlas   gmatrix: comparision error when used copy ctor");
	
	if(g2.xsize() != a2.xsize() || g2.ysize() != a2.ysize())
	  throw test_exception("gmatrix size mismatch (2)");	
	
	// compares individual values
	for(unsigned int j1=0;j1<N1;j1++){
	  for(unsigned int j2=0;j2<N2;j2++){
	    if( g2(j1,j2) != a2(j1,j2) )
	      throw test_exception("gmatrix values mismatch (2)");
	  }
	}
	
	// keeps smallening the size of the matrices to zero dimensions and
	// runs all the comparision tests after each resize()
	
	// chooses randomly either y resize or x resize
	
	if((rand() & 1) == 1){
	  
	  for(int s = N2 - 1; s >= 0; s--){ // doesn't test: resize_x(0)
	    fmat->resize_x(s);
	    amat->resize_x(s);
	    g2.resize_x(s);
	    a2.resize_x(s);
	    
	    if(fmat->ysize() != a2.ysize() || fmat->xsize() != a2.xsize())
	      throw test_exception("gmatrix size mismatch (3)");
	    
	    if(amat->ysize() != g2.ysize() || fmat->xsize() != g2.xsize())
	      throw test_exception("gmatrix size mismatch (4)");
	    
	    // compares values
	    //   within implementations
	    for(unsigned int j1=0;j1<a2.ysize();j1++){
	      for(unsigned int j2=0;j2<a2.xsize();j2++){
		if((*amat)(j1,j2) != a2(j1,j2))
		  throw test_exception("atlas gmatrix values mismatch after resize()");
	      }
	    }
	    
	    for(unsigned int j1=0;j1<g2.ysize();j1++){
	      for(unsigned int j2=0;j2<g2.xsize();j2++){
		if((*fmat)(j1,j2) != g2(j1, j2))
		  throw test_exception("generic gmatrix values mismatch after resize()");
	      }
	    }
	    
	    //   between implementations
	    for(unsigned int j1=0;j1<a2.ysize();j1++){
	      for(unsigned int j2=0;j2<a2.xsize();j2++){
		if(g2(j1,j2) != a2(j1,j2))
		  throw test_exception("in-between gmatrix impl. comparision: values mismatch after resize()");
	      }
	    }
	    
	    // tests "==" operator
	    
	    if( ( (*amat) != a2 ) == true )
	      throw test_exception("atlas gmatrix != operator error after resize()");
	    
	    if( ( (*fmat) != g2 ) == true )
	      throw test_exception("generic gmatrix != operator error after resize()");
	  }
	  
	  delete fmat;
	  delete amat;
	  
	}
	else{
	  
	  for(int s = N1 - 1; s >= 0; s--){ // doesn't test: resize_y(0)
	    fmat->resize_y(s);
	    amat->resize_y(s);
	    g2.resize_y(s);
	    a2.resize_y(s);
	    
	    if(fmat->ysize() != a2.ysize() || fmat->xsize() != a2.xsize())
	      throw test_exception("gmatrix size mismatch (3)");
	    
	    if(amat->ysize() != g2.ysize() || fmat->xsize() != g2.xsize())
	      throw test_exception("gmatrix size mismatch (4)");
	    
	    // compares values
	    //   within implementations
	    for(unsigned int j1=0;j1<a2.ysize();j1++){
	      for(unsigned int j2=0;j2<a2.xsize();j2++){
		if((*amat)(j1,j2) != a2(j1,j2))
		  throw test_exception("atlas gmatrix values mismatch after resize()");
	      }
	    }
	    
	    for(unsigned int j1=0;j1<g2.ysize();j1++){
	      for(unsigned int j2=0;j2<g2.xsize();j2++){
		if((*fmat)(j1,j2) != g2(j1, j2))
		  throw test_exception("generic gmatrix values mismatch after resize()");
	      }	
	    }
	    
	    //   between implementations
	    for(unsigned int j1=0;j1<a2.ysize();j1++){
	      for(unsigned int j2=0;j2<a2.xsize();j2++){
		if(g2(j1,j2) != a2(j1,j2))
		  throw test_exception("in-between gmatrix impl. comparision: values mismatch after resize()");
	      }
	    }
	    
	    // tests "==" operator
	    
	    if( ( (*amat) != a2 ) == true )
	      throw test_exception("atlas gmatrix != operator error after resize()");
	    
	    if( ( (*fmat) != g2 ) == true )
	      throw test_exception("generic gmatrix != operator error after resize()");
	  }
	  
	  delete fmat;
	  delete amat;
	  
	}
	
      }
      
    }
    
    
    
    
    std::cout << "GMATRIX ADD/SUB TESTS\n";
	
    // gmatrix addition and substraction tests
    {
      const unsigned int N1 = (rand() % 20) + 1;
      const unsigned int N2 = (rand() % 20) + 1;
      
      whiteice::math::gmatrix<float> a(N1,N2), b(N1,N2), c(N1,N2);
      whiteice::math::matrix< blas_real<float> > d(N1,N2), e(N1,N2), f(N1,N2);
      std::string str;
      
      for(unsigned int i=0;i<100;i++){ // number of iterations
      
	// initialize with random data
	for(unsigned int j1=0;j1<N1;j1++){
	  for(unsigned int j2=0;j2<N2;j2++){
	    
	    a(j1,j2) = rand() / ((float)RAND_MAX);
	    b(j1,j2) = rand() / ((float)RAND_MAX);
	    c(j1,j2) = rand() / ((float)RAND_MAX);
	    
	    d(j1,j2) = a(j1,j2);
	    e(j1,j2) = b(j1,j2);
	    f(j1,j2) = c(j1,j2);
	  }
	}
	
	
	// randomly selects on of the possible operations
	{
	  unsigned int optype = rand() % 4;
	  
	  if(optype == 0){      // '+' operator
	    str = "'+'-operator";
	    
	    c = a + b;
	    f = d + e;
	  }
	  else if(optype == 1){ // '-' operator
	    str = "'-'-operator";
	    
	    c = a - b;
	    f = d - e;
	  }
	  else if(optype == 2){ // '+=' operator
	    str = "'+='-operator";
	    
	    a += b;
	    d += e;
	  }
	  else if(optype == 3){ // '-=' operator
	    str = "'-='-operator";
	    
	    a -= b;
	    d -= e;
	  }
	  
	  // compares a <-> d, b <-> e, c <-> f
	  
	  for(unsigned int j1=0;j1<N1;j1++){
	    for(unsigned int j2=0;j2<N2;j2++){
	      if(a(j1,j2) != d(j1,j2) || 
		 b(j1,j2) != e(j1,j2) ||
		 c(j1,j2) != f(j1,j2))
	      {
		std::cout << str << " FAILED\n";
		std::cout << "a = " << a << std::endl;
		std::cout << "b = " << b << std::endl;
		std::cout << "c = " << c << std::endl;
		std::cout << "d = " << d << std::endl;
		std::cout << "e = " << e << std::endl;
		std::cout << "f = " << f << std::endl;
		throw test_exception("gmatrix addition/substraction tests: comparision error");
	      }
	    }
	  }
	}
      }
    }
    


    std::cout << "GMATRIX SCALAR MULTIPLICATION  TESTS\n";
    
    // gmatrix scalar multiplication tests
    {
      const unsigned int N = (rand() % 20) + 1;
      
      whiteice::math::gmatrix<float> a(N,N), b(N,N);
      float c;
      
      whiteice::math::matrix< blas_real<float> > d(N,N), e(N,N);
      blas_real<float> f;
      
      std::string str;
      
      for(unsigned int i=0;i<1000;i++){ // number of iterations
      
	// initialize with random data
	for(unsigned int j1=0;j1<N;j1++){
	  for(unsigned int j2=0;j2<N;j2++){
	    a(j1,j2) = rand() / ((float)RAND_MAX);
	    b(j1,j2) = rand() / ((float)RAND_MAX);
	    c = rand() / ((float)RAND_MAX);
	    
	    d(j1,j2) = a(j1,j2);
	    e(j1,j2) = b(j1,j2);
	    f = c;
	  }
	}
	
	
	// randomly selects on of the possible operations
	{
	  unsigned int optype = rand() % 5;
	  
	  if(optype == 0){      // 's * v' operator
	    str = "'s*v'-operator";
	    
	    b = c * a;
	    e = f * d;
	  }
	  else if(optype == 1){ // 'v * s' operator
	    str = "'v*s'-operator";
	    
	    b = a * c;
	    e = d * f;
	  }
	  else if(optype == 2){ // 'v *= s' operator
	    str = "'v*=s'-operator";
	    
	    a *= c;
	    d *= f;
	  }
	  else if(optype == 3){ // 'v / s' operator
	    str = "'v/s'-operator";
	    
	    b = a / c;
	    e = d / f;
	  }
	  else if(optype == 4){ // 'v /= s' operator
	    str = "'v/=s'-operator";
	    
	    a /= c;
	    d /= f;
	  }
	  
	  // compares a <-> d, b <-> e
	  
	  for(unsigned int j1=0;j1<N;j1++){
	    for(unsigned int j2=0;j2<N;j2++){
	      float _f1 = a(j1,j2);
	      float _f2 = d(j1,j2).c[0];
	      float _e1 = b(j1,j2);
	      float _e2 = e(j1,j2).c[0];
	      
	      if(whiteice::math::abs(_f1 - _f2) > 0.001 ||
		 whiteice::math::abs(_e1 - _e2) > 0.001){
		
		std::cout << str << " FAILED\n";
		std::cout << "index = " << j1 << ", " << j2 << std::endl;
		std::cout << "a = " << a << std::endl;
		std::cout << "b = " << b << std::endl;
		std::cout << "c = " << c << std::endl;
		std::cout << "d = " << d << std::endl;
		std::cout << "e = " << e << std::endl;
		std::cout << "f = " << f << std::endl;
		throw test_exception("matrix multiplication/divide tests: comparision error");
	      }
	    }
	  }
	}
      }
    }
    
    
    std::cout << "GMATRIX MUL, INV TESTS\n";
    
    // gvertex inner/cross product tests
    {
      const unsigned int N1 = (rand() % 10) + 1;
      const unsigned int N2 = N1;
      
      whiteice::math::gmatrix<float> a(N1,N2), b(N1,N2), c(N1,N2);
      whiteice::math::matrix< blas_real<float> > d(N1,N2), e(N1,N2), f(N1,N2);
      std::string str;
      
      for(unsigned int i=0;i<100;i++){ // number of iterations
	
	a.resize(N1,N2);
	b.resize(N1,N2);
	c.resize(N1,N2);
	d.resize(N1,N2);
	e.resize(N1,N2);
	f.resize(N1,N2);
	
	
	// initialize with random data
	for(unsigned int j1=0;j1<N1;j1++){
	  for(unsigned int j2=0;j2<N2;j2++){
	    
	    a(j1,j2) = rand() / ((float)RAND_MAX);
	    b(j1,j2) = rand() / ((float)RAND_MAX);
	    c(j1,j2) = rand() / ((float)RAND_MAX);
	    
	    d(j1,j2) = a(j1,j2);
	    e(j1,j2) = b(j1,j2);
	    f(j1,j2) = c(j1,j2);
	  }
	}
	
	
	// randomly selects on of the possible operations
	{	  
	  
	  unsigned int optype = rand() % 3;
	  
	  if(optype == 0){      // '*' operator
	    str = "'*'-operator";
	    
	    c = a * b;
	    f = d * e;
	  }
	  else if(optype == 1){ // 'inv' operator (if possible)
	    
	    str = "'calculates inverse'";
	    
	    c = a;
	    c.inv();
	    
	    f = d;
	    f.inv();
	  }
	  else if(optype == 2){ // '*=' operator
	    str = "'*='-operator";
	    
	    a *= b;
	    d *= e;
	  }
	  
	  
	  // compares a <-> d, b <-> e, c <-> f
	  
	  {
	    bool ok = true;
	    
	    for(unsigned int j1=0;j1<N1;j1++){
	      for(unsigned int j2=0;j2<N2;j2++){
		
		float E[2];
		
		E[0] = a(j1,j2); E[1] = d(j1,j2).c[0];
		float f1 = whiteice::math::abs(E[0] - E[1]);
		
		E[0] = b(j1,j2); E[1] = e(j1,j2).c[0];
		float f2 = whiteice::math::abs(E[0] - E[1]);
		
		E[0] = c(j1,j2); E[1] = f(j1,j2).c[0];
		float f3 = whiteice::math::abs(E[0] - E[1]);
		
		if(f1 >= 0.001 || f2 >= 0.001 || f3 >= 0.001){
		  ok = false;
		  std::cout << "gmatrix mismatch " 
			    << j1 << " , " << j2 << std::endl;
		}
		
	      }
	    }
	    
	    
	    if(ok == false){
	      std::cout << str << " FAILED\n";
	      std::cout << "(may be caused by differences in simd and ieee floating point arithmetic)\n";
	      std::cout << "a = " << a << std::endl;
	      std::cout << "b = " << b << std::endl;
	      std::cout << "c = " << c << std::endl;
	      std::cout << "d = " << d << std::endl;
	      std::cout << "e = " << e << std::endl;
	      std::cout << "f = " << f << std::endl;	      
	      throw test_exception("inner/cross product tests: comparision error");
	    }
	  }
	}
      }    
    }
    
    
    std::cout << "GMATRIX & GVERTEX TESTS" << std::endl;
    
    // GMATRIX & GVERTEX TESTS    
    // M*x, x*M, x * y' (outer product)
    {
      // tests with blas_real<float> againt
      // generic gmatrix/gvertex code
      
      const unsigned int N1 = ((rand() % 20) + 1);
      const unsigned int N2 = ((rand() % 20) + 1);
      
      whiteice::math::vertex< blas_real<float> > v[2];
      whiteice::math::matrix< blas_real<float> > M;
      
      whiteice::math::gvertex<float> u[2];
      whiteice::math::gmatrix<float> N;
      
      u[0].resize(N1); u[1].resize(N1);
      v[0].resize(N1); v[1].resize(N1);
      
      M.resize(N2,N1);
      N.resize(N2,N1);
      
      for(unsigned int i=0;i<1000;i++){
	// initializes data randomly
	
	u[0].resize(N1); u[1].resize(N1);
	v[0].resize(N1); v[1].resize(N1);
	
	M.resize(N2,N1);
	N.resize(N2,N1);
	
	// gmatrix
	for(unsigned int j1=0;j1<N2;j1++){
	  for(unsigned int j2=0;j2<N1;j2++){
	    
	    M(j1,j2) = rand() / ((float)RAND_MAX);
	    N(j1,j2) = M(j1,j2).c[0];
	  }
	}
	
	for(unsigned int j1=0;j1<N1;j1++){
	  u[0][j1] = rand() / ((float)RAND_MAX);
	  u[1][j1] = rand() / ((float)RAND_MAX);
	  
	  v[0][j1] = u[0][j1];
	  v[1][j1] = u[1][j1];
	}
      }
      
      
      std::string str;
      
      // selects operation randomly
      {	
	unsigned int optype = rand() % 3;
	
	if(optype == 0){ // M*x
	  v[1] = M*v[0];
	  u[1] = N*u[0];
	  
	  str = "M*x operation";
	}
	else if(optype == 1){ // x*M
	  
	  // resizes vectors
	  u[0].resize(N2);
	  v[0].resize(N2);
	  
	  for(unsigned int k=0;k<N2;k++){
	    u[0][k] = rand() / ((float)RAND_MAX);
	    v[0][k] = u[0][k];
	  }
	  
	  
	  v[1] = v[0]*M;
	  u[1] = u[0]*N;
	  
	  str = "x*M operation";
	}
	else if(optype == 2){ // x*y'
	  
	  M = v[0].outerproduct(v[1]);
	  N = u[0].outerproduct(u[1]);
	  
	  str = "x*y' operation";
	}
      }
      
      
      // checks for equality
      {
	bool ok = true;
	
	// matrices
	for(unsigned int j1=0;j1<M.ysize();j1++){
	  for(unsigned int j2=0;j2<M.xsize();j2++){
	    float mf = M(j1,j2).c[0];
	    float nf = N(j1,j2);
	    
	    if(whiteice::math::abs(mf - nf) > 0.001){
	      ok = false;
	    }
	    
	  }
	}
	
	
	if(ok == false){
	  std::cout << "ERROR: " << str << " FAILED\n";
	  std::cout << "(gmatrix mismatch)" << std::endl;
	  std::cout << "M = " << M << std::endl;
	  std::cout << "N = " << N << std::endl;
	  
	  test_exception("Operation result mismatch");
	}
	
	
	// verteces
	for(unsigned int k=0;k<2;k++){
	  for(unsigned int j1=0;j1<v[k].size();j1++){
	    
	    float mf = v[k][j1].c[0];
	    float nf = u[k][j1];
	    
	    
	    if(whiteice::math::abs(mf - nf) > 0.001){
	      ok = false;
	    }
	  }
	}
	
	
	if(ok == false){
	  std::cout << "ERROR: " << str << " FAILED\n";
	  std::cout << "(gvertex mismatch)" << std::endl;
	  
	  std::cout << "u[0] = " << u[0] << std::endl;
	  std::cout << "u[1] = " << u[1] << std::endl;
	  std::cout << "v[0] = " << v[0] << std::endl;
	  std::cout << "v[1] = " << v[1] << std::endl;
	  
	  test_exception("Operation result mismatch");
	}
	
      }
    }
    
    
    
  }
  catch(test_exception& e){
    std::cout << "TESTCASE FAILED: " << e.what() << std::endl;
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception: " << e.what() << std::endl;
  }
  
  
  try{
    std::cout << "ATLAS QUATERNION TESTS" << std::endl;
    
    std::cout << "ERROR: NOT IMPLEMENTED!" << std::endl;
  }
  catch(test_exception& e){
    std::cout << "TESTCASE FAILED: " << e.what() << std::endl;
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception: " << e.what() << std::endl;
  }
}


////////////////////////////////////////////////////////////////////////////////


void blas_compile_tests()
{  
  {
    std::cout << "ATLAS primitives test\n";
    
    if(sizeof(blas_real<float>) != sizeof(float))
      std::cout << "ERROR: wrong atlas primitive size, real float"
		<< std::endl;
	
    if(sizeof(blas_complex<float>) != 2*sizeof(float))
      std::cout << "ERROR: wrong atlas primitive size, complex float"
		<< std::endl;

    if(sizeof(blas_real<double>) != sizeof(double))
      std::cout << "ERROR: wrong atlas primitive size, real double"
		<< std::endl;
	
    if(sizeof(blas_complex<double>) != 2*sizeof(double))
      std::cout << "ERROR: wrong atlas primitive size, complex double"
		<< std::endl;
       
    
    // some calculation tests
    
    blas_complex<float> c;
    c = 1;
    blas_real<float>* r = new blas_real<float>[1024];
    
    r[10] = 10.0f;
    r[10] += 15.0f;
    std::cout << "10 + 15 = " << r[10] << std::endl;
    std::cout << "random element " << r[11] << std::endl;
    std::cout << std::endl;
    
    delete[] r;
  }
  
  // ATLAS gvertex test
  {
    std::cout << "ATLAS gvertex test\n";
    
    vertex< blas_real<float> > a, b, c, d(10);
    vertex<>* xx;
    xx = new vertex<>[10];
    
    for(unsigned int i=0;i<10;i++)
      xx[i].resize(100);
    
    delete[] xx; xx = 0;
    
    a.resize(10);
    b.resize(10);
    c.resize(10);
    
    std::cout << "A = " << a << std::endl;
    std::cout << "|A| = " << a.norm() << std::endl;
    
    for(unsigned int i=0;i<a.size();i++){
      float f = -((signed)i)*2.0f;
      a[i] = i;  
      b[i] = f;
    }
    
    std::cout << "A = " << a << std::endl;
    std::cout << "B = " << b << std::endl;
    std::cout << "|A| = " << a.norm() << std::endl;
    
    std::cout << "C = " << c << std::endl;
    c = blas_real<float>(2.0f) * a + b;
    std::cout << "C = " << c << std::endl;
    std::cout << "|C| = " << c.norm() << std::endl;
    
    a.normalize();
    b.normalize();
    c.normalize();    
    std::cout << "|A| = " << a.norm() << std::endl;
    std::cout << "|B| = " << b.norm() << std::endl;
    std::cout << "|C| = " << c.norm() << std::endl;
    
    std::cout << "A = " << a << std::endl;
    std::cout << "B = " << b << std::endl;
    std::cout << "C = " << c << std::endl;
    
    a *= 2;
    std::cout << "|A| = " << a.norm() << std::endl;
    std::cout << "A = " << a << std::endl;
    
    a /= 2;
    std::cout << "|A| = " << a.norm() << std::endl;
    std::cout << "A = " << a << std::endl;
    
    d = b;
    
    b += a; // -> zero
    std::cout << "|B| = " << b.norm() << std::endl;
    std::cout << "B = " << b << std::endl;
    
    d = d.abs();
    std::cout << "|D| = " << d.norm() << std::endl;
    std::cout << "D = " << d << std::endl;    
    
    c = a;
    std::cout << "|C| = " << c.norm() << std::endl;
    std::cout << "C = " << c << std::endl;
    
    b = a * c; // dot product (c = a -> b = |a|^2 = 1)
    std::cout << "|B| = " << b.norm() << std::endl;
    std::cout << "B = " << b << std::endl;
    
    a *= c;
    std::cout << "|A| = " << a.norm() << std::endl;
    std::cout << "A = " << a << std::endl;    
    
  }
  
  
  // ATLAS gmatrix test
  {
    std::cout << "ATLAS gmatrix test\n";    
    matrix< blas_real<float> > A, B, C;
    
    A(1,0) = 69;
    std::cout << "A = " << A << std::endl;
    B.identity();
    std::cout << "B = " << B << std::endl;    
    B = blas_real<float>(2.0f) *  B;
    std::cout << "B = " << B << std::endl;
    C = B*A;
    std::cout << "C = " << C << std::endl;
    std::cout << "trace(C) = " << C.trace() << std::endl;
    std::cout << "det(C) = " << C.det() << std::endl;
    std::cout << "det(B) = " << B.det() << std::endl;
    
    // TODO: test inverse (and optimize it)
  }
  
  // ATLAS gmatrix * ATLAS gvertex test
  {
    std::cout << "ATLAS gmatrix&gvertex test\n";
    matrix< blas_real<float> > A(4,4),B,C;
    vertex< blas_real<float> > a(4),b,c;
    
    A.identity();
    a[0] = -31415729;
    a[1] = 12;
    a[2] = 2;
    a[3] = 3;
    
    A(1,2) = -2.1;

    std::cout << "A = " << A << std::endl;
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    
    b = A*a;
    
    std::cout << "b = " << b << std::endl;
    
    c = a*A; // a is interpreted as a horizontal vector
    
    std::cout << "c = " << c << std::endl;
  }
}



//////////////////////////////////////////////////////////////////////

void correlation_test()
{
  
  // test autocorrelation with a list of vectors
  try{
    std::vector< vertex< blas_real<float> > > vectors;
    matrix< blas_real<float> > R, pR;
    vertex< blas_real<float> > v;
    
    // TEST 1: correctness test

    // 10 vectors
    v.resize(5);
    v[0] = -1.0; v[1] = 5.1; v[2] = 8.1; v[3] = -4.2; v[4] = -11.8;
    gpu_sync();
    vectors.push_back(v);

    v.resize(5);
    v[0] = 1.21; v[1] = -4.21; v[2] = -8.61; v[3] = 14.1; v[4] = 91.1;
    gpu_sync();
    vectors.push_back(v);
    
    v.resize(5);
    v[0] = 16.1; v[1] = -92.1; v[2] = 76.1; v[3] = -41.4; v[4] = -47.1;
    gpu_sync();
    vectors.push_back(v);
    
    v.resize(5);
    v[0] = 92.6; v[1] = -72.1; v[2] = -81.1; v[3] = 5.12; v[4] = 62.2;
    gpu_sync();
    vectors.push_back(v);
    
    v.resize(5);
    v[0] = -16.1; v[1] = 12.1; v[2] = -41.1; v[3] = -71.1; v[4] = 31.1;
    gpu_sync();
    vectors.push_back(v);
    
    v.resize(5);
    v[0] = -84.1; v[1] = 41.1; v[2] = 25.1; v[3] = 41.1; v[4] = -81.1;
    gpu_sync();
    vectors.push_back(v);
    
    v.resize(5);
    v[0] = -14.1; v[1] = 5.1; v[2] = 8.42; v[3] = -6.31; v[4] = 11.9;
    gpu_sync();
    vectors.push_back(v);
    
    v.resize(5);
    v[0] = 42.1; v[1] = -56.7; v[2] = 61.2; v[3] = -41.1; v[4] = -7.21;
    gpu_sync();
    vectors.push_back(v);
    
    v.resize(5);
    v[0] = -43.1; v[1] = 12.1; v[2] = -42.1; v[3] = 86.1; v[4] = -41.1;
    gpu_sync();
    vectors.push_back(v);
    
    v.resize(5);
    v[0] = 18.2; v[1] = -19.5; v[2] = 22.1; v[3] = -21.1; v[4] = 53.1;
    gpu_sync();
    vectors.push_back(v);
    
    // correct correlation matrix for above data
    pR.resize(5,5);
    
    pR(0,0) =  2032.85; pR(0,1) = -1515.62; pR(0,2) =  -307.78; pR(0,3) =  -821.92; pR(0,4) =  1370.97;
    pR(1,0) = -1515.62; pR(1,1) =  1932.77; pR(1,2) =  -491.71; pR(1,3) =   794.34; pR(1,4) =  -461.06;
    pR(2,0) =  -307.78; pR(2,1) =  -491.71; pR(2,2) =  2090.45; pR(2,3) =  -642.69; pR(2,4) = -1025.97;
    pR(3,0) =  -821.92; pR(3,1) =   794.34; pR(3,2) =  -642.69; pR(3,3) =  1828.85; pR(3,4) =  -637.98;
    pR(4,0) =  1370.97; pR(4,1) =  -461.06; pR(4,2) = -1025.97; pR(4,3) =  -637.98; pR(4,4) =  2677.25;

    gpu_sync();
    
    if(autocorrelation<>(R, vectors) == false){
      std::cout << "ERROR: autocorrelation calculation failed (test1)"
		<< std::endl;
      return;
    }
    
    pR -= R;
    
    if(norm_inf(pR) > 0.01){
      std::cout << "ERROR: incorrect autocorrelation matrix (test1)"
		<< std::endl;
      std::cout << "calculated R: " << R << std::endl;
      std::cout << "error: " << pR << std::endl;
      return;
    }
    else{
      std::cout << "Autocorrelation calculation OK. (test1" << std::endl;
    }
    
    
    // TEST2: properties test (symmetry and right size)
    
    for(unsigned int i=0;i<100;i++){
      vectors.clear();
      v.resize( (rand() % 10) + 1);
      
      for(unsigned int j=0;j<50;j++){
	
	for(unsigned int k=0;k<v.size();k++)
	  v[k] = (((float)rand())/((float)RAND_MAX) - 0.5f);
	
	vectors.push_back(v);
      }
      
      if(autocorrelation(R, vectors) == false){
	std::cout << "ERROR: autocorrelation matrix calculation failed (test2)"
		  << std::endl;
	std::cout << "Iteration " << i << std::endl;
	return;
      }
      
      // correct size test
      if(R.xsize() != v.size() || R.ysize() != v.size()){
	std::cout << "ERROR: wrong size autocorrelation matrix (test2)"
		  << std::endl;
	std::cout << "Iteration " << i << std::endl;
	return;
      }
      
      // symmetry test
      for(unsigned int y=1;y<v.size();y++){
	for(unsigned int x=0;x<y;x++){
	  if(whiteice::math::abs(R(x,y) - whiteice::math::conj(R(y,x))) > 0.01){
	    std::cout << "ERROR: non symmetric autocorrelation matrix (test2)"
		      << std::endl;
	    std::cout << "Iteration " << i << std::endl;
	    return;
	  }
	}
      }
    }
    
    
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
    return;
  }
  
  
  
  // tests autocorrelation with matrix'es row vectors
  try{
    matrix< blas_real<float> > V;
    matrix< blas_real<float> > R, pR;
    
    // TEST 1: correctness test
    V.resize(10, 5);

    // 10 vectors 
    V(0,0) = -1.0;  V(0,1) = 5.1;   V(0,2) = 8.1;   V(0,3) = -4.2;  V(0,4) = -11.8;
    V(1,0) = 1.21;  V(1,1) = -4.21; V(1,2) = -8.61; V(1,3) = 14.1;  V(1,4) = 91.1;
    V(2,0) = 16.1;  V(2,1) = -92.1; V(2,2) = 76.1;  V(2,3) = -41.4; V(2,4) = -47.1;
    V(3,0) = 92.6;  V(3,1) = -72.1; V(3,2) = -81.1; V(3,3) = 5.12;  V(3,4) = 62.2;
    V(4,0) = -16.1; V(4,1) = 12.1;  V(4,2) = -41.1; V(4,3) = -71.1; V(4,4) = 31.1;
    V(5,0) = -84.1; V(5,1) = 41.1;  V(5,2) = 25.1;  V(5,3) = 41.1;  V(5,4) = -81.1;
    V(6,0) = -14.1; V(6,1) = 5.1;   V(6,2) = 8.42;  V(6,3) = -6.31; V(6,4) = 11.9;
    V(7,0) = 42.1;  V(7,1) = -56.7; V(7,2) = 61.2;  V(7,3) = -41.1; V(7,4) = -7.21;
    V(8,0) = -43.1; V(8,1) = 12.1;  V(8,2) = -42.1; V(8,3) = 86.1;  V(8,4) = -41.1;
    V(9,0) = 18.2;  V(9,1) = -19.5; V(9,2) = 22.1;  V(9,3) = -21.1; V(9,4) = 53.1;

    gpu_sync();
    
    // correct correlation matrix for above data
    pR.resize(5,5);
    
    pR(0,0) =  2032.85; pR(0,1) = -1515.62; pR(0,2) =  -307.78; pR(0,3) =  -821.92; pR(0,4) =  1370.97;
    pR(1,0) = -1515.62; pR(1,1) =  1932.77; pR(1,2) =  -491.71; pR(1,3) =   794.34; pR(1,4) =  -461.06;
    pR(2,0) =  -307.78; pR(2,1) =  -491.71; pR(2,2) =  2090.45; pR(2,3) =  -642.69; pR(2,4) = -1025.97;
    pR(3,0) =  -821.92; pR(3,1) =   794.34; pR(3,2) =  -642.69; pR(3,3) =  1828.85; pR(3,4) =  -637.98;
    pR(4,0) =  1370.97; pR(4,1) =  -461.06; pR(4,2) = -1025.97; pR(4,3) =  -637.98; pR(4,4) =  2677.25;

    gpu_sync();
    
    if(autocorrelation<>(R, V) == false){
      std::cout << "ERROR: autocorrelation calculation failed (test3)"
		<< std::endl;
      return;
    }
    
    auto delta = pR - R;
    
    if(norm_inf(delta) > 0.01){
      std::cout << "ERROR: incorrect autocorrelation matrix (test3)"
		<< std::endl;
      std::cout << "calculated R: " << R << std::endl;
      std::cout << "correct    R: " << pR << std::endl;
      std::cout << "error: " << delta << std::endl;
      return;
    }
    
    
    
    // TEST2: properties test (symmetry and right size)
    
    for(unsigned int i=0;i<100;i++){
      V.resize(50, (rand() % 10) + 1);
      
      for(unsigned int j=0;j<50;j++)
	for(unsigned int k=0;k<V.xsize();k++)
	  V(j,k) = (((float)rand())/((float)RAND_MAX) - 0.5f);

      if(autocorrelation(R, V) == false){
	std::cout << "ERROR: autocorrelation matrix calculation failed (test4)"
		  << std::endl;
	std::cout << "Iteration " << i << std::endl;
	return;
      }
      
      // correct size test
      if(R.xsize() != V.xsize() || R.ysize() != V.xsize()){
	std::cout << "ERROR: wrong size autocorrelation matrix (test4)"
		  << std::endl;
	std::cout << "Iteration " << i << std::endl;
	return;
      }
      
      // symmetry test
      for(unsigned int y=1;y<V.xsize();y++){
	for(unsigned int x=0;x<y;x++){
	  if(whiteice::math::abs(R(x,y) - whiteice::math::conj(R(y,x))) > 0.01){
	    std::cout << "ERROR: non symmetric autocorrelation matrix (test4)"
		      << std::endl;
	    std::cout << "Iteration " << i << std::endl;
	    return;
	  }
	}
      }
    }
    
    
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
    return;
  }
  
  
  // comparision test between implementations
  try{  
    matrix< blas_real<float> > V;
    std::vector< vertex< blas_real<float> > > vectors;
    vertex< blas_real<float> > v;
    
    matrix< blas_real<float> > R1, R2 , E;
    
    
    for(unsigned int i=0;i<100;i++){
      
      unsigned int N = (rand() % 100) + 1; // number of vectors
      unsigned int D = (rand() % 10) + 1;  // dimension

      v.resize(D);
      vectors.clear();
      V.resize(N,D);
      
      // creates data
      for(unsigned int n=0;n<N;n++){
	for(unsigned int d=0;d<D;d++){
	  v[d] = ((float)rand()) / ((float)RAND_MAX);
	  V(n,d) = v[d];
	}
	
	vectors.push_back(v);
      }
      
      // calculates autocorrelation matrices
      
      if(autocorrelation(R1, vectors) == false){
	std::cout << "ERROR: autocorrelation matrix calculation failed (test5) 1/2"
		  << std::endl;
	std::cout << "Iteration " << i << std::endl;
	return;
      }
      
      if(autocorrelation(R2, V) == false){
	std::cout << "ERROR: autocorrelation matrix calculation failed (test5) 2/2"
		  << std::endl;
	std::cout << "Iteration " << i << std::endl;
	return;
      }
      
      E = R1;
      E -= R2;
      
      if(norm_inf(E) > 0.01){
	std::cout << "ERROR: autocorrelation calculation result mismatch (test5)"
		  << std::endl;
	std::cout << "R1 = " << R1 << std::endl;
	std::cout << "R2 = " << R2 << std::endl;
	std::cout << "difference = " << E << std::endl;
      }

    }
    
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
    return;
  }
  
  
  std::cout << "AUTOCORRELATION TESTS: PASSED" << std::endl;
}



//////////////////////////////////////////////////////////////////////



void own_unexpected()
{
  std::cout << "TESTCASE FATAL ERROR: unexpected exception: calling terminate()" << std::endl;
  std::cout << std::flush;
  std::terminate();
}


void own_terminate()
{
  std::cout << "testcase terminate() activated." << std::endl;
  std::cout << std::flush;
}






