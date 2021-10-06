/*
 * HugeLinear optimizer testcases
 */

#include <stdio.h>
#include <unistd.h>
#include <vector>

#include "HugeLinear.h"

using namespace whiteice;

//////////////////////////////////////////////////////////////////////

bool simple_test();


int main(void)
{
  simple_test();

  return 0;;
}

//////////////////////////////////////////////////////////////////////

class MemoryDataSource : public whiteice::DataSourceInterface
{
public:
  MemoryDataSource(const std::vector< math::vertex< math::blas_real<float> > >& x,
		   const std::vector< math::vertex< math::blas_real<float> > >& y){
    if(x.size() == y.size()){
      this->x = x;
      this->y = y;
    }
  }
  
  ~MemoryDataSource(){
    x.clear();
    y.clear();
  }
  
  virtual const unsigned long getNumber() const // number of data vector pairs (x,y)
  {
    return x.size();
  }
  
  virtual const unsigned long getInputDimension() const  // x input vector dimension
  {
    if(x.size() <= 0) return 0;
    else return x[0].size();
  }
  
  virtual const unsigned long getOutputDimension() const // y output vector dimension
  {
    if(y.size() <= 0) return 0;
    else return y[0].size();
  }
  
  // gets index:th data points or return false (bad index or unknown error)
  virtual const bool getData(const unsigned long index,
			     math::vertex< math::blas_real<float> > & x,
			     math::vertex< math::blas_real<float> >& y) const
  {
    if(index >= this->x.size()) return false;

    x = this->x[index];
    y = this->y[index];
    
    return true;
  }
  
private:
  std::vector< math::vertex< math::blas_real<float> > > x;
  std::vector< math::vertex< math::blas_real<float> > > y;
  
};

//////////////////////////////////////////////////////////////////////

bool simple_test()
{
  printf("HugeLinear first test: tests code runs without erros.\n");

  // generates simple test problem that fits perfectly to a linear problem

  std::vector< math::vertex< math::blas_real<float> > > datax;
  std::vector< math::vertex< math::blas_real<float> > > datay;

  {
    whiteice::RNG< math::blas_real<float> > rng;
    
    math::matrix< math::blas_real<float> > A;
    math::vertex< math::blas_real<float> > b;

    unsigned int inputDim = 1 + (rng.rand() % 100);
    unsigned int outputDim = 1 + (rng.rand() % 10);

    A.resize(outputDim, inputDim);
    b.resize(outputDim);

    rng.normal(A);
    rng.normal(b);

    const unsigned long N = 20000; // number of datapoints

    for(unsigned long i=0;i<N;i++){
      math::vertex< math::blas_real<float> > x, y;
      x.resize(inputDim);
      rng.normal(x);

      y = A*x + b;

      datax.push_back(x);
      datay.push_back(y);
    }
  }

  whiteice::HugeLinear solver;
  MemoryDataSource source(datax, datay);
  
  if(solver.startOptimize(&source) == false){
    printf("Starting solver FAILED.\n");
    return false;
    
  }
  else{
    printf("Succesfully started solver.\n");
  }
  
  unsigned int iteration_seen = 0;

  while(solver.isRunning()){
    sleep(1);

    const unsigned int iter = solver.getIterations();

    if(iter > iteration_seen){
      iteration_seen = iter;

      printf("Solver error after %d iterations. MSE: %f.\n",
	     iter,
	     solver.estimateSolutionMSE());
    }
  }

  solver.stopOptimize();

  printf("Final solver error after %d iterations. MSE: %f.\n",
	 solver.getIterations(),
	 solver.estimateSolutionMSE());

  return true;
}
