
#include "HugeLinear.h"
#include <math.h>


namespace whiteice
{
  
  HugeLinear::HugeLinear()
  {
  }
  
  HugeLinear::~HugeLinear()
  {
  }

  bool HugeLinear::startOptimize(DataSourceInterface* data)
  {
    return false;
  }
  
  bool HugeLinear::isRunning()
  {
    return false;
  }
  
  bool HugeLinear::stopOptimize()
  {
    return false;
  }
  
  bool HugeLinear::getSolution(math::matrix< math::blas_real<float> >& A,
			       math::vertex< math::blas_real<float> >& b)
  {
    return false;
  }
  
  float HugeLinear::estimateSolutionMSE()
  {
    return INFINITY;
  }
  
  // removes solution and resets to empty HugeLinear
  void HugeLinear::reset()
  {
    
  }

  
  void HugeLinear::optimizer_loop()
  {
    
  }
  
  
};
