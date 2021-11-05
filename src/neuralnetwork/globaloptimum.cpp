
#include "globaloptimum.h"
#include "RNG.h"
#include "NNGradDescent.h"
#include "linear_ETA.h"


namespace whiteice
{
  // learns N global linear optimizers (parameter K) u=vec(A,b) parameters u using randomized
  // neural network weights w and randomly N(0,I) generated training data (M training datapoints)
  // learn globally optimal linear optimum mapping from f(u)-> w and trains original problem using
  // globally optimal solution u and gets preset neural network weights w using learnt mapping
  // assumes input data is N(0,I) distributed + assumes output data is close to N(0,I) too.
  template <typename T>
  bool global_optimizer_pretraining(nnetwork<T>& net,
				    const dataset<T>& data, 
				    const unsigned int N,
				    const unsigned int M,
				    const unsigned int K)
  {
    if(N < 100) return false;
    if(M < 100) return false;
    if(K <= 1) return false;

    // 1. learns N global linear optimizes {u} and {w}
    std::vector< math::vertex<T> > linear_parameters_u;
    std::vector< math::vertex<T> > nnetwork_parameters_w;

    linear_parameters_u.resize(N);
    nnetwork_parameters_w.resize(N);
    
    whiteice::RNG<T> rng;
    
    whiteice::linear_ETA<> eta;
    eta.start(0.0f, (float)N);
    unsigned int counter = 0;

    bool failure = false;

#pragma omp parallel
    {

      // OpenMPfied code
#pragma omp for nowait schedule(auto)
      for(unsigned int n=0;n<N;n++){

	if(failure) continue;
	
	{
	  whiteice::math::blas_real<float> t = eta.estimate();
	  float tf = 0.0f;
	  whiteice::math::convert(tf, t);
	  
	  printf("Training data generation iteration %d/%d (ETA: %f minutes).\n",
		 counter+1, N, tf/60.0f);
	}
	
	dataset<T> train_data;
	nnetwork<T> nnet(net);

	// generates training data
	{
	  train_data.createCluster("input", nnet.input_size());
	  train_data.createCluster("output", nnet.output_size());

	  math::vertex<T> v, w;
	  v.resize(nnet.input_size());
	  w.resize(nnet.output_size());

	  // creates random XOR like non-linearities and select Nth digit from random input nonlins
	  std::vector< std::vector<unsigned int> > pairs;
	  std::vector< std::vector<unsigned int> > digit;

	  const unsigned int problem = rng.rand() % 3;

	  if(problem == 0){ // XOR-like non-linearity
	  
	    for(unsigned int i=0;i<nnet.output_size();i++){
	      std::vector<unsigned int> p;
	      
	      unsigned int pN = 1 + (rng.rand() % 3);
	      
	      for(unsigned int pi=0;pi<pN;pi++){
		unsigned int p1 = rng.rand() % nnet.input_size();
		p.push_back(p1);
	      }
	      
	      pairs.push_back(p);
	    }

	    for(unsigned int m=0;m<M;m++){
	      rng.normal(v);
	      
	      // nnet.calculate(v, w);
	      for(unsigned int i=0;i<w.size();i++){
		T value = T(1.0f);
		
		// is this XOR-like non-linearity
		for(unsigned int pi=0;pi<pairs[i].size();pi++)
		  value *= v[ pairs[i][pi] ];
		
		w[i] = whiteice::math::pow(whiteice::math::abs(value), T(1.0f)/T((float)pairs[i].size()));
		if(value < T(0.0f)) w[i] = -w[i];
	      }
	      
	      train_data.add(0, v);
	      train_data.add(1, w);
	    }

	  }
	  else if(problem == 1){ // D:th digit non-linearity

	    for(unsigned int i=0;i<nnet.output_size();i++){
	      std::vector<unsigned int> p;
	      
	      const unsigned int k = rng.rand() % nnet.input_size(); // random input
	      const unsigned int d = 1 + (rng.rand() % 3); // digit = [1,3]:th decimal position
	      
	      p.push_back(k);
	      p.push_back(d);
	      
	      digit.push_back(p);
	    }

	    for(unsigned int m=0;m<M;m++){
	      rng.normal(v);
	      
	      // nnet.calculate(v, w);
	      for(unsigned int i=0;i<w.size();i++){
		T value = T(0.0f);
		value = v[ digit[i][0] ]*whiteice::math::pow(T(10.0f), T((float)digit[i][1]));
		int k = 0;
		whiteice::math::convert(k, value);
		
		w[i] = (float)(k % 10);
	      }
		    
	      train_data.add(0, v);
	      train_data.add(1, w);
	    }
	  }
	  else{ // problem == 2 // overlapping gaussians (two rings)
	    
	    
	    for(unsigned int m=0;m<M;m++){
	      T r = (float)(rng.rand() & 1);
	      if(r >= 1.0f) r = 20.0f;
	      rng.normal(v);
	      v = r*v; // ring of radius 1 or 20
	      
	      w.zero();
	      
	      //rand_net.calculate(v, w);
	      for(unsigned int i=0;i<w.size() && i<v.size();i++){
		w[i] = r;
	      }

	      train_data.add(0, v);
	      train_data.add(1, w);
	    }
	    
	  }
	  
	  
	  if(train_data.preprocess(0, dataset<T>::dnMeanVarianceNormalization) == false){
	    failure = true;
	    continue;
	  }
	  
	  if(train_data.preprocess(1, dataset<T>::dnMeanVarianceNormalization) == false){
	    failure = true;
	    continue;
	  }
	  
	  {
	    nnet.randomize();
	    
	    // trains neural network
	    whiteice::math::NNGradDescent<T> grad;
	    grad.setUseMinibatch(true);
	    grad.startOptimize(train_data, nnet, 1);

	    while(grad.hasConverged(0.01) == false){
	      sleep(1);
	      T error = T(0.0);
	      unsigned int Nconverged = 0;
	      
	      if(grad.getSolutionStatistics(error, Nconverged) == false){
		failure = true;
		continue;
	      }
#if 0
	      else{
		float errorf = 0.0f;
		whiteice::math::convert(errorf, error);
		printf("Optimizing error %d: %f\n", Nconverged, errorf);
	      }
#endif
	    }

	    grad.stopComputation();

	    {
	      nnetwork<T> nn;
	      T error;
	      unsigned int Nconverged;
	      
	      if(grad.getSolution(nn, error, Nconverged) == false){
		failure = true;
		continue;
	      }

	      nnet = nn;
	      
	      float errorf = 0.0f;
	      whiteice::math::convert(errorf, error);
	      
	      printf("Optimization converged (%d: error = %f). STOP.\n",
		     Nconverged, errorf);
	    }

	  }

	  // now solution has converged
	}

	
	// vectorizes parameters and stores them to vector lists
	{
	  math::matrix<T> A;
	  math::vertex<T> b;

	  // solves linear optimization problem
	  if(global_linear_optimizer(train_data, K, A, b) == false){
	    printf("ERROR: global_linear_optimizer() call failed.\n");
	    failure = true;
	    continue;
	  }

	  math::vertex<T> u1;
	  if(A.vec(u1) == false){ // vectorizes matrix (row1, row2, row3, row(COL:th))
	    failure = true;
	    continue;
	  }
	  
	  // math::vertex<T> u2 = b;
	  
	  math::vertex<T> u;
	  u.resize(u1.size() + b.size());
	  if(u.write_subvertex(u1, 0) == false){
	    failure = true;
	    continue;
	  }
	  
	  if(u.write_subvertex(b, u1.size()) == false){
	    failure = true;
	    continue;
	  }

	  // neural network weights
	  math::vertex<T> w;

	  if(nnet.exportdata(w) == false){
	    failure = true;
	    continue;
	  }
	  
	  linear_parameters_u[n] = u; // OpenMPfied code
	  nnetwork_parameters_w[n] = w;
	}

	// OpenMPfied code
#pragma omp critical (ieworweoirjeowAAATHHD)
	{
	  counter++;
	  eta.update((float)(counter));
	}
	
      } // for(N)
      
    }

    if(failure) return false;

      
    // 2. now we have N linear optimum and non-linear local optimum weights
    // calculate linear optimum mapping f(u) = w, w = A*discretize(u) + b
    math::matrix<T> A;
    math::vertex<T> b;
    math::vertex<T> umean, ustdev;

    {
      printf("Computing linear to non-linear function parameter mapping.\n");
      
      // linear_parameters_u => nnetwork_parameters_w
      dataset<T> dmap;
      
      if(dmap.createCluster("input", linear_parameters_u[0].size()) == false) return false;
      if(dmap.createCluster("output", nnetwork_parameters_w[0].size()) == false) return false;
      
      if(dmap.add(0, linear_parameters_u) == false) return false;
      if(dmap.add(1, nnetwork_parameters_w) == false) return false;

      if(dmap.preprocess(0, dataset<T>::dnMeanVarianceNormalization) == false) return false;

      if(global_linear_optimizer(dmap, K, A, b) == false) return false;

      if(data_statistics(linear_parameters_u, umean, ustdev) == false)
	return false;
      
    }
    
    
    // 3. we have now linear optimum mapping f(u) = w, map original problem to
    // linear optimum solution and solve u = vec(AA,bb), then solve initial weights w
    {
      printf("Global linear optimizer parameter mapping.\n");
      
      math::matrix<T> AA;
      math::vertex<T> bb;
      
      if(global_linear_optimizer(data, K, AA, bb) == false) return false;
      
      // vectorizes AA and bb
      math::vertex<T> u1;
      if(AA.vec(u1) == false) // vectorizes matrix (row1, row2, row3, row(COL:th))
	return false;
      
      // math::vertex<T> u2 = bb;
      
      math::vertex<T> u;
      u.resize(u1.size() + bb.size());
      if(u.write_subvertex(u1, 0) == false) return false;
      if(u.write_subvertex(bb, u1.size()) == false) return false;
      
      // solve weights w using w = A*discretize(u) + b
      math::vertex<T> du;
      du.resize(u.size()*K);
      du.zero();
      
      for(unsigned int i=0;i<u.size();i++){
	int k = discretize(u[i], K, umean[i], ustdev[i]);
	if(k < 0) return false;
	du[i*K + k] = T(1.0f);
      }
      
      math::vertex<T> w = A*du + b;
      
      // imports pretrained weights
      if(net.importdata(w) == false) return false;
    }

    
    return true;
  }

  

  // solves global linear optimum by discretizing data to K bins per dimension
  // it is assumed that input data x is normalized to have zero mean = 0 and
  // unit variance/standard deviation = 1
  // solves global optimum of pseudolinear "y = Ax + b" by solving least squares problem
  template <typename T>
  bool global_linear_optimizer(const whiteice::dataset<T>& data,
			       const unsigned int K,
			       whiteice::math::matrix<T>& A,
			       whiteice::math::vertex<T>& b)
  {
    if(K <= 1) return false;
    if(data.getNumberOfClusters() <= 1) return false;
    if(data.size(0) != data.size(1)) return false;

    std::vector< enum whiteice::dataset<T>::data_normalization > dn;
    if(data.getPreprocessings(0, dn) == false) return false;
    if(dn.size() != 1) return false;
    if(dn[0] != whiteice::dataset<T>::dnMeanVarianceNormalization)
      return false;

    
    // discretizes problem variables (only input)
    std::vector< math::vertex<T> > xdata;
    std::vector< math::vertex<T> > ydata;
    
    {
      T mean  = T(0.0f);
      T stdev = T(1.0f);

      for(unsigned int j=0;j<data.size(0);j++){
	const math::vertex<T>& v = data.access(0, j);
	math::vertex<T> u;
	u.resize(data.dimension(0)*K);
	u.zero();

	for(unsigned int i=0;i<v.size();i++){
	  int k = discretize(v[i], K, mean, stdev);
	  if(k < 0) return false;
	  else if(k >= (signed)K) k = K-1;
	  u[i*K + k] = T(1.0f);
	}

	xdata.push_back(u);
	ydata.push_back(data.access(1,j));
      }
    }
    

    // solves linear problem "y = Ax + b", "min(A,b) 0.5*||y - A*x - b||^2"
    // solution is A = Cyx*pseudoinverse(Cxx), b = mean(y) - A*mean(x)
    {
      math::matrix<T> Cxx, Cyx;
      math::vertex<T> mx, my;

      if(mean_covariance_estimate(mx, Cxx, xdata) == false)
	return false;

      if(mean_crosscorrelation_estimate(mx, my, Cyx, xdata, ydata) == false)
	return false;

      if(Cxx.symmetric_pseudoinverse(T(0.0001f)) == false)
	return false;

      A = Cyx*Cxx;
      b = my - A*mx;
    }

    return true;
  }
  


  // discretizes each variable to K bins: [-3*stdev, 3*stdev]/K
  template <typename T>
  bool discretize_problem(const unsigned int K,
			  const std::vector< math::vertex<T> >& input,
			  std::vector< math::vertex<T> >& inputDiscrete,
			  whiteice::math::vertex<T>& mean,
			  whiteice::math::vertex<T>& stdev)
  {
    if(K <= 1) return false; // must have at least 2 bins to have meaningful discretization
    if(input.size() <= 1) return false;

    inputDiscrete.clear();
    
    math::vertex<T> x2;
    {
      mean.resize(input[0].size());
      x2.resize(input[0].size());
      stdev.resize(input[0].size());
      mean.zero();
      x2.zero();
      stdev.zero();
      
      for(const auto& x : input){
	mean += x;
	for(unsigned int j=0;j<x2.size();j++)
	  x2[j] += x[j]*x[j];
      }
      
      mean /= T(input.size());
      x2 /= T(input.size());

      for(unsigned int j=0;j<x2.size();j++){
	stdev[j] = x2[j] - mean[j]*mean[j];
	stdev[j] = sqrt(abs(stdev[j]));
	if(stdev[j] <= T(0.001))
	  stdev[j] = T(0.001);
      }
    }

    // now discretizes data using mean and stdev
    {
      for(const auto& x : input){
	math::vertex<T> d;
	d.resize(x.size()*K);
	d.zero();

	for(unsigned int j=0;j<x.size();j++){
	  T s = (x[j] - mean[j])/(T(3.0)*stdev[j]); // [-1,1] is the used range.
	  s = whiteice::math::round(T(K/2.0f)*s + T(K/2.0f)); // [-K/2, K/2] + K/2 = [0,K]
	  int k = 0;
	  if(s > T((float)K)) s = T((float)K);
	  whiteice::math::convert(k, s); // converts T to integer

	  if(k < 0) k = 0;
	  if(k >= (int)K) k = K-1;

	  d[K*j + k] = T(1.0f);
	}

	inputDiscrete.push_back(d);
	
      }
    }
    
    
    return true;
  }
  

  // discretizes each variable to K bins: [-3*stdev, 3*stdev]/K
  // discretization is calculated (K/2)*((x-mean)/3*stdev)+K/2 => [0,K[ (clipped to interval)
  // it is recommended that K is even number
  template <typename T>
  int discretize(const T& x, const unsigned int K, const T& mean, const T& stdev)
  {
    if(K <= 1) return -1;
    
    T sd = stdev;
    
    if(sd <= T(0.001))
      sd = T(0.001);
    
    T s = (x - mean)/(T(3.0)*sd); // [-1,1] is the used range.
    
    s = whiteice::math::round(T(K/2.0f)*s + T(K/2.0f)); // [-K/2, K/2] + K/2 = [0,K]
    int k = 0;
    if(s > T((float)K)) s = T((float)K);
    whiteice::math::convert(k, s); // converts T to integer
    
    if(k < 0) k = 0;
    if(k >= (int)K) k = K-1;

    return k;
  }
  
  
  // statistics to calculate discretization
  template <typename T>
  bool data_statistics(const std::vector< math::vertex<T> >& input,
		       whiteice::math::vertex<T>& mean,
		       whiteice::math::vertex<T>& stdev)
  {

    if(input.size() <= 1) return false;

    math::vertex<T> x2;
    {
      mean.resize(input[0].size());
      x2.resize(input[0].size());
      stdev.resize(input[0].size());
      mean.zero();
      x2.zero();
      stdev.zero();
      
      for(const auto& x : input){
	mean += x;
	for(unsigned int j=0;j<x2.size();j++)
	  x2[j] += x[j]*x[j];
      }
      
      mean /= T(input.size());
      x2 /= T(input.size());

      for(unsigned int j=0;j<x2.size();j++){
	stdev[j] = x2[j] - mean[j]*mean[j];
	stdev[j] = sqrt(abs(stdev[j]));
	if(stdev[j] <= T(0.001))
	  stdev[j] = T(0.001);
      }
    }

    return true;
  }

  //////////////////////////////////////////////////////////////////////

  template bool global_optimizer_pretraining
  (nnetwork< math::blas_real<float> >& net,
   const dataset< math::blas_real<float> >& data, 
   const unsigned int N,
   const unsigned int M,
   const unsigned int K);

  template bool global_optimizer_pretraining
  (nnetwork< math::blas_real<double> >& net,
   const dataset< math::blas_real<double> >& data, 
   const unsigned int N,
   const unsigned int M,
   const unsigned int K);

  
  
  
  template bool global_linear_optimizer
  (const whiteice::dataset< math::blas_real<float> >& data,
   const unsigned int K,
   whiteice::math::matrix< math::blas_real<float> >& A,
   whiteice::math::vertex< math::blas_real<float> >& b);
  
  template bool global_linear_optimizer
  (const whiteice::dataset< math::blas_real<double> >& data,
   const unsigned int K,
   whiteice::math::matrix< math::blas_real<double> >& A,
   whiteice::math::vertex< math::blas_real<double> >& b);
  
  
  template bool discretize_problem(const unsigned int K,
				   const std::vector< math::vertex< math::blas_real<float> > >& input,
				   std::vector< math::vertex< math::blas_real<float> > >& inputDiscrete,
				   whiteice::math::vertex< math::blas_real<float> >& mean,
				   whiteice::math::vertex< math::blas_real<float> >& stdev);
  
  template bool discretize_problem(const unsigned int K,
				   const std::vector< math::vertex< math::blas_real<double> > >& input,
				   std::vector< math::vertex< math::blas_real<double> > >& inputDiscrete,
				   whiteice::math::vertex< math::blas_real<double> >& mean,
				   whiteice::math::vertex< math::blas_real<double> >& stdev);
  


  
  template int discretize(const math::blas_real<float>& x,
			  const unsigned int K,
			  const math::blas_real<float>& mean,
			  const math::blas_real<float>& stdev);

  template int discretize(const math::blas_real<double>& x,
			  const unsigned int K,
			  const math::blas_real<double>& mean,
			  const math::blas_real<double>& stdev);
  

  
  template 
  bool data_statistics(const std::vector< math::vertex< math::blas_real<float> > >& input,
		       whiteice::math::vertex< math::blas_real<float> >& mean,
		       whiteice::math::vertex< math::blas_real<float> >& stdev);
  
  template 
  bool data_statistics(const std::vector< math::vertex< math::blas_real<double> > >& input,
		       whiteice::math::vertex< math::blas_real<double> >& mean,
		       whiteice::math::vertex< math::blas_real<double> >& stdev);  

};
