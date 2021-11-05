
#include "rLBFGS_nnetwork.h"
#include "deep_ica_network_priming.h"

#include "eig.h"
#include "EnsembleMeans.h"


namespace whiteice
{

  template <typename T>
  rLBFGS_nnetwork<T>::rLBFGS_nnetwork(const nnetwork<T>& nn,
				      const dataset<T>& d,
				      const unsigned int deepness_,
				      bool overfit,
				      bool negativefeedback) :
    whiteice::math::LBFGS<T>(overfit), deepness(deepness_),
    net(nn), data(d)
  {
    this->negativefeedback = negativefeedback;
    
    assert(data.getNumberOfClusters() == 2);

    // checks network has correct architecture
    if(deepness > 1)
    {
      const unsigned int RDIM = net.input_size()-data.dimension(0);

      assert(RDIM >= 1);
      
      assert(net.input_size() == data.dimension(0)+RDIM);
      assert(net.output_size() == data.dimension(1)+RDIM);
    }
    else{
      assert(net.input_size() == data.dimension(0));
      assert(net.output_size() == data.dimension(1));
    }
    

    // divides data to to training and testing sets
    ///////////////////////////////////////////////
    {
      
      dtrain = data;
      dtest  = data;
      
      dtrain.clearData(0);
      dtrain.clearData(1);
      dtest.clearData(0);
      dtest.clearData(1);
      
      
      for(unsigned int i=0;i<data.size(0);i++){
	const unsigned int r = (rand() & 3);
	
	if(r != 0){ // 75% will to training data
	  math::vertex<T> in  = data.access(0,i);
	  math::vertex<T> out = data.access(1,i);
	  
	  dtrain.add(0, in,  true);
	  dtrain.add(1, out, true);
	}
	else{ // 25% will go to testing data
	  math::vertex<T> in  = data.access(0,i);
	  math::vertex<T> out = data.access(1,i);
	  
	  dtest.add(0, in,  true);
	  dtest.add(1, out, true);
	}
      }
      
      // we cannot never have zero training or testing set size
      // in such a small cases (very little data) we just use
      // all the data both for training and testing and overfit
      if(dtrain.size(0) == 0 || dtest.size(0) == 0){
	dtrain = data;
	dtest  = data;
      }
    }
    
  }

  
  template <typename T>
  rLBFGS_nnetwork<T>::~rLBFGS_nnetwork()
  {
  }


  template <typename T>
  T rLBFGS_nnetwork<T>::getError(const math::vertex<T>& x) const
  {
    T e = T(0.0f);

    if(deepness <= 1){
    
#pragma omp parallel shared(e)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> err;
	T esum = T(0.0f);
	
	
	// E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(auto)
	for(unsigned int i=0;i<dtest.size(0);i++){
	  nnet.input() = dtest.access(0, i);
	  nnet.calculate(false);
	  err = dtest.access(1, i) - nnet.output();
	  
	  err = (err*err);
	  esum += T(0.5f)*err[0];
	}
	
#pragma omp critical (dsjairerew)
	{
	  e += esum;
	}
	
      }
    }
    else{ // recurrent neural network structure
      
#pragma omp parallel shared(e)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> err;
	T esum = T(0.0f);
	
	const unsigned int INPUT_DATA_DIM = dtest.dimension(0);
	const unsigned int OUTPUT_DATA_DIM = dtest.dimension(1);
	const unsigned int RDIM = nnet.output_size() - OUTPUT_DATA_DIM;

	math::vertex<T> input, output;
	input.resize(dtest.dimension(0)+RDIM);
	output.resize(RDIM);
	
	// E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(auto)
	for(unsigned int i=0;i<dtest.size(0);i++){
	  input.zero();
	  input.write_subvertex(dtest.access(0,i), 0);

	  // recurrency: feebacks output back to inputs and
	  //             calculates error
	  for(unsigned int d = 0;d<deepness;d++){
	    nnet.input() = input;
	    nnet.calculate(false);
	    
	    nnet.output().subvertex(output, dtest.dimension(1), RDIM);
	    assert(input.write_subvertex(output, INPUT_DATA_DIM));
	  }

	  nnet.output().subvertex(err, 0, dtest.dimension(1));
	  err -= dtest.access(1, i);
	  err = (err*err);
	  esum += T(0.5f)*err[0];
	}
	
#pragma omp critical (reiote2qweq)
	{
	  e += esum; // per each recurrency
	}
	
      }
      
    }
    
    e /= T( (float)dtest.size(0) ); // per N
    
    return e;
  }

  

  template <typename T>
  T rLBFGS_nnetwork<T>::U(const math::vertex<T>& x) const
  {
    T e = T(0.0f);

    const bool use_minibatch = true;

    if(use_minibatch){
      const unsigned int MINIBATCHSIZE = 100;

      if(deepness <= 1){
#pragma omp parallel
	{
	  whiteice::nnetwork<T> nnet(this->net);
	  
	  nnet.importdata(x);
	  math::vertex<T> err;
	  T esum = T(0.0f);
	  
	  // E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(auto)
	  for(unsigned int i=0;i<MINIBATCHSIZE;i++){
	    const unsigned int index =
	      whiteice::math::LBFGS<T>::rng.rand() % dtrain.size(0);
	    nnet.input() = dtrain.access(0, index);
	    nnet.calculate(false);
	    err = dtrain.access(1, index) - nnet.output();
	    err = (err*err); // /T(dtrain.size(0));
	    esum += T(0.5f)*err[0];
	  }
	  
#pragma omp critical (fsjogwrgbwe)
	  {
	    e += esum;
	  }
	
	}
      
      }
      else{ // recurrent neural network
	
#pragma omp parallel shared(e)
	{
	  whiteice::nnetwork<T> nnet(this->net);
	  
	  nnet.importdata(x);
	  math::vertex<T> err;
	  T esum = T(0.0f);
	  
	  const unsigned int INPUT_DATA_DIM = dtrain.dimension(0);
	  const unsigned int OUTPUT_DATA_DIM = dtrain.dimension(1);
	  const unsigned int RDIM = nnet.output_size() - OUTPUT_DATA_DIM;
	  
	  math::vertex<T> input, output;
	  input.resize(nnet.input_size());
	  output.resize(RDIM);
	  
	  // E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(auto)
	  for(unsigned int i=0;i<MINIBATCHSIZE;i++){
	    const unsigned int index =
	      whiteice::math::LBFGS<T>::rng.rand() % dtrain.size(0);
	    input.zero();
	    assert(input.write_subvertex(dtrain.access(0,index), 0));
	    
	    // recurrency: feebacks output back to inputs and
	    //             calculates error
	    for(unsigned int d = 0;d<deepness;d++){
	      nnet.input() = input;
	      nnet.calculate(false);
	      
	      if(d == (deepness - 1))
	      {
		// only the last error term E(N) matters
		nnet.output().subvertex(err, 0, dtrain.dimension(1));
		err -= dtrain.access(1, index);
		
		err = (err*err);
		esum += T(0.5f)*err[0];
	      }
	      
	      
	      nnet.output().subvertex(output, dtrain.dimension(1),RDIM);
	      assert(input.write_subvertex(output, INPUT_DATA_DIM));
	    }
	    
	  }
	  
#pragma omp critical (rewuiotymce)
	  {
	    e += esum;
	  }
	  
	}
	
      }
      
      e /= T(MINIBATCHSIZE);
    }
    else{ // don't use minibatch

      if(deepness <= 1){
#pragma omp parallel
	{
	  whiteice::nnetwork<T> nnet(this->net);
	  
	  nnet.importdata(x);
	  math::vertex<T> err;
	  T esum = T(0.0f);
	  
	  // E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(auto)
	  for(unsigned int i=0;i<dtrain.size(0);i++){
	    nnet.input() = dtrain.access(0, i);
	    nnet.calculate(false);
	    err = dtrain.access(1, i) - nnet.output();
	    err = (err*err); // /T(dtrain.size(0));
	    esum += T(0.5f)*err[0];
	  }
	  
#pragma omp critical (gjkvmrore)
	  {
	    e += esum;
	  }
	
	}
      
      }
      else{ // recurrent neural network
	
#pragma omp parallel shared(e)
	{
	  whiteice::nnetwork<T> nnet(this->net);
	  
	  nnet.importdata(x);
	  math::vertex<T> err;
	  T esum = T(0.0f);
	  
	  const unsigned int INPUT_DATA_DIM = dtrain.dimension(0);
	  const unsigned int OUTPUT_DATA_DIM = dtrain.dimension(1);
	  const unsigned int RDIM = nnet.output_size() - OUTPUT_DATA_DIM;
	  
	  math::vertex<T> input, output;
	  input.resize(nnet.input_size());
	  output.resize(RDIM);
	  
	  // E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(auto)
	  for(unsigned int i=0;i<dtrain.size(0);i++){
	    input.zero();
	    assert(input.write_subvertex(dtrain.access(0,i), 0));
	    
	    // recurrency: feebacks output back to inputs and
	    //             calculates error
	    for(unsigned int d = 0;d<deepness;d++){
	      nnet.input() = input;
	      nnet.calculate(false);
	      
	      if(d == (deepness - 1))
	      {
		// only the last error term E(N) matters
		nnet.output().subvertex(err, 0, dtrain.dimension(1));
		err -= dtrain.access(1, i);
		
		err = (err*err);
		esum += T(0.5f)*err[0];
	      }
	      
	      
	      nnet.output().subvertex(output, dtrain.dimension(1),RDIM);
	      assert(input.write_subvertex(output, INPUT_DATA_DIM));
	    }
	    
	  }
	  
#pragma omp critical (cmbnreorpww)
	  {
	    e += esum;
	  }
	  
	}
	
      }
      
#if 0
      {
	// regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
	T alpha = T(0.01);
	auto err = T(0.5)*alpha*(x*x);
	e += err[0];
      }
#endif

      e /= T(dtrain.size(0));
    } 
    
    return (e);    
  }

  template <typename T>
  math::vertex<T> rLBFGS_nnetwork<T>::Ugrad(const math::vertex<T>& x) const
  {

    if(deepness <= 1){ // non-recurrent neural network (normal gradient)
      math::vertex<T> sumgrad;
      sumgrad = x;
      sumgrad.zero();
      
#pragma omp parallel shared(sumgrad)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> grad, err;
	math::vertex<T> sgrad;
	sgrad = x;
	sgrad.zero();
	
#pragma omp for nowait schedule(auto)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  nnet.input() = dtrain.access(0, i);
	  nnet.calculate(true);
	  err = nnet.output() - dtrain.access(1,i);
	  
	  if(nnet.mse_gradient(err, grad) == false){
	    std::cout << "gradient failed." << std::endl;
	    assert(0); // FIXME
	  }

	  sgrad += grad;
	}
	
#pragma omp critical (otiepwrmcoid)
	{
	  sumgrad += sgrad;
	}
	
      }

#if 0
      {
	// regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
	T alpha = T(0.01);
	auto err = alpha*x;

	sumgrad += alpha*x;
      }
#endif

      sumgrad /= T(dtrain.size(0));

      return sumgrad;
    }
    else{
      // recurrent neural network!
      math::vertex<T> sumgrad;
      sumgrad = x;
      sumgrad.zero();

      const unsigned int INPUT_DATA_DIM = dtrain.dimension(0);
      const unsigned int OUTPUT_DATA_DIM = dtrain.dimension(1);
      const unsigned int RDIM = net.output_size() - OUTPUT_DATA_DIM;
      const unsigned int RDIM2 = net.input_size() - INPUT_DATA_DIM;
      assert(RDIM == RDIM2);
      
#pragma omp parallel shared(sumgrad)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> grad, err;
	math::vertex<T> sgrad;
	sgrad = x;
	sgrad.zero();
	grad = x;

	math::vertex<T> input, output;
	input.resize(dtrain.dimension(0)+RDIM);
	output.resize(RDIM);

	math::matrix<T> UGRAD;
	UGRAD.resize(dtrain.dimension(1)+RDIM, nnet.gradient_size());

	math::matrix<T> URGRAD;
	URGRAD.resize(RDIM, nnet.gradient_size());

	math::matrix<T> UYGRAD;
	UYGRAD.resize(dtrain.dimension(1), nnet.gradient_size());

	math::matrix<T> FGRAD;
	FGRAD.resize(dtrain.dimension(1)+RDIM, nnet.gradient_size());

	math::matrix<T> FRGRAD;
	FRGRAD.resize(RDIM, nnet.output_size());

	math::matrix<T> FGRADTMP;
	FGRADTMP.resize(dtrain.dimension(1)+RDIM, RDIM);

#pragma omp for nowait schedule(auto)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  UGRAD.zero();
	  grad.zero();
	  input.zero();
	  input.write_subvertex(dtrain.access(0,i), 0);
	  
	  for(unsigned int d=0;d<deepness;d++){
	    
	    assert(nnet.jacobian(input, FGRAD) == true);
	    // df/dw (dtrain.dimension(1)+RDIM, nnet.gradient_size())

	    {
	      assert(nnet.gradient_value(input, FGRADTMP) == true);
	      // df/dinput (dtrain.dimension(1)+RDIM,dtrain.dimension(0)+RDIM)

	      // df/dr
	      assert(FGRADTMP.submatrix(FRGRAD,
					dtrain.dimension(0), 0,
					RDIM, nnet.output_size()) == true);

	      // KAPPA_r = I

	      // df/dr (dtrain.dimension(1)+RDIM, RDIM)
	      // dU/dw (dtrain.dimension(1)+RDIM, nnet.gradient_size())

	      // KAPPA_r operation to UGRAD to select only R terms
	      assert(UGRAD.submatrix(URGRAD,
				     0,dtrain.dimension(1),
				     nnet.gradient_size(), RDIM) == true);
	    }

	    // dU(n+1)/dw = df/dw + df/dr * KAPPA_r * dU(n)/dw
	    UGRAD = FGRAD + FRGRAD*URGRAD;

	    nnet.input() = input;
	    nnet.calculate(false);
	    
	    if(d == (deepness - 1))
	    { // only last E(N) error term matters
	      // calculate gradient value
	      nnet.output().subvertex(err, 0, dtrain.dimension(1));
	      err -= dtrain.access(1,i);

	      // selects only Y terms from UGRAD
	      assert(UGRAD.submatrix
		     (UYGRAD,
		      0,0,
		      nnet.gradient_size(), dtrain.dimension(1)));

	      grad = err*UYGRAD;

	      sgrad += grad;
	    }

	    nnet.output().subvertex(output, dtrain.dimension(1),RDIM);
	    assert(input.write_subvertex(output, INPUT_DATA_DIM));
	  }

	}
	
#pragma omp critical (poirifmejs)
	{
	  sumgrad += sgrad;
	}
	
      }

#if 0
      {
	// regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
	T alpha = T(0.01);
	auto err = alpha*x;

	sumgrad += alpha*x;
      }
#endif

      sumgrad /= T(dtrain.size(0));

      return sumgrad;
    }


  }
  

#if 0
  template <typename T>
  math::vertex<T> rLBFGS_nnetwork<T>::Ugrad(const math::vertex<T>& x) const
  {
    math::vertex<T> sumgrad;
    sumgrad = x;
    sumgrad.zero();

    // we collect distribution of gradients and clusterize it and take majority direction
    // this should mean our gradint (machine learning pattern) is "less confused" so that
    // attempts to improve nnetwork is less likely to suffer
    // from "multiple good directions" problem or noise..
    
    std::vector< math::vertex<T> > pgradients;
    std::vector< math::vertex<T> > ngradients;

    // we sample approximately only 100 samples from all possible gradients
    const T gradprob = T(100.0)/T(dtrain.size(0));

    // number of clusters is random and varies between
    // 1-2 (50% for total gradient, 25% majority, 25% minority cluster)
    // const unsigned int NUMCLUSTERS = 1 + (rng.rand() & 1);
    const unsigned int NUMCLUSTERS = 1;

    // 50% probability to clusterize gradients and
    // try to follow it instead of using global mean gradient
    // (this denoises gradient and reduces pull
    //  to multiple different directions)
#if 0
    if(rng.rand() & 1)
      NUMCLUSTERS = 3; 
    else
      NUMCLUSTERS = 1;
#endif
    

    if(deepness <= 1){

      // positive phase/gradient
      // (calculate distribution of positive gradients)
#pragma omp parallel shared(sumgrad) shared(pgradients)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> grad, err;
	std::vector< math::vertex<T> > grads; // grads direction
	
#pragma omp for nowait schedule(auto)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  nnet.input() = dtrain.access(0, i);
	  nnet.calculate(true);
	  err = dtrain.access(1,i) - nnet.output();
	  
	  if(nnet.gradient(err, grad) == false){
	    std::cout << "gradient failed." << std::endl;
	    assert(0); // FIXME
	  }

	  if(rng.uniform() < gradprob) // keep only p% of gradients..
	    grads.push_back(grad);
	}
	
#pragma omp critical (mkfjhiwmdsa)
	{
	  for(unsigned int i=0;i<grads.size();i++){
	    {
	      auto g = grads[i];
	      // g.normalize();
	      pgradients.push_back(g);
	    }
	  }
	}
	
      }

      std::vector<bool> cluster;
      unsigned int chosenCluster;
      whiteice::EnsembleMeans<T> em;

      // we clusterize and use majority cluster of gradients
      {
	whiteice::math::vertex<T> c;
	T p;
	
	em.learn(NUMCLUSTERS, pgradients); // 3 groups and we keep majority cluster
	chosenCluster = (unsigned int)em.getProbabilisticCluster(c, p);
	// chosenCluster = (unsigned int)em.getMajorityCluster(c, p);
	
	// em.clusterize(pgradients, cluster);
	cluster.resize(dtrain.size(0));
      }

      math::matrix<T> sigma2;
      sigma2.resize(net.output_size(), net.output_size());
      sigma2.zero();
      
      math::vertex<T> m(net.output_size());
      m.zero();

      unsigned int numGradients = 0;
      
      // positive phase/gradient
      // (calculate majority clusters gradient
#pragma omp parallel shared(sumgrad) shared(pgradients)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> sgrad, grad, err;
	std::vector< math::vertex<T> > grads; // grads direction
	
	math::matrix<T> sum_sigma2;
	sum_sigma2.resize(net.output_size(), net.output_size());
	sum_sigma2.zero();
	
	math::vertex<T> sum_m(net.output_size());
	sum_m.zero();
	
	sgrad = x;
	sgrad.zero();
	
#pragma omp for nowait schedule(auto)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  {
	    nnet.input() = dtrain.access(0, i);
	    nnet.calculate(true);
	    err = dtrain.access(1,i) - nnet.output();
	    
	    if(nnet.gradient(err, grad) == false){
	      std::cout << "gradient failed." << std::endl;
	      assert(0); // FIXME
	    }

	    auto norm_grad = grad;
	    // norm_grad.normalize();

	    if(em.getCluster(norm_grad) == chosenCluster){
	      sum_sigma2 += err.outerproduct(err);
	      sum_m += err; // should not be calculated??
	      
	      sgrad += grad; // /T(dtrain.size(0));

	      cluster[i] = true;
	      numGradients++;
	    }
	    else{
	      cluster[i] = false;
	    }
	  }
	}
	
#pragma omp critical (xmnrewoijokcsrwe)
	{
	  sumgrad += sgrad;
	  sigma2 += sum_sigma2;
	  m += sum_m;
	}
	
      }
      
      sigma2 /= T(numGradients);
      m /= T(numGradients);
      sigma2 -= m.outerproduct(m);
      
      // sigma2 is error covariance matrix
      // (we assume there are only few output dimensions)
      // we need to sample from N(O, Sigma2) solve X*sqrt(D)*n
      auto D = sigma2;
      math::matrix<T> X;
      
      while(symmetric_eig(D, X) == false){
	for(unsigned int i=0;i<m.size();i++)
	  sigma2(i,i) += T(1.0); // dummy regularizer..
	D = sigma2;
      }
      
      sigma2 = D;
      
      for(unsigned int i=0;i<m.size();i++){
	sigma2(i,i) = sqrt(abs(sigma2(i,i)));
      }
      
      sigma2 = X*sigma2;
      

#if 1
      // negative phase/gradient
#pragma omp parallel shared(sumgrad) shared(ngradients)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> sgrad, grad, err;
	std::vector< math::vertex<T> > grads; // grads direction
	
	sgrad = x;
	sgrad.zero();
	
#pragma omp for nowait schedule(auto)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  if(cluster[i]){ // only selected cluster
	    
	    // generates negative particle
	    auto x = dtrain.access(0, rng.rand() % dtrain.size(0));
	    
	    nnet.input() = x;
	    nnet.calculate(true);
	    auto y = nnet.output();
	    
	    math::vertex<T> n(y.size());
	    rng.normal(n);
	    y += sigma2 * n; // adds properly correlated noise..
	    
	    err = y - nnet.output();
	    
	    if(nnet.gradient(err, grad) == false){
	      std::cout << "gradient failed." << std::endl;
	      assert(0); // FIXME
	    }

	    // grads.push_back(-grad);	  
	    sgrad -= grad; // /T(dtrain.size(0));
	  }
	  
	}
	
#pragma omp critical (lporkcmeajifew)
	{
	  //for(unsigned int i=0;i<grads.size();i++)
	  // ngradients.push_back(grads[i]);
	  
	  sumgrad += sgrad;
	}
	
      }
#endif // disables negative phase
      
      {
	T alpha = T(0.01f);
	sumgrad += alpha*x;
      }

      sumgrad /= T(numGradients);

      // sumgrad.normalize();
      
      return (sumgrad);
    }
    else{ // recurrent neural network!

      std::vector< math::matrix<T> > sigma2;
      std::vector< math::vertex<T> > m;
      sigma2.resize(deepness);
      m.resize(deepness);

      for(unsigned int d=0;d<deepness;d++){
	sigma2[d].resize(net.output_size(), net.output_size());
	sigma2[d].zero();
	m[d].resize(net.output_size());
	m[d].zero();
      }

      
      // positive phase/gradient
      // calculates distribution of positive phase gradients first
#pragma omp parallel shared(sumgrad) shared(pgradients)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> sgrad, grad, err;
	std::vector< math::vertex<T> > grads;
	
	sgrad = x;
	sgrad.zero();
	
#pragma omp for nowait schedule(auto)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  math::vertex<T> input;
	  input.resize(net.input_size());
	  input.zero();
	  input.write_subvertex(dtrain.access(0,i), 0);

	  if(rng.rand() & 1) // trains to use correct input (50% chance)
	    input.write_subvertex(dtrain.access(1,i), dtrain.dimension(0));

	  sgrad.zero();

	  for(unsigned int d=0;d<deepness;d++){
	    nnet.input() = input;
	    nnet.calculate(true);
	    err = dtrain.access(1,i) - nnet.output();
	    
	    if(nnet.gradient(err, grad) == false){
	      std::cout << "gradient failed." << std::endl;
	      assert(0); // FIXME
	    }
	    
	    sgrad += grad; // /T(dtrain.size(0));

	    input.write_subvertex(nnet.output(), dtrain.dimension(0));
	  }

	  if(rng.uniform() < gradprob) // keep only p% of gradients
	    grads.push_back(sgrad);
	  
	}
	
#pragma omp critical (motewjofwkocsw)
	{
	  // positive gradients
	  for(unsigned int i=0;i<grads.size();i++){
	    {
	      auto g = grads[i];
	      // g.normalize();
	      pgradients.push_back(g);
	    }
	  }
	}
      }


      // next we clusterize positive phase gradients
      std::vector<bool> cluster;
      unsigned int chosenCluster;
      whiteice::EnsembleMeans<T> em;
      
      // we clusterize and use majority cluster of gradients
      {
	whiteice::math::vertex<T> c;
	T p;
	
	em.learn(NUMCLUSTERS, pgradients); // 3 groups and we keep majority cluster
	chosenCluster = (unsigned int)em.getProbabilisticCluster(c, p);
	// chosenCluster = (unsigned int)em.getMajorityCluster(c, p);
	
	// em.clusterize(pgradients, cluster);
	cluster.resize(dtrain.size(0));
      }

      
      unsigned int numGradients = 0;
      
      // positive phase/gradient
#pragma omp parallel shared(sumgrad) shared(sigma2) shared(m) shared(numGradients)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> sgrad, grad, err;
	
	std::vector< math::matrix<T> > sum_sigma2;
	sum_sigma2.resize(deepness);

	std::vector< math::vertex<T> > sum_m;
	sum_m.resize(deepness);

	for(unsigned int d=0;d<deepness;d++){
	  sum_sigma2[d].resize(net.output_size(), net.output_size());
	  sum_sigma2[d].zero();
	  sum_m[d].resize(net.output_size());
	  sum_m[d].zero();
	}
	
	sgrad = x;
	sgrad.zero();
	
#pragma omp for nowait schedule(auto)
	for(unsigned int i=0;i<dtrain.size(0);i++){	  
	  {
	    math::vertex<T> input;
	    input.resize(net.input_size());
	    input.zero();
	    input.write_subvertex(dtrain.access(0,i), 0);
	    
	    if(rng.rand() & 1) // trains to use correct input (50% chance)
	      input.write_subvertex(dtrain.access(1,i), dtrain.dimension(0));
	    
	    auto tmpgrad = sgrad;
	    tmpgrad.zero();
	    
	    std::vector< math::vertex<T> > tmperrors;
	    
	    for(unsigned int d=0;d<deepness;d++){
	      nnet.input() = input;
	      nnet.calculate(true);
	      err = dtrain.access(1,i) - nnet.output();
	      
	      if(nnet.gradient(err, grad) == false){
		std::cout << "gradient failed." << std::endl;
		assert(0); // FIXME
	      }

	      tmperrors.push_back(err);
	      
	      // sum_sigma2[d] += err.outerproduct(err);
	      // sum_m.at(d) += err;
	      
	      // sgrad += grad; // /T(dtrain.size(0));
	      tmpgrad += grad;
	      
	      input.write_subvertex(nnet.output(), dtrain.dimension(0));
	    }

	    auto norm_grad = tmpgrad;
	    // norm_grad.normalize();

	    if(em.getCluster(norm_grad) == chosenCluster){

	      for(unsigned int d=0;d<deepness;d++){
		sum_sigma2[d] += tmperrors[d].outerproduct();
		sum_m[d] += err; // should not be calculated??
	      }
	      
	      sgrad += tmpgrad;

	      cluster[i] = true;
	      numGradients++;
	    }
	    else{
	      cluster[i] = false;
	    }


	  }
	  
	}
	
#pragma omp critical (mckfsepofwfgr)
	{
	  sumgrad += sgrad;

	  for(unsigned int d=0;d<deepness;d++){ 
	    sigma2[d] += sum_sigma2[d];
	    m[d] += sum_m[d];
	  }
	}
	
      }


      for(unsigned int d=0;d<deepness;d++){
      
	sigma2[d] /= T((double)numGradients);
	m[d] /= T((double)numGradients);
	sigma2[d] -= m[d].outerproduct(m[d]);
	
	// sigma2 is error covariance matrix
	// (we assume there are only few output dimensions)
	// we need to sample from N(O, Sigma2) solve X*sqrt(D)*n
	auto D = sigma2[d];
	math::matrix<T> X;
	
	while(symmetric_eig(D, X) == false){
	  for(unsigned int i=0;i<m.size();i++)
	    sigma2[d](i,i) += T(1.0); // dummy regularizer..
	  D = sigma2[d];
	}
      
	sigma2[d] = D;
	
	for(unsigned int i=0;i<m[d].size();i++){
	  sigma2[d](i,i) = sqrt(abs(sigma2[d](i,i)));
	}
	
	sigma2[d] = X*sigma2[d];
      }
      
#if 1
      // negative phase/gradient
#pragma omp parallel shared(sumgrad)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> sgrad, grad, err;
	
	sgrad = x;
	sgrad.zero();
	
#pragma omp for nowait schedule(auto)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  if(cluster[i])
	  {
	    // generates negative particle
	    
	    const unsigned int index = rng.rand() % dtrain.size(0);
	    auto x = dtrain.access(0, index);
	    
	    math::vertex<T> input;
	    input.resize(dtrain.dimension(0)+dtrain.dimension(1));
	    input.zero();
	    input.write_subvertex(x, 0);
	    
	    if(rng.rand() & 1) // trains to use correct input (50% chance)
	      input.write_subvertex(dtrain.access(1,index), dtrain.dimension(0));
	    
	    for(unsigned int d=0;d<deepness;d++){
	      nnet.input() = input;
	      nnet.calculate(true);
	      auto y = nnet.output();
	      
	      math::vertex<T> n(y.size());
	      rng.normal(n);
	      y += sigma2[d] * n; // adds properly correlated noise..
	      
	      err = y - nnet.output();
	      
	      if(nnet.gradient(err, grad) == false){
		std::cout << "gradient failed." << std::endl;
		assert(0); // FIXME
	      }
	      
	      sgrad -= grad; // /T(dtrain.size(0));
	      
	      // input.write_subvertex(nnet.output(), dtrain.dimension(0));
	      input.write_subvertex(y, dtrain.dimension(0)); // INCLUDE NOISE IN FEEDBACK??
	    }
	  }
	}
	
#pragma omp critical (mvmfekpfwdvsh)
	{
	  sumgrad += sgrad;
	}
	
      }      
#endif // disable negative phase gradient calculations

      // regularizer prior
      {
	T alpha = T(0.01f);
	sumgrad += alpha*x;
      }
      
      sumgrad /= T(numGradients); // (normalize to mean value)
      
      // sumgrad.normalize();
      
      return (sumgrad);
    }

  }
#endif
  
  
  template <typename T>
  bool rLBFGS_nnetwork<T>::heuristics(math::vertex<T>& x) const
  {
    if(negativefeedback){
      whiteice::nnetwork<T> nnet(this->net);
      nnet.importdata(x);
      
      T alpha = T(0.5f);
      negative_feedback_between_neurons(nnet, dtrain, alpha);
      
      nnet.exportdata(x);
    }

    return true;
  }
  
  
  template class rLBFGS_nnetwork< math::blas_real<float> >;
  template class rLBFGS_nnetwork< math::blas_real<double> >;

  
};
