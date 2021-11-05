/*
 * t-SNE dimension reduction algorithm.
 * Mathmatical implementation notes/documentation in docs/TSNE_notes.tm (texmacs)
 * Tomas Ukkonen. 2020.
 * 
 */

#include "TSNE.h"

#include "linear_ETA.h"
#include "correlation.h"
#include "fastpca.h"
#include "ica.h"

namespace whiteice
{


  template <typename T>
  TSNE<T>::TSNE(const bool absolute_value)
  {
    // absolute value KL divergence should give better results (see TSNE_notes.tm)
    kl_absolute_value = absolute_value;
  }

  template <typename T>
  TSNE<T>::TSNE(const TSNE<T>& tsne)
  {
    this->kl_absolute_value = tsne.kl_absolute_value;
  }
  
  // dimension reduces samples to DIM dimensional vectors using t-SNE algorithm
  template <typename T>
  bool TSNE<T>::calculate(const std::vector< math::vertex<T> >& samples,
			  const unsigned int DIM,
			  std::vector< math::vertex<T> >& results,
			  const bool verbose,
			  LoggingInterface* const messages,
			  VisualizationInterface* const gui,
			  unsigned int* running_flag)
  {
    if(DIM <= 0) return false;
    if(samples.size() <= 0) return false;
    if(samples[0].size() <= DIM) return false;
    
    if(running_flag)
      if(*running_flag == 0)
	return false; // execution stopped

    this->verbose = verbose;
    

    whiteice::RNG<T> rng;

    // perplexity of 30 seem to give reasonable good results
    const T PERPLEXITY = T(30.0f);

    const unsigned int BUFLEN = 128;
    char buffer[BUFLEN];
    
    // visible points plotted by GUI
    std::vector<unsigned int> guipoints;

    if(gui)
    {
      guipoints.resize(samples.size());

#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<guipoints.size();i++){
	guipoints[i] = i;
      }

      if(guipoints.size() > 10000){ // only plots 10.000 random points

	// randomly reorders points
	for(unsigned int i=0;i<guipoints.size();i++){
	  const unsigned int index = rng.rand() % guipoints.size();
	  std::swap(guipoints[i], guipoints[index]);
	}
	
	guipoints.resize(10000); // keeps only 10.000 first points
      }
    }
    
    {
      float perplexityf = 0.0f;
      whiteice::math::convert(perplexityf, PERPLEXITY);
      snprintf(buffer, BUFLEN, "Estimating p-values (perplexity %f).\n",
	       perplexityf);

      if(verbose){
	printf("%s", buffer);
	fflush(stdout);
      }

      if(messages != NULL)
	messages->printMessage(buffer);
    }

    // preprocesses samples to keep 95% of the total variance of data
    // this mean we will drop low variability constant data or values
    // that are scaled to be very small
    std::vector< math::vertex<T> > xsamples;

    if(1)
    {
      math::matrix<T> PCA;
      math::vertex<T> m;

      T original_var, reduced_var;
      const bool regularize = true;
      const bool unit_variance_normalization = false;

      if(whiteice::math::pca_p(samples, 0.95f, PCA, m,
			       original_var, reduced_var,
			       regularize, unit_variance_normalization) == false){
	printf("ERROR: TSNE::calculate(): pca_p() failed with input data.\n");
	return false;
      }
      else{
	if(verbose){
	  float ovar, rvar;
	  whiteice::math::convert(ovar, original_var);
	  whiteice::math::convert(rvar, reduced_var);
	  
	  printf("TSNE: Preprocessing linear dimension reduction %d->%d (%f var -> %f var).\n",
		 PCA.xsize(), PCA.ysize(), ovar, rvar);
	}
	
	for(const auto& s : samples){
	  xsamples.push_back(PCA*(s - m));
	}
      }
    }
    else{
      for(const auto& s : samples){
	xsamples.push_back(s);
      }
    }
	 

    // calculate p-values
    std::vector< std::vector<T> > pij;
    
    if(calculate_pvalues(xsamples, PERPLEXITY, pij) == false){
      if(verbose){
	printf("Estimating p-values FAILED.\n");
	fflush(stdout);
      }
      return false;
    }

    
    std::vector< math::vertex<T> > yvalues;

    { // initializes y values as gaussian unit variance around zero N(0,I)
      yvalues.resize(samples.size());
      
      for(auto& y : yvalues){
	y.resize(DIM);
	rng.normal(y);
      }
    }
    
    
    // gradient ascend search of yvalues
    {
      std::vector< std::vector<T> > qij;
      T qsum = T(0.0f);

      std::vector< math::vertex<T> > ygrad;
      ygrad = yvalues; // dummy initializer

      unsigned int iter = 0;
      const unsigned int MAXITER = 10000;

      unsigned int noimprove_counter = 0;
      const unsigned int MAXNOIMPROVE = 100;

      T lrate = T(100.0f);
      T best_klvalue = T(1000.0f);
      T klvalue = T(1000.0f);
      T prev_round_klvalue = T(1000.f);

      // implements early stopping
      std::list<T> history;
      const unsigned int HISTORYSIZE = 200;
      const T CONVERGENCE_LIMIT = T(0.02f); // st.dev. is 2.0% of the mean

      whiteice::linear_ETA<double> eta;
      eta.start(0.0, MAXITER);
      eta.update(0.0);

      if(running_flag) if(*running_flag == 0) return false; // execution stopped

      if(gui && DIM >= 2){
	gui->show();
	gui->clear();
      }
      
      while(iter < MAXITER && noimprove_counter < MAXNOIMPROVE)
      {
	// calculates qij values
	if(calculate_qvalues(yvalues, qij, qsum) == false){
	  printf("Calculating q-values FAILED.\n");
	  fflush(stdout);
	  return false;
	}
	
	// calculate and report the current KL divergence
	{
	  prev_round_klvalue = klvalue;
	  
	  if(kl_divergence(pij, qij, klvalue) == false){
	    printf("Calculating KL-divergence FAILED.\n");
	    fflush(stdout);
	    return false;
	  }

	  history.push_back(abs(klvalue));
	  while(history.size() > HISTORYSIZE)
	    history.pop_front();

	  T convergence_statistic = T(0.0f);
	  
	  // calculates convergence statistic
	  if(history.size() >= HISTORYSIZE){
	    T m = T(0.0f);
	    T v = T(0.0f);

	    for(const auto& k : history)
	      m += k;

	    m /= history.size();

	    for(const auto& k : history)
	      v += (k - m)*(k - m);

	    v /= (history.size() - 1);
	    v = sqrt(real(v));

	    convergence_statistic = real(v/m);

	    if(real(convergence_statistic) < real(CONVERGENCE_LIMIT)){
	      // stops computation (changes in KL value are so small)
	      iter = MAXITER+100;
	    }
	  }


	  
	  if(iter == 0){
	    best_klvalue = klvalue;
	  }
	  
	  float klvaluef = 0.0f;
	  whiteice::math::convert(klvaluef, best_klvalue);

	  float convf = 0.0f;
	  whiteice::math::convert(convf, convergence_statistic);

	  if(iter > 0){
	    snprintf(buffer, BUFLEN,
		     "%d/%d: model kl divergence %f. [%f] ETA %f hour(s) (%f minute(s))\n",
		     iter, MAXITER,
		     klvaluef, convf,
		     eta.estimate()/3600.0, eta.estimate()/60.0);
	  }
	  else{
	    snprintf(buffer, BUFLEN, "%d/%d: model kl divergence %f [%f].\n",
		     iter, MAXITER, 
		     klvaluef, convf);	    
	  }

	  if(verbose){
	    printf("%s", buffer);
	    fflush(stdout);
	  }

	  if(messages != NULL){
	    messages->printMessage(buffer);
	  }
	  
	}

	
	{
	  // check if solution improved
	  if(klvalue <= best_klvalue){
	    best_klvalue = klvalue;	    
	    noimprove_counter = 0;
	  }
	  else{
	    noimprove_counter++;
	  }

	  // checks if caller provided flag became false (=> stop)
	  if(running_flag)
	    if(*running_flag == 0)
	      iter = MAXITER+100; // execution stopped
	  
	  // adaptive learning rate
	  if(klvalue < prev_round_klvalue){
	    lrate *= T(1.5f); // solution improved so increase learning rate
	  }
	  else{ // solution worsened so use smaller step length
	    lrate /= T(1.5f);
	  }
	}

	if(gui && DIM >= 2){
	  // converts points to floating point values
	  std::vector< math::vertex< math::blas_real<float> > > points;
	  points.resize(guipoints.size());

# pragma omp parallel for schedule(auto)
	  for(unsigned int k=0;k<guipoints.size();k++){
	    const unsigned int j = guipoints[k];
	    points[k].resize(yvalues[j].size());
	    for(unsigned int i=0;i<yvalues[j].size();i++){
	      whiteice::math::convert(points[k][i], yvalues[j][i]);
	    }
	  }
	  
	  gui->clear();
	  gui->adaptiveScatterPlot(points);
	  gui->updateScreen();
	}
      
	// calculate gradient
	// (minimize KL divergence to match distributions as well as possible)
	if(kl_gradient(pij, qij, qsum, yvalues, ygrad) == false){
	  if(verbose){
	    printf("Calculating KL gradient FAILED.\n");
	    fflush(stdout);
	  }
	  return false;
	}

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<ygrad.size();i++){
	  auto& y = yvalues[i];
	  const auto& g = ygrad[i];

	  y -= lrate*g;
	}

	iter++;
	eta.update(iter);
      }
    }

    // postprocesses low dimensional coordinate results using ICA
    // NOTE: results may not give as good as one may think because
    // data is in 2d/3d plane/cubes in a form of clusters (non-linear map)
    if(yvalues.size() > 0){

      whiteice::math::matrix<T> PCA;
      whiteice::math::vertex<T> m;
      T v1, v2;

      if(whiteice::math::pca(yvalues, yvalues[0].size(), PCA, m, v1, v2, true, true) == true){

	for(auto& v : yvalues){
	  v = PCA*(v - m);
	}

	whiteice::math::matrix<T> ICA;

	if(whiteice::math::ica(yvalues, ICA, false) == true){
	  for(auto& v : yvalues){
	    v = ICA*v;
	  }
	  
	}
	
      }
      
      
      
    }
    
    results = yvalues;
    
    return true;
  }
  
  
  // calculates p values for pj|i where i = index and sigma2 for index:th vector is given
  template <typename T>
  bool TSNE<T>::calculate_pvalue_given_sigma(const std::vector< math::vertex<T> >& x,
					     const unsigned int index, // to x vector
					     const T& sigma2,
					     std::vector<T>& pj) const
  {
    if(index >= x.size()) return false;
    
    pj.resize(x.size());
    
    T rsum = T(0.0f); // calculates rsum for this index

    // too small values cause SFE: arithemtic exception
    const T SIGMA2 = sigma2 < T(1e-30f) ? T(1e-30f) : sigma2;
    
    
#pragma omp parallel // shared(rsum)
    {
      T rs = T(0.0f);
      
      math::vertex<T> delta;
      delta.resize(x[0].size());
      delta.zero();

#pragma omp for nowait schedule(auto)
      for(unsigned int k=0;k<x.size();k++){
	if(index == k) continue;
	delta = x[k] - x[index];

	const T nrm2 = (delta*delta)[0];
	const T v = -nrm2/SIGMA2;
	const T pvalue = whiteice::math::exp(v, T(70.0f));
	
	pj[k] = pvalue;
	rs += pvalue;
      }
      
#pragma omp critical (rewioaaqwejovrevr)
      {
	rsum += rs;
      }
    }
    
    pj[index] = T(0.0f);

    if(rsum > T(0.0f)){
    
#pragma omp parallel for schedule(auto)
      for(unsigned int j=0;j<x.size();j++){
	if(index == j) continue;
	
	pj[j] = pj[j]/rsum;
      }
    }
    else if(x.size() > 1){ // handles all zeros case
      rsum = T(1.0f)/(x.size()-1);

#pragma omp parallel for schedule(auto)      
      for(unsigned int j=0;j<x.size();j++){
	if(index == j) continue;
	
	pj[j] = rsum;
      }
    }
    else
      return false;
      
    return true;
  }


  // calculates distribution's perplexity
  template <typename T>
  T TSNE<T>::calculate_perplexity(const std::vector<T>& pj) const
  {
    if(pj.size() == 0) return T(1.0f);

    T H = T(0.0f); // entropy

#pragma omp parallel // shared(H)
    {
      T h = T(0.0f);

#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<pj.size();i++){
	if(pj[i] > T(0.0f)){
	  h += -pj[i]*math::log(pj[i])/math::log(T(2.0f));
	}
      }

#pragma omp critical (anvcdshoirwereio)
      {
	H += h;
      }
    }

    const T perplexity = math::pow(T(2.0f),H);

    return perplexity;
  }


  // calculate x samples probability distribution p values
  // estimates sigma^2 variance parameter using distribution perplexity values
  template <typename T>
  bool TSNE<T>::calculate_pvalues(const std::vector< math::vertex<T> >& x,
				  const T perplexity,
				  std::vector< std::vector<T> >& pij) const
  {
    if(x.size() == 0) return false;
    

    // calculates p_j|i values with given perplexity
    // perplexity is used to estimate sigma_i parameter of rji/p_j|i
    std::vector< std::vector<T> > rji;
    rji.resize(x.size());

    T total_var = T(0.0f);
    
    // estimates maximum variance: used to search for optimal sigma value
    {
      math::vertex<T> mx;
      math::matrix<T> Cxx;

      if(math::mean_covariance_estimate(mx, Cxx, x) == false) // parallel now
	return false;

      total_var = T(0.0f);

      for(unsigned int i=0;i<Cxx.xsize();i++)
        total_var += Cxx(i,i);

      total_var /= T(x[0].size());
    }

    const T TOTAL_SIGMA2 = total_var;  // total variance in data
    const T MIN_SIGMA2   = TOTAL_SIGMA2/T(1.0f); // was: 10^9

    bool error = false;

#pragma omp parallel
    {
      std::vector<T> pj;

#pragma omp for nowait schedule(auto)
      for(unsigned int j=0;j<x.size();j++){
	if(error) continue; // stopped to error
	
	// searches for sigma2 variance value for x[j] centered distribution
	
	rji[j].resize(x.size());
	
	T sigma2_min = MIN_SIGMA2;
	T sigma2_max = TOTAL_SIGMA2; // maximum sigma value
	T perp_min, perp_max;
	
	{
	  // searches for minimum sigma2 value
	  calculate_pvalue_given_sigma(x, j, sigma2_min, pj);
	  perp_min = calculate_perplexity(pj);

	  while(perp_min > perplexity && sigma2_min > T(0.0f)){
	    sigma2_min /= T(2.0f);
	    calculate_pvalue_given_sigma(x, j, sigma2_min, pj);
	    perp_min = calculate_perplexity(pj);
	  }
	  
	  if(perp_min > perplexity){
	    if(verbose){
	      printf("Could not find minimum perplexity.\n");
	      fflush(stdout);
	    }
	    error = true;
	    continue;
	    // return false; // could not find minimum value
	  }
	  
	  // searches for maximum sigma2 value
	  calculate_pvalue_given_sigma(x, j, sigma2_max, pj);
	  perp_max = calculate_perplexity(pj);

	  while(perp_max < perplexity && sigma2_max < T(10e10f)){
	    sigma2_max *= T(2.0f);
	    calculate_pvalue_given_sigma(x, j, sigma2_max, pj);
	    perp_max = calculate_perplexity(pj);
	  }
	  
	  if(perp_max < perplexity){
	    if(verbose){
	      printf("Could not find maximum perplexity.\n");
	      fflush(stdout);
	    }
	    error = true;
	    continue;
	    // return false; // could not find maximum value
	  }
	}

	// no we have minimum and maximum value for sigma2 variance term
	// search for sigma2 term with target perplexity
	{
	  T sigma2_next = (sigma2_min + sigma2_max)/T(2.0f);
	  
	  calculate_pvalue_given_sigma(x, j, sigma2_next, pj);
	  T perp_next = calculate_perplexity(pj);

	  const T epsilon = T(1e-20f);

	  while(math::abs(perplexity - perp_next) > T(0.1f) &&
		math::abs(sigma2_max - sigma2_min) > epsilon)
	  {
	    if(perplexity < perp_next){
	      sigma2_max = sigma2_next;
	    }
	    else{ // perplexity > perp_next
	      sigma2_min = sigma2_next;
	    }
	    
	    sigma2_next = (sigma2_min + sigma2_max)/T(2.0f);
	    
	    calculate_pvalue_given_sigma(x, j, sigma2_next, pj);
	    perp_next = calculate_perplexity(pj);
	  }
	  
	  if(math::abs(sigma2_max - sigma2_min) <= epsilon){
	    if(verbose){
	      float perplexityf = 0.0f;
	      float perp_nextf = 0.0f;
	      whiteice::math::convert(perplexityf, perplexity);
	      whiteice::math::convert(perp_nextf, perp_next);
	      printf("Could not find sigma^2 value for the target perplexity.\n");
	      printf("Target perplexity: %f. Current perplexity: %f\n",
		     perplexityf, perp_nextf);
	      fflush(stdout);
	    }
	    error = true;
	    continue;
	    // return false; // could not find target perplexity
	  }

	  // we found sigma variance and probability distribution with target perplexity
	  rji[j] = pj;
	  rji[j][j] = T(0.0f);

	}
	  
      }

    }

    if(error) return false;

    
    // calculates symmetric p-values
    pij.resize(x.size());
    
#pragma omp parallel for schedule(auto)
    for(unsigned int i=0;i<x.size();i++){
      pij[i].resize(x.size());

      for(unsigned int j=0;j<x.size();j++){
	if(i == j) continue;

	pij[i][j] = (rji[i][j] + rji[j][i])/T(2.0f*x.size());
      }

      pij[i][i] = T(0.0f);
    }

    return true;
  }


  // calculate dimension reduced y samples probability distribution values q
  template <typename T>
  bool TSNE<T>::calculate_qvalues(const std::vector< math::vertex<T> >& y,
				  std::vector< std::vector<T> >& qij,
				  T& qsum) const
  {
    if(y.size() == 0) return false;
    
    // calculates qij terms
    qij.resize(y.size());

    qsum = T(0.0f);

    const T epsilon = T(1e-12);

#pragma omp parallel // shared(qsum)
    {
      math::vertex<T> delta;
      delta.resize(y[0].size());
      delta.zero();

      T qs = T(0.0f);

#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<y.size();i++){
	qij[i].resize(y.size());
	for(unsigned int j=0;j<y.size();j++){
	  if(i == j) continue; // skip same values
	  delta = y[i] - y[j];
	  auto nrm2 = (delta*delta)[0];
	  auto pvalue = T(1.0f)/(T(1.0f) + nrm2);

	  if(pvalue < epsilon)
	    pvalue = epsilon;

	  qij[i][j] = pvalue;
	  
	  qs += pvalue;
	}

	qij[i][i] = T(0.0f);
      }

#pragma omp critical (vmbnfweofasefg)
      {
	qsum += qs;
      }
    }
      
#pragma omp parallel for schedule(auto)
    for(unsigned int i=0;i<y.size();i++){
      for(unsigned int j=0;j<y.size();j++){
	qij[i][j] /= qsum;
      }
    }

    return true;
  }

  
  template <typename T>
  bool TSNE<T>::kl_divergence(const std::vector< std::vector<T> >& pij,
			      const std::vector< std::vector<T> >& qij,
			      T& klvalue) const
  {
    if(pij.size() <= 0 || qij.size() <= 0) return false;
    if(qij.size() != pij.size()) return false;
    if(pij.size() != pij[0].size()) return false;
    if(qij.size() != qij[0].size()) return false;

    // calculates KL divergence
    klvalue = T(0.0f);

#pragma omp parallel // shared(klvalue)
    {
      T klv = T(0.0f); // threadwise KL divergence terms

#pragma omp for nowait schedule(auto)  
      for(unsigned int i=0;i<pij.size();i++){
	for(unsigned int j=0;j<pij.size();j++){
	  if(i == j) continue; // skip same values
	  
	  if(pij[i][j] == T(0.0f)) // skip zero values (0*log(0) = 0)
	    continue;

	  // ignores zero q values (should not happen after the fix)
	  if(qij[i][j] > T(0.0f)){
	    if(kl_absolute_value)
	      klv += pij[i][j]*whiteice::math::abs(whiteice::math::log(pij[i][j]/qij[i][j]));
	    else
	      klv += pij[i][j]*whiteice::math::log(pij[i][j]/qij[i][j]);
	  }
	}
      }

#pragma omp critical (mgfdjkreowiaa)
      {
	klvalue += klv;
      }
      
    }
    
    return true;
  }


  // calculates gradients of KL diverence
  template <typename T>
  bool TSNE<T>::kl_gradient(const std::vector< std::vector<T> >& pij,
			    const std::vector< std::vector<T> >& qij,
			    const T& qsum,
			    const std::vector< math::vertex<T> >& y,
			    std::vector< math::vertex<T> >& ygrad) const
  {
    if(pij.size() != y.size()) return false;
    if(pij.size() <= 0 || y.size() <= 0) return false;
    if(pij.size() != pij[0].size()) return false;

    ygrad.resize(y.size());

    T Ps = T(0.0f);
    if(kl_absolute_value){

#pragma omp parallel // shared(Ps)
      {
	T psi = T(0.0f);

#pragma omp for nowait schedule(auto)	
	for(unsigned int k=0;k<y.size();k++){
	  for(unsigned int l=0;l<y.size();l++){
	    if(k == l) continue;
	    
	    T ratio = 1.0f;

	    if(qij[k][l] <= T(0.0f)) ratio = T(100.0f); // positive infinity means sign(x)=1
	    else if(pij[k][l] <= T(0.0f)) ratio = T(0.00001); // zero means sign(x)=-1
	    else ratio = pij[k][l]/qij[k][l];
	    
	    const T logratio = whiteice::math::log(ratio);

	    if(logratio >= T(0.0f))
	      psi += pij[k][l];
	    else
	      psi -= pij[k][l];
	  }
	}

#pragma omp critical (mbrjioewaakjrew)
	{
	  Ps += psi;
	}

      }
    }

    
    // calculates gradient for each y value
#pragma omp parallel
    {
      // thread-wise variables
      math::vertex<T> delta;
      delta.resize(y[0].size());

#pragma omp for nowait schedule(auto)
      for(unsigned int m=0;m<ygrad.size();m++){
	ygrad[m].resize(y[0].size());
	ygrad[m].zero();

	if(kl_absolute_value == false){
	  for(unsigned int j=0;j<y.size();j++){
	    if(m == j) continue;
	    
	    delta = y[m] - y[j];
	    
	    ygrad[m] += (pij[m][j]-qij[m][j])*(T(4.0f)/(T(1.0f) + (delta*delta)[0]))*delta;
	  }
	}
	else{

	  for(unsigned int j=0;j<y.size();j++){
	    if(m == j) continue;
	    
	    T ratio = 1.0f;

	    if(qij[m][j] <= T(0.0f)) ratio = T(100.0f); // positive infinity means sign(x)=1
	    else if(pij[m][j] <= T(0.0f)) ratio = T(0.00001); // zero means sign(x)=-1
	    else ratio = pij[m][j]/qij[m][j];
	    
	    const T logratio = whiteice::math::log(ratio);
	    
	    delta = y[m] - y[j];

	    // multiply pij gradient by sign(log(pij/qij))
	    if(logratio >= T(0.0f))
	      ygrad[m] += (pij[m][j]-Ps*qij[m][j])*(T(4.0f)/(T(1.0f) + (delta*delta)[0]))*delta;
	    else
	      ygrad[m] += (-pij[m][j]-Ps*qij[m][j])*(T(4.0f)/(T(1.0f) + (delta*delta)[0]))*delta;
	  }
	}

#if 0
	// calculates gradient of each KL divergence term
	for(unsigned int i=0;i<y.size();i++){
	  for(unsigned int j=0;j<y.size();j++){
	    if(i == j) continue; // skip same values
	    
	    if(pij[i][j] <= T(0.0f)) // skip zero values
	      continue;
	    
	    if(qij[i][j] <= T(0.0f)) // skip zero values (should not happen after the fix
	      continue;

	    const T ratio   = pij[i][j]/qij[i][j];
	    T scaling = -ratio;

	    if(kl_absolute_value){
	      T lr = whiteice::math::log(ratio);
	      if(lr < T(0.0f)) scaling = -scaling; // multiply by sign(lr)
	    }
	    
	    // gradient of qij term
	    if(m == i || m == j){
	      
	      // the first part of derivate
	      {
		const T extra_scaling = T(1.0f)/qsum;
		
		if(m == i)
		  delta = y[i] - y[j];
		else
		  delta = y[j] - y[i];
		
		T qterm = (T(1.0f) + (delta*delta)[0]);
		qterm = T(-2.0f)/(qterm*qterm);
		
		ygrad[m] += scaling*extra_scaling*qterm*delta;
	      }
	      
	      
	      // the second part of the derivate
	      {
		delta = y[i] - y[j];
		const T extra_scaling = -(T(1.0f)/(T(1.0f) + (delta*delta)[0]))/(qsum*qsum);
		
		// gradient
		for(unsigned int l=0;l<y.size();l++){
		  if(l == m) continue;
		  delta = y[m] - y[l];
		  
		  T qterm = (T(1.0f) + (delta*delta)[0]);
		  qterm = -T(4.0f)/(qterm*qterm);
		  
		  ygrad[m] += scaling*extra_scaling*qterm*delta;
		}
	      }
	      
	    }
	    else{ // m != i || m != j
	      
	      // no need for the first part of the derivate
	      
	      // the second part of the derivate
	      {
		delta = y[i] - y[j];
		const T extra_scaling = -(T(1.0f)/(T(1.0f) + (delta*delta)[0]))/(qsum*qsum);
		
		// gradient
		for(unsigned int l=0;l<y.size();l++){
		  if(l == m) continue;
		  delta = y[m] - y[l];
		  
		  T qterm = (T(1.0f) + (delta*delta)[0]);
		  qterm = -T(4.0f)/(qterm*qterm);
		  
		  ygrad[m] += scaling*extra_scaling*qterm*delta;
		}
	      }
	      
	    }
	  }
	}
#endif

      } // parallel OpenMP for-loop ends (ygrad[m])

      
    } // OpenMP block ends

    // calculated all qterms
    return true;
  }


  template class TSNE< math::blas_real<float> >;
  template class TSNE< math::blas_real<double> >;
  
};
