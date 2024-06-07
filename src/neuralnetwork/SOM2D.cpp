/*
 * SOM2D implementation
 *
 * - when implementation is stable / not compiled with electric fence
 *   enable posix_memalign() and disable malloc()s -> faster
 *   (efence doesn't work with posix_memalign())
 *
 */

#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept>
#include <exception>
#include <math.h>

#include "SOM2D.h"

#include "vertex.h"
#include "data_source.h"
#include "linear_ETA.h"
#include "conffile.h"
#include "Log.h"

#ifdef OPENBLAS
#include "cblas.h"
#endif

#ifdef INTELMKL
#include "mkl_blas.h"
#endif

//#include "dlib.h"


using namespace whiteice;
using namespace whiteice::math;



namespace whiteice
{

  SOM2D::SOM2D(unsigned int w, unsigned int h, unsigned int dim) :
    som_width(w) , som_height(h) , som_dimension(dim),
    som_widthf(w), som_heightf(h), som_dimensionf(dim)
  {
    somtable = NULL;

    som_widthf = som_width;
    som_heightf = som_height;
    som_dimensionf = som_dimension;
    
    // posix_memalign(&somtable, sizeof(void*)*2,
    // som_dimension*som_height*som_width*sizeof(whiteice::math::blas_real<float>));

#ifdef CUBLAS
    auto cudaErr = cudaMallocManaged
      (&somtable, som_dimension*som_height*som_width*sizeof(whiteice::math::blas_real<float>));
    
    if(cudaErr != cudaSuccess || somtable == NULL){
      whiteice::logging.error("SOM2D ctor: cudaMallocManaged() failed.");
      throw CUDAException("CUBLAS memory allocation failure.");
    }
    
#else
    somtable = (whiteice::math::blas_real<float>*)
      malloc(som_dimension*som_height*som_width*sizeof(whiteice::math::blas_real<float>));
    
    if(somtable == 0) throw std::bad_alloc();
#endif
    
    show_visualization = false;
    show_eta = true;
    graphics_on = false;
    
    umatrix = 0;
    
    randomize();
  }
  
  
  SOM2D::SOM2D(const SOM2D& som)
  {
    this->somtable = NULL;
    
    this->som_width = som.som_width;
    this->som_height = som.som_height;
    this->som_dimension = som.som_dimension;

    this->som_widthf = som.som_widthf;
    this->som_heightf = som.som_heightf;
    this->som_dimensionf = som.som_dimensionf;
    
    // posix_memalign(&somtable, sizeof(void*)*2,
    // som_dimension*som_height*som_width*sizeof(whiteice::math::blas_real<float>));

#if CUBLAS
    auto cudaErr = cudaMallocManaged
      (&somtable, som_dimension*som_height*som_width*sizeof(whiteice::math::blas_real<float>));
    
    if(cudaErr != cudaSuccess || somtable == NULL){
      whiteice::logging.error("SOM2D ctor: cudaMallocManaged() failed.");
      throw CUDAException("CUBLAS memory allocation failure.");
    }

    cudaErr = cudaMemcpy
      (somtable, som.somtable,
       som_dimension*som_width*som_height*sizeof(whiteice::math::blas_real<float>),
       cudaMemcpyDeviceToDevice);

    if(cudaErr != cudaSuccess){
      cudaFree(somtable);
      whiteice::logging.error("SOM2D ctor: cudaMemcpy() failed.");
      throw CUDAException("CUBLAS memcopy failed.");
    }

    gpu_sync();
    
#else
    this->somtable = (whiteice::math::blas_real<float>*)
      malloc(som_dimension*som_height*som_width*sizeof(whiteice::math::blas_real<float>));
    
    if(this->somtable == NULL) throw std::bad_alloc();

    memcpy((float*)this->somtable, (float*)som.somtable,
	   som_dimension*som_height*som_width*sizeof(whiteice::math::blas_real<float>));
#endif
    
    this->show_visualization = som.show_visualization;
    this->show_eta = som.show_eta;
    this->graphics_on = som.graphics_on;
    
    this->umatrix = NULL;

#ifdef CUBLAS
    if(som.umatrix){
      auto cudaErr = cudaMallocManaged
	(&umatrix, som_height*som_width*sizeof(whiteice::math::blas_real<float>));
      
      if(cudaErr != cudaSuccess || umatrix == NULL){
	cudaFree(somtable);
	whiteice::logging.error("SOM2D ctor: cudaMallocManaged() failed.");
	throw CUDAException("CUBLAS memory allocation failure.");
      }
      
      cudaErr = cudaMemcpy
	(umatrix, som.umatrix,
	 som_dimension*som_width*som_height*sizeof(whiteice::math::blas_real<float>),
	 cudaMemcpyDeviceToDevice);
      
      if(cudaErr != cudaSuccess){
	cudaFree(umatrix);
	cudaFree(somtable);
	whiteice::logging.error("SOM2D ctor: cudaMemcpy() failed.");
	throw CUDAException("CUBLAS memcopy failed.");
      } 

      gpu_sync();
    }
    
#else
    if(som.umatrix){
      this->umatrix = (whiteice::math::blas_real<float>*)
	malloc(som_width*som_height*sizeof(whiteice::math::blas_real<float>));
      memcpy((float*)this->umatrix, (float*)som.umatrix,
	     som_width*som_height*sizeof(whiteice::math::blas_real<float>));
    }
#endif
    
  }


  
  SOM2D::~SOM2D()
  {
#if CUBLAS
    if(somtable) cudaFree(somtable);
    if(umatrix) cudaFree(umatrix);
#else
    if(somtable) free(somtable);
    if(umatrix) free(umatrix);
#endif
    
    
#if 0    
    close_visualization();
#endif
  }
  
  
  // learns given data
  bool SOM2D::learn(const std::vector < vertex< whiteice::math::blas_real<float> > >& source,
		    bool full) 
  {        
    if(source.size() <= 0) return false;
    if(source[0].size() != som_dimension) return false;
    
    ETA<double>* eta = 0;
    
    if(show_eta){
      try{ eta = new linear_ETA<double>(); }
      catch(std::exception& e){ return false; }
    }
    
#if 0
    if(show_visualization)
      open_visualization();
#endif
    
    const unsigned int MAXSTEPS = 20000;
    const unsigned int CNGSTEPS = 5000*som_height*som_width;
    const unsigned int ETA_DELTA = MAXSTEPS / 100;
    
    learning_rate0 = 0.1f;
    hvariance0 = (som_height*som_width)/2.0f;
    hvariance_t1 = ((float)MAXSTEPS)/(logf(hvariance0));
    learning_rate_t2 = ((float)MAXSTEPS);
    
    float hvariance = hvariance0;
    float learning_rate = learning_rate0;

    std::list< whiteice::math::blas_real<float> > errors;
    whiteice::math::blas_real<float> convergence_ratio = 0.01; // 1% stdev
    bool convergence = false;
    
    if(eta) eta->start(0.0, (double)(MAXSTEPS + CNGSTEPS));

    
    // SELF-ORGANIZING PHASE
    
    for(unsigned int i=0, eta_counter=0;i<MAXSTEPS;i++, eta_counter++){
      if(convergence == true) break;
      
      if(eta){
	eta->update((double)i);

	if(eta_counter >= ETA_DELTA){
	  eta_counter = 0;
	  report_convergence(i, MAXSTEPS+CNGSTEPS, errors, eta, source);
	  
	  
	  // check for convergence
	  if(errors.size() >= 100){
	    whiteice::math::blas_real<float>  m = 0.0f, v = 0.0f;
	    
	    for(const auto& e : errors){
	      m += e;
	    }
	
	    m /= errors.size();
	    
	    for(const auto& e : errors){
	      v += (e - m)*(e - m);
	    }
	  
	    v /= (errors.size() - 1);
	  
	    v = sqrt(abs(v));
	
	    {
	      std::cout << "ERROR CONVERGENCE " << 100.0f*v/m << "% "
			<< "(limit " << 100.0f*convergence_ratio << "%)"
			<< std::endl;
	      std::cout << std::flush;;
	    }

	    
	    if(v/m <= convergence_ratio){
	      convergence = true;
	      break; // stdev is less than c% of mean (5%)
	    }
	  }
	}

      }
      
      
#if 0      
      draw_visualization();
#endif
      
      // FINDS WINNER FOR RANDOMLY CHOSEN DATA
      
      unsigned int dindex = rng.rand() % source.size();            
      unsigned int winner = find_winner(source[dindex].data);
      
      // UPDATES SOM LATTICE
      
      // winner coordinates as floats
      float wx = winner % som_width , wy = (int)(winner / som_width);
      
      // winner index (back) to somtable memory index
      winner *= som_dimension;
      float x = 0.0f, y = 0.0f;
      float h = 0.0f;
      
      
      for(unsigned int index=0;index<som_height*som_width*som_dimension;index += som_dimension){
	// calculates h() function for a given point

	x = (int)((index/som_dimension) % som_width);
	y = (int)((index/som_dimension) / som_width);
	
	h = wraparound_sqdistance(x, wx, y, wy) / (-2.0f * hvariance);
	h = learning_rate * expf(h);

	// std::cout << "h = " << h << std::endl;
	
	if(h > 0.001){
	  // printf("H(%d,%d) = %f\n", (int)x, (int)y, h);

#ifdef CUBLAS

	  float h1 = 1.0f - h;
	  auto s = cublasSscal(cublas_handle, (int)som_dimension,
			       (const float*)&h1, (float*)&(somtable[index]), 1);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("SOM2D<>::learn(): cublasSscal() failed.");
	    throw CUDAException("CUBLAS cublasSscal() call failed.");
	  }

	  s = cublasSaxpy(cublas_handle, som_dimension, (const float*)&h,
			  (const float*)(source[dindex].data), 1, 
			  (float*)&(somtable[index]), 1);

	  if(s != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("SOM2D<>::learn(): cublasSaxpy() failed.");
	    throw CUDAException("CUBLAS cublasSaxpy() call failed.");
	  }

	  gpu_sync();
	  
#else
	  // w -= h*w <=> w = (1-h) * w
	  cblas_sscal(som_dimension, (1 - h), (float*)&(somtable[index]), 1);
	  
	  // w += h*x (
	  cblas_saxpy(som_dimension,  h,
		      (float*)(source[dindex].data),  1,
		      (float*)&(somtable[index]), 1);
	  /*
	  cblas_saxpy(som_dimension,  h,
		      (float*)&(somtable[winner]),  1,
		      (float*)&(somtable[index]), 1);	  
	  */
#endif
	}
	
      }
    
      
      // UPDATES LEARNING PARAMETERS

      // NO UPDATE IN TRAINING PHASE
      //learning_rate = learning_rate0 * expf( i / (-learning_rate_t2));
      //hvariance = hvariance0 * expf(i/(-hvariance_t1)) * expf(i/(-hvariance_t1));

      // std::cout << "hvariance = " << hvariance << std::endl;
    }

    convergence = false;
    errors.clear();
    
    // CONVERGENCE PHASE
    if(full && convergence == false)
      for(unsigned int i=0, eta_counter=0;i<CNGSTEPS;i++, eta_counter++){
	if(convergence == true) break;

	// std::cout << "CONVERGENCE PHASE: " << i << " ETA_DELTA: " << eta_counter << std::endl;
	
	if(eta){
	  eta->update((double)(i+MAXSTEPS));
	  
	  if(eta_counter >= ETA_DELTA){
	    eta_counter = 0;
	    report_convergence(i+MAXSTEPS, MAXSTEPS+CNGSTEPS, errors, eta, source);
	    
	    // check for convergence
	    if(errors.size() >= 100){
	      whiteice::math::blas_real<float>  m = 0.0f, v = 0.0f;
	      
	      for(const auto& e : errors){
		m += e;
	      }
	      
	      m /= errors.size();
	      
	      for(const auto& e : errors){
		v += (e - m)*(e - m);
	      }
	      
	      v /= (errors.size() - 1);
	      
	      v = sqrt(abs(v));
	      
	      {
		std::cout << "ERROR CONVERGENCE " << 100.0f*v/m << "% "
			  << "(limit " << 100.0f*convergence_ratio << "%)"
			  << std::endl;
		std::cout << std::flush;;
	      }
	      
	      if(v/m <= convergence_ratio){
		convergence = true;
		break; // stdev is less than c% of mean (5%)
	      }
	    }
	  }
	}

#if 0	
	draw_visualization();
#endif
      
	// FINDS WINNER FOR RANDOMLY CHOSEN DATA
	
	unsigned int dindex = rng.rand() % source.size();            
	unsigned int winner = find_winner(source[dindex].data);
	
	// UPDATES SOM LATTICE
	
	// winner coordinates as floats
	float wx = winner % som_width , wy = (int)(winner / som_width);
	
	// winner index (back) to somtable memory index
	winner *= som_dimension;
	
	float x = 0, y = 0; // floating point coordinates of current index
	float h;
	
	
	for(unsigned int index=0;index<som_height*som_width*som_dimension;index += som_dimension){
	  
	  // calculates h() function for a given point
	  
	  h = wraparound_sqdistance(x, wx, y, wy) / (-2.0f * hvariance);
	  h = 0.1f * expf(h);

#ifdef CUBLAS
	  if(h > 0.001){
	    
	    float h1 = 1.0f - h;
	    auto s = cublasSscal(cublas_handle, (int)som_dimension,
				 (const float*)&h1, (float*)&(somtable[index]), 1);
	    
	    if(s != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("SOM2D<>::learn(): cublasSscal() failed.");
	      throw CUDAException("CUBLAS cublasSscal() call failed.");
	    }
	    
	    s = cublasSaxpy(cublas_handle, som_dimension, (const float*)&h,
			    (const float*)(source[dindex].data), 1, 
			    (float*)&(somtable[index]), 1);
	    
	    if(s != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("SOM2D<>::learn(): cublasSaxpy() failed.");
	      throw CUDAException("CUBLAS cublasSaxpy() call failed.");
	    }

	    gpu_sync();
	  }
	  
#else
	  // w = (1-h)*w + h*x => w = w + h*(x-w)
	  
	  if(h > 0.001){
	    // w -= h*w <=> w = (1 - h) * w
	    cblas_sscal(som_dimension, (1 - h), (float*)&(somtable[index]), 1);
	    
	    // w += h*x
	    cblas_saxpy(som_dimension,  h,
			(float*)(source[dindex].data), 1,
			(float*)&(somtable[index]), 1);
	  }
#endif
	  
	  
	  // updates coordinates
	  x += 1.0f;
	  
	  if(x >= som_widthf){
	    x = 0.0f;
	    y += 1.0f;
	  }
	}
	
	
	// UPDATES LEARNING PARAMETERS
	
	hvariance = 0.05 + ((CNGSTEPS - i)/((float)CNGSTEPS))*5;
	
	// std::cout << "hvariance = " << hvariance << std::endl;
      }
    
    
    if(eta){
      report_convergence(CNGSTEPS+MAXSTEPS, MAXSTEPS+CNGSTEPS, errors, eta, source);
      
      delete eta;
    }
    
    
    return true;
  }
    
  
  
  // randomizes som vertex values
  bool SOM2D::randomize() 
  {
    // calculates random values to between [-1,1]
    
    const unsigned int N = som_dimension*som_width*som_height;
    
    for(unsigned int i=0;i<N;i++)
      somtable[i] = rng.normal();
    
    // normalizes lengt of som vectors
    float len;
    
    for(unsigned int i=0;i<N;i += som_dimension){

#ifdef CUBLAS
      auto s = cublasSnrm2(cublas_handle, som_dimension,
			   (const float*)&(somtable[i]), 1, &len);

      if(s != CUBLAS_STATUS_SUCCESS){
	whiteice::logging.error("SOM2D::randomize(): cublasSnrm2() failed.");
	throw CUDAException("CUBLAS cublasSnrm2() failed.");
      }

      if(len != 0.0f){
	len = 1.0f/whiteice::math::sqrt(len);

	s = cublasSscal(cublas_handle, som_dimension, (const float*)&len,
			(float*)&(somtable[i]), 1);
	
	if(s != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("SOM2D::randomize(): cublasSscal() failed.");
	  throw CUDAException("CUBLAS cublasSscal() failed.");
	}
      }

      gpu_sync();
      
#else
      len = cblas_snrm2(som_dimension, (float*)&(somtable[i]), 1);
      if(len != 0.0f){
	len = 1.0f / whiteice::math::sqrt(len);
	cblas_sscal(som_dimension, len, (float*)&(somtable[i]), 1);
      }
#endif 
    }
    
    return true;
  }


  // randomsizes SOM vertex values to span two highest variance PCA eigenvectors
  bool SOM2D::randomize
  (const std::vector< whiteice::math::vertex< whiteice::math::blas_real<float> > >& data) 
  {
    if(som_dimension <= 2) return randomize();
    
    matrix<> PCA;
    vertex<> m;
    blas_real<float> v1, v2;
    
    if(pca(data, 2, PCA, m, v1, v2, true) == false){
      printf("PCA calculation failed.");
      randomize();
      return false;
    }

    printf("PCA computed: %d x %d\n", PCA.ysize(), PCA.xsize());

    // initial SOM vector space spans span(v1,v2) where
    // v1 and v2 are the largest eigenvectors

    vertex<> alpha;
    alpha.resize(2);

    if(PCA.pseudoinverse() == false)
      return false; // bad singular eigenvalues for data???
    
    auto& invPCA = PCA;

    for(int j=0;j<(int)som_height;j++){
      float fy = ((float)j/(float)(som_height-1));
      alpha[1] = 2.0f*fy - 1.0f;
	
      for(int i=0;i<(int)som_width;i++){
	float fx = ((float)i/(float)(som_width - 1));
	alpha[0] = 2.0f*fx - 1.0f;

	vertex<> som_vector = invPCA*alpha + m;

	const unsigned int index = (j*som_width + i)*som_dimension;

#ifdef CUBLAS
	auto err = cudaMemcpy((float*)&(somtable[index]), (const float*)&(som_vector[0]),
			      som_vector.size()*sizeof(whiteice::math::blas_real<float>),
			      cudaMemcpyDeviceToDevice);

	if(err != cudaSuccess){
	  whiteice::logging.error("SOM2d::randomize(): cudaMemcpy() failed.");
	  throw CUDAException("CUBLAS cudaMemcpy() failed.");
	}
	
#else
	memcpy((float*)&(somtable[index]), (float*)&(som_vector[0]),
	       som_vector.size()*sizeof(whiteice::math::blas_real<float>));
#endif
      }
    }

#ifdef CUBLAS
    gpu_sync();
#endif

    return true;
  }

  
  bool SOM2D::initializeHierarchical(const SOM2D& som_prev)
  {
    if(som_prev.width() > this->width() || som_prev.height() > this->height())
      return false; // SOM map is not smaller than current map

    float px = 0.0f, py = 0.0f;
    float dx = ((float)this->width())/((float)som_prev.width());
    float dy = ((float)this->height())/((float)som_prev.height());
    
    
    for(int j=0.0f;j<(int)som_height;j++, py += dy){
      px = 0.0f;
      int iy = (int)py;
      for(int i=0.0f;i<(int)som_width;i++, px += dx){
	int ix = (int)px;

	whiteice::math::vertex< whiteice::math::blas_real<float> > v = som_prev(ix, iy);
	whiteice::math::vertex< whiteice::math::blas_real<float> > e = v;
	whiteice::math::blas_real<float> s = 0.01f;
	rng.normal(e);
	v = v + s*e; // add random noise to lower dimensional SOM vectors
	
	this->setVector(i, j, v);
      }
    }

    return true;
  }
  

  // calculates average error by using the best match vector for given dataset's vectors
  whiteice::math::blas_real<float> SOM2D::getError(const std::vector< whiteice::math::vertex< whiteice::math::blas_real<float> > >& data) const 
  {
    whiteice::math::blas_real<float> error = 0.0f;

    // const unsigned int MINIBATCHSIZE = 500;
    
#pragma omp parallel shared(error)
    {
      whiteice::math::blas_real<float> err = 0.0f;

      //for(unsigned int i=0;i<MINIBATCHSIZE;i++){
#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<data.size();i++){
	//const unsigned int index = rng.rand() % data.size();
	const unsigned int index = i;
	const unsigned int somindex = find_winner_sub(data[index].data);
	//std::cout << "somindex = " << somindex << "/" << som_width*som_height << std::endl;
	auto delta = (*this)(somindex) - data[index];
	
	err += delta.norm();
      }
      
#pragma omp critical (mvsdkqwoqarit)
      {
	error += err;
      }
    }
    
    //error /= MINIBATCHSIZE;
    error /= data.size();

    return error;
  }


  whiteice::math::blas_real<float> SOM2D::Uvalue() const 
  {
    // calculates average distance between neighbourhood vertexes

    whiteice::math::blas_real<float> Uvalue = 0.0f;
    const int N = (int)(som_height*som_width);

#pragma omp parallel shared(Uvalue)    
    {
      whiteice::math::blas_real<float> uval = 0.0f; // per thread

#pragma omp for nowait schedule(auto)      
      for(int index=0;index<N;index++){
	int i = index % som_width;
	int j = index / som_width;
	
	auto v = (*this)(i,j);
	whiteice::math::blas_real<float> err = 0.0f;
	
	{
	  auto delta = ((*this)(i-1,j) - v).norm();
	  err += delta;
	}
	
	{
	  auto delta = ((*this)(i+1,j) - v).norm();
	  err += delta;
	}
	
	{
	  auto delta = ((*this)(i,j-1) - v).norm();
	  err += delta;
	}
	
	{
	  auto delta = ((*this)(i,j+1) - v).norm();
	  err += delta;
	}

	err /= 4.0f;

	uval += err;
      }

#pragma omp critical (mgeqiopqaq)
      {
	Uvalue += uval;
      }
      
    }
    
    Uvalue /= (som_height*som_width);
    
    return Uvalue;
  }
  
  
  float SOM2D::somdistance(const vertex< whiteice::math::blas_real<float> >& v1,
			   const vertex< whiteice::math::blas_real<float> >& v2) const 
  {
    if(v1.size() != som_dimension || v2.size() != som_dimension)
      return -1.0f; // (error)
       
    
    // don't make two separate activate() calls because
    // that would cause data to be read through twice
    // (with lots of data this causes more cache misses)
    
    const unsigned int N=som_dimension*som_height*som_width;
  
    unsigned int winner[2];
    float tmp, result[2];

#ifdef CUBLAS

    auto s = cublasSdot(cublas_handle, som_dimension,
			(const float*)v1.data, 1, (const float*)somtable, 1,
			(float*)&(result[0]));

    if(s != CUBLAS_STATUS_SUCCESS){
      whiteice::logging.error("SOM2D::somdistance(): cublasSdot() failed.");
      throw CUDAException("CUBLAS cublasSdot() failed.");
    }

    s = cublasSdot(cublas_handle, som_dimension,
		   (const float*)v2.data, 1, (const float*)somtable, 1,
		   (float*)&(result[1]));

    if(s != CUBLAS_STATUS_SUCCESS){
      whiteice::logging.error("SOM2D::somdistance(): cublasSdot() failed.");
      throw CUDAException("CUBLAS cublasSdot() failed.");
    }

    winner[0] = 0; winner[1] = 0;

    for(unsigned int i=som_dimension;i<N;i += som_dimension){
      
      auto s = cublasSdot(cublas_handle, som_dimension,
			  (const float*)v1.data, 1, (const float*)&(somtable[i]), 1,
			  (float*)&(tmp));
      
      if(s != CUBLAS_STATUS_SUCCESS){
	whiteice::logging.error("SOM2D::somdistance(): cublasSdot() failed.");
	throw CUDAException("CUBLAS cublasSdot() failed.");
      }

      if(result[0] < tmp){
	result[0] = tmp;
	winner[0] = i/som_dimension;
      }

      s = cublasSdot(cublas_handle, som_dimension,
		     (const float*)v2.data, 1, (const float*)&(somtable[i]), 1,
		     (float*)&(tmp));
      
      if(s != CUBLAS_STATUS_SUCCESS){
	whiteice::logging.error("SOM2D::somdistance(): cublasSdot() failed.");
	throw CUDAException("CUBLAS cublasSdot() failed.");
      }

      if(result[1] < tmp){
	result[1] = tmp;
	winner[1] = i/som_dimension;
      }
    }

    gpu_sync();
    
#else
    result[0] = cblas_sdot(som_dimension, (const float*)v1.data, 1, (float*)somtable, 1);
    result[1] = cblas_sdot(som_dimension, (const float*)v2.data, 1, (float*)somtable, 1);
    winner[0] = 0; winner[1] = 0;
    
    for(unsigned int i=som_dimension;i<N;i += som_dimension){
      
      tmp = cblas_sdot(som_dimension, (const float*)v1.data, 1, (float*)&(somtable[i]), 1);
      if(result[0] < tmp){
	result[0] = tmp;
	winner[0] = i/som_dimension;
      }
      
      tmp = cblas_sdot(som_dimension, (const float*)v2.data, 1, (float*)&(somtable[i]), 1);
      if(result[1] < tmp){
	result[1] = tmp;
	winner[1] = i/som_dimension;
      }
    }
#endif
    
    // finally converts indexes to coordinates
    
    float y1 = (float)( (winner[0] / som_width) );
    float y2 = (float)( (winner[1] / som_width) );
    float x1 = (float)( (winner[0] % som_width) );
    float x2 = (float)( (winner[1] % som_width) );
    
    return sqrtf(wraparound_sqdistance(x1, x2, y1, y2));
  }
  
  
  // returns winner vertex raw index for a given vertex
  unsigned int SOM2D::activate(const vertex< whiteice::math::blas_real<float> >& v) const 
  {
    unsigned int winner = find_winner(v.data);
    
    return winner;
  }


  // returns interpolated coordinates in SOM 2d map:
  // first find winner vertex (i,j) and calculates its closeness to data at points
  // at (i-1,j), (i+1,j), (i,j-1), (i,j+1). weight of each location is |x^t * |som(i,j)|
  bool SOM2D::smooth_coordinate(const whiteice::math::vertex< whiteice::math::blas_real<float> >& v,
				whiteice::math::vertex< whiteice::math::blas_real<float> > smooth_coordinate)
  {
    unsigned int i = 0, j = 0;
    
    if(index2coordinates(activate(v), i, j) == false)
      return false;

    
    whiteice::math::blas_real<float> w   = getActivity(v, i, j);
    whiteice::math::blas_real<float> wmn = getActivity(v, i-1, j);
    whiteice::math::blas_real<float> wpn = getActivity(v, i+1, j);
    whiteice::math::blas_real<float> wnm = getActivity(v, i, j-1);
    whiteice::math::blas_real<float> wnp = getActivity(v, i, j+1);

    auto x = (w*i + wmn*(i-1) + wpn*(i+1))/(w + wmn + wpn);
    auto y = (w*j + wnm*(j-1) + wnp*(j+1))/(w + wnm + wnp);

    // wrap around coordinates
    if(x < 0) x += som_width;
    if(x >= som_width) x -= som_width;
    if(y < 0) y += som_height;
    if(y >= som_height) y -= som_height;
    
    smooth_coordinate.resize(2);
    smooth_coordinate[0] = x;
    smooth_coordinate[1] = y;

    return true;
  }


  whiteice::math::blas_real<float> SOM2D::getActivity
  (const whiteice::math::vertex< whiteice::math::blas_real<float> >& v,
   unsigned int i, unsigned int j) const 
  {

    if(v.size() != som_dimension) return 0.0f;

    if(i >= som_width){ i -= (i/som_width)*som_width; }
    
    if(j < 0){ j += (-j/som_height)*som_height; }
    if(j >= som_height){ j -= (j/som_height)*som_height; }


    auto d = (*this)(i, j) - v;
    auto nrm = d.norm();
    
    auto p = exp(-nrm*nrm);

    return p;
  }
    
  
  // reads som vertex given lattice coordinate
  vertex< whiteice::math::blas_real<float> > SOM2D::operator()(int i, int j) const 
  {
    if(i < 0){ i += (-i/((int)som_width) + 1)*((int)som_width); }
    if(i >= (int)som_width){ i -= (i/((int)som_width))*((int)som_width); }

    if(j < 0){ j += (-j/((int)som_height) + 1)*((int)som_height); }
    if(j >= (int)som_height){ j -= (j/((int)som_height))*((int)som_height); }
    
    
    vertex< whiteice::math::blas_real<float> > r(som_dimension);

#ifdef CUBLAS

    auto e = cudaMemcpy((float*)r.data,
			(const float*)&(somtable[(i + j*som_width)*som_dimension]),
			som_dimension*sizeof(whiteice::math::blas_real<float>),
			cudaMemcpyDeviceToDevice);

    if(e != cudaSuccess){
      whiteice::logging.error("SOM2D::operator(): cudaMemcpy() failed.");
      throw CUDAException("CUBLAS cudaMemcpy() failed.");
    }

    gpu_sync();

#else
    memcpy((float*)r.data, (float*)&(somtable[(i + j*som_width)*som_dimension]),
	   som_dimension*sizeof(whiteice::math::blas_real<float>));
#endif
    
    return r;
  }


  // writes vertex given lattice coordinate
  bool SOM2D::setVector(int i, int j,
			const whiteice::math::vertex< whiteice::math::blas_real<float> >& v)
  {
    if(v.size() != som_dimension) return false;
    
    if(i < 0){ i += (-i/((int)som_width) + 1)*((int)som_width); }
    if(i >= (int)som_width){ i -= (i/((int)som_width))*((int)som_width); }
	
    if(j < 0){ j += (-j/((int)som_height) + 1)*((int)som_height); }
    if(j >= (int)som_height){ j -= (j/((int)som_height))*((int)som_height); }

    unsigned int index = 0;
    coordinates2index(i, j, index);
    index *= som_dimension;

#ifdef CUBLAS

    auto e = cudaMemcpy((float*)&(somtable[(i + j*som_width)*som_dimension]),
			(const float*)v.data,
			som_dimension*sizeof(whiteice::math::blas_real<float>),
			cudaMemcpyDeviceToDevice);

    if(e != cudaSuccess){
      whiteice::logging.error("SOM2D::operator(): cudaMemcpy() failed.");
      throw CUDAException("CUBLAS cudaMemcpy() failed.");
    }

    gpu_sync();
    
#else
    memcpy(((float*)&somtable[index]), (const float*)v.data,
	   som_dimension*sizeof(whiteice::math::blas_real<float>));
#endif

    return true;
  }
  
  
  // reads som vertex given direct raw index coordinate to a table
  vertex< whiteice::math::blas_real<float> > SOM2D::operator()(unsigned int index) const 
  {
    vertex< whiteice::math::blas_real<float> > v(som_dimension);

#ifdef CUBLAS

    auto e = cudaMemcpy((float*)v.data,
			(const float*)&(somtable[index*som_dimension]),
			som_dimension*sizeof(whiteice::math::blas_real<float>),
			cudaMemcpyDeviceToDevice);

    if(e != cudaSuccess){
      whiteice::logging.error("SOM2D::operator(): cudaMemcpy() failed.");
      throw CUDAException("CUBLAS cudaMemcpy() failed.");
    }

    gpu_sync();
    
#else
    memcpy((float*)v.data, (float*)&(somtable[index*som_dimension]),
	   som_dimension*sizeof(whiteice::math::blas_real<float>));
#endif
    
    return v;
  }
  
  
  bool SOM2D::index2coordinates(const unsigned int index, unsigned int& i, unsigned int& j) const 
  {
    if(index >= som_width*som_height)
      return false;

    
    j = index / som_width;
    i = index % som_width;

    return true;
  }

  
  // handles wrap-around property properly
  bool SOM2D::coordinates2index(const unsigned int i, const unsigned int j,
				unsigned int& index) const 
  {
    unsigned int ii=i;
    unsigned int jj=j;

    if(ii < 0) ii += (-ii/som_width + 1)*som_width;
    if(ii >= som_width) ii -= (ii/som_width)*som_width;

    if(jj < 0) jj += (-jj/som_height + 1)*som_height;
    if(jj >= som_height) jj -= (jj/som_height)*som_height;

    return (ii + jj*som_width);
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  
  // constant field names in SOM configuration files
  const std::string SOM_VERSION_CFGSTR  = "SOM_CONFIG_VERSION";
  const std::string SOM_SIZES_CFGSTR    = "SOM_SIZES";
  const std::string SOM_PARAMS_CFGSTR   = "SOM_PARAMS";
  const std::string SOM_ETA_CFGSTR      = "SOM_USE_ETA";
  const std::string SOM_ROWPROTO_CFGSTR = "SOM_ROW%d";  
  
  
  
  // loads SOM data from file , failure puts
  // SOM in unknown state!
  bool SOM2D::load(const std::string& filename) 
  {
    whiteice::conffile configuration;
    std::vector<int> ints;
    std::vector< float > floats;
    std::vector<std::string> strings;    
    
    if(!configuration.load(filename))
      return false;
    
    ints.clear();
    configuration.get(SOM_VERSION_CFGSTR, ints);
    
    if(ints.size() != 1) return false;
    
    // recognizes version 0.101 (= 101)
    if(ints[0] != 101)
      return false;
    
    ints.clear();
    if(!configuration.get(SOM_SIZES_CFGSTR, ints)) return false;
    if(ints.size() != 3) return false;    
    this->som_width = ints[0];
    this->som_height = ints[1];
    this->som_dimension = ints[2];
    
    this->som_widthf = som_width;
    this->som_heightf = som_height;
    this->som_dimensionf = som_dimension;
    
    {
      float* tmp = (float*)realloc((float*)somtable, sizeof(whiteice::math::blas_real<float>)*som_width*som_height*som_dimension);
				   
				   
      if(tmp == 0) return false;
      else somtable = (whiteice::math::blas_real<float>*)tmp;
    }
    
    floats.clear();
    if(!configuration.get(SOM_PARAMS_CFGSTR, floats)) return false;
    
    ints.clear();
    if(!configuration.get(SOM_ETA_CFGSTR, ints)) return false;
    if(ints.size() != 1) return false;
  
    this->show_eta = (bool)(ints[0]);
    
    char *buf = 0;
    
    try{
      // now starts loading actual som data
            
      buf = new char[50];
      
      for(unsigned int i=0;i<som_width*som_height;i++){
	sprintf(buf,SOM_ROWPROTO_CFGSTR.c_str(), i);
	floats.clear();
	
	if(!configuration.get(buf, floats)) return false;
	if(floats.size() != som_dimension) return false;
	
	for(unsigned int j=0;j<som_dimension;j++)
	  somtable[i*som_dimension + j] = floats[j];
      }

#ifdef CUBLAS
      gpu_sync();
#endif
      
      delete[] buf;
    }
    catch(std::exception& e){
      // probably bad_alloc...
      if(buf) delete[] buf;
      return false;
    }  
    
  return true;
  }
  
  
  // saves SOM data to file
  bool SOM2D::save(const std::string& filename) const 
  {  
    whiteice::conffile configuration;
    std::vector<int> ints;
    std::vector< float > floats;
    std::vector<std::string> strings;
    
    ints.clear();
    ints.push_back(101); // 1000 = 1.000 etc. 101 = 0.101
    if(!configuration.set(SOM_VERSION_CFGSTR, ints)) return false;
    
    ints.clear();
    ints.push_back(som_width);
    ints.push_back(som_height);
    ints.push_back(som_dimension);
    if(!configuration.set(SOM_SIZES_CFGSTR, ints)) return false;
    
    floats.clear();
    floats.push_back(0.0f); // dummy value
    if(!configuration.set(SOM_PARAMS_CFGSTR, floats)) return false;
    
    ints.clear();
    ints.push_back((int)show_eta);
    if(!configuration.set(SOM_ETA_CFGSTR, ints)) return false;

    char *buf = 0;
    
    try{
      // now starts saving actual som data
      // number of som rows etc. can be calculated from
      // SOM_SIZES
      
      buf = new char[50];
      
      floats.resize(som_dimension);
      
      for(unsigned int i=0;i<som_width*som_height;i++){      
	sprintf(buf,SOM_ROWPROTO_CFGSTR.c_str(), i);		
	
	// accuracy loss maybe problem here !!
	// (should add double and long double support
	//  to ConfFile + generic printable interface support)
	for(unsigned int j=0;j<som_dimension;j++)
	  floats[j] = (float)somtable[i*som_dimension + j].c[0];
	
	if(!configuration.set(buf, floats)){
	  delete[] buf;
	  return false;
	}
      }
      
      delete[] buf;
    }
    catch(std::exception& e){
      // probably bad_alloc...
      if(buf) delete[] buf;
      return false;
    }
    
    // saves configuration file and returns results
    return configuration.save(filename);
  }
  

  bool SOM2D::show(bool on) 
  {
#if 0
    if(graphics_on && on == false){
      close_visualization();
    }
    if(graphics_on == false && on == true){
      open_visualization();
      draw_visualization();
    }
#endif
    
    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  
  
  unsigned int SOM2D::find_winner(const whiteice::math::blas_real<float>* vmemory)
    const 
  {
    // calculates inner products and finds the biggest one
    const unsigned int N = som_dimension*som_width*som_height;
    
    unsigned int winner = 0;
    float tmp, result = 0;

#ifdef CUBLAS
    auto s = cublasSdot(cublas_handle, som_dimension,
			(const float*)vmemory, 1, (const float*)&(somtable[0]), 1,
			(float*)&result);
    
    if(s != CUBLAS_STATUS_SUCCESS){
      whiteice::logging.error("SOM2D::find_winner(): cublasSdot() failed.");
      throw CUDAException("CUBLAS cublasSdot() failed.");
    }
    
    for(unsigned int i=som_dimension;i<N;i+= som_dimension){
      
      auto s = cublasSdot(cublas_handle, som_dimension,
			  (const float*)vmemory, 1, (const float*)&(somtable[i]), 1,
			  (float*)&tmp);
      
      if(s != CUBLAS_STATUS_SUCCESS){
	whiteice::logging.error("SOM2D::find_winner(): cublasSdot() failed.");
	throw CUDAException("CUBLAS cublasSdot() failed.");
      }
      
      if(result < tmp){
	result = tmp;
	winner = i/som_dimension;
      }
      
    }

    gpu_sync();

#else    
    result = cblas_sdot(som_dimension, (float*)vmemory, 1, (float*)somtable, 1);    
    
    for(unsigned int i=som_dimension;i<N;i+= som_dimension){
      tmp = cblas_sdot(som_dimension, (float*)vmemory, 1, (float*)&(somtable[i]), 1);
      
      if(result < tmp){
	result = tmp;
	winner = i/som_dimension;
      }
    }

#endif
    
    return winner;
  }


  // finds vertex which has the smallest distance to given vertex: min(i) ||som_i - vmemory||
  unsigned int SOM2D::find_winner_sub(const whiteice::math::blas_real<float>* vmemory)
    const 
  {
    // calculates inner products and finds the biggest one
    const unsigned int N = som_dimension*som_width*som_height;
    
    unsigned int winner = 0;
    float tmp, result = 0;

#ifdef CUBLAS
    printf("WARN: cuBLAS code unlikely to be correct!\n");
    //#error "NOT IMPLEMENTED"
    auto s = cublasSdot(cublas_handle, som_dimension,
			(const float*)vmemory, 1, (const float*)&(somtable[0]), 1,
			(float*)&result);
    
    if(s != CUBLAS_STATUS_SUCCESS){
      whiteice::logging.error("SOM2D::find_winner(): cublasSdot() failed.");
      throw CUDAException("CUBLAS cublasSdot() failed.");
    }
    
    for(unsigned int i=som_dimension;i<N;i+= som_dimension){
      
      auto s = cublasSdot(cublas_handle, som_dimension,
			  (const float*)vmemory, 1, (const float*)&(somtable[i]), 1,
			  (float*)&tmp);
      
      if(s != CUBLAS_STATUS_SUCCESS){
	whiteice::logging.error("SOM2D::find_winner(): cublasSdot() failed.");
	throw CUDAException("CUBLAS cublasSdot() failed.");
      }
      
      if(result < tmp){
	result = tmp;
	winner = i/som_dimension;
      }
      
    }

    gpu_sync();

#else
    whiteice::math::blas_real<float>* delta =
      (whiteice::math::blas_real<float>*)malloc(sizeof(whiteice::math::blas_real<float>)*som_dimension);

    memcpy(delta, &(somtable[0]), sizeof(whiteice::math::blas_real<float>)*som_dimension);

    float alpha = -1.0f;
    cblas_saxpy(som_dimension, alpha, (float*)vmemory, 1, (float*)delta, 1);
    result = cblas_sdot(som_dimension, (float*)delta, 1, (float*)delta, 1);
    
    
    for(unsigned int i=som_dimension;i<N;i+= som_dimension){
      memcpy(delta, &(somtable[i]), sizeof(whiteice::math::blas_real<float>)*som_dimension);
      cblas_saxpy(som_dimension, alpha, (float*)vmemory, 1, (float*)delta, 1);
      tmp = cblas_sdot(som_dimension, (float*)delta, 1, (float*)delta, 1);

      if(result > tmp){
	result = tmp;
	winner = i/som_dimension;
      }
    }

    free(delta);

#endif

    //winner = rand() % (som_width*som_height);
    
    return winner;
  }
  
  
  // calculates squared wrap-a-round distance between two coordinates
  float SOM2D::wraparound_sqdistance(float x1, float x2, float y1, float y2) const 
  {
    // wrap'a'round distance

    if(x1 < 0.0f) x1 += ::floor(-x1/som_widthf + 1)*som_widthf;
    if(x2 < 0.0f) x2 += ::floor(-x2/som_widthf + 1)*som_widthf;
    if(y1 < 0.0f) y1 += ::floor(-y1/som_heightf + 1)*som_heightf;
    if(y2 < 0.0f) y2 += ::floor(-y2/som_heightf + 1)*som_heightf;

    if(x1 >= som_widthf)  x1 -= ::floor(x1/som_widthf)*som_widthf;
    if(x2 >= som_widthf)  x2 -= ::floor(x2/som_widthf)*som_widthf;
    if(y1 >= som_heightf) y1 -= ::floor(y1/som_heightf)*som_heightf;
    if(y2 >= som_heightf) y2 -= ::floor(y2/som_heightf)*som_heightf;

    float dx = x1 - x2;
    float dy = y1 - y2;

    if(dx <= 0.0f) dx = -dx;
    if(dy <= 0.0f) dy = -dy;

    if(dy >= som_heightf/2.0f){
      dy = som_heightf - dy;
    }

    if(dx >= som_widthf/2.0f){
      dx = som_widthf - dx;
    }
    
    return (dx*dx + dy*dy);
  }
  
  
  ////////////////////////////////////////////////////////////////////////////////
  
#if 0
  bool SOM2D::open_visualization() 
  {
    using namespace dlib_global;
    
    if(graphics_on) return true;

    umatrix = (float*)malloc(som_width*som_height*sizeof(whiteice::math::blas_real<float>));
    if(umatrix == 0) return false;
    
    dlib.extensions("verbose off");
    dlib.setName("SOM");
    if(!dlib.open(640, 480)) return false;
    
    dlib.clear();
    dlib.update();    
    
    graphics_on = true;
    
    return true;
  } 
  
  
  bool SOM2D::close_visualization() 
  {
    using namespace dlib_global;
    
    if(!graphics_on) return true;
    
    if(dlib.close() == false)
      return false;

    free(umatrix);
    umatrix = 0;        
    
    graphics_on = false;
    
    return true;
  }
  
  
  bool SOM2D::draw_visualization() 
  {
    using namespace dlib_global;
    
    if(!graphics_on) return false;
    
    float* vmemory = (float*)malloc(sizeof(whiteice::math::blas_real<float>)*som_dimension);
    if(vmemory == 0) return false;
    
    // calculates U-matrix visualization
    // uses square 4 neighbourhood
    
    float umatrix_max = 0.0f;
    
    // calculates simple non-border cases 
    for(unsigned int y=1,index=som_width*som_dimension;y<(som_height - 1);y++){
      index += som_dimension;
      for(unsigned int x=1;x<(som_width - 1);x++){
	
	cblas_scopy(som_dimension, (float*)&(somtable[index]), 1, (float*)vmemory, 1);
	cblas_sscal(som_dimension, -1.0f, (float*)vmemory, 1);
	cblas_saxpy(som_dimension, 0.25f, (float*)&(somtable[index + som_dimension]), 1,
		    (float*)vmemory, 1);
	cblas_saxpy(som_dimension, 0.25f, (float*)&(somtable[index - som_dimension]), 1,
		    (float*)vmemory, 1);
	cblas_saxpy(som_dimension, 0.25f, (float*)&(somtable[index + som_width*som_dimension]), 1,
		    (float*)vmemory, 1);
	cblas_saxpy(som_dimension, 0.25f, (float*)&(somtable[index - som_width*som_dimension]), 1,
		    (float*)vmemory, 1);
	
	umatrix[index/som_dimension] = cblas_snrm2(som_dimension, vmemory, 1);
	if(umatrix[index/som_dimension] > umatrix_max)
	  umatrix_max = umatrix[index/som_dimension];
	
	index += som_dimension;
      }
      index += som_dimension;
    }
    
    
    // calculates border cases.
    
    // y=0 line (no corners)
    for(unsigned int index=som_dimension;index<(som_width - 1)*som_dimension;index += som_dimension){

      cblas_scopy(som_dimension, &(somtable[index]), 1, vmemory, 1);
      cblas_sscal(som_dimension, -1.0f, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index - som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + som_width*som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + (som_height - 1)*som_width*som_dimension]), 1, vmemory, 1);
      
      umatrix[index/som_dimension] = cblas_snrm2(som_dimension, vmemory, 1);
      if(umatrix[index/som_dimension] > umatrix_max)
	umatrix_max = umatrix[index/som_dimension];
    }
    
    // y=(height-1) line (no corners)
    for(unsigned int index=som_dimension;index<(som_width - 1)*som_dimension;index += som_dimension){
      
      cblas_scopy(som_dimension, &(somtable[index + (som_height-1)*som_width*som_dimension]), 1, vmemory, 1);
      cblas_sscal(som_dimension, -1.0f, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f,
		  &(somtable[index + som_dimension + som_width*(som_height - 1)*som_dimension]), 1,
		  vmemory, 1);
      
      cblas_saxpy(som_dimension, 0.25f,
		  &(somtable[index - som_dimension + som_width*(som_height - 1)*som_dimension]), 1,
		  vmemory, 1);
      
      cblas_saxpy(som_dimension, 0.25f,
		  &(somtable[index - som_width*som_dimension + som_width*(som_height-1)*som_dimension]), 1,
		  vmemory, 1);
      
      cblas_saxpy(som_dimension, 0.25f,
		  &(somtable[index]), 1,
		  vmemory, 1);
      
      umatrix[index/som_dimension + (som_height-1)*som_width] = cblas_snrm2(som_dimension, vmemory, 1);      
      if(umatrix[index/som_dimension + (som_height-1)*som_width] > umatrix_max)
	umatrix_max = umatrix[index/som_dimension + (som_height-1)*som_width];
    }
    
    // x = 0 line (no corners)
    for(unsigned int index=som_width*som_dimension;
	index<som_width*(som_height-1)*som_dimension;
	index += som_width*som_dimension){
      
      cblas_scopy(som_dimension, &(somtable[index]), 1, vmemory, 1);
      cblas_sscal(som_dimension, -1.0f, vmemory, 1);
      
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + (som_width - 1)*som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + som_width*som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index - som_width*som_dimension]), 1, vmemory, 1);
      
      umatrix[index/som_dimension] = cblas_snrm2(som_dimension, vmemory, 1);
      if(umatrix[index/som_dimension] > umatrix_max)
	umatrix_max = umatrix[index/som_dimension];
    }
    
    // x = (width-1) line (no corners)
    for(unsigned int index=(2*som_width - 1)*som_dimension;
	index<som_width*(som_height-1)*som_dimension;
	index += som_width*som_dimension){

      cblas_scopy(som_dimension, &(somtable[index]), 1, vmemory, 1);
      cblas_sscal(som_dimension, -1.0f, vmemory, 1);
      
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index - som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + (1 - som_width)*som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + som_width*som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index - som_width*som_dimension]), 1, vmemory, 1);
      
      umatrix[index/som_dimension] = cblas_snrm2(som_dimension, vmemory, 1);
      if(umatrix[index/som_dimension] > umatrix_max)
	umatrix_max = umatrix[index/som_dimension];
    }
    
    // calculates (0,0)
    cblas_scopy(som_dimension, &(somtable[0]), 1, vmemory, 1);
    cblas_sscal(som_dimension, -1.0f, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f, &(somtable[(som_width - 1)*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f, &(somtable[som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f, &(somtable[som_width*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f, &(somtable[(som_height - 1)*som_width*som_dimension]), 1, vmemory, 1);
      
    umatrix[0] = cblas_snrm2(som_dimension, vmemory, 1);    
    if(umatrix[0] > umatrix_max)
      umatrix_max = umatrix[0];
    
    // calculates (0, width-1)
    cblas_scopy(som_dimension, &(somtable[(som_width - 1)*som_dimension]), 1, vmemory, 1);
    cblas_sscal(som_dimension, -1.0f, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f, &(somtable[0]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f, &(somtable[(som_width - 2)*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f, &(somtable[(2*som_width - 1)*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_width*som_height - 1)*som_dimension]), 1,
		vmemory, 1);
      
    umatrix[som_width - 1] = cblas_snrm2(som_dimension, vmemory, 1);
    if(umatrix[som_width - 1] > umatrix_max)
      umatrix_max = umatrix[som_width - 1];
    
    // calculates (height-1, 0);

    cblas_scopy(som_dimension, &(somtable[(som_height - 1)*som_width*som_dimension]), 1, vmemory, 1);
    cblas_sscal(som_dimension, -1.0f, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_width*(som_height - 1) + 1)*som_dimension]), 1, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_width*som_height - 1)*som_dimension]), 1, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_height - 2)*som_width*som_dimension]), 1, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[0]), 1, vmemory, 1);
      
    umatrix[(som_height-1)*som_width] = cblas_snrm2(som_dimension, vmemory, 1);
    if(umatrix[(som_height - 1)*som_width] > umatrix_max)
      umatrix_max = umatrix[(som_height - 1)*som_width];
    
    // calculates (height-1, width-1)

    cblas_scopy(som_dimension, &(somtable[(som_width*som_height - 1)*som_dimension]), 1, vmemory, 1);
    cblas_sscal(som_dimension, -1.0f, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_width*som_height - 2)*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_height - 1)*som_width*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_width - 1)*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[((som_height - 1)*som_width - 1)*som_dimension]), 1, vmemory, 1);
    
    umatrix[som_height*som_width - 1] = cblas_snrm2(som_dimension, vmemory, 1);
    if(umatrix[som_height*som_width - 1] > umatrix_max)
      umatrix_max = umatrix[som_height*som_width - 1];
    
    free(vmemory);
    
    if(umatrix_max == 0.0f)
      umatrix_max = 1.0f;
    
    // draws the umatrix visualization
    // umatrix values are between 0..1
    
    dlib.clear();
    
    const unsigned int W = dlib.width();
    const unsigned int ph = dlib.height()/som_height;
    const unsigned int pw = dlib.width()/som_width;
    unsigned int color = (((unsigned int)((umatrix[0]/umatrix_max)*0xFF))*(0x010101));
    
    for(unsigned int j=0,index = 0;j<som_height;j++){
      for(unsigned int i=0;i<som_width;i++,index++){
	
	for(unsigned int y=0;y<ph;y++){
	  for(unsigned int x=0;x<pw;x++){
	    dlib[i*pw + j*ph*W + x + y*W] = color;
	  }
	}
	
	
	color = (((unsigned int)((umatrix[index]/umatrix_max)*0xFF))*(0x010101));
      }
    }
    
    
    dlib.update();
    
    return true;
  }
#endif

  
  // size of the som lattice
  unsigned int SOM2D::width() const { return som_width; }
  unsigned int SOM2D::height() const { return som_height; }
  unsigned int SOM2D::dimension() const { return som_dimension; }
  unsigned int SOM2D::size() const { return (som_width*som_height); }
  
  

  
  ////////////////////////////////////////////////////////////////////////////////

  
  void SOM2D::report_convergence
  (const unsigned int ITER,
   const unsigned int MAXITER,
   std::list< whiteice::math::blas_real<float> >& errors,
   ETA<double>* eta,
   const std::vector< whiteice::math::vertex< whiteice::math::blas_real<float> > >& datasource)
  {
    std::cout << "SOM ITER: " << ITER << " / " << MAXITER  << " ";

    auto error = getError(datasource);
    auto uvalue = Uvalue();

    errors.push_back(uvalue);
    while(errors.size() > 100)
      errors.pop_front();

    std::cout << "(average error: " << error << ", average Uvalue: " << uvalue << ") ";
      
    double secs = eta->estimate();
    unsigned int mins  = 0;
    unsigned int hours = 0;
    unsigned int days = 0;
    unsigned int years = 0;
    
    years = (unsigned int)(secs/(365.242199*24.0*3600.0));
    secs -= years*(365.242199*24.0*3600.0);
    
    days  = (unsigned int)(secs/(24.0*3600.0));
    secs -= days*24.0*3600.0;
    
    hours = (unsigned int)(secs/3600.0);
    secs -= hours*3600.0;
    
    mins  = (unsigned int)(secs/60.0);
    secs -= mins*60.0;
    
    std::cout << "ETA: "; 
    
    if(years > 0){
      if(years == 1)
	std::cout << "1 year ";
      else if(years > 1)
	std::cout << years << " years ";
    }
    
    if(days > 0 || years > 0){
      if(days == 1)
	std::cout << "1 day ";
      else if(days > 1)
	std::cout << days << " days ";
    }
    
    if(hours > 0 || days > 0 || years > 0){
      if(hours == 1)
	std::cout << "1 hour ";
      else if(hours > 1)
	std::cout << hours << " hours ";
    }
    
    if(mins > 0 || hours > 0 || days > 0 || years > 0){
      if(mins == 1)
	std::cout << "1 min ";
      else if(mins > 0)
	std::cout << mins << " mins ";
    }
    
    if(mins > 0 || hours > 0 || days >> 0 || years > 0){
      secs = (unsigned int)secs;
    }
    
    
    if(secs > 1)
      std::cout << secs << " secs " << std::endl;
    else
      std::cout << secs << " sec " << std::endl;
  }



  // hierachical training to train tree SOM2D
  bool hierarchicalTraining(SOM2D* som,
			    std::vector<whiteice::math::vertex< whiteice::math::blas_real<float> > >& data)
  {
    if(som == NULL) return false;
    if(som->width() != som->height()) return false; // only handle symmetric plane

    if(som->width() <= 16 || som->height() <= 16){
      return som->learn(data, true);
    }

    int level = (int)::floorf(::log2f(som->width()));

    SOM2D* prev = NULL;

    for(int l=4;l<level;l++){
      int size = (int)::powf(2.0f, l);
      
      SOM2D* current = new SOM2D(size, size, som->dimension());

      if(prev != NULL)
	current->initializeHierarchical(*prev);
      else
	current->randomize();

      if(current->learn(data) == false) return false;


      if(prev) delete prev;
      prev = current;
    }

    if(prev != NULL)
      som->initializeHierarchical(*prev);
    else
      som->randomize();

    if(som->learn(data) == false) return false;

    if(prev) delete prev;

    return true;
  }
    
};
  


