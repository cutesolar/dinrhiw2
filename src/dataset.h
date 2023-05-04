/*
 * dataset
 * for learning data
 * has data preprocessing code
 *
 * dataset (should) keeps order of data
 */

#ifndef dataset_h
#define dataset_h

#include <vector>
#include "vertex.h"
#include "dinrhiw_blas.h"
#include "ica.h"

#include <stdexcept>
#include <exception>
#include <string>
#include <map>



namespace whiteice
{
  
  template <typename T = math::blas_real<float> >
    class dataset
    {
    public:
      // data preprocessing methods
      enum data_normalization { 
	dnMeanVarianceNormalization, // zero mean, unit variance
	dnSoftMax, // forces data to small interval
	dnCorrelationRemoval, // PCA
	dnLinearICA
      };
      
      
      // creates dataset with a given number of dimensions
      // data is a set of vectors
      dataset();
      dataset(unsigned int dimension) ;
      dataset(const dataset<T>& d);
      ~dataset() ;

      dataset<T>& operator=(const dataset<T>& d);

      
      bool createCluster(const std::string& name, const unsigned int dimension);

      bool resetCluster(const unsigned int index, const std::string& name, const unsigned int dimension);
      
      bool getClusterNames(std::vector<std::string>& names) const;
      
      unsigned int getCluster(const std::string& name) const;
      
      std::string getName(unsigned int index = 0) const;
      bool setName(const unsigned int index, const std::string& name);
      
      unsigned int getNumberOfClusters() const;
      
      bool removeCluster(std::string& name);
      bool removeCluster(unsigned int index);

      // copies preprocessing and other information to dataset but no data (perfect copy but no data)
      void copyAllButData(const dataset<T>& d);
      
      // adds data examples
      bool add(const math::vertex<T>& input,
	       bool nopreprocess = false) ;
      
      bool add(const std::vector<math::vertex<T> >& inputs,
	       bool nopreprocess = false) ;
      
      bool add(const std::string& input,
	       bool nopreprocess = false) ;
      
      bool add(const std::vector<std::string>& inputs,
	       bool nopreprocess = false) ;

      // adds data to clusters
      bool add(unsigned int index, const math::vertex<T>& input, 
	       bool nopreprocess = false) ;
      bool add(unsigned int index, const std::vector<math::vertex<T> >& inputs, 
	       bool nopreprocess = false) ;

      bool add(unsigned int index, const std::vector<T>& input,
	       bool nopreprocess = false) ;
      
      bool add(unsigned int index, const std::string& input, 
	       bool nopreprocess = false) ;
      bool add(unsigned int index, const std::vector<std::string>& inputs, 
	       bool nopreprocess = false) ;
      
      // creates empty dataset
      bool clear();
      
      // removes all data from dataset but no preprocessing
      // information
      bool clearData(unsigned int index = 0);
      
      // clears data and preprocessing information
      bool clearAll(unsigned int index = 0);

      // reduces data by taking sample or 'samples' samples from
      // each cluster (which must have equal size) and keeps
      // order of samples between different clusters the same
      // so ith element of cluster A match to ith element of cluster B
      bool downsampleAll(unsigned int samples) ;
      
      // removes data elements from dataset which have Infinite or NaN values
      // maintains pairing between cluster A and cluster B, 
      // if i:th element of A is bad, removes also i:th element of B
      bool removeBadData();

      // DEBUG: returns maximum absolute value in cluster index
      T max(unsigned int index) const;
      
      // returns data in cluster "index"
      bool getData(unsigned int index, std::vector< math::vertex<T> >& data) const ;
      
      /* defines dataset<T>::iterator */
      typedef typename std::vector< math::vertex<T> >::iterator iterator;
      typedef typename std::vector< math::vertex<T> >::const_iterator const_iterator;
      
      // iterators for dataset
      iterator begin(unsigned int index = 0) ;
      iterator end(unsigned int index = 0) ;
      const_iterator begin(unsigned int index = 0) const ;
      const_iterator end(unsigned int index = 0) const ;
      
      
      /*
       * saves and loads dataset(s) to/from disk.
       * Datasets are saved in own binary format
       * documented in dataset::load() (dataset.cpp)
       * 
       */
      bool load(const std::string& filename) ;
      bool save(const std::string& filename) const ;
      
      /*
       * exports dataset values as ascii data without preprocessing (specified cluster)
       * if raw = true, do not remove preprocessings from data before saving
       *
       * FIXME: exportAscii() don't work with superresolutional numbers
       */
      bool exportAscii(const std::string& filename,
		       const unsigned int cluster_index = 0,
		       const bool writeHeaders = false,
		       const bool raw = false) const ;

      /*
       * imports space, "," or ";" separated floating point numbers as vectors into cluster 0
       * which will be overwritten. Ignores the first line which may contain headers and
       * reads at most LINES of vertex data or unlimited amount of data (if set to 0).
       *
       * if realData is true then import real data and not complex data 
       * (one number per number instead of 2 numbers per number)
       *
       * NOTE: in general, importAscii() cannot load data written using exportAscii() because
       *       exportAscii() dumps data from a given cluster. 
       *                     However, if there is only a single
       *                     cluster then importAscii() can load data saved by exportAscii()
       * 
       * FIXME: importAscii() don't work with superresolutional numbers 
       */
      bool importAscii(const std::string& filename,
		       const int cluster_index = -1, // -1 means create new cluster
		       const unsigned int LINES=0,
		       const bool realData = false) ;
      
      // accesses data from cluster zero
      const math::vertex<T>& operator[](unsigned int index) const ;
      
      const math::vertex<T>& access(unsigned int cluster, unsigned int data) const ;
      math::vertex<T>& access(unsigned int cluster, unsigned int data);
      const math::vertex<T>& accessName(const std::string& clusterName, unsigned int dataElem) ;
      
      // accesses random element from specified cluster
      const math::vertex<T>& random_access(unsigned int index = 0) const ;
      
      inline unsigned int size(unsigned int index) const { // dataset size
	if(index >= clusters.size()) return 0;

	return clusters[index].data.size();
      }
      
      bool clear(unsigned int index) ; // data set clear  
      bool resize(unsigned int index, unsigned int nsize) ; // reduces size of data
      
      inline unsigned int dimension(unsigned int index) const { // dimension of data vectors
	if(index >= clusters.size()) return 0;

	return clusters[index].data_dimension;
      }
      
      
      bool getPreprocessings(unsigned int cluster,
			     std::vector<data_normalization>& preprocessings) const  ;
      
      // is data normalized with given operation
      bool hasPreprocess(unsigned int cluster, enum data_normalization norm) const {
    	  if(cluster >= clusters.size()) return false;
    	  else return is_normalized(cluster, norm);
      }

      // data preprocessing
      bool preprocess(unsigned int index,
		      enum data_normalization norm = whiteice::dataset<T>::dnMeanVarianceNormalization) ;
      // index = 0
      bool preprocess(enum data_normalization norm = whiteice::dataset<T>::dnMeanVarianceNormalization) ;
      
      // inverse preprocess everything, calculates new preprocessing parameters
      // and preprocesses everything with parameter data from the whole dataset
      // (dataset may grow after preprocessing)
      bool repreprocess(unsigned int index = 0) ;
      
      // converts data with same preprocessing as with dataset vectors
      bool preprocess(unsigned int index,
		      math::vertex<T>& vec) const ;
      
      bool preprocess(unsigned int index,
		      std::vector< math::vertex<T> >& group) const ;
      // index = 0
      
      bool preprocess(math::vertex<T>& vec) const ;
      
      bool preprocess(std::vector< math::vertex<T> >& group) const ;
  
      // inverse preprocess given data vector
      
      bool invpreprocess(unsigned int index,
			 math::vertex<T>& vec) const ;
      
      bool invpreprocess(unsigned int index,
			 std::vector< math::vertex<T> >& group) const ;
      
      // inverse preprocess mean m and covariance matrix COV
      bool invpreprocess(unsigned int index,
			 math::vertex<T>& m,
			 math::matrix<T>& COV) const ;

      // index = 0
      bool invpreprocess(math::vertex<T>& vec) const ;
      bool invpreprocess(std::vector< math::vertex<T> >& group) const ;
      
      // changes preprocessing to given a given list of preprocessings
      bool convert(unsigned int index = 0) ; // removes all preprocessings from data
      bool convert(unsigned int index,
		   std::vector<enum data_normalization> plist);

      // calculates preprocessings Wx + b linear gradient W if possible
      // (does not support dnSoftMax !!) [returns false if index out of range]

      bool preprocess_grad(unsigned int index, math::matrix<T>& W) const ;
      bool invpreprocess_grad(unsigned int index, math::matrix<T>& W) const ;

      // logs clustering statistics per cluster
      bool diagnostics(const int cluster = 0, const bool verbose = false) const ;
      
    private:
      // is data normalized with given operation?
      bool is_normalized(unsigned int index,
			 enum data_normalization norm) const ;
      
      // preprocessing functions and inverse preprocessing functions
      void mean_variance_removal(unsigned int index,
				 math::vertex<T>& vec) const;
      void inv_mean_variance_removal(unsigned int index,
				     math::vertex<T>& vec) const;
      void inv_mean_variance_removal_cov(unsigned int index,
					 math::matrix<T>& C) const;
      
      void soft_max(unsigned int index, math::vertex<T>& vec) const;
      void inv_soft_max(unsigned int index, math::vertex<T>& vec) const;
      void inv_soft_max_cov(unsigned int index, math::matrix<T>& C) const; // FIXME not implemented
      
      void whiten(unsigned int index, math::vertex<T>& vec) const;
      void inv_whiten(unsigned int index, math::vertex<T>& vec) const;
      void inv_whiten_cov(unsigned int index, math::matrix<T>& C) const;

      void ica(unsigned int index, math::vertex<T>& vec) const;
      void inv_ica(unsigned int index, math::vertex<T>& vec) const;
      void inv_ica_cov(unsigned int index, math::matrix<T>& C) const;      
      
      
      ////////////////////////////////////////////////////////////
      // individually labelled datasets classes
      
      struct cluster
      {
	std::string cname; // data class name;
	unsigned int cindex; // cluster index
	
	unsigned int data_dimension; // data dimension
	std::vector< math::vertex<T> > data; // vectors
	
	std::vector<enum data_normalization> preprocessings; // done to data
	
	// mean & variance removal parameters
	math::vertex<T> mean;
	math::vertex<T> variance;
	
	T softmax_parameter;
	
	math::matrix<T> Rxx; // correlation matrix
	math::matrix<T> Wxx, invWxx; // whitening matrix (calculated from R)
	math::matrix<T> ICA; // ICA solution;
	math::matrix<T> invICA;
      };
      
      
      std::vector<cluster> clusters;
      std::map<std::string, unsigned int> namemapping;
      
      
      
      // fileformat tag
      static const char* FILEID_STRING;
      
    };
  
  
  //////////////////////////////////////////////////////////////////////
  
  extern template class dataset< whiteice::math::blas_real<float> >;
  extern template class dataset< whiteice::math::blas_real<double> >;
  extern template class dataset< whiteice::math::blas_complex<float> >;
  extern template class dataset< whiteice::math::blas_complex<double> >;
  
  extern template class dataset< whiteice::math::superresolution<
				   whiteice::math::blas_real<float>,
				   whiteice::math::modular<unsigned int> > >;
  extern template class dataset< whiteice::math::superresolution<
				   whiteice::math::blas_real<double>,
				   whiteice::math::modular<unsigned int> > >;
  
  extern template class dataset< whiteice::math::superresolution<
				   whiteice::math::blas_complex<float>,
				   whiteice::math::modular<unsigned int> > >;
  extern template class dataset< whiteice::math::superresolution<
				   whiteice::math::blas_complex<double>,
				   whiteice::math::modular<unsigned int> > >;
    
  //extern template class dataset< float >;
  //extern template class dataset< double >;
  
}




#endif

