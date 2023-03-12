
#include <math.h>

#include "discretization.h"
#include "KMeans.h"


#include <unistd.h>

#include <vector>
#include <set>
#include <map>



namespace whiteice
{
  
  // discretizes real-valued or discrete data to binary data

  template <typename T>
  bool discretization(const std::vector< math::vertex<T> >& input,
		      const std::vector< math::vertex<T> >& output,
		      std::vector< std::vector<bool> >& inputResults,
		      std::vector< std::vector<bool> >& outputResults,
		      std::vector< math::vertex<T> >& conversion)
  {
    if(input.size() == 0 || output.size() == 0) return false;
    if(input.size() != output.size()) return false;

    if(output[0].size() != 1) return false; // output is one-dimensional line

    inputResults.clear();
    outputResults.clear();

    // discretization:
    // 1. detect discrete variables (N=30 different discrete cases max)
    // 2. calculate clustering for continuous variables (K=min(numdata/20, 100) clusters)
    // 3. discretize continuous variables based on distance to cluster center
    //    (5 discretizations per cluster/distance)
    // 4. binarize discrete (all) variables

    
    // 1.
    
    std::vector< std::set<T> > inputVars, outputVars;

    inputVars.resize(input[0].size());
    outputVars.resize(output[0].size());

    for(unsigned long long i=0;i<input.size();i++){
      for(unsigned int k=0;k<input[i].size();k++)
	inputVars[k].insert(input[i][k]);
    }

    for(unsigned long long i=0;i<output.size();i++){
      for(unsigned int k=0;k<output[i].size();k++)
	outputVars[k].insert(output[i][k]);
    }

    // checks sets which size are 30 or smaller and discretize them

    std::map<int, std::set<T> > idisc; // discretized variables and their set of discrete variables
    std::map<int, int> icont; // continuous variables i => index
    
    std::map<int, std::set<T> > odisc; // discretized variables and their set of discrete variables
    std::map<int, int> ocont; // continuous variables i, => index

    for(unsigned int i=0;i<inputVars.size();i++){
      if(inputVars[i].size() <= 30){
	idisc.insert(std::pair<int, std::set<T> >(i, inputVars[i]));
      }
      else icont.insert(std::pair<int,int>(i,icont.size()));
    }

    for(unsigned int i=0;i<outputVars.size();i++){
      if(outputVars[i].size() <= 30){
	odisc.insert(std::pair<int, std::set<T> >(i, outputVars[i]));
      }
      else ocont.insert(std::pair<int,int>(i,ocont.size()));
    }


    //////////////////////////////////////////////////////////////////////
    // INPUT DATA BINARY DISCRETIZATION

    std::vector< std::vector<bool> > icb_data; // input continuous binarized data (discretized)

    // binary discretizes input continuous variables
    if(icont.size() > 0){

      math::vertex<T> v;
      std::vector< math::vertex<T> > cdata;

      v.resize(icont.size());

      for(unsigned long long i=0;i<input.size();i++){
	for(unsigned int k=0;k<v.size();k++){
	  v[k] = input[i][icont.find(k)->second];
	}

	cdata.push_back(v);
      }


      // calculates KMeans clustering of input data with
      // K=min(numdata/20, 100) clusters clusters
      
      const unsigned int K = whiteice::math::pow(1.75f, (float)icont.size())*((1 + cdata.size()/20) < 100 ? (1 + cdata.size()/20) : 100); // was: 100

      whiteice::KMeans<T> kmeans;

      kmeans.startTrain(K, cdata);

      while(kmeans.isRunning()) sleep(1);

      kmeans.stopTrain();

      // calculate st.dev of distances to cluster means

      std::vector<T> distances;

      for(unsigned long long i=0;i<cdata.size();i++){
	T d = (cdata[i] - kmeans[ kmeans.getClusterIndex(cdata[i]) ]).norm();
	distances.push_back(d);
      }

      T distance_mean  = T(0.0f);
      T distance_stdev = T(0.0f);

      for(const auto& d : distances){
	distance_mean += d;
	distance_stdev += d*d;
      }

      distance_mean /= T(distances.size()); // E[x]
      distance_stdev /= T(distances.size()); // E[x^2]

      distance_stdev = distance_stdev - distance_mean*distance_mean; // Var[x] = E[x^2] - E[x]^2
      
      distance_stdev = math::sqrt(distance_stdev);

      const unsigned int L = 5;

      // now we have st.dev., divide st.dev. by 2.5 to find discretization unit distance
      T unit_distance = distance_stdev / T(L / 2.0f);

      for(unsigned long long i=0;i<cdata.size();i++){
	const unsigned int index = kmeans.getClusterIndex(cdata[i]); // cluster*5
	unsigned int k = 0;
	whiteice::math::convert(k, distances[i]/unit_distance); // 5 distances per cluster

	if(k >= L) k = L-1;

	std::vector<bool> binarized;
	binarized.resize(kmeans.size()*L);

	for(unsigned int i=0;i<binarized.size();i++){
	  binarized[i] = false;
	}

	binarized[index*L + k] = true;

	icb_data.push_back(binarized);
      }
    }


    // binary discretizes input discrete data
    std::vector< std::vector<bool> > idb_data;
    
    {
      unsigned int BINSIZE = 0;

      for(auto& m : idisc){
	BINSIZE += m.second.size();
      }

      std::vector<bool> binarized;
      
      binarized.resize(BINSIZE);

      
      for(unsigned long long i=0;i<input.size();i++){
	
	for(unsigned int i=0;i<binarized.size();i++){
	  binarized[i] = false;
	}

	unsigned int index = 0;

	for(unsigned int k=0;k<input[i].size();k++){
	  auto iter = idisc.find(k);
	  if(iter != idisc.end()){

	    unsigned int index2 = 0;
	    
	    for(auto iter2 = iter->second.begin();iter2!=iter->second.end();iter2++,index2++){
	      if(*iter2 == k) break;
	    }
	    
	    binarized[index + index2] = true;
	    
	    index += iter->second.size();
	  }
	}

	idb_data.push_back(binarized);
      }
      
    }


    // create binarized data vectors for input data

    for(unsigned long long i=0;i<idb_data.size();i++){
      std::vector<bool> bvector;

      unsigned int size = 0;

      if(idb_data.size() > 0) size += idb_data[0].size();
      if(icb_data.size() > 0) size += icb_data[0].size();

      bvector.resize(size);

      unsigned int index = 0;

      for(;index<idb_data[i].size();index++)
	bvector[index] = idb_data[i][index];

      for(;(index-idb_data[0].size())<icb_data[i].size();index++)
	bvector[index] = icb_data[i][index-idb_data[0].size()];

      inputResults.push_back(bvector);
    }
    
    
    //////////////////////////////////////////////////////////////////////
    // OUTPUT DATA BINARY DISCRETIZATION

    std::vector< std::vector<bool> > ocb_data; // output continuous binarized data (discretized)
	
    // binary discretizes output continuous variables
    if(ocont.size() > 0){

      math::vertex<T> v;
      std::vector< math::vertex<T> > cdata;

      v.resize(ocont.size());

      for(unsigned long long i=0;i<output.size();i++){
	for(unsigned int k=0;k<v.size();k++){
	  v[k] = output[i][ocont.find(k)->second];
	}

	cdata.push_back(v);
      }


      // calculates KMeans clustering of input data with
      // K=min(numdata/2, 5) clusters clusters
      
      //const unsigned int K = cdata.size()/2 < 5 ? (cdata.size()/2) : 5;
      const unsigned int K = 1; // only one cluster for a real-line

      whiteice::KMeans<T> kmeans;

      kmeans.startTrain(K, cdata);

      while(kmeans.isRunning()) sleep(1);

      kmeans.stopTrain();

      // calculate st.dev of distances to cluster means

      std::vector<T> distances;

      for(unsigned long long i=0;i<cdata.size();i++){
	T d = (cdata[i] - kmeans[ kmeans.getClusterIndex(cdata[i]) ])[0];
	distances.push_back(d);
      }

      T distance_mean  = T(0.0f);
      T distance_stdev = T(0.0f);

      for(const auto& d : distances){
	distance_mean += d;
	distance_stdev += d*d;
      }

      distance_mean /= T(distances.size()); // E[x]
      distance_stdev /= T(distances.size()); // E[x^2]

      distance_stdev = distance_stdev - distance_mean*distance_mean; // Var[x] = E[x^2] - E[x]^2
      
      distance_stdev = math::sqrt(distance_stdev);


      const int KL = 5;

      // now we have st.dev., divide st.dev. by 2.5 to find discretization unit distance
      T unit_distance = distance_stdev / T(KL/2.0f);

      for(unsigned long long i=0;i<cdata.size();i++){
	const unsigned int index = kmeans.getClusterIndex(cdata[i]); // cluster*5
	int k = 0;
	T dk = T(0.0f);
	whiteice::math::convert(dk, distances[i]/unit_distance); // 5 distances per cluster

	k = (int)dk.c[0];

	if(k >=  KL) k =  KL;
	if(k <= -KL) k = -KL;

	k += KL;

	std::vector<bool> binarized;
	binarized.resize(kmeans.size()*(KL*2+1));

	for(unsigned int i=0;i<binarized.size();i++)
	  binarized[i] = false;

	binarized[index*(KL*2+1) + k] = true;

	ocb_data.push_back(binarized);
      }


      // conversion table! (for continuous data)
      conversion.resize(kmeans.size()*(KL*2+1));
      
      for(unsigned long long i=0;i<kmeans.size();i++){
	for(unsigned int j=0;j<(KL*2+1);j++){
	  conversion[i*(KL*2+1)+j] = kmeans[i];
	}
      }
      
    }


    // binary discretizes output discrete data
    std::vector< std::vector<bool> > odb_data;
    
    {
      unsigned int BINSIZE = 0;

      for(auto& m : odisc){
	BINSIZE += m.second.size();
      }

      std::vector<bool> binarized;
      
      binarized.resize(BINSIZE);

      
      for(unsigned long long i=0;i<output.size();i++){
	
	for(unsigned int i=0;i<binarized.size();i++)
	  binarized[i] = false;

	unsigned int index = 0;

	for(unsigned int k=0;k<output[i].size();k++){
	  auto iter = odisc.find(k);
	  if(iter != odisc.end()){

	    unsigned int index2 = 0;
	    
	    for(auto iter2 = iter->second.begin();iter2!=iter->second.end();iter2++,index2++){
	      if(*iter2 == k) break;
	    }

	    binarized[index + index2] = true;
	    
	    index += iter->second.size();
	  }
	}

	odb_data.push_back(binarized);
      }

      auto index = conversion.size();
      conversion.resize(conversion.size()+BINSIZE);

      // std::map<int, std::set<T> > odisc; // discretized variables and their set of discrete variables
      for(const auto& m : odisc){

	for(const auto& l : m.second){
	  conversion[index] = l;
	    
	  index++;
	}
      }
      
    }


    // create binarized data vectors for input data

    for(unsigned long long i=0;i<odb_data.size();i++){
      std::vector<bool> bvector;

      unsigned int size = 0;

      if(odb_data.size() > 0) size += odb_data[0].size();
      if(ocb_data.size() > 0) size += ocb_data[0].size();

      bvector.resize(size);

      unsigned int index = 0;

      for(;index<ocb_data[i].size();index++)
	bvector[index] = ocb_data[i][index];

      
      for(;(index-ocb_data[0].size())<odb_data[i].size();index++)
	bvector[index] = odb_data[i][index-ocb_data[0].size()];



      outputResults.push_back(bvector);
    }

    
    
    return true;
  }
  
  

  template bool discretization< math::blas_real<float> >
  (const std::vector< math::vertex< math::blas_real<float> > >& input,
   const std::vector< math::vertex< math::blas_real<float> > >& output,
   std::vector< std::vector<bool> >& inputResults,
   std::vector< std::vector<bool> >& outputResults,
   std::vector< math::vertex< math::blas_real<float> > >& conversion);

  template bool discretization< math::blas_real<double> >
  (const std::vector< math::vertex< math::blas_real<double> > >& input,
   const std::vector< math::vertex< math::blas_real<double> > >& output,
   std::vector< std::vector<bool> >& inputResults,
   std::vector< std::vector<bool> >& outputResults,
   std::vector< math::vertex< math::blas_real<double> > >& conversion);
  
};
