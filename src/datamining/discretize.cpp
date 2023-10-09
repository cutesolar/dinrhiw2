
#include "discretize.h"

#include <set>
#include <vector>
#include <chrono>
#include <thread>

#include "dynamic_bitset.h"
#include "FrequentSetsFinder.h"
#include "list_source.h"
#include "KMeans.h"

using namespace std::chrono_literals;


namespace whiteice
{
  //////////////////////////////////////////////////////////////////////

  bool calculate_discretize(const std::vector< std::vector<std::string> >& data,
			    std::vector<struct discretization>& disc)
  {
    if(data.size() == 0) return false;

    std::vector< std::set<std::string> > elems;
    std::vector< std::vector<double> > numbers;
    std::vector<unsigned int> is_numeric;
    

    elems.resize(data[0].size());
    numbers.resize(data[0].size());
    is_numeric.resize(data[0].size());

    for(unsigned int i=0;i<data.size();i++){
      for(unsigned int j=0;j<data[i].size();j++){
	elems[j].insert(data[i][j]);

	char* p = NULL;
	double value = strtod(data[i][j].c_str(), &p);
	if(p != NULL && p != data[i][j].c_str()){
	  is_numeric[j]++;
	  numbers[j].push_back(value);
	}
      }
    }

    disc.resize(data[0].size());

    for(unsigned int i=0;i<disc.size();i++){
      if(elems[i].size() <= 20){
	disc[i].TYPE = 1;
	disc[i].elem.resize(elems[i].size());

	unsigned int index = 0;

	for(const auto& s : elems[i]){
	  disc[i].elem[index] = s;
	  index++;
	}
      }
      else if(is_numeric[i] == data.size()){
	disc[i].TYPE = 0;

	// pairs of items has on average 10.0 rows/freq
	// [not enough of frequent item which is 20.0] 
	const double ROWS = 50.0;  // was: 50
	const double DIV = sqrt(ROWS*data.size());

	double B = data.size()/DIV; // 4000.0;
	//B = math::pow(B, 1.0/2.0); 

	unsigned int BINS = (unsigned int)round(B);
	std::cout << "BINS: " << BINS << std::endl;
	
	if(BINS < 2) BINS = 2;
	else if(BINS > 100) BINS = 100;

	/*
	unsigned int BINS = data.size() / 1000; // was 200, 300, 500 don't work
	
	if(BINS < 2) BINS = 2;
	else if(BINS > 100) BINS = 100;
	*/
	
	disc[i].bins.resize(BINS);

	whiteice::KMeans<double> km;

	std::vector< std::vector<double> > data;

	for(const auto& n : numbers[i]){
	  std::vector<double> nn;
	  nn.push_back(n);
	  data.push_back(nn);
	}

	km.startTrain(BINS, data);

	while(km.isRunning()){
	  std::this_thread::sleep_for(100ms);
	}

	km.stopTrain();

	std::set<double> numset; // ordered list from smallest to largest

	for(unsigned long long k=0;k<km.size();k++){
	  numset.insert(km[k][0]);
	}
	
	//std::cout << "BINS: ";
	unsigned long long index = 0;
	double prev = -INFINITY;
	for(const auto& n : numset){
	  if(prev > -INFINITY){
	    disc[i].bins[index] = (n+prev)/2.0;
	    index++;
	  }
	  prev = n;
	  
	  //std::cout << n << " ";
	}

	disc[i].bins[index] = INFINITY; // last bin 
	
	//std::cout << std::endl;

	

	/*
	double mean = 0.0;
	double stdev = 0.0;

	for(const auto& s : numbers[i]){
	  mean += s;
	  stdev += s*s;
	}

	mean /= numbers[i].size();
	stdev /= numbers[i].size();
	stdev -= mean*mean;

	double binstart = 6.0*stdev;
	double binwide = binstart/(BINS/2);

	for(unsigned int j=0;j<(BINS/2);j++){
	  disc[i].bins[j] = -binstart/2.0 + binwide*j;
	}
	*/
	
      }
      else{
	disc[i].TYPE = 2;
	// ignore this column  
      }
    }
    
    return (disc.size() > 0);
  }

  
  //////////////////////////////////////////////////////////////////////

  
  // discretizes data and creates one-hot-encoding of discrete value in binary
  bool binarize(const std::vector< std::vector<std::string> >& data,
		const std::vector<struct discretization>& disc,
		std::vector< std::vector<double> >& result)
  {
    if(data.size() == 0) return false;
    if(data[0].size() != disc.size()) return false;

    unsigned int binary_size = 0;

    for(const auto& d : disc){
      if(d.TYPE == 0){
	binary_size += d.bins.size()+1;
      }
      else if(d.TYPE == 1){
	binary_size += d.elem.size();
      }
    }

    if(binary_size == 0) return false;

    
    for(unsigned int i=0;i<data.size();i++){
      std::vector<double> v;
      v.resize(binary_size);

      for(auto& vi : v)
	vi = 0.0;

      unsigned int index = 0;
      
      for(unsigned j=0;j<data[i].size();j++){
	if(disc[j].TYPE == 0){
	  char* p = NULL;
	  double value = strtod(data[i][j].c_str(), &p);

	  unsigned int counter = 0;
	  for(counter = 0;counter<disc[j].bins.size();counter++){
	    if(value < disc[j].bins[counter]) break;
	  }

	  v[index + counter] = 1.0;
	  index += disc[j].bins.size()+1;
	}
	else if(disc[j].TYPE == 1){

	  unsigned int counter = 0;
	  for(counter=0;counter<disc[j].elem.size();counter++)
	    if(disc[j].elem[counter] == data[i][j])
	      break;

	  if(counter < disc[j].elem.size()){
	    v[index+counter] = 1.0;
	  }

	  index += disc[j].elem.size();
	}
	
      }
      
      result.push_back(v);
    }


    return (result.size() > 0); 
  }

  //////////////////////////////////////////////////////////////////////

  using namespace whiteice;

  
  // creates dataset with frequent sets added as extra-variables
  bool enrich_data(const std::vector< std::vector<double> >& data,
		   std::set<whiteice::dynamic_bitset>& f, // frequent sets
		   std::vector< std::vector<double> >& result,
		   double freq_limit)
  {
    if(data.size() == 0) return false;
    if(freq_limit < 0.0 || freq_limit >= 1.0) return false;

    if(freq_limit == 0.0){
      freq_limit = 50.0/data.size(); // 50 cases for each variable minimum
    }

    std::vector<dynamic_bitset> fset;

    // calculates frequent itemsets
    {
      std::vector<dynamic_bitset> dbdata;
      
      for(const auto& d : data){
	dynamic_bitset x;
	x.resize(data[0].size());
	x.reset();
	
	for(unsigned int i=0;i<d.size();i++){
	  if(d[i] != 0.0) x.set(i, true);
	  else x.set(i, false);
	}
	
	dbdata.push_back(x);
      }
      
      list_source<dynamic_bitset>* source = new list_source<dynamic_bitset>(dbdata);
      
      
      whiteice::datamining::FrequentSetsFinder fsfinder(*source, fset, freq_limit);
      
      fsfinder.find();

      delete source;
    }

    // extend datasets to all subsets of frequent sets
    // std::set<dynamic_bitset> f;
    
    
    {
      for(unsigned int i=0;i<fset.size();i++){

	const unsigned int BITS = fset[i].count();
	
	dynamic_bitset b;
	b.resize(BITS);
	b.reset();

	b.inc();

	while(b.none() == false){

	  dynamic_bitset c;
	  c.resize(fset[i].size());
	  c.reset();

	  unsigned int k = 0;

	  for(unsigned int l=0;l<fset[i].size();l++){
	    if(fset[i][l]){

	      if(b[k]) c.set(l, true);
	      
	      k++;
	    }
	  }

	  f.insert(c);

	  b.inc();
	}
	
      }
    }

    // generates all frequent itemsets dataset
    {
      for(unsigned int j=0;j<data.size();j++){
	dynamic_bitset value;
	value.resize(f.size()*2);
	value.reset();

	unsigned int index = 0;

	for(const auto& b : f){

	  bool fdata = true;
	  bool or_data = false;

	  for(unsigned int i=0;i<b.size();i++){
	    if(b[i] && data[j][i] == 0.0){ fdata = false; }
	    if(b[i] && data[j][i] != 0.0){ or_data = true; }
	  }

	  if(fdata) value.set(2*index, true);
	  else value.set(2*index, false);
	  
	  if(or_data) value.set(2*index + 1, true);
	  else value.set(2*index + 1, false);

	  /*
	  if(fdata) value.set(4*index + 2, false);
	  else value.set(4*index + 2, true);
	  
	  if(or_data) value.set(4*index + 3, false);
	  else value.set(4*index + 3, true);
	  */
	  
	  index++;
	}

	// now we have one frequent item

	std::vector<double> r;
	r.resize(value.size());

	for(unsigned int i=0;i<r.size();i++){
	  if(value[i]) r[i] = 1.0;
	  else r[i] = 0.0;
	}

	result.push_back(r);
      }
    }
    
    if(result.size() == 0) return false;
    if(result[0].size() == 0) return false;

    return true;
  }


  // creates dataset with frequent sets added as extra-variables
  bool enrich_data_again(const std::vector< std::vector<double> >& data,
			 const std::set<whiteice::dynamic_bitset>& f, // frequent sets
			 std::vector< std::vector<double> >& result)
  {

    // generates all frequent itemsets dataset
    {
      for(unsigned int j=0;j<data.size();j++){
	dynamic_bitset value;
	value.resize(f.size()*2);
	value.reset();

	unsigned int index = 0;

	for(const auto& b : f){

	  bool fdata = true;
	  bool or_data = false;

	  for(unsigned int i=0;i<b.size();i++){
	    if(b[i] && data[j][i] == 0.0){ fdata = false; }
	    if(b[i] && data[j][i] != 0.0){ or_data = true; }
	  }

	  if(fdata) value.set(2*index, true);
	  else value.set(2*index, false);
	  
	  if(or_data) value.set(2*index + 1, true);
	  else value.set(2*index + 1, false);

	  /*
	  if(fdata) value.set(4*index + 2, false);
	  else value.set(4*index + 2, true);
	  
	  if(or_data) value.set(4*index + 3, false);
	  else value.set(4*index + 3, true);
	  */

	  index++;
	}

	// now we have one frequent item

	std::vector<double> r;
	r.resize(value.size());

	for(unsigned int i=0;i<r.size();i++){
	  if(value[i]) r[i] = 1.0;
	  else r[i] = 0.0;
	}

	result.push_back(r);
      }
    }
    
    if(result.size() == 0) return false;
    if(result[0].size() == 0) return false;

    return true;
  } 
  
};
