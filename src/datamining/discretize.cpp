
#include "discretize.h"

#include <set>

#include "dynamic_bitset.h"
#include "FrequentSetsFinder.h"
#include "list_source.h"



namespace whiteice
{
  //////////////////////////////////////////////////////////////////////

  bool calculate_discretize(const std::vector< std::vector<std::string> >& data,
			    std::vector<struct discretization>& disc)
  {
    if(data.size() == 0) return false;

    std::vector< std::set<std::string> > elems;
    std::vector< std::set<double> > numbers;
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
	  numbers[j].insert(value);
	}
      }
    }

    disc.resize(data[0].size());

    for(unsigned int i=0;i<disc.size();i++){
      if(is_numeric[i] == data.size()){
	disc[i].TYPE = 0;
	disc[i].bins.resize(10);

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
	double binwide = binstart/10.0;

	for(unsigned int j=0;j<10;j++){
	  disc[i].bins[j] = -binstart/2.0 + binwide*j;
	}
	
      }
      else if(elems[i].size() <= 20){
	disc[i].TYPE = 1;
	disc[i].elem.resize(elems[i].size());

	unsigned int index = 0;

	for(const auto& s : elems[i]){
	  disc[i].elem[index] = s;
	  index++;
	}
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
    std::set<dynamic_bitset> f;
    
    
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
	value.resize(f.size());
	value.reset();

	unsigned int index = 0;

	for(const auto& b : f){

	  bool fdata = true;

	  for(unsigned int i=0;i<b.size();i++){
	    if(b[i] && data[j][i] == 0.0){ fdata = false; break; }
	  }

	  if(fdata) value.set(index, true);
	  else value.set(index, false);

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
