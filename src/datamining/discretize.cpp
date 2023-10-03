
#include "discretize.h"

#include <set>


namespace whiteice
{


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

  
};
