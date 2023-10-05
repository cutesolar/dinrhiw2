

#include <vector>
#include <stdexcept>

#include "dynamic_bitset.h"
#include "data_source.h"
#include "fast_radixv.h"

#include "FrequentSetsFinder.h"
#include "timed_boolean.h"

#include "fpgrowth.h"


namespace whiteice
{
  namespace datamining
  {
    
    
    FrequentSetsFinder::
    FrequentSetsFinder(const whiteice::data_source<whiteice::dynamic_bitset>& _data,
		       std::vector< whiteice::dynamic_bitset >& _freqset,
		       float _freq_limit) :
      data(_data), freqset(_freqset), freq_limit(_freq_limit)
    {
      if(freq_limit < 0 || freq_limit > 1)
      {
	std::string s;
	s = "FrequentSetsFinder Ctor: ";
	s = s + "freq_limit and conf_limit must be within [0,1] interval";
	
	throw std::invalid_argument(s);
      }
      
      // current state of analysis/search
      level = 0; elem = 0;
      
      
      // finished is needed to tell that search was finished (finished()-call)
      // after clean() has removed internal data structures
      search_finished = false;
    }
    
    
    FrequentSetsFinder::~FrequentSetsFinder(){ }
    
    
    
    // finds (more) associative rules that can be found in TIME_LIMIT seconds
    // if TIME_LIMIT <= 0 amount of time used is unlimited.
    // returns true if any new rule could be found or false if no new rules
    // were added. Note that returning false may complite the search
    // (finished() returns true).
    bool FrequentSetsFinder::find(float TIME_LIMIT)
    {
      if(data.size() < 0) return false;

      std::vector< std::set<long long> > fdata;
      std::set< std::set<long long> > freq_sets;

      for(unsigned long long i=0;i<data.size();i++){
	std::set<long long> d;

	for(unsigned long long k=0;k<data[i].size();k++)
	  if(data[i][k]) d.insert(k);

	fdata.push_back(d);
      }

      if(whiteice::frequent_items(fdata, freq_sets, freq_limit) == false)
	return false;

      if(freq_sets.size() == 0) return false;

      for(const auto& f : freq_sets){
	whiteice::dynamic_bitset b;
	b.resize(data[0].size());
	b.reset();

	for(const auto& fi : f){
	  b.set(fi, true);
	}

	freqset.push_back(b);
      }


      return true;

      
#if 0
      if(data.size() < 0) return false;
      
      timed_boolean timeout(TIME_LIMIT, false);
      unsigned int numgroups = 0; // number of good sets found
      
      // converts frequency limit to number of occurances limit
      unsigned int nflimit = (unsigned int)(data.size() * freq_limit);
      
      const unsigned int M = data[0].size(); // number of bits in data
      
      
      // frequent set calculation (algo 2.14)
      // creating candidate set and checking for freq_limit (database pass)
      // has been combined here
      {
	fast_radixv<dynamic_bitset> sorter;
	dynamic_bitset zero, mask, z;
	zero.resize(data[0].size()); zero.reset();
	mask.resize(data[0].size());

	// fset = frequent sets
	// fcset = frequencies of frequent relations in fset
      
	unsigned int l = fset.size();
	      	
	while(timeout == false){
	  
	  if(fset.empty()){
	    // creates F1 set
	    
	    fset.resize(1);
	    fcset.resize(1);
	    
	    dynamic_bitset db; 
	    db.resize(M); db.reset();
	    
	    
	    // puts candidates into set from
	    // the smallest to the biggest
	    db.set(0);	  
	    
	    {
	      unsigned int N = calculate_counts(db);
	      if(N > nflimit){
		fset[0].push_back(db);
		fcset[0].push_back(N);
	      }
	    }
	    
	      
	    for(unsigned int i=1;i<M;i++){
	      db.reset(i-1);
	      db.set(i);
	      
	      unsigned int N = calculate_counts(db);
	      if(N > nflimit){
		fset[0].push_back(db);
		fcset[0].push_back(N);
	      }
	    }
	    
	    l++;
	  }
	  else{
	    if(fset[l-1].empty())
	      break; // all frequent sets have been generated
	    
	    fset.resize(l+1);
	    fcset.resize(l+1);
	    
	    mask.reset(); mask.set(M - 1); // mask
	    // sorts previous set (fast_radix)
	    sorter.sort(fset[l-1], mask, zero);
	    
	    // algo 2.18, previous (candidate set) should be in sorted order.
	    
	    for(unsigned int i=0;i<fset[l-1].size();i++){
	      for(unsigned int j=i+1;j<fset[l-1].size();j++){
		
		
		// check i and j share their l-1:th lowest on bits
		bool ok = true;
		
		{
		  unsigned int m = 0, k = l-1;
		  while(k > 0 && ok){
		    if(fset[l-1][i][m] != fset[l-1][j][m]){
		      ok = false; // ... don't share
		      break;
		    }
		    
		    
		    if(fset[l-1][i][m]) k--;
		    
		    m++;
		    
		    if(m == M && k > 0)
		      ok = false; // ... don't share		    		    
		  }
		}				
		
		if(ok){
		  dynamic_bitset w;
		  
		  z = fset[l-1][i] | fset[l-1][j];
		  
		  // checks if all z's subsets of size 'l' are
		  // in previous set
		  
		  unsigned int counter = 0;
		  
		  for(unsigned int k=0;k<fset[l-1].size();k++){
		    
		    w = z & fset[l-1][k];
		    
		    // ('l' common bits -> subset with size 'l' is in previous set)
		    if(w.count() >= l)
		      counter++;
		    
		  }
		  
		  
		  if(counter != l+1) // didn't find all subsets
		    ok = false;
		  
		  
		  // adds it to next set if set freq is bigger that fr_limit
		  // todo: optimize to calculate frequencies in a single pass
		  if(ok){
		    unsigned int N = calculate_counts(z);
		    if(N > nflimit){
		      fset[l].push_back(z);
		      fcset[l].push_back(N);
		      numgroups++;
		    }
		  }
		  
		}
	      }
	    }
	    
	    l++;
	  }
	  
	}
      }
      
      
      
      // checks if any new relations were found at all
      if(level < fset.size())
	if(numgroups == 0)
	  level = fset.size(); // no new groups -> no data to preprocess

      if(fset.size() >= 2)
	freqset = fset[fset.size()-2];
      else if(fset.size() > 1)
	freqset = fset[fset.size()-1];
      
      //for(unsigned int i=0;i<fset.size();i++){
      //	std::cout << i << " = " << fset[i].size() << std::endl;
      //}
      
      if(internal_isfinished())
	search_finished = true;
	
      
      return (freqset.size() > 0);
#endif
    }
    
    
    bool FrequentSetsFinder::finished() const 
    {
      return search_finished;
    }
    
    
    void FrequentSetsFinder::clean()
    {
      fset.clear();
      fcset.clear();
    }
    
    
    /************************************************************/
    
    
    // calculates number of times given pattern (subset) exists in data
    unsigned int FrequentSetsFinder::calculate_counts(const dynamic_bitset& bset) const 
    {
      unsigned int c = 0;
      unsigned int bsetc = bset.count();

      for(unsigned int i=0;i<data.size();i++)
	if((bset & data[i]).count() == bsetc) c++;
	  
      
      return c;
    }
    
    
    // returns true if last call of find() finished before time limit
    // and managed to find all possible rules
    bool FrequentSetsFinder::internal_isfinished() const 
    {
      // fset generation:
      if(fset.size() == 0) return false;
      if(fset[fset.size() - 1].size() != 0) return false;
      
      return (level+1 >= fset.size());
    }
    
    
  }
}



