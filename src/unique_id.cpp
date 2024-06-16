

#include <set>
#include <map>
#include <string>
#include <iostream>
#include <stdlib.h>

#include "unique_id.h"
#include "RNG.h"

using namespace std;

namespace whiteice
{

  // allocates numbers
  unsigned int unique_id::get(const std::string& name) const 
  {
    std::map< std::string, id_domain >::iterator i;
      
    i = idlist.find(name);
    
    if(i == idlist.end()) return 0; // not found    

    if(i->second.maxvalue > 0) // limited list (free numbers are listed)
    {
      set<unsigned int>::iterator j;

      j = i->second.ids.begin();
      
      if(j == i->second.ids.end())
	return false; // no numbers available
      
      unsigned int value = *j;  
      i->second.ids.erase(j);

      return value;
    }
    else{ // "unlimited list" (reserved numbers are listed)
      // allocates number - assumes number of numbers used
      // is typically much smaller than available set of numbers

      if(i->second.ids.size() == i->second.ids.max_size())
	return 0; // full
    
      // 4*10^9 - so even when there's 2*10^9 numbers (50%) allocated
      // it takes still only a few guesses before free number is found
      // (todo: support for given length.  times (native) word size)
      
      unsigned int candidate = whiteice::rng.rand(); // srand() must have been called!
      
      while(i->second.ids.find(candidate) !=
	    i->second.ids.end())
      {
	candidate = whiteice::rng.rand();
      }

      i->second.ids.insert(candidate);
      
      return candidate;
    }
  }
  

  // frees id numbers
  // limited list: freeing already free number is possible (returns true)
  //
  bool unique_id::free(const std::string& name, unsigned int id) const 
  {
    std::map< std::string, id_domain>::iterator i;
    if(id == 0) return false;
      
    i = idlist.find(name);    
    if(i == idlist.end()) return false; // domain not found

    if(i->second.maxvalue != 0) // limited set
    {
      if(id > i->second.maxvalue)
	return false;
      
      i->second.ids.insert(id);
      return true;
    }
    else{
      set<unsigned int>::iterator j;
      
      j = i->second.ids.find(id);

      if(j == i->second.ids.end())
	return false;

      i->second.ids.erase(j);
      
      return true;
    }
    
    return false;
  }
  

  // creates id number domains
  bool unique_id::create(const std::string& name, unsigned int number) const 
  {
    if(number == 0)
      return this->create(name);

    if(idlist.find(name) != idlist.end())
      return false; // already in use
    
    std::set<unsigned int>& created = idlist[name].ids; // creates it
    idlist[name].maxvalue = number;
    
    for(unsigned int i=1;i<=number;i++)
      created.insert(i);
    
    
    return true;
  };

  
  
  // unlimited (limited by memory/int size) number of id values
  bool unique_id::create(const std::string& name) const 
  {
    if(idlist.find(name) != idlist.end())
      return false; // already in use
    
    idlist[name].maxvalue = 0; // creates it: zero means infinity

    // ids number set represents now allocated numbers NOT available ones
    
    return true;    
  }
  
  
  // frees id number domains
  bool unique_id::free(const std::string& name) const 
  {
    std::map< std::string,
      id_domain >::iterator i;
    
    i = idlist.find(name);  
    
    if(i == idlist.end())
      return false; // does not exist
    
    
    idlist.erase(i);
    
    return true;
  }


  // returns if domain exists
  bool unique_id::exists(const std::string& name) const
  {
    std::map< std::string, id_domain >::iterator i;
      
    i = idlist.find(name);  
    
    if(i == idlist.end())
      return false; // does not exist

    return true;
  }

};


