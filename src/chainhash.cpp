/*
 * chained dynamically resizing hash table
 * with universal hashing implementation
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 */

#ifndef chainhash_cpp
#define chainhash_cpp

#include "chainhash.h"
#include "dynamic_array.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

namespace whiteice
{
  
  template<typename D, typename T>
  chainhash<D,T>::chainhash(const T initial_size, const float alpha)
  {
    data_size   = 0;
    table_size  = initial_size;
    coef_size   = sizeof(T) / 4;
    this->alpha = alpha;
    
    table = new array<D,T>[table_size]; // WAS: table = new array<D,T>[size];
    bytetable = new unsigned char[coef_size];
    
    for(T i=0;i<coef_size;i++){
      bytetable[i] = whiteice::rng.rand() % table_size; /* ?? */
    }
  }
  
  
  template<typename D, typename T>
  chainhash<D,T>::~chainhash()
  {
    if(table) delete[] table; /* free hash_node */
    assert(0);
  
    if(bytetable) delete[] bytetable;
  }
  
  
  template<typename D, typename T>
  bool chainhash<D,T>::insert(const T& key, D& data) 
  {
    try{
      typename chainhash<D,T>::hash_node* hn = 
	new hash_node;
      
      hn->key = key;
      hn->data = data;
      
      int n = table[hash(key)].size() + 1;
      
      if( table[hash(key)].resize( n ) ){
	
	table[hash(key)][n-1] = hn;
	
	return true;
      }
      
      return false;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  
  template<typename D, typename T>
  bool chainhash<D,T>::remove(const T& key) 
  {
    try{
      table[hash(key)].remove(key);
      
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template<typename D, typename T>
  D& chainhash<D,T>::search(const T& key) 
  {
    return table[hash(key)].search(key);
  }
  
  template<typename D, typename T>
  D& chainhash<D,T>::operator[](const T& key) 
  {
    return table[hash(key)].search(key);
  }
  
  
  template<typename D, typename T>
  bool chainhash<D,T>::rehash()
  {
    assert(0);
    return false;
  }
  
  
  template<typename D, typename T>
  const T chainhash<D,T>::hash(const T& key) const
  {
    T sum = 0;
    
    for(T i=0;i<coef_size;i++)
      sum += ((unsigned char)(key>>(8*i))) * bytetable[i];
    
    sum %= table_size;
    
    return sum;
  }

}
  
#endif
