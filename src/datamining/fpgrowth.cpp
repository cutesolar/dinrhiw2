/*
 * Own implementation of FP-Growth algorithm
 *
 * Datamines frequent patterns from data.
 *
 */


#include "fpgrowth.h"
#include "fptree.h"
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <math.h>


namespace whiteice
{

  bool frequent_items(const std::vector< std::set<long long> >& data,
		      std::set< std::set<long long> >& freq_sets,
		      double min_support)
  {
    if(data.size() == 0) return false;
    if(min_support < 0.0 || min_support > 1.0) return false;

    if(min_support == 0.0){
      min_support = 50.0 / data.size(); // 50 items default freq items
    }

    
    std::vector<Transaction> transactions;

    for(const auto& di : data){

      Transaction t;

      for(const auto& si : di){
	t.push_back(si);
      }

      transactions.push_back(t);      
    }


    const uint64_t minimum_support_threshold = (uint64_t)round(min_support*data.size());

    const FPTree fptree{ transactions, minimum_support_threshold };

    const std::set<Pattern> patterns = fptree_growth( fptree );

    
    std::multimap<long long, std::set<Item> > sets;

    for(auto& p : patterns){
      sets.insert(std::pair((long long)p.second, p.first));
    }

    
    freq_sets.clear();

    for(auto si = sets.rbegin();si != sets.rend(); si++){
      bool skip = false;

      for(auto pi = freq_sets.begin();pi != freq_sets.end(); pi++){

	// si subset is in pi set (already has entry) => skip
	if(std::includes(pi->begin(), pi->end(), si->second.begin(), si->second.end())){
	  skip = true;
	  break;
	}
	
      }

      if(skip == false){
	freq_sets.insert(si->second);
      }
    }
    

    return (freq_sets.size() > 0);
  }
  

#if 1

  struct fptree_item
  {
    long long item;
    long long count;

    struct fptree_item* parent;

    std::vector<struct fptree_item*> children;

    //////////////////////////////////////////////////////////////////////

    void print(){
      std::cout << "-" << std::endl;
      std::cout << "parent: " << parent << std::endl;
      std::cout << "this: " << this << std::endl;
      if(item >= 0) std::cout << item << " = " << count << std::endl;
      else std::cout << "*" << std::endl;
      std::cout << "-" << std::endl;

      
      for(auto& c : children)
	c->print();
    }

    //////////////////////////////////////////////////////////////////////

    bool add(struct fptree_item* node, struct fptree_item* parent, const std::multimap<long long, long long>& iset, std::multimap<long long, long long>::iterator item){
      if(item == iset.end()) return true;

      // std::cout << "add " << node->item << " " << item->second << std::endl;
									      
									      
      
      if(node->item == (item->second)){
	node->count++;
	item++;

	if(item == iset.end()) return true;

	for(unsigned long long i=0;i<node->children.size();i++){
	  if(node->children[i]->item == (item->second)){
	    return node->add(node->children[i], this, iset, item);
	  }
	}

	fptree_item* newitem = new fptree_item();
	newitem->item = item->second;
	newitem->count = 0;
	newitem->parent = node;
	node->children.push_back(newitem);

	return node->add(node->children[node->children.size()-1], this, iset, item);
	
      }
      else{

	for(unsigned long long i=0;i<node->children.size();i++){
	  if(node->children[i]->item == (item->second)){
	    return node->add(node->children[i], this, iset, item);
	  }
	}
	
	
	fptree_item* newitem = new fptree_item();
	newitem->item = item->second;
	newitem->count = 0;
	newitem->parent = node->parent;
	if(node->parent){
	  node->parent->children.push_back(newitem);
	  return node->add(newitem, node->parent, iset, item);
	}
	else{
	  node->children.push_back(newitem);
	  return node->add(newitem, node, iset, item); 
	}
      }
    }


    void find_sets(struct fptree_item* node, long long item, std::map<std::set<long long>, long long>& s) {
      if(node->item == item){
	std::set<long long> ss;

	// ss.insert(item);

	auto n = node;

	while(n->parent){
	  if(n->parent->item >= 0)
	    ss.insert(n->parent->item);
	  n = n->parent;
	}

	s.insert(std::pair(ss,node->count));
      }
      else{
	for(unsigned long long i=0;i<children.size();i++){
	  node->find_sets(children[i], item, s);
	}
      }
    }
    
  };


#if 0
  bool contains_single_path(fptree* tree){
    if(tree->children.size() == 0) return true;
    else if(tree->children.size() > 1) return false;
    else return contains_single_path(tree->children.front());
  }
  

  bool fptree_growth(fptree_item* tree,
		     
		     std::multimap<long long, std::set<long long> >& fsets)
  {
    if(tree->children.size() == 0) return;
    
    if(contains_single_path(tree)){

      std::multimap<long long, std::set<long long> > single_path_sets;

      while(tree->item < 0 && tree->children.size() > 0){
	tree = tree->children.front();
      }

      if(tree->item < 0){
	fsets = single_path_sets;
	return true;
      }

      while(tree){

	std::set<long long> ss;
	ss.insert(tree->item);
	
	single_path_sets.insert(std::pair(tree->count,  ss));

	for(const auto& s : single_path_sets){
	  std::set<long long> new_ss(s.second);
	  new_ss.insert(tree->item);

	  single_path_sets.insert(std::pair(tree->count, new_ss));
	}

	if(tree->children.size() > 0)
	  tree = tree->childrend.front();
	else
	  tree = nullptr;
	
      }

      fset = single_path_sets;
      
      return true;
    }
    else{


      /*
       * now constructs conditional sets by traveling fp-tree
       */
      
      // item->({items}, count)
      //std::map<long long, std::map<std::set<long long>, long long> > sets;

      for(auto i = itemfreq.rbegin();i!=itemfreq.rend();i++){
	
	// {items} -> count
	std::map<std::set<long long>, long long> s;
	
	tree->find_sets(&root, i->first, s);
	
	std::cout << "for item: " << i->first << std::endl;
	std::cout << "s.size() = " << s.size() << std::endl;
	for(auto& si : s){
	  for(auto& sij : si.first)
	    std::cout << sij << " ";
	  std::cout << " = " << si.second << std::endl;
	}
	
	//sets.insert(std::pair(i->first, s));

	// generates transactions

	std::vector< std::set<long long> > data;

	for(const auto& si : s){
	  for(long long j=0;j<si.second;j++){
	    data.push_back(si.first);
	  }
	}

	std::set< std::set<long long> > freq_sets;

	if(frequent_items(data, freq_sets, min_support) == false)
	  return false;
	
      }
      
      
      
    }
  }
  
#endif
  
  
  /*
   * datamines frequent itemsets using FP-Growth algorithm
   */ 
  bool frequent_items2(const std::vector< std::set<long long> >& data,
		       std::set< std::set<long long> >& freq_sets,
		       double min_support)
  {
    if(data.size() == 0) return false;
    if(min_support < 0.0 || min_support > 1.0) return false;

    if(min_support == 0.0){
      min_support = 50.0 / data.size(); // 50 items default freq items
    }

    fptree_item root;
    root.item = -1;
    root.count = 0;
    root.parent = nullptr;

    long long minfreq = (long long)round(min_support*data.size());
    if(minfreq <= 0) return false;

    // std::cout << "minfreq = " << minfreq << std::endl;

    /*
     * database pass, calculates and orders frequent items
     */

    // item number -> item count
    std::map<long long, long long> itemfreq;

    for(const auto& d : data){
      for(const auto& di : d){
	if(itemfreq.find(di) == itemfreq.end())
	  itemfreq[di] = -1;
	else
	  itemfreq[di]--;
      }
    }

    // item count -> item number
    std::multimap<long long, long long> inv_itemfreq;

    for(const auto& i : itemfreq){

      if(-i.second >= minfreq) // frequent enough item 
	inv_itemfreq.insert(std::pair<long long,long long>(i.second, i.first));
    }

    // removes non-frequent items
    for(auto i = itemfreq.begin();i != itemfreq.end();){

      auto ii = i;
      if(-i->second < minfreq){
	i++;
	itemfreq.erase(ii);
      }
      else i++;
    }

    // prints frequent items 
    if(0){
      for(const auto& i : itemfreq){
	std::cout << i.first << " = " << i.second << std::endl;
      }

      std::cout << "---" << std::endl;
	    
      for(const auto& i : inv_itemfreq){
	std::cout << i.second << " = " << i.first << std::endl;
      }
    }

    /*
     * adds transaction sets to fptree 
     */
    for(const auto& d : data){
      std::multimap<long long, long long> items; // freq -> item number

      for(const auto& di : d){
	auto f = itemfreq.find(di);

	if(f != itemfreq.end()){
	  items.insert(std::pair<long long, long long>(f->second, f->first));
	}
      }


      // now adds given items to tree
      root.add(&root, nullptr, items, items.begin());

      //std::cout << "==========" << std::endl;
      //root.print();
      //std::cout << "==========" << std::endl;
    }


    //std::multimap<long long, std::set<long long> > fsets;

    //if(fptree_growth(root, itemfreq, fsets) == false) return false;
    
#if 1


    /*
     * now constructs conditional sets by traveling fp-tree
     */

    // item->({items}, count)
    std::map<long long, std::map<std::set<long long>, long long> > sets;

    for(auto i = itemfreq.rbegin();i!=itemfreq.rend();i++){

      // {items} -> count
      std::map<std::set<long long>, long long> s;
      
      root.find_sets(&root, i->first, s);

      std::cout << "for item: " << i->first << std::endl;
      std::cout << "s.size() = " << s.size() << std::endl;
      for(auto& si : s){
	for(auto& sij : si.first)
	  std::cout << sij << " ";
	std::cout << " = " << si.second << std::endl;
      }

      sets.insert(std::pair(i->first, s));
    }

    /*
     * calculates frequent sets
     */
    freq_sets.clear();
    std::multimap<long long, std::set<long long> > fsets;
    
    for(const auto& si : sets){

      // {items} -> count
      const std::map<std::set<long long>, long long>& p = si.second;

      for(const auto& pi : p){
	long long total_count = 0;
	
	for(const auto& qi : p){
	  // if pi is in qi increases total_count   
	  if(std::includes(qi.first.begin(), qi.first.end(), pi.first.begin(), pi.first.end())){
	    total_count += qi.second;
	  }
	}

	if(total_count >= minfreq){
	  fsets.insert(std::pair(total_count, pi.first));
	}
      }
    }

    for(const auto& pi : fsets){
      bool add = true;

      /*
      for(const auto& qi : freq_sets){
	// if pi is in qi, don't add subset to the f-set    
	if(std::includes(qi.begin(), qi.end(), pi.second.begin(), pi.second.end())){
	  add = false;
	  break;
	}
      }
      */

      if(add){
	std::set<long long> f = pi.second;
	//f.insert(si.first);
	

	//std::cout << "adding: ";
	//for(auto& fi : f) std::cout << fi << " ";
	//std::cout << std::endl;
	
	freq_sets.insert(f);
      }
    }
    
    return (freq_sets.size() > 0);
#endif
  }

#endif
  
};


