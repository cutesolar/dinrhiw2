/*
 * tests for datamining code
 *
 */

#include <string>
#include <exception>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "AssociationRuleFinder.h"
#include "FrequentSetsFinder.h"
#include "list_source.h"
#include "discretize.h"
#include "RNG.h"

#include "fpgrowth.h"


void test_associationrulefinder();

void test_frequent_sets();

void test_enrich_data();

void test_fpgrowth();

using namespace whiteice;


int main()
{
  printf("DATAMINING CODE TESTS\n");
  
  srand(time(0));
  
  // test_associationrulefinder();

  test_frequent_sets();

  // test_fpgrowth();

  // test_enrich_data();
  
  return 0;
}


void test_fpgrowth()
{
  std::vector< std::set<long long> > data;
  std::set< std::set<long long> > fset;

  data.push_back({1, 6});
  data.push_back({1, 2, 3});
  data.push_back({1, 3, 5});
  data.push_back({1, 2, 6});
  data.push_back({1, 2, 3});
  data.push_back({1, 4, 7});
  data.push_back({1, 4, 9});

  if(whiteice::frequent_items(data, fset, 2.0/data.size()) == false){
    std::cout << "mining frequent itemsets FAILED!" << std::endl;
    return;
  }

  std::cout << "frequent items" << std::endl;

  for(const auto& s : fset){

    for(const auto& i : s){
      std::cout << i << " ";
    }

    std::cout << std::endl;
  }
  
}


void test_enrich_data()
{
  printf("TEST DISCRETIZATION\n");

  std::vector< std::vector<std::string> > data;
  
  for(unsigned int i=0;i<5000;i++){

    std::vector<std::string> vec;
    
    for(unsigned int k=0;k<10;k++){
      float v = whiteice::rng.normal().c[0];
      
      char buffer[80];
      sprintf(buffer, "%f", v);

      vec.push_back(std::string(buffer));
    }

    data.push_back(vec);
  }


  std::vector<struct whiteice::discretization> disc;

  if(whiteice::calculate_discretize(data, disc) == false){
    std::cout << "ERROR 1" << std::endl;
    return;
  }

  /*
  for(unsigned int k=0;k<disc.size();k++){
    std::cout << "k = " << k << " : " << disc[k].TYPE << std::endl;
    std::cout << "bins = " << disc[k].bins.size() << std::endl;
    
    for(unsigned int l=0;l<disc[k].bins.size();l++){
      std::cout << disc[k].bins[l] << std::endl;
    }
  }
  */

  std::vector< std::vector<double> > bindata;
  
  if(whiteice::binarize(data, disc, bindata) == false){
    std::cout << "ERROR 2" << std::endl;
    return;
  }

  std::cout << "binarized variables: " << bindata[0].size() << std::endl;

  /*
  for(unsigned int i=0;i<100;i++){
    for(unsigned int k=0;k<bindata[i].size();k++){
      std::cout << (int)bindata[i][k];
    }
    std::cout << std::endl;
  }
  */

  std::vector< std::vector<double> > results;
  std::set<whiteice::dynamic_bitset> f;

  if(whiteice::enrich_data(bindata, f, results) == false){
    std::cout << "ERROR 3" << std::endl;
    return;
  }

  std::cout << "rows in dataset: " << results.size() << std::endl;
  if(results.size() > 0)
    std::cout << "variables per row: " << results[0].size() << std::endl;

  std::vector<double> counts;
  counts.resize(results[0].size());

  for(auto& c : counts)
    c = 0.0;

  for(unsigned int i=0;i<results.size();i++){
    for(unsigned int k=0;k<results[i].size();k++){
      std::cout << (int)results[i][k];

      if(results[i][k]) counts[k]++;
    }
    
    std::cout << std::endl;
  }

  
  for(unsigned int k=0;k<counts.size();k++)
    std::cout << counts[k] << " ";
  
  std::cout << std::endl;
}


void test_frequent_sets()
{
  printf("TEST FREQUENT SETS FINDER\n");

  std::vector<dynamic_bitset> data;
  list_source<dynamic_bitset>* source;

  datamining::FrequentSetsFinder* fsfinder;
  std::vector<dynamic_bitset> fset;

  /*
  for(unsigned int i=0;i<5000;i++){
  dynamic_bitset x;
    x.resize(10);

    for(unsigned int i=0;i<x.size();i++)
      x.set(i, (bool)(rng.rand()&1));

    data.push_back(x);
  }
  */

  /*
  {
    dynamic_bitset x;
    x.resize(6);

    x.reset();
    x.set(0, true);
    x.set(1, true);
    x.set(2, true);
    data.push_back(x);

    x.reset();
    x.set(1, true);
    x.set(2, true);
    x.set(3, true);
    data.push_back(x);

    x.reset();
    x.set(3, true);
    x.set(4, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    x.set(1, true);
    x.set(3, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    x.set(1, true);
    x.set(2, true);
    x.set(4, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    x.set(1, true);
    x.set(2, true);
    x.set(3, true);
    data.push_back(x);
  }
  */

  /*
  {
    dynamic_bitset x;
    x.resize(5);

    x.reset();
    x.set(0, true);
    x.set(1, true);
    data.push_back(x);

    x.reset();
    x.set(1, true);
    x.set(2, true);
    x.set(3, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    x.set(2, true);
    x.set(3, true);
    x.set(4, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    x.set(3, true);
    x.set(4, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    x.set(1, true);
    x.set(2, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    x.set(1, true);
    x.set(2, true);
    x.set(3, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    x.set(1, true);
    x.set(2, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    x.set(1, true);
    x.set(3, true);
    data.push_back(x);

    x.reset();
    x.set(1, true);
    x.set(2, true);
    x.set(4, true);
    data.push_back(x);
  }
  */

  /*
  {
    dynamic_bitset x;
    x.resize(5);
    
    x.reset();
    x.set(0, true);
    x.set(1, true);
    x.set(3, true);
    x.set(4, true);
    data.push_back(x);

    x.reset();
    x.set(1, true);
    x.set(2, true);
    x.set(4, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    x.set(1, true);
    x.set(3, true);
    x.set(4, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    x.set(1, true);
    x.set(2, true);
    x.set(4, true);
    data.push_back(x);

    x.reset();
    x.set(0, true);
    x.set(1, true);
    x.set(2, true);
    x.set(3, true);
    x.set(4, true);
    data.push_back(x);

    x.reset();
    x.set(1, true);
    x.set(2, true);
    x.set(3, true);
    data.push_back(x);
  }
  */
  {
    dynamic_bitset x;
    x.resize(17);

    /*
      { f, a, c, d, g, i, m, p },
      { a, b, c, f, l, m, o },
      { b, f, h, j, o },
      { b, c, k, s, p },
      { a, f, c, e, l, p, m, n }
    */

    const int a = 0;
    const int b = 1;
    const int c = 2;
    const int d = 3;
    const int e = 4;
    const int f = 5;
    const int g = 6;
    const int h = 7;
    const int i = 8;
    const int j = 9;
    const int k = 10;
    const int l = 11;
    const int m = 12;
    const int n = 13;
    const int o = 14;
    const int p = 15;
    const int s = 16;
    
    x.reset();
    x.set(f, true);
    x.set(a, true);
    x.set(c, true);
    x.set(d, true);
    x.set(g, true);
    x.set(i, true);
    x.set(m, true);
    x.set(p, true);
    data.push_back(x);

    x.reset();
    x.set(a, true);
    x.set(b, true);
    x.set(c, true);
    x.set(f, true);
    x.set(l, true);
    x.set(m, true);
    x.set(o, true);
    data.push_back(x);

    x.reset();
    x.set(b, true);
    x.set(f, true);
    x.set(h, true);
    x.set(j, true);
    x.set(o, true);
    data.push_back(x);

    x.reset();
    x.set(b, true);
    x.set(c, true);
    x.set(k, true);
    x.set(s, true);
    x.set(p, true);
    data.push_back(x);

    x.reset();
    x.set(a, true);
    x.set(f, true);
    x.set(c, true);
    x.set(e, true);
    x.set(l, true);
    x.set(p, true);
    x.set(m, true);
    x.set(n, true);
    data.push_back(x);
  }

  source = new list_source<dynamic_bitset>(data);
  fsfinder = new datamining::FrequentSetsFinder(*source, fset, 3.0/data.size());

  fsfinder->find();

  std::cout << "Frequent sets found: " << fset.size() << std::endl;

  for(unsigned int i=0;i<fset.size();i++){
    std::cout << fset[i] << std::endl;
  }



  {
    std::set<whiteice::dynamic_bitset> f; // frequent sets

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

	  std::cout << "fsubset = " << c << std::endl;
	  f.insert(c);

	  b.inc();
	}
	
      }
    }

    
#if 0
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
#endif    
    
  }
  
}



void test_associationrulefinder()
{
  try{
    printf("ASSOCIATION RULE FINDER TESTS\n");
    
    
    // tests with logic operations
    // 3 input bits, 4 output bits, output bits are calculated with and/or/xor of input bits
    {
      std::vector<std::string> colnames; // names for columns of row vectors
      std::vector<dynamic_bitset> data;
      std::vector<datamining::rule> rules;
      
      list_source<dynamic_bitset>* source;
      datamining::AssociationRuleFinder* rulefinder;
      
      
      colnames.push_back("x");     // 0
      colnames.push_back("y");     // 1
      colnames.push_back("z");     // 2
      colnames.push_back("x * y"); // 3 (and)
      colnames.push_back("y * z"); // 4 (and)
      colnames.push_back("x + z"); // 5 (or)
      colnames.push_back("z");     // 6 (identity)
      
      dynamic_bitset x;
      x.resize(7);
      
      for(unsigned int i=0;i<5000;i++){
	x.reset();
	x.set(0, (bool)(rand() & 1) ); // x      50%
	x.set(1, (bool)(rand() & 1) ); // y      50%
	x.set(2, (bool)(rand() & 1) ); // z      50%
	x.set(3, x[0] & x[1] );        // x * y  25%
	x.set(4, x[1] & x[2] );        // y * z  25%
	x.set(5, x[0] | x[2] );        // x + z  25%
	x.set(6, x[2]);
	
	data.push_back(x);
      }
      
      
      printf("data size: %d\n", (int)data.size());
      
      source = new list_source<dynamic_bitset>(data);
      rulefinder = new datamining::AssociationRuleFinder(*source, rules, 0.2, 0.5);
      
      while(rulefinder->finished() == false){
	rulefinder->find(-1.0); // no time limits
	printf("total number of rules: %d\n", (int)rules.size());
      }
      
      rulefinder->clean();
      delete rulefinder;
      delete source;
      
      // shows found rules (in no specific order)
      for(unsigned int i=0;i<rules.size();i++){
	printf("RULE %3d (%3.2f , %3.2f)  ", i+1,
	       rules[i].frequency, rules[i].confidence);
	
	unsigned int k = 0;
	printf("(");
	for(unsigned int j=0;j<rules[i].x.size();j++){
	  if(rules[i].x[j]){
	    printf("%s", colnames[j].c_str());
	    
	    k++;
	  
	    if(k != rules[i].x.count())
	      printf(",");
	  }
	}
	printf(") => (");
	
	k = 0;
	for(unsigned int j=0;j<rules[i].y.size();j++){
	  if(rules[i].y[j]){
	    printf("%s", colnames[j].c_str());
	    
	    k++;
	    
	    if(k != rules[i].y.count())
	      printf(",");
	  }
	}
	
	printf(")\n");
      }
            
    }
    
    
    
    std::cout << "ASSOCIATION RULE FINDER TESTS PASSED" << std::endl;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }
}
