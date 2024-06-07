/*
 * some unit testing
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <errno.h>
#include <string>
#include <vector>

#ifndef WINOS

#include "point.h"
#include "static_array.h"
#include "dynamic_array.h"
#include "timed_boolean.h"
#include "primality_test.h"
#include "dynamic_bitset.h"
#include "fast_radix.h"
#include "fast_radixv.h"
#include "fast_gradix.h"
#include "priority_queue.h"
#include "augmented_data.h"
#include "rbtree.h"
#include "binary_tree.h"
#include "avltree.h"
#include "dataset.h"
#include "unique_id.h"
#include "conffile.h"
#include "list_source.h"
//#include "MemoryCompressor.h"

#else
// eclipse has different build process we test with ready compiled library
#include "dinrhiw.h"
#endif

#include "RNG.h"

using namespace whiteice;

void point_test();
void sarray_test();
void darray_test();
void stack_test();
void queue_test();
void test_timed_boolean();
void primality_test();
void print_bits();
void dynamic_bitset_test();
void test_priority_queue_and_fast_radix();
void test_fast_radixv();
void test_fast_gradix();
void test_rbtree();
void test_avltree();
void test_binary_tree();
void test_dataset();

// same as test_dataset() but with blas_complex<double>
// [complex numbers implementation is a buggy in the library]
void test_dataset_complex(); 

void test_dataset_ica();

void test_dataset_superreso(); // tests superresolutional number implementation [mostly save and load] 


void test_uniqueid();
void test_conffile();
//void test_compression();
void test_list_source();


// void test_optimum_binary_tree();
// void test_multivariable_optimum_binary_quadtree();
// void test_hashbased_optimum_n_branch_tree();
// void test_mutlivariable_hashbased_optimum_n_branch_tree();


/********************************************************************************/


int main()
{
  /* very simple tests / not complete/good tests,
   * 'seems to work tests'
   */
  unsigned int seed = time(0);
  printf("STARTING TESTS\n");
  
  // seed = 0x42d0592f; // seed which exposes conffile save() & load() bug.  
  // seed = 0x53a59716;
  // seed = 0x53a5b194;
  // seed = 0x53b7ac0c;
  seed = time(0);
  seed = 0x557abbe3; // exposes early buf in dataset<>
  

  printf("randomization seed is 0x%x\n", seed);
  srand(seed);

  test_dataset_superreso();
  
  test_dataset();
  test_dataset_ica();

  test_dataset_complex();
  
  point_test();
  sarray_test();
  darray_test();
  stack_test();
  queue_test();
  test_timed_boolean();
  primality_test();
  print_bits();
  dynamic_bitset_test();
  test_priority_queue_and_fast_radix();
  test_fast_radixv();
  test_fast_gradix();
  test_rbtree();
  test_binary_tree();

  // test_avltree();
  
  // test_uniqueid(); // FIXME fix this testcase
  test_conffile();
  //test_compression();
  test_list_source();
  
  
  return 0;
}



////////////////////////////////////////////////////////////


class test_exception : public std::exception
{
public:
  
  test_exception() 
  {
    reason = 0;
  }
  
  test_exception(const std::exception& e) 
  {
    reason = 0;
    
    const char* ptr = e.what();
    
    if(ptr)
      reason = (char*)malloc(sizeof(char)*(strlen(ptr) + 1));
    
    if(reason) strcpy(reason, ptr);
  }
  
  
  test_exception(const char* ptr) 
  {
    reason = 0;
    
    if(ptr){
      reason = (char*)malloc(sizeof(char)*(strlen(ptr) + 1));
      
      if(reason) strcpy(reason, ptr);
    }
  }
  
  
  virtual ~test_exception() 
  {
    if(reason) free(reason);
    reason = 0;
  }
  
  virtual const char* what() const throw() 
  {
    if(reason == 0) // null terminated value
      return ((const char*)&reason);

    return reason;
  }
  
private:
  
  char* reason;
  
};



////////////////////////////////////////////////////////////

// tests superresolutional number implementation [mostly save and load]
void test_dataset_superreso()
{
  printf("DATASET SUPERRESOLUTIONAL NUMBERS TEST.\n");
  
  try{
    using namespace whiteice;
    using namespace whiteice::math;
    
    dataset< superresolution< blas_complex<double>, modular<unsigned int> > > data;
    std::vector< vertex< superresolution< blas_complex<double>, modular<unsigned int> > > > list;
    vertex< superresolution< blas_complex<double>, modular<unsigned int> > > v;

    data.createCluster("test-sdata", 10);
    
    v.resize(data.dimension(0));

    for(unsigned int i=0;i<100;i++){
      for(unsigned int d=0;d<v.size();d++)
	for(unsigned int k=0;k<v[d].size();k++)
	  for(unsigned int l=0;l<v[d][k].size();l++)
	    v[d][k][l] = rng.uniformf();
      
      list.push_back(v);
    }

    if(data.add(0, list) == false){
      printf("ERROR: adding list of numbers to dataset FAILs.\n");
      return;
    }

    if(data.save("sdata.dat") == false){
      printf("ERROR: saving dataset FAILs.\n");
      return;
    }

    dataset< superresolution< blas_complex<double>, modular<unsigned int> > > data2;
    std::vector< vertex< superresolution< blas_complex<double>, modular<unsigned int> > > > list2;

    if(data2.load("sdata.dat") == false){
      printf("ERROR: loading dataset FAILs.\n");
      return;
    }

    if(data2.getNumberOfClusters() != 1){
      printf("ERROR: wrong number of clusters after load().\n");
      return;
    }

    if(data2.dimension(0) != data.dimension(0)){
      printf("ERROR: wrong data dimension after load().\n");
      return;
    }
    
    if(data2.getData(0, list2) == false){
      printf("ERROR: getData() FAILs after load().\n");
      return;
    }

    if(list.size() != list2.size()){
      printf("ERROR: data sizes mismatch after load(save(x)).\n");
      return;
    }

    if(list2[0].size() != list[0].size()){
      printf("ERROR: data sizes mismatch (2) after load(save(x)).\n");
      return;
    }
    
    for(unsigned int i=0;i<list.size();i++){

      auto u = list[i];
      auto v = list2[i];
      
      for(unsigned int d=0;d<v.size();d++){
	for(unsigned int k=0;k<v[d].size();k++){
	  if(abs(u[d][k]-v[d][k]) > 0.0001f){
	    printf("ERROR: data elements mismatch after load(save(x)).\n");
	    return;
	  }
	}
      }
    }


    printf("SUCCESS: load(save(x)) == x\n");
    
    return;
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  }
  
}

////////////////////////////////////////////////////////////

void test_dataset_ica()
{
  try{
    printf("ICA TEST DATASET FUNCTIONALITY\n");
    
    whiteice::dataset<> data;

    // 1. we create random dataset consisting of
    //    four (4) independent signals and mix them and
    //    store them to dataset

    std::string name = "source signals";
    
    if(data.createCluster(name, 4) == false)
      throw test_exception("createCluster failed.");

    name = "mixed signals";
	
    if(data.createCluster(name, 4) == false)
      throw test_exception("createCluster failed.");

    std::vector<math::blas_real<float> > t;
    t.resize(1000);

    math::matrix< math::blas_real<float> > A;
    A.resize(4,4);
    for(unsigned int j=0;j<A.ysize();j++)
      for(unsigned int i=0;i<A.xsize();i++)
	A(j,i) = ((float)rand())/((float)RAND_MAX);
    
    for(unsigned int i=0;i<t.size();i++){
      float T = 10.0f*((float)i/((float)t.size()));
      t[i] = T;
      math::vertex< math::blas_real<float> > d, x;
      d.resize(4);
      d[0] = sin(T);
      d[1] = cos(T)*cos(T)*cos(T);
      d[2] = ((float)rand())/((float)RAND_MAX);
      d[3] = 4.0*cos(sin(T));

      x = A*d; // mixes signals with mixing matrix A

      if(data.add(0, d) == false)
	throw test_exception("adding data to cluster failed");

      if(data.add(1, x) == false)
	throw test_exception("adding data to cluster failed");
    }


    // 2. check preprocessing with ICA works (do not return failure)
    {
      if(data.preprocess(1, whiteice::dataset<>::dnLinearICA) == false){
	std::cout << "ICA preprocessing of dataset FAILS."
		  << std::endl;
	return;
      }
      
    }
    
    // 3. store and load the dataset, remove ICA preprocessing and
    //    check that dataset vectors match after removal of all dataset
    //    preprocessings
    {
      whiteice::dataset<> data2(data);

      std::cout << "DATASET STORE AND LOAD TEST" << std::endl;
      
      // 1. saves and loads dataset data
      std::string fname = "ica_dataset.ds";
      
      if(data.save(fname) == false){
	std::cout << "ERROR: cannot save dataset file." << std::endl;
	return;
      }

      if(data.load(fname) == false){
	std::cout << "ERROR: cannot load dataset file." << std::endl;
	return;
      }

      std::cout << "DATASET SAVE() & LOAD() CALLS OK" << std::endl;
      
      // 2. check data2 and data objects MATCH
      if(data.getNumberOfClusters() != data2.getNumberOfClusters()){
	std::cout << "ERROR: number of clusters mismatch." << std::endl;
	return;
      }

      for(unsigned int i=0;i<data.getNumberOfClusters();i++){
	if(data.getName(i) != data2.getName(i)){
	  std::cout << "ERROR: cluster name mismatch." << std::endl;
	  return;
	}
	if(data.size(i) != data2.size(i)){
	  std::cout << "ERROR: cluster size mismatch." << std::endl;
	  return;
	}
	if(data.dimension(i) != data2.dimension(i)){
	  std::cout << "ERROR: cluster dimension mismatch." << std::endl;
	  return;
	}

	std::vector<whiteice::dataset<>::data_normalization> pp1;
	std::vector<whiteice::dataset<>::data_normalization> pp2;

	if(data.getPreprocessings(i, pp1) == false || data2.getPreprocessings(i, pp2) == false){
	  for(unsigned int j=0;j<pp1.size();j++){
	    if(pp1[j] != pp2[j]){
	      std::cout << "ERROR: preprocessings mismatch." << std::endl;
	      return;
	    }
	  }
	}

	std::cout << "DATA TESTS" << std::endl;

	// data test
	for(unsigned int index=0;index<data.size(i);index++){
	  math::vertex<> e = data.access(i, index) - data2.access(i, index);
	  if(e.norm() > 0.01){
	    std::cout << "ERROR: dataset cluster " << i << " data mismatch." << std::endl;
	    return;
	  }
	}
	   
      }
      
      //////////////////////////////////////////////////////////////////
      // 3. remove preprocessings, check data objects MATCH AGAIN
      
      for(unsigned int i=0;i<data.getNumberOfClusters();i++){
	if(data.convert(i) == false || data2.convert(i) == false){
	  std::cout << "ERROR: removing preprocessings from clusters FAILED: " << i << std::endl;
	  std::cout << "Number of clusters: "
		    << data.getNumberOfClusters() << " , "
		    << data2.getNumberOfClusters() << std::endl;
	  return;
	}
      }


      for(unsigned int i=0;i<data.getNumberOfClusters();i++){
	if(data.getName(i) != data2.getName(i)){
	  std::cout << "ERROR: cluster name mismatch." << std::endl;
	  return;
	}
	if(data.size(i) != data2.size(i)){
	  std::cout << "ERROR: cluster size mismatch." << std::endl;
	  return;
	}
	if(data.dimension(i) != data2.dimension(i)){
	  std::cout << "ERROR: cluster dimension mismatch." << std::endl;
	  return;
	}

	std::vector<whiteice::dataset<>::data_normalization> pp1;
	std::vector<whiteice::dataset<>::data_normalization> pp2;

	if(data.getPreprocessings(i, pp1) == false || data2.getPreprocessings(i, pp2) == false){
	  for(unsigned int j=0;j<pp1.size();j++){
	    if(pp1[j] != pp2[j]){
	      std::cout << "ERROR: preprocessings mismatch." << std::endl;
	      return;
	    }
	  }
	}

	// data test
	for(unsigned int index=0;index<data.size(i);index++){
	  math::vertex<> e = data.access(i, index) - data2.access(i, index);
	  if(e.norm() > 0.01){
	    std::cout << "ERROR: dataset cluster " << i
		      << " data mismatch (index = "
		      << index << ")" << std::endl;

	    std::cout << data.access(i, index) << " != " << data2.access(i, index) << std::endl;
	    return;
	  }
	}
	   
      }
    }
      
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  }
} 

////////////////////////////////////////////////////////////

void stack_test()
{
  try{
    printf("STACK TEST\n");
    
    stack<int> *s = 0;
    s = new dynamic_array<int>;
    if(s == 0)
      printf("ERROR: stack - bad pointer\n");
    
    delete s;
    
    
    s = new dynamic_array<int>(0);
    
    for(int i=0;i<5;i++){
      s->push(i);
    }
    
    if(s->size() != 5)
      printf("ERROR - stack size: %d\n", s->size());
    
    for(int i=4;i>=0;i--){
      int t = s->pop();
      if(t != i) printf("ERROR: wrong pop value: %d != %d ", t, i);
      if((signed)s->size() != i) printf("WRONG |s| = %d\n", s->size() );
    }
    
    if(s->size() != 0)
      printf("ERROR - stack size: %d\n", s->size());
    
    for(int i=0;i<5;i++){
      s->push(i);
    }
    
    if(s->size() != 5)
      printf("ERROR - stack size: %d\n", s->size());
    
    for(int i=4;i>=0;i--){
      int t = s->pop();
      if(t != i) printf("ERROR: wrong pop value: %d != %d ", t, i);
      if((signed)s->size() != i) printf("WRONG |s| = %d\n", s->size() );    
    }
    
    delete s;
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  }
}

/********************************************************************************/

void queue_test()
{
  try{
    printf("QUEUE TEST\n");
    queue<int> *q = new dynamic_array<int>(0);
    if(q == 0) printf("ERROR: queue - null pointer\n");
    delete q;
    
    q = new dynamic_array<int>;
    
    for(int i=0;i<5;i++)
      q->enqueue(i);
    
    if((signed)q->size() != 5)
      printf("ERRORa: queue size: %d\n", (signed)q->size());
    
    for(int i=4;i>=0;i--){
      int a = q->dequeue();
      if(a != 4-i) printf("ERROR: bad value. %d \n", q->dequeue() );
      if(i != (signed)q->size()) printf("ERROR wrong size: |q| = %d\n", (signed)q->size());
    }
    
    delete q;
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  }
}

/********************************************************************************/

void darray_test()
{
  try{
    printf("DYNAMIC ARRAY TEST\n");
    
    dynamic_array<float> a;
    dynamic_array<float> b(11);
    
    a.resize(9);
    
    if((signed)a.size() != 9 || (signed)b.size() != 11)
      printf("ERROR: 9,11 != %d,%d\n", a.size(), b.size());
    
    for(int i=0;i<(signed)a.size();i++){
      a[i] = 2*(i+1);
    }
    
    for(int i=0;i<(signed)a.size();i++){
      b[i] = a[i];
    }
    
    for(unsigned int i=a.size();i<b.size();i++)
      b[i] = 0.0f;
    
    for(int i=0;i<(signed)b.size();i++){
      if(i < (signed)a.size()){
	if(b[i] != 2*(i+1)) printf("ERRORa: bad value - %f\n", b[i]);
      }
      else{
	if(b[i] != 0.0f) printf("ERRORb: bad value - %f\n", b[i]);
      }
    }
    
    try{
      a[-1] = 32;
      printf("ERROR: no exception thrown\n");
    }
    catch(std::out_of_range& e){ }
    catch(std::exception& e){ printf("ERROR: wrong exception throwed\n"); }
    
    try{
      a[9] = 32;
      printf("ERROR: no exception thrown\n");
    }
    catch(std::out_of_range& e){ }
    catch(std::exception& e){ printf("ERROR: wrong exception throwed\n"); }
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  }
}

/********************************************************************************/

void sarray_test()
{
  try{
    printf("STATIC ARRAY TEST\n");
    
    static_array<float> stratic;
    static_array<float> b(10);
    
    for(int i=0;i<(signed)b.size();i++){
      b[i] = 0;
    }
    
    stratic.resize(8);
    
    if(stratic.size() != 8 || b.size() != 10)
      printf("ERROR: wrong sizes. 8,10 != %d,%d\n", stratic.size(), b.size());
    
    for(int i=0;i<(signed)stratic.size();i++){
      stratic[i] = 2*(i+1);
    }
    
    for(int i=0;i<(signed)stratic.size();i++){
      b[i] = stratic[i];
    }
    
    for(int i=0;i<(signed)b.size();i++){
      if(i < (signed)stratic.size()){
	if(b[i] != stratic[i]) printf("ERRORa: bad value - %f\n", b[i]);
      }
      else{
	if(b[i] != 0) printf("ERRORa: bad value - %f\n", b[i]);
      }
    }
    
    try{
      stratic[-1] = 32;
      printf("ERROR: no exception thrown\n");
    }
    catch(std::out_of_range& e){ }
    catch(std::exception& e){ printf("ERROR: wrong exception throwed\n"); }
    
    try{
      stratic[8] = 32;
      printf("ERROR: no exception thrown\n");
    }
    catch(std::out_of_range& e){ }
    catch(std::exception& e){ printf("ERROR: wrong exception throwed\n"); }  
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  }
  
}

/********************************************************************************/

void point_test()
{
  try{
    printf("POINT TEST\n");
    
    point2d<int> a,b;
    point3d<int> c;
    
    b.setY(2);
    a[0] = 1;
    a[1] = 2;
    
    b += a;
    
    if(b[0] != 1 || b[1] != 4){
      printf("ERROR: bad values\n");
      b.print(); // should be (1,4)
      printf("\n");
    }
    
    int k = a * b;
    
    if(k != 9){
      printf("ERROR: should be 9: %d\n", k);
    }
    
    c[0] = 2;
    c[1] = 3;
    c[2] = 3;
    
    if(c[0] != 2 || c[1] != 3 || c[2] != 3){
      printf("ERROR: bad values ");
      c.print();
      printf("\n");
    }
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  }
}

/********************************************************************************/

void test_timed_boolean()
{
  try{
    
    std::cout << "(SIMPLE) TIMED BOOLEAN TESTS" << std::endl;
    
    // TEST1: timed inversion and basic operations test after
    // timeout
    {
      timed_boolean timeout(2.0, false);
      
      if(timeout.time_left() <= 0.0){
	std::cout << "ERROR: time_left() <= 0.0 before timeout"
		  << std::endl;
	return;
      }
      
      if(timeout == true){
	std::cout << "ERROR: timeout != false before timeout"
		  << std::endl;
	return;
      }
      
      if(!timeout == false){
	std::cout << "ERROR: !timeout != true before timeout"
		  << std::endl;
      }
      
      
      // waits aprox 2.0 sec
      unsigned int counter = 0;      
      while(timeout == false){
	timed_boolean timeout2(0.5, false);
	while(timeout2 == false);
	std::cout << "waiting for master timeout..." << std::endl
		  << "counter:   " << counter << std::endl
		  << "time left: " << timeout.time_left()
		  << std::endl; 
	counter++;
      }
      
      
      
      if(timeout.time_left() > 0.0){
	std::cout << "ERROR: time_left() > 0 after timeout"
		  << std::endl;
	return;
      }
      
      if(timeout != true){
	std::cout << "ERROR: timeout != true after timeout"
		  << std::endl;
	return;
      }
      
      if(!timeout != false){
	std::cout << "ERROR: !timeout != false after timeout"
		  << std::endl;
      }
      
    }
    
    
    std::cout << "(SIMPLE) TIMED BOOLEAN TESTS OK" << std::endl;
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  } 
}


/********************************************************************************/

void primality_test()
{
  try{
    
    printf("PSEUDOPRIME TEST\n");
    
    if(pseudoprime<int>(100))
      printf("ERROR: 100 is prime  (no) ? %d\n", (int)pseudoprime<int>(100)  );
    
    if(!pseudoprime<int>(1009))
      printf("ERROR: 1009 is prime (yes)? %d\n", (int)pseudoprime<int>(1009) );
    
    if(!pseudoprime<int>(773))
      printf("ERROR: 773  is prime (yes)? %d\n", (int)pseudoprime<int>(773)  );
    
    if(!pseudoprime<int>(1531))
      printf("ERROR: 1531 is prime (yes)? %d\n", (int)pseudoprime<int>(1531) );
    
    if(!pseudoprime<int>(4211))
      printf("ERROR: 4211 is prime (yes)? %d\n", (int)pseudoprime<int>(4211) );
    
    if(pseudoprime<int>(36))
      printf("ERROR: 36   is prime (no) ? %d\n", (int)pseudoprime<int>(36)   );
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  }
}

/********************************************************************************/  

/*
 * tests nth bit
 * implementation
 */
void print_bits()
{
  try{
    int num = (32 + 3);
    
    printf("binary representation of %d = ", num);
    for(int i=31;i >= 0;i--){
      printf("%d", nth_bit<int>(num, i) );
    }
    
    printf("\n");
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  }
}


/********************************************************************************/



void dynamic_bitset_test()
{
  
  std::cout << "BITSET TEST" << std::endl;
  
  unsigned int t;
  
  
  // TEST CASE 1: set, reset, flip, resize, size, count, any, none, operator[]
  try{
    t = 1;
    
    std::vector< whiteice::dynamic_bitset > b;
    b.resize(10);

    
    // tests size() and []-operator
    for(unsigned int i=0;i<b.size();i++){
      b[i].resize(27*i+1);
      
      if(b[i].size() != (27*i + 1)){
	throw test_exception("size() returns wrong value after resize()");
      }
      
      for(unsigned int j=0;j<b[i].size();j++){
	try{
	  b[i].set(j, 0);
	}
	catch(std::out_of_range& e){
	  throw test_exception("operator[] throws exception on a correct range");
	}
      }
      
      try{
	std::cout << b[i][b[i].size() + 1] << std::endl;
	
	throw test_exception("operator[] doesn't throw exception with index out of range");
      }
      catch(std::out_of_range& e){
      }
    }       
    
    
    // tests reset, set, flip, count, any, none, []-operator
    for(unsigned int i=0;i<b.size();i++){
      b[i].reset();
      
      if(b[i].any())
	throw test_exception("zero bitset cannot have any() bits set");
      
      if(b[i].none() == false)
	throw test_exception("zero bitset must have none() bits set");
      
      if(b[i].count() != 0)
	throw test_exception("zero bitset must have zero count");
      
      b[i].set();
      
      if(b[i].any() == false)
	throw test_exception("all ones bitset must some ( any() ) bits set [1]");
      
      if(b[i].none())
	throw test_exception("all ones bitset cannot have none() bits set [1]");
      
      if(b[i].count() != b[i].size()){
	std::cout << b[i].count() << " != " << b[i].size() << std::endl;
	for(unsigned int j=0;j<b[i].size();j++)
	  std::cout << b[i][j];
	std::cout << std::endl;
	
	throw test_exception("all ones bitset must have count() equal to length [1]");
      }
      
      for(unsigned int j=0;j<b[i].size();j++){
	if(j & 1) b[i].reset(j);	
      }
      
      unsigned int k = b[i].count();
      b[i].flip();
      
      k += b[i].count();
      
      if(b[i].size() != k)
	throw test_exception("sum of zeros and ones in bitset must be equal to length");
      
      // flips things back to full one bitset and checks the result
      
      for(unsigned int j=0;j<b[i].size();j++){
	if(!(j & 1)) b[i].flip(j);
      }     
      
      // full ones tests
      
      if(b[i].any() == false)
	throw test_exception("all ones bitset must have some ( any() ) bits set [2]");
      
      if(b[i].none())
	throw test_exception("all ones bitset cannot have none() bits set [2]");
      
      if(b[i].count() != b[i].size())
	throw test_exception("all ones bitset muts have count() equal to length [2]");
    }
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }


  // TEST CASE 2: xor/and/or, shifting, equality
  try{
    t = 2;
    
    unsigned int bits = 10 + ( rand() % 128);
    
    whiteice::dynamic_bitset temp; temp.resize(bits);
    std::vector<whiteice::dynamic_bitset > b;
    b.resize(10);
    
    // inits Xbits long vectors with random data
    for(unsigned int i=0;i<b.size();i++){
      b[i].resize(bits);
      
      for(unsigned int j=0;j<b[i].size();j++){
	b[i].set(j, rand() & 1);
      }
      
    }
    
    
    b[0] = b[b.size()-1];
    
    // checks equality and unequality
    for(unsigned int i=0;i<b.size();i++){
      for(unsigned int j=0;j<b.size();j++){
	
	if(i == j){
	  if(b[i] != b[j])
	    throw test_exception("same vectors are unequal");
	}
	else{
	  if(b[i] == b[j]){
	    
	    for(unsigned int k=0;k<b[i].size();k++){
	      if(b[i][k] != b[j][k])
		throw test_exception("==operator returns true when bitsets are different");
	    }
	  }
	}
	
      }
    }
    
    
    // test: double xor should cancel
    for(unsigned int i=0;i<b.size();i++){
      temp = b[i];            
      
      unsigned int j = i;
      while(j == i) j = rand() % b.size();
	
      temp ^= b[j];
      
      if(temp == b[i]){
	char buf[200];
	sprintf(buf,"probably error (2^-%d prob. for success): bitset is same after xorring",
		bits);
	
	throw test_exception(buf);
      }
      
      temp ^= b[j];
      
      if(temp != b[i]){
	std::cout << std::hex;
	std::cout << temp << std::endl;
	std::cout << b[i] << std::endl;
	std::cout << b[j] << std::endl;
	std::cout << std::dec;
	throw test_exception("error: double xor doesn't return original value");
      }
      
    }
    
    
    // reinit with random data
    for(unsigned int i=0;i<b.size();i++){
      
      for(unsigned int j=0;j<b[i].size();j++)
	b[i].set(j, rand() & 1);
    }
    
    
    // ANDing and ORring tests
    for(unsigned int i=0;i<b.size();i++){
      temp = b[i];
      
      unsigned int j = i;
      while(j == i) j = rand() % b.size();
      
      temp |= b[j];
      
      for(unsigned int k=0;k<b[i].size();k++){
	b[i].set(k, b[i][k] || b[j][k]);
      }
      
      if(temp != b[i])
	throw test_exception("ORring bit by bit or as whole bitset returns incorrect results");


      temp = b[i];
      
      j = i;
      while(j == i) j = rand() % b.size();
      
      temp &= b[j];
      
      for(unsigned int k=0;k<b[i].size();k++){
	b[i].set(k, b[i][k] && b[j][k]);
      }
      
      if(temp != b[i])
	throw test_exception("ANDing bit by bit or as whole bitset returns incorrect results");
                  
    }
    
    
    // reinit with random data
    for(unsigned int i=0;i<b.size();i++){
      
      for(unsigned int j=0;j<b[i].size();j++)
	b[i].set(j, rand() & 1);
    }
      

      
      

    // shifting test: shift by random length, compares with manual shifting      
    // left shift
    for(unsigned int i=0;i<b.size();i++){
      
      unsigned int sh = rand() % bits;
      
      whiteice::dynamic_bitset orig = b[i];
      
      temp = b[i];
      
      b[i] <<= sh;
      
      if(sh){
	for(unsigned int j=(b[i].size() - 1);j>=sh;j--){
	  temp.set(j, temp[j - sh]); 
	}
	
	for(unsigned int j=0;j<sh;j++){
	  temp.reset(j);
	}
      }
      
      if(temp != b[i]){
	std::cout << "*** ERROR: " << std::endl;
	std::cout << "orig = " << orig << std::endl;
	std::cout << "|b[i]| = " << b[i].size() << std::endl;
	std::cout << "temp = " << temp.to_string() << std::endl;
	std::cout << "b[i] = " << b[i].to_string() << std::endl;
	std::cout << "rshift: " << sh << std::endl;

	throw test_exception("left shifting gave bad result");
      }
    }
    
    
    // reinit with random data
    for(unsigned int i=0;i<b.size();i++){
	
      for(unsigned int j=0;j<b[i].size();j++)
	b[i].set(j, rand() & 1);
    }

    
    // right shift
    for(unsigned int i=0;i<b.size();i++){
      
      unsigned int sh = rand() % bits;
      
      whiteice::dynamic_bitset orig = b[i];
      
      temp = b[i];
      
      b[i] >>= sh;
	
      if(sh){
	for(unsigned int j=0;j<(b[i].size() - sh);j++){
	  temp.set(j, temp[j + sh] );
	}
	
	for(unsigned int j=(b[i].size() - sh);j<b[i].size();j++){
	  temp.reset( j );
	}
      }
      
      
      if(temp != b[i]){
	std::cout << "*** ERROR: " << std::endl;
	std::cout << "orig = " << orig << std::endl;
	std::cout << "|b[i]| = " << b[i].size() << std::endl;
	std::cout << "temp = " << temp.to_string() << std::endl;
	std::cout << "b[i] = " << b[i].to_string() << std::endl;
	std::cout << "rshift: " << sh << std::endl;
	
	throw test_exception("right shifting gave bad result");
      }
    }
    
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }


  // TEST CASE 3: to string conversion, to integer conversion, from int conversion,
  //              ctors
  try{
    t = 3;
    

    // hex string conversion test
    {
      unsigned int ai = 0xF1F0DAAA, bi = 0xFAABACE0;
      
      whiteice::dynamic_bitset a(ai);
      whiteice::dynamic_bitset b(bi);
      
      a += b;
      
      std::cout << "hex strings should be same" << std::endl;
      std::cout << "0x" << std::hex << ai << bi << std::endl;
      std::cout << std::hex << std::showbase << a << std::endl;
      std::cout << std::noshowbase << std::endl;
    }
    
    std::cout << "WARN: TEST CASE 3: non-critical tests NOT DONE" << std::endl;
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }



  // TEST CASE 4: random operations
  
  
  // change: with different bitset sizes <= sizeof(int)*8
  
  
  try{
    t = 4;
    
    whiteice::dynamic_bitset a,b,c;
    int A,B,C;
    
    std::vector<int> ops;
    
    a.resize(sizeof(int)*8);
    b.resize(sizeof(int)*8);
    c.resize(sizeof(int)*8);
    
    unsigned int iter = 0;    
    
    while(iter < 1000){
      
      A = rand();
      B = rand();
      C = rand();
      
      for(unsigned int i=0;i<a.size();i++){
	a.set(i, ((A >> i) & 1) );
	b.set(i, ((B >> i) & 1) );
	c.set(i, ((C >> i) & 1) );
      }
      
      
      ops.resize((rand() % 10) + 1);
      for(unsigned int i=0;i<ops.size();i++)
	ops[i] = rand() % 3; // 0 == and, 1 == or, 2 == xor

      for(unsigned int i=0;i<ops.size();i++){
	// 0 == abc, 1 == acb, 2 == bac, 3 == bca, 4 == cab, 5 == cba
	unsigned int order = rand() % 6;
	
	if(order == 0){
	  
	  if(ops[i] == 0){ // abc
	    a = b & c;
	    A = B & C;
	  }
	  else if(ops[i] == 1){
	    a = b | c;
	    A = B | C;
	  }
	  else if(ops[i] == 2){
	    a = b ^ c;
	    A = B ^ C;
	  }
	}
	else if(order == 1){ // acb
	  
	  if(ops[i] == 0){
	    a = c & b;
	    A = C & B;
	  }
	  else if(ops[i] == 1){
	    a = c | b;
	    A = C | B;
	  }
	  else if(ops[i] == 2){
	    a = c ^ b;
	    A = C ^ B;
	  }
	}
	else if(order == 2){ // bac
	  
	  if(ops[i] == 0){
	    b = a & c;
	    B = A & C;
	  }
	  else if(ops[i] == 1){
	    b = a | c;
	    B = A | C;
	  }
	  else if(ops[i] == 2){
	    b = a ^ c;
	    B = A ^ C;
	  }
	}
	else if(order == 3){ // bca
	  
	  if(ops[i] == 0){
	    b = c & a;
	    B = C & A;
	  }
	  else if(ops[i] == 1){
	    b = c | a;
	    B = C | A;
	  }
	  else if(ops[i] == 2){
	    b = c ^ a;
	    B = C ^ A;
	  }
	}
	else if(order == 4){ // cab
	  
	  if(ops[i] == 0){
	    c = a & b;
	    C = A & B;
	  }
	  else if(ops[i] == 1){
	    c = a | b;
	    C = A | B;
	  }
	  else if(ops[i] == 2){
	    c = a ^ b;
	    C = A ^ B;
	  }
	}
	else if(order == 5){ // cba
	  
	  if(ops[i] == 0){
	    c = b & a;
	    C = B & A;
	  }
	  else if(ops[i] == 1){
	    c = b | a;
	    C = B | A;
	  }
	  else if(ops[i] == 2){
	    c = b ^ a;
	    C = B ^ A;
	  }
	}
      }

      // checks result of sequence of operators
      
      for(unsigned int i=0;i<a.size();i++){
	if(a[i] != ((A >> i) & 1)){
	  throw test_exception("random sequence of xor/and/or operators gave wrong results");
	}
	   
	if(b[i] != ((B >> i) & 1)){
	  throw test_exception("random sequence of xor/and/or operators gave wrong results");
	}
	
	
	if(c[i] != ((C >> i) & 1)){
	  throw test_exception("random sequence of xor/and/or operators gave wrong results");
	}
      }
      
      iter++;
    }
    
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }
  
  
}



/********************************************************************************/


/*
 * proper test not done - code actual sorting test
 * (test against quicksort)
 */
void test_priority_queue_and_fast_radix()
{
  try{
    
    printf("PRIORITY QUEUE & RADIX SORT TEST\n");
    std::cout << std::dec << std::endl;
  
    priority_queue<int>* pq = 
      new priority_queue<int>(16);
    
    const unsigned int SIZE = 100;
    std::vector<int> floats;
    int* table  = new int[SIZE];
    
    fast_radix<int>* rs = 
      new fast_radix<int>(table, SIZE);
    
    for(unsigned int i=0;i<SIZE;i++){
      int v = 0;
      
      while(v < (signed)SIZE){ v = rand() % (SIZE*10); }
      floats.push_back( v );
      table[i]  = floats[i];
    }
    
    std::cout << "INIT DONE\n";
    
    for(unsigned int i=0;i<SIZE;i++){
      if(pq->insert(floats[i]) == false)
	std::cout << "PQ->insert : " << false << std::endl;
    }
    
    //std::cout << "INPUT VALUES: \n";
    for(unsigned int i=0;i<floats.size();i++);
      //std::cout << floats[i] << "  ";
      //std::cout << std::endl;
    
    //std::cout << "PRIORITY QUEUE OUTPUT VALUES: \n";
    while(!pq->empty()){
      /*int v =*/ pq->extract();
      
      //std::cout << v << "  ";
    }
    //std::cout << std::endl;
    
    if(!rs->sort()){
      std::cout << "RADIX SORT FAILED\n";
    } 
    
    //std::cout << "RADIX SORT OUTPUT VALUES: \n";
    for(unsigned int i=0;i<SIZE;i++){
      // std::cout << table[i] << "  ";
    }
    std::cout << std::endl;
    
    if(table) delete[] table;
    if(pq) delete pq;
    if(rs) delete rs;
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  }
}



/********************************************************************************/

class RN : public radixnode
{
public:
  virtual ~RN(){ }
  
  virtual const dynamic_bitset& radixkey() { return key; }
  
  dynamic_bitset key;
  std::string label;
};


void test_fast_gradix()
{
  try{
    std::cout << "FAST GENERIC KEYED RADIX TEST" << std::endl;
    bool test_ok = true;
    
    // generates test data
    const unsigned int N = 200;
    const unsigned int B = 512;
    RN* table = new RN[N];
    radixnode** ptable = new radixnode*[N];
    
    for(unsigned int i=0;i<N;i++)
      ptable[i] = &(table[i]);
    
    
    for(unsigned int i=0;i<N;i++){
      table[i].key.resize(B);
      
      for(unsigned int j=0;j<B;j++){
	if(rand() > (RAND_MAX/2))
	  table[i].key.set(j, true);
	else
	  table[i].key.set(j, false);
      }
      
      if(rand() > (RAND_MAX/2))
	table[i].label = "replicated";
      else
	table[i].label = "not replicated";
    }
    
    
    // sorts data
    fast_gradix sorter;
    sorter.set(ptable, N);
    
    
    if(sorter.sort()){
      // checks if list is ordered
      for(unsigned int i=1;i<N;i++){
	if(((RN*)ptable[i-1])->key.to_integer() < ((RN*)ptable[i])->key.to_integer()){
	  std::cout << "ERROR: LIST IS INCORRECT ORDER (loc: " << i << ")\n";
	  test_ok = false;
	  break;
	}
      }
    }
    else{
      std::cout << "ERROR: RADIX SORTER FAILED." << std::endl;
      test_ok = false;
    }
    
    delete[] table;
    delete[] ptable;
    
    if(test_ok)
      std::cout << "TESTS OK." << std::endl;
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  }
  
}


/********************************************************************************/

void test_fast_radixv()
{
  try{
    std::cout << "FAST RADIX SORT TEST FOR STD::VECTOR<T>" << std::endl;
    std::cout << std::dec;
    
    // numbers test
    {
      // std::cout << "integer" << std::endl;
      
      std::vector<unsigned int> numbers;
      
      for(unsigned int i=0;i<100;i++)
	numbers.push_back( rand() % 10000 );
      
      // std::cout << "start sorting..." << std::endl;
      
      fast_radixv<unsigned int> integer_sorter;
      unsigned int zero = 0;
      unsigned int mask = 1 << (sizeof(unsigned int)*8 - 1);
      integer_sorter.sort(numbers, mask, zero);
      
      // std::cout << "sorting done.." << std::endl;
      
      // checks sorting is correct
      for(unsigned int i=0;i<100;i++){
	for(unsigned int j=0;j<i;j++){
	  if(numbers[j] > numbers[i]){
	    std::cout << "integer sorting error: " 
		      << "indeces " << j << " " << i << std::endl
		      << "numbers " << numbers[j] << " "
		      << numbers[i] << std::endl;
	    return;
	  }
	}
      }
    }
    
    
    {
      // std::cout << "dynamic_bitset" << std::endl;
      
      std::vector<whiteice::dynamic_bitset> bitsets;
      const unsigned int BSLEN = 33;
      
      for(unsigned int i=0;i<100;i++){
	dynamic_bitset bs;
	bs.resize(BSLEN);
	bs.reset();
	
	for(unsigned int j=0;j<bs.size();j++)
	  if(rand() % 2) bs.set(j);
	
	bitsets.push_back(bs);
      }
      
      // std::cout << "start sorting..." << std::endl;
      
      fast_radixv<dynamic_bitset> bitset_sorter;
      dynamic_bitset zero; zero.resize(BSLEN);
      dynamic_bitset mask; mask.resize(BSLEN);
      zero.reset(); mask.set(BSLEN - 1);
      bitset_sorter.sort(bitsets, mask, zero);
      
      // std::cout << "sorting done.." << std::endl;
      
      // checks sorting is correct
      for(unsigned int i=0;i<100;i++){
	for(unsigned int j=0;j<i;j++){
	  if(bitsets[j].to_integer() > bitsets[i].to_integer()){
	    std::cout << "dynamic_bitset sorting error: "
		      << "indeces " << j << i << std::endl
		      << "numbers " 
		      << bitsets[j] << " " 
		      << bitsets[i] << std::endl;
	    return;
	  }
	}
      }
    }
    
    
    std::cout << "STD::VECTOR<T> RADIX SORT TESTS OK"
	      << std::endl;
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: uncaught exception " << e.what() << std::endl;
  }
}

/********************************************************************************/


void test_rbtree()
{
  try{
    // not done, should work
    printf("RED BLACK TREE TEST\n");
  
    std::vector< augmented_data<int, std::string> > data;    
    data.resize(100);
    
    // creation test
    rbtree< augmented_data<int, std::string> >* A;
    A = new rbtree< augmented_data<int, std::string> >;
    delete A;
    
    data[0].key()  = 1;
    data[0].data() = "push";
    
    data[1].key()  = 2;
    data[1].data() = "the";
    
    data[2].key()  = 3;
    data[2].data() = "limits";
    
    data[3].key()  = -1;
    data[3].data() = "ramson";
    
    data[4].key()  =  10001;
    data[4].data() = "dakata";

    data[5].key()  =  -101;
    data[5].data() = "desert";
    
    data[6].key()  = 12;
    data[6].data() = "push2";
    
    data[7].key()  = 22;
    data[7].data() = "the2";
    
    data[8].key()  = 35;
    data[8].data() = "limits2";
    
    data[9].key()  = -10;
    data[9].data() = "ramson2";
    
    data[10].key()  =  10101;
    data[10].data() = "dakata2";
    
    data[11].key()  =  -102;
    data[11].data() = "desert2";
    
    data[12].key()  = 3754;
    data[12].data() = "limits*";
    
    data[13].key()  = -1965;
    data[13].data() = "ramson1000";
    
    data[14].key()  =  1000175;
    data[14].data() = "dakata213";

    data[15].key()  =  -101234;
    data[15].data() = "desert101";
    
    data[16].key()  = 12532;
    data[16].data() = "remedy";
    
    data[17].key()  = 2253532;
    data[17].data() = "entertainment";
    
    data[18].key()  = 3523;
    data[18].data() = "rainbow";
    
    data[19].key()  = -105;
    data[19].data() = "six sky software";
    
    
    A = new rbtree< augmented_data<int, std::string> >();
    
    for(unsigned int i=0;i<20;i++){
      if(A->insert(data[i]) == false)
	std::cout << "1ERROR: insert failed." << std::endl;
      if((signed)A->size() != (signed)(i+1))
	std::cout << "1Wrong tree size: " << A->size() << std::endl;
      
    }
    
    augmented_data<int,std::string> ad;
    
    ad.key() = 6969696; ad.data() = "";
    if(A->remove(ad) == true)
      std::cout << "2ERROR: removing non-existing key succesfully\n";
    
    ad.key() = 0; ad.data() = "";
    if(A->remove(ad) == true)
      std::cout << "3ERROR: removing non-existing key succesfully\n";
    
    ad.key() = -103; ad.data() = "";
    if(A->remove(ad) == true)
      std::cout << "4ERROR: removing non-existing key succesfully\n";
    
    // search tests
    for(unsigned int i=0;i<20;i++){
      data[i+50].key() = data[i].key();
      data[i+50].data() = "";
    }
    
    for(unsigned int i=0;i<20;i++){
      if(A->search(data[i+50]) == false)
	std::cout << "5ERROR: searching value: " << data[i+50].key() << " failed\n";
      
      if(data[i+50].data() != data[i].data() && i != 11)
	std::cout << "6ERROR: bad accompaning data value: " 
		  << data[i+50].data() << " != "
		  << data[i].data() << "  | i = " << i << std::endl;
    }
    
    
    for(unsigned int i=0;i<20;i++){
      std::string str = data[i].data();
      data[i].data() = "";
      
      if(A->remove(data[i]) == false)
	std::cout << "7ERROR: remove failed." << std::endl;
      if((signed)A->size() != (signed)(19-i))
	std::cout << "8Wrong tree size: " << A->size() << std::endl;
      
      if(data[i].data() == str)
	std::cout << "9Deleted data not retrieved correctly. Found  '" 
		  << str << "'  instead (expected '" 
		  << data[i].data() << "') , i = " << i << std::endl;
		  
    }
        
    delete A;

    std::cout << "RANDOM test" << std::endl;
    
    {
      std::vector<int> data;
      rbtree<int>* B;      
      B = new rbtree<int>();      
      data.resize(2048);
      
      std::vector<int>::iterator i;
      i = data.begin();
      
      while(i != data.end()){
	(*i) = rand() % 8192;
	if(B->insert(*i) == false)
	  std::cout << "inserting data to red black tree failed." 
		    << std::endl;	
	i++;
      }
      
      i = data.begin();
      while(i != data.end()){
	if(B->search((*i)) == false)
	  std::cout << "searching for data: " 
		    << *i << " failed." << std::endl;
	  
	if(B->remove((*i)) == false)
	  std::cout << "removing data: "
		    << *i << " failed." << std::endl;
	  
	i++;
      }
      
      delete B;
    }
    
    std::cout << "OK" << std::endl;
  }
  catch(std::exception& e){
    std::cout << "1000ERROR: uncaught exception " << e.what() << std::endl;
  }
}


/********************************************************************************/

void test_dataset()
{
  printf("DATASET TESTS\n");

  using namespace whiteice::math;
  
  {
    dataset< blas_real<float> >* A;
    A = new dataset< blas_real<float> >(10);
    delete A;
    
    A = new dataset< blas_real<float> >(10);
    
    std::vector<math::vertex< blas_real<float> > > data;
    data.resize(100);
    
    for(unsigned int i=0;i<data.size();i++){
      data[i].resize(10);
      for(unsigned int j=0;j<data[i].size();j++)
	data[i][j] = ((float)rand())/((float)RAND_MAX);
    }
    
    std::vector<bool> bresults;
    std::vector<bool> bwanted;
    std::string test_str("0123456789");
    std::vector<std::string> test_strs;
    for(unsigned int i=0;i<10;i++)
      test_strs.push_back(test_str);

    // positive tests, doesn't test functionality
    // tests function calls works (force compilation)
    
    // add() tests
    
    bresults.push_back(A->add(data[0]));   bwanted.push_back(true); // 0
    bresults.push_back(A->add(data));      bwanted.push_back(true); // 1
    bresults.push_back(A->add(test_str));  bwanted.push_back(true); // 2
    bresults.push_back(A->add(test_strs)); bwanted.push_back(true); // 3
    
    A->begin(); A->end();
    (*A)[A->size(0)-1];
    if(A->dimension(0) != 10) printf("ERROR: BAD DIMENSION\n");
    
    bresults.push_back(A->preprocess()); bwanted.push_back(true);   // 4
    bresults.push_back(A->repreprocess()); bwanted.push_back(true); // 5
    bresults.push_back(A->preprocess(dataset<float>::dnSoftMax)); 
    bwanted.push_back(true); // softmax requires data has been normalized to [-1,1] range or something N(0,1)    // 6

    bresults.push_back(A->preprocess(data[0])); bwanted.push_back(true); // 7
    bresults.push_back(A->preprocess(data)); bwanted.push_back(true);    // 8

    bresults.push_back(A->invpreprocess(data)); bwanted.push_back(true); // 9
    bresults.push_back(A->invpreprocess(data[0])); bwanted.push_back(true); // 10
    
    // checks responses
    for(unsigned int i=0;i<bresults.size();i++){
      if(bresults[i] != bwanted[i]){
	printf("ERROR - WRONG RESPONSE, CASE %d\n", i);
	printf("RESULT: %d, WANTED: %d\n",
	       (unsigned int)bresults[i], 
	       (unsigned int)bwanted[i]);
      }
    }
    
    delete A;
  }
  
  
  // save and loading test
  {
    dataset< blas_real<float> >* A = 0;
    std::vector<math::vertex< blas_real<float> > > data;
    data.resize(100);
    
    for(unsigned int i=0;i<data.size();i++){
      data[i].resize(10);
      for(unsigned int j=0;j<data[i].size();j++)
	data[i][j] = ((float)rand())/((float)RAND_MAX);
    }
    
    
    A = new dataset< blas_real<float> >(10);
    
    if(A->add(data, true) == false){
      std::cout << "dataset error: adding new data failed." << std::endl;
      return;
    }

    if(A->size(0) != data.size()){
    	std::cout << "dataset error: incorrect size after add()" << std::endl;
    	return;
    }
    

    A->preprocess();


    if(A->save("dataset.bin") == false){
      std::cout << "dataset error: data saving failed." << std::endl;
      return;
    }

    for(unsigned int i=0;i<A->size(0);i++){
      for(unsigned int j=0;j<data[i].size();j++){
	data[i][j] = (*A)[i][j];
      }
    }
    
    delete A;
    
    A = new dataset< blas_real<float> >(10);
    
    if(A->load("dataset.bin") == false){
      std::cout << "dataset error: data loading failed." << std::endl;
      return;
    }
    
    
    for(unsigned int i=0;i<A->size(0);i++){
      for(unsigned int j=0;j<data[i].size();j++){
	if((*A)[i][j] != data[i][j]){
	  std::cout << "dataset error: data corruption" << std::endl;
	  j = data[i].size();
	  i = 100;
	  return;
	}
      }
    }
    
    
    delete A;
    
    printf("DATASET BASIC SAVE&LOAD() IS OK\n");
    fflush(stdout);

  }
  
  
  
  // multicluster dataset tests
  // added to version 1 
  // (other tests also work with dataset version 0)
  {
    dataset< blas_real<double> > data;
    
    // create N>10 clusters, sets N params
    
    const unsigned int N = (rand() % 32) + 12;
    std::vector<std::string> names;
    std::vector<unsigned int> dims;
    std::vector<std::string> snames;

    try{
      
    for(unsigned int i=0;i<N;i++){
      std::string tmp;
      tmp.resize(4 + rand() % 16);
      
      for(unsigned int j=0;j<tmp.length();j++){
	char ch;
	do{ ch = rand() % 256; }while(!isalpha(ch));
	tmp[j] = ch;
      }
      
      unsigned int d = (rand() % 32) + 2;
      
      if(data.createCluster(tmp, d) == false){
	std::cout << "Creating cluster failed: "
		  << "name : " << tmp 
		  << " , dim: " << d << std::endl;
	return;
      }
      
      dims.push_back(d);
      names.push_back(tmp);
    }
    
    
    if(data.getNumberOfClusters() != N){
      std::cout << "dataset::getNumberOfClusters() returned bad value"
		<< std::endl;
      return;
    }
    
    // checks params are ok
    // tests: size(), dimension(), getCluster(), getClusterNames()
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      if(data.size(i) != 0){
	std::cout << "dataset error: non-zero initial cluster size"
		  << std::endl;
	return;
      }
      
      if(data.dimension(i) != dims[i]){
	std::cout << "dataset error: bad dimension for new cluster"
		  << std::endl
		  << "cluster " << i << " : " << dims[i] << " != "
		  << data.dimension(i) << std::endl;
	return;
      }
      
      if(data.getCluster(names[i]) != i){
	std::cout << "dataset error: couldn't find cluster based on its name"
		  << std::endl
		  << "cluster " << i << " : '" << names[i] << "'"
		  << std::endl
		  << "call returns: " << data.getCluster(names[i])
		  << std::endl;
	return;
      }
    }
        
    
    {
      if(data.getClusterNames(snames) == false){
	std::cout << "dataset error: getClusterNames() call failed."
		  << std::endl;
	return;
      }
      
      if(snames.size() != names.size()){
	std::cout << "dataset error: number of names returned by getClusterNames() is incorrect"
		  << std::endl;
	return;
      }
      
      
      for(unsigned int j=0;j<snames.size();j++){
	if(snames[j] != snames[j]){
	  std::cout << "dataset error: wrong name in namelist"
		    << std::endl;
	  return;
	}
      }
    }
     
    {
      // remove 2 random clusters
      
      for(unsigned int i=0;i<2;i++){
	unsigned int index = 1 + rand() % (data.getNumberOfClusters() - 1);
	
	if(data.removeCluster(index) == false){
	  std::cout << "dataset error: cluster removal failed (1)." << std::endl;
	  std::cout << "cluster index: " << index << std::endl;
	  std::cout << "cluster size : " << data.getNumberOfClusters() << std::endl;
	  return;
	}
	
	std::vector<unsigned int>::iterator i_iter = dims.begin();
	std::vector<std::string>::iterator  s_iter = snames.begin();
	
	for(unsigned int j=0;j<index;j++){
	  i_iter++;
	  s_iter++;
	}
	
	dims.erase(i_iter);
	snames.erase(s_iter);
      }
    }

    }
    catch(std::exception& e){
      std::cout << "ERROR during: create N>10 clusters, sets N params calls." << std::endl;
      std::cout << "Exception: " << e.what() << std::endl;
      std::cout << std::flush;
      fflush(stdout);
    }
    
    
    //////////////////////////////////////////////////
    // add [0,K] data to each cluster

    try{
    
    std::vector<unsigned int> datasizes;
    
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      unsigned int K = 100 + rand() % 100;
      
      math::vertex< blas_real<double> > v;
      std::vector< math::vertex< blas_real<double> > > grp;
      v.resize(data.dimension(i));
      grp.resize(K/4 + 1);
      
      for(unsigned int k=0;k<grp.size();k++){
	grp[k].resize(data.dimension(i));
	for(unsigned l=0;l<data.dimension(i);l++)
	  grp[k][l] = (rand()/((float)RAND_MAX));
      }
      
      
      for(unsigned l=0;l<data.dimension(i);l++)
	v[l] = (rand()/((float)RAND_MAX));
      
      // add(vertex)
      for(unsigned int k=0;k<(K/2);k++)
	data.add(i, v);
      
      // add(vector<vertex>)
      for(unsigned int k=0;k<(K/2);k++)
	data.add(i, grp);
      
      datasizes.push_back(data.size(i));
    }
    
    
    // remove 2 random clusters
        
    {
      for(unsigned int i=0;i<2;i++){
	unsigned int index = 1 + rand() % (data.getNumberOfClusters() - 1);
	
	if(data.removeCluster(index) == false){
	  std::cout << "dataset error: cluster removal failed (2)." << std::endl;
	  std::cout << "cluster index: " << index << std::endl;
	  std::cout << "cluster size : " << data.getNumberOfClusters() << std::endl;
	  
	  return;
	}
	
	std::vector<unsigned int>::iterator i_iter = dims.begin();
	std::vector<std::string>::iterator  s_iter = snames.begin();
	std::vector<unsigned int>::iterator i2_iter = datasizes.begin();
	
	for(unsigned int j=0;j<index;j++){
	  i_iter++;
	  s_iter++;
	  i2_iter++;
	}
	
	
	datasizes.erase(i2_iter);
	dims.erase(i_iter);
	snames.erase(s_iter);
      }
    }
    
    
    
    //////////////////////////////////////////////////
    // check parameters are still correct
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      if(data.size(i) != datasizes[i]){
	std::cout << "dataset error: non-zero initial cluster size"
		  << std::endl;
	return;
      }
      
      if(data.dimension(i) != dims[i]){
	std::cout << "dataset error: bad dimension for new cluster"
		  << std::endl;
	return;
      }
      
      if(data.getCluster(snames[i]) != i){
	std::cout << "dataset error: couldn't find cluster based on its name (2)"
		  << std::endl;
	return;
      }
    }
    

    {
      std::vector<std::string> cnames;
      if(data.getClusterNames(cnames) == false){
	std::cout << "dataset error: getClusterNames() call failed."
		  << std::endl;
	return;
      }
      
      if(cnames.size() != snames.size()){
	std::cout << "dataset error: number of names returned by getClusterNames()"
		  << " is incorrect."
		  << std::endl;
	return;
      }
      
      
      for(unsigned int j=0;j<snames.size();j++){
	if(cnames[j] != snames[j]){
	  std::cout << "dataset error: wrong name in namelist"
		    << std::endl;
	  return;
	}
      }
    }

    }
    catch(std::exception& e){
      std::cout << "ERROR during: add [0,K] data to each cluster calls." << std::endl;
      std::cout << "Exception: " << e.what() << std::endl;
      std::cout << std::flush;
      fflush(stdout);
    }
    

    try{
    
    // check preprocess is ok
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      dataset< blas_real<double> >::data_normalization dn;
      dn = (dataset< blas_real<double> >::data_normalization)(rand()%3);
      
      if(data.size(i) > 2*data.dimension(i)){
	// there is enough data for all preprocessing methods

	if(data.preprocess(i,dn) == false){
	  std::cout << "dataset error: preprocessing cluster "
		    << i << " using method " << dn
		    << " failed." 
		    << " cluster size is "
		    << data.size(i) << "." << std::endl;
	  
	  
	  std::cout << "normalization method: ";
	  
	  if(dn == dataset< blas_real<double> >::dnMeanVarianceNormalization){
	    std::cout << "dnMeanVarianceNormalization" << std::endl;
	  }
	  else if(dn == dataset< blas_real<double> >::dnSoftMax){
	    std::cout << "dnSoftMax" << std::endl;
	  }
	  else if(dn == dataset< blas_real<double> >::dnCorrelationRemoval){
	    std::cout << "dnCorrelationRemoval" << std::endl;
	  }
	  else if(dn == dataset< blas_real<double> >::dnLinearICA){
	    std::cout << "dnLinearICA" << std::endl;
	  }	  
	  else{
	    std::cout << "unknown method (should be error)" << std::endl;
	  }
	  
	  return;
	}
	
      }
    }
    
    }
    catch(std::exception& e){
      std::cout << "Exception happended during preprocess(cluster) calls." << std::endl;
      std::cout << "Exception: " << e.what() << std::endl;
      std::cout << std::flush;
      fflush(stdout);
    }

    
    try{
    
    // check invpreprocess(preprocess(x)) = x
    std::cout << "START invpreprocess(preprocess(x)) == x test." << std::endl;
    std::cout << std::flush;
    fflush(stdout);
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      math::vertex< blas_real<double> > v, u, w;

      ////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////
#if 0
      // prints cluster debugging info
      std::cout << std::endl;
      std::cout << "Cluster: " << i << std::endl;
      std::cout << "Cluster size: " << data.size(i) << std::endl;
      std::cout << "Cluster dimensions: " << data.dimension(i) << std::endl;
      
#endif
      
      ////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////
      
      v.resize(data.dimension(i));
      
      for(unsigned int n=0;n<v.size();n++)
	v[n] = rand()/((float)RAND_MAX);
      
      u = v;
      w = v;

      if(data.preprocess(i, u) == false){
	std::cout << "dataset error: preprocess vector of cluster "
		  << i << std::endl;
	std::cout << std::flush;
	return;
      }

      if(data.invpreprocess(i, u) == false){
	std::cout << "dataset error: invpreprocess of vector failed."
		  << " ( " << i << " cluster)" << std::endl;
	std::cout << std::flush;
	return;
      }

      v -= u;
      
      if(v.norm() > 0.1){
	std::cout << "dataset error: invpreprocess(preprocess(x)) == x "
		  << "(" << i << "/" << data.getNumberOfClusters() << " cluster)"
		  << std::endl;
	
	std::cout << "original = " << w << std::endl;
	std::cout << "invpreprocess(process(x)) = " << u << std::endl;
	std::cout << "delta [error] = " << v << std::endl;
	std::cout << std::flush;
	fflush(stdout);

	
	std::cout << "FIXME: PCA preprocessing is known to be buggy!" << std::endl;
	
	std::vector<dataset< blas_real<double> >::data_normalization> pp;
	
	if(data.getPreprocessings(i, pp)){
	  std::cout << "Preprocessings: " << std::endl;
	  std::cout << std::flush;
	  fflush(stdout);
	  
	  for(std::vector<dataset< blas_real<double> >::data_normalization>::iterator 
		j = pp.begin(); j != pp.end(); j++){
	    // mean-variance norm
	    if(*j == dataset< blas_real<double> >::dnMeanVarianceNormalization){
	      std::cout << "mean-variance normalization" << std::endl;
	    }
	    else if(*j == dataset< blas_real<double> >::dnSoftMax){ // soft-max
	      std::cout << "soft-max normalization" << std::endl;
	    }
	    else if(*j == dataset< blas_real<double> >::dnCorrelationRemoval){ // PCA
	      std::cout << "correlation-removal normalization" << std::endl;
	    }
	    else if(*j == dataset< blas_real<double> >::dnLinearICA){ // ICA
	      std::cout << "independent components (ICA) normalization" << std::endl;
	    }
	  }
	  
	  std::cout << std::flush;
	  fflush(stdout);
	}
	
	
	// show diagnostics of this cluster
	// data.diagnostics(i, true);
	
	// return;
      }
      else{
	std::cout << "dataset::invpreprocess(preprocess(x)) == x. Good." << std::endl;
	std::cout << std::flush;
	fflush(stdout);
      }
    }

    std::cout << "END invpreprocess(preprocess(x)) == x test." << std::endl;
    std::cout << std::flush;
    fflush(stdout);

    }
    catch(std::exception& e){
      std::cout << "Exception happended during invprocess(process(x)) == x tests." << std::endl;
      std::cout << "Exception: " << e.what() << std::endl;
      std::cout << std::flush;
      fflush(stdout);
    }

    
    // test removal, access to unexisting cluster fails
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      if(data.removeCluster(data.getNumberOfClusters() + rand() % 100)){
	std::cout << "dataset error: removal of unexisting cluster is ok"
		  << std::endl;
	return;
      }
    }
    
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      for(unsigned int j=0;j<data.size(i);j++){
	try{
	  data.access(i, data.size(i) + (rand()%100));
	  std::cout << "dataset error: access to unexisting data was ok"
		    << std::endl;
	  return;
	}
	catch(std::exception& e){ }
      }
      
      try{
	data.access(data.getNumberOfClusters() + (rand()%100),
		    rand()&data.size(i));
	std::cout << "dataset error: access to unexisting cluster was ok"
		  << std::endl;
	return;
      }
      catch(std::exception& e){ }
    }
    
    
    // check adding of badly formated data fails (use all add() calls)
    // add N random clusters (use different add()s for adding all)
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      math::vertex< blas_real<double> > v;
      std::vector< math::vertex< blas_real<double> > > grp;
      
      v.resize(data.dimension(i) + rand()%100 + 1);
      if(data.add(i, v)){
	std::cout << "dataset error: adding ill formated data is ok"
		  << std::endl;
	return;
      }
      
      grp.resize(rand()%100 + 1);
      
      for(unsigned j=0;j<grp.size();j++){
	v.resize(data.size(i) + rand()%100 + 1);
	grp[j] = v;
      }
      
      grp[0].resize(data.size(i));
      
      if(data.add(i, grp)){
	std::cout << "dataset error: adding set of ill formated data is ok"
		  << std::endl;
	return;
	
      }
    }
    
    
    // make copy of multicluster dataset (compare)
    dataset< blas_real<double> >* copy;
    try{
      copy = new dataset< blas_real<double> >(data);
    }
    catch(std::exception& e){
      std::cout << "dataset error: creating copy of dataset failed: "
		<< e.what() << std::endl;
      return;
    }
    
    if(copy->getNumberOfClusters() != data.getNumberOfClusters()){
      std::cout << "dataset error: bad copy: number of clusters"
		<< std::endl;
      return;
    }
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      if(data.size(i) != copy->size(i)){
	std::cout << "dataset error: bad copy: cluster size"
		  << std::endl;
	return;
      }
      
      
      if(data.dimension(i) != copy->dimension(i)){
	std::cout << "dataset error: bad copy: cluster dimension mismatch"
		  << std::endl
		  << "cluster " << i << " : " 
		  << data.getName(i) << " != "
		  << copy->dimension(i)
		  << std::endl;
	return;
      }
      
      
      if(data.getName(i) != copy->getName(i)){
	std::cout << "dataset error: bad copy: cluster name mismatch"
		  << std::endl
		  << "cluster " << i << " : " 
		  << data.getName(i) << " != "
		  << copy->getName(i)
		  << std::endl;
      }
      
      
      dataset< blas_real<double> >::iterator a = data.begin(i);
      dataset< blas_real<double> >::iterator b = copy->begin(i);
      
      while(a != data.end(i) && b != copy->end(i)){
	if(*a != *b){
	  std::cout << "dataset error: bad copy: cluster vector"
		    << std::endl;
	  return;
	}
	
	a++;
	b++;
      }
      
      if(a != data.end(i) || b != copy->end(i)){
	std::cout << "dataset error: bad copy: real cluster size"
		  << std::endl;
	return;
      }
    }
    
    
    
    // save multicluster dataset
    std::string filename = "dataset1file.bin";
    
    if(!data.save(filename)){
      std::cout << "dataset error: file saving failed: "
		<< filename << std::endl;
      return;
    }
    
    
    // removes clusters one by one
    while(data.getNumberOfClusters() > 1){
      data.removeCluster(rand()%data.getNumberOfClusters());
    }
    
    
    // loads multicluster dataset
    
    if(!data.load(filename)){
      std::cout << "dataset error: file loading failed: "
		<< filename << std::endl;
      return;
    }
    
    
    // makes full check that everything is as it should be
    
    if(copy->getNumberOfClusters() != data.getNumberOfClusters()){
      std::cout << "dataset error: loading: number of clusters"
		<< std::endl;
      return;
    }
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      if(data.size(i) != copy->size(i)){
	std::cout << "dataset error: loading: cluster size mismatch"
		  << std::endl
		  << "cluster " << i << " : " 
		  << data.getName(i) << " != "
		  << copy->size(i)
		  << std::endl;
	return;
      }

      if(data.dimension(i) != copy->dimension(i)){
	std::cout << "dataset error: loading: cluster dimension mismatch"
		  << std::endl
		  << "cluster " << i << " : " 
		  << data.getName(i) << " != "
		  << copy->dimension(i)
		  << std::endl;
	return;
      }
      
      if(data.getName(i) != copy->getName(i)){
	std::cout << "dataset error: loading: cluster name mismatch"
		  << std::endl
		  << "cluster " << i << " : " 
		  << data.getName(i) << " != "
		  << copy->getName(i)
		  << std::endl;
      }
      
      
      dataset< blas_real<double> >::iterator a = data.begin(i);
      dataset< blas_real<double> >::iterator b = copy->begin(i);
      unsigned int counter = 0;
      
      while(a != data.end(i) && b != copy->end(i)){
	math::vertex< blas_real<double> > delta(*b);
	delta -= *a;
	
	if(delta.norm() > 0.0001){
	  std::cout << "dataset error: loading: cluster vector "
		    << "cluster " << i << " : data " << counter
		    << std::endl;
	  std::cout << "original: " << (*b) << std::endl;
	  std::cout << "save&loaded: " << (*a) << std::endl;
	  std::cout << "|delta|: " << delta.norm() << std::endl;
	  
	  return;
	}
	
	a++;
	b++;
	counter++;
      }
      
      if(a != data.end(i) || b != copy->end(i)){
	std::cout << "dataset error: loading: real cluster size"
		  << std::endl;
	return;
      }
    }
    
    // free's copy
    delete copy;
    copy = 0;
    
    
    // checks loading of unexisting file fails.
    
    if(data.load("rqr0q2348349249___Vdffkl.rwreAop0")){
      std::cout << "dataset error: loading of unexisting file is ok."
		<< std::endl;
      return;
    }
  }


  // dataset: test exportAscii() and importAscii() implementations
  {
    std::cout << "DATASET TESTING exportAscii() and importAscii()" << std::endl;

    whiteice::dataset< blas_real<float> > test, loaded;
    const unsigned int DATADIM = 10;

    test.createCluster("test cluster with a longer name than usual", DATADIM);
    
    for(unsigned int i=0;i<1000;i++){
      whiteice::math::vertex< blas_real<float> > v(test.dimension(0));

      for(unsigned int j=0;j<v.size();j++){
	v[j] = (float)rand()/((float)RAND_MAX) - 0.5f;
      }

      if(test.add(0, v) == false){
	std::cout << "ERROR cannot add vertex to dataset. Index: " << i << std::endl;
	break;
      }
    }

    std::string asciiFilename = "exportedData-test.ds";

    if(test.exportAscii(asciiFilename, 0, true) == false){ // writes headers to ASCII file
      std::cout << "ERROR: cannot export data to ascii file" << std::endl;
    }
    else{
      std::cout << "dataset::exportAscii() successful." << std::endl;
    }

    if(loaded.importAscii(asciiFilename) == false){
      std::cout << "ERROR: cannot import ASCII data from file," << std::endl;
    }
    else{
      std::cout << "dataset::importAscii() successful." << std::endl;
    }
    
    if(loaded.getNumberOfClusters() <= 0){
      std::cout << "ERROR: zero getNumberOfClusters() after importAscii()." << std::endl;
    }
    else{
      std::cout << "dataset::getNumberOfClusters() is ok after importAscii()." << std::endl;
    }

    if(loaded.dimension(0) != test.dimension(0)){
      std::cout << "ERROR: data (3)" << std::endl;
    }
    else{
      std::cout << "dataset::dimension(0) match after importAscii()." << std::endl;
    }

    if(loaded.size(0) != test.size(0)){
      std::cout << "ERROR: cluster size mismatch after importAscii() (4)" << std::endl;
    }
    else{
      std::cout << "dataset::size() match after importAscii()." << std::endl;
    }

    {
      bool error = false;
      
      for(unsigned int j=0;j<loaded.size(0);j++){
	auto l = loaded[j];
	auto t = test[j];
	
	auto delta = l - t;
	
	if(delta.norm() > 0.01f){
	  printf("ERROR: data corrupted in importAscii(exportAscii(data)). Index %d. Error: %f\n",
		 j, delta.norm().c[0]);
	  error = true;
	  break;
	}
      }
      
      if(error == false)
	printf("Comparision: importAscii() returns correct data after exportAscii(). Good.\n");
      
    }

    //////////////////////////////////////////////////////////////////////
    // tries to load ASCII file AGAIN (with existing data structure) and checks everything works ok.
    
    if(loaded.importAscii(asciiFilename, 0) == false){
      std::cout << "ERROR: cannot import ASCII data from file (2.1)" << std::endl;
    }
    
    if(loaded.getNumberOfClusters() <= 0){
      std::cout << "ERROR: cannot import ASCII data from file (2.2)" << std::endl;
    }

    if(loaded.dimension(0) != test.dimension(0)){
      std::cout << "ERROR: cannot import ASCII data from file (2.3)" << std::endl;
    }

    if(loaded.size(0) != test.size(0)){
      std::cout << "ERROR: cannot import ASCII data from file (2.4)" << std::endl;
    }

    for(unsigned int j=0;j<loaded.size(0);j++){
      auto l = loaded[j];
      auto t = test[j];

      auto error = l - t;

      if(error.norm() > 0.01f){
	printf("ERROR: data corrupted in importAscii(exportAscii(data)) (2). Index %d. Error: %f\n",
	       j, error.norm().c[0]);
	break;
      }
    }
    
  }
  
  
}


/********************************************************************************/

void test_dataset_complex()
{
  printf("*********************** DATASET TESTS (COMPLEX NUMBERS) [has problems]\n");

  printf("Dataset basic tests\n");
  fflush(stdout);
  
  {
    dataset< math::blas_complex<float> >* A;
    A = new dataset< math::blas_complex<float> >(10);
    delete A;
    
    A = new dataset< math::blas_complex<float> >(10);
    
    std::vector<math::vertex< math::blas_complex<float> > > data;
    data.resize(100);
    
    for(unsigned int i=0;i<data.size();i++){
      data[i].resize(10);
      for(unsigned int j=0;j<data[i].size();j++){
	data[i][j].real( ((float)rand())/((float)RAND_MAX) );
	data[i][j].imag( ((float)rand())/((float)RAND_MAX) );
      }
    }
    
    std::vector<bool> bresults;
    std::vector<bool> bwanted;
    std::string test_str("0123456789");
    std::vector<std::string> test_strs;
    for(unsigned int i=0;i<10;i++)
      test_strs.push_back(test_str);

    // positive tests, doesn't test functionality
    // tests function calls works (force compilation)
    
    // add() tests
    
    bresults.push_back(A->add(data[0]));   bwanted.push_back(true); // 0
    bresults.push_back(A->add(data));      bwanted.push_back(true); // 1
    bresults.push_back(A->add(test_str));  bwanted.push_back(true); // 2
    bresults.push_back(A->add(test_strs)); bwanted.push_back(true); // 3

    A->begin(); A->end();
    (*A)[A->size(0)-1];
    if(A->dimension(0) != 10) printf("ERROR: BAD DIMENSION\n");

    bresults.push_back(A->preprocess()); bwanted.push_back(true);   // 4

    bresults.push_back(A->repreprocess()); bwanted.push_back(true); // 5

    bresults.push_back(A->preprocess(dataset<float>::dnSoftMax)); // 6
    // softmax requires data has been normalized to [-1,1] range or something N(0,1) 
    bwanted.push_back(true); // 6

    bresults.push_back(A->preprocess(data[0])); bwanted.push_back(true); // 7
    bresults.push_back(A->preprocess(data)); bwanted.push_back(true);    // 8

    bresults.push_back(A->invpreprocess(data)); bwanted.push_back(true); // 9
    bresults.push_back(A->invpreprocess(data[0])); bwanted.push_back(true); // 10
    
    // checks responses
    for(unsigned int i=0;i<bresults.size();i++){
      if(bresults[i] != bwanted[i]){
	printf("ERROR - WRONG RESPONSE, CASE %d\n", i);
	printf("RESULT: %d, WANTED: %d\n",
	       (unsigned int)bresults[i], 
	       (unsigned int)bwanted[i]);
      }
    }
    
    delete A;
  }
  
  
  // save and loading test
  printf("Dataset save() and load()ing tests\n");
  fflush(stdout);
  
  {
    dataset< math::blas_complex<float> >* A = 0;
    std::vector<math::vertex< math::blas_complex<float> > > data;
    data.resize(100);
    
    for(unsigned int i=0;i<data.size();i++){
      data[i].resize(10);
      for(unsigned int j=0;j<data[i].size();j++){
	data[i][j].real( ((float)rand())/((float)RAND_MAX) );
	data[i][j].imag( ((float)rand())/((float)RAND_MAX) );
      }
    }
    
    
    A = new dataset< math::blas_complex<float> >(10);
    
    if(A->add(data, true) == false){
      std::cout << "dataset error: adding new data failed." << std::endl;
      return;
    }

    if(A->size(0) != data.size()){
    	std::cout << "dataset error: incorrect size after add()" << std::endl;
    	return;
    }
    

    A->preprocess();


    if(A->save("dataset.bin") == false){
      std::cout << "dataset error: data saving failed." << std::endl;
      return;
    }

    for(unsigned int i=0;i<A->size(0);i++){
      for(unsigned int j=0;j<data[i].size();j++){
	data[i][j] = (*A)[i][j];
      }
    }
    
    delete A;
    
    A = new dataset< math::blas_complex<float> >(10);
    
    if(A->load("dataset.bin") == false){
      std::cout << "dataset error: data loading failed." << std::endl;
      return;
    }
    
    
    for(unsigned int i=0;i<A->size(0);i++){
      for(unsigned int j=0;j<data[i].size();j++){
	if((*A)[i][j] != data[i][j]){
	  std::cout << "dataset error: data corruption" << std::endl;
	  j = data[i].size();
	  i = 100;
	}
      }
    }
    
    
    delete A;
    
    printf("DATASET BASIC SAVE&LOAD() IS OK\n");
    fflush(stdout);

  }
  
  
  
  // multicluster dataset tests
  // added to version 1 
  // (other tests also work with dataset version 0)
  printf("Dataset multicluster dataset tests\n");
  fflush(stdout);
  
  
  {
    dataset< math::blas_complex<double> > data;
    
    // create N>10 clusters, sets N params
    
    const unsigned int N = (rand() % 32) + 12;
    std::vector<std::string> names;
    std::vector<unsigned int> dims;
    std::vector<std::string> snames;

    try{
      
    for(unsigned int i=0;i<N;i++){
      std::string tmp;
      tmp.resize(4 + rand() % 16);
      
      for(unsigned int j=0;j<tmp.length();j++){
	char ch;
	do{ ch = rand() % 256; }while(!isalpha(ch));
	tmp[j] = ch;
      }
      
      unsigned int d = (rand() % 32) + 2;
      
      if(data.createCluster(tmp, d) == false){
	std::cout << "Creating cluster failed: "
		  << "name : " << tmp 
		  << " , dim: " << d << std::endl;
	return;
      }
      
      dims.push_back(d);
      names.push_back(tmp);
    }
    
    
    if(data.getNumberOfClusters() != N){
      std::cout << "dataset::getNumberOfClusters() returned bad value"
		<< std::endl;
      return;
    }
    
    // checks params are ok
    // tests: size(), dimension(), getCluster(), getClusterNames()
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      if(data.size(i) != 0){
	std::cout << "dataset error: non-zero initial cluster size"
		  << std::endl;
	return;
      }
      
      if(data.dimension(i) != dims[i]){
	std::cout << "dataset error: bad dimension for new cluster"
		  << std::endl
		  << "cluster " << i << " : " << dims[i] << " != "
		  << data.dimension(i) << std::endl;
	return;
      }
      
      if(data.getCluster(names[i]) != i){
	std::cout << "dataset error: couldn't find cluster based on its name"
		  << std::endl
		  << "cluster " << i << " : '" << names[i] << "'"
		  << std::endl
		  << "call returns: " << data.getCluster(names[i])
		  << std::endl;
	return;
      }
    }
        
    
    {
      if(data.getClusterNames(snames) == false){
	std::cout << "dataset error: getClusterNames() call failed."
		  << std::endl;
	return;
      }
      
      if(snames.size() != names.size()){
	std::cout << "dataset error: number of names returned by getClusterNames() is incorrect"
		  << std::endl;
	return;
      }
      
      
      for(unsigned int j=0;j<snames.size();j++){
	if(snames[j] != snames[j]){
	  std::cout << "dataset error: wrong name in namelist"
		    << std::endl;
	  return;
	}
      }
    }
     
    {
      // remove 2 random clusters
      
      for(unsigned int i=0;i<2;i++){
	unsigned int index = 1 + rand() % (data.getNumberOfClusters() - 1);
	
	if(data.removeCluster(index) == false){
	  std::cout << "dataset error: cluster removal failed (1)." << std::endl;
	  std::cout << "cluster index: " << index << std::endl;
	  std::cout << "cluster size : " << data.getNumberOfClusters() << std::endl;
	  return;
	}
	
	std::vector<unsigned int>::iterator i_iter = dims.begin();
	std::vector<std::string>::iterator  s_iter = snames.begin();
	
	for(unsigned int j=0;j<index;j++){
	  i_iter++;
	  s_iter++;
	}
	
	dims.erase(i_iter);
	snames.erase(s_iter);
      }
    }

    }
    catch(std::exception& e){
      std::cout << "ERROR during: create N>10 clusters, sets N params calls." << std::endl;
      std::cout << "Exception: " << e.what() << std::endl;
      std::cout << std::flush;
      fflush(stdout);
    }
    
    
    //////////////////////////////////////////////////
    // add [0,K] data to each cluster

    printf("Dataset: add [0,K] data to each cluster tests\n");
    fflush(stdout);

    try{
    
    std::vector<unsigned int> datasizes;
    
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      unsigned int K = 100 + rand() % 100;
      
      math::vertex< math::blas_complex<double> > v;
      std::vector< math::vertex< math::blas_complex<double> > > grp;
      v.resize(data.dimension(i));
      grp.resize(K/4 + 1);
      
      for(unsigned int k=0;k<grp.size();k++){
	grp[k].resize(data.dimension(i));
	for(unsigned l=0;l<data.dimension(i);l++){
	  grp[k][l].real( (rand()/((float)RAND_MAX)) );
	  grp[k][l].imag( (rand()/((float)RAND_MAX)) );
	}
      }
      
      
      for(unsigned l=0;l<data.dimension(i);l++){
	v[l].real( (rand()/((float)RAND_MAX)) );
	v[l].imag( (rand()/((float)RAND_MAX)) );
      }
      
      // add(vertex)
      for(unsigned int k=0;k<(K/2);k++)
	data.add(i, v);
      
      // add(vector<vertex>)
      for(unsigned int k=0;k<(K/2);k++)
	data.add(i, grp);
      
      datasizes.push_back(data.size(i));
    }
    
    
    // remove 2 random clusters
        
    {
      for(unsigned int i=0;i<2;i++){
	unsigned int index = 1 + rand() % (data.getNumberOfClusters() - 1);
	
	if(data.removeCluster(index) == false){
	  std::cout << "dataset error: cluster removal failed (2)." << std::endl;
	  std::cout << "cluster index: " << index << std::endl;
	  std::cout << "cluster size : " << data.getNumberOfClusters() << std::endl;
	  
	  return;
	}
	
	std::vector<unsigned int>::iterator i_iter = dims.begin();
	std::vector<std::string>::iterator  s_iter = snames.begin();
	std::vector<unsigned int>::iterator i2_iter = datasizes.begin();
	
	for(unsigned int j=0;j<index;j++){
	  i_iter++;
	  s_iter++;
	  i2_iter++;
	}
	
	
	datasizes.erase(i2_iter);
	dims.erase(i_iter);
	snames.erase(s_iter);
      }
    }
    
    
    
    //////////////////////////////////////////////////
    // check parameters are still correct
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      if(data.size(i) != datasizes[i]){
	std::cout << "dataset error: non-zero initial cluster size"
		  << std::endl;
	return;
      }
      
      if(data.dimension(i) != dims[i]){
	std::cout << "dataset error: bad dimension for new cluster"
		  << std::endl;
	return;
      }
      
      if(data.getCluster(snames[i]) != i){
	std::cout << "dataset error: couldn't find cluster based on its name (2)"
		  << std::endl;
	return;
      }
    }
    

    {
      std::vector<std::string> cnames;
      if(data.getClusterNames(cnames) == false){
	std::cout << "dataset error: getClusterNames() call failed."
		  << std::endl;
	return;
      }
      
      if(cnames.size() != snames.size()){
	std::cout << "dataset error: number of names returned by getClusterNames()"
		  << " is incorrect."
		  << std::endl;
	return;
      }
      
      
      for(unsigned int j=0;j<snames.size();j++){
	if(cnames[j] != snames[j]){
	  std::cout << "dataset error: wrong name in namelist"
		    << std::endl;
	  return;
	}
      }
    }

    }
    catch(std::exception& e){
      std::cout << "ERROR during: add [0,K] data to each cluster calls." << std::endl;
      std::cout << "Exception: " << e.what() << std::endl;
      std::cout << std::flush;
      fflush(stdout);
    }


    printf("Dataset: preprocess() tests. (PCA will fail for now with complex numbers).\n");
    fflush(stdout);

    try{
    
    // check preprocess is ok
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      dataset< math::blas_complex<double> >::data_normalization dn;
      dn = (dataset< math::blas_complex<double> >::data_normalization)(rand()%3);
      
      if(data.size(i) > 2*data.dimension(i)){
	// there is enough data for all preprocessing methods

	if(data.preprocess(i,dn) == false){
	  std::cout << "dataset error: preprocessing cluster "
		    << i << " using method " << dn
		    << " failed." 
		    << " cluster size is "
		    << data.size(i) << "." << std::endl;
	  
	  
	  std::cout << "normalization method: ";
	  
	  if(dn == dataset< math::blas_complex<double> >::dnMeanVarianceNormalization){
	    std::cout << "dnMeanVarianceNormalization" << std::endl;
	  }
	  else if(dn == dataset< math::blas_complex<double> >::dnSoftMax){
	    std::cout << "dnSoftMax" << std::endl;
	  }
	  else if(dn == dataset< math::blas_complex<double> >::dnCorrelationRemoval){
	    std::cout << "dnCorrelationRemoval" << std::endl;
	  }
	  else if(dn == dataset< math::blas_complex<double> >::dnLinearICA){
	    std::cout << "dnLinearICA" << std::endl;
	  }	  
	  else{
	    std::cout << "unknown method (should be error)" << std::endl;
	  }
	  
	  // return;
	}
	
      }
    }
    
    }
    catch(std::exception& e){
      std::cout << "Exception happended during preprocess(cluster) calls." << std::endl;
      std::cout << "Exception: " << e.what() << std::endl;
      std::cout << std::flush;
      fflush(stdout);
    }

    
    try{
    
    // check invpreprocess(preprocess(x)) == x
    std::cout << "START invpreprocess(preprocess(x)) == x test." << std::endl;
    std::cout << std::flush;
    fflush(stdout);
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      math::vertex< math::blas_complex<double> > v, u, w;

      ////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////

#if 0
      // prints cluster debugging info
      std::cout << std::endl;
      std::cout << "Cluster: " << i << std::endl;
      std::cout << "Cluster size: " << data.size(i) << std::endl;
      std::cout << "Cluster dimensions: " << data.dimension(i) << std::endl;
#endif
      
      ////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////
      
      v.resize(data.dimension(i));
      
      for(unsigned int n=0;n<v.size();n++){
	v[n].real( rand()/((float)RAND_MAX) );
	v[n].imag( rand()/((float)RAND_MAX) );
      }
      
      u = v;
      w = v;

      if(data.preprocess(i, u) == false){
	std::cout << "dataset error: preprocess vector of cluster "
		  << i << std::endl;
	std::cout << std::flush;
	return;
      }

      if(data.invpreprocess(i, u) == false){
	std::cout << "dataset error: invpreprocess of vector failed."
		  << " ( " << i << " cluster)" << std::endl;
	std::cout << std::flush;
	return;
      }

      v -= u;

      
      if(abs(v.norm()) > 0.1){
	std::cout << "dataset error: invpreprocess(preprocess(x)) == x "
		  << "(" << i << "/" << data.getNumberOfClusters() << " cluster)"
		  << std::endl;
	
	std::cout << "original = " << w << std::endl;
	std::cout << "invpreprocess(process(x)) = " << u << std::endl;
	std::cout << "delta [error] = " << v << std::endl;
	std::cout << std::flush;
	fflush(stdout);

	
	std::cout << "FIXME: PCA preprocessing is known to be buggy!" << std::endl;
	
	std::vector<dataset< math::blas_complex<double> >::data_normalization> pp;
	
	if(data.getPreprocessings(i, pp)){
	  std::cout << "Preprocessings: " << std::endl;
	  std::cout << std::flush;
	  fflush(stdout);
	  
	  for(std::vector<dataset< math::blas_complex<double> >::data_normalization>::iterator 
		j = pp.begin(); j != pp.end(); j++){
	    // mean-variance norm
	    if(*j == dataset< math::blas_complex<double> >::dnMeanVarianceNormalization){
	      std::cout << "mean-variance normalization" << std::endl;
	    }
	    else if(*j == dataset< math::blas_complex<double> >::dnSoftMax){ // soft-max
	      std::cout << "soft-max normalization" << std::endl;
	    }
	    else if(*j == dataset< math::blas_complex<double> >::dnCorrelationRemoval){ // PCA
	      std::cout << "correlation-removal normalization" << std::endl;
	    }
	    else if(*j == dataset< math::blas_complex<double> >::dnLinearICA){ // ICA
	      std::cout << "independent components (ICA) normalization" << std::endl;
	    }
	  }
	  
	  std::cout << std::flush;
	  fflush(stdout);
	}
	
	
	// show diagnostics of this cluster
	// data.diagnostics(i, true);
	
	// return;
      }
      else{
	std::cout << "dataset::invpreprocess(preprocess(x)) == x. Good." << std::endl;
	std::cout << std::flush;
	fflush(stdout);
      }
    }

    std::cout << "END invpreprocess(preprocess(x)) == x test." << std::endl;
    std::cout << std::flush;
    fflush(stdout);

    }
    catch(std::exception& e){
      std::cout << "Exception happended during invprocess(process(x)) == x tests." << std::endl;
      std::cout << "Exception: " << e.what() << std::endl;
      std::cout << std::flush;
      fflush(stdout);
    }

    
    // test removal, access to unexisting cluster fails
    printf("Dataset: Test bad calls fails.\n");
    fflush(stdout);
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      if(data.removeCluster(data.getNumberOfClusters() + rand() % 100)){
	std::cout << "dataset error: removal of unexisting cluster is ok"
		  << std::endl;
	return;
      }
    }
    
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      for(unsigned int j=0;j<data.size(i);j++){
	try{
	  data.access(i, data.size(i) + (rand()%100));
	  std::cout << "dataset error: access to unexisting data was ok"
		    << std::endl;
	  return;
	}
	catch(std::exception& e){ }
      }
      
      try{
	data.access(data.getNumberOfClusters() + (rand()%100),
		    rand()&data.size(i));
	std::cout << "dataset error: access to unexisting cluster was ok"
		  << std::endl;
	return;
      }
      catch(std::exception& e){ }
    }
    
    
    // check adding of badly formated data fails (use all add() calls)
    // add N random clusters (use different add()s for adding all)
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      math::vertex< math::blas_complex<double> > v;
      std::vector< math::vertex< math::blas_complex<double> > > grp;
      
      v.resize(data.dimension(i) + rand()%100 + 1);
      if(data.add(i, v)){
	std::cout << "dataset error: adding ill formated data is ok"
		  << std::endl;
	return;
      }
      
      grp.resize(rand()%100 + 1);
      
      for(unsigned j=0;j<grp.size();j++){
	v.resize(data.size(i) + rand()%100 + 1);
	grp[j] = v;
      }
      
      grp[0].resize(data.size(i));
      
      if(data.add(i, grp)){
	std::cout << "dataset error: adding set of ill formated data is ok"
		  << std::endl;
	return;
	
      }
    }
    
    
    // make copy of multicluster dataset (compare)
    dataset< math::blas_complex<double> >* copy;
    try{
      copy = new dataset< math::blas_complex<double> >(data);
    }
    catch(std::exception& e){
      std::cout << "dataset error: creating copy of dataset failed: "
		<< e.what() << std::endl;
      return;
    }
    
    if(copy->getNumberOfClusters() != data.getNumberOfClusters()){
      std::cout << "dataset error: bad copy: number of clusters"
		<< std::endl;
      return;
    }
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      if(data.size(i) != copy->size(i)){
	std::cout << "dataset error: bad copy: cluster size"
		  << std::endl;
	return;
      }
      
      
      if(data.dimension(i) != copy->dimension(i)){
	std::cout << "dataset error: bad copy: cluster dimension mismatch"
		  << std::endl
		  << "cluster " << i << " : " 
		  << data.getName(i) << " != "
		  << copy->dimension(i)
		  << std::endl;
	return;
      }
      
      
      if(data.getName(i) != copy->getName(i)){
	std::cout << "dataset error: bad copy: cluster name mismatch"
		  << std::endl
		  << "cluster " << i << " : " 
		  << data.getName(i) << " != "
		  << copy->getName(i)
		  << std::endl;
      }
      
      
      dataset< math::blas_complex<double> >::iterator a = data.begin(i);
      dataset< math::blas_complex<double> >::iterator b = copy->begin(i);
      
      while(a != data.end(i) && b != copy->end(i)){
	if(*a != *b){
	  std::cout << "dataset error: bad copy: cluster vector"
		    << std::endl;
	  return;
	}
	
	a++;
	b++;
      }
      
      if(a != data.end(i) || b != copy->end(i)){
	std::cout << "dataset error: bad copy: real cluster size"
		  << std::endl;
	return;
      }
    }
    
    
    
    // save multicluster dataset
    printf("Dataset: save multicluster dataset.\n");
    fflush(stdout);
    
    std::string filename = "dataset1file.bin";
    
    if(!data.save(filename)){
      std::cout << "dataset error: file saving failed: "
		<< filename << std::endl;
      return;
    }
    
    
    // removes clusters one by one
    while(data.getNumberOfClusters() > 1){
      data.removeCluster(rand()%data.getNumberOfClusters());
    }
    
    
    // loads multicluster dataset
    
    if(!data.load(filename)){
      std::cout << "dataset error: file loading failed: "
		<< filename << std::endl;
      return;
    }
    
    
    // makes full check that everything is as it should be
    
    if(copy->getNumberOfClusters() != data.getNumberOfClusters()){
      std::cout << "dataset error: loading: number of clusters"
		<< std::endl;
      return;
    }
    
    for(unsigned int i=0;i<data.getNumberOfClusters();i++){
      if(data.size(i) != copy->size(i)){
	std::cout << "dataset error: loading: cluster size mismatch"
		  << std::endl
		  << "cluster " << i << " : " 
		  << data.getName(i) << " != "
		  << copy->size(i)
		  << std::endl;
	return;
      }

      if(data.dimension(i) != copy->dimension(i)){
	std::cout << "dataset error: loading: cluster dimension mismatch"
		  << std::endl
		  << "cluster " << i << " : " 
		  << data.getName(i) << " != "
		  << copy->dimension(i)
		  << std::endl;
	return;
      }
      
      if(data.getName(i) != copy->getName(i)){
	std::cout << "dataset error: loading: cluster name mismatch"
		  << std::endl
		  << "cluster " << i << " : " 
		  << data.getName(i) << " != "
		  << copy->getName(i)
		  << std::endl;
      }
      
      
      dataset< math::blas_complex<double> >::iterator a = data.begin(i);
      dataset< math::blas_complex<double> >::iterator b = copy->begin(i);
      unsigned int counter = 0;

      while(a != data.end(i) && b != copy->end(i)){
	math::vertex< math::blas_complex<double> > delta(*b);
	delta -= *a;

	if(abs(delta.norm()) > 0.001){
	  std::cout << "dataset error: loading: cluster vector "
		    << "cluster " << i << " : data " << counter
		    << std::endl;
	  std::cout << "original: " << (*b) << std::endl;
	  std::cout << "save&loaded: " << (*a) << std::endl;
	  std::cout << "|delta|: " << abs(delta.norm()) << std::endl;
	  
	  return;
	}
	
	a++;
	b++;
	counter++;
      }
      
      if(a != data.end(i) || b != copy->end(i)){
	std::cout << "dataset error: loading: real cluster size"
		  << std::endl;
	return;
      }
    }
    
    // free's copy
    delete copy;
    copy = 0;
    
    
    // checks loading of unexisting file fails.
    printf("Testing loading unexisting file fails..\n");
    fflush(stdout);
    
    if(data.load("rqr0q2348349249___Vdffkl.rwreAop0")){
      std::cout << "dataset error: loading of unexisting file is ok."
		<< std::endl;
      return;
    }
  }


  // dataset: test exportAscii() and importAscii() implementations
  {
    std::cout << "DATASET TESTING exportAscii() and importAscii()" << std::endl;

    whiteice::dataset< math::blas_complex<float> > test, loaded;
    const unsigned int DATADIM = 10;

    test.createCluster("test cluster with a longer name than usual", DATADIM);
    
    for(unsigned int i=0;i<1000;i++){
      whiteice::math::vertex< math::blas_complex<float> > v(test.dimension(0));

      for(unsigned int j=0;j<v.size();j++){
	v[j].real( (float)rand()/((float)RAND_MAX) - 0.5f );
	v[j].imag( (float)rand()/((float)RAND_MAX) - 0.5f );
      }

      if(test.add(0, v) == false){
	std::cout << "ERROR cannot add vertex to dataset. Index: " << i << std::endl;
	break;
      }
    }

    std::string asciiFilename = "exportedData-test.ds";

    if(test.exportAscii(asciiFilename, 0, true) == false){ // writes headers to ASCII file
      std::cout << "ERROR: cannot export data to ascii file" << std::endl;
    }
    else{
      std::cout << "dataset::exportAscii() successful." << std::endl;
    }

    if(loaded.importAscii(asciiFilename) == false){
      std::cout << "ERROR: cannot import ASCII data from file," << std::endl;
    }
    else{
      std::cout << "dataset::importAscii() successful." << std::endl;
    }
    
    if(loaded.getNumberOfClusters() <= 0){
      std::cout << "ERROR: zero getNumberOfClusters() after importAscii()." << std::endl;
    }
    else{
      std::cout << "dataset::getNumberOfClusters() is ok after importAscii()." << std::endl;
    }

    if(loaded.dimension(0) != test.dimension(0)){
      std::cout << "ERROR: data (3)" << std::endl;
    }
    else{
      std::cout << "dataset::dimension(0) match after importAscii()." << std::endl;
    }

    if(loaded.size(0) != test.size(0)){
      std::cout << "ERROR: cluster size mismatch after importAscii() (4)" << std::endl;
    }
    else{
      std::cout << "dataset::size() match after importAscii()." << std::endl;
    }

    {
      bool error = false;
      
      for(unsigned int j=0;j<loaded.size(0);j++){
	auto l = loaded[j];
	auto t = test[j];
	
	auto delta = l - t;
	
	if(abs(delta.norm()) > 0.01f){
	  printf("ERROR: data corrupted in importAscii(exportAscii(data)). Index %d. Error: %f\n",
		 j, delta.norm().c[0]);
	  error = true;
	  break;
	}
      }
      
      if(error == false)
	printf("Comparision: importAscii() returns correct data after exportAscii(). Good.\n");
      
    }

    //////////////////////////////////////////////////////////////////////
    // tries to load ASCII file AGAIN (with existing data structure) and checks everything works ok.
    
    if(loaded.importAscii(asciiFilename, 0) == false){
      std::cout << "ERROR: cannot import ASCII data from file (2.1)" << std::endl;
    }
    
    if(loaded.getNumberOfClusters() <= 0){
      std::cout << "ERROR: cannot import ASCII data from file (2.2)" << std::endl;
    }

    if(loaded.dimension(0) != test.dimension(0)){
      std::cout << "ERROR: cannot import ASCII data from file (2.3)" << std::endl;
    }

    if(loaded.size(0) != test.size(0)){
      std::cout << "ERROR: cannot import ASCII data from file (2.4)" << std::endl;
    }

    for(unsigned int j=0;j<loaded.size(0);j++){
      auto l = loaded[j];
      auto t = test[j];

      auto error = l - t;

      if(abs(error.norm()) > 0.01f){
	printf("ERROR: data corrupted in importAscii(exportAscii(data)) (2). Index %d. Error: %f\n",
	       j, error.norm().c[0]);
	break;
      }
    }
    
  }
  
  
}


/********************************************************************************/


/* copy of test_rbtree() */
void test_binary_tree()
{
  try{
    // not done, should work
    printf("BINARY TREE TEST\n");
  
    std::vector< augmented_data<int, std::string> > data;    
    data.resize(100);
    
    // creation test
    binarytree< augmented_data<int, std::string> >* A;
    A = new binarytree< augmented_data<int, std::string> >;
    delete A;
    
    data[0].key()  = 1;
    data[0].data() = "push";
    
    data[1].key()  = 2;
    data[1].data() = "the";
    
    data[2].key()  = 3;
    data[2].data() = "limits";
    
    data[3].key()  = -1;
    data[3].data() = "ramson";
    
    data[4].key()  =  10001;
    data[4].data() = "dakata";

    data[5].key()  =  -101;
    data[5].data() = "desert";
    
    data[6].key()  = 12;
    data[6].data() = "push2";
    
    data[7].key()  = 22;
    data[7].data() = "the2";
    
    data[8].key()  = 35;
    data[8].data() = "limits2";
    
    data[9].key()  = -10;
    data[9].data() = "ramson2";
    
    data[10].key()  =  10101;
    data[10].data() = "dakata2";
    
    data[11].key()  =  -102;
    data[11].data() = "desert2";

    data[12].key()  = 3754;
    data[12].data() = "limits*";
    
    data[13].key()  = -1965;
    data[13].data() = "ramson1000";
    
    data[14].key()  =  1000175;
    data[14].data() = "dakata213";

    data[15].key()  =  -101234;
    data[15].data() = "desert101";
    
    data[16].key()  = 12532;
    data[16].data() = "remedy";
    
    data[17].key()  = 2253532;
    data[17].data() = "entertainment";
    
    data[18].key()  = 3523;
    data[18].data() = "rainbow";
    
    data[19].key()  = -105;
    data[19].data() = "six sky software";
    
    
    A = new binarytree< augmented_data<int, std::string> >();
    
    for(unsigned int i=0;i<20;i++){
      if(A->insert(data[i]) == false)
	std::cout << "1ERROR: insert failed." << std::endl;
      if((signed)A->size() != (signed)(i+1))
	std::cout << "1Wrong tree size: " << A->size() << std::endl;
      
    }
    
    augmented_data<int,std::string> ad;
    
    ad.key() = 6969696; ad.data() = "";
    if(A->remove(ad) == true)
      std::cout << "2ERROR: removing non-existing key succesfully\n";
    
    ad.key() = 0; ad.data() = "";
    if(A->remove(ad) == true)
      std::cout << "3ERROR: removing non-existing key succesfully\n";
    
    ad.key() = -103; ad.data() = "";
    if(A->remove(ad) == true)
      std::cout << "4ERROR: removing non-existing key succesfully\n";
    
    // search tests
    for(unsigned int i=0;i<20;i++){
      data[i+50].key() = data[i].key();
      data[i+50].data() = "";
    }
    
    for(unsigned int i=0;i<20;i++){
      if(A->search(data[i+50]) == false)
	std::cout << "5ERROR: searching value: " << data[i+50].key() << " failed\n";
      
      if(data[i+50].data() != data[i].data() && i != 11)
	std::cout << "6ERROR: bad accompaning data value: " 
		  << data[i+50].data() << " != "
		  << data[i].data() << "  | i = " << i << std::endl;
    }
    
    
    for(unsigned int i=0;i<20;i++){
      std::string str = data[i].data();
      data[i].data() = "";
      
      if(A->remove(data[i]) == false)
	std::cout << "7ERROR: remove failed." << std::endl;
      if((signed)A->size() != (signed)(19-i))
	std::cout << "8Wrong tree size: " << A->size() << std::endl;
      
      if(data[i].data() == str)
	std::cout << "9Deleted data not retrieved correctly. Found  '" 
		  << str << "'  instead (expected '" 
		  << data[i].data() << "') , i = " << i << std::endl;
		  
    }
        
    delete A;

    std::cout << "RANDOM test" << std::endl;
    
    {
      std::vector<int> data;
      binarytree<int>* B;      
      B = new binarytree<int>();      
      data.resize(2048);
      
      std::vector<int>::iterator i;
      i = data.begin();
      
      while(i != data.end()){
	(*i) = rand() % 8192;
	if(B->insert(*i) == false)
	  std::cout << "inserting data to red black tree failed." 
		    << std::endl;	
	i++;
      }
      
      i = data.begin();
      while(i != data.end()){
	if(B->search((*i)) == false)
	  std::cout << "searching for data: " 
		    << *i << " failed." << std::endl;
	  
	if(B->remove((*i)) == false)
	  std::cout << "removing data: "
		    << *i << " failed." << std::endl;
	  
	i++;
      }
      
      delete B;
    }
    
    std::cout << "OK" << std::endl;
  }
  catch(std::exception& e){
    std::cout << "1000ERROR: uncaught exception " << e.what() << std::endl;
  }
}


/* avl tree tests */
void test_avltree()
{
  printf("AVL TREE TEST\n");
  
  std::vector<int> data;
  data.resize(0);
  
  for(unsigned int i=0;i<200;i++)
    data.push_back(i);
  
  
  // random changes in input order
  {
    int temp;
    for(unsigned int i=0;i<data.size();i++){
      unsigned int i1 = rand() % data.size();
      unsigned int i2 = rand() % data.size();
      temp = data[i1];
      data[i1] = data[i2];
      data[i2] = temp;
    }
  }
  
  
  // creation and simple usage test
  avltree<int>* A;
  
  A = new avltree<int>;
  delete A;
  
  A = new avltree<int>;

  for(unsigned int i=0;i<data.size();i++){
    
    if(A->insert(data[i]) == false)
      std::cout << "1ERROR: insert failed." << std::endl;
    if((signed)A->size() != (signed)(i+1))
      std::cout << "1Wrong tree size: " << A->size() << std::endl;
  }

  
  A->ordered_list();
  A->list();
  
  
  for(int i=100;i<200;i++){
    printf("LIST BEGIN\n");
    A->ordered_list();
    A->list();
    printf("STARTING REMOVAL %d\n", i);
    A->remove(i);
    printf("STOPPING REMOVAL %d\n", i);
    A->ordered_list();
    printf("LIST END\n");
  }
  
  
  if(A->maximum() != 99)
    std::cout << "Wrong maximum value "
	      << A->maximum() << std::endl;
  
  if(A->minimum() != 0)
    std::cout << "Wrong minimum value "
	      << A->minimum() << std::endl;
	      
  
  if(A->size() != 100)
    std::cout << "Wrong list size " 
	      << A->size() << std::endl;
  
  // removes rest of the elements from A
  {
    for(int i=0;i<100;i++){
      printf("LIST BEGIN\n");
      A->ordered_list();
      A->list();
      printf("STARTING REMOVAL %d\n", i);
      A->remove(i);
      printf("STOPPING REMOVAL %d\n", i);
      A->ordered_list();
      printf("LIST END\n");
    }
  }  
  
  delete A;
  
  
#if 0  
  std::vector< augmented_data<int, std::string> > data;    
  data.resize(100);
  
  // creation test
  binarytree< augmented_data<int, std::string> >* A;
  A = new binarytree< augmented_data<int, std::string> >;
  delete A;
  
  data[0].key()  = 1;
  data[0].data() = "push";
  
  data[1].key()  = 2;
  data[1].data() = "the";
  
  data[2].key()  = 3;
  data[2].data() = "limits";
  
  data[3].key()  = -1;
  data[3].data() = "ramson";
  
  data[4].key()  =  10001;
  data[4].data() = "dakata";
  
  data[5].key()  =  -101;
  data[5].data() = "desert";
  
  data[6].key()  = 12;
  data[6].data() = "push2";
  
  data[7].key()  = 22;
  data[7].data() = "the2";
  
  data[8].key()  = 35;
  data[8].data() = "limits2";
  
  data[9].key()  = -10;
  data[9].data() = "ramson2";
  
  data[10].key()  =  10101;
  data[10].data() = "dakata2";
  
  data[11].key()  =  -102;
  data[11].data() = "desert2";
  
  data[12].key()  = 3754;
  data[12].data() = "limits*";
  
  data[13].key()  = -1965;
  data[13].data() = "ramson1000";
  
  data[14].key()  =  1000175;
  data[14].data() = "dakata213";
  
  data[15].key()  =  -101234;
  data[15].data() = "desert101";
  
  data[16].key()  = 12532;
  data[16].data() = "remedy";
    
  data[17].key()  = 2253532;
  data[17].data() = "entertainment";
    
  data[18].key()  = 3523;
  data[18].data() = "rainbow";
  
  data[19].key()  = -105;
  data[19].data() = "six sky software";
    
  
  A = new binarytree< augmented_data<int, std::string> >();
  
  for(unsigned int i=0;i<20;i++){
    if(A->insert(data[i]) == false)
      std::cout << "1ERROR: insert failed." << std::endl;
    if((signed)A->size() != (signed)(i+1))
      std::cout << "1Wrong tree size: " << A->size() << std::endl;
    
  }
  
  augmented_data<int,std::string> ad;
  
  ad.key() = 6969696; ad.data() = "";
  if(A->remove(ad) == true)
    std::cout << "2ERROR: removing non-existing key succesfully\n";
  
  ad.key() = 0; ad.data() = "";
  if(A->remove(ad) == true)
    std::cout << "3ERROR: removing non-existing key succesfully\n";
  
  ad.key() = -103; ad.data() = "";
  if(A->remove(ad) == true)
    std::cout << "4ERROR: removing non-existing key succesfully\n";
  
  // search tests
  for(unsigned int i=0;i<20;i++){
    data[i+50].key() = data[i].key();
    data[i+50].data() = "";
  }
    
  for(unsigned int i=0;i<20;i++){
    if(A->search(data[i+50]) == false)
      std::cout << "5ERROR: searching value: " << data[i+50].key() << " failed\n";
    
    if(data[i+50].data() != data[i].data() && i != 11)
      std::cout << "6ERROR: bad accompaning data value: " 
		<< data[i+50].data() << " != "
		<< data[i].data() << "  | i = " << i << std::endl;
  }
  
  
  for(unsigned int i=0;i<20;i++){
    std::string str = data[i].data();
    data[i].data() = "";
    
    if(A->remove(data[i]) == false)
      std::cout << "7ERROR: remove failed." << std::endl;
    if((signed)A->size() != (signed)(19-i))
      std::cout << "8Wrong tree size: " << A->size() << std::endl;
    
    if(data[i].data() == str)
      std::cout << "9Deleted data not retrieved correctly. Found  '" 
		<< str << "'  instead (expected '" 
		<< data[i].data() << "') , i = " << i << std::endl;
    
  }
  
  delete A;
  
  std::cout << "RANDOM test" << std::endl;
  
  {
    std::vector<int> data;
    binarytree<int>* B;      
    B = new binarytree<int>();      
      data.resize(2048);
      
      std::vector<int>::iterator i;
      i = data.begin();
      
      while(i != data.end()){
	(*i) = rand() % 8192;
	if(B->insert(*i) == false)
	  std::cout << "inserting data to red black tree failed." 
		    << std::endl;	
	i++;
      }
      
      i = data.begin();
      while(i != data.end()){
	if(B->search((*i)) == false)
	  std::cout << "searching for data: " 
		    << *i << " failed." << std::endl;
	
	if(B->remove((*i)) == false)
	  std::cout << "removing data: "
		    << *i << " failed." << std::endl;
	
	i++;
      }
      
      delete B;
  }
  
  std::cout << "OK" << std::endl;
#endif
  
}


/**************************************************/
#if 0

// FIXME make this test to compile


class unique_id_test : whiteice::unique_id
{
public:
  
  unique_id_test(){
    this->create(std::string("list1"), 100);
    this->create(std::string("list2"), 10);
  }
  
  ~unique_id_test(){
    this->free(std::string("list2"));
    this->free(std::string("list1"));
  }
  
  unsigned int get(unsigned int listnum){
    if(listnum == 1){
      return this->get(std::string("list1"));
    }
    else if (listnum == 2){
      return this->get(std::string("list2"));
    }
    else return 0;
  }
  
  bool free(unsigned int listnum, int number){
    if(listnum == 1){
      return this->free(std::string("list1"),
			     number);
    }
    else if (listnum == 2){
      return this->free(std::string("list2"),
			     number);
    }
    else return 0;    
  }
  
};


void test_uniqueid()
{
  // simple unique_id tests
  unique_id_test T;
  bool ok = true;
  
  std::cout << "TESTING unique_id class.\n";
  
  int i = T.get(1);
  int j[11];
  
  for(unsigned int k=0;k<11;k++){
    j[k] = 0;
    j[k] = T.get(2);
  }
  
  bool ri = T.free(1, i);
  
  bool rj[11];
  
  for(unsigned int k=0;k<11;k++){
    rj[k] = false;
    if(j[k])
      rj[k] = T.free(2, j[k]);
  }
  
  if(i == 0){
    std::cout << "UNIQUE_ID TEST: getting single id. FAILED\n";
    ok = false;
  }
  
  for(unsigned int k=0;k<10;k++){
    if(j[k] == 0){
      std::cout << "UNIQUE_ID TEST: getting group of ids (" << k << "). FAILED\n";
      ok = false;
    }
  }
  
  if(j[10] != 0){
    std::cout << "UNIQUE_ID TEST: getting value from empty list possible. FAILED\n";
    ok = false;
  }
  
  if(ri == false){
    std::cout << "UNIQUE_ID TEST: allocating single (free) id failed. FAILED\n";
    ok = false;
  }
  
  for(unsigned int k=0;k<10;k++){
    if(rj[k] == false){
      std::cout << "UNIQUE_ID TEST: freeing group of ids.(" << k << ") FAILED\n";
      ok = false;
    }
  }
  
  if(rj[10] == true){
    std::cout << "UNIQUE_ID TEST: freeing non-existing value possible. FAILED\n";
    ok = false;
  }
  
  if(T.free(1,0) == true){
    std::cout << "UNIQUE_ID TEST: freeing non-existing value possible. FAILED\n";
    ok = false;
  }
  
  if(T.free(1,10000) == true){
    std::cout << "UNIQUE_ID TEST: freeing non-existing value possible. FAILED\n";
    ok = false;
  }
  
  // unlimited list tests
  
  std::cout << "UNIQUE_ID TEST:  WARNING unlimited list features not tested!\n";
  
  
  if(ok)
    std::cout << "UNIQUE ID TESTS PASSED.\n";    
}
#endif

/********************************************************************************/

void conffile_create_good_varname(std::string& str);
void conffile_create_good_string(std::string& str);


void test_conffile()
{
  bool ok = true;
  
  std::cout << "TESTING conffile class.\n";
  
  
  // set(), exists(), get(), clear()
  {
    whiteice::conffile configuration;
  
    //////////////////////////////////////////////////////////////////////
    // negative exist() check
    {
      std::cout << "CONFFILE EXISTS TEST" << std::endl;
      
       if(configuration.exists("") == true){
	 ok = false;
	 std::cout << "exists() test FAILED\n";
       }

       char buf[1000];

       for(unsigned int i=0;i<10;i++){

	 // creates random string
	 const unsigned int L = rand() % 800;

	 for(unsigned int j=0;j<L;j++)
	   buf[j] = rand() & 0xFF;

	 buf[L] = '\0';

	 if(configuration.exists(std::string(buf)) == true){
	   ok = false;
	   std::cout << "exists() random check test FAILED\n";
	 }

       }
    }
    
        

    //////////////////////////////////////////////////////////////////////
    // integer test
    {
      std::cout << "CONFFILE INTEGER TEST" << std::endl;
      
      // generates random data
      
      std::vector< std::vector<int> > values;
      std::vector< std::string > names;
      
      values.resize((rand() % 20) + 1);
      names.resize(values.size());
      
      for(unsigned int i=0;i<values.size();i++){
	values[i].resize((rand() % 20) + 1);
	
	for(unsigned int j=0;j<values[i].size();j++){
	  values[i][j] = (rand() % 7482) - 12000;
	}
	
	names[i].resize((rand() % 20) + 1);
	
	conffile_create_good_varname(names[i]);
      }
      
      
      // puts it into conffile in a random order       
      
      std::vector<int> order;
      order.resize(values.size());
      
      for(unsigned int i=0;i<order.size();i++){
	order[i] = i;
      }
      
      for(unsigned int i=0;i<order.size();i++){
	unsigned int a, b;
	a = rand() % order.size();
	b = rand() % order.size();
	
	std::swap<int>(order[a],order[b]);
      }
      
      for(unsigned int i=0;i<values.size();i++){
	const unsigned int index = order[i];
	
	if(configuration.set(names[index], values[index]) == false){
	  ok = false;
	  std::cout << "conffile integer - setting pair "
		    << "(" << names[index] << " , ";
	  
	  std::cout << "(";
	  
	  for(unsigned int k=0;k<values.size();k++){
	    std::cout << values[index][k] << ",";
	  }
	  
	  std::cout << ") FAILED\n";
	}
      }
      
      
      // positive exists() check
      
      for(unsigned int i=0;i<names.size();i++){
	if(configuration.exists(names[i]) == false){
	  ok = false;
	  std::cout << "conffile integer - name existence check FAILED\n"
		    << "( " << names[i] << " )" << std::endl;
	}
      }
      
      // negative exists() check not done
      
      // positive get() check
      for(unsigned int i=0;i<values.size();i++){
	std::vector<int> tmp;
	
	if(configuration.get(names[i], tmp) == false){
	  ok = false;
	  std::cout << "conffile integer - get() FAILED\n";
	}
	
	if(tmp.size() != values[i].size()){
	  ok = false;
	  std::cout << "conffile integer - get() returned wrong size vector (ERROR).\n";
	}
	
	for(unsigned int j=0;j<tmp.size();j++){
	  if(tmp[j] != values[i][j]){
	    ok = false;
	    std::cout << "conffile integer - get() returned corrupted data (ERROR).\n";
	  }
	}
      }
      
      
      configuration.clear();
      
      // negative get() check
      for(unsigned int i=0;i<values.size();i++){
	std::vector<int> tmp;
	
	if(configuration.get(names[i], tmp) == true){
	  ok = false;
	  std::cout << "conffile integer - get() successful after clear (ERROR)\n";
	}
      }
      
      
    }
    
    
    //////////////////////////////////////////////////////////////////////
    // float test
    {
      std::cout << "CONFFILE FLOAT TEST" << std::endl;
      
      // generates random data
      
      std::vector< std::vector<float> > values;
      std::vector< std::string > names;
      
      values.resize((rand() % 20) + 1);
      names.resize(values.size());
      
      for(unsigned int i=0;i<values.size();i++){
	values[i].resize((rand() % 20) + 1);
	
	for(unsigned int j=0;j<values[i].size();j++){
	  values[i][j] = (rand() % 7482)/7482.0 - 0.5;
	}
	
	names[i].resize((rand() % 20) + 1);
	
	conffile_create_good_varname(names[i]);	
      }
      
      
      // puts it into conffile in a random order       
      
      std::vector<int> order;
      order.resize(values.size());
      
      for(unsigned int i=0;i<order.size();i++){
	order[i] = i;
      }
      
      for(unsigned int i=0;i<order.size();i++){
	unsigned int a, b;
	a = rand() % order.size();
	b = rand() % order.size();
	
	std::swap<int>(order[a],order[b]);
      }
      
      for(unsigned int i=0;i<values.size();i++){
	const unsigned int index = order[i];
	
	if(configuration.set(names[index], values[index]) == false){
	  ok = false;
	  std::cout << "conffile float - setting pair "
		    << "(" << names[index] << " , ";
	  
	  std::cout << "(";
	  
	  for(unsigned int k=0;k<values[index].size();k++){
	    std::cout << values[index][k] << ",";
	  }
	  
	  std::cout << ")";
	  
	  std::cout << " FAILED\n";
	}	 
      }
      
      
      // positive exists() check
      
      for(unsigned int i=0;i<names.size();i++){
	if(configuration.exists(names[i]) == false){
	  ok = false;
	  std::cout << "conffile float - name existence check FAILED\n"
		    << "( " << names[i] << " )" << std::endl;
	}
      }
      
      // negative exists() check not done
      
      // positive get() check
      for(unsigned int i=0;i<values.size();i++){
	std::vector<float> tmp;
	
	if(configuration.get(names[i], tmp) == false){
	  ok = false;
	  std::cout << "conffile float - get() FAILED\n";
	}
	
	if(tmp.size() != values[i].size()){
	  ok = false;
	  std::cout << "conffile float - get() returned wrong size vector (ERROR).\n";
	}
	
	for(unsigned int j=0;j<tmp.size();j++){
	  if(tmp[j] != values[i][j]){
	    ok = false;
	    std::cout << "conffile float - get() returned corrupted data (ERROR).\n";
	  }
	}
      }
      
      
      configuration.clear();
      
      // negative get() check
      for(unsigned int i=0;i<values.size();i++){
	std::vector<float> tmp;
	
	if(configuration.get(names[i], tmp) == true){
	  ok = false;
	  std::cout << "conffile float - get() successful after clear (ERROR)\n";
	}
      }
      
      
    }
    
    
    //////////////////////////////////////////////////////////////////////
    // string test        
    
    {
      std::cout << "CONFFILE STRING TEST" << std::endl;
      
      // generates random data
      
      std::vector< std::vector<std::string> > values;
      std::vector< std::string > names;
      
      values.resize((rand() % 20) + 1);
      names.resize(values.size());
      
      for(unsigned int i=0;i<values.size();i++){
	values[i].resize((rand() % 20) + 1);
	
	for(unsigned int j=0;j<values[i].size();j++){
	  values[i][j].resize((rand() % 20) + 1);

	  conffile_create_good_string(values[i][j]);
	}
	
	names[i].resize((rand() % 20) + 1);

	conffile_create_good_varname(names[i]);	
      }
      
      
      // puts it into conffile in a random order       
      
      std::vector<int> order;
      order.resize(values.size());
      
      for(unsigned int i=0;i<order.size();i++){
	order[i] = i;
      }
      
      for(unsigned int i=0;i<order.size();i++){
	unsigned int a, b;
	a = rand() % order.size();
	b = rand() % order.size();
	
	std::swap<int>(order[a],order[b]);
      }
      
      for(unsigned int i=0;i<values.size();i++){
	const unsigned int index = order[i];
	
	if(configuration.set(names[index], values[index]) == false){
	  ok = false;
	  std::cout << "conffile string - setting pair "
		    << "(" << names[index] << " , ";
	  
	  
	  std::cout << "(";
	    
	  for(unsigned int k=0;k<values[index].size();k++)
	    std::cout << values[index][k] << ",";
	  
	  std::cout << ")";
	  
	  std::cout << ") FAILED\n";
	}	 
      }
      
      
      // positive exists() check
      
      for(unsigned int i=0;i<names.size();i++){
	if(configuration.exists(names[i]) == false){
	  ok = false;
	  std::cout << "conffile string - name existence check FAILED\n"
		    << "( " << names[i] << " )" << std::endl;
	}
      }
      
      // negative exists() check not done
      
      // positive get() check
      for(unsigned int i=0;i<values.size();i++){
	std::vector<std::string> tmp;
	
	if(configuration.get(names[i], tmp) == false){
	  ok = false;
	  std::cout << "conffile string - get() FAILED\n";
	}
	
	if(tmp.size() != values[i].size()){
	  ok = false;
	  std::cout << "conffile string - get() returned wrong size vector (ERROR).\n";
	}
	
	for(unsigned int j=0;j<tmp.size();j++){
	  if(tmp[j] != values[i][j]){
	    ok = false;
	    std::cout << "conffile string - get() returned corrupted data (ERROR).\n";
	  }
	}
      }
      
      
      configuration.clear();
      
      // negative get() check
      for(unsigned int i=0;i<values.size();i++){
	std::vector<std::string> tmp;
	
	if(configuration.get(names[i], tmp) == true){
	  ok = false;
	  std::cout << "conffile string - get() successful after clear (ERROR)\n";
	}
      }
    }
    

    //////////////////////////////////////////////////////////////////////
    // mixed test
    
    {
      // implement after separated tests are passed
      
      
      
    }
    
  }
  
  
  // load(), save()
  {
    std::cout << "CONFFILE SAVE&LOAD TEST" << std::endl;    
    std::cout << "SAVING" << std::endl;
    
    std::vector< std::string > inames;
    std::vector< std::vector<int> > ivalues;
    
    std::vector< std::string > fnames;
    std::vector< std::vector<float> > fvalues;
    
    std::vector< std::string > snames;
    std::vector< std::vector< std::string > > svalues;
    
    // generates random data
    
    inames.resize((rand() % 20) + 1);
    fnames.resize((rand() % 20) + 1);
    snames.resize((rand() % 20) + 1);
    
    ivalues.resize(inames.size());
    fvalues.resize(fnames.size());
    svalues.resize(snames.size());
    
    // creates random data
    {
      std::vector<std::string>::iterator names;
      std::vector< std::vector<int> >::iterator ints;
      std::vector< std::vector<float> >::iterator floats;
      std::vector< std::vector<std::string> >::iterator strings;
      
      // creates random integers
    
      for(names = inames.begin(), ints = ivalues.begin();
	  names != inames.end();
	  names++, ints++)
      {
	names->resize((rand() % 20) + 1);
	
	conffile_create_good_varname(*names);
	
	ints->resize((rand() % 20) + 1);
	
	for(unsigned int i=0;i<ints->size();i++)
	  (*ints)[i] = rand();
      }
      
      
      // creates random floats
      
      for(names = fnames.begin(), floats = fvalues.begin();
	  names != fnames.end();
	  names++, floats++)
      {
	names->resize((rand() % 20) + 1);
	
	do{
	  conffile_create_good_varname(*names);
	  
	  // checks for duplicates
	  bool dupe = false;
	  for(unsigned int i=0;i<inames.size();i++)
	    if(inames[i] == *names)
	      dupe = true;
	  
	  if(dupe) continue;
	  
	  break;
	}
	while(true);
	
	floats->resize((rand() % 20) + 1);
	
	for(unsigned int i=0;i<floats->size();i++)
	  (*floats)[i] = (rand() % 3232)/3232.0 - 0.5;
      }
      
      
      // creates random strings
      
      for(names = snames.begin(), strings = svalues.begin();
	  names != snames.end();
	  names++, strings++)
      {
	names->resize((rand() % 20) + 1);
	
	do{
	  conffile_create_good_varname(*names);
	  
	  // checks for duplicates
	  bool dupe = false;
	  for(unsigned int i=0;i<inames.size();i++)
	    if(inames[i] == *names)
	      dupe = true;
	  
	  for(unsigned int i=0;i<fnames.size();i++)
	    if(fnames[i] == *names)
	      dupe = true;
	  
	  if(dupe) continue;
	  
	  break;
	}
	while(true);
	
	strings->resize((rand() % 20) + 1);
	
	for(unsigned int i=0;i<strings->size();i++)
	  conffile_create_good_string((*strings)[i]);
      }
      
    }
    
    
    // puts it into configuration file
    {
      whiteice::conffile configuration;
      
      for(unsigned int i=0;i<inames.size();i++){
	if(configuration.set(inames[i], ivalues[i]) == false){
	  ok = false;
	  
	  std::cout << "conffile save()/load() - integer data set() FAILURE\n";
	}
      }
      
      for(unsigned int i=0;i<fnames.size();i++){
	if(configuration.set(fnames[i], fvalues[i]) == false){
	  ok = false;
	  
	  std::cout << "conffile save()/load() - float data set() FAILURE\n";
	}
      }
      
      for(unsigned int i=0;i<snames.size();i++)
	if(configuration.set(snames[i], svalues[i]) == false){
	  ok = false;
	  
	  std::cout << "conffile save()/load() - string data set() FAILURE\n";
	}
      
      // checks if "configuration_test.cfg" exists and removes it
      
      {
	struct stat buf;
	
	if(stat("configuration_test.cfg", &buf) != 0){
	  
	  if(errno != ENOENT){ // file does exists
	    ok = false;
	    std::cout << "conffile save()/load() - stat() FAILED\n";
	  }
	}
	else{ // removes existing file
	  if(remove("configuration_test.cfg") != 0){
	    ok = false;
	    std::cout << "conffile save()/load() - remove() FAILED\n";
	  }
	}
      }
	
      
      // saves it
      
      if(configuration.save("configuration_test.cfg") == false){
	ok = false;
	std::cout << "conffile save()/load() - save() FAILURE\n";
      }
      
      configuration.clear();

    }
    
    
    // loads data
    {
      whiteice::conffile configuration;
      
      std::cout << "LOADING" << std::endl;
      
      if(configuration.load("configuration_test.cfg") == false){
	ok = false;
	std::cout << "conffile save()/load() - load() FAILURE\n";
      }

      // removes existing file
      if(remove("configuration_test.cfg") != 0){
	ok = false;
	std::cout << "conffile save()/load() - remove() of existing configuration file FAILED\n";
      }
      
      // checks that values were read correctly from the configuration file
      
      // integers
      for(unsigned int i=0;i<inames.size();i++){
	if(configuration.exists(inames[i]) == false){
	  ok = false;
	  std::cout << "conffile save()/load() - integer exist() check after load FAILED\n";
	}
      }

      // floats
      for(unsigned int i=0;i<fnames.size();i++){
	if(configuration.exists(fnames[i]) == false){
	  ok = false;
	  std::cout << "conffile save()/load() - float exist() check after load FAILED\n";
	}
      }
      
      // strings
      for(unsigned int i=0;i<snames.size();i++){
	if(configuration.exists(snames[i]) == false){
	  ok = false;
	  std::cout << "conffile save()/load() - string exist() check after load FAILED\n";
	}
      }
      
      // negative exist() tests not done
      
      // checks int values were loaded correctly
      for(unsigned int i=0;i<ivalues.size();i++){
	std::vector<int> tmp;
	
	if(configuration.get(inames[i], tmp) == false){
	  ok = false;
	  std::cout << "conffile save()/load() - int get() check after load FAILED\n";
	}
	
	if(tmp.size() != ivalues[i].size()){
	  ok = false;
	  std::cout << "conffile save()/load() - int get() vector size check after load FAILED\n";
	}
	
	for(unsigned int j=0;j<tmp.size();j++){
	  if(ivalues[i][j] != tmp[j]){
	    ok = false;
	    std::cout << "conffile save()/load() - int get() vector mismatch after load (ERROR)\n";
	  }
	}
      }


      // checks float values were loaded correctly
      for(unsigned int i=0;i<fvalues.size();i++){
	std::vector<float> tmp;
	
	if(configuration.get(fnames[i], tmp) == false){
	  ok = false;
	  std::cout << "conffile save()/load() - float get() check after load FAILED\n";
	}
	
	if(tmp.size() != fvalues[i].size()){
	  ok = false;
	  std::cout << "conffile save()/load() - float get() vector size check after load FAILED\n";
	}
	
	for(unsigned int j=0;j<tmp.size();j++){
	  if(fvalues[i][j] != tmp[j]){
	    ok = false;
	    std::cout << "conffile save()/load() - float get() vector mismatch after load (ERROR)\n";
	  }
	}
      }

      
      // checks string values were loaded correctly
      for(unsigned int i=0;i<svalues.size();i++){
	std::vector<std::string> tmp;
	
	if(configuration.get(snames[i], tmp) == false){
	  ok = false;
	  std::cout << "conffile save()/load() - string get() check after load FAILED\n";
	}
	
	if(tmp.size() != svalues[i].size()){
	  ok = false;
	  std::cout << "conffile save()/load() - string get() vector size check after load FAILED\n";
	}
	
	for(unsigned int j=0;j<tmp.size();j++){
	  if(svalues[i][j] != tmp[j]){
	    ok = false;
	    std::cout << "conffile save()/load() - string get() vector mismatch after load (ERROR)\n";
	    std::cout << "   got: " << tmp[j] << std::endl;
	    std::cout << "wanted: " << svalues[i][j] << std::endl;
	  }
	}
      }
    }
    
  }
  
  
  // (out of/limited resources) checks not done
  
  
  
  if(ok)
    std::cout << "conffile TESTS PASSED.\n";
}




#if 0
void test_compression()
{
  try{
    
    std::cout << "MemoryCompressor (zlib compression) tests"
	      << std::endl;
    
    // random data to pre-allocated memory area and back
    for(unsigned int j=0;j<32;j++)
    {
      char* buffer  = 0;
      char* cbuffer = 0;
      char* buffer2 = 0;
      
      unsigned int size = rand() % 16386;
      unsigned int csize = 0;
      
      MemoryCompressor* mc = new MemoryCompressor();
      
      
      buffer = (char*)malloc(sizeof(char) * size);
      cbuffer = (char*)malloc(sizeof(char) * size);
      buffer2 = (char*)malloc(sizeof(char) * size);
      
      
      if(buffer == 0 || cbuffer == 0 || buffer2 == 0){
	if(buffer) free(buffer);
	if(cbuffer) free(cbuffer);
	if(buffer2) free(buffer2);
	
	std::cout << "Memory allocation failure during basic test"
		  << std::endl;
	return;
      }
      
      for(unsigned int i=0;i<size;i++){
	buffer[i] = rand() % 256;
	cbuffer[i] = rand() % 256;
	buffer2[i] = rand() % 256;
      }
      
      mc->setMemory(buffer , size);
      mc->setTarget(((void*)cbuffer), size);
      
      
      if(mc->compress() == false){
	std::cout << "Basic memory compression failed" 
		  << std::endl;
	return;
      }
      
      cbuffer = (char*)( mc->getTarget(csize) );
      mc->setMemory(buffer2, size);
      
      if(mc->decompress() == false){
	std::cout << "Basic memory decompression failed"
		  << std::endl;
	return;
      }
      
      cbuffer = (char*)( mc->getTarget(csize) );
      buffer2 = (char*)( mc->getMemory(size) );
      
      // compares buffer and buffer2
      
      for(unsigned int i=0;i<size;i++){
	unsigned char ch1 = buffer[i];
	unsigned char ch2 = buffer2[i];
	
	if(ch1 != ch2){
	  std::cout << "decompress(compress(x)) != x"
		    << std::endl;
	  std::cout << "index: " << i << " size: " << size << std::endl;
	  return;
	}
      }
      
      
      free(buffer);
      free(cbuffer);
      free(buffer2);
      
      delete mc;
    }
    
    
    // random data to memory compressor allocated memory area
    // and back
    {
      char* buffer  = 0;
      char* cbuffer = 0;
      char* buffer2 = 0;
      
      unsigned int size = rand() % 16386;
      unsigned int csize = 0;
      
      MemoryCompressor* mc = new MemoryCompressor();
      
      
      buffer = (char*)malloc(sizeof(char) * size);
      
      buffer2 = (char*)malloc(sizeof(char) * size);
      
      
      if(buffer == 0 || buffer2 == 0){
	if(buffer) free(buffer);
	
	if(buffer2) free(buffer2);
	
	std::cout << "Memory allocation failure during advanced test"
		  << std::endl;
	return;
      }
      
      for(unsigned int i=0;i<size;i++){
	buffer[i] = rand() % 256;	
	buffer2[i] = rand() % 256;
      }
      
      mc->setMemory(buffer , size);
      
      
      
      if(mc->compress() == false){
	std::cout << "Advanced memory compression failed" 
		  << std::endl;
	return;
      }
      
      
      mc->setMemory((void*)buffer2, size);
      cbuffer = (char*)( mc->getTarget(csize) );
      
      if(mc->decompress() == false){
	std::cout << "Advanced memory decompression failed"
		  << std::endl;
	return;
      }
            
      buffer2 = (char*)( mc->getMemory(size) );
      cbuffer = (char*)( mc->getTarget(csize) );
      
      // compares buffer and buffer2
      
      for(unsigned int i=0;i<size;i++){
	if(buffer[i] != buffer2[i]){
	  std::cout << "decompress(compress(x)) != x"
		    << std::endl;
	  std::cout << "index: " << i << " size: " << size << std::endl;
	  return;
	}
      }
      
      
      free(buffer);
      free(cbuffer);
      free(buffer2);
      
      delete mc;
    }
    
    
    std::cout << "MEMORY COMPRESSION TESTS PASSED" << std::endl;
    
  }
  catch(std::exception& e){
    std::cout << "Uncaught exception: "
	      << e.what() << std::endl;
  }
}
#endif






void conffile_create_good_varname(std::string& str)
{
  for(unsigned int i=0;i<str.size();i++){
    
    unsigned int a;
    
    do{ a = rand() & 0xFF; } while(!isalpha(a) && !isdigit(a) && a != '_');
    
    str[i] = a;
  }
}


void conffile_create_good_string(std::string& str)
{
  str.resize((rand() % 20) + 1);
  
  for(unsigned int i=0;i<str.size();i++){
    
    unsigned int a;
    
    do{ a = rand() & 0xFF; } while(!isprint(a));
    
    if(a == 0) std::cout << "GENERATED ZERO!!!\n";
    
    str[i] = a;
  }
}


/********************************************************************************/


void test_list_source()
{
  try{
    std::cout << "LIST_SOURCE TESTS" << std::endl;
    
    std::vector<unsigned int> list;
    for(unsigned int i=0;i<50;i++) list.push_back(i);
    
    list_source<unsigned int> ls(list);
    
    for(unsigned int i=0;i<50;i++){
      if(ls[i] != i){
	std::cout << "ERROR: list_source gave bad data" << std::endl;
	return;
      }
    }
    
    if(ls.size() != 50){
      std::cout << "ERROR: list_source has wrong number of data"
		<< std::endl;
      return;
    }
    
    try{
      unsigned int t = ls[50];
      
      std::cout << "ERROR: list_source didn't throw exception with out of range index"
		<< std::endl;
      std::cout << t << std::endl;
      
      return;
    }
    catch(std::out_of_range& e){ /* ok */ }
      
    std::cout << "LIST_SOURCE TESTS PASSED" << std::endl;
  }
  catch(std::exception& e){
    std::cout << "Unexcepted exception: " << e.what() << std::endl;
  }
}

/********************************************************************************/
