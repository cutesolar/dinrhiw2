

#include "decisiontree.h"

#include <stdexcept>
#include <system_error>
#include <functional>

#include <iostream>
#include <stdio.h>

#include "RNG.h"



namespace whiteice
{

  
  DecisionTree::DecisionTree()
  {
    running = false;
    worker_thread = nullptr;
    inputs = nullptr;
    outcomes = nullptr;
    
    {
      std::lock_guard<std::mutex> lock(tree_mutex);
      tree = nullptr;
    }
  }

  
  DecisionTree::~DecisionTree()
  {
    stopTrain();

    {
      std::lock_guard<std::mutex> lock(tree_mutex);

      if(tree){
	tree->deleteChilds();
	delete tree;
      }
      
      tree = nullptr;
    }

    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      
      if(worker_thread)
	delete worker_thread;
      
      worker_thread = nullptr;
    }
    
  }

  
  bool DecisionTree::startTrain(const std::vector< std::vector<bool> >& inputs,
				const std::vector< std::vector<bool> >& outcomes)
  {
    if(running) return false;
    if(inputs.size() <= 0) return false;
    if(inputs.size() != outcomes.size()) return false;
    if(inputs[0].size() <= 0) return false;
    if(outcomes[0].size() <= 0) return false;
    
    
    {
      std::lock_guard<std::mutex> lock(thread_mutex);

      if(running) return false;
    
      this->inputs = &inputs;
      this->outcomes = &outcomes;

      // clears tree structure [FIXME creation of thread may fail so we lose tree data structure]
      {
	std::lock_guard<std::mutex> lock(tree_mutex);
	if(tree){
	  tree->deleteChilds();
	  delete tree;
	}
	
	tree = nullptr;
      }

      try{
	running = true;
	worker_thread = new std::thread(std::bind(&DecisionTree::worker_thread_loop, this));

	return running;
      }
      catch(std::system_error& e){
	if(worker_thread) delete worker_thread;
	worker_thread = nullptr;
	running = false;
	return false;
      }
      
    }
      
    return false;
  }

  
  bool DecisionTree::stopTrain()
  {
    if(running == false){
      std::lock_guard<std::mutex> lock(thread_mutex);

      if(worker_thread){
	worker_thread->join();
	delete worker_thread;
	worker_thread = nullptr;
      }
      
      return false;
    }
    

    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      
      if(running == false) return false;

      running = false;

      if(worker_thread){
	worker_thread->join();
	delete worker_thread;
	worker_thread = nullptr;
      }

      inputs = nullptr;
      outcomes = nullptr;

      return true;
    }
    
    return false;
  }

  
  bool DecisionTree::isRunning() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);

    if(running) return true;
    else return false;
  }
  

  // classify input to target class of the most active variable in outcomes
  int DecisionTree::classify(std::vector<bool>& input) const
  {
    std::lock_guard<std::mutex> lock(tree_mutex);
    
    DTNode* current = tree;

    if(current == NULL) return -1;
    
    if(current->left0 == NULL && current->right1 == NULL)
      return current->outcome;
    

    while(current->decisionVariable >= 0){
      if(current->decisionVariable >= (int)input.size())
	return -1;

      if(current->decisionVariable2 >= 0 && current->decisionVariable2 < (int)input.size()){

	if(input[current->decisionVariable] == true && input[current->decisionVariable2] == true){
	  if(current->right1) current = current->right1;
	  else return current->outcome;
	}
	else{
	  if(current->left0) current = current->left0;
	  else return current->outcome;
	}
      }
      else{
	
	if(input[current->decisionVariable] == false){
	  if(current->left0) current = current->left0;
	  else return current->outcome;
	}
	else if(input[current->decisionVariable] == true){
	  if(current->right1) current = current->right1;
	  else return current->outcome;
	}
      }

    }

    return current->outcome;
  }

  
  bool DecisionTree::save(const std::string& filename) const
  {
    // for each node we save following information:
    // NODEID, DECISION_VARIABLE, OUTCOME_VARIABLE, LEFT0_NODEID, RIGHT1_NODEID
    // there are all int variables (2**31 values)

    std::vector<int> data;

    {
      std::lock_guard<std::mutex> lock(tree_mutex);
      
      int counter = 0;
      
      {
	std::mutex counter_mutex;
	
	tree->calculateNodeIDs(counter_mutex, counter);
      }
      
      
      data.resize(counter*6);
      
      if(tree->saveData(data) == false) return false;
    }

    // saves data vector to disk

    FILE* handle = fopen(filename.c_str(), "wb");
    if(handle == NULL) return false;
    
    if(fwrite(data.data(), sizeof(int), data.size(), handle) != data.size()){
      fclose(handle);
      return false;
    }

    fclose(handle);
    
    return true;
  }


#include <sys/stat.h>
  
  long long GetFileSize(const std::string& filename)
  {
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
  }
  
  
  bool DecisionTree::load(const std::string& filename)
  {
    // loads file to data vector
    std::vector<int> data;

    // gets filesize and resize vector accordingly
    long long filesize = GetFileSize(filename);
    if(filesize % sizeof(int) != 0) return false;

    data.resize(filesize / sizeof(int));

    // reads data from disk
    FILE* handle = fopen(filename.c_str(), "rb");
    if(handle == NULL) return false;

    if(fread(data.data(), sizeof(int), data.size(), handle) != data.size()){
      fclose(handle);
      return false;
    }

    fclose(handle);
    
    // recreates tree structure from data:
    // NODEID, DECISION_VARIABLE, OUTCOME_VARIABLE, LEFT0_NODEID, RIGHT1_NODEID
    // there are all int variables (2**31 values)

    DTNode* node = new DTNode();
    int nvalue = 0;

    if(node->loadData(data, nvalue) == false){

      node->deleteChilds();
      
      return false;
    }


    {
      std::lock_guard<std::mutex> lock(tree_mutex);
      
      if(tree){
	tree->deleteChilds();
	delete tree;
      }
      
      tree = node;
    }
    
    return true;
  }



  // returns true if node's parent's variables matches to data
  bool DecisionTree::matchData(const DTNode* n, const std::vector<bool>& data) const{

    while(n->parent != NULL){
      // while(n->parent != NULL){

      if(n->parent->left0 == n){
	// n->parent->decisionVariable == 0
	const int d2 = n->parent->decisionVariable2;

	if(d2 >= 0){
	  if((data[n->parent->decisionVariable] == true && data[d2] == true)) return false;
	}
	else{
	  if(data[n->parent->decisionVariable] == true) return false;
	}
      }
      else if(n->parent->right1 == n){
	// n->parent->decisionVariable == 1
	const int d2 = n->parent->decisionVariable2;

	if(d2 >= 0){
	  if((data[n->parent->decisionVariable] == false || data[d2] == false)) return false;
	}
	else{
	  if(data[n->parent->decisionVariable] == false) return false;
	}
      }

      n = n->parent;
    }

    return true;
  }
  

  bool DecisionTree::calculateGoodnessSplit(const DTNode* n,
					    int& split_variable, int& split_variable2,
					    float& split_goodness, int& node_outcome) const
  {
    if(n == NULL) return false;
    //if(n->variableSet.size() == 0) return false;

    int best_variable = -1;
    int best_variable2 = -1;
    float best_goodness = -100000000.0f;
    int best_outcome = -1;

    std::map<float, int> GINIs;
    

    std::vector<unsigned long long> rows; // rows where parent nodes are as set

#pragma omp parallel
    {
      std::vector<unsigned long long> r;

#pragma omp for nowait
      for(unsigned long long i=0;i<inputs->size();i++){
	if(matchData(n, (*inputs)[i])){ // checks if node's variable selection matches row
	  r.push_back(i);
	}
      }

#pragma omp critical
      {
	for(unsigned long long i=0;i<r.size();i++)
	  rows.push_back(r[i]);
      }
    }

    
    // calculate outcome for this node
    {
      
      // calculate outcome [p-values of current node]
      std::vector<float> pfull;
      
      pfull.resize((*outcomes)[0].size());
      
      for(auto& p : pfull) p = 0.0f;
      
      for(auto& r : rows){
	for(unsigned int k=0;k<pfull.size();k++)
	  if((*outcomes)[r][k]) pfull[k]++;
      }

      
      //std::cout << "pfull = ";
      if(rows.size())
	for(auto& p : pfull){
	  p /= (float)(rows.size());
	//std::cout << p << " ";
      }
      //std::cout << std::endl;
      
      float pbest = pfull[0];
      int pindex = 0;
      
      for(unsigned int i=0;i<pfull.size();i++){
	if(pbest < pfull[i]){
	  pbest = pfull[i];
	  pindex = i;
	}
      }
      
      best_outcome = pindex;

      node_outcome = best_outcome;
    }
    
    

    
    if(rows.size() <= 20)
      return false; // don't split node if there is less than 21 data points

    
    std::cout << "ROWS = " << rows.size() << std::endl;
    
    
    
    for(auto& candidateSplit : n->variableSet){

      std::set<unsigned long long> rows0; // data rows where variable is 0 and parent nodes are as set
      std::set<unsigned long long> rows1; // data rows where variable is 1 and parent nodes are as set
      
#pragma omp parallel 
      {
	std::set<unsigned long long> r0; // data rows where variable is 0 and parent nodes are as set
	std::set<unsigned long long> r1; // data rows where variable is 1 and parent nodes are as set

#pragma omp for nowait
	for(unsigned long long i=0;i<rows.size();i++){
	  if((*inputs)[rows[i]][candidateSplit] == false) r0.insert(rows[i]);
	  else r1.insert(rows[i]);
	}

#pragma omp critical
	{
	  rows0.insert(r0.begin(), r0.end());
	  rows1.insert(r1.begin(), r1.end());
	}
	
      }

      
      // calculates GINI index for the data rows
      
      const float weight0 = rows0.size() / (float)(rows0.size() + rows1.size());
      const float weight1 = rows1.size() / (float)(rows0.size() + rows1.size());

      if(weight0 >= 1.0f || weight1 >= 1.0f) continue; // must separate some rows..

      // calculates p-values for outcomes rows
      std::vector<float> p0, p1;

      p0.resize((*outcomes)[0].size());
      p1.resize((*outcomes)[0].size());

      for(auto& p : p0) p = 0.0f;
      
      for(auto& r : rows0){
	for(unsigned int k=0;k<p0.size();k++)
	  if((*outcomes)[r][k]) p0[k]++;
      }

      if(rows0.size())
	for(auto& p : p0) p /= (float)rows0.size();

      for(auto& p : p1) p = 0.0f;

      for(auto& r : rows1){
	for(unsigned int k=0;k<p1.size();k++)
	  if((*outcomes)[r][k]) p1[k]++;
      }

      if(rows1.size())
	for(auto& p : p1) p /= (float)rows1.size();

      // GINI value is gini = 1 - ||p||^2 is used for estimating splitting goodness

      float g0 = 0.0f;
      
      for(auto& p : p0) g0 += p*p;

      g0 = 1.0f - g0;

      float g1 = 0.0f;
      
      for(auto& p : p1) g1 += p*p;

      g1 = 1.0f - g1;


      const float GINI = weight0*g0 + weight1*g1;

      // std::cout << "GINI = " << GINI << std::endl;

      if(GINI > best_goodness){
	best_goodness = GINI;
	best_variable = candidateSplit;
	GINIs.insert(std::pair<float, int>(GINI, candidateSplit));
      }
    }
    
    

#if 1 
    // finds second variable which is best match together with first variable: O(26*N) computational complexity

    if(best_variable >= 0){

      const unsigned int NUM_COMB_VARIABLES = 25;

      for(unsigned int m=0;m<NUM_COMB_VARIABLES && m<GINIs.size();m++){

	int selected_variable = -1;

	auto iter = GINIs.end();

	for(unsigned int k=0;k<(m+1);k++) iter--;

	selected_variable = iter->second;

	//std::cout << iter->first << std::endl;
	
	
	for(auto& candidateSplit : n->variableSet){
	  
	  if(candidateSplit == selected_variable) continue; // skip same variable
	  
	  
	  std::set<unsigned long long> rows0; // data rows where variable is 0 and parent nodes are as set
	  std::set<unsigned long long> rows1; // data rows where variable is 1 and parent nodes are as set
	  
#pragma omp parallel 
	  {
	    std::set<unsigned long long> r0; // data rows where variable is 0 and parent nodes are as set
	    std::set<unsigned long long> r1; // data rows where variable is 1 and parent nodes are as set
	    
#pragma omp for nowait
	    for(unsigned long long i=0;i<rows.size();i++){
	      if((*inputs)[rows[i]][candidateSplit] == true && (*inputs)[rows[i]][selected_variable] == true)
		r1.insert(rows[i]);
	      else
		r0.insert(rows[i]);
	    }
	    
#pragma omp critical
	    {
	      rows0.insert(r0.begin(), r0.end());
	      rows1.insert(r1.begin(), r1.end());
	    }
	    
	  }

	  // calculates GINI index for the data rows
	  
	  const float weight0 = ((float)rows0.size()) / (float)(rows0.size() + rows1.size());
	  const float weight1 = ((float)rows1.size()) / (float)(rows0.size() + rows1.size());

	  //std::cout << "weight0 = " << weight0 << std::endl;
	  //std::cout << "weight1 = " << weight1 << std::endl;

	  if(weight0 >= 1.0f || weight1 >= 1.0f) continue; // must separate some rows..
	  
	  // calculates p-values for outcomes rows
	  std::vector<float> p0, p1;
	  
	  p0.resize((*outcomes)[0].size());
	  p1.resize((*outcomes)[0].size());
	  
	  for(auto& p : p0) p = 0.0f;
	  
	  for(auto& r : rows0){
	    for(unsigned int k=0;k<p0.size();k++)
	      if((*outcomes)[r][k]) p0[k]++;
	  }

	  if(rows0.size())
	    for(auto& p : p0) p /= (float)rows0.size();
	  
	  for(auto& p : p1) p = 0.0f;
	  
	  for(auto& r : rows1){
	    for(unsigned int k=0;k<p1.size();k++)
	      if((*outcomes)[r][k]) p1[k]++;
	  }

	  if(rows1.size())
	    for(auto& p : p1) p /= (float)rows1.size();
	  
	  // GINI value is gini = 1 - ||p||^2 is used for estimating splitting goodness
	  
	  float g0 = 0.0f;
	  
	  for(auto& p : p0) g0 += p*p;
	  
	  g0 = 1.0f - g0;
	  
	  float g1 = 0.0f;
	  
	  for(auto& p : p1) g1 += p*p;
	  
	  g1 = 1.0f - g1;
	  
	  
	  const float GINI = weight0*g0 + weight1*g1;

	  // std::cout << "GINI = " << GINI << std::endl;
	  
	  if(GINI > best_goodness){
	    best_goodness = GINI;
	    best_variable = selected_variable;
	    best_variable2 = candidateSplit;
	  }
	}

      }

    }
      
#endif
      
    
    split_variable = best_variable;
    split_variable2 = best_variable2;
    split_goodness = best_goodness;
    node_outcome = best_outcome;

    std::cout << "split_variables = " << split_variable << " " << split_variable2 << std::endl;

    return (split_variable >= 0); // found some split variable(s)
  }

  
  
  void DecisionTree::worker_thread_loop()
  {
    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      if(running == false) return;
    }
    
    
    // set thread priority (non-standard)
    {
      sched_param sch_params;
      int policy = SCHED_FIFO;
      
      pthread_getschedparam(pthread_self(),
			    &policy, &sch_params);
      
#ifdef linux
      policy = SCHED_IDLE; // in linux we can set idle priority
#endif	
      sch_params.sched_priority = sched_get_priority_min(policy);
      
      if(pthread_setschedparam(pthread_self(),
			       policy, &sch_params) != 0){
	// printf("! SETTING LOW PRIORITY THREAD FAILED\n");
      }
      
#ifdef WINOS
      SetThreadPriority(GetCurrentThread(),
			THREAD_PRIORITY_IDLE);
#endif	
    }
    

    std::lock_guard<std::mutex> lock(tree_mutex);
    
    tree = new DTNode(); 

    DTNode* current = tree;
    std::map<DTNode*, float> goodness;
    std::set<int> initialVariableSet;

    for(unsigned int i=0;i<(*inputs)[0].size();i++)
      initialVariableSet.insert((int)i);
    
    int var = -1, var2 = -1;
    int outcome = -1;
    float g = 0.0;
    
    current->variableSet = initialVariableSet;

    
    if(calculateGoodnessSplit(current, var, var2, g, outcome) == true){
      current->decisionVariable = var;
      current->decisionVariable2 = var2;
      current->outcome = outcome;
      goodness.insert(std::pair<DTNode*,float>(current, g));
    }

    
    
    while(goodness.size() > 0){

      {
	std::lock_guard<std::mutex> lock(thread_mutex);
	if(running == false) break;
      }
      
      auto iter = goodness.end();
      iter--;
      current = iter->first;
      goodness.erase(iter);

      current->left0 = nullptr;
      current->right1 = nullptr;

      // now split based on variable

      DTNode* left0 = new DTNode();
      DTNode* right1 = new DTNode();

      left0->parent = current;
      right1->parent = current;
      left0->variableSet = current->variableSet;
      right1->variableSet = current->variableSet;
      
      
#if 0
      if(current->decisionVariable2 < 0){
	left0->variableSet.erase(current->decisionVariable);
	right1->variableSet.erase(current->decisionVariable);
      }

      if((whiteice::rng.rand()&3) == 0){
	left0->variableSet.erase(current->decisionVariable);
	right1->variableSet.erase(current->decisionVariable);
      }
#endif
      

      //if(left0->variableSet.size() > 0)
      {
	current->left0 = left0;
	
	if(calculateGoodnessSplit(left0, var, var2, g, outcome) == true){
	  left0->decisionVariable = var;
	  left0->decisionVariable2 = var2;
	  left0->outcome = outcome;
	  
	  goodness.insert(std::pair<DTNode*,float>(left0, g));
	}
	else{
	  left0->decisionVariable = var;
	  left0->decisionVariable2 = var2;
	  left0->outcome = outcome;
	  current->left0 = left0;
	  //goodness.insert(std::pair<DTNode*,float>(left0, g));
	}
	
      }
      //else delete left0;

      //if(right1->variableSet.size() > 0)
      {
	current->right1 = right1;
	
	if(calculateGoodnessSplit(right1, var, var2, g, outcome) == true){
	  right1->decisionVariable = var;
	  right1->decisionVariable2 = var2;
	  right1->outcome = outcome;
	  
	  goodness.insert(std::pair<DTNode*,float>(right1, g));
	}
	else{
	  right1->decisionVariable = var;
	  right1->decisionVariable2 = var2;
	  right1->outcome = outcome;
	  current->right1 = right1;
	  //goodness.insert(std::pair<DTNode*,float>(right1, g));
	}
      }
      //else delete right1;
      
    }

    printf("CALCULATED DECISION TREE\n");
    tree->printTree();

    
    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      running = false;
    }
    
    
  }
  
  
};
