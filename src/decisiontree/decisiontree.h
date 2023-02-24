/*
 * experimental decision tree for binary input data and 
 * probability distribution (p_1..p_N) of output labels
 *
 */

#ifndef __whiteice_decision_tree
#define __whiteice_decision_tree


#include <vector>
#include <string>

#include <mutex>
#include <thread>

#include <set>
#include <map>


namespace whiteice
{
  class DTNode
  {
  public:
    DTNode(){
      decisionVariable = -1;
      outcome = -1;
      nodeid = -1;
      
      parent = NULL;
      left0 = NULL;
      right1 = NULL;
    }

    // deletes tree's child nodes
    void deleteChilds(){
      if(left0) left0->deleteChilds();
      if(right1) right1->deleteChilds();

      delete left0;
      delete right1;
    }

    
    void calculateNodeIDs(std::mutex& counter_mutex, int& counter){

      {
	std::lock_guard<std::mutex> lock(counter_mutex);
	this->nodeid = counter;
	counter++;
      }
      
      if(left0) left0->calculateNodeIDs(counter_mutex, counter);
      if(right1) right1->calculateNodeIDs(counter_mutex, counter);
      
    }
    

    bool saveData(std::vector<int>& data){
      
      if(left0) if(left0->saveData(data) == false) return false;
      if(right1) if(right1->saveData(data) == false) return false;

      if(nodeid*5 + 5 > (int)data.size()) return false;

      data[this->nodeid*5 + 0] = this->nodeid;
      data[this->nodeid*5 + 1] = this->decisionVariable;
      data[this->nodeid*5 + 2] = this->outcome;

      if(left0) data[this->nodeid*5 + 3] = left0->nodeid;
      else data[this->nodeid*5 + 3] = -1;
      
      if(right1) data[this->nodeid*5 + 4] = right1->nodeid;
      else data[this->nodeid*5 + 4] = -1;

      return true;
    }

    
    bool loadData(std::vector<int>& data, int& counter)
    {
      if(counter*5 + 5  > (int)data.size()) return false;
      
      this->nodeid = data[counter*5 + 0];
      if(this->nodeid != counter) return false;
      this->decisionVariable = data[counter*5 + 1];
      this->outcome = data[counter*5 + 2];
      this->left0 = nullptr;
      this->right1 = nullptr;

      int origcounter = counter;

      if(data[counter*5 + 3] >= 0){

	left0 = new DTNode();
	left0->nodeid = data[counter*5 + 3];

	counter++;
	
	if(left0->loadData(data, counter) == false) return false;
	if(left0->nodeid != data[origcounter*5 + 3]) return false;

	left0->parent = this;
      }
      
      
      if(data[origcounter*5 + 4] >= 0){
	
	right1 = new DTNode();
	right1->nodeid = data[origcounter*5 + 4];

	counter++;

	if(right1->loadData(data, counter) == false) return false;
	if(right1->nodeid != data[origcounter*5 + 4]) return false;

	right1->parent = this;
      }
      
      return true;
    }


    void printTree(){
      printf("NODE %d (%llx): %d %d %llx %llx %llx\n",
	     nodeid, (long long)this, decisionVariable, outcome, (long long)parent, (long long)left0, (long long)right1);
      
      if(left0) left0->printTree();
      if(right1) right1->printTree();
      
    }
    
    
    int decisionVariable;
    std::set<int> variableSet;
    int outcome; // leaf-node's outcome
    int nodeid; // for saving the tree

    class DTNode *parent;  
    class DTNode *left0, *right1; // child nodes;
  };

  

  class DecisionTree
  {
  public:
    DecisionTree();
    virtual ~DecisionTree();

    bool startTrain(const std::vector< std::vector<bool> >& inputs, const std::vector< std::vector<bool> >& outcomes);
    bool stopTrain();
    bool isRunning() const;

    int classify(std::vector<bool>& input) const;

    bool save(const std::string& filename) const;
    bool load(const std::string& filename);
    
  private:

    // input data: pointers to const objects
    std::vector< std::vector<bool> > const * inputs;
    std::vector< std::vector<bool> > const * outcomes;

    // calculated decision tree
    DTNode* tree;
    mutable std::mutex tree_mutex;

    bool running;
    std::thread* worker_thread;
    mutable std::mutex thread_mutex;

    

    bool matchData(const DTNode* n, const std::vector<bool>& data) const;
    
    bool calculateGoodnessSplit(const DTNode* n,
				int& split_variable, float& split_goodness, int& node_outcome) const;
    
    
    void worker_thread_loop(); // worker thread function
    
  };

  
};

#endif
