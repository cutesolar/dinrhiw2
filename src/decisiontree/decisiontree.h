/*
 * experimental decision tree for binary input data and probability distribution (p_1..p_N) of discrete labels
 *
 */

#ifndef __whiteice_decision_tree
#define __whiteice_decision_tree


#include <vector>
#include <string>

#include <mutex>
#include <thread>

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
      right0 = NULL;
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

    void saveData(std::vector<int>& data){
      
      if(left0) left0->saveData(data);
      if(right1) right1->saveData(data);

      data[this->nodeid*5 + 0] = this->nodeid;
      data[this->nodeid*5 + 1] = this->decisionVariable;
      data[this->nodeid*5 + 2] = this->outcome;

      if(left0) data[this->nodeid*5 + 3] = left0->nodeid;
      else data[this->nodeid*5 + 3] = -1;
      
      if(right0) data[this->nodeid*5 + 4] = right0->nodeid;
      else data[this->nodeid*5 + 4] = -1;
    }

    bool loadData(std::vector<int>& data, int& counter){

      this->nodeid = data[counter*5 + 0];
      if(this->nodeid != counter) return false;
      this->decisionVariable = data[counter*5 + 1];
      this->outcome = data[counter*5 + 2];

      int origcounter = counter;

      if(data[counter*5 + 3] >= 0){

	left0 = new DTNode();
	left0->nodeid = data[counter*5 + 3];
	
	if(left0->loadData(data, ++counter) == false) return false;
	if(left0->nodeid != data[origcounter*5 + 3]) return false;
	
      }

      if(data[counter*5 + 4] >= 0){
	
	right1 = new DTNode();
	right1->nodeid = data[counter*5 + 4];

	if(right0->loadData(data, ++counter) == false) return false;
	if(right0->nodeid != data[origcounter*5 + 4]) return false;
	
      }
      
      return true;
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

    unsigned int classify(std::vector<bool>& input) const;

    bool save(const std::string& filename) const;
    bool load(const std::string& filename);
    
  private:

    // input data: pointers to const objects
    std::vector< std::vector<bool> > const * inputs
    std::vector< std::vector<bool> > const * outcomes;

    // calculated decision tree
    DTNode* tree;
    mutable std::mutex tree_lock;

    bool running;
    std::thread* worker_thread;
    std::mutex thread_mutex;
  };

  
};

#endif
