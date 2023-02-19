

#include "decisiontree.h"


namespace whiteice
{

  
  DecisionTree::DecisionTree(const unsigned int NUM_BIN_INPUTS,
			     const unsigned int NUM_DISCRETE_OUTCOMES)
  {
  }
  
  DecisionTree::~DecisionTree(){
    
  }

  
  bool DecisionTree::startTrain(const std::vector< std::vector<bool> >& input, const std::vector< std::vector<bool> >& outcomes)
  {
    return false;
  }
  
  bool DecisionTree::stopTrain()
  {
    return false;
  }
  
  bool DecisionTree::isRunning() const
  {
    return false;
  }
  
  unsigned int DecisionTree::classify(std::vector<bool>& input) const
  {
    return false;
  }
  
  bool DecisionTree::save(const std::string& filename) const
  {
    return false;
  }
  
  bool DecisionTree::load(const std::string& filename)
  {
    return false;
  }
  
  
};
