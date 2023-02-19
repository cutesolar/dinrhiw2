/*
 * experimental decision tree for binary input data and probability distribution (p_1..p_N) of discrete labels
 *
 */

#ifndef __whiteice_decision_tree
#define __whiteice_decision_tree


#include <vector>
#include <string>


namespace whiteice
{

  class DecisionTree
  {
  public:
    DecisionTree(const unsigned int NUM_BIN_INPUTS, const unsigned int NUM_DISCRETE_OUTCOMES = 2);
    virtual ~DecisionTree();

    bool startTrain(const std::vector< std::vector<bool> >& input, const std::vector< std::vector<bool> >& outcomes);
    bool stopTrain();
    bool isRunning() const;

    unsigned int classify(std::vector<bool>& input) const;

    bool save(const std::string& filename) const;
    bool load(const std::string& filename);
    
  private:

    
  };

  
};

#endif
