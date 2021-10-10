#ifndef FPTREE_HPP
#define FPTREE_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>


using Item = unsigned long;
// using Item = std::string;
using Transaction = std::vector<Item>;
using TransformedPrefixPath = std::pair<std::vector<Item>, uint64_t>;
using Pattern = std::pair<std::set<Item>, uint64_t>;


// caller must implement interface from which data vectors can be retrieved (possible from the disk)
class TransactionSourceInterface {
public:
  
  virtual const unsigned long getNumberOf() const = 0; // number of data vector pairs (x,y)
  
  // gets index:th data points or return false (bad index or unknown error)
  virtual const bool getData(const unsigned long index,
			     Transaction& dataset) const = 0;
};

class ConditionalTreeTransactions : public TransactionSourceInterface
{
public:
  ConditionalTreeTransactions(const std::vector<Transaction>& trans) : transactions(trans)
  {
    
  }

  virtual const unsigned long getNumberOf() const // number of data vector pairs (x,y)
  {
    return transactions.size();
  }
  
  // gets index:th data points or return false (bad index or unknown error)
  virtual const bool getData(const unsigned long index,
			     Transaction& dataset) const
  {
    if(index >= transactions.size()) return false;

    dataset = transactions[index];

    return true;
  }

private:

  const std::vector<Transaction>& transactions;
};


struct FPNode {
  const Item item;
  uint64_t frequency;
  std::shared_ptr<FPNode> node_link;
  std::weak_ptr<FPNode> parent;
  std::vector< std::shared_ptr<FPNode> > children;
  
  FPNode(const Item&, const std::shared_ptr<FPNode>&);
};

struct FPTree {
  std::shared_ptr<FPNode> root;
  std::map< Item, std::shared_ptr<FPNode> > header_table;
  uint64_t minimum_support_threshold;

  // FPTree(const std::vector<Transaction>&, uint64_t);

  FPTree(FPTree& t){
    assert(0); // not implemented, just work around compiler errors
    
    root = t.root;
    header_table = t.header_table;
    minimum_support_threshold = t.minimum_support_threshold;
  }
  
  FPTree(const TransactionSourceInterface* transactions,
	 uint64_t minimum_support_threshold);
  
  bool empty() const;
};


bool fptree_growth(const FPTree&, std::set<Pattern>& p);


#endif  // FPTREE_HPP
