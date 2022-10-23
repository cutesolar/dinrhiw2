/*
 * File datasource for crypting.
 * 
 */

#include <string>

//#include <dinrhiw/dinrhiw.h>
#include <dinrhiw.h>


template <unsigned int B>
class CryptoFileSource : public whiteice::data_source<whiteice::dynamic_bitset>
{
 public:
  CryptoFileSource(const std::string& filename,
		   bool encrypting, bool aarmour = false);
  
  ~CryptoFileSource();
  
  whiteice::dynamic_bitset& operator[](unsigned int index)
    { return (*DC)[index]; }
  
  const whiteice::dynamic_bitset& operator[](unsigned int index)
    const { return (*DC)[index]; }
  
  unsigned int size() const ;
  
  bool good() const ;
  
  void flush() const;
  
  // writes data to file with given filename
  bool write(const std::string& filename) const ;
  
  const whiteice::dynamic_bitset& getIV(){ return IV; }
  
 private:
  
  FILE* file;
  bool encrypting;   // true if we are encrypting data, 
                     // false otherwise
  bool asciiarmour;
  whiteice::dynamic_bitset IV;

  whiteice::RNG<>* rng = new whiteice::RNG<>(true); // secure random number source
  
  mutable unsigned char* buffer;
  mutable whiteice::crypto::DataConversion<B>* DC;
  mutable unsigned long long filesize;
  
};


extern template class CryptoFileSource<128u>;



