
/*
 * file datasource, pads the last block if zeros if needed
 * block size must be multiple of 8. Reads whole file to memory
 * and user of FileSource accesses memory
 * (reading page(s) to file, marking some dirty and writing dirty ones
 *  back to file when page(s) are 'swapped' out (new read/write) would
 *  be better)
 */

#include <string>

//#include <dinrhiw/dinrhiw.h>
#include <dinrhiw.h>


// block size, must be power of 8:s
template <unsigned int B>
class FileSource : public whiteice::data_source<whiteice::dynamic_bitset>
{
 public:
  
  FileSource(const std::string& filename);
  ~FileSource();
  
  whiteice::dynamic_bitset& operator[](unsigned int index)
    ;
  
  const whiteice::dynamic_bitset& operator[](unsigned int index)
    const ;
  
  unsigned int size() const ;
  
  bool good() const ;
  
  void flush() const;
  
  // writes data to file with given filename
  bool write(const std::string& filename) const ;
  
 private:
  
  FILE* file;
  
  mutable unsigned char* buffer;
  mutable whiteice::crypto::DataConversion<B>* DC;
  mutable unsigned int filesize;
  
};


extern template class FileSource<128>;





