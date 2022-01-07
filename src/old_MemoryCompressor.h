/*
 * MemoryCompressor class
 *
 * is a wrapper class for memory area. this memory
 * area can be then compressed or decompressed.
 *
 * uses zlib's non-lossy data compression
 * to compress given memory region to another
 * region
 */

#include "compressable.h"

#ifndef MemoryCompressor_h
#define MemoryCompressor_h

namespace whiteice
{
  
  class MemoryCompressor
    {
      public:
      MemoryCompressor();
      virtual ~MemoryCompressor();
      
      // sets source data memory region (nbytes long) for
      // the next compression operation or target area
      // for the next decompression operation. Memory area
      // must be allocated by malloc() and could be realloc()ated.
      void setMemory(void* ptr, unsigned int nbytes);
      
      // sets target data memory region for the compressed data
      // or source memory region for decompress() operation.
      // Memory regions must be realloc()atable if needed.
      void setTarget(void* ptr, unsigned int nbytes);
	
      void* getMemory();
      void* getTarget();
      void* getMemory(unsigned int& nbytes);
      void* getTarget(unsigned int& nbytes);
      
      // if available
      unsigned int getMemorySize();
      unsigned int getTargetSize();
      
      void clearMemory();
      void clearTarget();
      void clear();
      
      float ratio() const ;
      
      // compresses data. this *may* allocate memory
      // which caller must free() (getTarget())
      bool compress() ;
      
      // decompressed data. this *may* allocated memory
      // which caller must free() (getMemory())
      bool decompress() ;
      
    private:
      
      void* data;
      unsigned int datalen;
      
      void* compressed_data;
      unsigned int compressed_datalen;
      
    };
}

#endif
