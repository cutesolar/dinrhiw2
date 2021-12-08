
#include <stdio.h>
#include <stdlib.h>
#include <zlib.h> // -lz

#include "MemoryCompressor.h"


namespace whiteice
{
  
  
  MemoryCompressor::MemoryCompressor()
  {
    data = 0;
    compressed_data = 0;
    datalen = 0;
    compressed_datalen = 0;
  }
  
  
  MemoryCompressor::~MemoryCompressor(){ }
  
  
  // sets source data memory region (nbytes long) for
  // the next compression operation or target area
  // for the next decompression operation. Memory area
  // must be allocated by malloc() and could be realloc()ated.
  void MemoryCompressor::setMemory(void* ptr, unsigned int nbytes)
  {
    this->data = ptr;
    this->datalen = nbytes;
  }
  
  
  // sets target data memory region for the compressed data
  // or source memory region for decompress() operation.
  // Memory regions must be realloc()atable if needed.
  void MemoryCompressor::setTarget(void* ptr, unsigned int nbytes)
  {
    this->compressed_data = ptr;
    this->compressed_datalen = nbytes;
  }
    
  
  void* MemoryCompressor::getMemory(){ return this->data; }
  void* MemoryCompressor::getTarget(){ return this->compressed_data; }
  
    
  void* MemoryCompressor::getMemory(unsigned int& nbytes){
    nbytes = this->datalen;
    return this->data;
  }
    
  
  void* MemoryCompressor::getTarget(unsigned int& nbytes){
    nbytes = this->compressed_datalen;
    return this->compressed_data;
  }
  
  
  unsigned int MemoryCompressor::getMemorySize()
  {
    return this->datalen;    
  }
  
  
  unsigned int MemoryCompressor::getTargetSize()
  {
    return this->compressed_datalen;
  }
  
  
  void MemoryCompressor::clearMemory(){
    this->data = 0;
    this->datalen = 0;
  }    
  
  
  void MemoryCompressor::clearTarget()
  {
    this->compressed_data = 0;
    this->compressed_datalen = 0;
  }
  
  
  void MemoryCompressor::clear()
  {
    clearMemory();
    clearTarget();
  }
  
  
  float MemoryCompressor::ratio() const 
  {
    if(compressed_data != 0 && data != 0){
      return ((float)compressed_datalen / (float)datalen);
    }
    else return 1.0f;
  }
  
  
  bool MemoryCompressor::compress() 
  {
    if(this->data == 0 || this->datalen == 0)
      return false;
    
    bool allocated_memory = false;
    
    if(this->compressed_data == 0 && this->compressed_datalen != 0)
      return false;
    
    if(this->compressed_data == 0){
      this->compressed_data = // guess 66% compression
	(void*)malloc(sizeof(char)*(2*(this->datalen) + 1)/3);
      
      // cannot get enough memory
      if(this->compressed_data == 0)
	return false;
      
      this->compressed_datalen = (2*(this->datalen) + 1)/3;
      allocated_memory = true;
    }
    
    
    z_stream zs;
    
    zs.next_in  = (Bytef*)this->data;
    zs.next_out = (Bytef*)this->compressed_data;
    
    zs.avail_in  = this->datalen;
    zs.total_in  = 0;
    zs.avail_out = this->compressed_datalen;
    zs.total_out = 0;
    
    zs.zalloc = (alloc_func)0;
    zs.zfree  = (free_func)0;
    zs.opaque = (voidpf)0;
    zs.data_type = Z_BINARY; // (needed?)
    
    if(deflateInit(&zs, Z_DEFAULT_COMPRESSION) != Z_OK){
      if(allocated_memory){
	free(this->compressed_data);
	this->compressed_data = 0;
	this->compressed_datalen = 0;	
      }
      
      return false;
    }
    
    
    while(1){
      
      int ok = deflate(&zs, Z_FINISH);
      
      if(ok == Z_STREAM_END){
	break;
      }
      else if(ok != Z_OK){
	if(allocated_memory){
	  free(this->data);
	  this->data = 0;
	  this->datalen = 0;
	}
	
	return false;
      }
      
      // Z_OK
      
      unsigned int memsize_increase = (zs.avail_in + 1)/2;
      if(memsize_increase < 16) memsize_increase = 16;
      
      void *new_area = 
	(void*)realloc(this->compressed_data,
		       this->compressed_datalen + 
		       memsize_increase);
      
      if(new_area == 0){
	if(allocated_memory){
	  free(this->compressed_data);
	  this->compressed_data = 0;
	  this->compressed_datalen = 0;
	}
	
	return false;
      }
      
      
      zs.next_in   = (Bytef*)&(((unsigned char*)data)[zs.total_in]);
      zs.next_out  = (Bytef*)&(((unsigned char*)new_area)[zs.total_out]);
      zs.avail_out += memsize_increase;
      
      this->compressed_data = new_area;
      this->compressed_datalen += memsize_increase;
    }
    
    
    if(deflateEnd(&zs) != Z_OK){
      if(allocated_memory){
	free(this->compressed_data);
	this->compressed_data = 0;
	this->compressed_datalen = 0;
      }
      
      return false;
    }
    
    
    this->compressed_datalen = zs.total_out;
    void *new_area = (void*)realloc(this->compressed_data,
				    this->compressed_datalen);
    if(new_area)
      this->compressed_data = new_area;
    
    return true;
  }
  
  
  
  bool MemoryCompressor::decompress() 
  {
    if(this->compressed_data == 0 || this->compressed_datalen == 0)
      return false;
    
    if(this->data == 0 && this->datalen != 0)
      return false;
    
    
    bool allocated_memory = false;
    
    
    if(this->data == 0){
      this->data = 
	(void*)malloc(sizeof(char)*(this->compressed_datalen + 1)/2);
      
      // cannot get enough memory
      if(this->data == 0)
	return false;
      
      this->datalen = (this->compressed_datalen + 1)/2;
      allocated_memory = true;
    }
    
      
    z_stream zs;
    
    zs.next_in  = (Bytef*)this->compressed_data;
    zs.next_out = (Bytef*)this->data;
    
    zs.avail_in  = this->compressed_datalen;
    zs.total_in  = 0;
    zs.avail_out = this->datalen;
    zs.total_out = 0;
    
    zs.zalloc = (alloc_func)0;
    zs.zfree  = (free_func)0;
    zs.opaque = (voidpf)0;
    zs.data_type = Z_BINARY; // (needed?)
    
    if(inflateInit(&zs) != Z_OK){
      if(allocated_memory){
	free(this->data);
	this->data = 0;
	this->datalen = 0;
      }
      
      return false;
    }
    
    
    while(1){
      
      int ok = inflate(&zs, Z_SYNC_FLUSH);
      
      if(ok == Z_STREAM_END){
	break;
      }
      else if(ok != Z_OK){
	if(allocated_memory){
	  free(this->data);
	  this->data = 0;
	  this->datalen = 0;
	  }
	
	return false;
      }
      
      
      // Z_OK
      
      unsigned int memsize_increase = (zs.avail_in + 1)/2;
      if(memsize_increase < 16) memsize_increase = 16;
      
      void *new_area = 
	(void*)realloc(this->data,
		       this->datalen + 
		       memsize_increase);
      
      if(new_area == 0){
	if(allocated_memory){
	  free(this->data);
	  this->data = 0;
	  this->datalen = 0;
	}
	
	return false;
      }
      
      
      zs.next_in   = (Bytef*)&(((unsigned char*)compressed_data)[zs.total_in]);
      zs.next_out  = (Bytef*)&(((unsigned char*)new_area)[zs.total_out]);
      zs.avail_out += memsize_increase;
      
      this->data = new_area;
      this->datalen += memsize_increase;
    }
    
    
    if(inflateEnd(&zs) != Z_OK){
      if(allocated_memory){
	free(this->data);
	this->data = 0;
	this->datalen = 0;
      }
      
      return false;
    }
    
    
    this->datalen = zs.total_out;
    void *new_area = (void*)realloc(this->data,
				    this->datalen);
    if(new_area)
      this->data = new_area;
    
    return true;
  }
  
  
}







