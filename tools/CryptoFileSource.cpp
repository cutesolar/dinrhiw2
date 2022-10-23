/*
 * fileformat (low endian intel machine is assumed):
 * IDBLOCK 0x2da39b42642099xx 2 dwords (64(56) bit) + options
 *         the first byte is reserved for options
 *         B 0000 0000 "no options"
 *         bit 0: "bzip compression (not supported yet)"
 *         rest of the bits: unused. must be set to zero
 * REALLEN 2 dwords (64bit), this is used to remove padding and
 *                           for internal checking
 * IV      4 dword (128 bit (random data))
 * FILEDATA (CONTENT)
 * 
 * TOTAL HEADER SIZE IS: 32 bytes
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
//#include <sys/mman.h>
#include <unistd.h>
#include <string>

//#include <dinrhiw/dinrhiw.h>
#include <dinrhiw.h>

#include "CryptoFileSource.h"




template <unsigned int B>
CryptoFileSource<B>::CryptoFileSource(const std::string& filename,
				      bool encrypting, bool aarmour)
{
  this->encrypting = encrypting;
  this->asciiarmour = aarmour;
  this->filesize = 0;
  
  file = (FILE*)fopen(filename.c_str(), "rb");
  
  if(ferror(file))
    throw std::runtime_error("file i/o error");
  
  if(asciiarmour == false){
    
    if(encrypting){
      // size also calculates filesize
      unsigned int totalSize = (size()*B)/8 + 32; // padded size + header size
      
      buffer = new unsigned char[totalSize]; 
      
      // sets up headers
      ((unsigned int*)buffer)[0] = 0x64209900;
      ((unsigned int*)buffer)[1] = 0x2da39b42;
      *((unsigned long long*)(buffer + 8)) = filesize;
      
      IV.resize(128);
      
      for(unsigned int i=0;i<16;i++){
	// uses RNG::rand() for IV
	unsigned char ch = rng->rand() & 0xFF;
	IV.value(i) = ch;
	buffer[16 + i] = ch;
      }
      
      
      // reads in file data
      if(fread(buffer + 32, sizeof(char), filesize, file) != filesize)
	throw std::runtime_error("file i/o error");
      
      // pads unused data (final block) with random data
      int padSize = (size()*B)/8 - filesize;
      for(int i=0;i<padSize;i++){
	// uses RNG::rand() for padding
	buffer[32 + filesize + i] = (char)(rng->rand() % 0xFF);
      }
      
      this->DC = new whiteice::crypto::DataConversion<B>(buffer + 32, totalSize - 32);
    }
    else{ // decrypting
      // size also calculates filesize
      unsigned int totalSize = (size()*B)/8; // = padded size + headers
      
      buffer = new unsigned char[totalSize];
      
      // reads in file data
      if(fread(buffer, sizeof(char), totalSize, file) != filesize)
	throw std::runtime_error("file i/o error");
      
      // checks file ID
      if((((unsigned int*)buffer)[0] & 0xFFFFFF00) != 0x64209900 ||
	 ((unsigned int*)buffer)[1] != 0x2da39b42){
	throw std::runtime_error("corrupted encrypted file or file i/o error");
      }
      
      filesize = *((unsigned long long*)(buffer + 8));
      
      IV.resize(128);
      
      for(unsigned int i=0;i<16;i++)
	IV.value(i) = buffer[16 + i];
      
      this->DC = new whiteice::crypto::DataConversion<B>(buffer + 32, totalSize - 32);
    }
  }
  else{ // ASCII ARMOUR
    throw std::runtime_error("NO ASCII ARMOUR SUPPORT");
    // read data and recode it to binary (from ASCII)
    
    
  }
}
  
  

template <unsigned int B>
CryptoFileSource<B>::~CryptoFileSource()
{
  if(DC != 0){ delete DC; DC = 0; }
  if(file != 0){ fclose(file); file = 0; }
  if(buffer != 0){ free(buffer); buffer = 0; }
  if(rng != 0){ delete rng; rng = 0; }
}




/* returns number of blocks */
template <unsigned int B>
unsigned int CryptoFileSource<B>::size() const 
{
  if(filesize == 0){
    struct stat buf;
    
    if(fstat(fileno(file), &buf) != 0)
      return 0;
    
    filesize = buf.st_size;
  }
  
  return ((filesize*8+(B-1))/B); // converts to block size
}


template <unsigned int B>
bool CryptoFileSource<B>::good() const 
{
  if(ferror(file) != 0) return false;
  
  // checks file is regular file (non-special)
  struct stat buf;
  if(fstat(fileno(file), &buf) != 0)
    return false;
  
  // shoud be regular non-directory file
  if( (S_IFREG & buf.st_mode) == 0 ||
      (S_IFDIR & buf.st_mode) != 0) return false;
  
  // block size must be multiple of 8
  if((B % 8) != 0) return false;
  
  return true;
}


template <unsigned int B>
void CryptoFileSource<B>::flush() const
{
  if(DC) DC->flush();
}


template <unsigned int B>
bool CryptoFileSource<B>::write(const std::string& filename) const 
{
  flush();
  
  FILE* out = (FILE*)fopen(filename.c_str(), "wb");
  if(out == 0) return false;
  
  if(asciiarmour == false){
    if(encrypting){
      if(fwrite(buffer, sizeof(char), (size()*B)/8 + 32, out) != (size()*B)/8 + 32)
	return false;
    }
    else{
      if(fwrite(buffer + 32, sizeof(char), filesize, out) != filesize)
	return false;
    }
  }
  else{
    // NOT IMPLEMENTED YET.
    // NEED TO DECODE EACH BYTE TO ASCII DATA AND
    // WRITE IT
    
    return false;
    
    if(encrypting){
      
    }
    else{
      
    }
  }
  
  fclose(out);
  
  return true;
}


/**************************************************/

template class CryptoFileSource<128u>;

/**************************************************/
