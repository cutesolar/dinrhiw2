/*
 * tests cryptography
 */

#include <iostream>
#include <climits>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dynamic_bitset.h"
#include "DES.h"
#include "AES.h"
#include "PAD.h"
#include "SHA.h"
#include "RSA.h"
#include "DSA.h"
#include "integer.h"

////////////////////////////////////////////////////////////

void des_test();
void aes_test();
void pad_test();
void rsa_test();
void test_sha();
void dsa_test();


void change_endianess(whiteice::uint32& x);
void change_endianess(whiteice::uint64& x);

////////////////////////////////////////////////////////////

// number of bits in a dynamic_bitset, must be multitiple of 8 bits !
class DynamicBitsetVectorSource : public whiteice::data_source<whiteice::dynamic_bitset>
{
public:
  DynamicBitsetVectorSource(std::vector<whiteice::dynamic_bitset>* data){ this->data = data; }
  virtual ~DynamicBitsetVectorSource(){ }
           
  
  whiteice::dynamic_bitset& operator[](unsigned int index) 
  { return (*data)[index]; }
  const whiteice::dynamic_bitset& operator[](unsigned int index) const 
  { return (*data)[index]; }
  
  // number of dynamic_bitsets available
  unsigned int size() const { return data->size(); }
  
  // tells if the data source access is working correctly (-> can read more data)
  bool good() const { return true; }
  
  void flush() const { }
  
private:
  std::vector<whiteice::dynamic_bitset>* data;
};




int main()
{
  std::cout << "CRYPTO TESTS" 
	    << std::endl << std::endl;
  
  srand(time(0));
  
  pad_test();
  des_test();
  aes_test();
  
  test_sha();
  
  rsa_test();
  
  dsa_test();
  
  return 0;
}



////////////////////////////////////////////////////////////


class test_exception : public std::exception
{
public:
  
  test_exception() 
  {
    reason = 0;
  }
  
  test_exception(const std::exception& e) 
  {
    reason = 0;
    
    const char* ptr = e.what();
    
    if(ptr)
      reason = (char*)malloc(sizeof(char)*(strlen(ptr) + 1));
    
    if(reason) strcpy(reason, ptr);
  }
  
  
  test_exception(const char* ptr) 
  {
    reason = 0;
    
    if(ptr){
      reason = (char*)malloc(sizeof(char)*(strlen(ptr) + 1));
      
      if(reason) strcpy(reason, ptr);
    }
  }
  
  
  virtual ~test_exception() 
  {
    if(reason) free(reason);
    reason = 0;
  }
  
  virtual const char* what() const throw()
  {
    if(reason == 0) // null terminated value
      return ((const char*)&reason);

    return reason;
  }
  
private:
  
  char* reason;
  
};



////////////////////////////////////////////////////////////


void pad_test()
{
  using namespace whiteice::crypto;
  using namespace whiteice;
  
  unsigned int t = 1;
  
  // TEST 1: PAD encryption & decryption test
  try{        
    PAD pad;
    
    Keyschedule<dynamic_bitset>* padkey;
    dynamic_bitset key;
    dynamic_bitset data;
    dynamic_bitset result;
    
    key.resize((rand() % 128) + 128);
    data.resize((rand() % 128) + 128);
    result.resize((rand() % 128) + 128);
    
    std::cout << "PAD ENCRYPT/DECRYPT TESTS" << std::endl;
    
    for(unsigned int i=0;i<1000;i++){
      
      for(unsigned int j=0;j<key.size();j++)
	key.set(j, rand() & 1);
      
      for(unsigned int j=0;j<data.size();j++)
	data.set(j, rand() & 1);
      
      result = data;
      
      padkey = new PADKey(key);
      
      pad.encrypt(data, *padkey);
      pad.decrypt(data, *padkey);
      
      delete padkey;
      
      if(data != result)
	throw test_exception("PAD encryption/decryption failed.");
    }
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
}


////////////////////////////////////////////////////////////


void des_test()
{
  using namespace whiteice::crypto;
  using namespace whiteice;
  
  unsigned int t = 0;

  // TEST 0: DES constants tests
  try{
    
    // tests that IP1 is inverse of IP0
    
    for(unsigned int i=0;i<64;i++){
      if( DES::IP1[DES::IP0[i] - 1] - 1 != i){
	std::cout << "DES IP: bad permutation, bit " 
		  << i << std::endl;
	
	throw test_exception("DES initial permutation check failed\n");
      }
	
    }
  
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
  t = 1;
  
  // TEST 1: DES Keyschedule
  try{
    std::cout << "DES and 3DES  KEYSCHEDULE GENERATION TEST" 
	      << std::endl;
    
    std::cout << "ERROR: obtain correct values from 3rd party implementation"
	      << std::endl;
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }

  t++;
  
  
  // TEST 2: DES encryption tests
  try{
    std::cout << "DES and 3DES ENCRYPTION TEST" 
	      << std::endl;
    
    std::cout << "ERROR: obtain correct values from 3rd party implementation"
	      << std::endl;
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }


  t++;
  
  
  // TEST 3: DES decryption tests
  try{
    std::cout << "DES DECRYPTION TEST" << std::endl;
    
    DES des;
    
    DESKey* deskey;
    dynamic_bitset key;
    dynamic_bitset data;
    dynamic_bitset result;    
    
    key.resize(56);
    data.resize(64);
    result.resize(64);
    
    for(unsigned int i=0;i<50;i++){
      
      for(unsigned int j=0;j<key.size();j++)
	key.set(j, rand() & 1);
      
      for(unsigned int j=0;j<data.size();j++)
	data.set(j, rand() & 1);
      
      result = data;
      
      deskey = new DESKey(key);
      
      des.encrypt(data, *deskey);
      des.decrypt(data, *deskey);
      
      if(data != result)
	throw test_exception("DES encryption/decryption failed.");
      
      delete deskey;
    }
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }

  t++;

  // TEST 4: 3DES decryption tests
  try{
    std::cout << "3DES DECRYPTION TEST" << std::endl;
    
    TDES des;
    
    NDESKey* tdeskey;
    dynamic_bitset key;
    dynamic_bitset data;
    dynamic_bitset result;
    
    key.resize(3*56);
    data.resize(64);
    result.resize(64);
    
    for(unsigned int i=0;i<1000;i++){
      
      for(unsigned int j=0;j<key.size();j++)
	key.set(j, rand() & 1);
      
      for(unsigned int j=0;j<data.size();j++)
	data.set(j, rand() & 1);
      
      result = data;
      
      tdeskey = new NDESKey(key); // key = 3*56 -> triple DES
      
      std::cout.flush();
      
      if(des.encrypt(data, *tdeskey) == false){
	delete tdeskey;
	throw test_exception("3DES encryption failure.");
      }
      
      if(des.decrypt(data, *tdeskey) == false){
	delete tdeskey;
	throw test_exception("3DES decryption failure.");
      }
      
      delete tdeskey;
      
      if(data != result){
	throw test_exception("3DES encryption/decryption error.");
      }
    }
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
}


////////////////////////////////////////////////////////////


void aes_test()
{
  using namespace whiteice::crypto;
  using namespace whiteice;
  
  dynamic_bitset IV;
  IV.resize(128);
  
  
  unsigned int t = 0;
  
  // TEST 0: AES constants tests
  try{
    
    // checks SBOX2 is inverse of SBOX1 and vica versa        
    
    for(unsigned int i=0;i<255;i++){
      
      unsigned int j = i;      
      
      j = AES::SBOX1[j];
      j = AES::SBOX2[j];
      
      if(i != j){
	std::cout << "SBOX2[SBOX1[]] failure, value "
		  << i << std::endl;
	
	throw test_exception("AES constants checking error");
      }
      
      j = i;
      j = AES::SBOX2[j];
      j = AES::SBOX1[j];
      
      if(i != j){
	std::cout << "SBOX2[SBOX1[]] failure, value "
		  << i << std::endl;
	
	throw test_exception("AES constants checking error");
      }
    }
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
  t = 1;
  
  // TEST 1: AES Keyschedule test
  // (tests against examples given in FIPS 197
  //  AES  specification)
  try{
    AESKey* aeskey;
    dynamic_bitset db;
    
    std::cout << "AES KEYSCHEDULE GENERATION TESTS" << std::endl;
    
    //////////////////////////////////////////////////
    // 128 bit key test
    
    std::cout << "AES 128 bit key schedule test" << std::endl;
    
    db.resize(128);
    
    db.value(0 ) = 0x2B; db.value(1 ) = 0x7E; db.value(2 ) = 0x15; db.value(3 ) = 0x16;
    db.value(4 ) = 0x28; db.value(5 ) = 0xAE; db.value(6 ) = 0xD2; db.value(7 ) = 0xA6;
    db.value(8 ) = 0xAB; db.value(9 ) = 0xF7; db.value(10) = 0x15; db.value(11) = 0x88;
    db.value(12) = 0x09; db.value(13) = 0xCF; db.value(14) = 0x4F; db.value(15) = 0x3C;
    
    aeskey = new AESKey(db);        
    
    // 
    // std::cout << std::hex << std::endl;
    // std::cout << "key   : " << db << std::endl;
    // 
    // for(unsigned int i=0;i<aeskey->size();i++)
    //   std::cout << "key " << i << " : " << (*aeskey)[i] << std::endl;
    // 
    
    
    // tests for correct round keys
    
    // the 1st generated round key (2nd round key)
    
    if( (*aeskey)[ 1].value( 3) != 0x17 )
      throw test_exception("1st generated round key mismatch");  
    if( (*aeskey)[ 1].value( 2) != 0xFE )
      throw test_exception("1st generated round key mismatch");    
    if( (*aeskey)[ 1].value( 1) != 0xFA )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 0) != 0xA0 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 7) != 0xB1 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 6) != 0x2C )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 5) != 0x54 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 4) != 0x88 )
      throw test_exception("1st generated round key mismatch");
    
    if( (*aeskey)[ 1].value(11) != 0x39 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value(10) != 0x39 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 9) != 0xA3 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 8) != 0x23 )
      throw test_exception("1st generated round key mismatch");
    
    if( (*aeskey)[ 1].value(15) != 0x05 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value(14) != 0x76 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value(13) != 0x6C )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value(12) != 0x2A )
      throw test_exception("1st generated round key mismatch");
    
    // the last round key
    
    if( (*aeskey)[10].value( 3) != 0xA8 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[10].value( 2) != 0xF9 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[10].value( 1) != 0x14 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[10].value( 0) != 0xD0 )
      throw test_exception("lst generated round key mismatch");
    
    if( (*aeskey)[10].value( 7) != 0x89 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[10].value( 6) != 0x25 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[10].value( 5) != 0xEE )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[10].value( 4) != 0xC9 )
      throw test_exception("lst generated round key mismatch");
    
    if( (*aeskey)[10].value(11) != 0xC8 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[10].value(10) != 0x0C )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[10].value( 9) != 0x3F )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[10].value( 8) != 0xE1 )
      throw test_exception("lst generated round key mismatch");
    
    if( (*aeskey)[10].value(15) != 0xA6 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[10].value(14) != 0x0C )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[10].value(13) != 0x63 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[10].value(12) != 0xB6 )
      throw test_exception("lst generated round key mismatch");
    
    delete aeskey; aeskey = 0;
    
    //////////////////////////////////////////////////
    // 192 bit key test
    
    std::cout << "AES 192 bit key schedule test" << std::endl;
    
    db.resize(192);
    
    db.value( 3) = 0xF7; db.value( 2) = 0xB0; db.value( 1) = 0x73; db.value( 0) = 0x8E;
    db.value( 7) = 0x52; db.value( 6) = 0x64; db.value( 5) = 0x0E; db.value( 4) = 0xDA;
    db.value(11) = 0x2B; db.value(10) = 0xF3; db.value( 9) = 0x10; db.value( 8) = 0xC8;
    db.value(15) = 0xE5; db.value(14) = 0x79; db.value(13) = 0x90; db.value(12) = 0x80;
    db.value(19) = 0xD2; db.value(18) = 0xEA; db.value(17) = 0xF8; db.value(16) = 0x62;
    db.value(23) = 0x7B; db.value(22) = 0x6B; db.value(21) = 0x2C; db.value(20) = 0x52;
    
    
    aeskey = new AESKey(db);
    
    // tests for correct round keys
    
    // the 1st generated round key (2nd round key)
    
    if( (*aeskey)[ 1].value( 3) != 0xD2 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 2) != 0xEA )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 1) != 0xF8 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 0) != 0x62 )
      throw test_exception("1st generated round key mismatch");
    
    if( (*aeskey)[ 1].value( 7) != 0x7B )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 6) != 0x6B )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 5) != 0x2C )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 4) != 0x52 )
      throw test_exception("1st generated round key mismatch");
      
    
    
    if( (*aeskey)[ 1].value(11) != 0xF7 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value(10) != 0x91 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 9) != 0x0C )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value( 8) != 0xFE )
      throw test_exception("1st generated round key mismatch");
    
    if( (*aeskey)[ 1].value(15) != 0xA5 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value(14) != 0xF5 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value(13) != 0x02 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 1].value(12) != 0x24 )
      throw test_exception("1st generated round key mismatch");
      
    
    // the last round key
    
    if( (*aeskey)[12].value(  3) != 0x6F )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[12].value(  2) != 0xA0 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[12].value(  1) != 0x8B )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[12].value(  0) != 0xE9 )
      throw test_exception("lst generated round key mismatch");
    
    if( (*aeskey)[12].value(  7) != 0x3C )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[12].value(  6) != 0x77 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[12].value(  5) != 0x8C )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[12].value(  4) != 0x44 )
	throw test_exception("lst generated round key mismatch");
    
    if( (*aeskey)[12].value( 11) != 0x04 )
      throw test_exception("lst generated round key mismatch");    
    if( (*aeskey)[12].value( 10) != 0x72 )
	throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[12].value(  9) != 0xCC )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[12].value(  8) != 0x8E )
      throw test_exception("lst generated round key mismatch");
    
    if( (*aeskey)[12].value( 15) != 0x02 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[12].value( 14) != 0x22 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[12].value( 13) != 0x00 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[12].value( 12) != 0x01 )
      throw test_exception("lst generated round key mismatch");
    
    delete aeskey; aeskey = 0;
    
    //////////////////////////////////////////////////
    // 256 bit key test
    
    std::cout << "AES 256 bit key schedule test" << std::endl;
    
    db.resize(256);
    db.value( 3) = 0x10; db.value( 2) = 0xEB; db.value( 1) = 0x3D; db.value( 0) = 0x60;
    db.value( 7) = 0xBE; db.value( 6) = 0x71; db.value( 5) = 0xCA; db.value( 4) = 0x15;
    db.value(11) = 0xF0; db.value(10) = 0xAE; db.value( 9) = 0x73; db.value( 8) = 0x2B;
    db.value(15) = 0x81; db.value(14) = 0x77; db.value(13) = 0x7D; db.value(12) = 0x85;
    db.value(19) = 0x07; db.value(18) = 0x2C; db.value(17) = 0x35; db.value(16) = 0x1F;
    db.value(23) = 0xD7; db.value(22) = 0x08; db.value(21) = 0x61; db.value(20) = 0x3B;
    db.value(27) = 0xA3; db.value(26) = 0x10; db.value(25) = 0x98; db.value(24) = 0x2D;
    db.value(31) = 0xF4; db.value(30) = 0xDF; db.value(29) = 0x14; db.value(28) = 0x09;
    
    
    aeskey = new AESKey(db);
    
    // tests for correct round keys
    
    // the 1st generated round key (3rd round key)
    
    if( (*aeskey)[ 2].value( 3) != 0x11 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 2].value( 2) != 0x54 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 2].value( 1) != 0xA3 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 2].value( 0) != 0x9B )
      throw test_exception("1st generated round key mismatch");
    
    if( (*aeskey)[ 2].value( 7) != 0xAF )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 2].value( 6) != 0x25 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 2].value( 5) != 0x69 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 2].value( 4) != 0x8E )
      throw test_exception("1st generated round key mismatch");
    
    if( (*aeskey)[ 2].value(11) != 0x5F )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 2].value(10) != 0x8B )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 2].value( 9) != 0x1A )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 2].value( 8) != 0xA5 )
      throw test_exception("1st generated round key mismatch");
    
    if( (*aeskey)[ 2].value(15) != 0xDE )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 2].value(14) != 0xFC )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 2].value(13) != 0x67 )
      throw test_exception("1st generated round key mismatch");
    if( (*aeskey)[ 2].value(12) != 0x20 )
      throw test_exception("1st generated round key mismatch");
    
    // the last round key
    
    if( (*aeskey)[14].value( 3) != 0xD1 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[14].value( 2) != 0x90 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[14].value( 1) != 0x48 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[14].value( 0) != 0xFE )
      throw test_exception("lst generated round key mismatch");
    
    if( (*aeskey)[14].value( 7) != 0x0B )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[14].value( 6) != 0x8D )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[14].value( 5) != 0x18 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[14].value( 4) != 0xE6 )
      throw test_exception("lst generated round key mismatch");
    
    if( (*aeskey)[14].value(11) != 0x44 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[14].value(10) != 0xF3 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[14].value( 9) != 0x6D )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[14].value( 8) != 0x04 )
      throw test_exception("lst generated round key mismatch");
    
    if( (*aeskey)[14].value(15) != 0x1E )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[14].value(14) != 0x63 )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[14].value(13) != 0x6C )
      throw test_exception("lst generated round key mismatch");
    if( (*aeskey)[14].value(12) != 0x70 )
      throw test_exception("lst generated round key mismatch");
    
    delete aeskey; aeskey = 0;
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  t = t + 1;
  
  
  // TEST 2: AES encryption tests
  // (against examples in FIPS 197 AES specification)
  try{
    AES aes;
    
    AESKey* aeskey;
    dynamic_bitset key;
    dynamic_bitset data;
    dynamic_bitset result;
    
    std::cout << "AES ENCRYPTION TESTS" << std::endl;
    
    //////////////////////////////////////////////////
    // 128 bit key test (this example is taken from FIPS 197)
    
    std::cout << "AES 128 bit encryption test" << std::endl;
    
    // key
    key.resize(128);
    
    for(unsigned int i=0;i<key.blocks();i++)
      key.value(i) = i;
    
    aeskey = new AESKey(key);
    
    // data
    data.resize(128);
    
    for(unsigned int i=0;i<data.blocks();i++)
      data.value(i) = (i + (0x10 * i));
    
    //std::cout << "PLAINTEXT: " << std::hex << data << std::endl;
    //std::cout << "KEY      : " << key << std::endl;
    //std::cout << std::dec;
    //
    //
    //std::cout << "AES-128 TEST ROUND KEYS:" << std::endl;
    //
    //for(unsigned int i=0;i<aeskey->size();i++)
    //  std::cout << std::dec << "key " << i << " : " 
    //		<< std::hex << (*aeskey)[i] << std::endl;
    //
    //std::cout << std::dec << std::endl;
    //
    
    
    // result
    result.resize(128);
    result.value(15) = 0x5a;
    result.value(14) = 0xc5;
    result.value(13) = 0xb4;
    result.value(12) = 0x70;
    result.value(11) = 0x80;
    result.value(10) = 0xb7;
    result.value( 9) = 0xcd;
    result.value( 8) = 0xd8;
    result.value( 7) = 0x30;
    result.value( 6) = 0x04;
    result.value( 5) = 0x7b;
    result.value( 4) = 0x6a;
    result.value( 3) = 0xd8;
    result.value( 2) = 0xe0;
    result.value( 1) = 0xc4;
    result.value( 0) = 0x69;
    
    aes.encrypt(data, *aeskey);
    
    
    //std::cout << std::hex << std::noshowbase << std::endl;
    //std::cout << "CRYPTED  : " << data << std::endl;
    //std::cout << "CORRECT  : " << result << std::endl;
    
    
    if(data != result)
      throw test_exception("AES-128 encryption gave wrong results");
    
    delete aeskey;
    
    
    //////////////////////////////////////////////////
    // 192 bit key test
    
    std::cout << "AES 192 bit encryption test" << std::endl;
    
    // key
    key.resize(192);
    
    for(unsigned int i=0;i<key.blocks();i++)
      key.value(i) = i;
    
    aeskey = new AESKey(key);
    
    // data
    data.resize(128);
    
    for(unsigned int i=0;i<data.blocks();i++)
      data.value(i) = (i + (0x10 * i));
    
    // result
    result.resize(128);
    result.value(15) = 0x91;
    result.value(14) = 0x71;
    result.value(13) = 0x0D;
    result.value(12) = 0xEC;
    result.value(11) = 0xA0;
    result.value(10) = 0x70;
    result.value( 9) = 0xAF;
    result.value( 8) = 0x6E;
    result.value( 7) = 0xE0;
    result.value( 6) = 0xDF;
    result.value( 5) = 0x4C;
    result.value( 4) = 0x86;
    result.value( 3) = 0xA4;
    result.value( 2) = 0x7C;
    result.value( 1) = 0xA9;
    result.value( 0) = 0xDD;
    
    aes.encrypt(data, *aeskey);
    
    if(data != result)
      throw test_exception("AES-192 encryption gave wrong results");
    
    
    delete aeskey;
    
    
    //////////////////////////////////////////////////
    // 256 bit key test
    
    std::cout << "AES 256 bit encryption test" << std::endl;
    
    // key
    key.resize(256);
    
    for(unsigned int i=0;i<key.blocks();i++)
      key.value(i) = i;
    
    aeskey = new AESKey(key);
    
    // data
    data.resize(128);
    
    for(unsigned int i=0;i<data.blocks();i++)
      data.value(i) = (i + (0x10 * i));
    
    // result
    result.resize(128);
    result.value(15) = 0x89;
    result.value(14) = 0x60;
    result.value(13) = 0x49;
    result.value(12) = 0x4B;
    result.value(11) = 0x90;
    result.value(10) = 0x49;
    result.value( 9) = 0xFC;
    result.value( 8) = 0xEA;
    result.value( 7) = 0xBF;
    result.value( 6) = 0x45;
    result.value( 5) = 0x67;
    result.value( 4) = 0x51;
    result.value( 3) = 0xCA;
    result.value( 2) = 0xB7;
    result.value( 1) = 0xA2;
    result.value( 0) = 0x8E;
    
    aes.encrypt(data, *aeskey);
    
    if(data != result)
      throw test_exception("AES-256 encryption gave wrong results");    
    
    delete aeskey;
    
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t 
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
  
  t = t + 1;
  
  // TEST 3: AES decryption tests (assumes encryption works)
  // (randomly generated (data, key) pairs)
  try{
    AES aes;
    
    AESKey* aeskey;
    dynamic_bitset key;
    dynamic_bitset data;
    dynamic_bitset result;
    
    std::cout << "AES DECRYPTION TESTS" << std::endl;
    
    
    //////////////////////////////////////////////////
    // 128 bit key test
    
    key.resize(128);
    data.resize(128);
    result.resize(128);
    
    // tests with 10 different examples
    for(unsigned int k=0;k<10;k++){
      
      for(unsigned int i=0;i<key.size();i++)
	key.set(i, rand() & 1);
      
      for(unsigned int i=0;i<data.size();i++)
	data.set(i, rand() & 1);
      
      aeskey = new AESKey(key);
      
      result = data;
      
      aes.encrypt(data, *aeskey);
      aes.decrypt(data, *aeskey);
      
      delete aeskey;
      
      if(data != result)
	throw test_exception("AES-128 decryption gave wrong results");    
    }
    
    //////////////////////////////////////////////////
    // 192 bit key test
    
    key.resize(192);
    data.resize(128);
    result.resize(128);    
    
    for(unsigned int k=0;k<10;k++){
      
      for(unsigned int i=0;i<key.size();i++)
	key.set(i, rand() & 1);
      
      for(unsigned int i=0;i<data.size();i++)
	data.set(i, rand() & 1);
      
      aeskey = new AESKey(key);
      
      result = data;
      
      aes.encrypt(data, *aeskey);
      aes.decrypt(data, *aeskey);
      
      delete aeskey;
      
      if(data != result)
	throw test_exception("AES-192 decryption gave wrong results");    
    }
    
    
    //////////////////////////////////////////////////
    // 256 bit key test
    
    key.resize(256);
    data.resize(128);
    result.resize(128);
    
    for(unsigned int k=0;k<10;k++){
      
      for(unsigned int i=0;i<key.size();i++)
	key.set(i, rand() & 1);
      
      for(unsigned int i=0;i<data.size();i++)
	data.set(i, rand() & 1);
      
      aeskey = new AESKey(key);
      
      result = data;
      
      aes.encrypt(data, *aeskey);
      aes.decrypt(data, *aeskey);
      
      delete aeskey;
      
      if(data != result)
	throw test_exception("AES-256 decryption gave wrong results");    
    }
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t 
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
  
  t = t + 1;
  
  // TEST 4: AES-128 batch encrypt/decrypt batch tests
  try{
    AES aes;
    
    AESKey* aeskey;
    dynamic_bitset key;
    std::vector<dynamic_bitset> data;
    std::vector<dynamic_bitset> result;
    
    
    data.resize(32);
    result.resize(32);
    key.resize(128);
    
    //////////////////////////////////////////////////
    // encrypt one by one, decrypt as a batch
    
    std::cout << "AES BATCH DECRYPTION TESTS" << std::endl;
    
    {
      for(unsigned int i=0;i<key.size();i++)
	key.set(i, rand() & 1);
      
      aeskey = new AESKey(key);
      
      for(unsigned int k=0;k<data.size();k++){
	data[k].resize(128);
	result[k].resize(128);
	
	for(unsigned int i=0;i<data[k].size();i++)
	  data[k].set(i, rand() & 1);            
	
	result[k] = data[k];
      }
      
      for(unsigned int k=0;k<data.size();k++)
	aes.encrypt(data[k], *aeskey);
      
      DynamicBitsetVectorSource* dbvs = new DynamicBitsetVectorSource(&data);
      
      aes.decrypt(*dbvs, *aeskey, IV, ECBmode);
      
      delete dbvs;
      delete aeskey;
      
      // compares results
      for(unsigned int k=0;k<data.size();k++){
	if(data[k] != result[k])
	  throw test_exception("*BATCH* AES-128 decryption gave wrong results.");
      }
    }
    
      
    
    
    
    //////////////////////////////////////////////////
    // encrypt as a batch, decrypt one by one
    
    std::cout << "AES BATCH ENCRYPTION TESTS" << std::endl;
    
    {
      for(unsigned int i=0;i<key.size();i++)
	key.set(i, rand() & 1);
      
      aeskey = new AESKey(key);
      
      for(unsigned int k=0;k<data.size();k++){
	data[k].resize(128);
	result[k].resize(128);
	
	for(unsigned int i=0;i<data[k].size();i++)
	  data[k].set(i, rand() & 1);            
	
	result[k] = data[k];
      }
      
      
      DynamicBitsetVectorSource* dbvs = new DynamicBitsetVectorSource(&data);
      aes.encrypt(*dbvs, *aeskey, IV, ECBmode);
      
      delete dbvs;
      
      for(unsigned int k=0;k<data.size();k++)
	aes.decrypt(data[k], *aeskey);
      
      delete aeskey;
      
      // compares results
      for(unsigned int k=0;k<data.size();k++){
	if(data[k] != result[k])
	  throw test_exception("*BATCH* AES-128 encryption gave wrong results.");
      }
    }
    
    
    for(int modeNumber=(int)ECBmode;modeNumber<=(int)CTRmode;modeNumber++){
      
      std::cout << "AES MODE OF OPERATION: ";
      
      if(modeNumber == ECBmode) std::cout << "ECB mode" << std::endl;
      else if(modeNumber == CFBmode) std::cout << "CFB mode" << std::endl;
      else if(modeNumber == CBCmode) std::cout << "CBC mode" << std::endl;
      else if(modeNumber == OFBmode) std::cout << "OFB mode" << std::endl;
      else if(modeNumber == CTRmode) std::cout << "CTR mode" << std::endl;
      
      //////////////////////////////////////////////////
      // encrypt batch of data, decrypt batch of data
      
      std::cout << "AES BATCH ENCRYPT-DECRYPT TEST" << std::endl;
      
      {
	for(unsigned int i=0;i<key.size();i++)
	  key.set(i, rand() & 1);
	
	aeskey = new AESKey(key);
	
	for(unsigned int k=0;k<data.size();k++){
	  data[k].resize(128);
	  result[k].resize(128);
	  
	  for(unsigned int i=0;i<data[k].size();i++)
	    data[k].set(i, rand() & 1);
	  
	  result[k] = data[k];
	}
	
	
	DynamicBitsetVectorSource* dbvs = new DynamicBitsetVectorSource(&data);
	aes.encrypt(*dbvs, *aeskey, IV, (ModeOfOperation)modeNumber);
	aes.decrypt(*dbvs, *aeskey, IV, (ModeOfOperation)modeNumber);
	
	delete dbvs;
	delete aeskey;
	
	// compares results
	for(unsigned int k=0;k<data.size();k++){
	  if(data[k] != result[k])
	    throw test_exception("*BATCH* AES-128 decrypt(encrypt(x)) gave wrong results.");
	}
	
      }
    }
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t 
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
}
  
  
////////////////////////////////////////////////////////////


void test_sha()
{
  // TEST 1
  // SHA repeated running and function tests
  try{
    std::cout << "SHA FUNCTIONS PRIMITIVE TESTS" << std::endl;
    
    whiteice::crypto::SHA SHA160(160);
    whiteice::crypto::SHA SHA256(256);
    whiteice::crypto::SHA SHA384(384);
    whiteice::crypto::SHA SHA512(512);
    
    unsigned char* message;
    char hash160[20], hash256[32], hash384[48], hash512[64];
    char xhash[64], ehash[64];
    unsigned int len = rand() % 1432;
    
    message = (unsigned char*)malloc(sizeof(char) * len);
    
    for(unsigned int i=0;i<len;i++)
      message[i] = rand() % 256;
    
    for(unsigned int i=0;i<64;i++)
      ehash[i] = rand() % 256; // wrong hash (with very high probability)
    
    len *= 8; // bytes -> bits    
    
    if(SHA160.hash(&message, len, (unsigned char*)hash160) == false)
      throw test_exception("SHA160 hash() function error.");
    
    if(SHA256.hash(&message, len, (unsigned char*)hash256) == false)
      throw test_exception("SHA256 hash() function error.");
    
    if(SHA384.hash(&message, len, (unsigned char*)hash384) == false)
      throw test_exception("SHA384 hash() function error.");
    
    if(SHA512.hash(&message, len, (unsigned char*)hash512) == false)
      throw test_exception("SHA512 hash() function error.");
    
    // calculates again and use check()
    
    if(SHA160.hash(&message, len, (unsigned char*)xhash) == false)
      throw test_exception("SHA160 hash() function error.");
    
    if(SHA160.check((unsigned char*)hash160, (unsigned char*)xhash) == false)
      throw test_exception("unexpected SHA160 check() failure.");
    
    if(SHA160.check((unsigned char*)hash160, (unsigned char*)ehash) == true)
      throw test_exception("unexpected SHA160 check() success.");
    
    
    if(SHA256.hash(&message, len, (unsigned char*)xhash) == false)
      throw test_exception("SHA256 hash() function error.");
    
    if(SHA256.check((unsigned char*)hash256, (unsigned char*)xhash) == false)
      throw test_exception("unexpected SHA256 check() failure.");
    
    if(SHA256.check((unsigned char*)hash256, (unsigned char*)ehash) == true)
      throw test_exception("unexpected SHA256 check() success.");
    
    
    if(SHA384.hash(&message, len, (unsigned char*)xhash) == false)
      throw test_exception("SHA384 hash() function error.");

    if(SHA384.check((unsigned char*)hash384, (unsigned char*)xhash) == false)
      throw test_exception("unexpected SHA384 check() failure.");
    
    if(SHA384.check((unsigned char*)hash384, (unsigned char*)ehash) == true)
      throw test_exception("unexpected SHA384 check() success.");
    
    
    if(SHA512.hash(&message, len, (unsigned char*)xhash) == false)
      throw test_exception("SHA512 hash() function error.");
    
    if(SHA512.check((unsigned char*)hash512, (unsigned char*)xhash) == false)
      throw test_exception("unexpected SHA512 check() failure.");
    
    if(SHA512.check((unsigned char*)hash512, (unsigned char*)ehash) == true)
      throw test_exception("unexpected SHA512 check() success.");    
    
    
    if(SHA160.bits() != 160) throw test_exception("SHA bits() function error.");
    if(SHA256.bits() != 256) throw test_exception("SHA bits() function error.");
    if(SHA384.bits() != 384) throw test_exception("SHA bits() function error.");
    if(SHA512.bits() != 512) throw test_exception("SHA bits() function error.");

    if(SHA160.bytes() != 20) throw test_exception("SHA bytes() function error.");
    if(SHA256.bytes() != 32) throw test_exception("SHA bytes() function error.");
    if(SHA384.bytes() != 48) throw test_exception("SHA bytes() function error.");
    if(SHA512.bytes() != 64) throw test_exception("SHA bytes() function error.");
    
    free(message); // known bug above: if exception occurs leaks message memory
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception. " 
	      << e.what() << std::endl;
  }
  
  
  // TEST 2
  // SHA correctness tests against FIPS 180-2 examples
  try{
    std::cout << "SHA CORRECTNESS TESTS" << std::endl;
   
    {
      whiteice::crypto::SHA SHA160(160);
      
      unsigned char* message;
      unsigned char hash160[20];      
      unsigned char xhash[20];
      
      // short (one block) message test
      
      unsigned int len = 3;
      
      message = (unsigned char*)malloc(sizeof(char) * (len+1));
      strcpy((char*)message,"abc");
      
      ((whiteice::uint32*)xhash)[0] = 0xa9993e36;
      ((whiteice::uint32*)xhash)[1] = 0x4706816a;
      ((whiteice::uint32*)xhash)[2] = 0xba3e2571;
      ((whiteice::uint32*)xhash)[3] = 0x7850c26c;
      ((whiteice::uint32*)xhash)[4] = 0x9cd0d89d;
      
      for(unsigned int t=0;t<5;t++)
	change_endianess( ((whiteice::uint32*)xhash)[t] );
      
      if(SHA160.hash(&message, len, (unsigned char*)hash160) == false)
	throw test_exception("SHA160 hash() function error.");
      
      if(SHA160.check(hash160, xhash) == false)
	throw test_exception("SHA160 check() failed with correct hash [short]");
      
      free(message);
      
      // long message test
      
      len = 1000000; // 1 million
      message = (unsigned char*)malloc(sizeof(char) * (len+1));
      for(unsigned int i=0;i<len;i++)
	message[i] = 'a';
      
      ((whiteice::uint32*)xhash)[0] = 0x34aa973c;
      ((whiteice::uint32*)xhash)[1] = 0xd4c4daa4;
      ((whiteice::uint32*)xhash)[2] = 0xf61eeb2b;
      ((whiteice::uint32*)xhash)[3] = 0xdbad2731;
      ((whiteice::uint32*)xhash)[4] = 0x6534016f;
      
      for(unsigned int t=0;t<5;t++)
	change_endianess( ((whiteice::uint32*)xhash)[t] );
      
      if(SHA160.hash(&message, len, (unsigned char*)hash160) == false)
	throw test_exception("SHA160 hash() function error.");
      
      if(SHA160.check(hash160, xhash) == false)
	throw test_exception("SHA160 check() failed with correct hash [long]");
      
      free(message);
    }
    
    
    {
      whiteice::crypto::SHA SHA256(256);
      
      unsigned char* message;
      unsigned char hash256[32];
      unsigned char xhash[32];
      
      // short (one block) message test
      
      unsigned int len = 3;
      
      message = (unsigned char*)malloc(sizeof(char) * (len+1));
      strcpy((char*)message,"abc");
      
      ((whiteice::uint32*)xhash)[0] = 0xba7816bf;
      ((whiteice::uint32*)xhash)[1] = 0x8f01cfea;
      ((whiteice::uint32*)xhash)[2] = 0x414140de;
      ((whiteice::uint32*)xhash)[3] = 0x5dae2223;
      ((whiteice::uint32*)xhash)[4] = 0xb00361a3;
      ((whiteice::uint32*)xhash)[5] = 0x96177a9c;
      ((whiteice::uint32*)xhash)[6] = 0xb410ff61;
      ((whiteice::uint32*)xhash)[7] = 0xf20015ad;

      for(unsigned int t=0;t<8;t++)
	change_endianess( ((whiteice::uint32*)xhash)[t] );
      
      if(SHA256.hash(&message, len, (unsigned char*)hash256) == false)
	throw test_exception("SHA256 hash() function error.");
      
      if(SHA256.check(hash256, xhash) == false)
	throw test_exception("SHA256 check() failed with correct hash [short]");
      
      free(message);
      
      
      // long message test
      
      len = 1000000; // 1 million
      message = (unsigned char*)malloc(sizeof(char) * (len+1));
      for(unsigned int i=0;i<len;i++)
	message[i] = 'a';
      
      ((whiteice::uint32*)xhash)[0] = 0xcdc76e5c;
      ((whiteice::uint32*)xhash)[1] = 0x9914fb92;
      ((whiteice::uint32*)xhash)[2] = 0x81a1c7e2;
      ((whiteice::uint32*)xhash)[3] = 0x84d73e67;
      ((whiteice::uint32*)xhash)[4] = 0xf1809a48;
      ((whiteice::uint32*)xhash)[5] = 0xa497200e;
      ((whiteice::uint32*)xhash)[6] = 0x046d39cc;
      ((whiteice::uint32*)xhash)[7] = 0xc7112cd0;

      for(unsigned int t=0;t<8;t++)
	change_endianess( ((whiteice::uint32*)xhash)[t] );
      
      
      if(SHA256.hash(&message, len, (unsigned char*)hash256) == false)
	throw test_exception("SHA256 hash() function error.");
      
      if(SHA256.check(hash256, xhash) == false)
	throw test_exception("SHA256 check() failed with correct hash [long]");
      
      free(message);
    }
    
    
    {
      whiteice::crypto::SHA SHA384(384);
      
      unsigned char* message;
      unsigned char hash384[48];
      unsigned char xhash[48];
      
      // short (one block) message test
      
      unsigned int len = 3;
      
      message = (unsigned char*)malloc(sizeof(char) * (len+1));
      strcpy((char*)message,"abc");
      
      ((whiteice::uint64*)xhash)[0] = 0xcb00753f45a35e8bULL;
      ((whiteice::uint64*)xhash)[1] = 0xb5a03d699ac65007ULL;
      ((whiteice::uint64*)xhash)[2] = 0x272c32ab0eded163ULL;
      ((whiteice::uint64*)xhash)[3] = 0x1a8b605a43ff5bedULL;
      ((whiteice::uint64*)xhash)[4] = 0x8086072ba1e7cc23ULL;
      ((whiteice::uint64*)xhash)[5] = 0x58baeca134c825a7ULL;
      
      for(unsigned int t=0;t<6;t++)
	change_endianess( ((whiteice::uint64*)xhash)[t] );
      
      
      if(SHA384.hash(&message, len, (unsigned char*)hash384) == false)
	throw test_exception("SHA384 hash() function error.");
      
      if(SHA384.check(hash384, xhash) == false)
	throw test_exception("SHA384 check() failed with correct hash [short]");
      
      free(message);
      
      
      // long message test
      
      len = 1000000; // 1 million
      message = (unsigned char*)malloc(sizeof(char) * (len+1));
      for(unsigned int i=0;i<len;i++)
	message[i] = 'a';
      
      ((whiteice::uint64*)xhash)[0] = 0x9d0e1809716474cbULL;
      ((whiteice::uint64*)xhash)[1] = 0x086e834e310a4a1cULL;
      ((whiteice::uint64*)xhash)[2] = 0xed149e9c00f24852ULL;
      ((whiteice::uint64*)xhash)[3] = 0x7972cec5704c2a5bULL;
      ((whiteice::uint64*)xhash)[4] = 0x07b8b3dc38ecc4ebULL;
      ((whiteice::uint64*)xhash)[5] = 0xae97ddd87f3d8985ULL;
      
      for(unsigned int t=0;t<6;t++)
	change_endianess( ((whiteice::uint64*)xhash)[t] );
      
      
      if(SHA384.hash(&message, len, (unsigned char*)hash384) == false)
	throw test_exception("SHA384 hash() function error.");
      
      if(SHA384.check(hash384, xhash) == false)
	throw test_exception("SHA384 check() failed with correct hash [long]");
      
      free(message);
    }
    
    
    {
      whiteice::crypto::SHA SHA512(512);
      
      unsigned char* message;
      unsigned char hash512[64];
      unsigned char xhash[64];
      
      // short (one block) message test
      
      unsigned int len = 3;
      
      message = (unsigned char*)malloc(sizeof(char) * (len+1));
      strcpy((char*)message,"abc");
      
      ((whiteice::uint64*)xhash)[0] = 0xddaf35a193617abaULL;
      ((whiteice::uint64*)xhash)[1] = 0xcc417349ae204131ULL;
      ((whiteice::uint64*)xhash)[2] = 0x12e6fa4e89a97ea2ULL;
      ((whiteice::uint64*)xhash)[3] = 0x0a9eeee64b55d39aULL;
      ((whiteice::uint64*)xhash)[4] = 0x2192992a274fc1a8ULL;
      ((whiteice::uint64*)xhash)[5] = 0x36ba3c23a3feebbdULL;
      ((whiteice::uint64*)xhash)[6] = 0x454d4423643ce80eULL;
      ((whiteice::uint64*)xhash)[7] = 0x2a9ac94fa54ca49fULL;
      
      for(unsigned int t=0;t<8;t++)
	change_endianess( ((whiteice::uint64*)xhash)[t] );
      
      if(SHA512.hash(&message, len, (unsigned char*)hash512) == false)
	throw test_exception("SHA512 hash() function error.");
      
      if(SHA512.check(hash512, xhash) == false)
	throw test_exception("SHA512 check() failed with correct hash [short]");
      
      free(message);
      
      
      // long message test
      
      len = 1000000; // 1 million
      message = (unsigned char*)malloc(sizeof(char) * (len+1));
      for(unsigned int i=0;i<len;i++)
	message[i] = 'a';
      
      ((whiteice::uint64*)xhash)[0] = 0xe718483d0ce76964ULL;
      ((whiteice::uint64*)xhash)[1] = 0x4e2e42c7bc15b463ULL;
      ((whiteice::uint64*)xhash)[2] = 0x8e1f98b13b204428ULL;
      ((whiteice::uint64*)xhash)[3] = 0x5632a803afa973ebULL;
      ((whiteice::uint64*)xhash)[4] = 0xde0ff244877ea60aULL;
      ((whiteice::uint64*)xhash)[5] = 0x4cb0432ce577c31bULL;
      ((whiteice::uint64*)xhash)[6] = 0xeb009c5c2c49aa2eULL;
      ((whiteice::uint64*)xhash)[7] = 0x4eadb217ad8cc09bULL;
      
      for(unsigned int t=0;t<8;t++)
	change_endianess( ((whiteice::uint64*)xhash)[t] );
      
      if(SHA512.hash(&message, len, (unsigned char*)hash512) == false)
	throw test_exception("SHA512 hash() function error.");
      
      if(SHA512.check(hash512, xhash) == false)
	throw test_exception("SHA512 check() failed with correct hash [long]");
      
      free(message);
    }

    
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception. " 
	      << e.what() << std::endl;
  }
  
}


////////////////////////////////////////////////////////////


void rsa_test()
{
  using namespace whiteice::crypto;
  using namespace whiteice;
  
  unsigned int t = 0;
  
  
  t = 1;
  
  // TEST 1: RSA key generation test
  try{
    std::cout << "WARNING: "
	      << "RSA key generation tests not done."
	      << std::endl;
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
  t = 2;
  
  // TEST 2: RSA encryption comparision against
  // other implementation
  try{
    std::cout << "WARNING: "
	      << "RSA encryption comparision tests not done."
	      << std::endl;
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
  t = 3;
  
  // TEST 3: RSA decrypt(encrypt(x)) == x identity test
  try{
    std::cout << "RSA DECRYPT(ENCRYPT(X)) = X TEST" << std::endl;
    
    RSA rsa;
    RSAKey* rsakey;
    
    integer data;
    integer result;
    
    for(unsigned int i=0;i<5;i++){
      
      // new key
      // parameter is length of the modulus, p and q are aprox half of this value
      rsakey = new RSAKey( (rand() % 1024) + 512);
      
      // random value
      data = rand();
      data %= rsakey->publickey()[1]; // makes sure random value is within correct interval
	
      result = data;
      
      if(rsa.encrypt(result, *rsakey) == false)
	throw test_exception("RSA encryption failure.");
	 
      if(rsa.decrypt(result, *rsakey) == false)
	throw test_exception("RSA decryption failure.");
      
      delete rsakey;
      
      if(data != result)
	throw test_exception("RSA encryption/decryption error.");
    }
      
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
}


////////////////////////////////////////////////////////////


void dsa_test()
{
  using namespace whiteice::crypto;
  using namespace whiteice;
  
  unsigned int t = 0;
  
  
  t = 1;
  
  // TEST 1: DSA key generation test
  try{

    std::cout << "DSA key generation test" << std::endl;
    std::cout << "WARN: DSA key generation test assumes DSAKey ctor does self testing"
	      << std::endl;
    
    DSAKey* dsakey;
    
    for(unsigned int i=0;i<10;i++){
      dsakey = new DSAKey(); // generates 
      delete dsakey;
    }
      
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
  t = 2;
  
  // TEST 2: DSA signing comparision against
  // other implementation
  try{
    std::cout << "DSA signing comparision test" << std::endl;
    std::cout << "WARNING: "
	      << "DSA signing comparision tests not done."
	      << std::endl;
    
    
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
  t = 3;
  
  // TEST 3: DSA verify(x, sign(x)) test
  try{
    std::cout << "DSA VERIFY(X, SIGN(X)) = TRUE TESTS" << std::endl;
    
    DSA* dsa;
    
    unsigned char* data;
    unsigned int dlen;
    
    
    for(unsigned int i=0;i<5;i++){
      
      // creates new dsa object & with a new dsakey
      dsa = new DSA();
      
      dlen = rand() % 1024;
      
      data = (unsigned char*)malloc(dlen);
      
      for(unsigned int i=0;i<dlen;i++)
	data[i] = rand() % 0xFF;
      
      std::vector<integer> signature;
      
      if(dsa->sign(&data, dlen, signature) == false)
	throw test_exception("DSA signing failure.");
	 
      if(dsa->verify(&data, dlen, signature) == false)
	throw test_exception("DSA verify failure.");

      free(data);
      
      delete dsa;
    }
    
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
  t = 4;
  
  // TEST 4: DSA verify(modify(x), sign(x)) = false test
  try{
    std::cout << "DSA VERIFY(MODIFY(X), SIGN(X)) = FALSE TESTS" << std::endl;
    
    DSA* dsa;
    
    unsigned char* data;
    unsigned int dlen;
    
    
    for(unsigned int i=0;i<5;i++){
      
      // creates new dsa object & with a new dsakey
      dsa = new DSA();
      
      dlen = rand() % 1024;
      
      data = (unsigned char*)malloc(dlen);
      
      for(unsigned int i=0;i<dlen;i++)
	data[i] = rand() & 0xFF;
      
      std::vector<integer> signature;
      
      if(dsa->sign(&data, dlen, signature) == false) // sign
	throw test_exception("DSA signing failure.");
      
      data[(rand() % dlen)] = rand() & 0xFF; // modify
      
      if(dsa->verify(&data, dlen, signature) == true) // verify
	throw test_exception("DSA verify succeeded with changed message.");

      free(data);
      
      delete dsa;
    }
    
    
  }
  catch(test_exception& e){
    std::cout << "Testcase " << t
	      << " failed: " << e.what() << std::endl;
  }
  catch(std::exception& e){    
    std::cout << "Unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
  
}













/************************************************************/
/************************************************************/


void change_endianess(whiteice::uint32& x)
{
  x = ((x >> 24) | ((x >> 8) & 0xFF00) | ((x & 0xFF00) << 8) | (x << 24));
}


void change_endianess(whiteice::uint64& x)
{
  x = ((x >> 56) | ((x >> 40) & 0xFF00) | ((x >> 24) & 0xFF0000) |
       ((x >> 8) & 0xFF000000) | ((x & 0xFF000000) << 8) |
       ((x & 0xFF0000) << 24) | ((x & 0xFF00) << 40) | ((x & 0xFF) << 56));
}

