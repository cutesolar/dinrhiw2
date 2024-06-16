/*
 * Advanced Encyption Standard (AES)
 */

#include "AES.h"
#include "dynamic_bitset.h"
#include <stdexcept>
#include <new>
#include "global.h"
#include "RNG.h"

#include <stdlib.h>


namespace whiteice
{
  namespace crypto
  {
    
    AES::AES()
    {
      GFMULTI = new unsigned char[256*256]; // 65 Kb
      
      gf_precalculate(); // precalculates GF(2^8) multiplication table

      // self_test();
    }
    
    AES::~AES()
    {
      s.reset();
      
      if(GFMULTI) delete[] GFMULTI;
    }
    
    
    // data must be X bit and Keyschedule should be AESKey
    bool AES::encrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) 
    {
      try{
	// input data must be 128 bits
	if(data.size() != 128) return false;
	
	// AES key must be at least 128 bits
	// (sizes 128, 192, 256 are offical ones,
	//  values in between maybe insecure, values greater than above
	//  may not increase security considerably (afaik at least rijndael-512 
	//  should be implemented somewhat differently)
	// (must have at least 2 rounds)
	if(k.size() < 2) return false;
	
	
	unsigned int r = 0;
	
	data ^= k[r];        // AddRoundKey()
	
	for(r=1;r<(k.size() - 1);r++){
	  substitute(data);  // SubBytes()
	  shift_rows(data);  // ShiftRows()
	  mix_columns(data); // MixColumns()
	  
	  data ^= k[r];      // AddRoundKey()
	}
	
	substitute(data);    // SubBytes()
	shift_rows(data);    // ShiftRows()
	
	data ^= k[r];
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    bool AES::decrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) 
    {
      try{
	// input data must be 128 bits
	if(data.size() != 128) return false;
	
	// number of keys/rounds must be at least two
	if(k.size() < 2) return false;
	
	unsigned int r = k.size() - 1;
	
	data ^= k[r];            // AddRoundKey()
	
	for(r=k.size() - 2;r>=1;r--){
	  inv_shift_rows(data);  // InvShiftRows()
	  inv_substitute(data);  // InvSubBytes()
	  data ^= k[r];          // AddRoundKey()
	  inv_mix_columns(data); // InvMixColumns()
	}
	
	inv_shift_rows(data);    // InvShiftRows()
	inv_substitute(data);    // InvSubBytes()
	
	data ^= k[r];            // AddRoundKey()
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }	
    }    
    

    
    bool AES::encrypt(data_source<dynamic_bitset>& data,
		      const Keyschedule<dynamic_bitset>& k, const dynamic_bitset& IV,
		      ModeOfOperation mode) 
    {
      try{
	if(data.size() <= 0) return true;
	
	// input data must be 128 bits
	if(data[0].size() != 128) return false;
	
	// AES key must be at least 128 bits
	// (sizes 128, 192, 256 are offical ones,
	//  values in between maybe insecure, values greater than above
	//  may not increase security considerably (afaik at least rijndael-512 
	//  should be implemented somewhat differently)
	
	// (must have at least 2 rounds)
	if(k.size() < 2) return false;			
	  	
	dynamic_bitset di;
	
	if(mode == CTRmode){
	  dynamic_bitset CTR = IV;
	  
	  for(unsigned int i=0;i<data.size();i++){
	    unsigned int r = 0;
	    
	    di = CTR ^ k[r];  // AddRoundKey()
	    CTR.inc(); // increases counter
	    
	    for(r=1;r<(k.size() - 1);r++){
	      substitute(di);  // SubBytes()
	      shift_rows(di);  // ShiftRows()
	      mix_columns(di); // MixColumns()
	      di ^= k[r];      // AddRoundKey()
	    }
	    
	    substitute(di);    // SubBytes()
	    shift_rows(di);    // ShiftRows()
	    
	    di ^= k[r];        // di = encrypt(CTR)
	    
	    data[i] ^= di;
	  }
	  
	}
	else if(mode == CBCmode){
	  dynamic_bitset prev = IV;
	  
	  for(unsigned int i=0;i<data.size();i++){
	    unsigned int r = 0;
	    
	    di = (data[i] ^ prev) ^ k[r];  // add prev value and AddRoundKey()
	    
	    for(r=1;r<(k.size() - 1);r++){
	      substitute(di);  // SubBytes()
	      shift_rows(di);  // ShiftRows()
	      mix_columns(di); // MixColumns()
	      di ^= k[r];      // AddRoundKey()
	    }
	    
	    substitute(di);    // SubBytes()
	    shift_rows(di);    // ShiftRows()
	    
	    prev    = di ^ k[r];
	    data[i] = prev;
	  }
	  
	}
	else if(mode == OFBmode){
	  dynamic_bitset z = IV;
	  
	  for(unsigned int i=0;i<data.size();i++){
	    unsigned int r = 0;
	    
	    di = z ^ k[r];  // AddRoundKey()
	    
	    for(r=1;r<(k.size() - 1);r++){
	      substitute(di);  // SubBytes()
	      shift_rows(di);  // ShiftRows()
	      mix_columns(di); // MixColumns()
	      di ^= k[r];      // AddRoundKey()
	    }
	    
	    substitute(di);    // SubBytes()
	    shift_rows(di);    // ShiftRows()
	    
	    z = di ^ k[r];
	    
	    data[i] ^= z;
	  }
	  
	}
	else if(mode == CFBmode){
	  dynamic_bitset prev = IV;
	  	  
	  for(unsigned int i=0;i<data.size();i++){
	    unsigned int r = 0;
	    
	    di = prev ^ k[r];  // AddRoundKey()
	    
	    for(r=1;r<(k.size() - 1);r++){
	      substitute(di);  // SubBytes()
	      shift_rows(di);  // ShiftRows()
	      mix_columns(di); // MixColumns()
	      di ^= k[r];      // AddRoundKey()
	    }
	    
	    substitute(di);    // SubBytes()
	    shift_rows(di);    // ShiftRows()
	    
	    di ^= k[r]; // di = encrypt(prev)
	    
	    data[i] ^= di;
	    
	    prev = data[i];
	  }
	  
	}
	else if(mode == ECBmode){
	  
	  for(unsigned int i=0;i<data.size();i++){
	    unsigned int r = 0;
	    
	    di = data[i] ^ k[r];  // AddRoundKey()
	    
	    for(r=1;r<(k.size() - 1);r++){
	      substitute(di);  // SubBytes()
	      shift_rows(di);  // ShiftRows()
	      mix_columns(di); // MixColumns()
	      di ^= k[r];      // AddRoundKey()
	    }
	    
	    substitute(di);    // SubBytes()
	    shift_rows(di);    // ShiftRows()
	    
	    data[i] = di ^ k[r];
	  }
	  
	}
	else return false;
	
	
	data.flush();
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }      
    }
    
    
    
    bool AES::decrypt(data_source<dynamic_bitset>& data,
		      const Keyschedule<dynamic_bitset>& k, const dynamic_bitset& IV,
		      ModeOfOperation mode) 
    {
      try{
	if(data.size() <= 0) return true;
	
	// input data must be 128 bits
	if(data[0].size() != 128) return false;
	
	// number of keys/rounds must be at least two
	if(k.size() < 2) return false;
	
	if(IV.size() != 128 && mode != ECBmode) 
	  return false; // IV length must be same as block size (if used)

	
	dynamic_bitset di;
	
	
	if(mode == CTRmode){
	  
	  dynamic_bitset CTR = IV;
	  
	  for(unsigned int i=0;i<data.size();i++){
	    unsigned int r = 0;
	    
	    di = CTR ^ k[r];  // AddRoundKey()
	    CTR.inc(); // increases counter
	    
	    for(r=1;r<(k.size() - 1);r++){
	      substitute(di);  // SubBytes()
	      shift_rows(di);  // ShiftRows()
	      mix_columns(di); // MixColumns()
	      di ^= k[r];      // AddRoundKey()
	    }
	    
	    substitute(di);    // SubBytes()
	    shift_rows(di);    // ShiftRows()
	    
	    di ^= k[r];        // di = encrypt(CTR)
	    
	    data[i] ^= di;
	  }
	  
	}
	else if(mode == CBCmode){
	  dynamic_bitset temp, prev = IV;
	  
	  for(unsigned int i=0;i<data.size();i++){
	    unsigned int r = k.size() - 1;
	    
	    temp = data[i];
	    di = data[i] ^ k[r];   // AddRoundKey()
	    
	    for(r=k.size() - 2;r>=1;r--){
	      inv_shift_rows(di);  // InvShiftRows()
	      inv_substitute(di);  // InvSubBytes()
	      di ^= k[r];          // AddRoundKey()
	      inv_mix_columns(di); // InvMixColumns()
	    }
	    
	    inv_shift_rows(di);    // InvShiftRows()
	    inv_substitute(di);    // InvSubBytes()
	    
	    data[i] = di ^ k[r];   // AddRoundKey()
	    
	    data[i] = data[i] ^ prev;
	    prev = temp;
	  }	  
	  
	}
	else if(mode == OFBmode){
	  dynamic_bitset z = IV;
	  
	  for(unsigned int i=0;i<data.size();i++){
	    unsigned int r = 0;
	    
	    di = z ^ k[r];  // AddRoundKey()
	    
	    for(r=1;r<(k.size() - 1);r++){
	      substitute(di);  // SubBytes()
	      shift_rows(di);  // ShiftRows()
	      mix_columns(di); // MixColumns()
	      di ^= k[r];      // AddRoundKey()
	    }
	    
	    substitute(di);    // SubBytes()
	    shift_rows(di);    // ShiftRows()
	    
	    z = di ^ k[r];
	    
	    data[i] ^= z;
	  }
	  
	}
	else if(mode == CFBmode){
	  dynamic_bitset prev = IV;
	  
	  for(unsigned int i=0;i<data.size();i++){
	    unsigned int r = 0;
	    
	    di = prev ^ k[r];  // AddRoundKey()
	    
	    for(r=1;r<(k.size() - 1);r++){
	      substitute(di);  // SubBytes()
	      shift_rows(di);  // ShiftRows()
	      mix_columns(di); // MixColumns()
	      di ^= k[r];      // AddRoundKey()
	    }
	    
	    substitute(di);    // SubBytes()
	    shift_rows(di);    // ShiftRows()
	    
	    di ^= k[r]; // di = encrypt(prev)
	    
	    prev = data[i];
	    data[i] ^= di;
	  }
	  
	}
	else if(mode == ECBmode){
	
	  for(unsigned int i=0;i<data.size();i++){
	    unsigned int r = k.size() - 1;
	    
	    di = data[i] ^ k[r];   // AddRoundKey()
	    
	    for(r=k.size() - 2;r>=1;r--){
	      inv_shift_rows(di);  // InvShiftRows()
	      inv_substitute(di);  // InvSubBytes()
	      di ^= k[r];          // AddRoundKey()
	      inv_mix_columns(di); // InvMixColumns()
	    }
	    
	    inv_shift_rows(di);    // InvShiftRows()
	    inv_substitute(di);    // InvSubBytes()
	    
	    data[i] = di ^ k[r];   // AddRoundKey()
	  }
	  
	}
	else return false;
	  
	  
	data.flush();
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }	
    }
    
    
    //////////////////////////////////////////////////////////////////////
    
    void AES::substitute(dynamic_bitset& data) const
    {
      const unsigned int N = data.blocks();      
      
      // gcc-3.4.3 compiled code never prints this message
      // (don't make substitute() calls -- too aggressive wrong optimization?,
      //  dynmaic_bitset still changes?)
      
      for(unsigned int i=0;i<N;i++)
	data.value(i) = SBOX1[data.value(i)];
	
    }
    
    
    void AES::inv_substitute(dynamic_bitset& data) const
    {
      const unsigned int N = data.blocks();      
      
      for(unsigned int i=0;i<N;i++)
	data.value(i) = SBOX2[data.value(i)];
    }
    
    
    void AES::shift_rows(dynamic_bitset& data) const
    {
      s = data;
      
      // assumes data is 128 bit long (inner loop: j<4, % 4 )
      
      data.value( 0x00 ) = s.value( 0x00 );
      data.value( 0x04 ) = s.value( 0x04 );
      data.value( 0x08 ) = s.value( 0x08 );
      data.value( 0x0c ) = s.value( 0x0c );

      data.value( 0x01 ) = s.value( 0x05 );
      data.value( 0x05 ) = s.value( 0x09 );
      data.value( 0x09 ) = s.value( 0x0d );
      data.value( 0x0d ) = s.value( 0x01 );

      data.value( 0x02 ) = s.value( 0x0a );
      data.value( 0x06 ) = s.value( 0x0e );
      data.value( 0x0a ) = s.value( 0x02 );
      data.value( 0x0e ) = s.value( 0x06 );

      data.value( 0x03 ) = s.value( 0x0f );
      data.value( 0x07 ) = s.value( 0x03 );
      data.value( 0x0b ) = s.value( 0x07 );
      data.value( 0x0f ) = s.value( 0x0b );
      
      
      //
      //// (rows, columns) = (i,j)
      //for(unsigned int i=0;i<4;i++){
      //	for(unsigned int j=0;j<4;j++){
      //	  data.value(i + 4*j) = s.value( i + 4 * mod(((signed int)j) + rowshifts[i], 4) );
      //	}
      //}
      //
      
    }
    
    
    void AES::inv_shift_rows(dynamic_bitset& data) const
    {
      s = data;
      
      // assumes data is 128 bit long (inner loop: j<4, % 4 )
      
      data.value( 0x00 ) = s.value( 0x00 );
      data.value( 0x04 ) = s.value( 0x04 );
      data.value( 0x08 ) = s.value( 0x08 );
      data.value( 0x0c ) = s.value( 0x0c );

      data.value( 0x01 ) = s.value( 0x0d );
      data.value( 0x05 ) = s.value( 0x01 );
      data.value( 0x09 ) = s.value( 0x05 );
      data.value( 0x0d ) = s.value( 0x09 );

      data.value( 0x02 ) = s.value( 0x0a );
      data.value( 0x06 ) = s.value( 0x0e );
      data.value( 0x0a ) = s.value( 0x02 );
      data.value( 0x0e ) = s.value( 0x06 );

      data.value( 0x03 ) = s.value( 0x07 );
      data.value( 0x07 ) = s.value( 0x0b );
      data.value( 0x0b ) = s.value( 0x0f );
      data.value( 0x0f ) = s.value( 0x03 );
      
      //
      // for(unsigned int i=0;i<4;i++){
      //	for(unsigned int j=0;j<4;j++){
      //	  data.value(i + 4*j) = s.value( i + 4 * mod(((signed int)j) - rowshifts[i], 4) );
      //	}
      //}
      //
    }
    
    
    void AES::mix_columns(dynamic_bitset& data) const
    {
      s = data;
            
      unsigned char a, b, c, d;
      
      for(unsigned int col=0;col<4;col++){
	a = s.value(0 + col*4);
	b = s.value(1 + col*4);
	c = s.value(2 + col*4);
	d = s.value(3 + col*4);
	
	a = GFMULTI[ 2*256 + a]; // {02} * {a} (galois field multi)
	b = GFMULTI[ 3*256 + b]; // {03} * {b}
	
	data.value(0 + col*4) = a ^ b ^ c ^ d;
      }
      
      for(unsigned int col=0;col<4;col++){
	a = s.value(0 + col*4);
	b = s.value(1 + col*4);
	c = s.value(2 + col*4);
	d = s.value(3 + col*4);
	
	b = GFMULTI[ 2*256 + b]; // {02} * {b}
	c = GFMULTI[ 3*256 + c]; // {03} * {c}
	
	data.value(1 + col*4) = a ^ b ^ c ^ d;
      }
      
      for(unsigned int col=0;col<4;col++){
	a = s.value(0 + col*4);
	b = s.value(1 + col*4);
	c = s.value(2 + col*4);
	d = s.value(3 + col*4);
	
	c = GFMULTI[ 2*256 + c]; // {02} * {c}
	d = GFMULTI[ 3*256 + d]; // {03} * {d}
	
	data.value(2 + col*4) = a ^ b ^ c ^ d;
      }
      
      for(unsigned int col=0;col<4;col++){
	a = s.value(0 + col*4);
	b = s.value(1 + col*4);
	c = s.value(2 + col*4);
	d = s.value(3 + col*4);
	
	a = GFMULTI[ 3*256 + a]; // {03} * {a}
	d = GFMULTI[ 2*256 + d]; // {02} * {d}	
	
	data.value(3 + col*4) = a ^ b ^ c ^ d;
      }
      
      s.reset();
    }
    
    
    void AES::inv_mix_columns(dynamic_bitset& data) const
    {
      s = data;
      
      unsigned char a, b, c, d;
      
      for(unsigned int col=0;col<4;col++){
	a = s.value(0 + col*4);
	b = s.value(1 + col*4);
	c = s.value(2 + col*4);
	d = s.value(3 + col*4);
	
	a = GFMULTI[0x0e * 256 + a]; // {0x0e} * {a} (galois field multi)
	b = GFMULTI[0x0b * 256 + b]; // {0x0b} * {b}
	c = GFMULTI[0x0d * 256 + c]; // {0x0d} * {c} 
	d = GFMULTI[0x09 * 256 + d]; // {0x09} * {d}
	
	data.value(0 + col*4) = a ^ b ^ c ^ d;
      }
      
      for(unsigned int col=0;col<4;col++){
	a = s.value(0 + col*4);
	b = s.value(1 + col*4);
	c = s.value(2 + col*4);
	d = s.value(3 + col*4);
	
	a = GFMULTI[0x09 * 256 + a]; // {0x09} * {a}
	b = GFMULTI[0x0e * 256 + b]; // {0x0e} * {b}
	c = GFMULTI[0x0b * 256 + c]; // {0x0b} * {c}
	d = GFMULTI[0x0d * 256 + d]; // {0x0d} * {d}
	
	data.value(1 + col*4) = a ^ b ^ c ^ d;
      }
      
      for(unsigned int col=0;col<4;col++){
	a = s.value(0 + col*4);
	b = s.value(1 + col*4);
	c = s.value(2 + col*4);
	d = s.value(3 + col*4);
	
	a = GFMULTI[0x0d * 256 + a]; // {0x0d} * {a}
	b = GFMULTI[0x09 * 256 + b]; // {0x09} * {b}
	c = GFMULTI[0x0e * 256 + c]; // {0x0e} * {c}
	d = GFMULTI[0x0b * 256 + d]; // {0x0b} * {d}
	
	data.value(2 + col*4) = a ^ b ^ c ^ d;
      }
      
      for(unsigned int col=0;col<4;col++){
	a = s.value(0 + col*4);
	b = s.value(1 + col*4);
	c = s.value(2 + col*4);
	d = s.value(3 + col*4);
	
	a = GFMULTI[0x0b * 256 + a]; // {0x0b} * {a}
	b = GFMULTI[0x0d * 256 + b]; // {0x0d} * {b}
	c = GFMULTI[0x09 * 256 + c]; // {0x09} * {c}
	d = GFMULTI[0x0e * 256 + d]; // {0x0e} * {d}
	
	data.value(3 + col*4) = a ^ b ^ c ^ d;
      }
      
    }
    
    
    // precalculates multiplication in GF(2^8)
    void AES::gf_precalculate()
    {
      unsigned char a;
      unsigned char b;
      
      unsigned int K;
      
      
      for(unsigned int i=0;i<256;i++){
	for(unsigned int j=0;j<256;j++){
	  
	  // calculates "i * j"
	  
	  a = i;
	  b = j;
	  
	  GFMULTI[i + j*256] = 0;
	  
	  K = 0;
	  
	  while(a){
	    if(a & 1){	      	      
	      // result += "polynom of j" * x^K
	      b = j;
	      
	      for(unsigned int k=0;k<K;k++)  // multiplication by x^K term
		xmulti(b);
	      
	      GFMULTI[i + j*256] ^= b; // addition in mod 2 is xor
	    }
	    
	    a >>= 1;
	    K++;
	  }
	  
	}
      }
      
    }
    
    
    // calculates "x * p(x)" in GF(2^8)
    // (where x^8 + m(x) = 0, degree(m(x)) < 8) m(x) = x^4 + x^3 + x + 1  (in AES)
    void AES::xmulti(unsigned char& p) const
    {
      // (idea for this came from NIST's FIPS 197 AES specification)
      
      if(p & 0x80){ // has the highest bit set (-> has x^8 after multi))
	p <<= 1;    // x * p(x) + drops possible x^8 term
	p ^= 0x1B;  // substitutes dropped x^8' with -m(x) and adds it
      }
      else{
	p <<= 1;    // shift is enough (no x^8 term)
      }
    }
    
    
    // calculates smallest positive conquence in modular arithmetic
    unsigned int AES::mod(int i, unsigned int modulo) const 
    {
      int r = (i % modulo);
      
      if(r < 0) r += modulo;
      
      return ((unsigned int)r);
    }
    
    
    void AES::self_test()
    {
      dynamic_bitset a, b;
      a.resize(128);
      b.resize(128);
      
      // substitute test
      for(unsigned int i=0;i<1000;i++){
	
	for(unsigned int j=0;j<a.blocks();j++){
	  a.value(j) = whiteice::rng.rand() & 0xFF;
	}
	b = a;
	
	substitute(a);
	inv_substitute(a);
	
	if(a != b){
	  std::cout << "AES SUBSTITUTE SELF TEST FAILED!"
		    << std::endl;
	}
      }
      
      
      // shift_rows test
      for(unsigned int i=0;i<1000;i++){
	
	for(unsigned int j=0;j<a.blocks();j++){
	  a.value(j) = whiteice::rng.rand() & 0xFF;
	}
	b = a;
	
	shift_rows(a);
	inv_shift_rows(a);
	
	if(a != b){
	  std::cout << "AES SHIFT ROWS SELF TEST FAILED!"
		    << std::endl;
	}
      }
      
      
      // mix_columns test
      for(unsigned int i=0;i<1000;i++){
	
	for(unsigned int j=0;j<a.blocks();j++){
	  a.value(j) = whiteice::rng.rand() & 0xFF;
	}
	b = a;
	
	mix_columns(a);
	inv_mix_columns(a);
	
	if(a != b){
	  std::cout << "AES MIX COLS SELF TEST FAILED!"
		    << std::endl;
	}
      }
      
      
    }

    
    //////////////////////////////////////////////////////////////////////
    
    
    void change_endianess(whiteice::uint32& x)
    {
      x = ((x >> 24) | ((x >> 8) & 0xFF00) | ((x & 0xFF00) << 8) | (x << 24));
    }
    
    
    AESKey::AESKey(const dynamic_bitset& key)
    {
      if(key.size() <= 0)
	throw std::invalid_argument("Key can't be zero bits long");
      
      if(key.size() % 32)
	throw std::invalid_argument("Rijndael key must be multiple of 32 bits");
      
      
      unsigned int Nk = (key.size() / 32);
      unsigned int Nr = Nk + 6;
                  
      
      keys.resize(Nr + 1);      
      
      
      // calculates round keys: the first key is same as given 
      {				
	const unsigned int Nb = 4;
	
	unsigned int i;	
	//unsigned int* w = new unsigned int[Nb*(Nr + 1)];
	whiteice::uint32* w = new whiteice::uint32[Nb*(Nr + 1)];
	
	i = 0;
	
	while(i < Nk){ // swaps order
	  
	  ((unsigned char*)w)[4*i + 0] = key.value(4*i + 0);
	  ((unsigned char*)w)[4*i + 1] = key.value(4*i + 1);
	  ((unsigned char*)w)[4*i + 2] = key.value(4*i + 2);
	  ((unsigned char*)w)[4*i + 3] = key.value(4*i + 3);
	  
	  i++;
	}
	
	i = Nk;
	
	while(i < Nb* (Nr + 1)){
	  w[i] = w[i-1];
	  
	  change_endianess(w[i]);
	  
	  if((i % Nk) == 0){
	    w[i] = substitute_word(rotate_word(w[i])) ^ rcon[(i/Nk) - 1];
	  }
	  else if(Nk > 6 && ((i % Nk) == 4)){
	    w[i] = substitute_word(w[i]);
	  }
	  
	  change_endianess(w[i]);
	  
	  w[i] ^= w[i - Nk];
	  
	  i++;
	}
	
	
	unsigned int j;
	i = 0; j = 0;

	keys[0].resize(Nb*32);
	
	
	for(unsigned int k=0;k<(Nb * (Nr + 1));k++){
	  
	  keys[i].value(j) = (w[k] & 0xFF);
	  
	  j++;
	  
	  if(j >= keys[i].blocks()){
	    i++;
	    j = 0;
	    
	    if(i < keys.size())
	      keys[i].resize(Nb*32);
	  }

	  
	  keys[i].value(j) = ((w[k] >> 8) & 0xFF);
	  
	  j++;
	  
	  if(j >= keys[i].blocks()){
	    i++;
	    j = 0;
	    
	    if(i < keys.size())
	      keys[i].resize(Nb*32);
	  }
	  
	  
	  keys[i].value(j) = ((w[k] >> 16) & 0xFF);
	  
	  j++;
	  
	  if(j >= keys[i].blocks()){
	    i++;
	    j = 0;
	    
	    if(i < keys.size())
	      keys[i].resize(Nb*32);
	  }
	  
	  
	  keys[i].value(j) = ((w[k] >> 24) & 0xFF);
	  
	  j++;
	  
	  if(j >= keys[i].blocks()){
	    i++;
	    j = 0;
	    
	    if(i < keys.size())
	      keys[i].resize(Nb*32);
	  }

	}
	
	
	
	// clears & frees memory
	for(unsigned int k=0;k<(Nb * (Nr + 1));k++)
	  w[k] = 0;
	
	delete[] w;
      }    
      
      
    }
    
    
    AESKey::AESKey(const AESKey& k)
    {
      this->keys.resize(k.keys.size());
      
      for(unsigned int i=0;i<keys.size();i++){
	this->keys[i].resize(k.keys[i].size());
	this->keys[i] = k.keys[i];
      }
    }
    
    
    AESKey::~AESKey()
    {
      // clears key values from the memory
      
      for(unsigned int i=0;i<keys.size();i++)
	keys[i].reset();
    }
    
    
    unsigned int AESKey::size() const 
    {
      return keys.size();
      
      return 0;
    }
    
    
    bool AESKey::resize(unsigned int s) 
    {
      // can only reduce number of keys
      
      if(s > keys.size()) return false;
      else if(s == keys.size()) return true;
      else{
	keys.resize(s);
	return true;
      }
    }
    
    
    unsigned int AESKey::keybits() const 
    {
      if(keys.size() <= 0) return 0;
      else return keys[0].size();
    }
    
    
    const dynamic_bitset& AESKey::operator[](unsigned int n)
      const 
    {
      if(n >= keys.size())
	throw std::out_of_range("index bigger than number of keys");
      
      return keys[n];
    }
    
    
    Keyschedule<dynamic_bitset>* AESKey::copy() const
    {
      return new AESKey(*this);
    }
    
    
    unsigned int AESKey::substitute_word(unsigned int x) const
    {
      unsigned int r = 0;
      
      //
      //r += (AES::SBOX1[(x >> 4) & 0x0F][(x >> 0) & 0x0F] <<  0); x >>= 8;
      //r += (AES::SBOX1[(x >> 4) & 0x0F][x & 0x0F] <<  8); x >>= 8;
      //r += (AES::SBOX1[(x >> 4) & 0x0F][x & 0x0F] << 16); x >>= 8;
      //r += (AES::SBOX1[(x >> 4) & 0x0F][x & 0x0F] << 24);
      //

      r += (AES::SBOX1[x & 0xFF] <<  0); x >>= 8;
      r += (AES::SBOX1[x & 0xFF] <<  8); x >>= 8;
      r += (AES::SBOX1[x & 0xFF] << 16); x >>= 8;
      r += (AES::SBOX1[x & 0xFF] << 24);
      
      return r;
    }
    
    
    unsigned int AESKey::rotate_word(unsigned int x) const
    {
      unsigned int r = x;
      
      r <<= 8;
      x >>= 24;
      
      r += x;
      
      return r;
    }
    
    
    ////////////////////////////////////////////////////////////////////////////////
    // AES constants

    const unsigned char AES::SBOX1[256] = 
    { 
      0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
      0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
      0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
      0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
      0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
      0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
      0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
      0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
      0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
      0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
      0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
      0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
      0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
      0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
      0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
      0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16 
    };

    // inverse of SBOX1
    const unsigned char AES::SBOX2[256] = 
    { 
      0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
      0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
      0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
      0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
      0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
      0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
      0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
      0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
      0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
      0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
      0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
      0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
      0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
      0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
      0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
      0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D 
    };
    
#if 0
    
    const unsigned char AES::SBOX1[16][16] = 
    { /*           0     1     2     3     4     5     6     7     8     9     A     B     C     D     E     F    */
      /* 0 */ { 0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76 },
      /* 1 */ { 0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0 },
      /* 2 */ { 0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15 },
      /* 3 */ { 0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75 },
      /* 4 */ { 0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84 },
      /* 5 */ { 0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF },
      /* 6 */ { 0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8 },
      /* 7 */ { 0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2 },
      /* 8 */ { 0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73 },
      /* 9 */ { 0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB },
      /* A */ { 0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79 },
      /* B */ { 0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08 },
      /* C */ { 0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A },
      /* D */ { 0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E },
      /* E */ { 0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF },
      /* F */ { 0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16 } 
    };

    // inverse of SBOX1
    const unsigned char AES::SBOX2[16][16] = 
    { /*           0     1     2     3     4     5     6     7     8     9     A     B     C     D     E     F    */
      /* 0 */ { 0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB },
      /* 1 */ { 0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB },
      /* 2 */ { 0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E },
      /* 3 */ { 0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25 },
      /* 4 */ { 0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92 },
      /* 5 */ { 0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84 },
      /* 6 */ { 0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06 },
      /* 7 */ { 0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B },
      /* 8 */ { 0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73 },
      /* 9 */ { 0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E },
      /* A */ { 0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B },
      /* B */ { 0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4 },
      /* C */ { 0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F },
      /* D */ { 0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF },
      /* E */ { 0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61 },
      /* F */ { 0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D } 
    };

#endif
    
    const unsigned int AES::rowshifts[4] = { 0, 1, 2, 3 };    
    
    
    // TODO: extend at least up to 32 values
    const unsigned int AESKey::rcon[10] = { 0x01000000, 0x02000000, 0x04000000, 0x08000000,
					    0x10000000, 0x20000000, 0x40000000, 0x80000000,
					    0x1B000000, 0x36000000 };
    
  };
};




