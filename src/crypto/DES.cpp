/*
 * Data Encryption Standard (DES)
 */

#include "DES.h"
#include <stdexcept>


namespace whiteice
{
  namespace crypto
  {
    
    // data must be 64 bit and Keyschedule should be DESKey
    bool DES::encrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) 
    {
      try{

	// input data must be 64 bits
	if(data.size() != 64) return false;      
	
	// keys must be 48 bits long
	if(k.keybits() != 48) return false;      
	
	// number of rounds keys (-> number of rounds)
	// must be at least 16
	if(k.size() < 16) return false;
	
	
	// performs initial permutation
	{
	  dynamic_bitset temp(data);
	  
	  for(unsigned int i=0;i<64;i++)
	    data.set(i, temp[IP0[i] - 1]);
	  
	  
	  temp.reset();
	}
	
	
	// DES encryption
	{
	  dynamic_bitset L, R, pR, pL;
	  
	  L.resize(32);
	  R.resize(32);
	  
	  pL.resize(32); // previous round
	  pR.resize(32);
	  
	  for(unsigned int i=0;i<32;i++){
	    pL.set(i, data[i]);
	    pR.set(i, data[i+32]);
	  }
	  
	  
	  for(unsigned int i=0;i<k.size();i++){
	    
	    // L_i = R_(i-1)	  
	    L = pR;
	    
	    // R_i = L_(i-1) xor f(R_(i-1), Ki)
	    if( f(pR, k[i]) == false ) return false;	  
	    R = pL ^ pR;
	    
	    pL = L;
	    pR = R;
	  }
	  
	  // puts crypted data in [pR pL] order
	  // into data (reversed order)
	  for(unsigned int i=0;i<32;i++){
	    data.set(i, pR[i]);
	    data.set(i+32, pL[i]);
	  }
	  
	  L.reset();
	  R.reset();
	  pL.reset();
	  pR.reset();
	}
	
	
	// performs final inverse permutation
	{
	  dynamic_bitset temp(data);
	  
	  for(unsigned int i=0;i<64;i++)
	    data.set(i, temp[IP1[i] - 1]);
	  
	  
	  temp.reset();
	}
	
	return true;
	
      }
      catch(std::exception& e){
	std::cout << "DES exception: " << e.what() << std::endl;
	return false;
      }
    }
    
    
    bool DES::decrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) 
    {
      try{
	
	// input data must be 64 bits
	if(data.size() != 64) return false;      
	
	// keys must be 48 bits long
	if(k.keybits() != 48) return false;      
	
	// number of rounds keys (-> number of rounds)
	// must be at least 16
	if(k.size() < 16) return false;
	
	
	// performs initial permutation
	{
	  dynamic_bitset temp(data);
	  
	  for(unsigned int i=0;i<64;i++){
	    data.set(i, temp[IP0[i] - 1]);
	  }
	  
	  temp.reset();
	}
	
	
	// DES decryption
	{
	  dynamic_bitset L, R, pR, pL;
	  
	  L.resize(32);
	  R.resize(32);
	  
	  pL.resize(32); // previous round
	  pR.resize(32);
	  
	  // load pR and pL values from the data
	  // where data is formated as [pR pL]
	  // (reversed order)
	  for(unsigned int i=0;i<32;i++){
	    pR.set(i, data[i]);
	    pL.set(i, data[i+32]);
	  }
	  
	  
	  
	  for(int i=(k.size() - 1);i>=0;i--){
	    
	    // R_(i-1) = L_i
	    R = pL;
	    
	    // R_i = L_(i-1) xor f(R_(i-1), Ki)
	    if( f(pL, k[i]) == false ) return false;	  
	    L = pR ^ pL;
	    
	    pL = L;
	    pR = R;
	  }
	  
	  for(unsigned int i=0;i<32;i++){
	    data.set(i, pL[i]);
	    data.set(i+32, pR[i]);
	  }
	  
	  L.reset();
	  R.reset();
	  pL.reset();
	  pR.reset();
	}
	
	
	// performs final inverse permutation
	{
	  dynamic_bitset temp(data);
	  
	  for(unsigned int i=0;i<64;i++){
	    data.set(i, temp[IP1[i] - 1]);
	  }
	  
	  temp.reset();
	}
	
	return true;
	
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    bool DES::encrypt(data_source<dynamic_bitset>& data,
		      const Keyschedule<dynamic_bitset>& k,
		      const dynamic_bitset& IV,
		      ModeOfOperation mode) 
    {
      if(IV.size() != 64 && mode != ECBmode)
	return false; // IV size is not same as block size and non ECB mode
      
      if(mode == CTRmode){
	dynamic_bitset temp, CTR = IV;
	
	for(unsigned int i=0;i<data.size();i++){
	  temp = CTR;
	  if(!encrypt(temp, k)) return false;
	  CTR.inc();
	  data[i] ^= temp;
	}
      }
      else if(mode == CBCmode){
	dynamic_bitset prev = IV;
	
	for(unsigned int i=0;i<data.size();i++){
	  data[i] ^= prev;
	  if(!encrypt(data[i], k)) return false;
	  prev = data[i];
	}
      }
      else if(mode == OFBmode){
	dynamic_bitset z = IV;
	
	for(unsigned int i=0;i<data.size();i++){
	  if(!encrypt(z, k)) return false;
	  data[i] ^= z;
	}
      }
      else if(mode == CFBmode){
	dynamic_bitset y = IV;
	
	for(unsigned int i=0;i<data.size();i++){
	  if(!encrypt(y, k)) return false;
	  data[i] ^= y;
	  y = data[i];
	}
	
      }
      else if(mode == ECBmode){
	      
	for(unsigned int i=0;i<data.size();i++){
	  if(!encrypt(data[i],k)) return false;
	}
	
      }
      else return false;
      
      return true;
    }
    

        
    bool DES::decrypt(data_source<dynamic_bitset>& data,
		      const Keyschedule<dynamic_bitset>& k,
		      const dynamic_bitset& IV,
		      ModeOfOperation mode) 
    {
      if(IV.size() != 64 && mode != ECBmode)
	return false; // IV size is not same as block size and non ECB mode
      
      if(mode == CTRmode){
	dynamic_bitset temp, CTR = IV;
	
	for(unsigned int i=0;i<data.size();i++){
	  temp = CTR;
	  if(!encrypt(temp, k)) return false;
	  CTR.inc();
	  data[i] ^= temp;
	}
      }
      else if(mode == CBCmode){
	dynamic_bitset y, prev = IV;
	
	for(unsigned int i=0;i<data.size();i++){
	  y = data[i];
	  if(!decrypt(data[i], k)) return false;
	  data[i] ^= prev;
	  prev = y;
	}
      }
      else if(mode == OFBmode){
	dynamic_bitset z = IV;
	
	for(unsigned int i=0;i<data.size();i++){
	  if(!encrypt(z, k)) return false;
	  data[i] ^= z;
	}
      }
      else if(mode == CFBmode){
	dynamic_bitset temp, y = IV;
	
	for(unsigned int i=0;i<data.size();i++){
	  if(!encrypt(y, k)) return false;
	  temp = data[i];
	  data[i] ^= y;
	  y = temp;
	}
	
      }
      else if(mode == ECBmode){
	
	for(unsigned int i=0;i<data.size();i++){
	  if(!decrypt(data[i],k)) return false;
	}
	
      }
      else return false;
      
      return true;
    }
    
    
    // A is data (32bits), J is key (48bits)
    bool DES::f(dynamic_bitset& A, const dynamic_bitset& J) 
    {
      dynamic_bitset EA;
      
      try{	
	EA.resize(48);
	
	// expands A
	
	for(unsigned int i=0;i<48;i++)
	  EA.set(i, A[E[i] - 1]);
	
	// EA = EA xor J
	EA ^= J;
	
	// S boxes (substitution)
	{
	  unsigned int value;
	  unsigned int i1, i2;
	  unsigned int ji, ai;
	  
	  ji = ai = 0;
	  
	  
	  //////////////////////////////////////////////
	  // S1  
	  
	  i1 = 
	    (((unsigned int)EA[0 + ji]) << 1) + EA[5 + ji];
	  
	  i2 = 
	    (((unsigned int)EA[1 + ji]) << 3) +
	    (((unsigned int)EA[2 + ji]) << 2) +
	    (((unsigned int)EA[3 + ji]) << 1) +
	    (((unsigned int)EA[4 + ji]) << 0);
	  
	  value = S1[i1][i2];
	  
	  A.set(ai + 0, value & 1); value >>= 1;
	  A.set(ai + 1, value & 1); value >>= 1;
	  A.set(ai + 2, value & 1); value >>= 1;
	  A.set(ai + 3, value & 1); value >>= 1;
	  
	  ji += 6;
	  ai += 4;
	  
	  
	  //////////////////////////////////////////////
	  // S2
	  
	  i1 = 
	    (((unsigned int)EA[0 + ji]) << 1) + EA[5 + ji];
	  
	  i2 = 
	    (((unsigned int)EA[1 + ji]) << 3) +
	    (((unsigned int)EA[2 + ji]) << 2) +
	    (((unsigned int)EA[3 + ji]) << 1) +
	    (((unsigned int)EA[4 + ji]) << 0);
	  
	  value = S2[i1][i2];
	  
	  A.set(ai + 0, value & 1); value >>= 1;
	  A.set(ai + 1, value & 1); value >>= 1;
	  A.set(ai + 2, value & 1); value >>= 1;
	  A.set(ai + 3, value & 1); value >>= 1;
	  
	  ji += 6;
	  ai += 4;
	  
	  
	  //////////////////////////////////////////////
	  // S3
	  
	  i1 = 
	    (((unsigned int)EA[0 + ji]) << 1) + EA[5 + ji];
	  
	  i2 = 
	    (((unsigned int)EA[1 + ji]) << 3) +
	    (((unsigned int)EA[2 + ji]) << 2) +
	    (((unsigned int)EA[3 + ji]) << 1) +
	    (((unsigned int)EA[4 + ji]) << 0);
	  
	  value = S3[i1][i2];
	  
	  A.set(ai + 0, value & 1); value >>= 1;
	  A.set(ai + 1, value & 1); value >>= 1;
	  A.set(ai + 2, value & 1); value >>= 1;
	  A.set(ai + 3, value & 1); value >>= 1;
	  
	  ji += 6;
	  ai += 4;
	  
	  
	  //////////////////////////////////////////////
	  // S4
	  
	  i1 = 
	    (((unsigned int)EA[0 + ji]) << 1) + EA[5 + ji];
	  
	  i2 = 
	    (((unsigned int)EA[1 + ji]) << 3) +
	    (((unsigned int)EA[2 + ji]) << 2) +
	    (((unsigned int)EA[3 + ji]) << 1) +
	    (((unsigned int)EA[4 + ji]) << 0);
	  
	  value = S4[i1][i2];
	  
	  A.set(ai + 0, value & 1); value >>= 1;
	  A.set(ai + 1, value & 1); value >>= 1;
	  A.set(ai + 2, value & 1); value >>= 1;
	  A.set(ai + 3, value & 1); value >>= 1;
	  
	  ji += 6;
	  ai += 4;
	
	  
	  //////////////////////////////////////////////
	  // S5
	  
	  i1 = 
	    (((unsigned int)EA[0 + ji]) << 1) + EA[5 + ji];
	  
	  i2 = 
	    (((unsigned int)EA[1 + ji]) << 3) +
	    (((unsigned int)EA[2 + ji]) << 2) +
	    (((unsigned int)EA[3 + ji]) << 1) +
	    (((unsigned int)EA[4 + ji]) << 0);
	  
	  value = S5[i1][i2];
	  
	  A.set(ai + 0, value & 1); value >>= 1;
	  A.set(ai + 1, value & 1); value >>= 1;
	  A.set(ai + 2, value & 1); value >>= 1;
	  A.set(ai + 3, value & 1); value >>= 1;
	  
	  ji += 6;
	  ai += 4;
	  
	  //////////////////////////////////////////////
	  // S6
	  
	  i1 = 
	    (((unsigned int)EA[0 + ji]) << 1) + EA[5 + ji];
	  
	  i2 = 
	    (((unsigned int)EA[1 + ji]) << 3) +
	    (((unsigned int)EA[2 + ji]) << 2) +
	    (((unsigned int)EA[3 + ji]) << 1) +
	    (((unsigned int)EA[4 + ji]) << 0);
	  
	  value = S6[i1][i2];
	  
	  A.set(ai + 0, value & 1); value >>= 1;
	  A.set(ai + 1, value & 1); value >>= 1;
	  A.set(ai + 2, value & 1); value >>= 1;
	  A.set(ai + 3, value & 1); value >>= 1;
	  
	  ji += 6;
	  ai += 4;
	  
	  
	  //////////////////////////////////////////////
	  // S7
	  
	  i1 = 
	    (((unsigned int)EA[0 + ji]) << 1) + EA[5 + ji];
	  
	  i2 = 
	    (((unsigned int)EA[1 + ji]) << 3) +
	    (((unsigned int)EA[2 + ji]) << 2) +
	    (((unsigned int)EA[3 + ji]) << 1) +
	    (((unsigned int)EA[4 + ji]) << 0);
	  
	  value = S7[i1][i2];
	  
	  A.set(ai + 0, value & 1); value >>= 1;
	  A.set(ai + 1, value & 1); value >>= 1;
	  A.set(ai + 2, value & 1); value >>= 1;
	  A.set(ai + 3, value & 1); value >>= 1;
	   
	  ji += 6;
	  ai += 4;
	  
	  //////////////////////////////////////////////
	  // S8
	  
	  i1 = 
	    (((unsigned int)EA[0 + ji]) << 1) + EA[5 + ji];
	  
	  i2 = 
	    (((unsigned int)EA[1 + ji]) << 3) +
	    (((unsigned int)EA[2 + ji]) << 2) +
	    (((unsigned int)EA[3 + ji]) << 1) +
	    (((unsigned int)EA[4 + ji]) << 0);
	  
	  value = S8[i1][i2];
	  
	  A.set(ai + 0, value & 1); value >>= 1;
	  A.set(ai + 1, value & 1); value >>= 1;
	  A.set(ai + 2, value & 1); value >>= 1;
	  A.set(ai + 3, value & 1); value >>= 1;
	  
	  ji += 6;
	  ai += 4;
	  
	  value = 0;
	  i1 = i2 = 0;
	}
	
	
	// P-box (permutation)
	{	  	  
	  EA.resize(32);
	  EA = A;
	  
	  for(unsigned int i=0;i<32;i++)
	    A.set(i, EA[P0[i] - 1]);
	}

	EA.reset();
	
	return true;
	
      }
      catch(std::exception& e){
	EA.reset();
	A.reset();
	
	return false;
      }
    }
    
    
    //////////////////////////////////////////////////
    // Triple DES extension to DES
    
    
    bool TDES::encrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) 
    {
      try{

      	// input data must be 64 bits
	if(data.size() != 64) return false;      
	
	// keys must be 48 bits long
	if(k.keybits() != 48) return false;      
	
	// number of rounds keys must be 3*16 = 48	
	if(k.size() != 48) return false;
	
	DES des;
	
	PartialKeyschedule k1(k, 0, 16);
	PartialKeyschedule k2(k, 16, 16);
	PartialKeyschedule k3(k, 32, 16);

	if(des.encrypt(data, k1) == false) return false;
	if(des.decrypt(data, k2) == false) return false;
	if(des.encrypt(data, k3) == false) return false;
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    bool TDES::decrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) 
    {
      try{
	// input data must be 64 bits
	if(data.size() != 64) return false;      
	
	// keys must be 48 bits long
	if(k.keybits() != 48) return false;      
	
	// number of rounds keys must be 3*16 = 48	
	if(k.size() != 48) return false;
	
	DES des;
	
	PartialKeyschedule k1(k, 0, 16);
	PartialKeyschedule k2(k, 16, 16);
	PartialKeyschedule k3(k, 32, 16);
	
	if(des.decrypt(data, k3) == false) return false;
	if(des.encrypt(data, k2) == false) return false;
	if(des.decrypt(data, k1) == false) return false;
	
	return true;	
      }
      catch(std::exception& e){
	return false;
      }
    }
    

    
    
    bool TDES::encrypt(data_source<dynamic_bitset>& data,
		      const Keyschedule<dynamic_bitset>& k,
		      const dynamic_bitset& IV,
		       ModeOfOperation mode) 
    {
      try{
      	// input data must be 64 bits
	if(data.size() != 64) return false;      
	
	// keys must be 48 bits long
	if(k.keybits() != 48) return false;      
	
	// number of rounds keys must be 3*16 = 48	
	if(k.size() != 48) return false;
	
	DES des;
	
	PartialKeyschedule k1(k, 0, 16);
	PartialKeyschedule k2(k, 16, 16);
	PartialKeyschedule k3(k, 32, 16);
	
	if(des.encrypt(data, k1, IV, mode) == false) return false;
	if(des.decrypt(data, k2, IV, mode) == false) return false;
	if(des.encrypt(data, k3, IV, mode) == false) return false;
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    bool TDES::decrypt(data_source<dynamic_bitset>& data,
		       const Keyschedule<dynamic_bitset>& k,
		       const dynamic_bitset& IV,
		       ModeOfOperation mode) 
    {
      try{
	// input data must be 64 bits
	if(data.size() != 64) return false;      
	

	// keys must be 48 bits long
	if(k.keybits() != 48) return false;      
	
	// number of rounds keys must be 3*16 = 48	
	if(k.size() != 48) return false;
	
	DES des;
	
	PartialKeyschedule k1(k, 0, 16);
	PartialKeyschedule k2(k, 16, 16);
	PartialKeyschedule k3(k, 32, 16);
	
	if(des.decrypt(data, k3, IV, mode) == false) return false;
	if(des.encrypt(data, k2, IV, mode) == false) return false;
	if(des.decrypt(data, k1, IV, mode) == false) return false;
	
	return true;	
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    //////////////////////////////////////////////////    
    // keyschedule for 56 bit DES
    
    
    DESKey::DESKey(const dynamic_bitset& key)
    {
      if(key.size() != 56)
	throw std::invalid_argument("DES key must be 56 bits long");
      
      this->key.resize(key.size());      
      this->key = key;
      
      calculate_roundkeys();
    }

    
    DESKey::DESKey(const DESKey& k)
    {
      this->key = k.key;
      
      for(unsigned int i=0;i<16;i++)
	this->derived_keys[i] = k.derived_keys[i];
    }
    
    
    DESKey::~DESKey()
    {
      this->key.reset();
    }
    
    
    unsigned int DESKey::size() const 
    {
      return 16;
    }
    
    
    unsigned int DESKey::keybits() const 
    {      
      return 48;
    }
    
    
    const dynamic_bitset& DESKey::operator[](unsigned int n)
      const 
    {
      // TODO: calculate and return (n+1):th round key
      
      if(n >= 16)
	throw std::out_of_range("no so many DES round keys");
      
      return derived_keys[n];
    }
    
    
    
    void DESKey::calculate_roundkeys()
    {
      for(unsigned int i=0;i<16;i++)
	derived_keys[i].resize(48);
      
      dynamic_bitset D, C;
      dynamic_bitset sum;
      
      D.resize(28);
      C.resize(28);
      sum.resize(56);
      
      // calculates D0 and C0 with 'permuted choice 1'
      
      for(unsigned int i=0;i<28;i++){
	C.set(i, key[PC1_C0[i] - 1]); 
	D.set(i, key[PC1_D0[i] - 1]);
      }
      
      // creates 16 round keys
      for(unsigned int i=0;i<16;i++){
	
	C.cyclicshift(left_shifts[i]);
	D.cyclicshift(left_shifts[i]);
	
	sum = C + D;
	
	// calculates round key from Ci and Di with 'permuted choice 2'
	for(unsigned int j=0;j<48;j++){
	  derived_keys[i].set(j, sum[PC2[j] - 1]);
	}
      }
      
    }

    
    Keyschedule<dynamic_bitset>* DESKey::copy() const { return new DESKey(*this); }
    
    
    //////////////////////////////////////////////////
    // keyschedule code for N-DES
    
    
    NDESKey::NDESKey(const dynamic_bitset& key)
    {      
      if(key.size() % 56)
	throw std::invalid_argument("DES key must be multiple of 56 bits");
      
      std::vector<dynamic_bitset> rawkeys;
      
      {
	rawkeys.resize(key.size() / 56);
	
	for(unsigned int i=0;i<rawkeys.size();i++){
	  rawkeys[i].resize(56);
	  
	  for(unsigned int j=0;j<56;j++){
	    rawkeys[i].set(j, key[j + 56*i]);
	  }
	}
      }
      
      
      // tries to allocate memory / create keys
      
      unsigned int i = 0;
      
      try{	
	this->keys.resize(rawkeys.size());
      
	for(i=0;i<keys.size();i++)
	  this->keys[i] = new DESKey(rawkeys[i]);
	
      }
      catch(std::exception& e){
	
	// frees already allocated keys
	for(unsigned int j=0;j<i;j++)
	  delete (this->keys[i]);

	this->keys.resize(0);
	
	throw e; // rethrows exception
      }
      
    }
    
    
    NDESKey::NDESKey(const std::vector<dynamic_bitset>& keys)
    {
      if(keys.size() <= 0)
	throw std::invalid_argument("Must have at least one DES key");
      
      this->keys.resize(keys.size());
      
      
      // tries to allocate memory / create keys
      
      unsigned int i = 0;
      
      try{
      
	for(i=0;i<keys.size();i++)
	  this->keys[i] = new DESKey(keys[i]);
	
      }
      catch(std::exception& e){
	
	// frees already allocated keys
	for(unsigned int j=0;j<i;j++)
	  delete (this->keys[i]);

	this->keys.resize(0);
	
	throw e; // rethrows exception
      }
      
    }

    
    NDESKey::NDESKey(const NDESKey& k)
    {
      keys.resize(k.keys.size());
      
      // tries to allocate memory / create keys
      
      unsigned int i = 0;
      
      try{
      
	for(i=0;i<keys.size();i++)
	  this->keys[i] = new DESKey(*(k.keys[i]));
	
      }
      catch(std::exception& e){
	
	// frees already allocated keys
	for(unsigned int j=0;j<i;j++)
	  delete (this->keys[i]);
	
	keys.resize(0);
	
	throw e; // rethrows exception
      }      
    }
      
    
    NDESKey::~NDESKey()
    {
      for(unsigned int i=0;i<keys.size();i++){
	delete keys[i];
      }
    }
    
    
    unsigned int NDESKey::size() const 
    {
      if(keys.size() > 0)
	return ( ( keys[0]->size() ) * keys.size() );
      else
	return 0;
    }
    
    
    unsigned int NDESKey::keybits() const 
    {
      if(keys.size() > 0)
	return keys[0]->keybits();
      else
	return 0;
    }
    
    
    const dynamic_bitset& NDESKey::operator[](unsigned int n)
      const 
    {
      // each key has 16 rounds

      unsigned int k = n % keys[0]->size();
      n /= keys[0]->size();
      
      if(n >= keys.size())
	throw std::out_of_range("DES round key out of range - not so many keys");
      
      return (*(keys[n]))[k];
    }    
    
    
    Keyschedule<dynamic_bitset>* NDESKey::copy() const { return new NDESKey(*this); }
    
    
    //////////////////////////////////////////////////
    
    
    PartialKeyschedule::PartialKeyschedule(const Keyschedule<dynamic_bitset>& k,
					   unsigned int begin, unsigned int len)
    {
      if(k.size() < begin + len || len == 0 || k.size() == 0)
	throw std::invalid_argument("Bad keyschedule partition (zero length or doesn't fit)");
      
      this->begin = begin;
      this->len = len;
      
      ks = k.copy();
    }
    
    
    PartialKeyschedule::PartialKeyschedule(const PartialKeyschedule& k)
    {
      this->begin = k.begin;
      this->len = k.len;
      
      ks = k.ks->copy();
    }
    
    
    PartialKeyschedule::~PartialKeyschedule()
    {
      if(ks) delete ks;
    }
    
    
    unsigned int PartialKeyschedule::size() const 
    {
      return len;
    }
    
    unsigned int PartialKeyschedule::keybits() const 
    {
      return ks->keybits();
    }
    
    
    const dynamic_bitset& PartialKeyschedule::operator[](unsigned int n)
      const 
    {
      if(n >= len)
	throw std::out_of_range("PartialKeyschedule: tried to access round key out of range");
      
      return (*ks)[begin + n];
    }
    
    
    Keyschedule<dynamic_bitset>* PartialKeyschedule::copy() const { return new PartialKeyschedule(*this); }
    
    //////////////////////////////////////////////////
    // DES constants
    
    // initial permutation
    const unsigned int DES::IP0[64] =
    {
      58, 50, 42, 34, 26, 18, 10,  2,
      60, 52, 44, 36, 28, 20, 12,  4,
      62, 54, 46, 38, 30, 22, 14,  6,
      64, 56, 48, 40, 32, 24, 16,  8,
      57, 49, 41, 33, 25, 17,  9,  1,
      59, 51, 43, 35, 27, 19, 11,  3,
      61, 53, 45, 37, 29, 21, 13,  5,
      63, 55, 47, 39, 31, 23, 15,  7 };
    
    
    // temporal (find real initial permutation)
    const unsigned int DES::IP1[64] =
    {
      40,  8, 48, 16, 56, 24, 64, 32,
      39,  7, 47, 15, 55, 23, 63, 31,
      38,  6, 46, 14, 54, 22, 62, 30,
      37,  5, 45, 13, 53, 21, 61, 29,
      36,  4, 44, 12, 52, 20, 60, 28,
      35,  3, 43, 11, 51, 19, 59, 27,
      34,  2, 42, 10, 50, 18, 58, 26,
      33,  1, 41,  9, 49, 17, 57, 25
    };
    
    
    const unsigned int DES::E[48] =
    { 32,  1,  2,  3,  4,  5,
      4,   5,  6,  7,  8,  9,
      8,   9, 10, 11, 12, 13,
      12, 13, 14, 15, 16, 17,
      16, 17, 18, 19, 20, 21,
      20, 21, 22, 23, 24, 25,
      24, 25, 26, 27, 28, 29,
      28, 29, 30, 31, 32,  1 
    };
    
    
    const unsigned int DES::S1[4][16] =
    { { 14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7 },
      {  0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8 },
      {  4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0 },
      { 15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13 } };
    
    const unsigned int DES::S2[4][16] =
    { { 15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10 },
      {  3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5 },
      {  0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15 },
      { 13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9 } };
    
    const unsigned int DES::S3[4][16] =
    { { 10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8 },
      { 13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1 },
      { 13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7 },
      {  1, 10,  13, 0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12 } };
    
    const unsigned int DES::S4[4][16] = 
    { {  7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15 },
      { 13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9 },
      { 10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4 },
      {  3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14 } };
    
    const unsigned int DES::S5[4][16] = 
    { {  2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9 },
      { 14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6 },
      {  4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14 },
      { 11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3 } };
    
    const unsigned int DES::S6[4][16] = 
    { { 12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11 },
      { 10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8 },
      {  9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6 },
      {  4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13 } };
    
    const unsigned int DES::S7[4][16] = 
    { {  4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1 },
      { 13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6 },
      {  1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2 },
      {  6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12 } };
    
    const unsigned int DES::S8[4][16] = 
    { { 13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7 },
      {  1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2 },
      {  7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8 },
      {  2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11 } };
    
    
    const unsigned int DES::P0[32] = 
    { 16,  7, 20, 21,
      29, 12, 28, 17,
      1,  15, 23, 26,
      5,  18, 31, 10,
      2,  8,  24, 14,
      32, 27, 3,  9,
      19, 13, 30, 6,
      22, 11, 4,  25 };
    
    
    // PERMUTED CHOICE 1 for key schedule 
    // tables assume that parity bits aren't part of 
    // the key (key is 56 bits)
    
    const unsigned int DESKey::PC1_C0[28] = 
    { 50, 43, 36, 29, 22, 15,  8, 
       1, 51, 44, 37, 30, 23, 16,
       9,  2, 52, 45, 38, 31, 24,
      17, 10,  3, 53, 46, 39, 32 
    }; 
    
    const unsigned int DESKey::PC1_D0[28] =
    { 
      56, 49, 42, 35, 28, 21, 14, 
      7,  55, 48, 41, 34, 27, 20,
      13,  6, 54, 47, 40, 33, 26,
      19, 12,  5, 25, 18, 11,  4 
    };
    
    
    // PERMUTED CHOICE 2 for key schedule
    
    const unsigned int DESKey::PC2[56] = 
    {
      14, 17, 11, 24,  1,  5,
       3, 28, 15,  6, 21, 10,
      23, 19, 12,  4, 26,  8,
      16,  7, 27, 20, 13,  2,
      41, 52, 31, 37, 47, 55,
      30, 40, 51, 45, 33, 48,
      44, 49, 39, 56, 34, 53,
      46, 42, 50, 36, 29, 32
    };
    
    
    const unsigned int DESKey::left_shifts[16] =
    {
      1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
    };
    
    
    // PERMUTATED CHOICE 1 for key with parity bits
    //    
    //    static const unsigned int DESKey::PC1_C0[28] = 
    //    { 57, 49, 41, 33, 25, 17, 09,
    //      01, 58, 50, 42, 34, 26, 18,
    //      10, 02, 59, 51, 43, 35, 27,
    //      19, 11, 03, 60, 52, 44, 36 };
    //    
    //    static const unsigned int DESKey::PC1_D0[28] =
    //    { 63, 55, 47, 39, 31, 23, 15,
    //      07, 62, 54, 46, 38, 30, 22,
    //      14, 06, 61, 53, 45, 37, 29,
    //      21, 13, 05, 28, 20, 12, 04 };
    //
    
  };
};



