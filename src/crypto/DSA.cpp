
#include "DSA.h"
#include "SHA.h"
#include "RNG.h"

#include <stdlib.h>

namespace whiteice
{
  namespace crypto
  {
    
    DSA::DSA() // creates new key
    {
      key = new DSAKey();
    }
    
    
    DSA::DSA(const DSA& dsa) // gets key from the DSA
    {
      key = new DSAKey(*(dsa.key));
    }
    
    
    
    DSA::DSA(const DSAKey& dsakey)
    {
      key = new DSAKey(dsakey);
    }
    
    
    DSA::~DSA(){
      delete key;
    }

    
    
    // allocates memory with malloc() and calculates signature for len bytes long
    // message. A pointer *signature is updated to point into allocated memory area of
    // slen bytes long.
    bool DSA::sign(unsigned char** message, unsigned int len,
		   std::vector<integer>& signature) const 
    {
      try{
	
	if(message == 0 || *message == 0 || len == 0)
	  return false;
	
	// calculate SHA-1 of the message
	
	integer sha;       
	
	if(!calculate_sha(sha, message, len))
	  return false;
	
	
	// generates random number k
	integer k;
	
	{
	  integer qminus;
	  unsigned int kbits = qminus.bits();
	  
	  qminus = key->publickey()[1]; // q
	  qminus -= 1;
	  
	  do{
	    k.setbit(0); // bigger than zero
	    
	    for(unsigned int i=1;i<kbits;i++)
	      k.setbit(i, random_bit());
	  }
	  while(k >= qminus);
	}
	
	// calculates signature (gamma, epsilon)
	
	integer gamma, epsilon;
	integer k_inverse;
	
	gamma = key->publickey()[2]; // alpha
	modular_exponentation(gamma, k, key->publickey()[0]);
	gamma %= key->publickey()[1]; // q
	
	epsilon = sha + (key->privatekey()[0]) * gamma;
	epsilon %= key->publickey()[1];
	
	modular_inverse(k_inverse, k, key->publickey()[1]);
	epsilon = (epsilon * k_inverse) % key->publickey()[1];
	
	signature.resize(2);
	signature[0] = gamma;
	signature[1] = epsilon;
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    // returns true if the signature of a message matches with the given message
    bool DSA::verify(unsigned char** message, unsigned int len,
		     const std::vector<integer>& signature) const 
    {
      try{
	
	if(message == 0 || *message == 0 ||
	   len == 0 || signature.size() != 2)
	{
	  return false;
	}

	
	integer e1, e2;
	
	// calculates SHA-1
	
	if(!calculate_sha(e1, message, len))
	  return false;
	
	// calculates exponents for signature verification
	
	e1 %= key->publickey()[1]; // q
	
	modular_inverse(e2, signature[1], key->publickey()[1]); // q
	
	e1 = (e1 * e2) % key->publickey()[1];
	
	e2 = (signature[0] * e2) % key->publickey()[1]; // q
	
	// checks signature
	
	integer result = key->publickey()[2]; // alpha
	
	modular_exponentation(result, e1, key->publickey()[0]);
	
	e1 = key->publickey()[3]; // beta
	
	modular_exponentation(e1, e2, key->publickey()[0]);
	
	result = (result * e1) % key->publickey()[0];
	result %= key->publickey()[1];
	
	return (result == signature[0]);
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    
    bool DSA::getKey(DSAKey& dsakey) const 
    {
      try{
	dsakey = *key;
	return true;
      }
      catch(std::exception& e){ return false; }
    }
    
    
    bool DSA::calculate_sha(integer& sha,
			    unsigned char **message,
			    unsigned int len) const 
    {
      // calculate SHA-1 of the message
      
      try
      {
	unsigned char* sha_signature;
	
	sha_signature = (unsigned char*)malloc((160/8));
	if(sha_signature == 0) return false;
	
	SHA sha_hash(160); // SHA-1
	
	unsigned char* m2 = *message;
	  
	if(!sha_hash.hash(&m2, len, sha_signature)){
	  free(sha_signature);
	  return false;
	}
	
	if(m2 != *message){
	  *message = m2;
	  
	  m2 = (unsigned char*)realloc(*message, len);
	  
	  if(m2) *message = m2;
	}
	
	// convers *signature into a number
	// (assumes char is 8 bits wide)
	sha = 0;
	
	for(unsigned int i=0;i<(160/8);i++){
	  sha += sha_signature[i];
	  sha <<= 8;
	}
	
	free(sha_signature);
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    bool DSA::random_bit() const 
    {
      // terrible unsecure (& slow) way to generate bits
      // (useful for testing)
      
      return (whiteice::rng.rand() & 1);
    }
    
    
    /**********************************************************************/
    
    
    DSAKey::DSAKey() // generates new public/private key pair
    {
      this->pk.resize(4);
      this->sk.resize(1);
      
      // generates q
      
      generate_random_prime(pk[1],160);
      
      // generates p
      {
	integer x;
	
	do{
	  
	  generate_random_even(x, 864); // random number
	  
	  // calculates number p, (p-1) = 0 (mod pk[1] = q)
	  
	  pk[0] = x * pk[1]; // even number
	  pk[0] += 1; // odd number (can be prime)
	  
	  if(pk[0].bits() != 1024) continue;
	  if(probably_prime(pk[0]) == false) continue;
	  
	  // pk[0] is 1024bit prime
	  
	  break;
	  
	}
	while(1);
	
	// final check	
      }
      
      
      integer pminus = pk[0];
      pminus -= 1;
      
      // generates alpha
      
      random_number_smaller_than(pk[2], pk[0]);
      pminus /= pk[1];
      modular_exponentation(pk[2], pminus, pk[0]);
      
      // generates a (smaller than q)
      
      random_number_smaller_than(sk[0], pk[1]);
      
      // generates beta
      
      pk[3] = pk[2];
      
      modular_exponentation(pk[3], sk[0], pk[0]);
      
      // temporary check
      
      if(!check_key_values())
	throw std::invalid_argument("Bad DSA key generation");
    }
    
    
    // public key only
    DSAKey::DSAKey(const integer& p,     const integer& q,
		   const integer& alpha, const integer& beta)
    {
      this->pk.resize(4);
      this->sk.resize(0);
      
      this->pk[0] = p;
      this->pk[1] = q;
      this->pk[2] = alpha;
      this->pk[3] = beta;      
      
      if(!check_key_values())
	throw std::invalid_argument("Bad DSA key values");
    }
    
    // public & private key
    DSAKey::DSAKey(const integer& p,     const integer& q,
		   const integer& alpha, const integer& beta,
		   const integer& a)
    {
      this->pk.resize(4);
      this->sk.resize(1);
      
      this->pk[0] = p;
      this->pk[1] = q;
      this->pk[2] = alpha;
      this->pk[3] = beta;
      
      this->sk[0] = a;
      
      if(!check_key_values())
	throw std::invalid_argument("Bad DSA key values");      
    }
    
    
    DSAKey::DSAKey(const DSAKey& dsakey)
    {
      this->pk.resize(dsakey.pk.size());
      this->sk.resize(dsakey.sk.size());
      
      for(unsigned int i=0;i<pk.size();i++)
	this->pk[i] = dsakey.pk[i];
      
      for(unsigned int i=0;i<sk.size();i++)
	this->sk[i] = dsakey.sk[i];
    }
    
    DSAKey::~DSAKey()
    {
      // clears memory
      
      for(unsigned int i=0;i<pk.size();i++)
	pk[i] = 0;
      
      for(unsigned int i=0;i<sk.size();i++)
	sk[i] = 0;
      
      pk.resize(0);
      sk.resize(0);
    }
    
    
    // returns number of keys in a keyschedule
    // or negative value for infinity
    unsigned int DSAKey::size() const 
    {
      return (pk.size() + sk.size());
    }
    
    
    // returns number of bits in a single key      
    unsigned int DSAKey::keybits() const 
    {
      // returns number of bits in p
      
      return this->pk[0].bits();
    }
    
    
    // gets n:th key from the key schedule
    const integer& DSAKey::operator[](unsigned int n) const 
    {
      // 0 = p, 1 = q, 2 = alpha, 3 = beta
      // if(hasSecretKey): 4 = a
      
      if(n < 4){
	return pk[n];
      }
      else if(sk.size() > 0 && n == 4){
	return sk[0];
      }
      else{
	throw std::out_of_range("Key index value too big");
      }
    }
    
    
    
    DSAKey& DSAKey::operator=(const DSAKey& dsakey) 
    {
      this->pk.resize(dsakey.pk.size());
      this->sk.resize(dsakey.sk.size());
      
      for(unsigned int i=0;i<pk.size();i++)
	this->pk[i] = dsakey.pk[i];
      
      for(unsigned int i=0;i<sk.size();i++)
	this->sk[i] = dsakey.sk[i];
      
      return (*this);
    }
    
    
    // Keyschedules can copy itself
    Keyschedule<integer>* DSAKey::copy() const
    {
      return new DSAKey(*this);
    }
    
    
    // gets n:th key from the key schedule
    const std::vector<integer>& DSAKey::publickey() const 
    {
      return pk;
    }
      
    
      // gets n:th key from the key schedule
    const std::vector<integer>& DSAKey::privatekey() const 
    {
      return sk;
    }
    

    
    bool DSAKey::check_key_values() const 
    {
      if(pk.size() != 4) return false;
      if(sk.size() > 1) return false;
      
      if(sk.size() == 1){
	// checks beta is calculated correctly
	
	integer beta = pk[2];
	
	modular_exponentation(beta, sk[0], pk[0]);
	
	if(beta != pk[3]) return false;
      }
      
      // checks p and q are primes
      if(probably_prime(pk[0]) == false) return false;
      if(probably_prime(pk[1]) == false) return false;
      
      // checks alpha q:th root of one in mod p
      {
	integer alpha = pk[2];
	
	modular_exponentation(alpha, pk[1], pk[0]);
	
	if(alpha != 1) return false;
      }
      
      // checks (p - 1) % q == 0
      
      {
	integer pminus = pk[0];
	pminus -= 1;
	
	if((pminus % pk[1]) != 0) return false;
      }
      
      return true;
    }
    
    
    
    void DSAKey::generate_random_prime(integer& a,
				       unsigned int abits) const 
    {
      do{
	a = 0;
	
	a.setbit(0); // odd number
	
	for(unsigned int i=1;i<(abits-1);i++)
	  a.setbit(i, random_bit());
	
	a.setbit(abits - 1); // number uses abits bits	
      }
      while(probably_prime(a) == false);
    }
    
    
    void DSAKey::generate_random_even(integer& a, unsigned int abits) const 
    {
      a = 0;
      
      a.setbit(0, 0); // even number
      
      for(unsigned int i=1;i<(abits-1);i++)
	a.setbit(i, random_bit());
      
      a.setbit(abits - 1); // forces abits:th bit to be one
    }
    
    
    void DSAKey::random_number_smaller_than(integer& a, const integer& b) const 
    {
      const unsigned int abits = b.bits();
      
      do{
	a = 0;
	
	for(unsigned int i=0;i<abits;i++)
	  a.setbit(i, random_bit());
      }
      while(a >= b);
    }
      
    
    bool DSAKey::random_bit() const 
    {
      // terrible unsecure (& slow) way to generate bits
      // (useful for testing)
      
      return (whiteice::rng.rand() & 1);
    }
    
    
  };
};
