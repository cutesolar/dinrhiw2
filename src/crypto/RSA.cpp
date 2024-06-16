
#include <exception>
#include <stdexcept>
#include <stdlib.h>

#include "RSA.h"
#include "RNG.h"


namespace whiteice
{
  namespace crypto
  {
    
    
    RSA::RSA(){ }
    
    RSA::~RSA(){ }
    
    
    bool RSA::encrypt(integer& data, const Keyschedule<integer>& k) 
    {
      //// must be public key
      //if(typeid(PublicPrivateKeyPair<integer>).before(typeid(k)) == false)
      //return false;
      //
      //const PublicPrivateKeyPair<integer>& pk = 
      // dynamic_cast< const PublicPrivateKeyPair<integer>& >(k);
      
      
      try{
	
	const RSAKey& pk = dynamic_cast< const RSAKey& >(k);
	
	if(pk.publickey().size() != 2)
	return false;
	
	// Key format assumed (RSAKey)
	//  publickey[0] = b exponent, publickey [1] = n
	
	modular_exponentation(data, pk.publickey()[0], pk.publickey()[1]);
	
	return true;
	
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    bool RSA::decrypt(integer& data, const Keyschedule<integer>& k) 
    {
      try{
	
	const RSAKey& pk = dynamic_cast< const RSAKey& >(k);
	
	if(pk.publickey().size() != 2 || pk.privatekey().size() != 3)
	  return false;
	
	// Key format assumed (RSAKey)
	//  publickey[0] = b exponent, publickey [1] = n
	// privatekey[0] = a exponent, privatekey[1,2] = p, q
	
	modular_exponentation(data, pk.privatekey()[0], pk.publickey()[1]);

	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    bool RSA::encrypt(data_source<integer>& data, const Keyschedule<integer>& k) 
    {
      try{
	
	const RSAKey& pk = dynamic_cast< const RSAKey& >(k);
	
	if(pk.publickey().size() != 2)
	  return false;
	
	// Key format assumed (RSAKey)
	//  publickey[0] = b exponent, publickey [1] = n
	
	for(unsigned int i=0;i<data.size();i++)
	  modular_exponentation(data[i], pk.publickey()[0], pk.publickey()[1]);
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    bool RSA::decrypt(data_source<integer>& data, const Keyschedule<integer>& k) 
    {
      try{
	
	const RSAKey& pk = dynamic_cast< const RSAKey& >(k);
	
	if(pk.publickey().size() != 2 || pk.privatekey().size() != 3)
	  return false;
	
	// Key format assumed (RSAKey)
	//  publickey[0] = b exponent, publickey [1] = n
	// privatekey[0] = a exponent, privatekey[1,2] = p, q      
	
	for(unsigned int i=0;i<data.size();i++)
	  modular_exponentation(data[i], pk.privatekey()[0], pk.publickey()[1]);
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    

    //////////////////////////////////////////////////
    
    
    // generates new public / private key pair
    RSAKey::RSAKey(unsigned int bits)
    {
      // RSA key generation
      
      // Key format assumed (RSAKey)
      //  publickey[0] = b exponent, publickey [1] = n
      // privatekey[0] = a exponent, privatekey[1,2] = p, q
      
      integer phi;      
      
      public_keydata.resize(2);
      private_keydata.resize(3);
      this->bits = bits;
      
      generate_prime(private_keydata[1], bits/2);                    // generates prime p
      
      do{
	generate_prime(private_keydata[2], bits/2);
      }
      while(private_keydata[2] == private_keydata[1]);             // generates prime q , q != p
      
      public_keydata[1] = private_keydata[1] * private_keydata[2]; // n = p * q
      
      std::cout << public_keydata[1].bits() << " : " << public_keydata[1] << std::endl;
      
      phi = (private_keydata[1] - 1) * (private_keydata[2] - 1);   // phi(n) = (p-1) * (q-1)
      
      choose_random_mod_invertible_number(public_keydata[0], phi); // b = invertible < phi(n) number
      
      modular_inverse(private_keydata[0], public_keydata[0], phi); // a = inverse of b mod phi(n)
      
    }
    
    
    // only public key available
    RSAKey::RSAKey(const std::vector<integer>& public_key)
    {
      public_keydata.resize(public_key.size());
      
      for(unsigned int i=0;i<public_key.size();i++)
	public_keydata[i] = public_key[i];
      
      bits = ((public_keydata[1].bits() + 1) / 2) * 2;
    }
    
    
    RSAKey::RSAKey(const std::vector<integer>& public_key,
		   const std::vector<integer>& private_key)
    {
      public_keydata.resize(public_key.size());
      private_keydata.resize(private_key.size());
      
      for(unsigned int i=0;i<public_key.size();i++)
	public_keydata[i] = public_key[i];

      for(unsigned int i=0;i<private_key.size();i++)
	private_keydata[i] = private_key[i];
      
      bits = ((public_keydata[1].bits() + 1) / 2) * 2;
    }
    
    
    RSAKey::~RSAKey()
    {
      // safely clears the cryptographic keys to zero
      // before same memory can be given to someone else
      // by operating system
      
      for(unsigned int i=0;i<public_keydata.size();i++)
	reset(public_keydata[i]);
      
      for(unsigned int i=0;i<private_keydata.size();i++)
	reset(private_keydata[i]);
      
      bits = 0;
    }
    
    

    const std::vector<integer>& RSAKey::publickey() const 
    {
      return public_keydata;
    }
    

    const std::vector<integer>& RSAKey::privatekey() const 
    {
      return private_keydata;
    }
    
    

    unsigned int RSAKey::size() const 
    {
      return (private_keydata.size() + public_keydata.size());
    }
    

    unsigned int RSAKey::keybits() const 
    {
      return bits;
    }
    

    const integer& RSAKey::operator[](unsigned int n) const 
    {
      if(n >= private_keydata.size() + public_keydata.size())
	throw std::out_of_range("Error no so many key parameter values in RSA");
      
      if(n < public_keydata.size())
	return public_keydata[n];
      else
	return private_keydata[n];
    }
      
    
    Keyschedule<integer>* RSAKey::copy() const
    {
      RSAKey* key = new RSAKey(public_keydata, private_keydata);
      
      return key;
    }
    

    /**********************************************************************/

    
    void RSAKey::generate_prime(integer& a, unsigned int a_bits) const 
    {
      do{
	a = 0;
	
	a.setbit(0); // odd number
	
	for(unsigned int i=1;i<a_bits;i++)
	  a.setbit(i, random_bit());
	
	a.setbit(a_bits - 1); // number uses a_bits 'bits'
	
      }
      while(probably_prime(a) == false);
    }

    
    bool RSAKey::random_bit() const 
    {
      // terrible unsecure way to generate bits
      // (useful for testing)
      
      return (whiteice::rng.rand() & 1);
    }
    
    
    void RSAKey::choose_random_mod_invertible_number(integer& x,
						     const integer& phi) const 
    {
      // generates equally probable
      // integer x E [1, phi[  gcd(x,phi) = 1
      
      integer g;
      const unsigned int B = phi.bits();
      
      do{
	x = 0;
	
	for(unsigned int i=0;i<B;i++)
	  x.setbit(i, random_bit());
	
	if(x < phi) gcd(g, x, phi);
	else        continue;
	  
      }
      while(g != 1);
    }

    
    void RSAKey::reset(integer& a) const 
    {
      const unsigned int B = a.bits();
      
      for(unsigned int i=0;i<B;i++)
	a.clrbit(i);
    }
    
    

    
  };
};






