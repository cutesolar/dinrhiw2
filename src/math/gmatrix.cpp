
#ifndef gmatrix_cpp
#define gmatrix_cpp

#include "gmatrix.h"
#include "blade_math.h"

#include <list>
#include <exception>
#include <stdexcept>
#include <cassert>

// using namespace std;


namespace whiteice
{
  namespace math
  {

    
    template <typename T, typename S>
    gmatrix<T,S>::gmatrix(const unsigned int ysize,
			  const unsigned int xsize)
    {
      data.resize(ysize);
      
      for(unsigned int i=0;i<ysize;i++){
	data[i].resize(xsize);
	for(unsigned int j=0;j<xsize;j++)
	  data[i][j] = T(0);
      }
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>::gmatrix(const gmatrix<T,S>& M)
    {
      data.resize(M.data.size());
      
      for(unsigned int j=0;j<M.data.size();j++){
	
	data[j].resize(M[j].size());
	
	for(unsigned int i=0;i<M[j].size();i++)
	  data[j][i] = M.data[j][i];
      }    
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>::gmatrix(const gvertex<T,S>& diagonal)
    {
      this->resize(diagonal.size(),
		   diagonal.size());
      
      for(unsigned int i=0;i<diagonal.size();i++){
	(*this)[i][i] = diagonal[i];
      }
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>::~gmatrix(){ }
    
    
    template <typename T, typename S>
    gmatrix<T,S> gmatrix<T,S>::operator+(const gmatrix<T,S>& m) const
      
    {
      if(data.size() != m.data.size() ||
	 data[0].size() != m.data[0].size()){
	
	throw illegal_operation("'+' operator: gmatrix size mismatch");
      }
      
      gmatrix<T,S> r(*this);
      
      for(unsigned int j=0;j<m.data.size();j++)
	for(unsigned i=0;i<m.data[j].size();i++)
	  r.data[j][i] += m.data[j][i];
      
      return r;
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S> gmatrix<T,S>::operator-(const gmatrix<T,S>& m) const
      
    {
      if(data.size() != m.data.size() ||
	 data[0].size() != m.data[0].size()){
	
	throw illegal_operation("'-'-operator");
      }
      
      gmatrix<T,S> r(*this);
      
      for(unsigned j=0;j<data.size();j++)
	for(unsigned i=0;i<data[0].size();i++){
	  
	  r.data[j][i] -= m.data[j][i];
	}
      
      return r;
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S> gmatrix<T,S>::operator*(const gmatrix<T,S>& m) const
      
    {	
      gmatrix<T,S> r(*this);    
      r *= m;    
      return r;
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S> gmatrix<T,S>::operator/(const gmatrix<T,S>& m) const
      
    {
      gmatrix<T,S> r(*this);    
      r /= m;    
      return r;
    }
    

    template <typename T, typename S>
    gmatrix<T,S> gmatrix<T,S>::operator!() const {    
      throw illegal_operation("'!'-operator");
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S> gmatrix<T,S>::operator-() const
      
    {
      gmatrix<T,S> m(xsize(), ysize());
      
      for(unsigned j=0;j<data.size();j++)
	for(unsigned i=0;i<data[j].size();i++)
	  m.data[j][i] = -data[j][i];
      
      return m;
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::operator+=(const gmatrix<T,S>& m)
      
    {
      if(data.size() != m.data.size() ||
	 data[0].size() != m.data[0].size())
	throw illegal_operation("gmatrix '+='-operator - gmatrix size mismatch");
      
      
      for(unsigned j=0;j<data.size();j++)
	for(unsigned i=0;i<data[j].size();i++)
	  data[j][i] += m.data[j][i];
      
      return (*this);
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::operator-=(const gmatrix<T,S>& m)
      
    {
      if(data.size() != m.data.size() || 
	 data[0].size() != m.data[0].size()){
	
	throw illegal_operation("gmatrix: '-='-opearator: gmatrix size mismatch");
      }
      
      for(unsigned j=0;j<data.size();j++)
	for(unsigned i=0;i<data[j].size();i++)
	  data[j][i] -= m.data[j][i];
      
      return (*this);
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::operator*=(const gmatrix<T,S>& m)
      
    {
      if(data[0].size() != m.data.size() ||
	 data.size() != m.data[0].size())
	throw illegal_operation("gmatrix '*='-operator - gmatrix size mismatch");
      
      
      // TODO:
      // strassen algorithm O(n^lg(7)) for gmatrix multiplication when n>50
      // other more complex gmatrix multiplication rutins O(n^2.376)
      //   - large linear gmatrixes
      // sparse gmatrixes have specific rutins check out those
      
      gmatrix<T,S> r(data.size(), m.data[0].size());
      
      for(unsigned int j=0;j<data.size();j++){
	
	for(unsigned int i=0;i<m.data[0].size();i++){
	  
	  r[j][i] = 0;
	  
	  for(unsigned int k=0;k<m.data.size();k++)
	    r[j][i] += data[j][k] * m[k][i];
	  
	}
      }
      
      resize_x(m.data[0].size());
      
      for(unsigned int j=0;j<data.size();j++)
	for(unsigned int i=0;i<data[j].size();i++)
	  data[j][i] = r[j][i];
      
      
      return (*this);
    }
    
    
    
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::operator/=(const gmatrix<T,S>& m)
      
    {
      gmatrix<T,S> n(m);
      
      (*this) *= n.inv();
      
      return (*this);
    }
  

    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::operator=(const gmatrix<T,S>& m) 
      
    {
      if(this != &m){
	if(data.size() != m.data.size())
	  resize_y(m.data.size());
	
	if(data.size()){
	  
	  if(data[0].size() != m.data[0].size())
	    resize_x(m.data[0].size());
	  
	  for(unsigned j=0;j<m.data.size();j++)
	    for(unsigned i=0;i<m.data[j].size();i++)
	      data[j][i] = m.data[j][i];
	}
      }
      
      return (*this);
    }
    
    
    template <typename T, typename S>
    bool gmatrix<T,S>::operator==(const gmatrix<T,S>& m) const
      
    {
      if(data.size() <= 0) return false; // empty matrices are same
      
      if(data.size() != m.data.size() || data[0].size() != m.data[0].size())
	throw uncomparable("gmatrix: '=='-operator: gmatrix size mismatch");
      
      for(unsigned int j=0;j<data.size();j++){
	for(unsigned int i=0;i<data[0].size();i++)
	  if(data[j][i] != m.data[i][j]) return false;
      }
      
      return true;
    }
    
    
    template <typename T, typename S>
    bool gmatrix<T,S>::operator!=(const gmatrix<T,S>& m) const
      
    {
      if(data.size() <= 0) return false; // empty matrices are same
      
      if(data.size() != m.data.size() || data[0].size() != m.data[0].size())
	throw uncomparable("gmatrix: '!='-operator: gmatrix size mismatch");           
      
      for(unsigned int j=0;j<data.size();j++){
	for(unsigned int i=0;i<data[0].size();i++)
	  if(this->data[j][i] == m.data[i][j]) return false;
      }
      
      return true;	
    }
    
    
    template <typename T, typename S>
    bool gmatrix<T,S>::operator>=(const gmatrix<T,S>& m) const {
      throw uncomparable("gmatrix: '>='-operator: matrices cannot be compared");
    }
    
    template <typename T, typename S>
    bool gmatrix<T,S>::operator<=(const gmatrix<T,S>& m) const {
      throw uncomparable("gmatrix: '<='-operator: matrices cannot be compared");
    }
    
    template <typename T, typename S>
    bool gmatrix<T,S>::operator< (const gmatrix<T,S>& m) const {
      throw uncomparable("gmatrix: '<'-operator: matrices cannot be compared");
    }
    
    template <typename T, typename S>
    bool gmatrix<T,S>::operator> (const gmatrix<T,S>& m) const {
      throw uncomparable("gmatrix: '>'-operator: matrices cannot be compared");
    }
    
    
    /***************************************************/

    
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::operator=(const S& s) 
    {
      const unsigned int H = data.size();
      if(H <= 0) return *this;
      
      for(unsigned int j=0;j<H;j++)
	data[j] = s;
      
      return *this;
    }
    
    
    
    template <typename T, typename S>
    gmatrix<T,S>  gmatrix<T,S>::operator* (const S& s) const 
    {
      gmatrix<T,S> m(*this);
      
      typename gvertex<gvertex<T,S>,S>::iterator i;
      
      for(i=m.data.begin();i!=m.data.end();i++)
	(*i) *= s;
      
      return m;
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S> operator*(const S& s, const gmatrix<T,S>& n)
      
    {
      gmatrix<T,S> m(n);
      for(unsigned int i=0;i<m.data.size();i++) m[i] = s*n[i];
      return m;
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>  gmatrix<T,S>::operator/ (const S& s) const 
      
    {
      gmatrix<T,S> m(*this);
      
      typename gvertex< gvertex<T,S>,S >::iterator i;
      
      for(i=m.data.begin();i!=m.data.end();i++)
	(*i) /= s;
      
      return m;
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::operator*=(const S& s) 
    {
      typename gvertex< gvertex<T,S>,S >::iterator i;
      
      for(i=data.begin();i!=data.end();i++)
	(*i) *= s;
      
      return *this;
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::operator/=(const S& s) 
    {
      typename gvertex< gvertex<T,S>,S >::iterator i;
      
      for(i=data.begin();i!=data.end();i++)
	(*i) /= s;
      
      return *this;
    }
  
    
    /***************************************************/
    
    template <typename T, typename S>
    gvertex<T,S> gmatrix<T,S>::operator*(const gvertex<T,S>& v) const
      
    {
      if(!data.size())
	throw std::invalid_argument("multiply: incompatible gvertex/gmatrix sizes");
      
      if(data[0].size() != v.size())
	throw std::invalid_argument("multiply: incompatible gvertex/gmatrix sizes");
      
      gvertex<T,S> r(data.size());
      
      for(unsigned int j=0;j<data.size();j++){
	gvertex<T,S> t;
	t = data[j]*v;
	r[j] = t[0];
      }
      
      return r;
    }

  
    template <typename T, typename S>
    gvertex<T,S>& gmatrix<T,S>::operator[](const unsigned int index)
      
    {
      return data[index];
    }
    
    
    template <typename T, typename S>
    const gvertex<T,S>& gmatrix<T,S>::operator[](const unsigned int index) const
      
    {
      return data[index];
    }
    
    
    template <typename T, typename S>
    T& gmatrix<T,S>::operator()(unsigned int y, unsigned int x)
      
    {
      return data[y][x];
    }
    
    
    template <typename T, typename S>
    const T& gmatrix<T,S>::operator()(unsigned int y, unsigned int x) const
      
    {
      return data[y][x];
    }
    
    
    /***************************************************/
    
    // crossproduct gmatrix M(z): M(z) * y = z x y
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::crossproduct(const gvertex<T,S>& v)
      
    {
      if(v.size() != 3)
	throw std::out_of_range("crossproduct() requires 3 dimensions");
      
      data[0][0] = S(0);
      data[0][1] = S(-v[2]);
      data[0][2] = S( v[1]);
      data[1][0] = S( v[2]);
      data[1][1] = S(0);
      data[1][2] = S(-v[0]);
      data[2][0] = S(-v[1]);
      data[2][1] = S( v[0]);
      data[2][2] = S(0);
      
      return *this;
    }
    
    
    // euclidean rotation XYZ gmatrix
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::rotation(const S& xr,
				       const S& yr,
				       const S& zr) 
    {
      if( (xsize() != 3 && ysize() != 3) ||
	  (xsize() != 4 && ysize() != 4) ){
	resize_y(4);
	resize_x(4);
      }
      
      S a = cos(xr);
      S b = sin(xr);
      S c = cos(yr);
      S d = sin(yr);
      S e = cos(zr);
      S f = sin(zr);
      
      data[0][0] = S( c*e);
      data[0][1] = S(-c*f);
      data[0][2] = S( d);
      data[1][0] = S( b*d*e + a*f);
      data[1][1] = S(-b*d*f + a*e);
      data[1][2] = S(-b*c);
      data[2][0] = S(-a*d*e + b*f);
      data[2][1] = S( a*d*f + b*e);
      data[2][2] = S( a*c);
      
      if(ysize() == 4){
	data[0][3] = S(0);
	data[1][3] = S(0);
	data[2][3] = S(0);
	data[3][0] = S(0);
	data[3][1] = S(0);
	data[3][2] = S(0);
	data[3][3] = S(1);      
      }
      
      return *this;
    }

    
    // 4x4 translation gmatrix
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::translation(const S& dx,
					  const S& dy,
					  const S& dz) 
    {
      if(ysize() != 4){
	resize_y(4);
	resize_x(4);
      }
      
      for(unsigned int j=0;j<3;j++)
	for(unsigned int i=0;i<4;i++){
	  if(j == i) data[j][i] = T(1);
	  else data[j][i] = T(0);
	}
    
      data[3][0] = dx;
      data[3][1] = dy;
      data[3][2] = dz;
      data[3][3] = T(1);
      
      return (*this);
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::abs() 
    {
      for(unsigned int j=0;j<data.size();j++){
	for(unsigned int i=0;i<data[0].size();i++)
	  data[j][i] = ::whiteice::math::abs(data[j][i]);
      }
      
      return (*this);
    }


    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::conj() 
    {
      for(unsigned int j=0;j<data.size();j++){
	for(unsigned int i=0;i<data[0].size();i++)
	  data[j][i] = ::whiteice::math::conj(data[j][i]);
      }
      
      return (*this);
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::transpose() 
    {
      const unsigned int orig_x = xsize();
      const unsigned int orig_y = ysize();
      
      if(orig_y >= orig_x){
	resize_x(orig_y);
	
	for(unsigned int i=0;i<orig_x;i++){
	  for(unsigned j=i;j<orig_y;j++){
	    // swaps elements
	    std::swap<T>(data[j][i],data[i][j]);
	  }
	}
      }
      else{
	resize_y(orig_x);
	
	for(unsigned j=0;j<orig_y;j++){
	  for(unsigned int i=j;i<orig_x;i++){
	    // swaps elements
	    std::swap<T>(data[j][i],data[i][j]);
	  }
	}
      }
      
      resize_x(orig_y);
      resize_y(orig_x);
      
      return *this;
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::hermite() 
    {
      this->transpose();
      this->conj();
      
      return (*this);
    }    
    
    
    template <typename T, typename S>
    T gmatrix<T,S>::det() const 
    {
      if(ysize() != xsize())
	throw std::logic_error("gmatrix::determinate() - non square gmatrix");
      
      const unsigned int N = data.size();
      
      // calculates determinant using half-implemented
      // gauss-jordan elimination
      // (upper triangle gmatrix -> trace is determinate)
      
      gmatrix<T,S> copy(*this); // copy of gmatrix
      T det = T(1);
      
      for(unsigned int i=0;i<N;i++){
	if(copy[i][i] == T(0)){ // resort
	  // (tries to) finds non-zero entry
	  for(unsigned int j=i+1;j<N;j++){
	    
	    if(copy[j][i] != T(0)){ // swaps values
	      S temp;

	      for(unsigned int k=0;k<N;k++){
		temp = copy[j][k];
		copy[j][k] = copy[i][k];
		copy[i][k] = temp;
	      }
	      
	      det *= T(-1);
	    }
	  }
	}
	
	
	// diagonal entry is zero -> det = 0
	if(copy[i][i] == T(0)) return T(0);            
	
	T t = copy[i][i];
	
	// sets a_ii element to 1
	for(unsigned int k=0;k<N;k++)
	  copy[i][k] /= t;
	
	
	det *= t;
	
	// eliminates lower rows
	for(unsigned int j=i+1;j<N;j++){
	  copy[j] -= copy[j][i]*copy[i];
	}
      }
      
      return det;
    }
    
    
    
    template <typename T, typename S>
    gmatrix<T,S>&  gmatrix<T,S>::inv() 
    {
      // simple and slow: gaussian elimination - works for small matrices
      // big ones: start to use atlas (don't bother to reinvent wheel)
      
      if(ysize() != xsize())
	throw std::logic_error("gmatrix::inv() - non square gmatrix");
      
      if(det() == T(0)) // slow
	throw std::logic_error("gmatrix:inv() - singular gmatrix");
      
      const unsigned int N = data.size();
      
      // gauss-jordan elimination
      
      gmatrix<T,S> copy(*this);
      this->identity();
      
      unsigned int table[N];
      for(unsigned int i=0;i<N;i++)
	table[i] = i;
      
      
      
      for(unsigned int i=0;i<N;i++){
	if(copy[i][i] == T(0)){ // reorder rows
	  
	  // (tries to) finds non-zero entry
	  for(unsigned int j=i+1;j<N;j++){
	    
	    if(copy[j][i] != T(0)){ // swaps values
	      // S temp;
	    
	      for(unsigned int k=0;k<N;k++){
		
		std::swap<T>(copy[j][k], copy[i][k]);
		std::swap<T>(data[j][k], data[i][k]);
			     
		// temp = copy[j][k];
		// copy[j][k] = copy[i][k];
		// copy[i][k] = temp;
		// 
		// temp = data[j][k];
		// data[j][k] = data[i][k];
		// data[i][k] = temp;
	      }
	      
	      // saves row change
	      table[j] = i;
	      table[i] = j;
	    }
	  }
	}
	
	// sets a_ii = 1
	{
	  T t = copy[i][i];
	  
	  for(unsigned int j=0;j<N;j++){
	    copy[i][j] /= t;
	    data[i][j] /= t;
	  }
	}
	
	
	if(i >= 1){
	  // eliminates lower row columns to zero
	  for(unsigned int j=0;j<i;j++){
	    T k = copy[j][i];
	    copy[j] -= k * copy[i];
	    data[j] -= k * data[i];
	  }
	}
	
	
	if(i < N-1){
	  // eliminates upper row columns to zero
	  for(unsigned int j=i+1;j<N;j++){
	    T k = copy[j][i];
	    copy[j] -= k * copy[i];
	    data[j] -= k * data[i];
	  }
	}
	
      }
      
      
      // restores old row order (slow)
      {
	unsigned int i = 0;
	unsigned int x;
	
	while(i < N){
	  
	  // checks if rest of the table is in correct order
	  for(;i<N && table[i] == i;i++);
	  
	  if(i < N){ // fix row one
	    
	    // finds pair
	    for(x=i+1;x<N;x++){
	      if(table[x] == i)
	  	break;
	    }
	    
	    std::swap< gvertex<T> >(data[i],data[x]);
	    
	    // don't bother to update table 
	    // - just continue from the next position
	    i = i + 1; 
	  }
	  
	}
	
      }
      
      return (*this);
    }
    
    
    template <typename T, typename S>
    T gmatrix<T,S>::trace() const 
    {
      if(xsize() != ysize())
	throw std::logic_error("gmatrix::trace() non square gmatrix");
      
      T tr = T(0);
      
      for(unsigned int i=0;i<ysize();i++){
	tr += data[i][i];
      }
      
      return tr;
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::zero()
    {
      if(data.size() == 0) return (*this);
      
      const unsigned int N = data.size();
      const unsigned int M = data[0].size();
      
      for(unsigned int j=0;j<N;j++){
	for(unsigned int i=0;i<M;i++){
	  data[j][i] = T(0);
	}
      }
      
      return (*this);
    }
    
    
    template <typename T, typename S>
    gmatrix<T,S>& gmatrix<T,S>::identity()
    {
      
      for(unsigned int j=0;j<data.size();j++){
	for(unsigned int i=0;i<data[0].size();i++){
	  if(i == j) data[j][i] = T(1);	  
	  else data[j][i] = T(0);
	}
      }
      
      return (*this);
    }
    
    
    
    template <typename T, typename S>
    unsigned int gmatrix<T,S>::xsize() const 
    {
      if(data.size() <= 0)
	return 0;
      
      return data[0].size();
    }
    
    
    template <typename T, typename S>
    unsigned int gmatrix<T,S>::size() const 
    {
      return data.size();
    }
    
    template <typename T, typename S>
    unsigned int gmatrix<T,S>::ysize() const 
    {
      return data.size();
    }
    
    
    template <typename T, typename S>
    bool gmatrix<T,S>::resize(unsigned int y, unsigned int x) 
    {
      if(x != 0 && y != 0){
      
	unsigned int old_y = data.size();
	unsigned int old_x = 0;
	if(old_y > 0) old_x = data[0].size();
	
	
	data.resize(y);
	
	if(old_x == x){
	  if(old_y < y){
	    for(unsigned int i=old_y;i<y;i++)
	      data[i].resize(x);
	  }
	}
	else{
	  for(unsigned int i=0;i<y;i++)
	    data[i].resize(x);
	}
	
	return true;
      }
      else{
	data.resize(0);
	
	return true;
      }
    }
    
    
    template <typename T, typename S>
    bool gmatrix<T,S>::resize_x(unsigned int d) 
    {
      if(d != 0)
	for(unsigned int i=0;i<data.size();i++)
	  data[i].resize(d);            
      else
	data.resize(0);
      
      return true;
    }
    
    template <typename T, typename S>
    bool gmatrix<T,S>::resize_y(unsigned int d) 
    {
      unsigned int old_d = data.size();
      unsigned int size_x = 0;
      
      if(old_d > 0) size_x = data[0].size();
      
      data.resize(d);
      
      if(old_d < d){
	for(unsigned int i=old_d;i<d;i++){
	  data[i].resize(size_x);
	}
      }
      
      return true;
    }
    
    
    template <typename T, typename S>
    void gmatrix<T,S>::normalize() 
    {
      typename gvertex< gvertex<T,S>, S>::iterator i = data.begin();
      
      while(i != data.end()){
	i->normalize();
	i++;
      }
    }
    
    
    /***************************************************/
    
    
    template <typename T, typename S>
    std::ostream& operator<<(std::ostream& ios, const gmatrix<T,S>& M)
    {
      ios << "[";
      
      for(unsigned int j=0;j<M.ysize();j++){
	for(unsigned int i=0;i<M.xsize();i++){
	  ios << " " << M[j][i];
	}
      
	ios << "; ";
      }
      
      ios << "]";
      
      return ios;
    }
    
    
    template <typename T, typename S, typename L, typename M>
    bool convert(gmatrix<T,S>& B, const gmatrix<L,M>& A) 
    {
      try{
	if(B.resize(A.ysize(), A.xsize()) == false)
	  return false;
	
	for(unsigned int j=0;j<A.ysize();j++)
	  for(unsigned int i=0;i<A.xsize();i++)
	    B[j][i] = static_cast<L>(A[j][i]);
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
  }
}

  
#endif

