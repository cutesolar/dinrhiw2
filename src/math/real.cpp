/*
 * abtrary precision real number simulation implementation
 *
 */

#include "real.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <mutex>


namespace whiteice
{
  namespace math
  {
    
    
    realnumber::realnumber(unsigned long int prec){
      if(!prec) mpf_init(data);
      else mpf_init2(data, prec);
    }
    
    
    realnumber::realnumber(const realnumber& r, unsigned long int prec){
      if(!prec) mpf_init2(data, r.getPrecision());
      else mpf_init2(data, prec);
      
      mpf_set(data, r.data);
    }
    
    
    realnumber::realnumber(signed long i, unsigned long int prec){
      if(!prec) mpf_init(data);
      else mpf_init2(data, prec);
      
      mpf_set_si(data, i);
    }
    
    
    realnumber::realnumber(unsigned long i, unsigned long int prec){
      if(!prec) mpf_init(data);
      else mpf_init2(data, prec);
      
      mpf_set_ui(data, i);
    }
    
    
    realnumber::realnumber(double d, unsigned long int prec){
      if(!prec) mpf_init(data);
      else mpf_init2(data, prec);
      
      mpf_set_d(data, d);
    }
    
    
    realnumber::realnumber(const std::string& s, unsigned long int prec){
      if(!prec) mpf_init(data);
      else mpf_init2(data, prec);
      
      mpf_set_str(data, s.c_str(), 10);
    }
    
    realnumber::realnumber(const mpf_t& d){
      mpf_init2(data, mpf_get_prec(d));
      mpf_set(data, d);
      
      // data[0] = d[0];
      
      // memcpy(data, d, sizeof(mpf_t));
    }
    
    
    realnumber::~realnumber(){ mpf_clear(data); }
    
    
    ////////////////////////////////////////////////////////////
    
    
    unsigned long int realnumber::getPrecision() const {
      return mpf_get_prec(data);
    }
    
    
    void realnumber::setPrecision(unsigned long int prec) {
      mpf_set_prec(data, prec);
    }
    
    
    // operators
    realnumber realnumber::operator+(const realnumber& r) const {
      mpf_t rval;
      unsigned long int p1 = mpf_get_prec(data);
      unsigned long int p2 = mpf_get_prec(r.data);
      if(p1 >= p2) mpf_init2(rval, p1);
      else mpf_init2(rval, p2);
      
      mpf_add(rval, data, r.data);
      
      auto ret = realnumber(rval);
      mpf_clear(rval);
      
      return ret;
    }
    
    
    realnumber realnumber::operator-(const realnumber& r) const {
      mpf_t rval;
      unsigned long int p1 = mpf_get_prec(data);
      unsigned long int p2 = mpf_get_prec(r.data);
      if(p1 >= p2) mpf_init2(rval, p1);
      else mpf_init2(rval, p2);
      
      mpf_sub(rval, data, r.data);

      auto ret = realnumber(rval);
      mpf_clear(rval);
      
      return ret;
    }
    
    
    realnumber realnumber::operator*(const realnumber& r) const {
      mpf_t rval;
      unsigned long int p1 = mpf_get_prec(data);
      unsigned long int p2 = mpf_get_prec(r.data);
      if(p1 >= p2) mpf_init2(rval, p1);
      else mpf_init2(rval, p2);
      
      mpf_mul(rval, data, r.data);

      auto ret = realnumber(rval);
      mpf_clear(rval);
      
      return ret;
    }
    
    
    realnumber realnumber::operator/(const realnumber& r) const {
      mpf_t rval;
      unsigned long int p1 = mpf_get_prec(data);
      unsigned long int p2 = mpf_get_prec(r.data);
      if(p1 >= p2) mpf_init2(rval, p1);
      else mpf_init2(rval, p2);

#if 0
      // No exceptions from division by zero!
      if(mpf_cmp_si(r.data, 0) == 0){ // division by zero
	throw illegal_operation("Division by zero");
      }
#endif
      
      mpf_div(rval, data, r.data);
      
      auto ret = realnumber(rval);
      mpf_clear(rval);
      
      return ret;
    }
    
    
    // complex conjugate
    realnumber realnumber::operator!() const {
      return realnumber(*this); // nothing t do
    }
    
    
    realnumber realnumber::operator-() const {
      mpf_t rval;
      mpf_init2(rval, mpf_get_prec(data));
      mpf_neg(rval, data);

      auto ret = realnumber(rval);
      mpf_clear(rval);
      
      return ret;
    }
    
    
    realnumber& realnumber::operator+=(const realnumber& r) {
      mpf_t temp;
      mpf_init2(temp, mpf_get_prec(data));
      mpf_set(temp, data);
      mpf_add(data, temp, r.data);
      mpf_clear(temp);
      
      return (*this);
    }
    
    realnumber& realnumber::operator-=(const realnumber& r) {
      mpf_t temp;
      mpf_init2(temp, mpf_get_prec(data));
      mpf_set(temp, data);
      mpf_sub(data, temp, r.data);
      mpf_clear(temp);
      
      return (*this);
    }
    
    realnumber& realnumber::operator*=(const realnumber& r) {
      mpf_t temp;
      mpf_init2(temp, mpf_get_prec(data));
      mpf_set(temp, data);
      mpf_mul(data, temp, r.data);
      mpf_clear(temp);
      
      return (*this);
    }
    
    realnumber& realnumber::operator/=(const realnumber& r) {
      mpf_t temp;
      mpf_init2(temp, mpf_get_prec(data));
      mpf_set(temp, data);
      mpf_div(data, temp, r.data);
      mpf_clear(temp);
      
      return (*this);
    }
    
    
    realnumber& realnumber::operator=(const realnumber& r) {
      if(this != &r){
	mpf_set_prec(data, mpf_get_prec(r.data));
	mpf_set(data, r.data);
      }
      
      return (*this);
    }
    
    
    /*************************************************************/
    
    
    bool realnumber::operator==(const realnumber& r) const {
      return (mpf_cmp(data, r.data) == 0);
    }
    
    bool realnumber::operator!=(const realnumber& r) const {
      return (mpf_cmp(data, r.data) != 0);
    }
    
    bool realnumber::operator>=(const realnumber& r) const {
      return (mpf_cmp(data, r.data) >= 0);
    }
    
    bool realnumber::operator<=(const realnumber& r) const {
      return (mpf_cmp(data, r.data) <= 0);
    }
    
    bool realnumber::operator< (const realnumber& r) const {
      return (mpf_cmp(data, r.data) < 0);
    }
    
    bool realnumber::operator> (const realnumber& r) const {
      return (mpf_cmp(data, r.data) > 0);
    }
    
    
    // scalar operation
    realnumber& realnumber::operator= (const double& s) {
      mpf_set_d(data, s);
      return (*this);
    }
    
    
    realnumber  realnumber::operator+ (const double& s) const {
      mpf_t temp, ss;
      mpf_init2(temp, mpf_get_prec(data));
      mpf_init2(ss, mpf_get_prec(data));
      mpf_set_d(ss, s);
      mpf_add(temp, data, ss);
      mpf_clear(ss);

      auto ret = realnumber(temp);
      mpf_clear(temp);
      
      return ret;
    }
    
    
    realnumber  realnumber::operator- (const double& s) const {
      mpf_t temp, ss;
      mpf_init2(temp, mpf_get_prec(data));
      mpf_init2(ss, mpf_get_prec(data));
      mpf_set_d(ss, s);
      mpf_sub(temp, data, ss);
      mpf_clear(ss);

      auto ret = realnumber(temp);
      mpf_clear(temp);
      
      return ret;
    }
    
    
    realnumber& realnumber::operator+=(const double& s) {
      mpf_t temp, ss;
      mpf_init2(temp, mpf_get_prec(data));
      mpf_init2(ss, mpf_get_prec(data));
      mpf_set(temp, data);
      mpf_set_d(ss, s);
      mpf_add(data, temp, ss);
      mpf_clear(ss);
      mpf_clear(temp);
      
      return (*this);
    }
    
    
    realnumber& realnumber::operator-=(const double& s) {
      mpf_t temp, ss;
      mpf_init2(temp, mpf_get_prec(data));
      mpf_init2(ss, mpf_get_prec(data));
      mpf_set(temp, data);
      mpf_set_d(ss, s);
      mpf_sub(data, temp, ss);
      mpf_clear(ss);
      mpf_clear(temp);
      
      return (*this);
    }
    
    
    realnumber  realnumber::operator* (const double& s) const {
      mpf_t ss, res;
      mpf_init2(ss, mpf_get_prec(data));
      mpf_init2(res, mpf_get_prec(data));
      mpf_set_d(ss, s);
      mpf_mul(res, ss, data);
      mpf_clear(ss);

      auto ret = realnumber(res);
      mpf_clear(res);
      
      return ret;
    }
    
    realnumber  realnumber::operator/ (const double& s) const {
      mpf_t ss, res;
      mpf_init2(ss, mpf_get_prec(data));
      mpf_init2(res, mpf_get_prec(data));
      mpf_set_d(ss, s);
      mpf_div(res, data, ss); // FIXED: was mpf_div(res,ss,data): s/data
      mpf_clear(ss);

      auto ret = realnumber(res);
      mpf_clear(res);
      
      return ret;
    }
    
    
    realnumber& realnumber::operator*=(const double& s) {
      mpf_t ss, temp;
      mpf_init2(ss, mpf_get_prec(data));
      mpf_init2(temp, mpf_get_prec(data));
      mpf_set(temp, data);
      mpf_set_d(ss, s);
      mpf_mul(data, temp, ss);
      mpf_clear(temp);
      mpf_clear(ss);
      
      return (*this);
    }
    
    
    realnumber& realnumber::operator/=(const double& s) {
      mpf_t ss, temp;
      mpf_init2(ss, mpf_get_prec(data));
      mpf_init2(temp, mpf_get_prec(data));
      mpf_set(temp, data);
      mpf_set_d(ss, s);
      mpf_div(data, temp, ss);
      mpf_clear(temp);
      mpf_clear(ss);
      
      return (*this);
    }
    
    
    // scalar comparisions
    bool realnumber::operator==(const double d) const {
      return (mpf_cmp_d(data, d) == 0);
    }
    
    
    bool realnumber::operator!=(const double d) const {
      return (mpf_cmp_d(data, d) != 0);
    }
    
    
    bool realnumber::operator>=(const double d) const {
      return (mpf_cmp_d(data, d) >= 0);
    }
    
    
    bool realnumber::operator<=(const double d) const {
      return (mpf_cmp_d(data, d) <= 0);
    }
    
    
    bool realnumber::operator< (const double d) const {
      return (mpf_cmp_d(data, d) < 0);
    }
    
    
    bool realnumber::operator> (const double d) const {
      return (mpf_cmp_d(data, d) > 0);
    }
    
    
    bool realnumber::operator==(const signed long int i) const {
      return (mpf_cmp_si(data, i) == 0);
    }
    
    
    bool realnumber::operator!=(const signed long int i) const {
      return (mpf_cmp_si(data, i) != 0);
    }
    
    
    bool realnumber::operator>=(const signed long int i) const {
      return (mpf_cmp_si(data, i) >= 0);
    }
    
    
    bool realnumber::operator<=(const signed long int i) const {
      return (mpf_cmp_si(data, i) <= 0);
    }
    
    
    bool realnumber::operator< (const signed long int i) const {
      return (mpf_cmp_si(data, i) < 0);
    }
    
    
    bool realnumber::operator> (const signed long int i) const {
      return (mpf_cmp_si(data, i) > 0);
    }
    
    
    
    realnumber& realnumber::abs() {
      mpf_t sd;
      mpf_init2(sd, mpf_get_prec(data));
      mpf_set(sd, data);
      mpf_abs(data, sd);
      mpf_clear(sd);
      
      return (*this);
    }
    
    
    realnumber& realnumber::ceil() {
      mpf_t sd;
      mpf_init2(sd, mpf_get_prec(data));
      mpf_set(sd, data);
      mpf_ceil(data, sd);
      mpf_clear(sd);
      
      return (*this);
    }
    
    
    realnumber& realnumber::floor() {
      mpf_t sd;
      mpf_init2(sd, mpf_get_prec(data));
      mpf_set(sd, data);
      mpf_floor(data, sd);
      mpf_clear(sd);
      
      return (*this);
    }
    
    
    realnumber& realnumber::trunc() {
      mpf_t sd;
      mpf_init2(sd, mpf_get_prec(data));
      mpf_set(sd, data);
      mpf_trunc(data, sd);
      mpf_clear(sd);
      
      return (*this);
    }

    // TODO: NOT TESTED, may fail
    realnumber& realnumber::round() {
      mpf_t sd;
      mpf_t sd2;
      mpf_init2(sd, mpf_get_prec(data));
      mpf_init2(sd2, mpf_get_prec(data));
      mpf_floor(sd, data); // data = -D.739203 => sd = -D-1    , -D.21 => sd = -D-1
      mpf_sub(sd2, data, sd); // SD = -D.739203 +D+1 = 0.271    , -D.21 +D+1 = +0.79
      mpf_set(sd, sd2);
      int compare_int = mpf_cmp_d(sd, 0.5); // sd >= 0.5 , fail, success
      mpf_floor(sd, data); // sd = -D-1, sd = -D-1
      if(compare_int > 0){
	mpf_add_ui(data, sd, 1); // not executed                , sd=-D-1+1 = -D
      }
      else{
	mpf_set(data, sd); // data = -D-1,                      not executed
      }

      mpf_clear(sd);
      mpf_clear(sd2);
      
      return (*this);
    }
    
    
    // returns sign of real number
    // returns 1 if r > 0, 0 if r == 0 and -1 if r < 0
    int realnumber::sign() const {
      return mpf_sgn(data);
    }

    static gmp_randstate_t __rndstate;
    class __init {
    public:
      __init(){
	gmp_randinit_default(__rndstate);

	std::chrono::time_point<std::chrono::system_clock> ts = std::chrono::system_clock::now();
	auto msvalue = std::chrono::duration_cast<std::chrono::microseconds>(ts.time_since_epoch()).count();
	gmp_randseed_ui(__rndstate, (unsigned long int)msvalue);
      }
    };
    static __init default_init;
    static std::mutex rnd_mutex;

    // overwrites number using [0,1[ given precision number
    // this is SLOW because of mutex lock around __rndstate
    realnumber& realnumber::random() {
      {
	std::lock_guard<std::mutex> lock(rnd_mutex);
	// auto prec = getPrecision();
	mpf_urandomb(data, __rndstate, 0 /*prec*/);
      }
      
      return (*this);
    }
    
    double& realnumber::operator[](const unsigned long index)
    {
      
      throw illegal_operation("whiteice::math::realnumber: no subelements");
    }
    
    const double& realnumber::operator[](const unsigned long index) const
      {
      throw illegal_operation("whitece::math::realnumber: no subelements");
    }
    
    
    //////////////////////////////////////////////////
    // conversions
    
    // rounds to closest double
    double realnumber::getDouble() const {
      return mpf_get_d(data);
    }
    
    // returns floor(realnumber) conversion to integer
    // integer realnumber::getInteger() const ; ****** TODO *******
    
    // returns human-readable and realnumber(std::string) ctor
    // understandable representation of realnumber
    std::string realnumber::getString(size_t ndigits) const 
    {
      mp_exp_t exp_ = 1;
      char* str = mpf_get_str(NULL, &exp_, 10, ndigits, data);
      std::string s;
      
      if(strlen(str)){
	if(mpf_sgn(data) >= 0){
	  
	  if(exp_ < 6 && exp_ >= 0){
	    
	    for(int k=0;k<exp_;k++){
	      s += " ";
	      s[k] = str[k];
	    }
	    
	    if(exp_ == 0)
	      s += "0";
	    
	    s += ".";
	    
	    if(str[exp_] == '\0')
	      s += "0";
	    else
	      s += &(str[exp_]);
	    
	  }
	  else if(exp_ > -4 && exp_ < 0){
	    exp_ = -exp_;
	    
	    s = "0.";
	    
	    for(int k=0;k<exp_;k++)
	      s += "0";
	    
	    s += str;
	    
	  }
	  else{
	    exp_--;
	    
	    s = " .";
	    s[0] = str[0];
	    s += &(str[1]);
	    
	    char buf[32];
	    sprintf(buf, "e%+.2ld", exp_);
	    s += buf;
	    
	  }
	}
	else{ // has negative - sign
	  
	  if(exp_ < 6 && exp_ >= 0){
	    s = "-";
	    
	    for(int k=0;k<exp_;k++){
	      s += " ";
	      s[k+1] = str[k+1];
	    }
	    
	    if(exp_ == 0)
	      s += "0";
	    
	    s += ".";
	    
	    if(str[exp_+1] == '\0')
	      s += "0";
	    else
	      s += &(str[exp_+1]);
	    
	    s += &(str[exp_+1]);
	    
	  }
	  else if(exp_ > -4 && exp_ < 0){
	    exp_ = -exp_;
	    
	    s = "-0.";
	    
	    for(int k=0;k<exp_;k++)
	      s += "0";
	    
	    s += &(str[1]);
	    
	  }
	  else{
	    exp_--;
	    
	    s = "- .";
	    s[1] = str[1];
	    s += &(str[2]);
	    
	    char buf[32];
	    sprintf(buf, "e%+.2ld", exp_);
	    s += buf;
	  }
	}
	
      }
      else
	s = "0";
      
      
      free(str);
      return s;
    }


    bool realnumber::printFile(FILE* output) const
    {
      if(output == NULL) return false;

      return (mpf_out_str(output, 10, 0, data) > 0);
    }
    
    bool realnumber::readFile(FILE* input)
    {
      if(input == NULL) return false;

      return (mpf_inp_str(data, input, 10) > 0);
    }
    
    
    ////////////////////////////////////////////////////////////
    
    
    std::ostream& operator<<(std::ostream& ios,
			     const whiteice::math::realnumber& r)
    {
      ios << r.getString(0);
      
      return ios;
    }
    
  };
};
