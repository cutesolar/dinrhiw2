
#ifndef __43_diffeqs_h
#define __43_diffeqs_h

#include <dinrhiw/dinrhiw.h>

bool create_random_diffeq_model(whiteice::nnetwork<>& diffeq, const unsigned int DIMENSIONS);

bool simulate_diffeq_model(whiteice::nnetwork<>& diffeq,
			   const whiteice::math::vertex<>& start,
			   const float TIME_LENGTH,
			   std::vector< whiteice::math::vertex<> >& data);


#endif
