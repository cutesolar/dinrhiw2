
#include <zlib.h>

int main(int argc, char** argv)
{
  z_stream zs;
  
  deflateInit(&zs, Z_DEFAULT_COMPRESSION);
  
  return 0;
}
