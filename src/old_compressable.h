/*
 * compressable interface
 */

#ifndef compressable_h
#define compressable_h

namespace whiteice
{
  class compressable
  {
  public:
    compressable(){ }
    virtual ~compressable(){ }
    
    /* tries to compress object.
     * compress must fail (and do nothing)
     * if object's data is already compressed.
     *
     * if compress() fails otherwise objects data
     * must still be intact
     */
    virtual bool compress()  = 0;
    
    /* tries to decompress object data 
     * decompress must fail (and do nothing) 
     * if object's data isn't compressed.
     *
     * if decompress() fails otherwise object's data
     * must still be intact (in a compressed form)
     */
    virtual bool decompress()  = 0;
    
    /* tells if data is currently compressed or not */
    virtual bool iscompressed() const  = 0;
  };
  
}

#endif


