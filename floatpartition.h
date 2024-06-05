#ifndef _FLOATPARTITION_H
#define _FLOATPARTITION_H

int floatPart(float x){
    int floatInt = (int)x;
    int decimalPart = (x-floatInt)*1e7;
    return decimalPart;
}



#endif