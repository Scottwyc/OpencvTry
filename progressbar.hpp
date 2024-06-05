
#ifndef _PROGRESSBAR_HPP
#define _PROGRESSBAR_HPP

#include <iostream>
using namespace std;

void ProgressShow(int loopNum, int i, int SectionNum=10){ // inside loop, get index i(0~loopNum-1)
            int section = loopNum/SectionNum;
            
            // at start
            if(i == 0){
                cout <<  "progress: 0%" ;  
                return;
            }
            
            if( (i+1)%section == 0){ 
                cout << "\r";  // return to the head
                cout << "progress: " << (i+1)/section*100 /SectionNum<<"%"; // those int!! should *100 first, or get 0!
            }
}

#endif