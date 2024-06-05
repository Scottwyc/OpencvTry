
#include <iostream>
#include <string>
using namespace std;

void testStringChar(){
    int a = stoi("1");
    int b = static_cast<int>('1'); // no, it is 49!! not what I need  
    int c = b - '0'; // use '0' to move!!
    cout << a+1 << endl;
    cout << b+1 << endl;
    cout << c << endl;
}

int main(){

    return 0;
}