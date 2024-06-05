// test c++ filesystem (C++17)

#include <filesystem>
#include <fstream>
#include <string>
#include <iostream>
#include <opencv2/core.hpp>
using namespace std;
namespace fs = std::filesystem;
// using namespace fs;

std::size_t number_of_files_in_directory(std::filesystem::path path){
    using std::filesystem::directory_iterator;
    return std::distance(directory_iterator(path), directory_iterator{});
}

void testFS(){
    fs::path currentPath = fs::current_path();
    cout << currentPath << endl;

    // ./ can represent current path!!

    string path1str = "./mikasa";
    fs::path path1(path1str);
    
    // iteration traverse
    fs::directory_iterator list(path1); // it works!!
    for(auto& it:list){  // resize 1,2,3,4
        fs::path file = it.path().filename();
        cout << it.path().filename() << endl;
        fs::path filename = file.stem();
        cout<< filename << endl;

        fs::path newFolder = currentPath/filename;
        string newFolderStr = newFolder.string();
        if( ! fs::exists( newFolder ) ){
            if( fs::create_directory(newFolder) ){
            cout << "create: "<< newFolder << endl;
            }
        }

        // operate the file. use new folder as output folder
        // fs::path outputPath(newFolder/fs::path("new.txt"));  // legal?
        // ofstream outfile( newFolderStr+"/new.txt" , fstream::out); // can create
        ofstream outfile(newFolderStr+"/new.txt" ); // can create
        if(outfile.is_open()){
            cout << "yes" << endl;
        }



    }
    // for iterator, it seems that, it is disposable for only one tiem?

    // int fileNum = number_of_files_in_directory(path1);
    // cout << number_of_files_in_directory(path1)<< endl;

    // for(int i=0; i<fileNum ; i++){

    // }


}



int main(){

    testFS();
    
    // std::cout << __cplusplus << std::endl; // c standard check  201402 only c++14


    return 0;
}
