// to give an automatica mat_at() without typename
// FMat_at return float of data !
// 
#ifndef MATAT_HPP_
#define MATAT_HPP_

#include <opencv2/core.hpp>
#include <iostream>
using namespace std;

// Function for accessing pixel values of CV_8U Mat
float FMat_at(cv::Mat& img, int row, int col) {
    if(img.type()==0){ // 8U
        return img.at<uchar>(row, col);
    }else if(img.type()==5){ // 32F
        return img.at<float>(row, col);
    }else{ // many other situation
        cout << " type error" << endl;
        exit(0);
    }
}

#endif
