// to reproduct RSV


#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include "MatAt.hpp"
#include "progressbar.hpp"
using namespace std;
using namespace cv;

const float  PI_F=3.14159265358979f;

class MyRSF{
    private:
        Mat Img;
        int digitT = 2;
        float delta_t = pow(0.1, digitT); // for gradient decline of Phi
        float epsilon = 1e-6;
        float sigma = 25.0; // ?

        float Lambda1; 
        float Lambda2; 
        float Nu;
        float Mu;

        Mat F1;  // fit the value(amplitude)
        Mat F2;
        Mat Phi;
        int Margin;
        int Max_Iter;


    public:
        MyRSF(Mat img, int margin =10, int max_iter=100, float lambda1=1, float lambda2=1, float nu=1, float mu=1){
            Img = img;
            Max_Iter = max_iter;
            Lambda1 = lambda1;
            Lambda2 = lambda2;
            Nu = nu;
            Mu = mu;
            F1 = Mat::ones(Img.size(), CV_32F);  // just all 1
            F2 = Mat::ones(Img.size(), CV_32F); 

            Margin = margin;
            InitPhi(margin);

        }

        void InitPhi(int margin){ // yes!  but actually the 0 in middle is not neccessary
            Phi = cv::Mat::ones(Img.size(), CV_32F);  // set Phi CV_32F for calculation
            Phi = -1*Phi;  // outside edge as -1, just white

            // first Rect, get 0 inside
            cv::Rect PhiInner1(margin,margin,Img.size().width-2*margin, Img.size().height-2*margin); // choose inner ROI
            Phi(PhiInner1) = 0*Phi(PhiInner1);  // edge as 0

            // second Rect, get -1 inside the edge, thickness of the edge is 1
            int thickness = 1;
            int marginInner = margin + thickness; // margin to the outer
            cv::Rect PhiInner2(marginInner, marginInner, Phi.size().width-2*marginInner, Phi.size().height-2*marginInner); // choose inner ROI
            Phi(PhiInner2).convertTo(Phi(PhiInner2), -1, 1.0, 1); // add 1, inner as +1
        }

        void ImgInfo(){  // cout the information
            cout << "type: "<< Img.type() << endl;
            cout << "channels:  " << Img.channels() << endl;
            cout << Img.size() <<  endl;
        }

        Mat getImg(){
            return Img;
        }

        void setEpsilon(float x=0.001){
            epsilon = x;
        }

        void setSigma(float sig=25.0){
            sigma = sig;
        }

        float Heaviside(float x){ // approximately distinguish: x>=0 get 1 and x<0 get 0
            return ( 1+(2/PI_F)*(atan(x/epsilon)) )/2.0;
        }

        float GaussianKernel(cv::Point2i a, cv::Point2i b ){
            float distanceSq = pow((a.x - b.x),2)+pow( a.y - b.y, 2) ;
            return (exp(-distanceSq/(2*sigma*sigma)))/(2*PI_F*sigma*sigma);
        }

        

        float updataF1(){ // return the change
            float numerator=0;
            float denominator=0;

            for (int i = 0; i < Img.rows; i++) {
                for (int j = 0; j < Img.cols; j++) {
                    // get local point, use the short name instead of at<>..
                    float phi_local = Phi.at<float>(i,j);

                    float u_local;
                    // if(Img.type()==0){ //8u
                    //     u_local = Img.at<uchar>(i,j);  // uchar to get pixel of CV_8U !!
                    // }else if(Img.type()==5){ // 32F
                    //     u_local = Img.at<float>(i,j);
                    // }

                    u_local = FMat_at(Img, i,j); 

                    numerator += u_local*Heaviside(phi_local);
                    denominator += Heaviside(phi_local);
                }
            }
            // float delta = abs(C1-numerator/denominator) ;
            // C1 = numerator/denominator ; 
            // return delta;
            return 0;
        }

        



};


int main(){

    return 0;
}

