// to reproduct CV algorithm
// the original picture may be CV_8U, what if I do not change it, directly use a CV_32F Phi on it? the calculation of Phi should can "include" the CV_8U type.
// PHi needs negative pixel, but there is no need to change the original picture!
// so we can try just same size between Img and Phi, but Phi with different type CV_32F for calculation, and when it comes to show Phi on Img, then make some convertion. 
// so Img CV_8U, while PHI and NGX, NGY, div :  CV_32F (for calcuation)
// maybe use CV_64F (double) for higher accuracy?

// let us try the experiment picture
// stoppage is not suitable! each time just tiny moves, so only use max_iter!!
// adjust last Phi,to distinguish negative(as 0, outside) and positive(255, inside) 
// max_iter may be 1e4, 1e5; 1e5 too large and slower, 1e4 is preferred
// for 1e4 max_iter, delt_t can not be too tiny like 0.01, 0.1 is suitable

// use vector for grid search of parameters!
// make testMyCVTotal() to systematically handle those pictures and use grid search!

// for different pictures, may use Threads !! to parallelly handle them! to try...

// from the result, it seeems that Phi is easier to check edges inside the original Phi?

#include <filesystem>
#include <string>
#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include "MatAt.hpp" // my hpp to uniformly get Mat_at
#include "progressbar.hpp" // to show progress

using namespace std;
using namespace cv;
namespace fs = std::filesystem; // rename

const float  PI_F=3.14159265358979f;

class myCV
{
    private:
        Mat Img;
        int Rows; // height of Img
        int Cols; // width of Img
        int type; // type of Img, 0:CV_8U, 5:CV_32F,   should make a type dict!
        int digitT = 2;
        float delta_t = pow(0.1, digitT); // for gradient decline of Phi
        float epsilon = 1e-6;
        // float sigma = 25.0;
        float C1;
        float C2;
        float Lambda1; 
        float Lambda2; 
        float Nu;
        Mat Phi;   // 2D 
        int Margin;
        int Max_Iter;

        // stop delta 
        float stopC1Delta=1e-6;
        float stopC2Delta=1e-6;
        float stopPhiDelta = 1e-6;

    public:
        myCV(Mat img = Mat(), int margin =10, int max_iter=100, int thickness = 1, float c1=1, float c2=1, float lambda1=1, float lambda2=1, float nu=1){
            // c1, c2, Phi to update
            Img = img;
            type = Img.type();
            Rows = Img.rows;
            Cols = Img.cols;
            if(Img.channels()>1){
                ExtractChannel();
                cout << "multichannel, have already extracted one channel picture." << endl;
                exit(0);
            }
            // no change on Img, just set Phi!

            // cv::convertScaleAbs(Img, Img, 1.0, -128); // from CV_8U(0-255) to CV_8S(-128-127)
            // Img.convertTo(Img, CV_32F);  // float for later calculation

            // the picture changes!

            C1 = c1; // inside  first 0?
            C2 = c2; // outside first 255?
            Lambda1 = lambda1;
            Lambda2 = lambda2;
            Nu = nu;
            // Phi = cv::Mat::ones(Img.size(), CV_32F); // just white // set Phi CV_32F for calculation
            // cv::Rect PhiInner(margin,margin,Phi.size().width-2*margin, Phi.size().height-2*margin); // choose inner ROI
            // Phi(PhiInner) = -1*Phi(PhiInner);  // get -1*! get black inside?

            Margin = margin;
            InitPhi(Margin, thickness); // Init Phi

            Max_Iter = max_iter;
        }

        void InitPhi(int margin, int thickness){ // yes!  but actually the 0 in middle is not neccessary
            Phi = cv::Mat::ones(Img.size(), CV_32F);  // set Phi CV_32F for calculation
            Phi = -1*Phi;  // outside edge as -1, just white

            // first Rect, get 0 inside
            cv::Rect PhiInner1(margin,margin,Phi.size().width-2*margin, Phi.size().height-2*margin); // choose inner ROI
            Phi(PhiInner1) = 0*Phi(PhiInner1);  // edge as 0

            // second Rect, get -1 inside the edge, thickness of the edge is 1
            int marginInner = margin + thickness; // margin to the outer
            cv::Rect PhiInner2(marginInner, marginInner, Phi.size().width-2*marginInner, Phi.size().height-2*marginInner); // choose inner ROI
            Phi(PhiInner2).convertTo(Phi(PhiInner2), -1, 1.0, 1); // add 1, inner as +1
        }

        void ImgInfo(){  // cout the information
            cout << "type: "<< Img.type() << endl;
            cout << "channels:  " << Img.channels() << endl;
            cout << Img.size() <<  endl;
        }

        void ExtractChannel(){ // get 1 channel 
            std::vector<cv::Mat> channels;  // vector of each channel
            cv::split(Img, channels);
            cv::Mat oneChannel = channels[0];
            oneChannel.convertTo(oneChannel, CV_32F); // convert to 32-bit floating point
            Img = oneChannel;
            // cout << "channels: "  << Img.channels() << endl; 
            cv::imshow("OneChannel",Img);
            cv::waitKey(0);
            cv::imwrite("onechannel.png", Img, {cv::IMWRITE_PNG_COMPRESSION, 0});
        }

        void setEpsilon(float x=0.001){
            epsilon = x;
        }

        float updataC1(){ // return the change
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
            float delta = abs(C1-numerator/denominator) ;
            C1 = numerator/denominator ; 
            return delta;
        }

        float updataC2(){ // return change
            float numerator=0;
            float denominator=0;

            for (int i = 0; i < Img.rows; i++) {
                for (int j = 0; j < Img.cols; j++) {
                    // get local point, use the short name instead of at<>..
                    float phi_local = Phi.at<float>(i,j);
                    // float u_local = Img.at<uchar>(i,j);  // uchar for CV_8U
                    float u_local = FMat_at(Img, i,j); 
                    
                    numerator += u_local*(1-Heaviside(phi_local));
                    denominator += (1-Heaviside(phi_local));
                }
            }
            float delta = abs(C2-numerator/denominator) ;
            C2 = numerator/denominator ; 
            return delta;
        }
        

        float Heaviside(float x){ // approximately distinguish: x>=0 get 1 and x<0 get 0
            return ( 1+(2/PI_F)*(atan(x/epsilon)) )/2.0;
        }

        void resetPhi(){ // to reset the phi into -1 or 1 ??
            for (int i = 0; i < Img.rows; i++) {
                for (int j = 0; j < Img.cols; j++) {
                    float phi_local = Phi.at<float>(i,j);
                    Phi.at<float>(i,j) = (2/PI_F)*(atan(phi_local/epsilon))  ; // into -1 or 1
                }
            }
        }

        float Dif_H(float x){ // no use?
            return (1/PI_F)*( epsilon/( x*x+ epsilon*epsilon) ) ;
        }

        void NormalizedGradient(Mat Phi, Mat& Gx, Mat& Gy){ // Gx, Gy to store gradient in x/y order
            // original Phi, only +1, -1

            // Calculate the gradient in the x and y directions
            cv::Sobel(Phi, Gx, CV_32F, 1, 0); //X order
            cv::Sobel(Phi, Gy, CV_32F, 0, 1); // Y order

            // Access the gradient values
            for (int i = 0; i < Img.rows; i++) {
                for (int j = 0; j < Img.cols; j++) {
                    float gx = Gx.at<float>(i, j);
                    float gy = Gy.at<float>(i, j);
                    float len = sqrt(gx*gx + gy*gy);  // len of the gradient
                    if(len < 1e-6){ // gx=gy=0
                        // len = 1 ; // for 0 in denominator! the gx,gy get 0 naturally
                        Gx.at<float>(i, j) = 0;
                        Gy.at<float>(i, j) = 0; // set 0
                    }else{
                        Gx.at<float>(i, j) = gx/len;
                        Gy.at<float>(i, j) = gy/len; // normalized
                    }  
                }
            }
        }

        void divNormalizedPhi(Mat& div, Mat NGx, Mat NGy){ // normalized Gx,Gy
            // get div 
            div = cv::Mat::zeros(Img.size(),CV_32F); // initialize! // 32F for float TYPE

            cv::Mat gradientX, gradientY; // gx, gy in Mat
            cv::Sobel(NGx, gradientX, CV_32F, 1, 0); //X order for Phi_x
            cv::Sobel(NGy, gradientY, CV_32F, 0, 1); // Y order for Phi_y

            // Access the gradient values
            for (int i = 0; i < Img.rows; i++) {
                for (int j = 0; j < Img.cols; j++) {
                    float gx = gradientX.at<float>(i, j);
                    float gy = gradientY.at<float>(i, j);
                    div.at<float>(i,j) = gx+gy;  // get div
                }
            }
        }

        void setDeltT(int digit_T=2){ // set delta t for gradient decline
            digitT = digit_T;
            delta_t = pow(0.1, digitT);
        }

        float updatePhi(){
            Mat NGX, NGY;
            NormalizedGradient(Phi, NGX, NGY);
            // cout << "NGX" << endl;
            // cout << NGX << endl;
            // cout << "NGY" << endl;
            // cout << NGY << endl;

            Mat div; 
            divNormalizedPhi(div, NGX, NGY);  // get div

            // cout << "div" << endl;
            // cout << div << endl;

            float delta=0;

            for (int i = 0; i < Img.rows; i++) {
                for (int j = 0; j < Img.cols; j++) {
                    // get local point, use the short name instead of at<>..
                    float phi_local = Phi.at<float>(i,j);
                    float div_local =  div.at<float>(i,j);  // div local!
                    float u_local = FMat_at(Img, i,j); 

                    // updat Phi
                    float delta_local =  Dif_H(phi_local)*( Nu*div_local - Lambda1*pow(u_local-C1 ,2) + Lambda2*pow(u_local-C2, 2) );
                    Phi.at<float>(i,j) += delta_t* delta_local;
                     
                    // use Heavide() for distinguishing Phi each time?
                    // Phi.at<float>(i,j) = Heaviside(Phi.at<float>(i,j));

                    delta += delta_local;
                }
            }
            return delta;
        }

        void setStop(float stopc1=1e-6, float stopc2=1e-6, float stopPhi=1e-6){
            stopC1Delta = stopc1;
            stopC2Delta = stopc2;
            stopPhiDelta = stopPhi;
        }

        bool stop(float deltaC1, float deltaC2, float deltaPhi){
            if( deltaC1 < stopC1Delta && deltaC2 < stopC2Delta && deltaPhi < stopPhiDelta ){ // may more strict?
                return true;
            }else{
                return false;
            }
        }

        void fit( int max_iter = 100 ){
            // cout<< "Phi first:" <<endl;
            // cout << fixed << setprecision(2) << Phi << endl;

            bool finishFlag = false;

            Max_Iter = max_iter;

            int maxtimes;
            for(int i=0; i<Max_Iter; i++){
                // cout << i << " " ; // show progress
                ProgressShow(Max_Iter, i);

                float deltaC1 = updataC1(); 
                float deltaC2 = updataC2();
                float deltaPhi = updatePhi();
                
                // ban stoppage  // it seems that, each time just small changes, so stop condition is not suitable!
                // if(stop( deltaC1, deltaC2, deltaPhi) && false) { 
                //     finishFlag = true;
                //     maxtimes = i+1;
                //     cout << endl;
                //     cout << "finish at index "<< i+1 << endl;
                //     break;
                // }
            }

            // transform Phi 32F to 8U   // no need?
            // Phi.convertTo(Phi, CV_8U);  // to 8U  // the value??

            LastPhiAdjust();

            fitInfoReport(finishFlag, maxtimes);

            // // save Phi
            // cv::imwrite(cv::format("Phi_Mar%d_C1_%d_C2_%d_Iter%d_deltT_1e-%d.png", Margin ,static_cast<int>(C1), static_cast<int>(C2),Max_Iter, digitT ), Phi);
        }

        void LastPhiAdjust(){  // make Phi easier to show the edge, change from 32F into 8U?
            // edge between negative and positive
            // directly, negative(outside) into 0, positive(inside) into 255 
            for(int i=0; i<Phi.rows; i++){
                for(int j=0; j<Phi.cols; j++){
                    float phi_local = Phi.at<float>(i,j);
                    if(phi_local >=0){
                        Phi.at<float>(i,j) = 255; // white
                    }else{
                        Phi.at<float>(i,j) = 0;  // black
                    }
                }
            }
            Phi.convertTo(Phi, CV_8U); // no value change, just change type
        }

        void fitInfoReport(bool finishFlag, int maxtimes){
            // CV reports  
            cout << endl;

            // for total test, no need to show, just see the Phi picture  
            // if(Img.size().width < 45 ){ // too much then do not show
            //     cout<< "Phi last:" << endl;
            //     cout <<  Phi << endl;
            // }
            
            cout<< "Img size: "<< Img.size() << endl;
            cout << "Phi original margin: " << Margin << endl;
            cout << "Delta_t: " << delta_t << endl;
            // cout << cv::format("stop condition: stopC1Delta: %f, stopC2Delta: %f, stopPhiDelta: %f ", stopC1Delta, stopC2Delta, stopPhiDelta)  << endl;
            // cout << "Max_Iteration: " << Max_Iter << endl;
            cout << "C1: " << C1 << " C2: " << C2 << endl;

            if(finishFlag){
                cout << "finish up to stop condition, at times: "<< maxtimes << endl;
            }else{
                cout << "finish up to max_iter: " << Max_Iter << endl;
            }
        }

        void LastPhiOutput(string pathbase){ // save Phi in the designated folder
            // save Phi
            cv::imwrite( pathbase + "/" + cv::format("Phi_Mar%d_C1_%d_C2_%d_Iter%d_deltT_1e-%d.png", Margin ,static_cast<int>(C1), static_cast<int>(C2),Max_Iter, digitT ), Phi);
        }
        Mat getPhi(){
            return Phi;
        }

        void showPhi(string windowName = "Phi"){  // can set the windowname
            cv::namedWindow(windowName, cv::WINDOW_NORMAL); // then drag for changing size
            imshow(windowName, Phi);
            cv::waitKey(0);
        }

        void showImg(string windowName = "Img"){
            cv::namedWindow(windowName, cv::WINDOW_NORMAL);
            imshow(windowName, Img);
            cv::waitKey(0);
        }

        void PhiOntoImg(string windowName = "Phi on Img"){ // add? adddWeighted?
            // cout << Phi.size() << endl;
            // cout << Img.size() << endl;
            // cout << "channels1 " << Phi.channels() << endl;  
            // cout << "channels2 "  << Img.channels() << endl; 

            // convert Img
            cv::Mat ImgF;
            Img.convertTo(ImgF, CV_32F);  // float to be compatible with Phi

            cv::Mat dst = Phi;
            // cv::add(Phi, Img, dst);
            cv::addWeighted(ImgF, 1.0, Phi, 1.0, 0, dst);
            cv::namedWindow(windowName, cv::WINDOW_NORMAL); // then drag for changing size
            cv::imshow(windowName, dst); 
            cv::waitKey(0);
        }

        void ColorPhiOnImg(){
            Mat dst = Mat(Img.size(), CV_32FC3); //  3 channels
            cvtColor(Img, dst, COLOR_GRAY2BGR); // Img into rgb
            Mat ColorPhi = Mat::zeros( Img.size(), CV_32FC3); //  3 channels
            
            // transform Phi, edge turn into red: channel[2]
            for (int i = 0; i < Img.rows; i++) {
                for (int j = 0; j < Img.cols; j++) {
                    if( abs( Phi.at<float>(i,j) ) < 1e-6 ){ // where Phi is 0, put red
                        ColorPhi.at<Vec3b>(i, j)[0] = 0;  // blue
                        ColorPhi.at<Vec3b>(i, j)[1] = 0;  // green
                        ColorPhi.at<Vec3b>(i, j)[2] = 255;  // red
                    }else{ // white
                        ColorPhi.at<Vec3b>(i, j)[0] = 255;  // blue
                        ColorPhi.at<Vec3b>(i, j)[1] = 255;  // green
                        ColorPhi.at<Vec3b>(i, j)[2] = 255;  // red
                    }
                }
            }

            // show ColorPhi
            cv::namedWindow("ColorPhi", cv::WINDOW_NORMAL); 
            cv::imshow("ColorPhi", ColorPhi);
            cv::waitKey(0);

            cout << ColorPhi.size() << endl;
            cout << dst.size() << endl;
            cout << "channels1 " << ColorPhi.channels() << endl;  
            cout << "channels2 "  << dst.channels() << endl; 

            cv::addWeighted(dst, 1.0, ColorPhi, 1.0, 0, dst);
            cv::namedWindow("ColorPhi on Img", cv::WINDOW_NORMAL); // then drag for changing size
            cv::imshow("ColorPhi on Img", dst); 
            cv::waitKey(0);
        }

        //should test colorPhi!! it is strange

        Mat GetImgMat(){
            return Img;
        }

        int getHeight(){ // rows of Img
            return Img.rows;
        }
        
        int getWidth(){ // cols of Img
            return Img.cols; 
        }

        void scaleSize(float scale){ // should operate before Phi operation
            int h = Img.cols;
            int w = Img.rows;

            // resize(Img, Img, Size(int(w*scale), int(h*scale)),0,0,INTER_LINEAR);
            resize(Img, Img, cv::Size(Img.size().width*scale, Img.size().height*scale) ,0,0,INTER_LINEAR);
            // cv::namedWindow("Scale", cv::WINDOW_NORMAL); 
            cv::imshow("Scale", Img);
            cv::imwrite("mikasa1_onechannell_resize4.png", Img);
            cv::waitKey(0);
        }

};

void testMyCV(){
    cv::Mat img = cv::imread("mikasa1_onechannell_resize4.png", cv::IMREAD_UNCHANGED);  // yse!! original 1 channel remains
    // cout << "type: " << img.type() << endl;  // type: 0 for CV_8U
    myCV cv0 = myCV(img, 1); // set margin
    // cv0.showImg();
    cout << cv0.GetImgMat().size() <<  endl;
    cout << cv0.GetImgMat() << endl;
    // cv0.ExtractChannel();
    // cv0.scaleSize(0.5);
    // cv0.showPhi();

    // cv0.showPhi();
    // cv0.PhiOntoImg();
    cout<<"Phi original" << endl;
    cout << cv0.getPhi() << endl;

    // Mat div0= cv0.GetImgMat();
    // Mat NGX0 = cv0.GetImgMat();
    // Mat NGY0 = cv0.GetImgMat();
    // cv0.NormalizedGradient(cv0.getPhi(), NGX0, NGY0); // Normalized Gradient
    // cv0.divNormalizedPhi(div0, NGX0, NGY0);
    // cout << "NGX0" << endl;
    // cout << NGX0 << endl;
    // cout << "NGY0" << endl;
    // cout << NGY0 << endl;

    // cout << "div" << endl;
    // cout << div0 << endl;

    cv0.fit(); // fit 
    cv0.showPhi(); // just showing Phi is enough!
    // cv0.PhiOntoImg();    
}

void testMyCV2(){
    // for Inhomogeneous picture like "mikasa1_onechannell_resize4.png", CV is bad
    // suitable margin is still important for the case!
    // margin 1  for resize4 [29 x 40] 
    // margin 10 for resize3; [59*80]
    // margin 30 for resize2 [118 x 160]
    // margin 

    cv::Mat img = cv::imread("mikasa1_onechannell_resize4.png", cv::IMREAD_UNCHANGED);  // yse!! original 1 channel remains // mikasa1_onechannell_resize3 // 6_onechannel
    // should only use 1 channel
    myCV cv1 = myCV(img, 1); // margin of Phi edge to outer:5, thickness of Phi edge is 1  // a margin bigger than 1 is better for cv! suitable margin is important
    // cv1.showImg();
    cv1.ImgInfo();

    // cout<<"Phi original" << endl;
    // cout << cv1.getPhi() << endl;
    // cv1.showPhi("Original Phi");

    cv1.setDeltT(1);  // 0.1 is suitable for 10000 iter, 0.001 too small
    // cv1.setStop(1e-8,1e-8, 1e-8); // set stop condition // seem to be useless, for each move is fairly small
    cv1.fit(10000); // max_iter
    
    cv1.showImg();
    cv1.showPhi("Last Phi"); // just showing Phi is enough!
    // cv1.PhiOntoImg();    
    // cv1.ColorPhiOnImg();
}

void testMyCV4(){ // use vector to grid search nice parameters!
    cv::Mat img = cv::imread("mikasa1_onechannell_resize3.png", cv::IMREAD_UNCHANGED); 

    vector<int>MarList({1,10, 20}); // vector
    vector<int>DeltaTdigit({1,2,3}); 
    for(auto digit : DeltaTdigit){  // iterator ! auto!
        for(auto mar : MarList){
            myCV cv = myCV(img, mar); // margin of Phi edge to outer:5, thickness of Phi edge is 1  // a margin bigger than 1 is better for cv! suitable margin is important

            cv.ImgInfo();
            cv.setDeltT(digit);  // 0.1 is enough for 10000 iter?
            cv.fit(1e4); // max_iter
        }
    }

}

void testMyCVTotal(){ // total test, different size of mikasa picture, automatically store Phi results in the corresponding folder
    // in ./mikasaTest or ./mikasa :  4 pic: resize1,2,3,4 respectively: 396*534, 118*160; 59*80; 29*40
    // how to set marginList?
    // for max_iter: 10000, delta_t: 0.1 is suitable, 0.01 is too small!

    fs::path path1 = "./mikasaTest"; // test // ./ can represent the current path!
    
    // iteration traverse
    fs::directory_iterator list(path1); // it works!!
    for(auto& it:list){  // resize 1,2,3,4
        fs::path file = it.path().filename();
        fs::path filenameNoExtension = file.stem();
        // cout<< filename << endl;

        string newFolderStr = "./PhiRes_" + filenameNoExtension.string(); // folderName
        fs::path newFolder(newFolderStr);
        if( ! fs::exists( newFolder ) ){
            if( fs::create_directory(newFolder) ){
            cout << "create: "<< newFolder << endl;
            }
        }

        // operate the file. use new folder as output folder
        cv::Mat img = cv::imread(file.string(), cv::IMREAD_UNCHANGED); 

        vector<int>DeltaTdigit({1,2}); 

        // different MarList for different size?
        vector<int>MarList; 
        vector<int>MarList1({1,10,30,50,100});
        vector<int>MarList2( {1,10,20,30,40});
        vector<int>MarList3({1,5,10,20});
        vector<int>MarList4({1,5,10});    

        string filenameNoExtensionStr = filenameNoExtension.string();
        int resizeNum =  static_cast<int>(filenameNoExtensionStr.at(filenameNoExtensionStr.size()-1)) - '0';
        switch (resizeNum){
            case 1:
                MarList = MarList1;
                break;
            case 2:
                MarList = MarList2;
                break;
            case 3:
                MarList = MarList3;
                break;
            case 4:
                MarList = MarList4;
                break;
            default:
                break;

        }

        for(auto digit : DeltaTdigit){  // iterator ! auto!
            for(auto mar : MarList){
                myCV cv = myCV(img, mar); // margin of Phi edge to outer:5, thickness of Phi edge is 1  // a margin bigger than 1 is better for cv! suitable margin is important

                cv.ImgInfo();
                cv.setDeltT(digit);  // 0.1 is enough for 10000 iter, 0.01 too small
                cv.fit(1e4); // max_iter
                cv.LastPhiOutput(newFolderStr);
            }
        }

    }

}

void testMyCV3(){
    cv::Mat img = cv::imread("6_onechannel.png", cv::IMREAD_UNCHANGED);  // get one channel
    myCV cv2 = myCV(img, 5);
    cv2.ImgInfo();

    cv2.setEpsilon(1e-6);
    cv2.setDeltT(0.001);
    cv2.setStop(1e-6,1e-6, 1e-6); // set stop condition
    cv2.fit(10000); // max_iter
    
    // but Phi does not change?
    cv2.showImg();
    cv2.showPhi("Last Phi"); // just showing Phi is enough!
    cv2.PhiOntoImg();  

}

class CVLearning
{
    private:
        Mat Img;
    //Mat 是一个类，由两个数据部分组成：
    //矩阵头(fixed size)：包含矩阵尺寸，存储方法，存储地址等信息
	//数据部分(data: uchar*pointer)：一个指向存储所有像素值的矩阵（根据所选存储方法的不同矩阵可以是不同的维数）的指针。
    public:
        CVLearning(Mat img){
            Img = img;
        }

        Mat getImg(){
            // Mat* p = &Img;
            return Img;
        }

        // aka ?? static
        static void onLightness(int lightness, void*userdata){ //user data is pointer to the data
            Mat image = *((Mat*)userdata);  //?
            Mat dst = Mat::zeros(image.size(), image.type());
            Mat m = Mat::zeros(image.size(), image.type());
            addWeighted(image, 1.0, m, 0, lightness, dst);
            imshow("Lightness adjustment", dst); //close this window first!! or occurs exceptations.
            cv::waitKey(0);
        }

        void trackingBar( Mat& image){ // use citation to pass in the "Mat imageoutside", and can change it!
            namedWindow("try", WINDOW_AUTOSIZE);
            int lightness = 50;
            int max_value = 100;
            int contrast_value = 100;
            createTrackbar("value_bar:", "try", &lightness, max_value, onLightness, (void*)(&image));
            onLightness(50, &image); 
        }

        void scaleSize(float scale){
            int h = this->Img.cols;
            int w = this->Img.rows;
            Mat zoom;

            resize(this->Img, zoom, Size(int(w*scale), int(h*scale)),0,0,INTER_LINEAR);
            cv::namedWindow("Scale", cv::WINDOW_NORMAL); 
            cv::imshow("Scale", zoom);
            cv::waitKey(0);
        }

};

void testCV(){
    cv::Mat img = cv::imread("./mikasa1.png");  // current path
	if (img.empty())
		std::cout << "image is empty or the path is invalid!" << std::endl;
	cv::imshow("Origin", img);  // showing at first seems to be important?

    Mat blurDst;
    GaussianBlur(img,blurDst,Size(9,9),11,11);  // blur the picture!
    imshow("GaussianDst",blurDst);

    cout << img.size() << endl;

	cv::waitKey(0);
    cv::waitKey(2000);
	cv::destroyAllWindows();
}

int main(){
    cout << "I am a fool but I can make it" << endl;

    // testMyCV();
    // testMyCV2();
    // testMyCV3();
    // testMyCV4();

    testMyCVTotal();

    cout << "YES!" << endl;

    return 0;
}

