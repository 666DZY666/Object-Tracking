#include "kcf/kcftracker.hpp"
#include "eco/eco.hpp"
#include "eco/parameters.hpp"

/*#ifdef USE_CAFFE
#include "goturn/network/regressor.h"
#include "goturn/tracker/tracker.h"
#endif
*/
#include "inputs/readdatasets.hpp"
#include "inputs/readvideo.hpp"
//#include "inputs/openpose.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

// Convert to string
int main(int argc, char **argv)
{
    int key, flag = 0, flag1 = 0, flag2 = 0, count = 0;
    char f;
/*
    // Read using openpose============================================
    cv::Rect2f bboxGroundtruth;
    cv::Mat frame, frameDraw;

    OpenPose openpose;
    openpose.IniRead(bboxGroundtruth);
    VideoCapture capture(0); // open the default camera
    if (!capture.isOpened()) // check if we succeeded
        return -1;
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    capture >> frame;
    //frame.copyTo(frameDraw);
    //rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
    //imshow("Tracking", frameDraw);
*/

    // Read from Video and choose a bbox===============================
    //::google::InitGoogleLogging(argv[0]);
    Rect2f bboxGroundtruth;
    Mat frame, frameDraw;
    VideoCapture capture;
    //VideoCapture capture(0);
    printf("Camera-c or Video-v: ");
    scanf("%c",&f);
    if(f == 'c'){
        capture.open(1);
    }
    else if(f == 'v'){
        capture.open("media/2.mp4");
    }

    if (!capture.isOpened())
    {
        std::cout << "Capture device failed to open!" << std::endl;
        return -1;
    }
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 1080);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    capture >> frame;
    //frameDraw.copyTo(frame);

    string window_name = "Object Tracking";
    namedWindow(window_name, 0);
    ReadVideo readvideo;
    //readvideo.IniRead(bboxGroundtruth, frameDraw, window_name, capture);
    
    //************************* Read from the datasets ****************************
/*  //::google::InitGoogleLogging(argv[0]);
   
    cv::Rect2f bboxGroundtruth;
    cv::Mat frame, frameDraw;
    ReadDatasets readdatasets;
    readdatasets.IniRead(bboxGroundtruth, frame);
    frame.copyTo(frameDraw);
    readdatasets.DrawGroundTruth(bboxGroundtruth, frameDraw);
*/
    //**************************** 創建 Trackers ***********************************
/*  // Create Opencv tracker:
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN"};
    string trackerType = trackerTypes[2];
    Ptr<cv::Tracker> opencvtracker;
    if (trackerType == "BOOSTING")
        opencvtracker = cv::TrackerBoosting::create();
    if (trackerType == "MIL")
        opencvtracker = cv::TrackerMIL::create();
    if (trackerType == "KCF")
        opencvtracker = cv::TrackerKCF::create();
    if (trackerType == "TLD")
        opencvtracker = cv::TrackerTLD::create();
    if (trackerType == "MEDIANFLOW")
        opencvtracker = cv::TrackerMedianFlow::create();
    if (trackerType == "GOTURN")
        opencvtracker = cv::TrackerGOTURN::create();
    Rect2d opencvbbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
    opencvtracker->init(frame, opencvbbox);
*/
/*  // Create DSSTTracker:
    DSST = true;
    kcf::KCFTracker dssttracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
    Rect2d dsstbbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
    dssttracker.init(frame, dsstbbox);
*/
/*#ifdef USE_CAFFE
    // Create GOTURN tracker:
    const string model_file = "goturn/nets/deploy.prototxt";
    const string pretrain_file = "goturn/nets/goturun_tracker.caffemodel";
    int gpu_id = 0;
    Regressor regressor(model_file, pretrain_file, gpu_id, false);
    goturn::Tracker goturntracker(false);
    cv::Rect goturnbbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
    BoundingBox bbox_gt;
    BoundingBox bbox_estimate_uncentered;
    bbox_gt.getRect(goturnbbox);
    goturntracker.Init(frame, bbox_gt, &regressor);
#endif
*/
    //**************************** 創建 Trackers ***********************************
    // Create ECO trakcer;
    eco::ECO ecotracker;
    eco::EcoParameters parameters;
    Rect2f ecobbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
    //parameters.max_score_threshhold = 0.1;
    // when use cn feature:
    parameters.cn_features.fparams.tablename = "/home/nvidia/Develop/Project/Tracker/OpenTracker/eco/look_tables/CNnorm.txt";
    //ecotracker.init(frame, ecobbox, parameters);

    // Create KCF Tracker —— HOG + LAB:
    bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false; //HOG + LAB(color)
    kcf::KCFTracker kcftracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
    Rect2d kcfbbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
    //kcftracker.init(frame, kcfbbox);
    
    Point pointInterest, pointInterest1;
    while (frame.data)
    {
        //frame.copyTo(frameDraw);
        key = cvWaitKey(6);
        //******************************** 初始化 Trackers ***********************************
        //ECO —— HOG
        //攝像頭
        if(key == 'a'){
            //frame.copyTo(frameDraw);
            flag = 1;
            flag1 = 0;
            parameters.useDeepFeature = false;
            parameters.useHogFeature = true;
            parameters.useColorspaceFeature = false;//not have
            parameters.useCnFeature = false;//RGB
            parameters.useIcFeature = false;//Gray
            parameters.use_scale_filter = false;
            parameters.use_gpu = false;
            parameters.gpu_id = 0;

            readvideo.IniRead(bboxGroundtruth, frame, window_name, capture, flag1);
            //ECO初始化
            Rect2f ecobbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
            ecotracker.init(frame, ecobbox, parameters);
            pointInterest.x = ecobbox.x + ecobbox.width / 2;
            pointInterest.y = ecobbox.y + ecobbox.height / 2;
            pointInterest1.x = pointInterest.x;
            pointInterest1.y = pointInterest.y;
        }
        //ECO —— HOG
        //視頻
        if(key == 'f'){
            //frame.copyTo(frameDraw);
            flag = 1;
            flag1 = 1;
            parameters.useDeepFeature = false;
            parameters.useHogFeature = true;
            parameters.useColorspaceFeature = false;//not have
            parameters.useCnFeature = false;//RGB
            parameters.useIcFeature = false;//Gray
            parameters.use_scale_filter = false;
            parameters.use_gpu = false;
            parameters.gpu_id = 0;

            readvideo.IniRead(bboxGroundtruth, frame, window_name, capture, flag1);
            //ECO初始化
            Rect2f ecobbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
            ecotracker.init(frame, ecobbox, parameters);
            pointInterest.x = ecobbox.x + ecobbox.width / 2;
            pointInterest.y = ecobbox.y + ecobbox.height / 2;
            pointInterest1.x = pointInterest.x;
            pointInterest1.y = pointInterest.y;
        }
        //ECO —— HOG + CN
        //攝像頭
        if(key == 's'){
            //frame.copyTo(frameDraw);
            flag = 1;
            flag1 = 0;
            parameters.useDeepFeature = false;
            parameters.useHogFeature = true;
            parameters.useColorspaceFeature = false;//not have
            parameters.useCnFeature = true;//RGB
            parameters.useIcFeature = true;//Gray
            parameters.use_scale_filter = false;
            parameters.use_gpu = false;
            parameters.gpu_id = 0;

            readvideo.IniRead(bboxGroundtruth, frame, window_name, capture, flag1);
            //ECO初始化
            Rect2f ecobbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
            ecotracker.init(frame, ecobbox, parameters);
            pointInterest.x = ecobbox.x + ecobbox.width / 2;
            pointInterest.y = ecobbox.y + ecobbox.height / 2;
            pointInterest1.x = pointInterest.x;
            pointInterest1.y = pointInterest.y;
        }
        //ECO —— HOG + CN
        //視頻
        if(key == 'g'){
            //frame.copyTo(frameDraw);
            flag = 1;
            flag1 = 1;
            parameters.useDeepFeature = false;
            parameters.useHogFeature = true;
            parameters.useColorspaceFeature = false;//not have
            parameters.useCnFeature = true;//RGB
            parameters.useIcFeature = true;//Gray
            parameters.use_scale_filter = false;
            parameters.use_gpu = false;
            parameters.gpu_id = 0;

            readvideo.IniRead(bboxGroundtruth, frame, window_name, capture, flag1);
            //ECO初始化
            Rect2f ecobbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
            ecotracker.init(frame, ecobbox, parameters);
            pointInterest.x = ecobbox.x + ecobbox.width / 2;
            pointInterest.y = ecobbox.y + ecobbox.height / 2;
            pointInterest1.x = pointInterest.x;
            pointInterest1.y = pointInterest.y;
        }
        //ECO —— HOG + CN + CNN(GPU)
        //攝像頭
        else if(key == 'd'){
            //frame.copyTo(frameDraw);
            flag = 1;
            flag1 = 0;
            parameters.useDeepFeature = true;
            parameters.useHogFeature = true;
            parameters.useColorspaceFeature = false;//not have
            parameters.useCnFeature = true;//RGB
            parameters.useIcFeature = true;//Gray
            parameters.use_scale_filter = false;
            parameters.use_gpu = true;
            parameters.gpu_id = 0;
    
            readvideo.IniRead(bboxGroundtruth, frame, window_name, capture, flag1);
             //ECO初始化
            Rect2f ecobbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
            ecotracker.init(frame, ecobbox, parameters);
            pointInterest.x = ecobbox.x + ecobbox.width / 2;
            pointInterest.y = ecobbox.y + ecobbox.height / 2;
            pointInterest1.x = pointInterest.x;
            pointInterest1.y = pointInterest.y;
        }
        //KCF —— HOG + LAB
        //攝像頭
        else if(key == 'z'){
            //frame.copyTo(frameDraw);
            flag = 2;
            flag1 = 0;
            readvideo.IniRead(bboxGroundtruth, frame, window_name, capture, flag1);
            //KCF初始化
            Rect2d kcfbbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
            kcftracker.init(frame, kcfbbox);
            pointInterest.x = kcfbbox.x + kcfbbox.width / 2;
            pointInterest.y = kcfbbox.y + kcfbbox.height / 2;
            pointInterest1.x = pointInterest.x;
            pointInterest1.y = pointInterest.y;
        }
        //KCF —— HOG + LAB
        //視頻
        else if(key == 'x'){
            //frame.copyTo(frameDraw);
            flag = 2;
            flag1 = 1;
            readvideo.IniRead(bboxGroundtruth, frame, window_name, capture, flag1);
            //KCF初始化
            Rect2d kcfbbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
            kcftracker.init(frame, kcfbbox);
            pointInterest.x = kcfbbox.x + kcfbbox.width / 2;
            pointInterest.y = kcfbbox.y + kcfbbox.height / 2;
            pointInterest1.x = pointInterest.x;
            pointInterest1.y = pointInterest.y;
        }
        //ECO + KCF
        //攝像頭
        else if(key == 'q'){
            //frame.copyTo(frameDraw);
            flag = 3;
            flag1 = 0;
            parameters.useDeepFeature = false;
            parameters.useHogFeature = true;
            parameters.useColorspaceFeature = false;//not have
            parameters.useCnFeature = true;//RGB
            parameters.useIcFeature = true;//Gray
            parameters.use_scale_filter = false;
            parameters.use_gpu = false;
            parameters.gpu_id = 0;
           
            readvideo.IniRead(bboxGroundtruth, frame, window_name, capture, flag1);
            //ECO初始化
            Rect2f ecobbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
            ecotracker.init(frame, ecobbox, parameters);
            //KCF初始化
            Rect2d kcfbbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
            kcftracker.init(frame, kcfbbox);
            pointInterest.x = kcfbbox.x + kcfbbox.width / 2;
            pointInterest.y = kcfbbox.y + kcfbbox.height / 2;
            pointInterest1.x = pointInterest.x;
            pointInterest1.y = pointInterest.y;
        }
        //ECO + KCF
        //視頻
        else if(key == 'w'){
            //frame.copyTo(frameDraw);
            flag = 3;
            flag1 = 1;
            parameters.useDeepFeature = false;
            parameters.useHogFeature = true;
            parameters.useColorspaceFeature = false;//not have
            parameters.useCnFeature = true;//RGB
            parameters.useIcFeature = true;//Gray
            parameters.use_scale_filter = false;
            parameters.use_gpu = false;
            parameters.gpu_id = 0;
           
            readvideo.IniRead(bboxGroundtruth, frame, window_name, capture, flag1);
            //ECO初始化
            Rect2f ecobbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
            ecotracker.init(frame, ecobbox, parameters);
            //KCF初始化
            Rect2d kcfbbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
            kcftracker.init(frame, kcfbbox);
            pointInterest.x = kcfbbox.x + kcfbbox.width / 2;
            pointInterest.y = kcfbbox.y + kcfbbox.height / 2;
            pointInterest1.x = pointInterest.x;
            pointInterest1.y = pointInterest.y;
        }
        else if(key == 'o'){
            flag2 = 1;
        }
        else if(key == 'p'){
            flag2 = 0;
        }
        else if(key == 'e'){
            flag = 0;
        }

        //********************************* Tracking ******************************************
        //ECO
        if(flag == 1 || flag == 3){
	    double timeeco = (double)getTickCount();
            bool okeco = ecotracker.update(frame, ecobbox);
            float fpseco = getTickFrequency() / ((double)getTickCount() - timeeco);
	    if (okeco){
                //frame.copyTo(frameDraw);
                rectangle(frame, ecobbox, Scalar(0, 0, 255), 3);

                pointInterest.x = ecobbox.x + ecobbox.width / 2;
                pointInterest.y = ecobbox.y + ecobbox.height / 2;
                circle(frame, pointInterest, 2, Scalar(0, 0, 255), 2);
         
                if(flag2){
                    count++;
                    if(count > 2){
                       line(frame, pointInterest1, pointInterest, Scalar(0, 0, 255), 2);
                       pointInterest1.x = pointInterest.x;
                       pointInterest1.y = pointInterest.y;
                       count = 0;
                    }               
                }
	    }
	    else{
	        putText(frame, "!!!NO Target!!!", cv::Point(130, 20), FONT_HERSHEY_SIMPLEX,
		            0.5, Scalar(0, 0, 255), 2);
	    }
              // Display FPS
	    ostringstream os;
	    os << float(fpseco);
	    putText(frame, "FPS: " + os.str(), Point(10, 20), FONT_HERSHEY_SIMPLEX,
		        0.5, Scalar(0, 0, 255), 2);
        }
        //KCF
        if(flag == 2 || flag == 3){
            double timerkcf = (double)getTickCount();
            bool okkcf = kcftracker.update(frame, kcfbbox);
            float fpskcf = getTickFrequency() / ((double)getTickCount() - timerkcf);
            if (okkcf){
                //frame.copyTo(frameDraw);
                rectangle(frame, kcfbbox, Scalar(0, 255, 0), 3);

                pointInterest.x = kcfbbox.x + kcfbbox.width / 2;
                pointInterest.y = kcfbbox.y + kcfbbox.height / 2;
                circle(frame, pointInterest, 2, Scalar(0, 255, 0), 2);
                
                if(flag2){
                    count++;
                    if(count > 2){
                        line(frame, pointInterest1, pointInterest, Scalar(0, 255, 0), 2);
                        pointInterest1.x = pointInterest.x;
                        pointInterest1.y = pointInterest.y;
                        count = 0;
                    }    
                }
            }
            else{
	        putText(frame, "!!!NO Target!!!", cv::Point(130, 50), FONT_HERSHEY_SIMPLEX,
		            0.5, Scalar(0, 255, 0), 2);
            }
	      // Display FPS
            ostringstream os1;
            os1 << float(fpskcf);
	    putText(frame, "FPS: " + os1.str(), Point(10, 50), FONT_HERSHEY_SIMPLEX,
		        0.5, Scalar(0, 255, 0), 2);
        }
/*      //Opencv
        double timercv = (double)getTickCount();
        bool okopencv = opencvtracker->update(frame, opencvbbox);
        float fpscv = getTickFrequency() / ((double)getTickCount() - timercv);
        if (okopencv){
            rectangle(frameDraw, opencvbbox, Scalar(255, 0, 0), 2, 1);
        }
        else{
            putText(frameDraw, "Opencv tracking failure detected", cv::Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 0), 2);
        }
*/
/*      //DSST
        double timerdsst = (double)getTickCount();
        bool okdsst = dssttracker.update(frame, dsstbbox);
        float fpsdsst = getTickFrequency() / ((double)getTickCount() - timerdsst);
        if (okdsst){
            rectangle(frameDraw, dsstbbox, Scalar(0, 0, 255), 2, 1);
        }
        else{
            putText(frameDraw, "DSST tracking failure detected", cv::Point(10, 110), FONT_HERSHEY_SIMPLEX,
                    0.75, Scalar(0, 0, 255), 2);
        }
*/
/*#ifdef USE_CAFFE
        //GOTURN=====================
        double timergoturn = (double)getTickCount();
        goturntracker.Track(frame, &regressor, &bbox_estimate_uncentered);
        bbox_estimate_uncentered.putRect(goturnbbox);
        float fpsgoturn = getTickFrequency() / ((double)getTickCount() - timergoturn);
        rectangle(frameDraw, goturnbbox, Scalar(255, 255, 0), 2, 1);
#endif
*/
/*      // Draw the label of trackers
        putText(frameDraw, "Opencv ", cv::Point(frameDraw.cols - 180, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 0), 2);
        line(frameDraw, cv::Point(frameDraw.cols - 100, 50), cv::Point(frameDraw.cols - 10, 50), Scalar(255, 0, 0), 2, 1);
	putText(frameDraw, "KCF ", cv::Point(frameDraw.cols - 180, 75), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 2);
	line(frameDraw, cv::Point(frameDraw.cols - 100, 75), cv::Point(frameDraw.cols - 10, 75), Scalar(0, 255, 0), 2, 1);
	putText(frameDraw, "DSST ", cv::Point(frameDraw.cols - 180, 100), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
	line(frameDraw, cv::Point(frameDraw.cols - 100, 100), cv::Point(frameDraw.cols - 10, 100), Scalar(0, 0, 255), 2, 1);
        putText(frameDraw, "ECO ", cv::Point(frameDraw.cols - 180, 125), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 255), 2);
	line(frameDraw, cv::Point(frameDraw.cols - 100, 125), cv::Point(frameDraw.cols - 10, 125), Scalar(255, 0, 255), 2, 1);
	#ifdef USE_CAFFE
            putText(frameDraw, "GOTURN ", cv::Point(frameDraw.cols - 180, 150), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 0), 2);
            line(frameDraw, cv::Point(frameDraw.cols - 100, 150), cv::Point(frameDraw.cols - 10, 150), Scalar(255, 255, 0), 2, 1);
	#endif
	//Display frameDraw.=========================================================
	cvNamedWindow("Tracking", CV_WINDOW_NORMAL);
*/
        imshow("Object Tracking", frame);
        waitKey(1);
        //Read the next frame
        capture >> frame;
        if (frame.empty())
            return false;
    }
#ifdef USE_MULTI_THREAD
    void *status;
    int rc = pthread_join(ecotracker.thread_train_, &status);
    if (rc)
    {
         cout << "Error:unable to join," << rc << std::endl;
         exit(-1);
    }
#endif
    cvDestroyWindow("OpenTracker");
    return 0;
}
