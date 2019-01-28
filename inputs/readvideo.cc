#include "readvideo.hpp"

using namespace cv;
using namespace std;

bool ReadVideo::drawing_now_flag_;
bool ReadVideo::bbox_get_flag_;
cv::Rect2f ReadVideo::bbox_;
ReadVideo::ReadVideo(){};
ReadVideo::~ReadVideo(){};

void ReadVideo::IniRead(Rect2f &bboxGroundtruth, Mat &frame, string window_name, VideoCapture &capture, int flag1)
{
	ReadVideo::drawing_now_flag_ = false;
	ReadVideo::bbox_get_flag_ = false;
	//bool flag = false;
	// Register mouse callback
	cvNamedWindow(window_name.c_str(), CV_WINDOW_AUTOSIZE);
       // cv::imshow(window_name.c_str(), frame);
	cvSetMouseCallback(window_name.c_str(), ReadVideo::mouseHandler, NULL);
	
	cv::Mat temp;
	frame.copyTo(temp);
	while (!ReadVideo::bbox_get_flag_)
	{
                if(!flag1){
                    capture >> frame;  
                }             
                putText(frame, "!Choose Target!", cv::Point(10, 20), FONT_HERSHEY_SIMPLEX,0.5, Scalar(255, 0, 0), 2);
		rectangle(frame, bbox_, cv::Scalar(0, 0, 255), 3);
		imshow(window_name, frame);
                waitKey(1);
                if(flag1){
                    temp.copyTo(frame);  
                }
/*		int c = cvWaitKey(1);
		if (c == 27)
			break;
		if (c == 65)
		{
			printf("debug2\n");
			capture >> frame;
			frame.copyTo(temp);
			//imshow(window_name, frame);
			//continue;
		}
*/
	}
	// Remove callback
	cvSetMouseCallback(window_name.c_str(), NULL, NULL);
	printf("bbox:%d, %d, %d, %d\n", bbox_.x, bbox_.y, bbox_.width, bbox_.height);
	bboxGroundtruth.x = bbox_.x;
	bboxGroundtruth.y = bbox_.y;
	bboxGroundtruth.width = bbox_.width;
	bboxGroundtruth.height = bbox_.height;	
        
        bbox_.x = 0;
        bbox_.y = 0;
        bbox_.width = 0;
        bbox_.height = 0;
}
