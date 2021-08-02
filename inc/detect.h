#ifndef __OPENCV_DNN_YOLOV5_H__
#define	__OPENCV_DNN_YOLOV5_H__

#include <iostream>
#include <vector>
using namespace std;
#include <opencv2/opencv.hpp>

using namespace cv;

#define	SIGMOID(x) (float)(1.0f/(1+exp(-x)))

typedef struct {
	int left;
	int top;
	int right;
	int bottom;
	float score;
	int type;
} BBox;

class YOLOv5
{
	private:
		int				mInputWidth;
		int				mInputHeight;
		int				mClassNum;
		vector<int>		mScales;
		vector<int>		mOutputWidth;
		vector<int>		mOutputHeight;
		vector<int>		mAnchors;
		dnn::Net		mNet;

	public:
		YOLOv5(char *model, int width, int height, int classNum, int *anchors);
		~YOLOv5();
		Mat preProcess(Mat img);
		vector<BBox> process(Mat img);
		vector<BBox> postProcess(vector<Mat> outputs);
};
#endif
