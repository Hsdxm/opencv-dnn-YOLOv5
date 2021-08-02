#include "detect.h"
int main(int argc, char** argv)
{
	int anchors[] = {10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326};

	YOLOv5 *v5 = new YOLOv5(argv[1], 640,480,10,anchors);

    Mat frame = imread(argv[2]);
	vector<BBox> results = v5->process(frame);

	for (size_t i=0;i<results.size();i++)
	{
		int left = results[i].left;
		int right = results[i].right;
		int top = results[i].top;
		int bottom = results[i].bottom;
		rectangle(frame, Point(left,top),Point(right,bottom),Scalar(0,0,255),4,4,0);
	}
	resize(frame,frame,Size(1920,1080));
	imshow("result",frame);
	waitKey(0);
    return 0;
}
