#include "detect.h"
int main(int argc, char** argv)
{
	//int anchors[] = {10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326};
	int anchors[]= {3,3, 5,5, 9,8, 7,13, 13,12, 20,16,26,27, 48,42, 98,98};

	YOLOv5 *v5 = new YOLOv5(argv[1], 416,384,4,anchors);

	VideoCapture cap(argv[2]);
	if (!cap.isOpened())	{
		printf("[%s %d] open video error: %s\n", __FILE__, __LINE__, argv[2]);
		return -1;
	}

    Mat frame;
	while(1)	{
		cap.read(frame);
		if (frame.empty())
			break;
		vector<BBox> results = v5->process(frame);

		for (size_t i=0;i<results.size();i++)
		{
			int left = results[i].left;
			int right = results[i].right;
			int top = results[i].top;
			int bottom = results[i].bottom;
			rectangle(frame, Point(left,top),Point(right,bottom),Scalar(0,0,255),4,4,0);
		}
		if (frame.cols > 1920 || frame.rows > 1080)
			resize(frame,frame,Size(1920,1080));
		imshow("result",frame);
		waitKey(40);
	}
    return 0;
}
