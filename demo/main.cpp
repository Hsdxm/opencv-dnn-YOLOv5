#include <iostream>
#include <fstream>
using namespace std;

#include "detect.h"

string replaceStr(string str, string A, string B)
{
	int pos = str.find(A);
	string result = str;
	while(pos != -1)
	{
		result = result.replace(pos,A.length(),B);
		pos = result.find(A);
	}
	return result;
}

int main(int argc, char** argv)
{
	//int anchors[] = {10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326};
	//int anchors[]= {3,3, 5,5, 9,8, 7,13, 13,12, 20,16,26,27, 48,42, 98,98};
	int anchors[]= {8,3,10,9,24,4, 21,12,46,9,25,25, 67,21,53,47,135,69};

	YOLOv5 *v5 = new YOLOv5(argv[1], 640,640,10,anchors);

	ifstream in(argv[2]);
	string imagePath;
	int counter = 0;
	while(getline(in,imagePath))	{
		
		Mat frame = imread(imagePath);
		if (frame.empty())
			break;
		vector<BBox> results = v5->process(frame);
		string txtPath = replaceStr(imagePath,"jpg","txt");
		cout << ++counter << "  " <<imagePath <<endl;

		FILE *f = fopen(txtPath.c_str(),"w");
		for (size_t i=0;i<results.size();i++)
		{
			int left = results[i].left;
			int right = results[i].right;
			int top = results[i].top;
			int bottom = results[i].bottom;
		//	rectangle(frame, Point(left,top),Point(right,bottom),Scalar(0,0,255),4,4,0);
		
			char text[512]={0};
			float x = (results[i].left + results[i].right)/ (2.*frame.cols);
			float y = (results[i].top +  results[i].bottom)/(2.*frame.rows);
			float w = (results[i].right - results[i].left) / (1.*frame.cols);
			float h = (results[i].bottom - results[i].top)/ (1.*frame.rows);
			snprintf(text,sizeof(text),"%d %f %f %f %f\n",results[i].type,x,y,w,h);
			fwrite(text,1,strlen(text),f);
		}
		fclose(f);
		/*
		if (frame.cols > 1920 || frame.rows > 1080)
			resize(frame,frame,Size(1920,1080));
		imshow("result",frame);
		waitKey(0);
		*/
	}
    return 0;
}
