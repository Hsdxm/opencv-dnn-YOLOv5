#include "detect.h"

std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net)
{
    static std::vector<cv::String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
         
        //get the names of all the layers in the network
        std::vector<cv::String> layersNames = net.getLayerNames();
         
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
	for (auto a=names.begin();a!=names.end();a++)
		cout << *a <<endl;
    return names;
}

bool cmp(const BBox& a, const BBox& b)
{
	return a.score > b.score;
}

float iou(BBox a, BBox b)
{
	Rect r1 = Rect(Point(a.left, a.top),Point(a.right, a.bottom));
	Rect r2 = Rect(Point(b.left, b.top),Point(b.right, b.bottom));
	Rect inter = r1 & r2;
	if (inter.area() <= 0)
		return 0.;
	return inter.area()*1. / (r1.area() + r2.area() - inter.area());
}

void doNMS(vector<BBox>& boxes)
{
	for (size_t i=0;i<boxes.size();i++)
		for (size_t j=i+1;j<boxes.size();j++)
			if (iou(boxes[i],boxes[j]) > 0.45)
			{
				boxes.erase(boxes.begin()+j--);
			}
}

void dumpBBoxes(vector<BBox> boxes)
{
	for (size_t i=0;i<boxes.size();i++)
		printf("%d  %d  %d  %d\n", boxes[i].left, boxes[i].top,boxes[i].right,boxes[i].bottom);
}

YOLOv5::YOLOv5(char *model, int width, int height, int classNum, int *anchors)
{
	mNet = dnn::readNetFromONNX(model);
	for (int i=0;i<18;i++)
		mAnchors.push_back(anchors[i]);

	mInputWidth = width;
	mInputHeight = height;
	mClassNum = classNum;
	
	mScales.push_back(8);
	mScales.push_back(16);
	mScales.push_back(32);

	for (int i=0;i<3;i++)
	{
		mOutputWidth.push_back(mInputWidth/mScales[i]);
		mOutputHeight.push_back(mInputHeight/mScales[i]);
	}
}

YOLOv5::~YOLOv5()
{
}

Mat YOLOv5::preProcess(Mat img)
{
	Mat dst = Mat::zeros(mInputHeight,mInputWidth,CV_8UC3);;
	float scaleSize = min(1.*mInputWidth/img.cols, 1.*mInputHeight/img.rows);
	int dstWidth = img.cols * scaleSize;
	int dstHeight = img.rows*scaleSize;
	int wIndex = (mInputWidth - dstWidth)/2;
	int hIndex = (mInputHeight - dstHeight)/2;
	Mat resized;
	resize(img,resized, Size(dstWidth,dstHeight));
	dst.setTo(Scalar(127,127,127));
	Rect r = Rect(wIndex,hIndex,dstWidth,dstHeight);
	resized.copyTo(dst(r));
	return dst;
}

vector<BBox> YOLOv5::postProcess(vector<Mat> outputs)
{
	vector<BBox> boxes;
	//第i个输出层
	for (size_t i=0;i<outputs.size();i++)
	{
		float *data = (float*)(outputs[i].data);
		size_t index = 0;
		//第d个维度
		for (size_t d= 0;d<3;d++)
		{
			//列
			for (size_t m=0;m<mOutputHeight[i];m++)
			{
				//行
				for (size_t n=0;n<mOutputWidth[i];n++)
				{
					float objConf = SIGMOID(data[index+4]);
					if (objConf > 0.1)
					{
						float x = (float)(SIGMOID(data[index+0]) * 2 - 0.5 + n) * mScales[i];
						float y = (float)(SIGMOID(data[index+1]) * 2 - 0.5 + m) * mScales[i];
						float w = (float)pow(SIGMOID(data[index+2])*2,2) * mAnchors[6*i+2*d];
						float h = (float)pow(SIGMOID(data[index+3])*2,2) * mAnchors[6*i+2*d+1];
						//类别
						for (size_t c = 0;c<10;c++)
						{
							float score = SIGMOID(data[index+5+c]) * objConf;
							if (score  > 0.1)
							{
								BBox box;
								box.left = x-w/2.;
								box.right = x+w/2;
								box.top = y-h/2.;
								box.bottom = y+h/2.;
								box.score = score;
								box.type = c;
								boxes.push_back(box);
							}
						}
					}
					index += (5+10);
				}
			}
		}
	}
	
	if (boxes.size() > 1)
	{
		std::sort(boxes.begin(),boxes.end(), cmp);
		doNMS(boxes);
	}
	return boxes;
}

vector<BBox> YOLOv5::process(Mat img)
{
	Mat resized = preProcess(img);
	Mat blob;
	dnn::blobFromImage(resized, blob, 1/255., Size(mInputWidth, mInputHeight), Scalar(0.,0.,0.), true, false);
	mNet.setInput(blob);
	vector<Mat> outputs;
	mNet.forward(outputs, getOutputsNames(mNet));
	vector<BBox> boxes = postProcess(outputs);
	
	float scaleSize = min(1.*mInputWidth/img.cols, 1.*mInputHeight/img.rows);
	int dstWidth = img.cols * scaleSize;
	int dstHeight = img.rows*scaleSize;
	int wIndex = (mInputWidth - dstWidth)/2;
	int hIndex = (mInputHeight - dstHeight)/2;
	for (size_t i=0;i<boxes.size();i++)
	{
		boxes[i].left  = (boxes[i].left - wIndex) * img.cols / dstWidth;
		boxes[i].right  = (boxes[i].right - wIndex) * img.cols / dstWidth;
		boxes[i].top  = (boxes[i].top - hIndex) * img.rows / dstHeight;
		boxes[i].bottom  = (boxes[i].bottom - hIndex) * img.rows / dstHeight;
	}
	return boxes;
}

/*
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
*/
