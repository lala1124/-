#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_provider_factory.h>
#include <tensorrt_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/videoio.hpp>


//#include <windows.h>

#define Fire (0,0,255)
#define Smoke (0,255,255)
using namespace std;
using namespace cv;
using namespace Ort;


int GPUcheck(void);
int Cap_choose(void);

int c, cap_choose,res,sig,name;
string webcap,jieguofile;
VideoCapture capture;
Mat frame;
bool  GPU = 0;
struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
};

typedef struct param {    //作为线程参数传入的结构体
	int *array;
	int start, end;
}param;

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

class check
{
public:
	check(Net_config config);
	void detect(Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	vector<string> class_names;
	int num_class;

	float confThreshold;
	float nmsThreshold;
	vector<float> input_image_;
	void normalize_(Mat img);
	void nms(vector<BoxInfo>& input_boxes);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "check");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

check::check(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	string classesFile = "data.names";
	string model_path = config.modelpath;
	std::string widestr = std::string(model_path.begin(), model_path.end());
	if (GPU == 1)
	{
		//OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0);
		OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);         //GPU加速
		
	}
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->nout = output_node_dims[0][2];
	this->num_proposal = output_node_dims[0][1];

	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
}

void check::normalize_(Mat img)
{
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;
			}
		}
	}
}


void check::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void check::detect(Mat& frame)
{
	Mat dstimg;
	sig = 0;
	resize(frame, dstimg, Size(this->inpWidth, this->inpHeight));
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	/////generate proposals
	vector<BoxInfo> generate_boxes;
	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, k = 0; ///cx,cy,w,h,box_score, class_score
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	for (n = 0; n < this->num_proposal; n++)   ///特征图尺度
	{
		float box_score = pdata[4];
		if (box_score > this->confThreshold)
		{
			int max_ind = 0;
			float max_class_socre = 0;
			for (k = 0; k < num_class; k++)
			{
				if (pdata[k + 5] > max_class_socre)
				{
					max_class_socre = pdata[k + 5];
					max_ind = k;
				}
			}
			max_class_socre *= box_score;
			if (max_class_socre > this->confThreshold)
			{ 
				float cx = pdata[0] * ratiow;  ///cx
				float cy = pdata[1] * ratioh;   ///cy
				float w = pdata[2] * ratiow;   ///w
				float h = pdata[3] * ratioh;  ///h
				
				float xmin = cx - 0.5 * w;
				float ymin = cy - 0.5 * h;
				float xmax = cx + 0.5 * w;
				float ymax = cy + 0.5 * h;
				generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, max_ind });
			}
		}
		pdata += nout;
	}

	nms(generate_boxes);
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		int ymax = int(generate_boxes[i].y2);
		rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label] + ":" + label;
		putText(frame, label, Point(xmin, ymax + 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
		sig = 1;
	}
}

void cap()
{
	int choose;
	//capture.open("res2.avi");
	//VideoCapture capture("rtsp://admin:302wustxx@10.160.65.241/Streaming/Channels/1");
	//VideoCapture capture(0);
	/*if(cap_choose == 1)
		capture.open(0);*/
	/*if (cap_choose == 2)
		capture.open(webcap);*/
	/*if (capture.isOpened())
		capture >> frame;*/
	while (capture.isOpened())
	{
		//if(go = 0)
		capture >> frame;
		if (cap_choose == 3)
			waitKey(10);                 //视频文件设置播放速度
		if (frame.empty())
			break;
		//waitKey(16);
		if (c == 'q')
			break;
	}
	res = 3;
}


string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method)
{
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + to_string(capture_width) + ", height=(int)" +
           to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + to_string(flip_method) + " ! video/x-raw, width=(int)" + to_string(display_width) + ", height=(int)" +
           to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

void test()
{
	Mat tframe;
	struct timeval start, finish;
	int i = 0;
	static int name = 0;
	float  duration;
	Net_config check_nets = { 0.25, 0.45, "best.onnx" };  
	check net(check_nets);
	char str[2];
	string fps;
	float f = 0;
	while (capture.isOpened())
	{
		if (!frame.empty())
		{
			tframe = frame;
			gettimeofday( &start, NULL );
			net.detect(tframe);
			gettimeofday( &finish, NULL );
			duration = (float)(finish.tv_sec - start.tv_sec)*1000;
			duration += (float)(finish.tv_usec - start.tv_usec)/1000;
			fps = "FPS: " + std::to_string(1000/duration);
			putText(tframe, fps, Point(300, 50),
				FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
				2);
			namedWindow("output", WINDOW_NORMAL);
			imshow("output", tframe);
			c = waitKey(1);
			if (c == 'q')
				break;

		}
		if (res == 3)
			break;
	}
	destroyAllWindows();
	f = 1;
}

int Cap_choose(void)
{
	char choose1;
	string ip, username, password,video1;
	res = 0;
	c = 'x';
	cout << "****请选择检测方式：1本地摄像头；2网络摄像头（Streaming/Channels/1）；3视频文件"<<";4自定义网络摄像头；q退出；****" << endl;
	cin >> choose1;
	cap_choose = 0;
	if (choose1 == '1')
	{
		int capture_width = 1280 ;
		int capture_height = 720 ;
		int display_width = 1280 ;
		int display_height = 720 ;
		int framerate = 60 ;
		int flip_method = 0 ;

		//创建管道
		string pipeline = gstreamer_pipeline(capture_width,
    capture_height,
    display_width,
    display_height,
    framerate,
    flip_method);
		std::cout << "使用gstreamer管道: \n\t" << pipeline << "\n";

		//管道与视频流绑定
		capture.open(pipeline, cv::CAP_GSTREAMER);
		cap_choose = 1;
		
		
	}
	if (choose1 == '2')
	{
		cout << "****请输入摄像头ip地址****" << endl;
		cin >> ip;
		cout <<  "****请输入用户名****" << endl;
		cin >> username;
		cout << "****输入密码****" << endl;
		cin >> password;
		webcap = "rtsp://" + username + ":" + password + "@" + ip + "/Streaming/Channels/1";
		capture.open(webcap);
		cap_choose = 2;
	}
	if (choose1 == '3')
	{
		cout << "****输入视频文件绝对路径****" << endl;
		cin >> video1;
		capture.open( video1);
		cap_choose = 3;
	}
	if (choose1 == '4')
	{
		cout << "请输入IP" << endl;
		cin >> webcap;
		capture.open(webcap);
		cap_choose = 4;
	}
	if (choose1 == 'q')
		return 999;
	if (capture.isOpened())
	{
		return 0;
	}
	else
		cout << "****无法打开****" << endl;
	return 0;
}

int GPUcheck(void)
{
	char GPU_check;
	cout << "****是否开启GPU加速：输入1开启，输入2使用CPU" << endl;
	GPUCheck:
	cin >> GPU_check;
	if (GPU_check == '1')
	{
		GPU = 1;
		return 0;
	}
	if (GPU_check == '2')
	{
		GPU = 0;
		return 0;
	}
	else
	{
		cout << "****输入有误，请重新输入****" << endl;
		goto GPUCheck;
	}

}

int main()
{
	int key = 0;
	char GPU_check;
	string q;
	cout << "***************烟火检测系统v1.0***************" << endl;
	GPUcheck();
	while (1)
	{
		key = Cap_choose();
		thread th1(cap);
    		thread th2(test);
		th1.join(); 
   		th2.join();
		if (key == 999)
			return 0;
	}
	destroyAllWindows();

}
   
#pragma once
