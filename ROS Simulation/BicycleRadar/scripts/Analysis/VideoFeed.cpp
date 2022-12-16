#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"
#include <cmath>

using namespace cv;
using namespace std;

const size_t width = 300;
const size_t height = 300;

std::vector<std::string> Labels;
std::unique_ptr<tflite::Interpreter> interpreter;
Mat_<double> cameraMatrix(3,3);
Mat_<double> distCoeffs(1,5);
Mat undistort_frame;

static double focal_camera;         // mtx(1,2)
#define Camera_Pose_Height  0.915    // m


static bool getFileContent(std::string fileName)
{
	std::ifstream in(fileName.c_str());
	if(!in.is_open()) return false;

	std::string str;
	while (std::getline(in, str))
	{
		if(str.size()>0) Labels.push_back(str);
	}
	in.close();
	return true;
}

double Calculate_object_distance(double Pix_Bottom)
{
    double Distance;
    Distance = (Camera_Pose_Height * focal_camera) / fabs( (Pix_Bottom - cameraMatrix(1,2)) ) ;
    return Distance;
}

void detect_from_video(Mat &src)
{
    Mat image;
    int cam_width =src.cols;
    int cam_height=src.rows;

    cv::resize(src, image, Size(width,height));
    memcpy(interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());

    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);

//        cout << "tensors size: " << interpreter->tensors_size() << "\n";
//        cout << "nodes size: " << interpreter->nodes_size() << "\n";
//        cout << "inputs: " << interpreter->inputs().size() << "\n";
//        cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";
//        cout << "outputs: " << interpreter->outputs().size() << "\n";

    interpreter->Invoke();

    const float* detection_locations = interpreter->tensor(interpreter->outputs()[0])->data.f;
    const float* detection_classes=interpreter->tensor(interpreter->outputs()[1])->data.f;
    const float* detection_scores = interpreter->tensor(interpreter->outputs()[2])->data.f;
    const int    num_detections = *interpreter->tensor(interpreter->outputs()[3])->data.f;

    //cout << "number of detections: " << num_detections << "\n";

    const float confidence_threshold = 0.48;
    for(int i = 0; i < num_detections; i++){
        if(detection_scores[i] > confidence_threshold){
            int  det_index = (int)detection_classes[i]+1;
            float y1=detection_locations[4*i  ]*cam_height;
            float x1=detection_locations[4*i+1]*cam_width;
            float y2=detection_locations[4*i+2]*cam_height;
            float x2=detection_locations[4*i+3]*cam_width;

            Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
            rectangle(src,rec, Scalar(0, 0, 255), 1, 8, 0);

            //cout<< Calculate_object_distance(y2,y1) << endl;
            //cout<<"mid_y: "<<cameraMatrix(1,2)<<" Y2: "<<y2<<" Y1: "<<y1<<endl;
            line(src , Point(317 ,228) , Point(x1+(x2-x1)/2.0,y2) , Scalar(0,0,255),1);
            //putText(src, format("%s", Labels[det_index].c_str()), Point(x1, y1-5) ,FONT_HERSHEY_SIMPLEX,0.5, Scalar(0, 255, 0), 1, 8, 0);

            putText(src, format("%0.2f m",Calculate_object_distance(y2) ), Point(x1+(x2-x1)/2.0,y2+10) ,FONT_HERSHEY_SIMPLEX,0.5, Scalar(0, 255, 0), 1, 8, 0);
        }
    }
}

int main(int argc,char ** argv)
{
    float f;
    float FPS[16];
    int i;
    int Fcnt=0;
    Mat frame;

    cameraMatrix(0,0) = 382.25802374;
    cameraMatrix(0,1) = 0;
    cameraMatrix(0,2) = 317.55530562;
    cameraMatrix(1,0) = 0;
    cameraMatrix(1,1) = 379.57912433;
    cameraMatrix(1,2) = 228.13804584;
    cameraMatrix(2,0) = 0;
    cameraMatrix(2,1) = 0;
    cameraMatrix(2,2) = 1;

    focal_camera = ( cameraMatrix(1,1) + cameraMatrix(0,0) ) / 2.0;

    //distCoeffs(0 , 0) = -0.32621363 ;
    //distCoeffs(0 , 1) =  0.14117533 ;
    //distCoeffs(0 , 2) = -0.00088709 ;
    //distCoeffs(0 , 3) =  0.00128622 ;
    //distCoeffs(0 , 4) = -0.03548023 ;

    distCoeffs(0 , 0) = 0 ;
    distCoeffs(0 , 1) = 0 ;
    distCoeffs(0 , 2) = 0 ;
    distCoeffs(0 , 3) = 0 ;
    distCoeffs(0 , 4) = 0 ;

    chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("detect.tflite");

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    interpreter->AllocateTensors();

	bool result = getFileContent("COCO_labels.txt");
	if(!result)
	{
        cout << "loading labels failed";
        exit(-1);
	}

    VideoCapture cap("traffic.mp4");
    //VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_HEIGHT,480);
    cap.set(CAP_PROP_FRAME_WIDTH,640);
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }

    cout << "Start grabbing, press ESC on Live window to terminate" << endl;

    while(1){
        cap >> frame;
        if (frame.empty()) {
            cerr << "End of movie" << endl;
            break;
        }

        // Calibrate && undistort image.
        undistort(frame, undistort_frame, cameraMatrix, distCoeffs);

        // detect image by tflite model
        detect_from_video(undistort_frame);

        Tend = chrono::steady_clock::now();
        //calculate frame rate
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();

        Tbegin = chrono::steady_clock::now();

        FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++)
        {
            f+=FPS[i];
        }

        putText(undistort_frame, format("FPS %0.2f",f/16),Point(10,20),FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 0, 255));

        //show output
        imshow("RPi 4 - 2.0 GHz - 2 Mb RAM", undistort_frame);

        char esc = waitKey(1);
        if(esc == 27) break;
    }

    cout << "Closing the camera" << endl;

    // When everything done, release the video capture and write object
    cap.release();

    destroyAllWindows();
    cout << "Bye!" << endl;

    return 0;
}
