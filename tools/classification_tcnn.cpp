#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gflags/gflags.h>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;


DEFINE_string(model, "", "Model file(.prototxt) path.");
DEFINE_string(weights, "", "Weights file(.caffemodel) path.");
DEFINE_string(meanfile, "", "Mean file(.png) path.");
DEFINE_string(stdfile, "", "STD file(.png) path.");

/* Pair (label, confidence) representing a prediction. */
class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& std_file);

  std::vector<cv::Point> Classify(const cv::Mat& img, const cv::Rect& faceRect, int N = 5);

 private:
  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  cv::Mat std_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& std_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  if (mean_file != ""){
    mean_ = cv::imread(mean_file);
  }

  if (std_file != ""){
    std_ = cv::imread(std_file);
  }
}

/* Return the top N predictions. */
std::vector<cv::Point> Classifier::Classify(const cv::Mat& img, const cv::Rect& faceRect, int N) {
  std::vector<float> output = Predict(img);

  std::vector<cv::Point> predictions;
  for(int i = 0; i < output.size(); i+=2){
    float x = (output[i] + 0.5) * faceRect.width + faceRect.x;
    float y = (output[i+1] + 0.5) * faceRect.height + faceRect.y;
    predictions.push_back(cv::Point(int(x), int(y)));
  }

  return predictions;
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_){
    cv::resize(sample, sample_resized, input_geometry_);
  } else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat mean_float;
  if (num_channels_ == 3)
    mean_.convertTo(mean_float, CV_32FC3);
  else
    mean_.convertTo(mean_float, CV_32FC1);

  cv::Mat sample_mean;
  if (!mean_.empty()){
    cv::subtract(sample_float, mean_float, sample_mean);
  }

  cv::Mat std_float;
  if (num_channels_ == 3)
    std_.convertTo(std_float, CV_32FC3);
  else
    std_.convertTo(std_float, CV_32FC1);

  cv::Mat sample_normalized;
  if (!std_.empty()){
    cv::divide(sample_mean, std_float, sample_normalized);
  }

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);

  string model_file   = FLAGS_model;
  string trained_file = FLAGS_weights;
  string mean_file    = FLAGS_meanfile;
  string std_file    = FLAGS_stdfile;
  Classifier classifier(model_file, trained_file, mean_file, std_file);

  string file = argv[1];

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;

  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
  dlib::cv_image<dlib::bgr_pixel> cimg(img);
  std::vector<dlib::rectangle> dets = detector(cimg);
  for(int i = 0; i < dets.size(); i++){
    dlib::rectangle det = dets[i];

    cv::Rect faceRect(det.left(), det.top(), det.width(), det.height());
    cv::Mat faceOnly = img(faceRect);

    std::vector<cv::Point> predictions = classifier.Classify(faceOnly, faceRect);
    for(int i = 0; i < predictions.size(); i++){
      if (i > 0){
        std::cout << ",";
      }
      std::cout << predictions[i].x;
      std::cout << "," << predictions[i].y;
    }
    std::cout << std::endl;

//    for(int i = 0; i < predictions.size(); i++){
//      std::cout << "prediction" << i << " = " << predictions[i] << std::endl;
//      cv::circle(img, predictions[i], 2, cv::Scalar(0,200,0), -1);
//    }
//    cv::imwrite("output"+std::to_string(i)+".png", img);
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
