#ifndef CAFFE_ID_DATA_LAYER_HPP_
#define CAFFE_ID_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class IdDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit IdDataLayer(const LayerParameter& param)
    : BaseDataLayer<Dtype>(param), image_id_(0) {}
//      : BaseDataLayer<Dtype>(param), has_new_data_(false) {}

  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "IdData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

//  virtual void AddDatumVector(const vector<Datum>& datum_vector);
//#ifdef USE_OPENCV
//  virtual void AddMatVector(const vector<cv::Mat>& mat_vector,
//      const vector<int>& labels);
//#endif  // USE_OPENCV

  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
//  void Reset(Dtype* data, Dtype* label, int n);
//  void set_batch_size(int new_size);
  void set_image_id(int image_id);

//  int batch_size() { return batch_size_; }
//  int channels() { return channels_; }
//  int height() { return height_; }
//  int width() { return width_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  int batch_size_, channels_, height_, width_, size_;
//  Dtype* data_;
//  Dtype* labels_;
//  int n_;
//  size_t pos_;
//  Blob<Dtype> added_data_;
//  Blob<Dtype> added_label_;
//  bool has_new_data_;
  int image_id_;
};

}  // namespace caffe

#endif  // CAFFE_ID_DATA_LAYER_HPP_