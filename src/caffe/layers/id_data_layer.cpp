#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <vector>

#include "caffe/layers/id_data_layer.hpp"

#include <lmdb.h>
#include "boost/scoped_ptr.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"


namespace caffe {


template <typename Dtype>
void IdDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  channels_ = this->layer_param_.memory_data_param().channels();
  height_ = this->layer_param_.memory_data_param().height();
  width_ = this->layer_param_.memory_data_param().width();
  size_ = channels_ * height_ * width_;
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  vector<int> label_shape(1, batch_size_);
  top[0]->Reshape(batch_size_, channels_, height_, width_);
  top[1]->Reshape(label_shape);
//  added_data_.Reshape(batch_size_, channels_, height_, width_);
//  added_label_.Reshape(label_shape);
//  data_ = NULL;
//  labels_ = NULL;
//  added_data_.cpu_data();
//  added_label_.cpu_data();
}

//template <typename Dtype>
//void IdDataLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector) {
//  CHECK(!has_new_data_) <<
//      "Can't add data until current data has been consumed.";
//  size_t num = datum_vector.size();
//  CHECK_GT(num, 0) << "There is no datum to add.";
//  CHECK_EQ(num % batch_size_, 0) <<
//      "The added data must be a multiple of the batch size.";
//  added_data_.Reshape(num, channels_, height_, width_);
//  added_label_.Reshape(num, 1, 1, 1);
//  // Apply data transformations (mirror, scale, crop...)
//  this->data_transformer_->Transform(datum_vector, &added_data_);
//  // Copy Labels
//  Dtype* top_label = added_label_.mutable_cpu_data();
//  for (int item_id = 0; item_id < num; ++item_id) {
//    top_label[item_id] = datum_vector[item_id].label();
//  }
//  // num_images == batch_size_
//  Dtype* top_data = added_data_.mutable_cpu_data();
//  Reset(top_data, top_label, num);
//  has_new_data_ = true;
//}

//#ifdef USE_OPENCV
//template <typename Dtype>
//void IdDataLayer<Dtype>::AddMatVector(const vector<cv::Mat>& mat_vector,
//    const vector<int>& labels) {
//  size_t num = mat_vector.size();
//  CHECK(!has_new_data_) <<
//      "Can't add mat until current data has been consumed.";
//  CHECK_GT(num, 0) << "There is no mat to add";
//  CHECK_EQ(num % batch_size_, 0) <<
//      "The added data must be a multiple of the batch size.";
//  added_data_.Reshape(num, channels_, height_, width_);
//  added_label_.Reshape(num, 1, 1, 1);
//  // Apply data transformations (mirror, scale, crop...)
//  this->data_transformer_->Transform(mat_vector, &added_data_);
//  // Copy Labels
//  Dtype* top_label = added_label_.mutable_cpu_data();
//  for (int item_id = 0; item_id < num; ++item_id) {
//    top_label[item_id] = labels[item_id];
//  }
//  // num_images == batch_size_
//  Dtype* top_data = added_data_.mutable_cpu_data();
//  Reset(top_data, top_label, num);
//  has_new_data_ = true;
//}
//#endif  // USE_OPENCV

//template <typename Dtype>
//void IdDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
//  CHECK(data);
//  CHECK(labels);
//  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
//  // Warn with transformation parameters since a memory array is meant to
//  // be generic and no transformations are done with Reset().
//  if (this->layer_param_.has_transform_param()) {
//    LOG(WARNING) << this->type() << " does not transform array data on Reset()";
//  }
//  data_ = data;
//  labels_ = labels;
//  n_ = n;
//  pos_ = 0;
//}

//template <typename Dtype>
//void IdDataLayer<Dtype>::set_batch_size(int new_size) {
//  CHECK(!has_new_data_) <<
//      "Can't change batch_size until current data has been consumed.";
//  batch_size_ = new_size;
//  added_data_.Reshape(batch_size_, channels_, height_, width_);
//  added_label_.Reshape(batch_size_, 1, 1, 1);
//}

template <typename Dtype>
void IdDataLayer<Dtype>::set_image_id(int image_id) {
  image_id_ = image_id;
}

template <typename Dtype>
void IdDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  boost::scoped_ptr<db::DB> db(db::GetDB(this->layer_param_.data_param().backend()));
  db->Open(this->layer_param_.data_param().source(), db::READ);
  boost::scoped_ptr<db::Transaction> txn(db->NewTransaction());

  string value;
  Datum datum;
  string key_str = format_int(image_id_, 8);
  txn->Get(key_str, value);
  datum.ParseFromString(value);

  LOG(INFO) << "dbg>----------------------------------------------";
  LOG(INFO) << "dbg>channels="<< datum.channels();
  LOG(INFO) << "dbg>channels="<< datum.channels();
  LOG(INFO) << "dbg>height="<< datum.height();
  LOG(INFO) << "dbg>width="<< datum.width();
  LOG(INFO) << "dbg>encoded="<< datum.encoded();
  LOG(INFO) << "dbg>label="<< datum.label();
  LOG(INFO) << "dbg>image_id_="<< image_id_;


  top[0]->Reshape(1, datum.channels(), datum.height(), datum.width());
  top[1]->Reshape(1, 1, 1, 1);
  this->data_transformer_->Transform(datum, top[0]);
  Dtype label = static_cast<Dtype>(datum.label());
  top[1]->set_cpu_data(&label);

//  caffe_copy(top[0]->count(), datum.float_data(), top[0]->mutable_cpu_data());

//  caffe_set(top[1]->count(), datum.label(), top[1]->mutable_cpu_data());
//  top[0]->set_cpu_data(datum.data());
//  top[1]->set_cpu_data(datum.label());
  

//  CHECK(data_) << "IdDataLayer needs to be initialized by calling Reset";
//  top[0]->Reshape(batch_size_, channels_, height_, width_);
//  top[1]->Reshape(batch_size_, 1, 1, 1);
//  top[0]->set_cpu_data(data_ + pos_ * size_);
//  top[1]->set_cpu_data(labels_ + pos_);
//  pos_ = (pos_ + batch_size_) % n_;
//  if (pos_ == 0)
//    has_new_data_ = false;
}

INSTANTIATE_CLASS(IdDataLayer);
REGISTER_LAYER_CLASS(IdData);

}  // namespace caffe
