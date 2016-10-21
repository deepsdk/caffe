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
}

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
  Dtype* top_label = top[1]->mutable_cpu_data();
  top_label[0] = datum.label();
}

INSTANTIATE_CLASS(IdDataLayer);
REGISTER_LAYER_CLASS(IdData);

}  // namespace caffe
