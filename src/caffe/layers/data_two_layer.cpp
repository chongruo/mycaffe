#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void DataTwoLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  Datum datum2;
  CHECK(prefetch_data_.count());
  CHECK(prefetch_data2_.count());
  Dtype* top_data = prefetch_data_.mutable_cpu_data();
  Dtype* top_data2 = prefetch_data2_.mutable_cpu_data();

  const Dtype scale = this->layer_param_.data_two_param().scale();
  const int batch_size = this->layer_param_.data_two_param().batch_size(); const int crop_size = this->layer_param_.data_two_param().crop_size();
  const bool mirror = this->layer_param_.data_two_param().mirror();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = datum_channels_;
  const int height = datum_height_;
  const int width = datum_width_;
  const int size = datum_size_;
  const Dtype* mean = data_mean_.cpu_data();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    switch (this->layer_param_.data_two_param().backend()) {
    case DataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      datum.ParseFromString(iter_->value().ToString());
      CHECK(iter2_);
      CHECK(iter2_->Valid());
      datum2.ParseFromString(iter2_->value().ToString());
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datum.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    const string& data = datum.data();
    const string& data2 = datum2.data();
    if (crop_size) {
      CHECK(data.size()) << "Image cropping only support uint8 data";
      CHECK(data2.size()) << "Image cropping only support uint8 data";
      int h_off, w_off;
      // We only do random crop when we do training.
      if (phase_ == Caffe::TRAIN) {
        h_off = PrefetchRand() % (height - crop_size);
        w_off = PrefetchRand() % (width - crop_size);
      } else {
        h_off = (height - crop_size) / 2;
        w_off = (width - crop_size) / 2;
      }
      if (mirror && PrefetchRand() % 2) {
        // Copy mirrored version
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = ((item_id * channels + c) * crop_size + h)
                              * crop_size + (crop_size - 1 - w);
              int data_index = (c * height + h + h_off) * width + w + w_off;

              Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
              top_data[top_index] = (datum_element - mean[data_index]) * scale;
              Dtype datum2_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data2[data_index]));
              top_data2[top_index] = (datum2_element) * scale;
            }
          }
        }
      } else {
        // Normal copy
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = ((item_id * channels + c) * crop_size + h)
                              * crop_size + w;
              int data_index = (c * height + h + h_off) * width + w + w_off;

              Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
              top_data[top_index] = (datum_element - mean[data_index]) * scale;
              Dtype datum2_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data2[data_index]));
              top_data2[top_index] = (datum2_element) * scale;
            }
          }
        }
      }
    } else {
      // we will prefer to use data() first, and then try float_data()
      if (data.size()) {
        for (int j = 0; j < size; ++j) {
          Dtype datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[j]));
          top_data[item_id * size + j] = (datum_element - mean[j]) * scale;
        }
      } else {
        for (int j = 0; j < size; ++j) {
          top_data[item_id * size + j] =
              (datum.float_data(j) - mean[j]) * scale;
        }
      }
      if (data2.size()) {
        for (int j = 0; j < size; ++j) {
          Dtype datum2_element =
              static_cast<Dtype>(static_cast<uint8_t>(data2[j]));
          top_data2[item_id * size + j] = (datum2_element) * scale;
        }
      } else {
        for (int j = 0; j < size; ++j) {
          top_data2[item_id * size + j] =
              (datum2.float_data(j)) * scale;
        }
      }
    }

    // go to the next iter
    switch (this->layer_param_.data_two_param().backend()) {
    case DataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      iter2_->Next();
      if (!iter2_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter2_->SeekToFirst();
      }
      break;
    case DataParameter_DB_LMDB:
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
  }
}

template <typename Dtype>
DataTwoLayer<Dtype>::~DataTwoLayer<Dtype>() {
  JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_two_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void DataTwoLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);

  // Initialize DB
  switch (this->layer_param_.data_two_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options;
    options.create_if_missing = false;
    options.max_open_files = 100;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_two_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.data_two_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.data_two_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();

    leveldb::DB* db2_temp;
    leveldb::Options options2;
    options2.create_if_missing = false;
    options2.max_open_files = 100;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_two_param().source2();
    leveldb::Status status2 = leveldb::DB::Open(
        options, this->layer_param_.data_two_param().source2(), &db2_temp);
    CHECK(status2.ok()) << "Failed to open leveldb "
                       << this->layer_param_.data_two_param().source2() << std::endl
                       << status.ToString();
    db2_.reset(db2_temp);
    iter2_.reset(db2_->NewIterator(leveldb::ReadOptions()));
    iter2_->SeekToFirst();
    }
    break;
  case DataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.data_two_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.data_two_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_two_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_two_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.data_two_param().backend()) {
      case DataParameter_DB_LEVELDB:
        iter_->Next();
        iter2_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        if (!iter2_->Valid()) {
          iter2_->SeekToFirst();
        }
        break;
      case DataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  Datum datum2;
  switch (this->layer_param_.data_two_param().backend()) {
  case DataParameter_DB_LEVELDB:
    datum.ParseFromString(iter_->value().ToString());
    datum2.ParseFromString(iter2_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // image
  int crop_size = this->layer_param_.data_two_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.data_two_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
    prefetch_data_.Reshape(this->layer_param_.data_two_param().batch_size(),
        datum.channels(), crop_size, crop_size);
    (*top)[1]->Reshape(this->layer_param_.data_two_param().batch_size(),
                       datum2.channels(), crop_size, crop_size);
    prefetch_data2_.Reshape(this->layer_param_.data_two_param().batch_size(),
        datum2.channels(), crop_size, crop_size);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.data_two_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    prefetch_data_.Reshape(this->layer_param_.data_two_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
    (*top)[1]->Reshape(
        this->layer_param_.data_two_param().batch_size(), datum2.channels(),
        datum2.height(), datum2.width());
    prefetch_data2_.Reshape(this->layer_param_.data_two_param().batch_size(),
        datum2.channels(), datum2.height(), datum2.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  LOG(INFO) << "output data(top[1]) size: " << (*top)[1]->num() << ","
      << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
      << (*top)[1]->width();

  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();
  CHECK_GT(datum_height_, crop_size);
  CHECK_GT(datum_width_, crop_size);

  datum2_channels_ = datum2.channels();
  datum2_height_ = datum2.height();
  datum2_width_ = datum2.width();
  datum2_size_ = datum2.channels() * datum2.height() * datum2.width();
  CHECK_GT(datum2_height_, crop_size);
  CHECK_GT(datum2_width_, crop_size);
  // check if we want to have mean
  if (this->layer_param_.data_two_param().has_mean_file()) {
    const string& mean_file = this->layer_param_.data_two_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_.mutable_cpu_data();
  prefetch_data2_.mutable_cpu_data();
  data_mean_.cpu_data();

  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void DataTwoLayer<Dtype>::CreatePrefetchThread() {
  phase_ = Caffe::phase();
  const bool prefetch_needs_rand = (phase_ == Caffe::TRAIN) &&
      (this->layer_param_.data_two_param().mirror() ||
       this->layer_param_.data_two_param().crop_size());
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  CHECK(!StartInternalThread()) << "Pthread execution failed";
}

template <typename Dtype>
void DataTwoLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!WaitForInternalThreadToExit()) << "Pthread joining failed";
}

template <typename Dtype>
unsigned int DataTwoLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype DataTwoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(prefetch_data2_.count(), prefetch_data2_.cpu_data(),
               (*top)[1]->mutable_cpu_data());
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DataTwoLayer, Forward);
#endif

INSTANTIATE_CLASS(DataTwoLayer);

}  // namespace caffe