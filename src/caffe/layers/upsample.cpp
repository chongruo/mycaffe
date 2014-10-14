#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void UpSampleLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Set the max number of top blobs before calling base Layer::SetUp.
  // If doing MAX pooling, we can optionally output an extra top Blob
  // for the mask.  Otherwise, we only have one top Blob.
  Layer<Dtype>::SetUp(bottom, top);
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  (*top)[0]->Reshape(bottom[0]->num(), channels_, 2*height_, 2*width_);

}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
Dtype UpSampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  return Dtype(0.);
}

template <typename Dtype>
void UpSampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  }
}


#ifdef CPU_ONLY
STUB_GPU(UpSampleLayer);
#endif

INSTANTIATE_CLASS(UpSampleLayer);


}  // namespace caffe
