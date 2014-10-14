// Copyright 2014 BVLC and contributors.
//
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void PixelSoftmaxLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "PixelSoftmax Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "PixelSoftmax Layer takes a single blob as output.";
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  // sum_multiplier_
  sum_multiplier_.Reshape(1, bottom[0]->channels(),1,1);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  for (int i = 0; i < sum_multiplier_.count(); ++i) {
    multiplier_data[i] = 1.;
  }
  // temp_data_
  temp_top_blob_.Reshape(bottom[0]->num(), bottom[0]->channels(),1,1);
  temp_bottom_blob_.Reshape(bottom[0]->num(), bottom[0]->channels(),1,1);
  scale_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
Dtype PixelSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  //LOG(INFO) << "I am in pixel_softmax_layer.cpp Forward_cpu";
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  Dtype* temp_top_data = temp_top_blob_.mutable_cpu_data();
  Dtype* temp_bottom_data = temp_bottom_blob_.mutable_cpu_data();

  int temp_num = 0;
  int temp_channel = 0;
  int num = bottom[0]->num();
  int dim = bottom[0]->channels();
  memcpy(top_data, bottom_data, sizeof(Dtype) * bottom[0]->count());
  // we need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.

  // calculate, obtaining temp_blob:(num,channels,1,1)
  for (int loc_x = 0; loc_x < bottom[0]->height(); ++loc_x){
    for (int loc_y = 0; loc_y < bottom[0]->width(); ++loc_y){
          // copy channels for each loc to temp_bottom_blob_
	  for (int i = 0; i < temp_bottom_blob_.count(); ++i){
             temp_num = ( i / bottom[0]->channels() );
             temp_channel = ( i % bottom[0]->channels() );
             temp_bottom_data[i] = bottom_data[ bottom[0]->offset(temp_num,temp_channel,loc_x,loc_y) ];
          }          

          // calculate
		  for (int i = 0; i < num; ++i) {
		    scale_data[i] = temp_bottom_data[i*dim];
		    for (int j = 0; j < dim; ++j) {
		      scale_data[i] = max(scale_data[i], temp_bottom_data[i * dim + j]);
		    }
		  }
		  // subtraction
		  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
		    scale_data, sum_multiplier_.cpu_data(), 1., temp_top_data);
		  // Perform exponentiation
		  caffe_exp<Dtype>(num * dim, temp_top_data, temp_top_data);
		  // sum after exp
		  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., temp_top_data,
		      sum_multiplier_.cpu_data(), 0., scale_data);
		  // Do division
		  for (int i = 0; i < num; ++i) {
		    caffe_scal<Dtype>(dim, Dtype(1.) / scale_data[i], temp_top_data + i * dim);
		  }

         //  copy temp_top_data to top_data
	 for (int i = 0; i < temp_top_blob_.count(); ++i){
             temp_num = ( i / bottom[0]->channels() );
             temp_channel = ( i % bottom[0]->channels() );
             top_data[ (*top)[0]->offset(temp_num,temp_channel,loc_x,loc_y) ] = temp_top_data[i];
         }          

         if(0){
  	   for (int i = 0; i< num; ++i){
	     for (int c =0; c< dim; ++c){
               LOG(INFO)<<i<<" "<< loc_x<<" "<<loc_y<<" "<<c<<": "<<\
			top_data[ (*top)[0]->offset(i,c,loc_x,loc_y) ]; 
   	      } 
            }
	 }
     }
  //LOG(INFO) << "pixel_softmax_layer.cpp Forward_cpu --loc_x " << loc_x;
  }
  //LOG(INFO) << "pixel_softmax_layer.cpp Forward_cpu  at last";
  return Dtype(0);
}

template <typename Dtype>
void PixelSoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  //LOG(INFO) << "I am in pixel_softmax_layer.cpp Backward_cpu";
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  Dtype* temp_top_diff = temp_top_blob_.mutable_cpu_diff();
  Dtype* temp_top_data = temp_top_blob_.mutable_cpu_data();
  Dtype* temp_bottom_diff = temp_bottom_blob_.mutable_cpu_diff();

  int temp_num = 0;
  int temp_channel = 0;
  int num = top[0]->num();
  int dim = top[0]->channels();
  memcpy(bottom_diff, top_diff, sizeof(Dtype) * top[0]->count());

  for (int loc_x = 0; loc_x < top[0]->height(); ++loc_x){
    for (int loc_y = 0; loc_y < top[0]->width(); ++loc_y){
          // copy channels for each loc to temp_top_blob_
	  for (int i = 0; i < temp_top_blob_.count(); ++i){
             temp_num = ( i / top[0]->channels() );
             temp_channel = ( i % top[0]->channels() );
             temp_top_diff[i] = top_diff[ top[0]->offset(temp_num,temp_channel,loc_x,loc_y) ];
             temp_top_data[i] = top_data[ top[0]->offset(temp_num,temp_channel,loc_x,loc_y) ];
          }          

		  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
		  for (int i = 0; i < num; ++i) {
		    scale_data[i] = caffe_cpu_dot<Dtype>(dim, temp_top_diff + i * dim,
			temp_top_data + i * dim);
		  }
		  // subtraction
		  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
		      scale_data, sum_multiplier_.cpu_data(), 1., temp_bottom_diff);
		  // elementwise multiplication
		  caffe_mul<Dtype>(top[0]->count(), temp_bottom_diff, temp_top_data, temp_bottom_diff);

	 //  copy temp_top_data to top_data
	 for (int i = 0; i < temp_top_blob_.count(); ++i){
             temp_num = ( i / top[0]->channels() );
             temp_channel = ( i % top[0]->channels() );
             bottom_diff[ (*bottom)[0]->offset(temp_num,temp_channel,loc_x,loc_y) ] = temp_bottom_diff[i];
         }          
     }
  }

  LOG(INFO) << "I am in pixel_softmax_layer.cpp Backward_cpu at last";
}


INSTANTIATE_CLASS(PixelSoftmaxLayer);


}  // namespace caffe
