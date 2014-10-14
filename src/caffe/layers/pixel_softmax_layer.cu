// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
__global__ void kernel_copyb(const int lnum, 
      const int channels, const int height, const int width,
      const int loc_x, const int loc_y, const Dtype* source, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, lnum) {
      int temp_num = ( index / channels);
      int temp_channel = ( index % channels );
      int offset = ((temp_num * channels + temp_channel) * height + loc_x ) * width + loc_y;
      dest[index] = source[ offset ]; 
  }
}

template <typename Dtype>
__global__ void kernel_bcopyb(const int lnum, 
      const int channels, const int height, const int width,
      const int loc_x, const int loc_y, const Dtype* dsource, Dtype* ddest,
      const Dtype* fsource, Dtype* fdest) {
  CUDA_KERNEL_LOOP(index, lnum) {
      int temp_num = ( index / channels);
      int temp_channel = ( index % channels );
      int offset = ((temp_num * channels + temp_channel) * height + loc_x ) * width + loc_y;
      fdest[index] = fsource[ offset ]; 
      ddest[index] = dsource[ offset ]; 
      
  }
}

template <typename Dtype>
__global__ void kernel_copyt(const int lnum, 
      const int channels, const int height, const int width,
      const int loc_x, const int loc_y, Dtype* dest, const Dtype* source) {
  CUDA_KERNEL_LOOP(index, lnum) {
      int temp_num = ( index / channels);
      int temp_channel = ( index % channels );
      int offset = ((temp_num * channels + temp_channel) * height + loc_x ) * width + loc_y;
      dest[ offset ] = source[index];
  }
}

template <typename Dtype>
__global__ void kernel_bcopyt(const int lnum, 
      const int channels, const int height, const int width,
      const int loc_x, const int loc_y, Dtype* fdest, const Dtype* fsource) {
  CUDA_KERNEL_LOOP(index, lnum) {
      int temp_num = ( index / channels);
      int temp_channel = ( index % channels );
      int offset = ((temp_num * channels + temp_channel) * height + loc_x ) * width + loc_y;
      fdest[ offset ] = fsource[index];
  }
}

template <typename Dtype>
__global__ void kernel_get_max(const int num, const int dim,
    const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num) {
    Dtype maxval = -FLT_MAX;
    for (int i = 0; i < dim; ++i) {
      maxval = max(data[index * dim + i], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_softmax_div(const int num, const int dim,
    const Dtype* scale, Dtype* data) {
  CUDA_KERNEL_LOOP(index, num * dim) {
    int n = index / dim;
    data[index] /= scale[n];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int num, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
Dtype PixelSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  LOG(INFO)<<"I am in pixel_softmax_layer.cu";
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data(); // (num,1,1,1)
  Dtype* temp_top_data = temp_top_blob_.mutable_gpu_data(); // (num,channels,1,1)
  Dtype* temp_bottom_data = temp_bottom_blob_.mutable_gpu_data(); // (num,channels,1,1)
   						// multiplier_data (1,channels,1,1)

  int num = bottom[0]->num();
  int dim = bottom[0]->channels(); 
  CUDA_CHECK(cudaMemcpy(top_data, bottom_data,
      sizeof(Dtype) * bottom[0]->count(), cudaMemcpyDeviceToDevice));

   //LOG(INFO)<<" I am here 111";
   for (int loc_x = 0; loc_x < bottom[0]->height(); ++loc_x){
          //LOG(INFO)<<" I am here 113 "<<",loc_x "<<loc_x;
     for (int loc_y = 0; loc_y < bottom[0]->width(); ++loc_y){
           //int loc_y = 0;

          //copy channels for each loc to temp_bottom_blob         
   	  //LOG(INFO)<< " i am a speaker "<< temp_bottom_blob_.count();
          kernel_copyb<Dtype><<<CAFFE_GET_BLOCKS(temp_bottom_blob_.count()),CAFFE_CUDA_NUM_THREADS>>>(
             temp_bottom_blob_.count(), bottom[0]->channels(), bottom[0]->height(),
             bottom[0]->width(), loc_x, loc_y, bottom_data, temp_bottom_data );

	  // we need to subtract the max to avoid numerical issues, compute the exp,
	  // and then normalize.
	  // Compute max
	  // NOLINT_NEXT_LINE(whitespace/operators)
          kernel_get_max<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
	      num, dim, temp_bottom_data, scale_data);
	  // subtraction
	  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
	      scale_data, sum_multiplier_.gpu_data(), 1., temp_top_data);
	  // Perform exponentiation
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(num * dim), CAFFE_CUDA_NUM_THREADS>>>(
	      num * dim, temp_top_data, temp_top_data);
	  // sum after exp
	  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., temp_top_data,
	      sum_multiplier_.gpu_data(), 0., scale_data);
	  // Do division
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  kernel_softmax_div<Dtype><<<CAFFE_GET_BLOCKS(num * dim),
				      CAFFE_CUDA_NUM_THREADS>>>(
	      num, dim, scale_data, temp_top_data);

   	  //LOG(INFO)<< " i am a speaker "<< temp_top_blob_.count();
          // copy temp_blob to top
          kernel_copyt<Dtype><<<CAFFE_GET_BLOCKS(temp_top_blob_.count()),CAFFE_CUDA_NUM_THREADS>>>(
             temp_top_blob_.count(), bottom[0]->channels(), bottom[0]->height(),
             bottom[0]->width(), loc_x, loc_y, top_data, temp_top_data );
 
        //  for ( int i = 0; i< num; ++i){
         //   for ( int c = 0;c< dim; ++c){
          //LOG(INFO)<< i<<" "<< loc_x <<" " << loc_y<<" " << ": "<< top_data[(*top)[0]->offset(0,0,loc_x,loc_y)];

          //int temp = (*top)[0]->offset(0,0,loc_x,loc_y);
          
          //LOG(INFO)<<  "in 158";
          //LOG(INFO)<< top_data[4]<< " fda ";
          //LOG(INFO)<<  "in 159";
            //}
          //}
       
      }
   }

  return Dtype(0);
}

// TODO(Yangqing): implement the GPU version of softmax.
template <typename Dtype>
void PixelSoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  Dtype* temp_top_diff = temp_top_blob_.mutable_gpu_diff();
  Dtype* temp_top_data = temp_top_blob_.mutable_gpu_data();
  Dtype* temp_bottom_diff = temp_bottom_blob_.mutable_gpu_diff();
  

  int num = top[0]->num();
  int dim = top[0]->channels();
  CUDA_CHECK(cudaMemcpy(bottom_diff, top_diff,
      sizeof(Dtype) * top[0]->count(), cudaMemcpyDeviceToDevice));
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
  // cuda dot returns the result to cpu, so we temporarily change the pointer
  // mode
  CUBLAS_CHECK(cublasSetPointerMode(Caffe::cublas_handle(),
     CUBLAS_POINTER_MODE_DEVICE));
  Dtype* scale_data = scale_.mutable_gpu_data();

  for (int loc_x = 0; loc_x < top[0]->height(); ++loc_x){
    for (int loc_y = 0; loc_y < top[0]->width(); ++loc_y){

          //copy channels for each loc to temp_bottom_blob         
          kernel_bcopyb<Dtype><<<CAFFE_GET_BLOCKS(temp_top_blob_.count()),CAFFE_CUDA_NUM_THREADS>>>(
             temp_top_blob_.count(), top[0]->channels(), top[0]->height(),
             top[0]->width(), loc_x, loc_y, top_data, temp_top_data,
             top_diff,temp_top_diff);

        if(0){
		 for (int i=0;i<temp_top_blob_.count();++i){
		     int temp_num = ( i / top[0]->channels() );
		     int temp_channel = ( i % top[0]-> channels());
		 }
        }

	  for (int i = 0; i < num; ++i) {
	    caffe_gpu_dot<Dtype>(dim, temp_top_diff + i * dim,
		temp_top_data + i * dim, scale_data + i);
	  }
	  CUBLAS_CHECK(cublasSetPointerMode(Caffe::cublas_handle(),
	      CUBLAS_POINTER_MODE_HOST));
	  // subtraction
	  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
	      scale_.gpu_data(), sum_multiplier_.gpu_data(), 1., temp_bottom_diff);
	  // elementwise multiplication
	  caffe_gpu_mul<Dtype>(top[0]->count(), temp_bottom_diff, temp_top_data, temp_bottom_diff);

          // copy temp_blob to top
          kernel_bcopyt<Dtype><<<CAFFE_GET_BLOCKS(temp_top_blob_.count()),CAFFE_CUDA_NUM_THREADS>>>(
             temp_top_blob_.count(), top[0]->channels(), top[0]->height(),
             top[0]->width(), loc_x, loc_y, bottom_diff, temp_bottom_diff );
      }
   }
}

INSTANTIATE_CLASS(PixelSoftmaxLayer);


}  // namespace caffe
