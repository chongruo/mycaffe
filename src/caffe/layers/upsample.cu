#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void UpSampleForward1(const int nthreads,const Dtype* bottom_data,
	const int num, const int channels, const int height,const int width,
	Dtype* top_data) {
     CUDA_KERNEL_LOOP(index,nthreads){
	int uw = index % (2*width);	
	int uh = index % (2*height);	
	int c = (index / (2*width) / (2*height)) % channels;
	int n = index / (2*width) / (2*height) / channels;

	int current_top_i = index - ((n * channels + c ) * (2*height) * (2*width));
	int current_down_starti = (n * channels + c) * height * width;
	if ((uh%2==0 && uw%2==0) || (uh%2==1 && uw%2==1)){
  		top_data[index] = bottom_data[current_down_starti + current_top_i/2];	
  	} 
     }
}

template <typename Dtype>
__global__ void UpSampleForward2(const int nthreads,const int num, 
  	const int channels, const int height,const int width,Dtype* top_data) {
     CUDA_KERNEL_LOOP(index,nthreads){
	int uw = index % (2*width);	
	int uh = index % (2*height);	
	int c = (index / (2*width) / (2*height)) % channels;
	int n = index / (2*width) / (2*height) / channels;

	if ((uh%2==0&& uw%2==1) || (uh%2==1 && uw%2==0)){
		Dtype temp = 0;
		Dtype addcount = 0;
		if (uh>0) { temp += top_data[index-(2*width)]; addcount += 1; }
		if (uh<(2*height-1)) { temp += top_data[index+(2*width)]; addcount += 1; }	
		if (uw>0) { temp += top_data[index-1]; addcount += 1; }
		if (uw<(2*width-1)) { temp += top_data[index+1];addcount += 1; }
  		top_data[index] = temp/addcount;
  	} 
     }
}

template <typename Dtype>
Dtype UpSampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
   UpSampleForward1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	count, bottom_data, bottom[0]->num(), channels_,
	height_, width_,top_data);

   UpSampleForward2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	count, bottom[0]->num(),channels_, height_, width_,top_data);
   
  CUDA_POST_KERNEL_CHECK;
  return Dtype(0.);
}


template <typename Dtype>
__global__ void UpSampleBackward1(const int nthreads, const Dtype* top_diff,
	const int num, const int channels, const int height, const int width,
	Dtype* bottom_diff){
     CUDA_KERNEL_LOOP(index,nthreads){
	int uw = index % (2*width);	
	int uh = index % (2*height);	
	int c = (index / (2*width) / (2*height)) % channels;
	int n = index / (2*width) / (2*height) / channels;
	
	int current_top_i = index - ((n * channels + c ) * (2*height) * (2*width));
	int current_down_starti = (n * channels + c) * height * width;
	if ((uh%2==0 && uw%2==0) || (uh%2==1 && uw%2==1)){
		bottom_diff[current_down_starti+current_top_i/2] += 0.5*top_diff[index];
  	} 
     }
}

template <typename Dtype>
__global__ void UpSampleBackward2(const int nthreads, const Dtype* top_diff,
	const int num, const int channels, const int height, const int width,
	Dtype* bottom_diff){
     CUDA_KERNEL_LOOP(index,nthreads){
	int uw = index % (2*width);	
	int uh = index % (2*height);	
	int c = (index / (2*width) / (2*height)) % channels;
	int n = index / (2*width) / (2*height) / channels;
	
	int current_top_i = index - ((n * channels + c ) * (2*height) * (2*width));
	int current_down_starti = (n * channels + c) * height * width;
	if ((uh%2==0&& uw%2==1) || (uh%2==1 && uw%2==0)){
		Dtype temp = 0;
		Dtype addcount = 0;
		if (uh>0) { addcount += 1; }
		if (uh<(2*height-1)) { addcount += 1; }	
		if (uw>0) { addcount += 1; }
		if (uw<(2*width-1)) { addcount += 1; }

		if (uh>0) { 
			bottom_diff[current_down_starti+(current_top_i-2*width)/2] += 0.5*temp/addcount;
		}
		if (uh<(2*height-1)) { 
			bottom_diff[current_down_starti+(current_top_i+2*width)/2] += 0.5*temp/addcount;
		}	
		if (uw>0) { 
			bottom_diff[current_down_starti+(current_top_i-1)/2] += 0.5*temp/addcount;
		}
		if (uw<(2*width-1)) { 
			bottom_diff[current_down_starti+(current_top_i+1)/2] += 0.5*temp/addcount;
		}
  	} 
     }
}

template <typename Dtype>
void UpSampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const int count = top[0]->count();
  const int bottom_count = (*bottom)[0]->count();
  caffe_gpu_set(bottom_count, Dtype(0.), bottom_diff);

  UpSampleBackward1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_, height_, width_, bottom_diff);
  UpSampleBackward2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_, height_, width_, bottom_diff);

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_CLASS(UpSampleLayer);


}  // namespace caffe
