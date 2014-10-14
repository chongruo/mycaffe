// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void PixelSoftmaxWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "PixelSoftmaxLoss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "PixelSoftmaxLoss Layer takes no blob as output.";
  pixel_softmax_bottom_vec_.clear();
  pixel_softmax_bottom_vec_.push_back(bottom[0]);
  pixel_softmax_top_vec_.push_back(&prob_);
  pixel_softmax_layer_->SetUp(pixel_softmax_bottom_vec_, &pixel_softmax_top_vec_);
}

template <typename Dtype>
Dtype PixelSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the pixel_softmax prob values.
  //LOG(INFO)<<"I am in pixel_softmaxLoss_layer.cpp";
  pixel_softmax_bottom_vec_[0] = bottom[0];
  pixel_softmax_layer_->Forward(pixel_softmax_bottom_vec_, &pixel_softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  int label_loc = 0; 
  int num = prob_.num();
  Dtype loss = 0;

  for (int loc_x = 0; loc_x < bottom[0]->height(); ++loc_x){
    for (int loc_y = 0; loc_y < bottom[0]->width(); ++loc_y){
	  for (int i = 0; i < num; ++i) {
            label_loc = static_cast<float>(label[bottom[1]->offset(i,0,loc_x,loc_y)]);
            
            if(0){
     	    for (int c = 0; c< bottom[0]->channels();++c){
                LOG(INFO)<< "prob_data: " << i << " " << loc_x << " " << loc_y << " " \
		 << c << " : "<< static_cast<float>(prob_data[prob_.offset(i,c,loc_x,loc_y)]);
  	    }
	    loss += -log(max(prob_data[prob_.offset(i,label_loc,loc_x,loc_y)], Dtype(FLT_MIN)));
	    }
          }
    }
   //LOG(INFO)<< "pxiel_softmaxLoss_layer.cpp --lox_x "<< loc_x;
  }
   
  
  if(0){
    for (int c = 0; c< bottom[0]->channels();++c){
	LOG(INFO)<< "prob_data: " << 11 << " " << 65<< " " << 67 << " " \
	 << c << " : "<< static_cast<float>(prob_data[prob_.offset(0,c,65,67)]);
    }
  }

   //LOG(INFO)<< "pxiel_softmaxLoss_layer.cpp at last";
  //return loss /  num; 
  return loss / ( num * bottom[0]->height() * bottom[0]->width() );
}

template <typename Dtype>
void PixelSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  LOG(INFO)<<"I am in pixel_softmaxLoss_layer.cpp Backward_cpu";
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
  const Dtype* label = (*bottom)[1]->cpu_data();

  int label_loc = 0;
  int num = prob_.num();

  for (int i = 0; i < num; ++i) {
     for (int loc_x = 0; loc_x < prob_.height(); ++loc_x){
        for (int loc_y = 0; loc_y < prob_.width(); ++loc_y){
          label_loc = static_cast<int>(label[(*bottom)[1]->offset(i,0,loc_x,loc_y)]);
   	  //LOG(INFO)<< "offset label_loc "<< "("<<loc_x<<","<<loc_y<<")"<<label_loc;
          bottom_diff[ prob_.offset(i,label_loc,loc_x,loc_y) ] -= 1;
          /*
          for (int c = 0; c<prob_.channels();++c){
	      LOG(INFO)<< " I am in swithloss backward_cpu";
              LOG(INFO)<<i<<" "<<loc_x<<","<<loc_y<<" "<< c<<" "<<bottom_diff[ prob_.offset(i,c,loc_x,loc_y) ];
	 }
         */
       }
    }
  //LOG(INFO)<<"I am in pixel_softmaxLoss_layer.cpp --loc_x "<<loc_x;
    
  }    
  
  // Scale down gradient
  caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
  LOG(INFO)<< "in 113";
  
  //LOG(INFO)<<"I am in pixel_softmaxLoss_layer.cpp backwwar   at last";
}


INSTANTIATE_CLASS(PixelSoftmaxWithLossLayer);


}  // namespace caffe
