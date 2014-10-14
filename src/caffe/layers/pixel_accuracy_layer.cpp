// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void PixelAccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Accuracy Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  (*top)[0]->Reshape(1, 2, 1, 1);
  true_label_blob_.Reshape( bottom[0]->num(), 1, 1, 1); 
  all_pred_.Reshape( bottom[0]->num(), 1, 1, 1); 
  loc_pred_.Reshape( 1, 1, bottom[0]->height(), bottom[0]->width() ); 
  eachloc_cal_.Reshape( 1, bottom[0]->channels(), 1, 1); 
}

template <typename Dtype>
Dtype PixelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  LOG(INFO)<<"I am in pixel_accuracy_layer.cpp";
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* true_label_data = true_label_blob_.mutable_cpu_data(); //(num,1,1,1)
  Dtype* all_pred_data= all_pred_.mutable_cpu_data();  // (num,1,1,1)
  Dtype* loc_pred_data= loc_pred_.mutable_cpu_data();  // (1,1,height,width))
  Dtype* each_loc_data = eachloc_cal_.mutable_cpu_data(); // (1,channel,1,1)
  int num = bottom[0]->num();
  int dim = bottom[1]->count() / bottom[1]->num();

  // get true_label, float
  int total_label = 11;
  LOG(INFO)<< "total_label: "<<total_label;
  for (int i = 0; i < true_label_blob_.count(); ++i){
     true_label_data[i] = static_cast<int>(total_label-1);
  }

  for (int i = 0; i < bottom[1]->num(); ++i){
    for (int pix = 0; pix < ( bottom[1]->count() / bottom[1]->num() ); ++pix){
       int temp_label = static_cast<int>(bottom_label[i*dim + pix]);
       if (temp_label != true_label_data[i]){
             //LOG(INFO)<<i<<" ("<<pix<<")";
   	     //LOG(INFO)<< "temp_label " <<temp_label;
             true_label_data[i] = temp_label;
 	     break;
       }
    }
  }
  /*
  for (int i =0; i< true_label_blob_.count();++i){
      LOG(INFO)<< " true label "<<i<<": "<<true_label_data[i];
  }
  */

    
  // get pred label for each loc of each num 
  for (int i=0; i < num; ++i){
       for ( int h = 0; h < bottom[0]-> height(); ++h){
          for (int w = 0; w < bottom[0]->width(); ++w){
              // copy 
              for ( int c = 0; c < bottom[0]->channels(); ++c){
                  int each_loc_offset = eachloc_cal_.offset(0,c,0,0);
                  int data_offset = bottom[0]->offset(i,c,h,w);
                  each_loc_data[each_loc_offset] = bottom_data[data_offset];
              }
              // get pred label for each pix
              Dtype maxval = -FLT_MAX;
              int max_id = 0;
              for (int c =0; c < bottom[0]->channels(); ++c){
                 int each_loc_offset = eachloc_cal_.offset(0,c,0,0);
                 //LOG(INFO)<<i<<" "<<maxval <<" , "<<each_loc_offset<<" "<<static_cast<Dtype>(each_loc_data[each_loc_offset]);
                 if (static_cast<Dtype>(each_loc_data[each_loc_offset]) > maxval){
      		      maxval = static_cast<Dtype>(each_loc_data[each_loc_offset]);
                      max_id = c;
                 } 
              }
              // copy to loc_pred
              int temp_offset = loc_pred_.offset(0,0,h,w);
              loc_pred_data[temp_offset] = max_id; 
          }
       } 
  
       if (0){
       LOG(INFO)<< " num"<<i;
       for ( int h = 0; h < bottom[0]-> height(); ++h){
          for (int w = 0; w < bottom[0]->width(); ++w){
              int temp_offset = loc_pred_.offset(0,0,h,w);
              LOG(INFO)<<"("<<h<<","<<w<<") :"<<loc_pred_data[temp_offset]; 
          }
       }
       }
              
      // get pred label for each num
      int a[total_label];
      for (int temp = 0; temp < total_label; ++temp){
             a[temp] = 0;
      }
      for ( int h = 0; h < bottom[0]-> height(); ++h){
          for (int w = 0; w < bottom[0]->width(); ++w){
              int temp_offset = loc_pred_.offset(0,0,h,w);
              a[ static_cast<int>(loc_pred_data[temp_offset]) ]++;
              //a[0]++; 
          }
      }
      
      /*
      LOG(INFO)<<" num "<<i;
      for (int m= 0; m < total_label;++m){
           LOG(INFO)<< "a["<<m<<"] "<< a[m];
      }
      */

      // get maximum 
      int max_num = 0;
      for (int temp = 0; temp < total_label; ++temp){
	  if (a[temp]>max_num){
  	      max_num = temp;
  	  } 
      }

      all_pred_data[i] = max_num;      
          
  }  

   //LOG(INFO)<< "num "<<num;
// TODO: get test loss
  for (int i = 0; i < num; ++i) {
    // Accuracy
   // LOG(INFO)<<"all_pred_data "<<i<<" :"<<all_pred_data[i]<<" "<< static_cast<int>(true_label_data[i]);
    if (all_pred_data[i] == static_cast<int>(true_label_data[i])) {
      ++accuracy;
    }
    logprob = 0;
  }
  LOG(INFO) << " ----- Accuracy: "<< accuracy;
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  (*top)[0]->mutable_cpu_data()[1] = logprob / num;
  // Accuracy layer should not be used as a loss function.
  LOG(INFO)<<" I am in pixel_accuracy_layer at last";
  return Dtype(0);

}

INSTANTIATE_CLASS(PixelAccuracyLayer);

}  // namespace caffe
