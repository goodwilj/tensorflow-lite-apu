/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/examples/cnn_inference/cnn_inference.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)
//#include <jpeglib.h>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

#include "tensorflow/lite/examples/cnn_inference/bitmap_helpers.h"



#define LOG(x) std::cerr

namespace tflite {
namespace cnn_inference {
/*
void write_JPEG_file(uint8_t* image_buffer, const char *filename, int image_width,int image_height,int quality){

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  FILE* outfile;
  int row_stride;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  if((outfile=fopen(filename,"wb"))==NULL){
    fprintf(stderr,"can't open %s\n", filename);
    exit(1);
  }

  jpeg_stdio_dest(&cinfo,outfile);

  cinfo.image_width=image_width;
  cinfo.image_height=image_height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo,quality,TRUE);
  jpeg_start_compress(&cinfo,TRUE);

  row_stride = image_width*3;
  JSAMPROW row_pointer;
  while(cinfo.next_scanline < cinfo.image_height){
    row_pointer = (JSAMPROW) &image_buffer[cinfo.next_scanline * row_stride];
    (void) jpeg_write_scanlines(&cinfo,&row_pointer,1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  jpeg_destroy_compress(&cinfo);
}
*/

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }


void RunInference(Settings* s) {
  if (!s->model_name.c_str()) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
    exit(-1);
  }
  LOG(INFO) << "Loaded model " << s->model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  interpreter->UseNNAPI(s->accel);

  if (s->verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }

  if (s->number_of_threads != -1) {
    interpreter->SetNumThreads(s->number_of_threads);
  }

  int image_width = 489;
  int image_height = 410;
  int image_channels = 3;
  int input_num_pixels = image_height*image_width*image_channels;
  std::vector<uint8_t> in = read_bmp(s->input_ppm_name, &image_width,
                                     &image_height, &image_channels, s);


  uint8_t* input_img = in.data();
  
  int input = interpreter->inputs()[0];
  if (s->verbose) LOG(INFO) << "input: " << input << "\n";

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (s->verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  if (s->verbose) PrintInterpreterState(interpreter.get());

  // get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_batchsize = dims->data[0];
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  LOG(INFO) << "wanted batch " << wanted_batchsize << "\n";
  LOG(INFO) << "wanted height " << wanted_height << "\n";
  LOG(INFO) << "wanted width " << wanted_width << "\n";
  LOG(INFO) << "wanted channels " << wanted_channels << "\n";

  for(int i=0; i<image_height*image_width*image_channels;i++)
    interpreter->typed_tensor<float>(input)[i]=input_img[i]/255.;

  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);
  if (interpreter->Invoke() != kTfLiteOk) {
    LOG(FATAL) << "Failed to invoke tflite!\n";
  }


  int output = interpreter->outputs()[0];
  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
  int output_batchsize = output_dims->data[0];
  int output_height = output_dims->data[1];
  int output_width = output_dims->data[2];
  int output_channels = output_dims->data[3];

  int output_num_pixels = output_height*output_width*output_channels;
  uint8_t output_image_uint8 [output_num_pixels];
  
  for(int i=0; i<output_num_pixels;i++)
    output_image_uint8[i]=(uint8_t) (interpreter->typed_output_tensor<float>(0)[i]*255);
  
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "invoked \n";
  LOG(INFO) << "ex time: "
            << (get_us(stop_time) - get_us(start_time)) / (1000)
            << " ms \n";
  //write_JPEG_file(output_image_uint8,(s->output_jpg_name).c_str(),output_width,output_height,15);

}

void display_usage() {
  LOG(INFO) << "jpeg_compression\n"
            << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
            << "--count, -c: loop interpreter->Invoke() for certain times\n"
            << "--input_mean, -b: input mean\n"
            << "--input_std, -s: input standard deviation\n"
            << "--input_image, -i: input_image_name.ppm\n"
            << "--output_image, -o: output_image_name.jpg"
            << "--tflite_model, -m: model_name.tflite\n"
            << "--threads, -t: number of threads\n"
            << "--verbose, -v: [0|1] print more information\n"
            << "\n";
}

int Main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"count", required_argument, nullptr, 'c'},
        {"verbose", required_argument, nullptr, 'v'},
        {"input_image", required_argument, nullptr, 'i'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"output_image", required_argument, nullptr, 'o'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "a:b:c:f:i:m:o:s:t:v:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_ppm_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'o':
        s.output_jpg_name = optarg;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }
  RunInference(&s);
  return 0;
}

}  // namespace jpeg_compression
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::cnn_inference::Main(argc, argv);
}
