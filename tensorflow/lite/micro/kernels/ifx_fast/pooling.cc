/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

/*
IFX FAST
Preinterpretation support:
* TAGS="record_model autodump" -- Run using these tags to write the used kernel
variants to a file
* TAGS="recorded_model" -- Use this TAG to use previously recorded kernel
variants
*     Advantages: Smaller binaries, since only the required kernels are compiled
*                 Smaller runtime, because many intermediate values are stored
==============================================================================*/

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/ifx_fast/pooling/pooling_impl.h"

namespace tflite {
namespace ops {
namespace micro {

TfLiteRegistration Register_AVERAGE_POOL_2D() {
  return {/*init=*/pooling::Init,
          /*free=*/nullptr,
          /*prepare=*/pooling::PrepareAverage,
          /*invoke=*/pooling::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_MAX_POOL_2D() {
  return {/*init=*/pooling::Init,
          /*free=*/nullptr,
          /*prepare=*/pooling::PrepareMax,
          /*invoke=*/pooling::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
