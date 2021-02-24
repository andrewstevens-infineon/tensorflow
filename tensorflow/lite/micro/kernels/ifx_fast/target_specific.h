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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_IFX_FAST_TARGET_SPECIFIC_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_IFX_FAST_TARGET_SPECIFIC_H_

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace micro {
namespace conv {

inline PaddingType RuntimePaddingType(TfLitePadding padding) {
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

}  // namespace conv

struct LayerOps {
  /*
   * Class that contains basic operations for kernels. This makes using an
   * advanced instruction (like MAC instructions) set easier.
   */
 public:
  static inline void reset_acc(int32_t* acc, int32_t val = 0) { *acc = val; }

  static inline void accumulate(int32_t* acc, int32_t mul1, int32_t mul2) {
    *acc += mul1 * mul2;
  }

  static inline int32_t get_acc(int32_t* acc) { return *acc; }
};

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif /* TENSORFLOW_LITE_MICRO_KERNELS_IFX_FAST_TARGET_SPECIFIC_H_ */
