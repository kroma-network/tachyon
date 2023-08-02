/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TACHYON_DEVICE_GPU_GPU_INIT_H_
#define TACHYON_DEVICE_GPU_GPU_INIT_H_

#include <string>

namespace tachyon::device::gpu {

class Platform;

// Initializes the GPU platform and returns true if the GPU
// platform could be initialized.
bool ValidateGPUMachineManager();

// Returns the GPU machine manager singleton, creating it and
// initializing the GPUs on the machine if needed the first time it is
// called.  Must only be called when there is a valid GPU environment
// in the process (e.g., ValidateGPUMachineManager() returns true).
Platform* GPUMachineManager();

// Returns the string describing the name of the GPU platform in use.
// This value is "CUDA" by default, and
// "ROCM" when TACHYON is built with `--config==rocm`
std::string GpuPlatformName();

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_GPU_INIT_H_
