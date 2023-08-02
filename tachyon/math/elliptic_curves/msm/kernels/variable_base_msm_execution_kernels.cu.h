#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_VARIABLE_BASE_MSM_EXECUTION_KERNELS_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_VARIABLE_BASE_MSM_EXECUTION_KERNELS_H_

#include <cmath>

// It is guided to use "umbrella" header instead of
// "third_party/gpus/cuda/include/cub/cub.cuh".
// See https://nvlabs.github.io/cub/#sec6
#include <cub/cub.cuh>

#include "tachyon/base/bits.h"
#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/scoped_async_memory.h"
#include "tachyon/device/gpu/scoped_event.h"
#include "tachyon/device/gpu/scoped_stream.h"
#include "tachyon/math/elliptic_curves/msm/kernels/variable_base_msm_setup_kernels.cu.h"

namespace tachyon::math::kernels {
namespace msm {
using namespace device::gpu;

template <typename Curve>
struct ExecutionConfig {
  gpuMemPool_t mem_pool;
  gpuStream_t stream;
  AffinePoint<Curve>* bases;
  typename AffinePoint<Curve>::ScalarField* scalars;
  JacobianPoint<Curve>* results;
  unsigned int log_scalars_count;
  gpuEvent_t h2d_copy_finished;
  cudaHostFn_t h2d_copy_finished_callback;
  void* h2d_copy_finished_callback_data;
  gpuEvent_t d2h_copy_finished;
  cudaHostFn_t d2h_copy_finished_callback;
  void* d2h_copy_finished_callback_data;
  bool force_min_chunk_size;
  unsigned int log_min_chunk_size;
  bool force_max_chunk_size;
  unsigned int log_max_chunk_size;
};

template <typename Curve>
struct ExtendedConfig {
  ExecutionConfig<Curve> execution_config;
  cudaPointerAttributes scalars_attributes;
  cudaPointerAttributes bases_attributes;
  cudaPointerAttributes results_attributes;
  unsigned int log_min_inputs_count;
  unsigned int log_max_inputs_count;
  cudaDeviceProp device_props;
};

unsigned int GetWindowsBitsCount(unsigned int log_scalars_count) {
  switch (log_scalars_count) {
    case 14:
      return 13;
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
      return 15;
    case 20:
    case 21:
      return 16;
    case 22:
      return 17;
    case 23:
      return 19;
    case 24:
    case 25:
      return 20;
    case 26:
      return 22;
    default:
      return std::max(log_scalars_count, 3u);
  }
}

unsigned int GetLogMinInputsCount(unsigned int log_scalars_count) {
  switch (log_scalars_count) {
    case 18:
    case 19:
      return 17;
    case 20:
      return 18;
    case 21:
    case 22:
    case 23:
    case 24:
      return 19;
    case 25:
    case 26:
      return 22;
    default:
      return log_scalars_count;
  }
}

template <typename ScalarField>
unsigned int GetWindowsCount(unsigned int window_bits_count) {
  return (ScalarField::Config::kModulusBits - 1) / window_bits_count + 1;
}

unsigned int GetOptimalLogDataSplit(unsigned int mpc,
                                    unsigned int source_window_bits,
                                    unsigned int target_window_bits,
                                    unsigned int target_windows_count) {
#define MAX_THREADS 32
#define MIN_BLOCKS 16
  unsigned int full_occupancy = mpc * MAX_THREADS * MIN_BLOCKS;
  unsigned int target = full_occupancy << 6;
  unsigned int unit_threads_count = target_windows_count << target_window_bits;
  unsigned int split_target =
      base::bits::Log2Ceiling(target / unit_threads_count);
  unsigned int split_limit = source_window_bits - target_window_bits - 1;
  return std::min(split_target, split_limit);
#undef MIN_BLOCKS
#undef MAX_THREADS
}

template <typename Curve,
          typename ScalarField = typename PointXYZZ<Curve>::ScalarField>
gpuError_t ScheduleExecution(const ExtendedConfig<Curve>& config,
                             const bool dry_run) {
  ExecutionConfig<Curve> ec = config.execution_config;
  gpuMemPool_t pool = ec.mem_pool;
  gpuStream_t stream = ec.stream;
  unsigned int log_scalars_count = ec.log_scalars_count;
  unsigned int scalars_count = 1 << log_scalars_count;
  unsigned int log_min_inputs_count = config.log_min_inputs_count;
  unsigned int log_max_inputs_count = config.log_max_inputs_count;
  unsigned int bits_count_pass_one = GetWindowsBitsCount(log_scalars_count);
  unsigned int signed_bits_count_pass_one = bits_count_pass_one - 1;
  unsigned int windows_count_pass_one =
      GetWindowsCount<ScalarField>(bits_count_pass_one);
  unsigned int buckets_count_pass_one = windows_count_pass_one
                                        << signed_bits_count_pass_one;
  unsigned int top_window_unused_bits =
      windows_count_pass_one * bits_count_pass_one -
      ScalarField::Config::kModulusBits;
  unsigned int extended_buckets_count_pass_one = buckets_count_pass_one +
                                                 windows_count_pass_one - 1 +
                                                 (1 << top_window_unused_bits);
  bool copy_scalars =
      config.scalars_attributes.type == cudaMemoryTypeUnregistered ||
      config.scalars_attributes.type == cudaMemoryTypeHost;
  bool copy_bases =
      config.bases_attributes.type == cudaMemoryTypeUnregistered ||
      config.bases_attributes.type == cudaMemoryTypeHost;
  bool copy_results =
      config.results_attributes.type == cudaMemoryTypeUnregistered ||
      config.results_attributes.type == cudaMemoryTypeHost;

  ScopedEvent execution_started_event;
  gpuError_t error;
  if (!dry_run) {
    execution_started_event = CreateEventWithFlags(gpuEventDisableTiming);
    error = gpuEventRecord(execution_started_event.get(), stream);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to gpuEventRecord()";
      return error;
    }
  }

  ScopedStream stream_copy_scalars;
  ScopedStream stream_copy_bases;
  ScopedStream stream_copy_finished;
  ScopedStream stream_sort_a;
  ScopedStream stream_sort_b;
  if (!dry_run) {
    if (copy_scalars) {
      stream_copy_scalars = CreateStream();
      error = gpuStreamWaitEvent(stream_copy_scalars.get(),
                                 execution_started_event.get());
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
        return error;
      }
    }
    if (copy_bases) {
      stream_copy_bases = CreateStream();
      error = gpuStreamWaitEvent(stream_copy_bases.get(),
                                 execution_started_event.get());
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
        return error;
      }
    }
    if (copy_scalars || copy_bases) {
      stream_copy_finished = CreateStream();
    }
    stream_sort_a = CreateStream();
    error =
        gpuStreamWaitEvent(stream_sort_a.get(), execution_started_event.get());
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
      return error;
    }
    stream_sort_b = CreateStream();
    error =
        gpuStreamWaitEvent(stream_sort_b.get(), execution_started_event.get());
    execution_started_event.reset();
  }

  ScopedAsyncMemory<PointXYZZ<Curve>> buckets_pass_one =
      MallocFromPoolAsync<PointXYZZ<Curve>>(buckets_count_pass_one, pool,
                                            stream);

  ScopedAsyncMemory<ScalarField> inputs_scalars;
  ScopedEvent event_scalars_free;
  ScopedEvent event_scalars_loaded;
  if (copy_scalars) {
    inputs_scalars = MallocFromPoolAsync<ScalarField>(
        1 << config.log_max_inputs_count, pool, stream);
    event_scalars_free = CreateEventWithFlags(gpuEventDisableTiming);
    event_scalars_loaded = CreateEventWithFlags(gpuEventDisableTiming);
  }

  ScopedAsyncMemory<AffinePoint<Curve>> inputs_bases;
  ScopedEvent event_bases_free;
  ScopedEvent event_bases_loaded;
  if (copy_bases) {
    inputs_bases = MallocFromPoolAsync<AffinePoint<Curve>>(
        1 << config.log_max_inputs_count, pool, stream);
    event_bases_free = CreateEventWithFlags(gpuEventDisableTiming);
    event_bases_loaded = CreateEventWithFlags(gpuEventDisableTiming);
  }

  for (unsigned int inputs_offset = 0, log_inputs_count = log_min_inputs_count;
       inputs_offset < scalars_count;) {
    unsigned int inputs_count = 1 << log_inputs_count;
    bool is_first_loop = inputs_offset == 0;
    bool is_last_loop = inputs_offset + inputs_count == scalars_count;
    unsigned int input_indexes_count = windows_count_pass_one
                                       << log_inputs_count;

    if (!dry_run) {
      if (is_first_loop &&
          (config.scalars_attributes.type == cudaMemoryTypeUnregistered ||
           config.bases_attributes.type == cudaMemoryTypeUnregistered))
        error =
            InitializeBuckets(buckets_pass_one.get(), buckets_count_pass_one, stream);
      if (error != gpuSuccess) return error;
      if (copy_scalars) {
        error = gpuStreamWaitEvent(stream_copy_scalars.get(),
                                   event_scalars_free.get());
        if (error != gpuSuccess) {
          GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
          return error;
        }
        const size_t inputs_size = sizeof(ScalarField) << log_inputs_count;
        error = cudaMemcpyAsync(inputs_scalars.get(), ec.scalars + inputs_offset,
                                inputs_size, gpuMemcpyHostToDevice,
                                stream_copy_scalars.get());
        if (error != gpuSuccess) {
          GPU_LOG(ERROR, error) << "Failed to gpuEventRecord()";
          return error;
        }
        error = gpuEventRecord(event_scalars_loaded.get(),
                               stream_copy_scalars.get());
        if (error != gpuSuccess) {
          GPU_LOG(ERROR, error) << "Failed to gpuEventRecord()";
          return error;
        }
      }
      if (copy_bases) {
        if (copy_scalars) {
          error = gpuStreamWaitEvent(stream_copy_bases.get(),
                                     event_scalars_loaded.get());
          if (error != gpuSuccess) {
            GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
            return error;
          }
        }
        error =
            gpuStreamWaitEvent(stream_copy_bases.get(), event_bases_free.get());
        if (error != gpuSuccess) {
          GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
          return error;
        }
        size_t bases_size = sizeof(AffinePoint<Curve>) << log_inputs_count;
        error = cudaMemcpyAsync(inputs_bases.get(), ec.bases + inputs_offset,
                                bases_size, gpuMemcpyHostToDevice,
                                stream_copy_bases.get());
        if (error != gpuSuccess) {
          GPU_LOG(ERROR, error) << "Failed to cudaMemcpyAsync()";
          return error;
        }
        error =
            gpuEventRecord(event_bases_loaded.get(), stream_copy_bases.get());
        if (error != gpuSuccess) {
          GPU_LOG(ERROR, error) << "Failed to gpuEventRecord()";
          return error;
        }
      }
      if (is_last_loop && (copy_bases || copy_scalars)) {
        if (copy_scalars) {
          error = gpuStreamWaitEvent(stream_copy_finished.get(),
                                     event_scalars_loaded.get());
          if (error != gpuSuccess) {
            GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
            return error;
          }
        }
        if (copy_bases) {
          error = gpuStreamWaitEvent(stream_copy_finished.get(),
                                     event_bases_loaded.get());
          if (error != gpuSuccess) {
            GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
            return error;
          }
        }
        if (ec.h2d_copy_finished) {
          error =
              gpuEventRecord(ec.h2d_copy_finished, stream_copy_finished.get());
          if (error != gpuSuccess) {
            GPU_LOG(ERROR, error) << "Failed to gpuEventRecord()";
            return error;
          }
        }
        if (ec.h2d_copy_finished_callback) {
          error = cudaLaunchHostFunc(stream_copy_finished.get(),
                                     ec.h2d_copy_finished_callback,
                                     ec.h2d_copy_finished_callback_data);
          if (error != gpuSuccess) {
            GPU_LOG(ERROR, error) << "Failed to cudaLaunchHostFunc()";
            return error;
          }
        }
      }
      if (is_first_loop &&
          config.scalars_attributes.type != cudaMemoryTypeUnregistered &&
          config.bases_attributes.type != cudaMemoryTypeUnregistered) {
        error =
            InitializeBuckets(buckets_pass_one.get(), buckets_count_pass_one, stream);
        if (error != gpuSuccess) return error;
      }
    }

    // compute bucket indexes
    ScopedAsyncMemory<unsigned int> bucket_indexes =
        MallocFromPoolAsync<unsigned int>(input_indexes_count + inputs_count,
                                          pool, stream);
    ScopedAsyncMemory<unsigned int> base_indexes =
        MallocFromPoolAsync<unsigned int>(input_indexes_count + inputs_count,
                                          pool, stream);
    if (!dry_run) {
      if (copy_scalars) {
        error = gpuStreamWaitEvent(stream, event_scalars_loaded.get());
      }
      error = ComputeBucketIndexes(
          copy_scalars ? inputs_scalars.get() : ec.scalars + inputs_offset,
          windows_count_pass_one, bits_count_pass_one,
          bucket_indexes.get() + inputs_count,
          base_indexes.get() + inputs_count, inputs_count, stream);
      if (error != gpuSuccess) return error;
      if (copy_scalars)
        error = gpuEventRecord(event_scalars_free.get(), stream);
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error) << "Failed to gpuEventRecord()";
        return error;
      }
    }

    if (is_last_loop && copy_scalars) {
      inputs_scalars.reset(stream);
    }

    // sort base indexes by bucket indexes
    ScopedAsyncMemory<uint8_t> input_indexes_sort_temp_storage;
    size_t input_indexes_sort_temp_storage_bytes = 0;
    error = cub::DeviceRadixSort::SortPairs(
        input_indexes_sort_temp_storage.get(),
        input_indexes_sort_temp_storage_bytes,
        bucket_indexes.get() + inputs_count, bucket_indexes.get(),
        base_indexes.get() + inputs_count, base_indexes.get(), inputs_count, 0,
        bits_count_pass_one);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to cub::DeviceRadixSort::SortPairs()";
      return error;
    }
    input_indexes_sort_temp_storage = MallocFromPoolAsync<uint8_t>(
        input_indexes_sort_temp_storage_bytes, pool, stream);
    if (!dry_run) {
      for (unsigned int i = 0; i < windows_count_pass_one; ++i) {
        unsigned int offset_out = i * inputs_count;
        unsigned int offset_in = offset_out + inputs_count;
        error = cub::DeviceRadixSort::SortPairs(
            input_indexes_sort_temp_storage.get(),
            input_indexes_sort_temp_storage_bytes,
            bucket_indexes.get() + offset_in, bucket_indexes.get() + offset_out,
            base_indexes.get() + offset_in, base_indexes.get() + offset_out,
            inputs_count, 0, bits_count_pass_one, stream);
        if (error != gpuSuccess) {
          GPU_LOG(ERROR, error) << "Failed to cub::DeviceRadixSort::SortPairs()";
          return error;
        }
      }
    }
    input_indexes_sort_temp_storage.reset(stream);

    // run length encode bucket runs
    ScopedAsyncMemory<unsigned int> unique_bucket_indexes =
        MallocFromPoolAsync<unsigned int>(extended_buckets_count_pass_one, pool,
                                          stream);
    ScopedAsyncMemory<unsigned int> bucket_run_lengths =
        MallocFromPoolAsync<unsigned int>(extended_buckets_count_pass_one, pool,
                                          stream);
    ScopedAsyncMemory<unsigned int> bucket_runs_count =
        MallocFromPoolAsync<unsigned int>(1, pool, stream);
    ScopedAsyncMemory<uint8_t> encode_temp_storage;
    size_t encode_temp_storage_bytes = 0;
    error = cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage.get(), encode_temp_storage_bytes,
        bucket_indexes.get(), unique_bucket_indexes.get(),
        bucket_run_lengths.get(), bucket_runs_count.get(), input_indexes_count);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to cub::DeviceRunLengthEncode::Encode()";
      return error;
    }
    encode_temp_storage =
        MallocFromPoolAsync<uint8_t>(encode_temp_storage_bytes, pool, stream);
    if (!dry_run) {
      error = cub::DeviceRunLengthEncode::Encode(
          encode_temp_storage.get(), encode_temp_storage_bytes,
          bucket_indexes.get(), unique_bucket_indexes.get(),
          bucket_run_lengths.get(), bucket_runs_count.get(),
          input_indexes_count, stream);
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error) << "Failed to cub::DeviceRunLengthEncode::Encode()";
        return error;
      }
    }
    encode_temp_storage.reset(stream);
    bucket_indexes.reset(stream);

    // compute bucket run offsets
    ScopedAsyncMemory<unsigned int> bucket_run_offsets =
        MallocFromPoolAsync<unsigned int>(extended_buckets_count_pass_one, pool,
                                          stream);
    ScopedAsyncMemory<uint8_t> scan_temp_storage;
    size_t scan_temp_storage_bytes = 0;
    error = cub::DeviceScan::ExclusiveSum(
        scan_temp_storage.get(), scan_temp_storage_bytes,
        bucket_run_lengths.get(), bucket_run_offsets.get(),
        extended_buckets_count_pass_one);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to cub::DeviceScan::ExclusiveSum()";
      return error;
    }
    scan_temp_storage =
        MallocFromPoolAsync<uint8_t>(scan_temp_storage_bytes, pool, stream);
    if (!dry_run) {
      error = cub::DeviceScan::ExclusiveSum(
          scan_temp_storage.get(), scan_temp_storage_bytes,
          bucket_run_lengths.get(), bucket_run_offsets.get(),
          extended_buckets_count_pass_one, stream);
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error) << "Failed to cub::DeviceScan::ExclusiveSum()";
        return error;
      }
    }
    scan_temp_storage.reset(stream);

    if (!dry_run) {
      error = RemoveZeroBuckets(
          unique_bucket_indexes.get(), bucket_run_lengths.get(),
          bucket_runs_count.get(), extended_buckets_count_pass_one, stream);
      if (error != gpuSuccess) return error;
    }
    bucket_runs_count.reset(stream);

    // sort run offsets by run lengths
    // sort run indexes by run lengths
    ScopedAsyncMemory<unsigned int> sorted_bucket_run_lengths =
        MallocFromPoolAsync<unsigned int>(extended_buckets_count_pass_one, pool,
                                          stream);
    ScopedAsyncMemory<unsigned int> sorted_bucket_run_offsets =
        MallocFromPoolAsync<unsigned int>(extended_buckets_count_pass_one, pool,
                                          stream);
    ScopedAsyncMemory<unsigned int> sorted_unique_bucket_indexes =
        MallocFromPoolAsync<unsigned int>(extended_buckets_count_pass_one, pool,
                                          stream);

    ScopedAsyncMemory<uint8_t> sort_offsets_temp_storage;
    size_t sort_offsets_temp_storage_bytes = 0;
    error = cub::DeviceRadixSort::SortPairsDescending(
        sort_offsets_temp_storage.get(), sort_offsets_temp_storage_bytes,
        bucket_run_lengths.get(), sorted_bucket_run_lengths.get(),
        bucket_run_offsets.get(), sorted_bucket_run_offsets.get(),
        extended_buckets_count_pass_one, 0, log_inputs_count + 1);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error)
          << "Failed to cub::DeviceRadixSort::SortPairsDescending()";
      return error;
    }
    sort_offsets_temp_storage = MallocFromPoolAsync<uint8_t>(
        sort_offsets_temp_storage_bytes, pool, stream);

    ScopedAsyncMemory<uint8_t> sort_indexes_temp_storage;
    size_t sort_indexes_temp_storage_bytes = 0;
    error = cub::DeviceRadixSort::SortPairsDescending(
        sort_indexes_temp_storage.get(), sort_indexes_temp_storage_bytes,
        bucket_run_lengths.get(), sorted_bucket_run_lengths.get(),
        unique_bucket_indexes.get(), sorted_unique_bucket_indexes.get(),
        extended_buckets_count_pass_one, 0, log_inputs_count + 1);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error)
          << "Failed to cub::DeviceRadixSort::SortPairsDescending()";
      return error;
    }
    sort_indexes_temp_storage = MallocFromPoolAsync<uint8_t>(
        sort_indexes_temp_storage_bytes, pool, stream);

    if (!dry_run) {
      ScopedEvent event_sort_inputs_ready =
          CreateEventWithFlags(gpuEventDisableTiming);
      error = gpuEventRecord(event_sort_inputs_ready.get(), stream);
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error) << "Failed to gpuEventRecord()";
        return error;
      }
      error = gpuStreamWaitEvent(stream_sort_a.get(),
                                 event_sort_inputs_ready.get());
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
        return error;
      }
      error = gpuStreamWaitEvent(stream_sort_b.get(),
                                 event_sort_inputs_ready.get());
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
        return error;
      }
      event_sort_inputs_ready.reset();
      error = cub::DeviceRadixSort::SortPairsDescending(
          sort_offsets_temp_storage.get(), sort_offsets_temp_storage_bytes,
          bucket_run_lengths.get(), sorted_bucket_run_lengths.get(),
          bucket_run_offsets.get(), sorted_bucket_run_offsets.get(),
          extended_buckets_count_pass_one, 0, log_inputs_count + 1,
          stream_sort_a.get());
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error)
            << "Failed to cub::DeviceRadixSort::SortPairsDescending()";
        return error;
      }
      error = cub::DeviceRadixSort::SortPairsDescending(
          sort_indexes_temp_storage.get(), sort_indexes_temp_storage_bytes,
          bucket_run_lengths.get(), sorted_bucket_run_lengths.get(),
          unique_bucket_indexes.get(), sorted_unique_bucket_indexes.get(),
          extended_buckets_count_pass_one, 0, log_inputs_count + 1,
          stream_sort_b.get());
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error)
            << "Failed to cub::DeviceRadixSort::SortPairsDescending()";
        return error;
      }
      ScopedEvent event_sort_a = CreateEventWithFlags(gpuEventDisableTiming);
      ScopedEvent event_sort_b = CreateEventWithFlags(gpuEventDisableTiming);
      error = gpuEventRecord(event_sort_a.get(), stream_sort_a.get());
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error) << "Failed to gpuEventRecord()";
        return error;
      }
      error = gpuEventRecord(event_sort_b.get(), stream_sort_b.get());
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error) << "Failed to gpuEventRecord()";
        return error;
      }
      error = gpuStreamWaitEvent(stream, event_sort_a.get());
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
        return error;
      }
      error = gpuStreamWaitEvent(stream, event_sort_b.get());
      if (error != gpuSuccess) {
        GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
        return error;
      }
    }

    sort_offsets_temp_storage.reset(stream);
    sort_indexes_temp_storage.reset(stream);
    bucket_run_lengths.reset(stream);
    bucket_run_offsets.reset(stream);
    unique_bucket_indexes.reset(stream);

    // aggregate buckets
    if (!dry_run) {
      if (copy_bases) {
        error = gpuStreamWaitEvent(stream, event_bases_loaded.get());
        if (error != gpuSuccess) {
          GPU_LOG(ERROR, error) << "Failed to gpuStreamWaitEvent()";
          return error;
        }
      }
      error = AggregateBuckets(
          is_first_loop, base_indexes.get(), sorted_bucket_run_offsets.get(),
          sorted_bucket_run_lengths.get(), sorted_unique_bucket_indexes.get(),
          copy_bases ? inputs_bases.get() : ec.bases + inputs_offset,
          buckets_pass_one.get(), buckets_count_pass_one, stream);
      if (error != gpuSuccess) return error;
      if (copy_bases) {
        error = gpuEventRecord(event_bases_free.get(), stream);
        if (error != gpuSuccess) {
          GPU_LOG(ERROR, error) << "Failed to gpuEventRecord()";
          return error;
        }
      }
    }
    base_indexes.reset(stream);
    sorted_bucket_run_offsets.reset(stream);
    sorted_bucket_run_lengths.reset(stream);
    sorted_unique_bucket_indexes.reset(stream);
    if (is_last_loop && copy_bases) {
      inputs_bases.reset(stream);
    }
    inputs_offset += inputs_count;
    if (!is_first_loop && log_inputs_count < log_max_inputs_count)
      log_inputs_count++;
  }

  ScopedAsyncMemory<PointXYZZ<Curve>> top_buckets =
      MallocFromPoolAsync<PointXYZZ<Curve>>(windows_count_pass_one, pool,
                                            stream);

  if (!dry_run) {
    if (copy_scalars) {
      stream_copy_scalars.reset();
      event_scalars_loaded.reset();
      event_scalars_free.reset();
    }
    if (copy_bases) {
      stream_copy_bases.reset();
      event_bases_loaded.reset();
      event_bases_free.reset();
    }
    if (copy_scalars || copy_bases) {
      stream_copy_finished.reset();
    }
    stream_sort_a.reset();
    stream_sort_b.reset();
    if (top_window_unused_bits != 0) {
      unsigned int top_window_offset = (windows_count_pass_one - 1)
                                       << signed_bits_count_pass_one;
      unsigned int top_window_used_bits =
          signed_bits_count_pass_one - top_window_unused_bits;
      unsigned int top_window_used_buckets_count = 1 << top_window_used_bits;
      unsigned int top_window_unused_buckets_count =
          (1 << signed_bits_count_pass_one) - top_window_used_buckets_count;
      unsigned int top_window_unused_buckets_offset =
          top_window_offset + top_window_used_buckets_count;
      for (unsigned int i = 0; i < top_window_unused_bits; ++i) {
        error =
            ReduceBuckets(buckets_pass_one.get() + top_window_offset,
                          1 << (signed_bits_count_pass_one - i - 1), stream);
        if (error != gpuSuccess) return error;
      }
      error = InitializeBuckets(
          buckets_pass_one.get() + top_window_unused_buckets_offset,
          top_window_unused_buckets_count, stream);
      if (error != gpuSuccess) return error;
    }
    error = ExtractTopBuckets(buckets_pass_one.get(), top_buckets.get(),
                              signed_bits_count_pass_one,
                              windows_count_pass_one, stream);
    if (error != gpuSuccess) return error;
  }

  unsigned int source_bits_count = signed_bits_count_pass_one;
  unsigned int source_windows_count = windows_count_pass_one;
  ScopedAsyncMemory<PointXYZZ<Curve>> source_buckets =
      std::move(buckets_pass_one);
  ScopedAsyncMemory<PointXYZZ<Curve>> target_buckets;
  for (unsigned int i = 0;; ++i) {
    unsigned int target_bits_count = (source_bits_count + 1) >> 1;
    unsigned int target_windows_count = source_windows_count << 1;
    unsigned int target_buckets_count = target_windows_count
                                        << target_bits_count;
    unsigned int log_data_split = GetOptimalLogDataSplit(
        config.device_props.multiProcessorCount, source_bits_count,
        target_bits_count, target_windows_count);
    unsigned int total_buckets_count = target_buckets_count << log_data_split;
    target_buckets = MallocFromPoolAsync<PointXYZZ<Curve>>(total_buckets_count,
                                                           pool, stream);
    if (!dry_run) {
      error = SplitWindows(source_bits_count, source_windows_count,
                           source_buckets.get(), target_buckets.get(),
                           total_buckets_count, stream);
      if (error != gpuSuccess) return error;
    }
    source_buckets.reset(stream);

    if (!dry_run) {
      for (unsigned int j = 0; j < log_data_split; ++j) {
        error = ReduceBuckets(target_buckets.get(), total_buckets_count >> (j + 1),
                              stream);
        if (error != gpuSuccess) return error;
      }
    }
    if (target_bits_count == 1) {
      ScopedAsyncMemory<JacobianPoint<Curve>> results;
      unsigned int result_windows_count =
          ScalarField::Config::kModulusBits;
      if (copy_results) {
        results = MallocFromPoolAsync<JacobianPoint<Curve>>(
            result_windows_count, pool, stream);
      }
      if (!dry_run) {
        error = LastPassGather(bits_count_pass_one, target_buckets.get(), top_buckets.get(),
                               copy_results ? results.get() : ec.results,
                               result_windows_count, stream);
        if (error != gpuSuccess) return error;
        if (copy_results) {
          error = cudaMemcpyAsync(
              ec.results, results.get(),
              sizeof(JacobianPoint<Curve>) * result_windows_count,
              gpuMemcpyDeviceToHost, stream);
          if (error != gpuSuccess) {
            GPU_LOG(ERROR, error) << "Failed to cudaMemcpyAsync()";
            return error;
          }
          if (ec.d2h_copy_finished) {
            error = gpuEventRecord(ec.d2h_copy_finished, stream);
            if (error != gpuSuccess) {
              GPU_LOG(ERROR, error) << "Failed to gpuEventRecord()";
              return error;
            }
          }
          if (ec.d2h_copy_finished_callback) {
            error = cudaLaunchHostFunc(stream, ec.d2h_copy_finished_callback,
                                       ec.d2h_copy_finished_callback_data);
            if (error != gpuSuccess) {
              GPU_LOG(ERROR, error) << "Failed to cudaLaunchHostFunc()";
              return error;
            }
          }
        }
      }
      if (copy_results) results.reset(stream);
      target_buckets.reset(stream);
      top_buckets.reset(stream);

      break;
    }
    source_buckets = std::move(target_buckets);
    source_bits_count = target_bits_count;
    source_windows_count = target_windows_count;
  }

  return gpuSuccess;
}

template <typename Curve>
gpuError_t ExecuteAsync(const ExecutionConfig<Curve>& config) {
  int device_id;
  gpuError_t error = cudaGetDevice(&device_id);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed to cudaGetDevice()";
    return error;
  }
  cudaDeviceProp props{};
  error = cudaGetDeviceProperties(&props, device_id);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed to cudaGetDeviceProperties()";
    return error;
  }
  unsigned int log_scalars_count = config.log_scalars_count;
  cudaPointerAttributes scalars_attributes{};
  cudaPointerAttributes bases_attributes{};
  cudaPointerAttributes results_attributes{};
  error = cudaPointerGetAttributes(&scalars_attributes, config.scalars);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed to cudaPointerGetAttributes()";
    return error;
  }
  error = cudaPointerGetAttributes(&bases_attributes, config.bases);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed to cudaPointerGetAttributes()";
    return error;
  }
  error = cudaPointerGetAttributes(&results_attributes, config.results);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed to cudaPointerGetAttributes()";
    return error;
  }
  bool copy_scalars = scalars_attributes.type == cudaMemoryTypeUnregistered ||
                      scalars_attributes.type == cudaMemoryTypeHost;
  unsigned int log_min_inputs_count =
      config.force_min_chunk_size ? config.log_min_chunk_size
      : copy_scalars              ? GetLogMinInputsCount(log_scalars_count)
                                  : log_scalars_count;
  if (config.force_max_chunk_size) {
    ExtendedConfig<Curve> extended_config = {
        config,
        scalars_attributes,
        bases_attributes,
        results_attributes,
        std::min(log_min_inputs_count, config.log_max_chunk_size),
        config.log_max_chunk_size,
        props};
    return ScheduleExecution(extended_config, false);
  }
  unsigned int log_max_inputs_count =
      copy_scalars ? max(log_min_inputs_count,
                         log_scalars_count == 0 ? 0 : log_scalars_count - 1)
                   : log_scalars_count;
  while (true) {
    ExtendedConfig<Curve> extended_config = {config,
                                             scalars_attributes,
                                             bases_attributes,
                                             results_attributes,
                                             log_min_inputs_count,
                                             log_max_inputs_count,
                                             props};
    gpuError_t error = ScheduleExecution(extended_config, true);
    if (error == cudaErrorMemoryAllocation) {
      log_max_inputs_count--;
      if (!copy_scalars) log_min_inputs_count--;
      if (log_max_inputs_count < GetLogMinInputsCount(log_scalars_count))
        return cudaErrorMemoryAllocation;
      continue;
    }
    if (error != gpuSuccess) {
      return error;
    }
    return ScheduleExecution(extended_config, false);
  }
}

}  // namespace msm
}  // namespace tachyon::math::kernels

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_MSM_EXECUTION_KERNELS_H_
