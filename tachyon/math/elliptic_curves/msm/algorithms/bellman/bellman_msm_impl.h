#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_BELLMAN_BELLMAN_MSM_IMPL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_BELLMAN_BELLMAN_MSM_IMPL_H_

#include <algorithm>
#include <cmath>
#include <utility>

// It is guided to use "umbrella" header instead of
// "third_party/gpus/cuda/include/cub/cub.cuh".
// See https://nvlabs.github.io/cub/#sec6
#include <cub/cub.cuh>  // NOLINT(build/include_order)

#include "tachyon/base/bits.h"
#include "tachyon/device/gpu/cuda/cub_helper.h"
#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/device/gpu/scoped_event.h"
#include "tachyon/device/gpu/scoped_stream.h"
#include "tachyon/math/elliptic_curves/msm/kernels/bellman/bellman_msm_kernels.cu.h"

namespace tachyon::math::bellman {

template <typename Curve>
struct ExecutionConfig {
  gpuMemPool_t mem_pool = nullptr;
  gpuStream_t stream = nullptr;
  const AffinePoint<Curve>* bases = nullptr;
  const typename AffinePoint<Curve>::ScalarField* scalars = nullptr;
  JacobianPoint<Curve>* results = nullptr;
  unsigned int log_scalars_count = 0;
  gpuEvent_t h2d_copy_finished = nullptr;
  cudaHostFn_t h2d_copy_finished_callback = nullptr;
  void* h2d_copy_finished_callback_data = nullptr;
  gpuEvent_t d2h_copy_finished = nullptr;
  cudaHostFn_t d2h_copy_finished_callback = nullptr;
  void* d2h_copy_finished_callback_data = nullptr;
  bool force_min_chunk_size = false;
  unsigned int log_min_chunk_size = 0;
  bool force_max_chunk_size = false;
  unsigned int log_max_chunk_size = 0;
};

template <typename Curve>
struct ExtendedConfig {
  ExecutionConfig<Curve> execution_config;
  cudaPointerAttributes scalars_attributes;
  cudaPointerAttributes bases_attributes;
  cudaPointerAttributes results_attributes;
  unsigned int log_min_inputs_count = 0;
  unsigned int log_max_inputs_count = 0;
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
                             bool dry_run) {
  using namespace device::gpu;
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
      config.scalars_attributes.type == gpuMemoryTypeUnregistered ||
      config.scalars_attributes.type == gpuMemoryTypeHost;
  bool copy_bases = config.bases_attributes.type == gpuMemoryTypeUnregistered ||
                    config.bases_attributes.type == gpuMemoryTypeHost;
  bool copy_results =
      config.results_attributes.type == gpuMemoryTypeUnregistered ||
      config.results_attributes.type == gpuMemoryTypeHost;

  ScopedStream stream_copy_scalars;
  ScopedStream stream_copy_bases;
  ScopedStream stream_copy_finished;
  ScopedStream stream_sort_a;
  ScopedStream stream_sort_b;
  gpuError_t error;
  if (!dry_run) {
    ScopedEvent execution_started_event =
        CreateEventWithFlags(gpuEventDisableTiming);
    RETURN_AND_LOG_IF_GPU_ERROR(
        gpuEventRecord(execution_started_event.get(), stream),
        "Failed to gpuEventRecord()");

    if (copy_scalars) {
      stream_copy_scalars = CreateStream();
      RETURN_AND_LOG_IF_GPU_ERROR(
          gpuStreamWaitEvent(stream_copy_scalars.get(),
                             execution_started_event.get()),
          "Failed to gpuStreamWaitEvent()");
    }

    if (copy_bases) {
      stream_copy_bases = CreateStream();
      RETURN_AND_LOG_IF_GPU_ERROR(
          gpuStreamWaitEvent(stream_copy_bases.get(),
                             execution_started_event.get()),
          "Failed to gpuStreamWaitEvent()");
    }

    if (copy_scalars || copy_bases) {
      stream_copy_finished = CreateStream();
    }

    stream_sort_a = CreateStream();
    RETURN_AND_LOG_IF_GPU_ERROR(
        gpuStreamWaitEvent(stream_sort_a.get(), execution_started_event.get()),
        "Failed to gpuStreamWaitEvent()");

    stream_sort_b = CreateStream();
    RETURN_AND_LOG_IF_GPU_ERROR(
        gpuStreamWaitEvent(stream_sort_b.get(), execution_started_event.get()),
        "Failed to gpuStreamWaitEvent()");
  }

  GpuMemory<PointXYZZ<Curve>> buckets_pass_one =
      GpuMemory<PointXYZZ<Curve>>::MallocFromPoolAsync(buckets_count_pass_one,
                                                       pool, stream);

  GpuMemory<ScalarField> inputs_scalars;
  ScopedEvent event_scalars_free;
  ScopedEvent event_scalars_loaded;
  if (copy_scalars) {
    inputs_scalars = GpuMemory<ScalarField>::MallocFromPoolAsync(
        1 << config.log_max_inputs_count, pool, stream);
    event_scalars_free = CreateEventWithFlags(gpuEventDisableTiming);
    event_scalars_loaded = CreateEventWithFlags(gpuEventDisableTiming);
  }

  GpuMemory<AffinePoint<Curve>> inputs_bases;
  ScopedEvent event_bases_free;
  ScopedEvent event_bases_loaded;
  if (copy_bases) {
    inputs_bases = GpuMemory<AffinePoint<Curve>>::MallocFromPoolAsync(
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
          (config.scalars_attributes.type == gpuMemoryTypeUnregistered ||
           config.bases_attributes.type == gpuMemoryTypeUnregistered)) {
        error = bellman::InitializeBuckets(buckets_pass_one.get(),
                                           buckets_count_pass_one, stream);
        if (UNLIKELY(error != gpuSuccess)) return error;
      }
      if (copy_scalars) {
        RETURN_AND_LOG_IF_GPU_ERROR(
            gpuStreamWaitEvent(stream_copy_scalars.get(),
                               event_scalars_free.get()),
            "Failed to gpuStreamWaitEvent()");
        const size_t inputs_size = sizeof(ScalarField) << log_inputs_count;
        CHECK(inputs_scalars.CopyFromAsync(
            &ec.scalars[inputs_offset], GpuMemoryType::kHost,
            stream_copy_scalars.get(), 0, inputs_size));
        RETURN_AND_LOG_IF_GPU_ERROR(gpuEventRecord(event_scalars_loaded.get(),
                                                   stream_copy_scalars.get()),
                                    "Failed to gpuEventRecord()");
      }
      if (copy_bases) {
        if (copy_scalars) {
          RETURN_AND_LOG_IF_GPU_ERROR(
              gpuStreamWaitEvent(stream_copy_bases.get(),
                                 event_scalars_loaded.get()),
              "Failed to gpuStreamWaitEvent()");
        }
        RETURN_AND_LOG_IF_GPU_ERROR(
            gpuStreamWaitEvent(stream_copy_bases.get(), event_bases_free.get()),
            "Failed to gpuStreamWaitEvent()");
        size_t bases_size = sizeof(AffinePoint<Curve>) << log_inputs_count;
        CHECK(inputs_bases.CopyFromAsync(
            &ec.bases[inputs_offset], GpuMemoryType::kHost,
            stream_copy_bases.get(), 0, bases_size));
        RETURN_AND_LOG_IF_GPU_ERROR(
            gpuEventRecord(event_bases_loaded.get(), stream_copy_bases.get()),
            "Failed to gpuEventRecord()");
      }
      if (is_last_loop && (copy_bases || copy_scalars)) {
        if (copy_scalars) {
          RETURN_AND_LOG_IF_GPU_ERROR(
              gpuStreamWaitEvent(stream_copy_finished.get(),
                                 event_scalars_loaded.get()),
              "Failed to gpuStreamWaitEvent()");
        }
        if (copy_bases) {
          RETURN_AND_LOG_IF_GPU_ERROR(
              gpuStreamWaitEvent(stream_copy_finished.get(),
                                 event_bases_loaded.get()),
              "Failed to gpuStreamWaitEvent()");
        }
        if (ec.h2d_copy_finished) {
          RETURN_AND_LOG_IF_GPU_ERROR(
              gpuEventRecord(ec.h2d_copy_finished, stream_copy_finished.get()),
              "Failed to gpuEventRecord()");
        }
        if (ec.h2d_copy_finished_callback) {
          RETURN_AND_LOG_IF_GPU_ERROR(
              cudaLaunchHostFunc(stream_copy_finished.get(),
                                 ec.h2d_copy_finished_callback,
                                 ec.h2d_copy_finished_callback_data),
              "Failed to cudaLaunchHostFunc()");
        }
      }
      if (is_first_loop &&
          config.scalars_attributes.type != gpuMemoryTypeUnregistered &&
          config.bases_attributes.type != gpuMemoryTypeUnregistered) {
        error = bellman::InitializeBuckets(buckets_pass_one.get(),
                                           buckets_count_pass_one, stream);
        if (UNLIKELY(error != gpuSuccess)) return error;
      }
    }

    // compute bucket indexes
    GpuMemory<unsigned int> bucket_indexes =
        GpuMemory<unsigned int>::MallocFromPoolAsync(
            input_indexes_count + inputs_count, pool, stream);
    GpuMemory<unsigned int> base_indexes =
        GpuMemory<unsigned int>::MallocFromPoolAsync(
            input_indexes_count + inputs_count, pool, stream);
    if (!dry_run) {
      if (copy_scalars) {
        RETURN_AND_LOG_IF_GPU_ERROR(
            gpuStreamWaitEvent(stream, event_scalars_loaded.get()),
            "Failed to gpuStreamWaitEvent()");
      }
      error = bellman::ComputeBucketIndexes(
          copy_scalars ? inputs_scalars.get() : &ec.scalars[inputs_offset],
          windows_count_pass_one, bits_count_pass_one,
          &bucket_indexes[inputs_count], &base_indexes[inputs_count],
          inputs_count, stream);
      if (UNLIKELY(error != gpuSuccess)) return error;
      if (copy_scalars) {
        RETURN_AND_LOG_IF_GPU_ERROR(
            gpuEventRecord(event_scalars_free.get(), stream),
            "Failed to gpuEventRecord()");
      }
    }

    if (is_last_loop && copy_scalars) {
      inputs_scalars.reset();
    }

    // sort base indexes by bucket indexes
    {
      GpuMemory<uint8_t> input_indexes_sort_temp_storage;
      size_t input_indexes_sort_temp_storage_bytes = 0;
      RETURN_AND_LOG_IF_GPU_ERROR(
          cub::DeviceRadixSort::SortPairs(
              input_indexes_sort_temp_storage.get(),
              input_indexes_sort_temp_storage_bytes,
              &bucket_indexes[inputs_count], bucket_indexes.get(),
              &base_indexes[inputs_count], base_indexes.get(), inputs_count, 0,
              bits_count_pass_one),
          "Failed to cub::DeviceRadixSort::SortPairs()");
      input_indexes_sort_temp_storage = GpuMemory<uint8_t>::MallocFromPoolAsync(
          input_indexes_sort_temp_storage_bytes, pool, stream);
      if (!dry_run) {
        for (unsigned int i = 0; i < windows_count_pass_one; ++i) {
          unsigned int offset_out = i * inputs_count;
          unsigned int offset_in = offset_out + inputs_count;
          RETURN_AND_LOG_IF_GPU_ERROR(
              cub::DeviceRadixSort::SortPairs(
                  input_indexes_sort_temp_storage.get(),
                  input_indexes_sort_temp_storage_bytes,
                  &bucket_indexes[offset_in], &bucket_indexes[offset_out],
                  &base_indexes[offset_in], &base_indexes[offset_out],
                  inputs_count, 0, bits_count_pass_one, stream),
              "Failed to cub::DeviceRadixSort::SortPairs()");
        }
      }
    }

    // run length encode bucket runs
    GpuMemory<unsigned int> unique_bucket_indexes =
        GpuMemory<unsigned int>::MallocFromPoolAsync(
            extended_buckets_count_pass_one, pool, stream);
    GpuMemory<unsigned int> bucket_run_lengths =
        GpuMemory<unsigned int>::MallocFromPoolAsync(
            extended_buckets_count_pass_one, pool, stream);
    GpuMemory<unsigned int> bucket_runs_count =
        GpuMemory<unsigned int>::MallocFromPoolAsync(1, pool, stream);
    if (dry_run) {
      error = CUB_TRY_ALLOCATE_WITH_POOL(
          pool, stream, cub::DeviceRunLengthEncode::Encode,
          bucket_indexes.get(), unique_bucket_indexes.get(),
          bucket_run_lengths.get(), bucket_runs_count.get(),
          input_indexes_count);
    } else {
      error = CUB_INVOKE_WITH_POOL(
          pool, stream, cub::DeviceRunLengthEncode::Encode,
          bucket_indexes.get(), unique_bucket_indexes.get(),
          bucket_run_lengths.get(), bucket_runs_count.get(),
          input_indexes_count, stream);
    }
    if (error != gpuSuccess) return error;
    bucket_indexes.reset();

    // compute bucket run offsets
    GpuMemory<unsigned int> bucket_run_offsets =
        GpuMemory<unsigned int>::MallocFromPoolAsync(
            extended_buckets_count_pass_one, pool, stream);
    if (dry_run) {
      error = CUB_TRY_ALLOCATE_WITH_POOL(
          pool, stream, cub::DeviceScan::ExclusiveSum, bucket_run_lengths.get(),
          bucket_run_offsets.get(), extended_buckets_count_pass_one);
    } else {
      error = CUB_INVOKE_WITH_POOL(
          pool, stream, cub::DeviceScan::ExclusiveSum, bucket_run_lengths.get(),
          bucket_run_offsets.get(), extended_buckets_count_pass_one, stream);
    }
    if (error != gpuSuccess) return error;

    if (!dry_run) {
      error = bellman::RemoveZeroBuckets(
          unique_bucket_indexes.get(), bucket_run_lengths.get(),
          bucket_runs_count.get(), extended_buckets_count_pass_one, stream);
      if (UNLIKELY(error != gpuSuccess)) return error;
    }
    bucket_runs_count.reset();

    // sort run offsets by run lengths
    // sort run indexes by run lengths
    GpuMemory<unsigned int> sorted_bucket_run_lengths =
        GpuMemory<unsigned int>::MallocFromPoolAsync(
            extended_buckets_count_pass_one, pool, stream);
    GpuMemory<unsigned int> sorted_bucket_run_offsets =
        GpuMemory<unsigned int>::MallocFromPoolAsync(
            extended_buckets_count_pass_one, pool, stream);
    GpuMemory<unsigned int> sorted_unique_bucket_indexes =
        GpuMemory<unsigned int>::MallocFromPoolAsync(
            extended_buckets_count_pass_one, pool, stream);

    if (dry_run) {
      error = CUB_TRY_ALLOCATE_WITH_POOL(
          pool, stream, cub::DeviceRadixSort::SortPairsDescending,
          bucket_run_lengths.get(), sorted_bucket_run_lengths.get(),
          bucket_run_offsets.get(), sorted_bucket_run_offsets.get(),
          extended_buckets_count_pass_one, 0, log_inputs_count + 1);
      if (error != gpuSuccess) return error;
      error = CUB_TRY_ALLOCATE_WITH_POOL(
          pool, stream, cub::DeviceRadixSort::SortPairsDescending,
          bucket_run_lengths.get(), sorted_bucket_run_lengths.get(),
          unique_bucket_indexes.get(), sorted_unique_bucket_indexes.get(),
          extended_buckets_count_pass_one, 0, log_inputs_count + 1, stream);
      if (error != gpuSuccess) return error;
    } else {
      ScopedEvent event_sort_inputs_ready =
          CreateEventWithFlags(gpuEventDisableTiming);
      RETURN_AND_LOG_IF_GPU_ERROR(
          gpuEventRecord(event_sort_inputs_ready.get(), stream),
          "Failed to gpuEventRecord()");
      RETURN_AND_LOG_IF_GPU_ERROR(
          gpuStreamWaitEvent(stream_sort_a.get(),
                             event_sort_inputs_ready.get()),
          "Failed to gpuStreamWaitEvent()");
      RETURN_AND_LOG_IF_GPU_ERROR(
          gpuStreamWaitEvent(stream_sort_b.get(),
                             event_sort_inputs_ready.get()),
          "Failed to gpuStreamWaitEvent()");
      event_sort_inputs_ready.reset();
      error = CUB_INVOKE_WITH_POOL(
          pool, stream, cub::DeviceRadixSort::SortPairsDescending,
          bucket_run_lengths.get(), sorted_bucket_run_lengths.get(),
          bucket_run_offsets.get(), sorted_bucket_run_offsets.get(),
          extended_buckets_count_pass_one, 0, log_inputs_count + 1);
      if (error != gpuSuccess) return error;
      error = CUB_INVOKE_WITH_POOL(
          pool, stream, cub::DeviceRadixSort::SortPairsDescending,
          bucket_run_lengths.get(), sorted_bucket_run_lengths.get(),
          unique_bucket_indexes.get(), sorted_unique_bucket_indexes.get(),
          extended_buckets_count_pass_one, 0, log_inputs_count + 1, stream);
      if (error != gpuSuccess) return error;
      ScopedEvent event_sort_a = CreateEventWithFlags(gpuEventDisableTiming);
      ScopedEvent event_sort_b = CreateEventWithFlags(gpuEventDisableTiming);
      RETURN_AND_LOG_IF_GPU_ERROR(
          gpuEventRecord(event_sort_a.get(), stream_sort_a.get()),
          "Failed to gpuEventRecord()");
      RETURN_AND_LOG_IF_GPU_ERROR(
          gpuEventRecord(event_sort_b.get(), stream_sort_b.get()),
          "Failed to gpuEventRecord()");
      RETURN_AND_LOG_IF_GPU_ERROR(
          gpuStreamWaitEvent(stream, event_sort_a.get()),
          "Failed to gpuStreamWaitEvent()");
      RETURN_AND_LOG_IF_GPU_ERROR(
          gpuStreamWaitEvent(stream, event_sort_b.get()),
          "Failed to gpuStreamWaitEvent()");
    }

    bucket_run_lengths.reset();
    bucket_run_offsets.reset();
    unique_bucket_indexes.reset();

    // aggregate buckets
    if (!dry_run) {
      if (copy_bases) {
        RETURN_AND_LOG_IF_GPU_ERROR(
            gpuStreamWaitEvent(stream, event_bases_loaded.get()),
            "Failed to gpuStreamWaitEvent()");
      }
      error = bellman::AggregateBuckets(
          is_first_loop, base_indexes.get(), sorted_bucket_run_offsets.get(),
          sorted_bucket_run_lengths.get(), sorted_unique_bucket_indexes.get(),
          copy_bases ? inputs_bases.get() : &ec.bases[inputs_offset],
          buckets_pass_one.get(), buckets_count_pass_one, stream);
      if (UNLIKELY(error != gpuSuccess)) return error;
      if (copy_bases) {
        RETURN_AND_LOG_IF_GPU_ERROR(
            gpuEventRecord(event_bases_free.get(), stream),
            "Failed to gpuEventRecord()");
      }
    }
    base_indexes.reset();
    sorted_bucket_run_offsets.reset();
    sorted_bucket_run_lengths.reset();
    sorted_unique_bucket_indexes.reset();
    if (is_last_loop && copy_bases) {
      inputs_bases.reset();
    }
    inputs_offset += inputs_count;
    if (!is_first_loop && log_inputs_count < log_max_inputs_count)
      log_inputs_count++;
  }

  GpuMemory<PointXYZZ<Curve>> top_buckets =
      GpuMemory<PointXYZZ<Curve>>::MallocFromPoolAsync(windows_count_pass_one,
                                                       pool, stream);

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
        error = bellman::ReduceBuckets(
            &buckets_pass_one[top_window_offset],
            1 << (signed_bits_count_pass_one - i - 1), stream);
        if (UNLIKELY(error != gpuSuccess)) return error;
      }
      error = bellman::InitializeBuckets(
          &buckets_pass_one[top_window_unused_buckets_offset],
          top_window_unused_buckets_count, stream);
      if (UNLIKELY(error != gpuSuccess)) return error;
    }
    error = bellman::ExtractTopBuckets(
        buckets_pass_one.get(), top_buckets.get(), signed_bits_count_pass_one,
        windows_count_pass_one, stream);
    if (UNLIKELY(error != gpuSuccess)) return error;
  }

  unsigned int source_bits_count = signed_bits_count_pass_one;
  unsigned int source_windows_count = windows_count_pass_one;
  GpuMemory<PointXYZZ<Curve>> source_buckets = std::move(buckets_pass_one);
  GpuMemory<PointXYZZ<Curve>> target_buckets;
  for (unsigned int i = 0;; ++i) {
    unsigned int target_bits_count = (source_bits_count + 1) >> 1;
    unsigned int target_windows_count = source_windows_count << 1;
    unsigned int target_buckets_count = target_windows_count
                                        << target_bits_count;
    unsigned int log_data_split = GetOptimalLogDataSplit(
        config.device_props.multiProcessorCount, source_bits_count,
        target_bits_count, target_windows_count);
    unsigned int total_buckets_count = target_buckets_count << log_data_split;
    target_buckets = GpuMemory<PointXYZZ<Curve>>::MallocFromPoolAsync(
        total_buckets_count, pool, stream);
    if (!dry_run) {
      error = bellman::SplitWindows(source_bits_count, source_windows_count,
                                    source_buckets.get(), target_buckets.get(),
                                    total_buckets_count, stream);
      if (UNLIKELY(error != gpuSuccess)) return error;
    }
    source_buckets.reset();

    if (!dry_run) {
      for (unsigned int j = 0; j < log_data_split; ++j) {
        error = bellman::ReduceBuckets(target_buckets.get(),
                                       total_buckets_count >> (j + 1), stream);
        if (UNLIKELY(error != gpuSuccess)) return error;
      }
    }
    if (target_bits_count == 1) {
      GpuMemory<JacobianPoint<Curve>> results;
      unsigned int result_windows_count = ScalarField::Config::kModulusBits;
      if (copy_results) {
        results = GpuMemory<JacobianPoint<Curve>>::MallocFromPoolAsync(
            result_windows_count, pool, stream);
      }
      if (!dry_run) {
        error = bellman::LastPassGather(
            bits_count_pass_one, target_buckets.get(), top_buckets.get(),
            copy_results ? results.get() : ec.results, result_windows_count,
            stream);
        if (UNLIKELY(error != gpuSuccess)) return error;
        if (copy_results) {
          RETURN_AND_LOG_IF_GPU_ERROR(
              cudaMemcpyAsync(
                  ec.results, results.get(),
                  sizeof(JacobianPoint<Curve>) * result_windows_count,
                  gpuMemcpyDeviceToHost, stream),
              "Failed to cudaMemcpyAsync()");
          if (ec.d2h_copy_finished) {
            RETURN_AND_LOG_IF_GPU_ERROR(
                gpuEventRecord(ec.d2h_copy_finished, stream),
                "Failed to gpuEventRecord()");
          }
          if (ec.d2h_copy_finished_callback) {
            RETURN_AND_LOG_IF_GPU_ERROR(
                cudaLaunchHostFunc(stream, ec.d2h_copy_finished_callback,
                                   ec.d2h_copy_finished_callback_data),
                "Failed to cudaLaunchHostFunc()");
          }
        }
      }
      if (copy_results) results.reset();
      target_buckets.reset();
      top_buckets.reset();

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
  RETURN_AND_LOG_IF_GPU_ERROR(cudaGetDevice(&device_id),
                              "Failed to cudaGetDevice()");
  cudaDeviceProp props{};
  RETURN_AND_LOG_IF_GPU_ERROR(cudaGetDeviceProperties(&props, device_id),
                              "Failed to cudaGetDeviceProperties()");
  unsigned int log_scalars_count = config.log_scalars_count;
  cudaPointerAttributes scalars_attributes{};
  cudaPointerAttributes bases_attributes{};
  cudaPointerAttributes results_attributes{};
  RETURN_AND_LOG_IF_GPU_ERROR(
      cudaPointerGetAttributes(&scalars_attributes, config.scalars),
      "Failed to cudaPointerGetAttributes()");
  RETURN_AND_LOG_IF_GPU_ERROR(
      cudaPointerGetAttributes(&bases_attributes, config.bases),
      "Failed to cudaPointerGetAttributes()");
  RETURN_AND_LOG_IF_GPU_ERROR(
      cudaPointerGetAttributes(&results_attributes, config.results),
      "Failed to cudaPointerGetAttributes()");
  bool copy_scalars = scalars_attributes.type == gpuMemoryTypeUnregistered ||
                      scalars_attributes.type == gpuMemoryTypeHost;
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

}  // namespace tachyon::math::bellman

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_BELLMAN_BELLMAN_MSM_IMPL_H_
