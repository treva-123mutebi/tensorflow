/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_PJRT_EVENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_PJRT_EVENT_H_

#include <functional>
#include <utility>

#include "absl/types/span.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace xla {

template <class T>
class PjRtEvent {
 public:
  struct ProfilingKeys {
    uint64_t traceme_context_id = -1;
  };
  using OnBlockStartFn = std::function<ProfilingKeys()>;
  using OnBlockEndFn = std::function<void(ProfilingKeys)>;
  explicit PjRtEvent(T t)
      : host_ctx_(nullptr),
        event_(tfrt::MakeAvailableAsyncValueRef<T>(t)),
        on_block_start_([]() { return ProfilingKeys(); }),
        on_block_end_([](ProfilingKeys) {}) {}

  explicit PjRtEvent(tfrt::HostContext* host_ctx, tfrt::AsyncValueRef<T> event,
                     OnBlockStartFn on_block_start, OnBlockEndFn on_block_end)
      : host_ctx_(host_ctx),
        event_(std::move(event)),
        on_block_start_(std::move(on_block_start)),
        on_block_end_(std::move(on_block_end)) {}

  T BlockHostUntilReady() {
    if (!event_.IsAvailable()) {
      host_ctx_->Await({event_.CopyRCRef()});
    }
    DCHECK(event_.IsConcrete());
    return *event_;
  }

  void OnReady(std::function<void(T)> callback) {
    event_.AndThen(
        [event = event_.CopyRef(), callback = std::move(callback)]() {
          DCHECK(event.IsConcrete());
          callback(*event);
        });
  }

  static tfrt::AsyncValueRef<T> MakeUnconstructedAVR() {
    return tfrt::MakeUnconstructedAsyncValueRef<T>();
  }

 private:
  tfrt::HostContext* host_ctx_;  // not owned
  tfrt::AsyncValueRef<T> event_;
  OnBlockStartFn on_block_start_;
  OnBlockEndFn on_block_end_;
};

std::unique_ptr<tfrt::HostContext> CreateHostContextForPjRtEvent();

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_EVENT_H_
