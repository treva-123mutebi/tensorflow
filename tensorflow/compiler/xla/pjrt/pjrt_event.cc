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

#include "tensorflow/compiler/xla/pjrt/pjrt_event.h"

#include <string>

#include "absl/synchronization/mutex.h"
#include "llvm/ADT/None.h"
#include "tensorflow/core/platform/logging.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace xla {

namespace {

// This class implements a dummy work queue that is only used for awaiting
// events.
class AwaitOnlyWorkQueue : public tfrt::ConcurrentWorkQueue {
 public:
  std::string name() const override { return "await-only"; }

  void AddTask(tfrt::TaskFunction work) override {
    LOG(FATAL) << "Work Queue may only be used for Await";
  }

  tfrt::Optional<tfrt::TaskFunction> AddBlockingTask(
      tfrt::TaskFunction work, bool allow_queuing) override {
    LOG(FATAL) << "Work Queue may only be used for Await";
    return llvm::None;
  }

  bool IsInWorkerThread() const override {
    LOG(FATAL) << "Work Queue may only be used for Await";
    return false;
  }

  int GetParallelismLevel() const override {
    LOG(FATAL) << "Work Queue may only be used for Await";
    return false;
  }

  void Await(
      tfrt::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> values) override {
    absl::Mutex mu;
    int values_remaining = values.size();

    // As each value becomes available, we can decrement our counts.
    for (auto& value : values) {
      value->AndThen([&mu, &values_remaining]() {
        {
          absl::MutexLock l(&mu);
          --values_remaining;
        }
      });
    }

    auto done = [&mu, &values_remaining]() {
      mu.AssertHeld();
      return values_remaining == 0;
    };
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(&done));
  }

  void Quiesce() override {}
};

}  // namespace

PjRtEventContext PjRtEventContext::Create() {
  return PjRtEventContext(std::make_unique<tfrt::HostContext>(
      [](const tfrt::DecodedDiagnostic& diag) {
        LOG(ERROR) << "Encountered runtime error: " << diag.message << "\n";
      },
      tfrt::CreateMallocAllocator(), std::make_unique<AwaitOnlyWorkQueue>()));
}

}  // namespace xla
