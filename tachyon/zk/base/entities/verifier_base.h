// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_ENTITIES_VERIFIER_BASE_H_
#define TACHYON_ZK_BASE_ENTITIES_VERIFIER_BASE_H_

#include <memory>
#include <utility>

#include "tachyon/base/logging.h"
#include "tachyon/zk/base/entities/entity.h"

namespace tachyon::zk {

template <typename PCS>
class VerifierBase : public Entity<PCS> {
 public:
  using TranscriptReader = typename PCS::TranscriptReader;
  using TranscriptWriter = typename PCS::TranscriptWriter;

  VerifierBase(PCS&& pcs, std::unique_ptr<TranscriptReader> reader)
      : Entity<PCS>(std::move(pcs)), reader_(std::move(reader)) {}

  TranscriptReader* GetReader() const override { return reader_.get(); }

  TranscriptWriter* GetWriter() const override {
    NOTREACHED();
    return nullptr;
  }

 protected:
  std::unique_ptr<TranscriptReader> reader_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_ENTITIES_VERIFIER_BASE_H_
