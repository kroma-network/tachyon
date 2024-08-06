#include "tachyon/base/profiler.h"

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace tachyon::base {

Profiler::Profiler() : Profiler(Options{}) {}

Profiler::Profiler(const Options& options)
    : trace_filepath_(options.output_path),
      trace_file_(trace_filepath_,
                  base::File::FLAG_CREATE_ALWAYS | base::File::FLAG_WRITE),
      max_size_kb_(options.max_size_kb) {}

Profiler::~Profiler() { Stop(); }

void Profiler::Init() {
  perfetto::TracingInitArgs args;
  args.backends |= perfetto::kInProcessBackend;
  perfetto::Tracing::Initialize(args);
  perfetto::TrackEvent::Register();
}

void Profiler::DisableCategories(std::string_view category) {
  track_event_cfg_.add_disabled_categories(std::string(category));
}

void Profiler::EnableCategories(std::string_view category) {
  track_event_cfg_.add_enabled_categories(std::string(category));
}

void Profiler::Start() {
  perfetto::TraceConfig cfg;
  cfg.add_buffers()->set_size_kb(max_size_kb_);

  auto* ds_cfg = cfg.add_data_sources()->mutable_config();
  ds_cfg->set_name("track_event");
  ds_cfg->set_track_event_config_raw(track_event_cfg_.SerializeAsString());

  tracing_session_ = perfetto::Tracing::NewTrace();
  tracing_session_->Setup(cfg, trace_file_.GetPlatformFile());
  tracing_session_->StartBlocking();
}

void Profiler::Stop() { tracing_session_->StopBlocking(); }

}  // namespace tachyon::base
