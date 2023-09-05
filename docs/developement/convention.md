# Convention

## GPU

### Naming

- If your gpu codes can be run independent to GPU vendors, please use a word `Gpu` not `Cuda` or `ROCm`.

### File Suffixes

- If a file contains kernels, there will be suffixes `.cu.xx` despite it can be run on `ROCm`.
- If a file contains device only codes, there will be suffixes `.cu.xx` despite it can be run on `ROCm`.
- If a file contains host and device codes, then suffixes are not appended.
  For examples, see files under `tachyon/devices/gpu`.
