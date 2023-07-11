load("@bazel_tools//tools/cpp:unix_cc_configure.bzl", "find_cc")

# This is taken from tensorflow/third_party/remote_config/common.bzl.
def execute(
        repository_ctx,
        cmdline,
        error_msg = None,
        error_details = None,
        allow_failure = False):
    """Executes an arbitrary shell command.

    Args:
      repository_ctx: the repository_ctx object
      cmdline: list of strings, the command to execute
      error_msg: string, a summary of the error if the command fails
      error_details: string, details about the error or steps to fix it
      allow_failure: bool, if True, an empty stdout result or output to stderr
        is fine, otherwise either of these is an error
    Returns:
      The result of repository_ctx.execute(cmdline)
    """
    result = raw_exec(repository_ctx, cmdline)
    if (result.stderr or not result.stdout) and not allow_failure:
        fail(
            "\n".join([
                error_msg.strip() if error_msg else "Repository command failed",
                result.stderr.strip(),
                error_details if error_details else "",
            ]),
        )
    return result

def raw_exec(repository_ctx, cmdline):
    """Executes a command via repository_ctx.execute() and returns the result.

    This method is useful for debugging purposes. For example, to print all
    commands executed as well as their return code.

    Args:
      repository_ctx: the repository_ctx
      cmdline: the list of args

    Returns:
      The 'exec_result' of repository_ctx.execute().
    """
    return repository_ctx.execute(cmdline)

def symlink_dir(repository_ctx, src_dir, dest_dir, src_files = []):
    files = repository_ctx.path(src_dir).readdir()
    for src_file in files:
        if len(src_files) == 0 or src_file.basename in src_files:
            repository_ctx.symlink(src_file, dest_dir + "/" + src_file.basename)

def symlink_dir_with_prefix(repository_ctx, src_dir, dest_dir, prefix):
    files = repository_ctx.path(src_dir).readdir()
    for src_file in files:
        if src_file.basename.startswith(prefix):
            repository_ctx.symlink(src_file, dest_dir + "/" + src_file.basename)

def _is_wellknown_vendor(vendor):
    return vendor == "unknown" or vendor == "pc"

def _is_wellknown_os(os):
    return os == "linux" or \
           os == "windows" or \
           os == "darwin" or \
           os == "illumos" or \
           os == "netbsd" or \
           os == "freebsd" or \
           os == "fuchsia" or \
           os == "android"

def _get_target_triple(repository_ctx):
    cc = find_cc(repository_ctx, overriden_tools = {})
    result = execute(repository_ctx, [cc, "-dumpmachine"]).stdout.strip().split("-")
    machine = ""
    vendor = ""
    os = ""
    if len(result) > 0:
        machine = result[0]
        result.pop(0)
    if len(result) > 0:
        if _is_wellknown_vendor(result[0]) or not _is_wellknown_os(result[0]):
            vendor = result[0]
            result.pop(0)
    os = "-".join(result[0:])
    return struct(
        machine = machine,
        vendor = vendor,
        os = os,
    )

def get_usr_lib_path_with_machine(repository_ctx):
    target_triple = _get_target_triple(repository_ctx)
    return "/usr/lib/%s-%s" % (target_triple.machine, target_triple.os)

def get_usr_include_path_with_machine(repository_ctx):
    target_triple = _get_target_triple(repository_ctx)
    return "/usr/include/%s-%s" % (target_triple.machine, target_triple.os)
