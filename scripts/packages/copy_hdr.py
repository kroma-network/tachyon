#!/usr/bin/env python3

import argparse
import os
import shutil
import sys


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('file', help='The file to replace with')
    argument_parser.add_argument('dst', help='The destination where the file moves to')
    argument_parser.add_argument('--strip', help='The include path to be stripped')
    args = argument_parser.parse_args()

    dst_file = args.file
    if len(args.strip) > 0:
        dst_file = dst_file[len(args.strip):]

    dst_file = os.path.join(args.dst, dst_file)
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    shutil.copyfile(args.file, dst_file)

    return 0


if __name__ == '__main__':
    sys.exit(main())
