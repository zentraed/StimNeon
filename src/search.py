#!/usr/bin/env python3

import re
import sys
from pathlib import Path

def find_bad_spanref_usage_in_file(filepath):
    with open(filepath, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        l = line.strip()
        # 1. Construction with temporary vector
        if re.search(r'CircuitInstruction\s*\(.*std::vector\s*<', l):
            print(f"{filepath}:{idx+1}: Potential BAD: CircuitInstruction constructed with temporary std::vector: {l}")
        # 2. Construction with local variable
        if re.search(r'CircuitInstruction\s*\(\s*[^,()]+,\s*[^,()]+,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[,)]', l):
            var = re.search(r'CircuitInstruction\s*\(\s*[^,()]+,\s*[^,()]+,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[,)]', l).group(1)
            # Try to filter out known good ones (e.g., .targets, target_buf etc); you may want to improve this.
            # You may wish to refine which variables are ok
            if not (".targets" in var or "target_buf" in var or "take_copy" in l):
                print(f"{filepath}:{idx+1}: Maybe BAD: CircuitInstruction targets from variable '{var}': {l}")
        # 3. Direct SpanRef from std::vector
        if re.search(r'SpanRef\s*<.*>\s*$[a-zA-Z_][a-zA-Z0-9_]*$', l) and "take_copy" not in l:
            print(f"{filepath}:{idx+1}: SpanRef constructed from variable: {l}")
        # 4. SpanRef from std::vector temporary
        if re.search(r'SpanRef\s*<.*>\s*\(\s*std::vector', l):
            print(f"{filepath}:{idx+1}: SpanRef constructed from temporary std::vector: {l}")

def walk_and_find_bad_spanref_usage(root='.'):
    for ext in ['*.cc', '*.cpp', '*.h', '*.hpp', '*.cxx', '*.tcc']:
        for file in Path(root).rglob(ext):
            find_bad_spanref_usage_in_file(str(file))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for fn in sys.argv[1:]:
            find_bad_spanref_usage_in_file(fn)
    else:
        walk_and_find_bad_spanref_usage()