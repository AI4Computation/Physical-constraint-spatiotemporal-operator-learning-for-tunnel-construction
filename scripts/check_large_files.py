#!/usr/bin/env python3
import os
import subprocess
import sys

# 默认阈值为 30MB，可通过环境变量 MAX_FILE_SIZE_MB 覆盖。
MAX_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "30"))
MAX_BYTES = MAX_MB * 1024 * 1024


def staged_files() -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    oversized = []
    for path in staged_files():
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        if size > MAX_BYTES:
            oversized.append((path, size))

    if not oversized:
        return 0

    print(f"错误：暂存区存在超过 {MAX_MB}MB 的文件：")
    for path, size in oversized:
        mb = size / (1024 * 1024)
        print(f"  - {path}: {mb:.2f} MB")

    print("建议：压缩文件、使用 Git LFS，或临时调大 MAX_FILE_SIZE_MB。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
