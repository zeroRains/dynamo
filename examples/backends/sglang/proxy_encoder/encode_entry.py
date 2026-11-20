import runpy

import patch_qwen  # noqa: F401  must be imported before dynamo.sglang

if __name__ == "__main__":
    runpy.run_module("dynamo.sglang", run_name="__main__", alter_sys=True)
