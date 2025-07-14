import os
import shutil
import subprocess
import sys
import time

sys.path.append(os.getcwd())

# Empty the output directory
shutil.rmtree("trash/benchmark", ignore_errors=True)
os.makedirs("trash/benchmark", exist_ok=True)

run_options = [
    {
        "type": "multi",
        "threads": 5,
        "batch_processing_size": 10,
    },
    {
        "type": "multi",
        "threads": 10,
        "batch_processing_size": 10,
    },
    {
        "type": "multi",
        "threads": 10,
        "batch_processing_size": 20,
    },
    {
        "type": "multi",
        "threads": 10,
        "batch_processing_size": 40,
    },
    {
        "type": "multi",
        "threads": 5,
        "batch_processing_size": 80,
    },
    {"type": "single"},
]

for options in run_options:
    print(f"Running with options: {options}")

    # Start the server
    server = subprocess.Popen(
        [
            "python",
            "start.py",
            "--batch",
            "--batch_processing_size",
            str(options.get("batch_processing_size", 10)),
            "--threads",
            str(options.get("threads", 1)),
        ]
    )

    time.sleep(3)

    # Start the edge server
    subprocess.Popen(
        [
            "python",
            "app/edge_device/first.py",
        ]
    )

    # Wait for the server to finish
    server.wait()

    print("Server finished")

    time.sleep(3)
