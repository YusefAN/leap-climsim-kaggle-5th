"""
download_data.py

This script downloads both parquet-processed files for both low-res and low-res acqua planet datasets.

- Ensure you have the Kaggle API installed and are authenticated.
- Datasets are private so access is needed for these.
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

def get_kaggle_file_names(dataset):
    result = subprocess.run(
        ["kaggle", "datasets", "files", dataset],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Error:", result.stderr)
        return []

    return [line.split()[0] for line in result.stdout.splitlines()[2:] if line]


lowres_datasets = [
    "marecserlin/leap-001-002",
    'ajobseeker/leap-003-part1',
    'ajobseeker/leap-003-part2',
    'ajobseeker/leap-004-part1',
    'ajobseeker/leap-004-part2',
    'ajobseeker/leap-005-part1',
    'ajobseeker/leap-005-part2',
    'marecserlin/leap-0006-0007-0008',
    
]

aqua_datasets = [
    'yusefkaggle/leap-ocean'
]

if __name__ == '__main__':
    os.makedirs('./raw_lowres', exist_ok=True)
    commands = [f"kaggle datasets download {data} -p ./raw_lowres --unzip" for data in lowres_datasets]


    os.makedirs('./raw_aqua', exist_ok=True)
    commands = [f"kaggle datasets download {data} -p ./raw_aqua --unzip" for data in aqua_datasets]


    def run_command(command):
        subprocess.run(command, shell=True)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(run_command, commands)