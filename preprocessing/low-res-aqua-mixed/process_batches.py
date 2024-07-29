"""
process_batches.py

Processes the parquet files for lowres and lowres aqua into batches of 1000 rows in Torch format.
Files specified use the same format/columns as in the train data.
"""


import os
import polars as pl
import torch

def process_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_files = [
        os.path.join(dirname, filename)
        for dirname, _, filenames in os.walk(input_dir)
        for filename in filenames
        if 'parquet' in filename
    ]

    pt_num = 0
    index_num = 0

    for f in all_files:
        try:
            df = pl.read_parquet(f).drop('sample_id')
            os.remove(f)
            print(f"Removing {f}")
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

        for idx, frame in enumerate(df.iter_slices(n_rows=1000)):
            data = frame.to_torch('tensor', dtype=pl.Float64)
            if data.shape[0] == 1000:
                torch.save(data, f'{output_dir}/{idx}{index_num}.pt')
                index_num += 1
                pt_num += 1
                if pt_num % 1000 == 0:
                    print(pt_num)

process_files('./raw_lowres/', './lowres_torch')
# process_files('./raw_aqua/', './ocean_torch')
