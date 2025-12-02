# # Copyright (C) KonaAI - All Rights Reserved
"""This module generates transaction data for invoices"""
import asyncio
import pathlib
import time
from uuid import uuid4

import dask.dataframe as dd
import humanize
from invoices_data_generation import generate_invoice_dataframe


root_path = pathlib.Path(__file__).resolve().parent
data_path = pathlib.Path(root_path, "data")
data_path.mkdir(parents=True, exist_ok=True)
print(f"Data will be saved to: {data_path}")


async def generate_invoice_samples(records: int):
    """Generate a sample of invoice data."""
    X = generate_invoice_dataframe(records=records)
    file_name = f"invoice_data_{uuid4()}.parquet"
    file_path = pathlib.Path(data_path, file_name)
    X.to_parquet(file_path)


async def run():
    """Run the invoice data generation asynchronously."""
    tasks = [generate_invoice_samples(records=10000) for _ in range(500)]
    await asyncio.gather(*tasks)

    df = dd.read_parquet(data_path)
    print(f"Total records generated: {len(df)}")
    size = df.memory_usage(deep=True).sum().compute()
    print(f"Total size of data: {humanize.naturalsize(size, binary=True)}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(run())
    end_time = time.perf_counter()
    print(f"Total time taken: {humanize.naturaldelta(end_time - start_time)}")
