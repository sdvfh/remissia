import os
from pathlib import Path

import psutil
from pyspark.sql import SparkSession


def get_spark():
    total_memory = int(psutil.virtual_memory().total / (1024**3) * 0.9)
    compressed_memory = min(int(0.9 * total_memory), 31)
    spark = (
        SparkSession.builder.config("spark.rdd.compress", "true")
        .config("spark.shuffle.compress", "true")
        .config("spark.memory.storageFraction", "0.30")
        .config("spark.memory.fraction", "	0.85")
        .config("spark.default.parallelism", str(int(os.cpu_count() * 2)))
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.dynamicAllocation.enabled", "false")
        .config("spark.sql.shuffle.partitions", "20")
        .config(
            "spark.driver.extraJavaOptions",
            f"-XX:InitialHeapSize=1g -XX:MaxHeapSize={compressed_memory}g -XX:+UseCompressedOops"
            " -XX:+UseG1GC -XX:+UnlockDiagnosticVMOptions -XX:InitiatingHeapOccupancyPercent=40"
            " -XX:OnOutOfMemoryError='kill -9 %p'",
        )
        .config(
            "spark.executor.extraJavaOptions",
            f"-XX:InitialHeapSize=1g -XX:MaxHeapSize={compressed_memory}g -XX:+UseCompressedOops"
            " -XX:+UseG1GC -XX:+UnlockDiagnosticVMOptions -XX:InitiatingHeapOccupancyPercent=40"
            " -XX:OnOutOfMemoryError='kill -9 %p'",
        )
        .config("spark.executor.memoryOverhead", f"{int(0.1 * total_memory)}g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.storage.level", "MEMORY_AND_DISK_SER")
        .config("spark.shuffle.spill.compress", "true")
        .config("spark.executors.memory", f"{int(0.9 * total_memory)}g")
        .config("spark.driver.memory", f"{int(0.9 * total_memory)}g")
        .getOrCreate()
    )
    return spark


def get_path():
    path = {"root": Path(__file__, "../..").resolve()}
    path["queries"] = path["root"] / "queries"
    path["datasets"] = path["root"] / "datasets"
    path["files"] = path["root"] / "files"
    return path
