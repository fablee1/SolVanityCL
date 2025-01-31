import json
import logging
import os
import platform
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from multiprocessing.pool import Pool

import pyopencl as cl

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
os.environ["PYOPENCL_NO_CACHE"] = "TRUE"
from pathlib import Path

import click
import numpy as np
from base58 import b58decode, b58encode
from nacl.signing import SigningKey

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")


class HostSetting:
    def __init__(self, kernel_source: str, iteration_bits: int) -> None:
        self.iteration_bits = iteration_bits
        self.iteration_bytes = np.ubyte(ceil(iteration_bits / 8))
        self.global_work_size = 1 << iteration_bits
        self.key32 = self.generate_key32()

        self.kernel_source = kernel_source

    def generate_key32(self):
        token_bytes = (
            secrets.token_bytes(32 - self.iteration_bytes)
            + b"\x00" * self.iteration_bytes
        )
        key32 = np.array([x for x in token_bytes], dtype=np.ubyte)
        return key32

    def increase_key32(self):
        current_number = int(bytes(self.key32).hex(), base=16)
        next_number = current_number + (1 << self.iteration_bits)
        _number_bytes = next_number.to_bytes(32, "big")
        new_key32 = np.array([x for x in _number_bytes], dtype=np.ubyte)
        carry_index = 0 - self.iteration_bytes
        if (new_key32[carry_index] < self.key32[carry_index]) and new_key32[
            carry_index
        ] != 0:
            new_key32[carry_index] = 0

        self.key32[:] = new_key32


def check_character(name: str, character: str):
    try:
        b58decode(character)
    except ValueError as e:
        logging.error(f"{str(e)} in {name}")
        sys.exit(1)
    except Exception as e:
        raise e


def get_kernel_source(ends_with: list, cl):
    with open(Path("opencl/kernel.cl"), "r") as f:
        source_lines = f.readlines()

    suffix_offsets = []
    suffix_lengths = []
    all_suffix_bytes = []

    current_offset = 0
    for sfx in ends_with:
        sb = list(sfx.encode())
        suffix_lengths.append(len(sb))
        suffix_offsets.append(current_offset)
        all_suffix_bytes.extend(sb)
        current_offset += len(sb)

    for i, s in enumerate(source_lines):
        if s.startswith("constant int SUFFIX_COUNT"):
            source_lines[i] = (
                f"constant int SUFFIX_COUNT = {len(ends_with)};\n"
            )
        if s.startswith("constant int SUFFIX_LENGTHS[]"):
            source_lines[i] = f"constant int SUFFIX_LENGTHS[] = {{{', '.join(map(str, suffix_lengths))}}};\n"
        if s.startswith("constant int SUFFIX_OFFSETS[]"):
            source_lines[i] = f"constant int SUFFIX_OFFSETS[] = {{{', '.join(map(str, suffix_offsets))}}};\n"
        if s.startswith("constant uchar SUFFIXES[]"):
            source_lines[i] = f"constant uchar SUFFIXES[] = {{{', '.join(map(str, all_suffix_bytes))}}};\n"

    source_str = "".join(source_lines)

    if "NVIDIA" in str(cl.get_platforms()) and platform.system() == "Windows":
        source_str = source_str.replace("#define __generic\n", "")

    if cl.get_cl_header_version()[0] != 1 and platform.system() != "Windows":
        source_str = source_str.replace("#define __generic\n", "")

    return source_str


def get_all_gpu_devices():
    
    devices = [
        device
        for platform in cl.get_platforms()
        for device in platform.get_devices(device_type=cl.device_type.GPU)
    ]
    return [d.int_ptr for d in devices]


def single_gpu_init(context, setting):
    searcher = Searcher(
        kernel_source=setting.kernel_source,
        index=0,
        setting=setting,
        context=context,
    )
    return [searcher.find()]


def multi_gpu_init(index: int, setting: HostSetting):
    # get all platforms and devices
    try:
        searcher = Searcher(
            kernel_source=setting.kernel_source,
            index=index,
            setting=setting,
        )

        return searcher.find()
    except Exception as e:
        logging.exception(e)
    return [0]


def save_result(outputs, output_dir):
    result_count = 0
    for output in outputs:
        if not output[0]:
            continue
        result_count += 1
        pv_bytes = bytes(output[1:])
        pv = SigningKey(pv_bytes)
        pb_bytes = bytes(pv.verify_key)
        pubkey = b58encode(pb_bytes).decode()

        logging.info(f"Found: {pubkey}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, f"{pubkey}.json").write_text(
            json.dumps(list(pv_bytes + pb_bytes))
        )
    return result_count


class Searcher:
    def __init__(
        self, *, kernel_source, index: int, setting: HostSetting, context=None
    ):
        device_ids = get_all_gpu_devices()
        # context and command queue
        if context:
            self.context = context
            self.gpu_chunks = 1
        else:
            self.context = cl.Context(
                [cl.Device.from_int_ptr(device_ids[index])],
            )
            self.gpu_chunks = len(device_ids)
        self.command_queue = cl.CommandQueue(self.context)

        self.setting = setting
        self.index = index

        # build program and kernel
        program = cl.Program(self.context, kernel_source).build()
        self.program = program
        self.kernel = cl.Kernel(program, "generate_pubkey")

    def filter_valid_result(self, outputs):
        valid_outputs = []
        for output in outputs:
            if not output[0]:
                continue
            valid_outputs.append(output)
        return valid_outputs

    def find(self):
        memobj_key32 = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            32 * np.ubyte().itemsize,
            hostbuf=self.setting.key32,
        )
        memobj_output = cl.Buffer(
            self.context, cl.mem_flags.READ_WRITE, 33 * np.ubyte().itemsize
        )

        memobj_occupied_bytes = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.array([self.setting.iteration_bytes]),
        )
        memobj_group_offset = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.array([self.index]),
        )
        output = np.zeros(33, dtype=np.ubyte)
        self.kernel.set_arg(0, memobj_key32)
        self.kernel.set_arg(1, memobj_output)
        self.kernel.set_arg(2, memobj_occupied_bytes)
        self.kernel.set_arg(3, memobj_group_offset)

        st = time.time()
        global_worker_size = self.setting.global_work_size // self.gpu_chunks
        cl.enqueue_nd_range_kernel(
            self.command_queue,
            self.kernel,
            (global_worker_size * 20,),
            None,
        )
        cl._enqueue_read_buffer(self.command_queue, memobj_output, output).wait()
        logging.info(
            f"GPU {self.index} Speed: {(global_worker_size * 20)/ ((time.time() - st) * 10**6):.2f} MH/s"
        )

        return output


@click.group()
def cli():
    pass


@cli.command(context_settings={"show_default": True})
@click.option(
    "--ends-with",
    type=str,
    help="Public key ends with the indicated suffix.",
    default="",
)
@click.option(
    "--count",
    type=int,
    help="Count of pubkeys to generate.",
    default=1,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Output directory.",
    default="./",
)
@click.option(
    "--iteration-bits",
    type=int,
    help="Number of the iteration occupied bits. Recommended 24, 26, 28, 30, 32. The larger the bits, the longer it takes to complete an iteration.",
    default=24,
)
@click.pass_context
def search_pubkey(
    ctx,
    ends_with: str,
    count: int,
    output_dir: str,
    iteration_bits: int,
):
    """Search Solana vanity pubkey"""

    if not ends_with:
        print("Please provides at least [starts with] or [ends with]\n")
        click.echo(ctx.get_help())
        sys.exit(1)

    suffix_list = []
    if ends_with.strip():
        suffix_list = [s.strip() for s in ends_with.split(",") if s.strip()]

    for sf in suffix_list:
        check_character("one-of-the-suffixes", sf)

    logging.info(
        f"Searching Solana pubkey that ends with '{suffix_list}'"
    )
    with Pool() as pool:
        gpu_counts = len(pool.apply(get_all_gpu_devices))

    kernel_source = get_kernel_source(suffix_list, cl)
    result_count = 0

    logging.info(f"Searching with {gpu_counts} OpenCL devices")

    with Pool(processes=gpu_counts) as pool:
        while result_count < count:
            results = pool.starmap(
                multi_gpu_init, [(x, HostSetting(kernel_source, iteration_bits)) for x in range(gpu_counts)]
            )
            result_count += save_result(results, output_dir)
            time.sleep(0.1)


@cli.command(context_settings={"show_default": True})
def show_device():
    """Show OpenCL devices"""

    platforms = cl.get_platforms()

    for p_index, platform_ in enumerate(platforms):
        print(f"Platform {p_index}: {platform_.name}")

        devices = platform_.get_devices()

        for d_index, device in enumerate(devices):
            print(f"- Device {d_index}: {device.name}")


if __name__ == "__main__":
    cli()
