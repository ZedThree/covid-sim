#!/usr/bin/env python3

from collections.abc import Iterable
from numbers import Number
from pathlib import Path
from typing import Union
import argparse
import hashlib
import re
import subprocess
import numpy as np
import io
from data_pipeline_api.standard_api import StandardAPI


# Wrap API
class CovidSimAPI:
    """Wrapper to handle translation between CovidSim and StandardAPI

    """

    def __init__(
        self,
        config_filename: Union[Path, str],
        input_filename: str,
        output_location: str,
    ):
        self.api = StandardAPI(config_filename)
        self.input_filename = Path(input_filename)
        self.output_location = Path(output_location)

        try:
            self.input_filename.unlink(missing_ok=True)
        except IsADirectoryError as e:
            raise RuntimeError("Input file must not be a directory") from e

    def write_input(self, name: str, data: Union[Iterable, Number]):
        """Write the (name, data) as a CovidSim input file
        """
        with open(self.input_filename, "a") as f:
            f.write(f"[{name}]")
            if isinstance(data, Iterable):
                f.write(" ".join(data))
            else:
                f.write(f"{data}")

    def read_estimate(self, data_product: str, translation: str):
        data = self.api.read_estimate(data_product)
        self.write_input(translation, data)

    def write_estimate(self, data_product: str, translation: str):
        # read output_location
        with open(self.output_location, "r") as f:
            contents = f.read()

        value = contents.find(translation)

        self.api.write_estimate(data_product, value)


PARAMETER_FILE_KEY_VALUE_RE = re.compile(r"(\[[^]]+\])([^[]+)", re.DOTALL)


def read_base_parameters(filename: str) -> dict:
    """Parse a CovidSim input file into a dict

    """

    with open(filename, "r") as f:
        contents = f.read()

    params = {}
    for key, value in PARAMETER_FILE_KEY_VALUE_RE.findall(contents):
        params[key] = np.genfromtxt(io.StringIO(value), dtype=None, encoding=None)

        # If it wasn't a simple number or array of numbers, then
        # either there was a string in there, or it's interpreted one
        # number as an integer, and now we've got a tuple
        if not np.issubdtype(params[key].dtype, np.number):
            if re.search(r"[a-zA-Z]", value) is not None:
                params[key] = value.strip()
            else:
                params[key] = np.genfromtxt(io.StringIO(value))

    return params


def write_parameter_file(filename: str, parameters: dict):
    """Write a dict as a CovidSim input file
    """
    with open(filename, "w") as f:
        for k, v in parameters.items():
            if isinstance(v, np.ndarray):
                v_str = re.sub(r"[\[\]]", " ", np.array_str(v, max_line_width=1e8))
            else:
                v_str = str(v)
            f.write(f"{k}\n{v_str}\n\n")


def run_covid_sim(
    covidsim, preparams_file, output_path, population_file, r, threads=2, seeds=None
):
    if seeds is None:
        seeds = [98798150, 729101, 17389101, 4797132]

    command = [
        f"{covidsim}",
        f"/c:{threads}",
        "/BM:bmp",
        f"/PP:{preparams_file}",
        "/P:empty.txt",
        f"/O:{output_path}",
        f"/D:{population_file}",
        f"/R:{r}",
    ] + list(map(str, seeds))

    print(" ".join(command))
    subprocess.run(command, check=True)


# Command line: /home/peter/Codes/Covid/covid-sim/build/src/CovidSim /c:2 /BM:bmp /PP:pre-params.txt /P:input-noint-params.txt /O:results-noint /D:pop.txt /M:pop.bin /A:admin-params.txt /S:network.bin /R:1.5 98798150 729101 17389101 4797132


def main(
    output_dir: Union[str, Path],
    covidsim: Union[str, Path],
    base_parameter_file: Union[str, Path],
    population_file: Union[str, Path],
    r: float,
    threads: int,
):
    # Read the data from the API
    # Write the data to the expected input file
    # Run the simulation
    # Read in the output files
    # Write the data to the API

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir()
    output_name = output_dir / "results"

    run_covid_sim(
        covidsim, base_parameter_file, output_name, population_file, r, threads
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("SCRC driver for CovidSim")
    parser.add_argument("covidsim", help="Path to covidsim executable", type=str)
    parser.add_argument(
        "--threads", "-j", help="Number of threads to run on", default=2, type=int
    )
    parser.add_argument(
        "--base",
        help="Base input file to use",
        default="combined_admin_pre_noint_params.txt",
    )
    parser.add_argument(
        "--output-dir", "-o", help="Output directory", type=str, default="results"
    )
    parser.add_argument("--population", "-p", help="Population file", type=str)
    parser.add_argument(
        "--reproduction-rate", "-r", help="Reproduction rate", type=float
    )

    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        covidsim=args.covidsim,
        base_parameter_file=args.base,
        population_file=args.population,
        threads=args.threads,
        r=args.reproduction_rate,
    )
