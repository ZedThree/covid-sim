#!/usr/bin/env python3

from pathlib import Path
from typing import Union
import argparse
import hashlib
import io
import re
import subprocess

import numpy as np
import pandas as pd

from data_pipeline_api.standard_api import StandardAPI


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
    covidsim,
    preparams_file,
    params_file,
    output_path,
    population_file,
    r,
    threads=2,
    seeds=None,
):
    """Wrap calling the covidsim executable

    """

    if seeds is None:
        seeds = [98798150, 729101, 17389101, 4797132]

    command = [
        f"{covidsim}",
        f"/c:{threads}",
        "/BM:bmp",
        f"/PP:{preparams_file}",
        f"/P:{params_file}",
        f"/O:{output_path}",
        f"/D:{population_file}",
        f"/R:{r}",
    ] + list(map(str, seeds))

    print(" ".join(command))
    subprocess.run(command, check=True)


def main(
    output_dir: Union[str, Path],
    covidsim: Union[str, Path],
    base_parameter_file: Union[str, Path],
    population_file: Union[str, Path],
    r: float,
    threads: int,
    config_file: str,
):
    """Main driver for covidsim

    - Read the data from the API
    - Write the data to the expected input file
    - Run the simulation
    - Read in the output files
    - Write the data to the API

    """

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir()
    output_name = output_dir / "results"

    parameters = read_base_parameters(base_parameter_file)

    api = StandardAPI(config_file)
    parameters["[Initial number of infecteds]"] = int(
        api.read_estimate("human/infected/number", "initial-infected-number")
    )

    preparams_file = output_dir / "preparams_file.txt"
    write_parameter_file(preparams_file, parameters)

    # Ensure the parameter file exists
    params_file = output_dir / "empty.txt"
    open(params_file, "a").close()

    run_covid_sim(
        covidsim, preparams_file, params_file, output_name, population_file, r, threads
    )

    for output in ["age", "severity.adunit", "severity.age", "severity"]:
        output_file = output_dir / f"results.avNE.{output}.xls"
        output_table = pd.read_csv(
            output_file, sep=r"\s+", skipfooter=1, engine="python"
        )
        api.write_table(output, output, output_table)
        with open(output_file, "rb") as f:
            sha = hashlib.sha512(f.read()).hexdigest()
            print(f"{output_file}\t{sha}")


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
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Datapipeline config file",
        default="config.yaml",
    )

    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        covidsim=args.covidsim,
        base_parameter_file=args.base,
        population_file=args.population,
        threads=args.threads,
        r=args.reproduction_rate,
        config_file=args.config,
    )
