""" Module to sequentially execute from feature engineering to evaluation step
"""
import subprocess
import os
import logging


logging.basicConfig(level=logging.INFO)


def check_files_exist():
    """Check if the files needed exist. Returns files location."""
    cwd = os.getcwd()
    assert os.path.isfile(cwd + "/src/preprocess/pre_process.py")
    assert os.path.isfile(cwd + "/src/train/train.py")
    assert os.path.isfile(cwd + "/src/evaluate/evaluate_model.py")
    assert os.path.isfile(cwd + "/src/register/register_model.py")
    logging.info("Los archivos necesarios para la ejecución existen")

    return (
        cwd + "/src/preprocess/pre_process.py",
        cwd + "/src/train/train.py",
        cwd + "/src/evaluate/evaluate_model.py",
        cwd + "/src/register/register_model.py",
    )


if __name__ == "__main__":
    # Check needed files exist, and returning they paths
    fe_dir, model_dir, eval_dir, reg_dir = check_files_exist()
    # Execute them sequentially
    cmd_str = f"python {fe_dir} && python {model_dir} && python {eval_dir} && python {reg_dir}"
    subprocess.run(cmd_str, shell=True)
