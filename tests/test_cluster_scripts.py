"""Cluster plumbing (CoE HPC, 2026-10-26): the Slurm templates render completely and are valid bash, the
walltime requeue is wired, and every shell script parses. No Slurm needed."""
import glob
import os
import re
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = dict(job="tb-skeleton", partition="gpuqs", gres="gpu:p100:1", time="02:00:00",
              account_line="#SBATCH -A coe", logdir="/scratch/x/slurm_logs", repo="/scratch/x/repo",
              experiment="skeleton")


def _render(name, **kw):
    return open(os.path.join(ROOT, "slurm", name)).read().format(**COMMON, **kw)


def _bash_n(text, tmp_path):
    p = tmp_path / "job.sh"
    p.write_text(text)
    r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_array_template_renders_and_requeues(tmp_path):
    t = _render("array.sbatch.tmpl", n=13, max_conc=8, cfg_list="/scratch/x/configs.txt", max_requeue=5)
    _bash_n(t, tmp_path)
    assert "{" not in re.sub(r"\$\{[A-Z_]+(:-[^}]*)?\}|\(\) \{|^\}$", "", t, flags=re.M)   # no unfilled fields
    assert "#SBATCH -A coe" in t and "#SBATCH --requeue" in t and "#SBATCH --signal=B:USR1@180" in t
    assert "trap requeue USR1" in t and 'scontrol requeue "$TASK"' in t and "-lt 5" in t
    assert "${CUDA_VISIBLE_DEVICES:-unset}" in t and "$$" not in t


def test_report_template_renders(tmp_path):
    t = _render("report.sbatch.tmpl", partition_cpu="defq")
    _bash_n(t, tmp_path)
    assert "-p defq" in t and "report_only=true" in t


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "scripts", "*.sh")) +
                                        glob.glob(os.path.join(ROOT, "slurm", "*.sbatch"))))
def test_shell_scripts_parse(path):
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "slurm", "*.sbatch"))))
def test_cluster_jobs_set_account_time_and_log_the_gpu(path):
    t = open(path).read()
    assert "#SBATCH -A coe" in t and re.search(r"#SBATCH -t \d", t) and "source scripts/hpc_env.sh" in t
    assert "log_gpu" in t


def test_setup_env_refuses_to_compile_on_a_login_node(tmp_path):
    env = {**os.environ, "TESTBED_SITE": "coe-hpc"}
    env.pop("SLURM_JOB_ID", None)
    r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "setup_env.sh"), "--llama", "--build-only"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "sbatch slurm/build_llama.sbatch" in r.stdout


def test_cpu_threads_follow_the_environment(monkeypatch):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=os.path.join(ROOT, "configs")):
        monkeypatch.delenv("TESTBED_CPU_THREADS", raising=False)
        assert compose(config_name="config").cpu_threads is None          # Colab / laptop: llama.cpp default
        monkeypatch.setenv("TESTBED_CPU_THREADS", "4")
        assert compose(config_name="config").cpu_threads == 4             # cluster job: its own CPUs
