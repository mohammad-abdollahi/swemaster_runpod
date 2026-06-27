"""
ApptainerRuntime: a Docker-free execution backend for R2E-Gym, for HPC clusters
(e.g. Alliance Canada / Nibi) that provide Apptainer instead of a Docker daemon.

Design
------
It subclasses DockerRuntime so that ALL the swebench/swegym/swerebench grading logic
(test_spec creation, get_logs_eval, _calculate_reward_*, parse_logs, setup_env_*) is
inherited unchanged. We override ONLY the container primitives:

  * __init__            -> set up attributes + test_spec (no docker daemon), then
                           locate/build a read-only .sif and a writable single-file overlay.
  * run / demux_run     -> `apptainer exec --fakeroot --overlay <img> <sif> bash -c ...`
  * apply_patch / reverse_patch / create_file / copy_to_container
                        -> stream data in over stdin (no docker put_archive / tar).
  * start_container / stop_container / reset / close -> overlay lifecycle.

State (agent edits to /testbed, `pip install -e .`, etc.) persists in the overlay file.
Each command runs as a fresh `apptainer exec`; because per-task commands are sequential,
re-using one overlay file across calls is safe and needs no long-running instance/daemon.

Key knobs come from env vars (set by env.sh):
  SIF_DIR, OVERLAY_DIR, APPTAINER_OVERLAY_MB, APPTAINER_ALLOW_BUILD
"""

import os
import re
import shlex
import subprocess
import uuid
from pathlib import Path

from r2egym.agenthub.runtime.docker import (
    DockerRuntime,
    DOCKER_PATH,
)
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub import CMD_TIMEOUT

SIF_DIR = Path(os.environ.get("SIF_DIR", "/scratch/swemaster/sif"))
OVERLAY_DIR = Path(os.environ.get("OVERLAY_DIR", "/scratch/swemaster/overlays"))
OVERLAY_MB = int(os.environ.get("APPTAINER_OVERLAY_MB", "6144"))  # writable layer per task
ALLOW_BUILD = os.environ.get("APPTAINER_ALLOW_BUILD", "1") == "1"
APPTAINER_BIN = os.environ.get("APPTAINER_BIN", "apptainer")
# Dir of pre-downloaded wheels (e.g. chardet) bind-mounted into every container so
# offline compute nodes can `pip install --no-index`. chardet is REQUIRED: the agent's
# str_replace_editor tool imports it at startup; without it the primary edit tool fails.
WHEEL_DIR = os.environ.get("WHEEL_DIR", "")
# Some clusters need fakeroot; allow disabling for debugging.
USE_FAKEROOT = os.environ.get("APPTAINER_FAKEROOT", "1") == "1"

_ANSI = re.compile(r"\x1b\[[0-9;]*m|\r")
# apptainer prints these to stderr on every fakeroot exec; strip so they don't pollute
# the agent's observation or the swebench log parser.
_APPTAINER_NOISE = re.compile(
    r"^(INFO|WARNING|DEBUG|VERBOSE):\s+(User not listed in /etc/subuid|"
    r"Using fakeroot command|Could not|Converting|Mounting|gocryptfs).*$\n?",
    re.MULTILINE,
)


def _clean(text: str) -> str:
    return _APPTAINER_NOISE.sub("", text)


def _sif_name(docker_image: str, instance_id: str) -> str:
    base = instance_id or docker_image.split(":")[-1].split("/")[-1]
    return base.replace("/", "_").replace(":", "_")


class ApptainerRuntime(DockerRuntime):
    def __init__(
        self,
        ds,
        repo_path: str = "/testbed",
        alt_path: str = "/root",
        docker_image: str = None,
        command="/bin/bash",
        ip: str = None,
        logger=None,
        backend: str = "apptainer",
        use_lsp: bool = False,
        **kwargs,
    ):
        assert ds, "Dataset entry (ds) required"
        self.backend = "apptainer"
        self.logger = logger or get_logger("ApptainerRuntime")
        self.use_lsp = use_lsp
        self.ds = ds

        # --- image + dataset-type detection (mirrors DockerRuntime.__init__) ---
        if "docker_image" in ds:
            ds_image = ds["docker_image"]
        elif "image_name" in ds:
            ds_image = ds["image_name"]
        else:
            raise ValueError(f"No docker image found in ds: {ds.get('instance_id')}")
        self.instance_id = ds.get("instance_id")
        self.docker_image = docker_image or ds_image
        self.swebench_verified = "swebench" in self.docker_image or "epoch-research" in self.docker_image
        self.swesmith = "swesmith" in self.docker_image
        self.swerebench = "swerebench" in self.docker_image
        self.swegym = "xingyaoww" in self.docker_image
        if self.swesmith:
            self.swebench_verified = False

        # --- test_spec (reuse parent's exact construction by delegating) ---
        self._build_test_spec()

        # --- runtime params (mirror parent) ---
        self.repo_path = repo_path
        self.alt_path = alt_path
        self.command = command
        import json as _json
        from r2egym.commit_models.diff_classes import ParsedCommit
        self.repo_name = (
            ds["repo"] if (self.swebench_verified or self.swesmith or self.swegym or self.swerebench)
            else ds["repo_name"]
        )
        if not self.swesmith and not self.swegym and not self.swerebench:
            self.commit_json = ds["parsed_commit"] if self.swebench_verified else ds["parsed_commit_content"]
            self.commit = ParsedCommit(**_json.loads(self.commit_json))

        self.ip = ip
        self.docker_kwargs = kwargs

        # --- apptainer artifacts ---
        self.sif = str(SIF_DIR / f"{_sif_name(self.docker_image, self.instance_id)}.sif")
        # overlay is per (job, instance): include a run tag so parallel runs over the SAME
        # instances (e.g. SFT and RL evals at once) never share/corrupt an overlay file.
        run_tag = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        self.overlay = str(OVERLAY_DIR / f"{_sif_name(self.docker_image, self.instance_id)}.{run_tag}.overlay.img")
        self.container_name = _sif_name(self.docker_image, self.instance_id)
        SIF_DIR.mkdir(parents=True, exist_ok=True)
        OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

        self.start_container(self.docker_image, command, self.container_name, **kwargs)

        # Initialize the environment (inherited setup_env_swebench etc., uses self.run)
        self.setup_env()
        if self.use_lsp:
            (self.setup_lsp if (self.swebench_verified or self.swesmith or self.swegym or self.swerebench)
             else self.uv_setup_lsp)()
        self.logger.info(f"Apptainer env ready: repo={self.repo_name} image={self.docker_image}")

    # ------------------------------------------------------------------ #
    # test_spec construction (copied dispatch from DockerRuntime.__init__)
    # ------------------------------------------------------------------ #
    def _build_test_spec(self):
        import json as _json
        if self.swegym:
            from swebench_fork_swegym.harness.test_spec import TestSpec
            if "make_test_spec" in self.ds:
                self.test_spec = TestSpec(**_json.loads(self.ds["make_test_spec"]))
            else:
                from r2egym.agenthub.runtime.docker import make_test_spec_swegym
                self.test_spec = make_test_spec_swegym(self.ds)
        elif self.swebench_verified:
            from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
            if "make_test_spec" in self.ds:
                self.test_spec = TestSpec(**_json.loads(self.ds["make_test_spec"]))
            else:
                self.test_spec = make_test_spec(self.ds)
        elif self.swerebench:
            from swebench_fork_swerebench.harness.test_spec.test_spec import TestSpec
            from r2egym.agenthub.runtime.docker import make_test_spec_swerebench
            if "make_test_spec" in self.ds:
                self.test_spec = TestSpec(**_json.loads(self.ds["make_test_spec"]))
            else:
                self.test_spec = make_test_spec_swerebench(self.ds)

    # ------------------------------------------------------------------ #
    # overlay / image lifecycle
    # ------------------------------------------------------------------ #
    def _create_overlay(self):
        # NOTE: must be created WITH --fakeroot so the overlay upper/work dirs are owned
        # for the root-mapped namespace; otherwise `exec --fakeroot --overlay` fails with
        # "overlay upper dir ... is not writable". (Validated on Nibi: no /etc/subuid entry.)
        if os.path.exists(self.overlay):
            os.remove(self.overlay)
        argv = [APPTAINER_BIN, "overlay", "create", "--size", str(OVERLAY_MB)]
        if USE_FAKEROOT:
            argv.append("--fakeroot")
        argv.append(self.overlay)
        subprocess.run(argv, check=True, capture_output=True)

    def _ensure_sif(self):
        if os.path.exists(self.sif):
            return
        if not ALLOW_BUILD:
            raise FileNotFoundError(
                f"SIF missing and APPTAINER_ALLOW_BUILD=0: {self.sif}. Pre-build it on a login node."
            )
        uri = self.docker_image
        if not uri.startswith(("docker://", "oras://", "library://")):
            uri = "docker://" + uri
        self.logger.info(f"Building SIF {self.sif} from {uri}")
        subprocess.run([APPTAINER_BIN, "build", "--force", self.sif, uri], check=True)

    def start_container(self, docker_image=None, command=None, ctr_name=None, **kwargs):
        self._ensure_sif()
        self._create_overlay()

    def stop_container(self):
        try:
            if os.path.exists(self.overlay):
                os.remove(self.overlay)
        except OSError as e:
            self.logger.warning(f"overlay cleanup failed: {e}")

    def reset(self):
        # pristine image state = fresh overlay
        self._create_overlay()

    def close(self, delete_sif: bool = False):
        self.stop_container()
        if delete_sif:
            try:
                if os.path.exists(self.sif):
                    os.remove(self.sif)
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    # core exec primitive
    # ------------------------------------------------------------------ #
    def _argv(self, inner_cmd, demux=False):
        argv = [APPTAINER_BIN, "--quiet", "exec"]
        if USE_FAKEROOT:
            argv.append("--fakeroot")
        if WHEEL_DIR and os.path.isdir(WHEEL_DIR):
            argv += ["--bind", f"{WHEEL_DIR}:/opt/wheels:ro"]
        argv += ["--overlay", self.overlay, self.sif, "/bin/bash", "-c", inner_cmd]
        return argv

    def run(self, code: str, timeout: int = CMD_TIMEOUT, args: str = "", workdir=None, type: str = None):
        exec_workdir = self.repo_path if workdir is None else workdir
        # mirror DockerRuntime: timeout wraps `code args`; PATH forced inside.
        inner = f"export PATH={DOCKER_PATH}; cd {shlex.quote(exec_workdir)} 2>/dev/null; timeout {timeout} {code} {args}"
        self.logger.info(f"[apptainer] {code[:120]}")
        try:
            # IMPORTANT: merge stderr INTO stdout (interleaved, execution order) to match
            # docker exec_run's combined stream. swebench grading splits the log on the
            # test_cmd (a `set -x` trace line on stderr); separate capture would reorder it
            # after the test output and the split would discard the results.
            p = subprocess.run(
                self._argv(inner), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=timeout + 30,
            )
        except subprocess.TimeoutExpired:
            return f"The command took too long to execute (>{timeout}s)", "-1"
        except Exception as e:  # noqa
            return f"Error: {repr(e)} for command: {code}", "-1"

        output = _clean((p.stdout or b"").decode("utf-8", errors="replace"))
        rc = p.returncode
        if rc == 124:
            return f"The command took too long to execute (>{timeout}s)", "-1"
        if rc != 0:
            return output, f"Error: Exit code {rc}"
        return _ANSI.sub("", output), str(rc)

    def demux_run(self, code: str, timeout: int = CMD_TIMEOUT, args: str = "", workdir=None):
        exec_workdir = self.repo_path if workdir is None else workdir
        inner = f"export PATH={DOCKER_PATH}; cd {shlex.quote(exec_workdir)} 2>/dev/null; timeout {timeout} {code} {args}"
        try:
            p = subprocess.run(self._argv(inner), capture_output=True, timeout=timeout + 30)
        except Exception as e:  # noqa
            return f"Error: {repr(e)}", f"Error: {repr(e)}", "-1"
        stdout = (p.stdout or b"").decode("utf-8", errors="replace")
        stderr = (p.stderr or b"").decode("utf-8", errors="replace")
        rc = p.returncode
        if rc != 0:
            return stdout, stderr, f"Error: Exit code {rc}"
        return stdout, stderr, str(rc)

    # ------------------------------------------------------------------ #
    # file / patch helpers (stdin streaming instead of docker put_archive)
    # ------------------------------------------------------------------ #
    def _run_stdin(self, inner_cmd: str, data: str, timeout: int = CMD_TIMEOUT):
        argv = self._argv(inner_cmd)
        try:
            p = subprocess.run(argv, input=data.encode("utf-8"), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=timeout + 30)
        except Exception as e:  # noqa
            return f"Error: {repr(e)}", "-1"
        out = _clean((p.stdout or b"").decode("utf-8", "replace"))
        if p.returncode != 0:
            return out, f"Error: Exit code {p.returncode}"
        return _ANSI.sub("", out), "0"

    def copy_to_container(self, src_path: str, dest_path: str):
        with open(src_path, "r", errors="replace") as f:
            content = f.read()
        dest_dir = os.path.dirname(dest_path) or "/"
        inner = f"mkdir -p {shlex.quote(dest_dir)} && cat > {shlex.quote(dest_path)}"
        return self._run_stdin(inner, content)

    def create_file(self, file_path: str, content: str):
        dest = file_path if file_path.startswith("/") else f"/{file_path}"
        dest_dir = os.path.dirname(dest) or "/"
        inner = f"mkdir -p {shlex.quote(dest_dir)} && cat > {shlex.quote(dest)}"
        return self._run_stdin(inner, content)

    def apply_patch(self, patch: str):
        inner = f"export PATH={DOCKER_PATH}; cd {shlex.quote(self.repo_path)} && git apply --whitespace=fix -"
        return self._run_stdin(inner, patch)

    def reverse_patch(self, patch: str):
        inner = f"export PATH={DOCKER_PATH}; cd {shlex.quote(self.repo_path)} && git apply -R -"
        return self._run_stdin(inner, patch)

    # ------------------------------------------------------------------ #
    # offline/fast environment setup (override parent's broken external-mirror pip)
    # ------------------------------------------------------------------ #
    def setup_env_swebench(self):
        # Parent's version does `pip install chardet -i http://pypi-mirror.weizhipin.com/...`
        # which is an internal mirror that hangs ~100s on every instance and breaks offline.
        # We only need: executable run_tests.sh, alt_path="/", and the conda env symlink.
        self.run("chmod +x /run_tests.sh", timeout=60)
        self.alt_path = "/"  # swebench dockers keep /run_tests.sh at /
        self.run("ln -s /opt/miniconda3/envs/testbed /root/.venv 2>/dev/null; true", timeout=60)
        # chardet is REQUIRED by the str_replace_editor tool. Install offline from the
        # bind-mounted wheel dir (/opt/wheels) so the editor tool works on offline nodes.
        out, _ = self.run(
            "python -c 'import chardet' 2>/dev/null && echo present || "
            "python -m pip install --no-index --find-links /opt/wheels chardet 2>&1 | tail -1",
            timeout=120,
        )
        self.logger.info(f"chardet setup: {out.strip()[:120]}")
