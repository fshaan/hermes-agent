"""Subprocess-per-test isolation plugin.

Why this exists
---------------
``pytest-xdist`` workers are long-lived processes. Module-level dicts/sets
and ContextVars leak between tests on the same worker, causing the classic
"works alone, flakes in CI" failure pattern. The historic mitigation was a
giant ``_reset_module_state`` autouse fixture in ``conftest.py`` that
manually cleared each known state bucket. That approach is fragile (every
new module-level dict needs a corresponding line in conftest) and ugly.

This plugin replaces that fixture with true process isolation: each test
runs in a fresh Python interpreter via ``multiprocessing.Process`` with
the ``spawn`` start method. We deliberately do not offer a fork fast-path,
even on POSIX — fork inherits threads, FDs, signal handlers, and module
state from the parent, which defeats the whole point of "fresh per test"
and is the exact failure mode this plugin exists to prevent. ``spawn`` is
also the only context that works on Windows, so this keeps one code path
across all platforms.

The child process:
  1. Sets ``HERMES_ISOLATE_CHILD=1`` so this plugin is a no-op there
     (otherwise the child would try to spawn its own grandchildren —
     fork bomb).
  2. Boots a fresh interpreter, re-imports pytest and the test module,
     then invokes ``pytest.main([nodeid])`` against an in-process plugin
     that captures reports.
  3. Serializes test reports via ``pytest_report_to_serializable`` and
     pushes them through an ``mp.Queue`` back to the parent.

The parent:
  1. Reads serialized reports off the queue.
  2. Rehydrates them via ``pytest_report_from_serializable`` and emits via
     ``pytest_runtest_logreport`` so xdist / terminal reporter / JUnit all
     see normal-looking results.
  3. Honors ``isolate_timeout`` (ini key) — kills the child if it hangs and
     synthesizes a failure report.

Performance
-----------
Per-test overhead is dominated by Python interpreter startup +
collecting one nodeid (~200-1000 ms). xdist parallelism (``-n auto``)
amortizes this across cores in one CI job; for the full ~17 k-test
suite we shard across multiple parallel GHA jobs via ``pytest-split``
so total wall time fits in the 30-minute job timeout.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from typing import Any, List, Optional, Tuple

import _pytest.runner
import pytest


# ── Env-var sentinel ────────────────────────────────────────────────────────
# Set in every child process so the plugin disables itself there. Without
# this, every child would try to spawn its own grandchild (fork bomb) when
# pytest_runtest_protocol fires.
_CHILD_SENTINEL = "HERMES_ISOLATE_CHILD"

# Default timeout for one test. Overridable via the ``isolate_timeout``
# ini key in pyproject.toml.
_DEFAULT_TIMEOUT = 30.0


# ── pytest plugin hooks ─────────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI/ini options for subprocess isolation."""
    group = parser.getgroup("hermes-isolate", "Subprocess-per-test isolation")
    group.addoption(
        "--no-isolate",
        action="store_true",
        dest="no_isolate",
        default=False,
        help=(
            "Disable subprocess-per-test isolation. Tests will run in the "
            "xdist worker process, sharing module-level state with siblings."
        ),
    )
    parser.addini(
        "isolate_timeout",
        "Per-test timeout in seconds for the isolation subprocess. "
        "If the child exceeds this it is killed and a failure report is "
        "synthesized.",
        type="string",
        default=str(_DEFAULT_TIMEOUT),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Sanity-check the env and neutralize pytest-timeout on the parent side."""
    # Only run on the *parent* — children inherit envvar and short-circuit.
    if os.environ.get(_CHILD_SENTINEL) == "1":
        return

    # spawn must be available — it's the only context that works on Windows
    # and we deliberately use it on POSIX too for cross-platform symmetry.
    try:
        mp.get_context("spawn")
    except (ValueError, AttributeError) as exc:  # pragma: no cover
        raise pytest.UsageError(
            "hermes-isolate: multiprocessing 'spawn' context is unavailable "
            f"on this platform. ({exc})"
        ) from exc

    # Disable pytest-timeout in the parent process. It's the wrong layer
    # for us: pytest-timeout arms a SIGALRM for each test via a hookwrapper
    # around ``pytest_runtest_protocol``, expecting the test to actually
    # run in this process. We intercept that hook and spawn a subprocess
    # instead — so the alarm has no test to interrupt, and any SIGALRM
    # that fires here raises ``Failed: Timeout`` from inside our hook,
    # which xdist surfaces as "worker_internal_error" + "found no
    # collectors" for every subsequent test on that worker.
    #
    # We don't need pytest-timeout in the parent because:
    #   1. The child subprocess gets its own ``--timeout`` baked in
    #      (see ``_spawn_child_entrypoint``), so Python-level hangs IN
    #      the test still produce clean ``Failed: Timeout`` reports.
    #   2. The parent enforces ``proc.join(timeout + 5)`` and kills the
    #      child if it overruns, synthesizing a failure report on its
    #      behalf. That's our backstop for child crashes too.
    #
    # pytest-timeout caches the resolved timeout in
    # ``config._env_timeout`` during its own ``pytest_configure``, which
    # may have already run by the time we get here (hook order between
    # entry_point plugins isn't guaranteed). Setting both ``option.timeout``
    # and the cached private attributes covers both code paths in
    # ``pytest_timeout._get_item_settings``: it reads ``_env_timeout``
    # first, then falls back to the option.
    if hasattr(config.option, "timeout"):
        config.option.timeout = 0
    # ``_env_timeout`` is a private attribute but stable across
    # pytest-timeout 2.x. Setting it to ``None`` (not 0) matches what
    # pytest-timeout itself uses when no timeout is configured — its
    # ``_get_item_settings`` then takes the no-timeout branch instead
    # of treating 0 as a configured-but-disabled value.
    config._env_timeout = None  # type: ignore[attr-defined]
    config._env_timeout_method = None  # type: ignore[attr-defined]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: Optional[pytest.Item]
) -> Optional[bool]:
    """Intercept test execution; run in a spawned subprocess.

    Returning ``True`` tells pytest "I handled this — skip the normal
    runtestprotocol". Returning ``None`` falls through to the default.
    """
    # Disable in child processes (fork-bomb prevention) and when user
    # passed --no-isolate.
    if os.environ.get(_CHILD_SENTINEL) == "1":
        return None
    if item.config.getoption("no_isolate", default=False):
        return None

    timeout = _parse_timeout(item.config.getini("isolate_timeout"))

    reports = _run_in_subprocess(item, timeout)

    # Emit reports through the normal pytest channels so xdist + terminal
    # reporter + junit etc. all see them as if the test ran normally.
    ihook = item.ihook
    ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    for rep in reports:
        ihook.pytest_runtest_logreport(report=rep)
    ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


# ── Internal: subprocess machinery ──────────────────────────────────────────


def _parse_timeout(raw: Any) -> float:
    """Coerce the ini value to a float; fall back to the default."""
    try:
        value = float(raw)
        if value <= 0:
            raise ValueError
        return value
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT


def _run_in_subprocess(
    item: pytest.Item, timeout: float
) -> List[pytest.TestReport]:
    """Spawn a fresh Python process, run the test there, collect reports.

    The child re-boots Python and runs ``pytest.main([nodeid])`` (~200-1000
    ms per test depending on import graph) — slower than fork, but the
    isolation is real: no inherited threads, FDs, signal handlers, or
    module state. We rely on ``-n auto`` xdist parallelism to amortize
    the spawn cost across cores.
    """
    ctx = mp.get_context("spawn")
    result_q: "mp.Queue[Tuple[str, Any]]" = ctx.Queue()

    rootdir = str(item.config.rootpath)
    nodeid = item.nodeid

    proc = ctx.Process(
        target=_spawn_child_entrypoint,
        args=(result_q, rootdir, nodeid, os.getpid(), timeout),
        # Helpful in process listings; spawn ignores name visually on most
        # systems but JUnit output uses it for pid attribution.
        name=f"pytest-isolate:{nodeid}",
        daemon=False,
    )
    proc.start()

    timed_out = False
    collected: List[Any] = []
    # We enforce the timeout ourselves via ``proc.join(timeout + 5)`` —
    # the +5s pad gives the child's own ``--timeout`` a chance to fire
    # first and emit a clean ``Failed: Timeout`` report before we kill
    # it. pytest-timeout in the parent has already been neutralized in
    # ``pytest_configure`` (it would otherwise raise ``Failed: Timeout``
    # from inside this hook and crash the xdist worker, leading to
    # "found no collectors" for every subsequent test).
    try:
        proc.join(timeout + 5.0)
        if proc.is_alive():
            timed_out = True
            proc.terminate()
            proc.join(5)
            if proc.is_alive():  # pragma: no cover — terminate refused
                proc.kill()
                proc.join()

        # Drain the queue. There may be zero items on timeout/crash.
        while True:
            try:
                kind, payload = result_q.get_nowait()
            except Exception:  # queue.Empty raises across mp contexts
                break
            if kind == "report":
                collected.append(payload)
            elif kind == "error":
                # Child raised before producing reports — synthesize a
                # failure below.
                collected.append(("__error__", payload))
    finally:
        result_q.close()
        result_q.join_thread()

    # Convert serializable reports back to live TestReport instances.
    reports: List[pytest.TestReport] = []
    for payload in collected:
        if isinstance(payload, tuple) and payload and payload[0] == "__error__":
            reports.append(_synthesize_crash_report(item, payload[1]))
            continue
        rep = item.config.hook.pytest_report_from_serializable(
            config=item.config, data=payload
        )
        reports.append(rep)

    if timed_out and not reports:
        reports.append(
            _synthesize_crash_report(
                item, f"Test exceeded isolate_timeout ({timeout:.1f}s)"
            )
        )
    elif not reports and proc.exitcode not in (0, None):
        reports.append(
            _synthesize_crash_report(
                item, f"Child exited with code {proc.exitcode} and no reports"
            )
        )

    return reports


def _synthesize_crash_report(item: pytest.Item, message: str) -> pytest.TestReport:
    """Build a failed TestReport when the child died before sending one."""
    longrepr = f"hermes-isolate: {message}"
    call_info = _pytest.runner.CallInfo.from_call(
        lambda: (_ for _ in ()).throw(RuntimeError(longrepr)), "call"
    )
    return _pytest.runner.pytest_runtest_makereport(item, call_info)


# ── Internal: child entrypoint ──────────────────────────────────────────────
# Module-level so ``spawn`` can pickle it (lambdas/closures don't pickle).


def _spawn_child_entrypoint(
    result_q: "mp.Queue", rootdir: str, nodeid: str, parent_pid: int, timeout: float
) -> None:
    """Run a single test in a spawned (fresh-interpreter) process.

    Must be importable at the top level for ``spawn`` to find it via
    ``mp.spawn``'s pickling machinery.
    """
    os.environ[_CHILD_SENTINEL] = "1"
    os.environ["PYTEST_PARENT_PID"] = str(parent_pid)

    # Capture stdout/stderr so we can ship it back to the parent if
    # the child crashes — keeps the parent's terminal clean while
    # preserving diagnostics for when something does go wrong.
    import io, sys
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = captured_out
    sys.stderr = captured_err

    try:
        # Move into the rootdir so relative test paths resolve correctly.
        os.chdir(rootdir)

        # Use pytest.main in-process. ``-p no:cacheprovider`` keeps the
        # child from racing on .pytest_cache. ``-p no:xdist`` is critical:
        # the child must NOT try to spawn its own xdist workers.
        # ``--no-header --no-summary -q`` keeps output noise down (the
        # parent re-renders reports anyway via logreport).
        #
        # We pass ``--timeout`` explicitly because ``-o addopts=`` purges
        # the parent's addopts (which carry the timeout config). Without
        # it, a hanging test would only be caught by the parent-side
        # ``proc.join`` timeout, surfaced as a generic SIGTERM crash. With
        # it, pytest-timeout fires inside the child and produces a clean
        # "Timeout" failure report.
        argv = [
            nodeid,
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "-p",
            "no:hermes_isolate",  # belt-and-suspenders against re-entry
            "--no-header",
            "-q",
            "-o",
            "addopts=",  # purge parent's addopts (would re-add -n auto)
            f"--timeout={timeout:.1f}",
            "--timeout-method=signal",
        ]

        collector = _ReportCollector(result_q)
        # Note: we DO want the child to load tests/conftest.py — that's
        # what provides _hermetic_environment + _live_system_guard. The
        # in-process plugin just intercepts reports.
        exit_code = pytest.main(argv, plugins=[collector])

        # If pytest.main exited without producing reports, surface the
        # captured stdout/stderr so the parent can synthesize a useful
        # error message instead of "Child exited with code N and no
        # reports". This catches collection errors, plugin crashes, and
        # anything else that bypasses the normal report-emit path.
        if not collector.sent_any:
            details = (
                f"child pytest.main exited {exit_code} without reports\n"
                f"--- captured stdout ---\n{captured_out.getvalue()}\n"
                f"--- captured stderr ---\n{captured_err.getvalue()}"
            )
            result_q.put(("error", details))
    except BaseException as exc:  # noqa: BLE001 — must catch everything
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            result_q.put((
                "error",
                f"child crashed: {tb}\n"
                f"--- captured stdout ---\n{captured_out.getvalue()}\n"
                f"--- captured stderr ---\n{captured_err.getvalue()}",
            ))
        except Exception:
            # Queue may already be closed if the parent gave up.
            pass
    finally:
        sys.stdout = real_stdout
        sys.stderr = real_stderr


class _ReportCollector:
    """In-process pytest plugin that ships TestReport objects via queue."""

    def __init__(self, result_q: "mp.Queue") -> None:
        self._q = result_q
        self._config: Optional[pytest.Config] = None
        self.sent_any = False

    @pytest.hookimpl
    def pytest_configure(self, config: pytest.Config) -> None:
        self._config = config

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:  # noqa: D401
        # Serialize through the config's hook so plugins that extend the
        # serialization (xdist, etc.) participate.
        data: Any
        if self._config is not None:
            try:
                data = self._config.hook.pytest_report_to_serializable(
                    config=self._config, report=report
                )
            except Exception:
                data = None
        else:
            data = None

        if data is None:
            # Last-resort minimal serialization. Better than dropping the
            # report and silently "passing" a broken test.
            data = {
                "$report_type": "TestReport",
                "nodeid": report.nodeid,
                "when": report.when,
                "outcome": report.outcome,
                "longrepr": str(report.longrepr) if report.longrepr else None,
            }

        try:
            self._q.put(("report", data))
            self.sent_any = True
        except Exception:
            pass


# ── Identification helpers (for testing the plugin itself) ─────────────────


__all__ = [
    "pytest_addoption",
    "pytest_configure",
    "pytest_runtest_protocol",
]
