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
import signal
import traceback
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Tuple

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
    """Sanity-check the env on the parent side."""
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

    # Suspend any pending SIGALRM (armed by pytest-timeout's hookwrapper
    # prelude) for the entire duration of our hook. The actual test runs
    # in the spawned child, not in this process, so any SIGALRM that
    # fires here would crash the xdist worker rather than time-out the
    # test. The child gets its own ``--timeout`` so Python-level hangs
    # still produce clean Failed: Timeout reports, and we enforce a
    # parent-side kill via ``proc.join`` as a backstop. See the docstring
    # on ``_suspend_sigalrm`` for the gory details.
    with _suspend_sigalrm():
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


@contextmanager
def _suspend_sigalrm() -> Iterator[None]:
    """Disarm any pending SIGALRM in this thread for the duration of the block.

    pytest-timeout (when ``--timeout-method=signal``) installs a SIGALRM
    handler and arms ``ITIMER_REAL`` for each test via a hookwrapper that
    runs around our ``pytest_runtest_protocol``. That timer is meant to
    interrupt the test code, but our isolation hook intercepts the
    protocol BEFORE the test runs in this process — the test runs in the
    spawned child instead. If the SIGALRM fires while we're blocked on
    ``proc.join`` (or any of the queue-drain / cleanup that follows), it
    raises ``Failed: Timeout`` from inside the hook and crashes the xdist
    worker (xdist's worker_internal_error then marks every subsequent
    test as "found no collectors").

    We disarm the timer and reset the handler to SIG_DFL on entry, and
    leave them that way on exit. pytest-timeout's own
    ``pytest_timeout_cancel_timer`` postlude (which runs after our hook
    returns) sets the same final state — ``setitimer(0)`` + ``SIG_DFL``
    — so leaving ``SIG_DFL`` in place is what pytest-timeout would have
    done anyway. Trying to "restore" the prior alarm with the prior
    remaining time would re-arm it for a value computed before our
    multi-second join, defeating the whole point of suspending it.

    No-op on platforms without SIGALRM (Windows).
    """
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        # Windows path — no SIGALRM, no risk of the bug.
        yield
        return

    # signal.signal() and signal.setitimer() can only be called from the
    # main thread (Python raises ValueError otherwise). xdist workers run
    # the test loop on the main thread of their subprocess, so this is
    # normally fine — but if anything has spun up a thread that ends up
    # invoking our hook (unlikely but possible), fall back to a no-op
    # rather than crashing.
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    # Disarm any pending alarm and reset the handler. Any SIGALRM that's
    # already pending in the kernel queue will be delivered to SIG_DFL
    # (terminate process), but pytest-timeout uses ``setitimer`` not
    # ``alarm`` — so disarming the timer + resetting the handler in this
    # order means no signal is ever delivered to us.
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    yield
    # Intentionally do not restore: pytest-timeout's wrapper postlude
    # will run ``pytest_timeout_cancel_timer`` next, which sets the same
    # final state we're already in.


def _run_in_subprocess(
    item: pytest.Item, timeout: float
) -> List[pytest.TestReport]:
    """Spawn a fresh Python process, run the test there, collect reports.

    The child re-boots Python and runs ``pytest.main([nodeid])``. Slower
    than fork (~200-1000 ms per test depending on import graph) but the
    isolation is real — no inherited threads, FDs, or module state. We
    rely on ``-n auto`` xdist parallelism + CI sharding (pytest-split)
    to keep total wall time reasonable.
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
    # The entire parent-side wait loop runs under _suspend_sigalrm() because:
    #   * proc.join blocks for ~timeout seconds — SIGALRM lands here
    #   * the queue-drain after the join, plus terminate/kill fallbacks,
    #     can also be slow enough to overlap with the alarm fire
    #   * pytest-timeout's SIGALRM raises ``Failed: Timeout`` from inside
    #     whatever stack frame is current when it fires. If that's our hook,
    #     the xdist worker crashes and xdist marks every subsequent test
    #     it would have assigned to that worker as "found no collectors".
    # We enforce the timeout ourselves via the ``timeout`` argument to
    # ``proc.join``, padded by 5s so the child's own pytest-timeout has a
    # chance to fire first and emit a clean Failed: Timeout report.
    with _suspend_sigalrm():
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

        # If pytest.main exited cleanly with no reports captured, surface
        # an explicit error so the parent doesn't think the test passed.
        if not collector.sent_any and exit_code != 0:
            result_q.put(
                ("error", f"child pytest.main exited {exit_code} without reports")
            )
    except BaseException as exc:  # noqa: BLE001 — must catch everything
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            result_q.put(("error", f"child crashed: {tb}"))
        except Exception:
            # Queue may already be closed if the parent gave up.
            pass


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
