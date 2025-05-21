"""Vox tests"""

import copy
import io
import os
import pathlib
import stat
import subprocess
import sys
import types
from typing import Callable, Self, cast

import pytest
from xonsh.built_ins import XonshSession
from xonsh.environ import Env
from xonsh.platform import ON_WINDOWS
from xonsh.pytest.tools import skip_if_on_conda, skip_if_on_msys
from xonsh.tools import EnvPath

from xontrib_uvox import UvoxHandler
from xontrib_uvox.uvoxapi import Uvox


class XonshSessionSafe(XonshSession):
    env: Env

    @property
    def path(self) -> EnvPath:
        res = self.env["PATH"]
        if not isinstance(res, EnvPath):
            raise ValueError(
                f'Something is wrong with your Xonsh session: `${{...}}["PATH"]` is {res!r}'
            )
        return res

    @path.setter
    def path(self, paths: list[str] | EnvPath):
        self.env["PATH"] = EnvPath(paths)

    @classmethod
    def from_xession(cls: type[Self], xession: XonshSession) -> Self:
        res = copy.copy(xession)
        res.__class__ = cls
        return cast(Self, res)


@pytest.fixture
def xession_safe(xession):
    return XonshSessionSafe.from_xession(xession=xession)


# FIXME: this is fishy, it should also return a session intead of modifying the current one and we
# could type it as safe session.
@pytest.fixture
def venv_home(tmp_path: pathlib.Path, xession_safe: XonshSessionSafe) -> pathlib.Path:
    """Path where VENVs are created"""
    home = tmp_path / "venvs"
    home.mkdir()
    # Set up an isolated venv home
    xession_safe.env["VIRTUALENV_HOME"] = str(home)
    return home


# FIXME: this is fishy, it should also return a session intead of modifying the current one and we
# could type it as safe session.
@pytest.fixture
def uvox(xession_safe: XonshSessionSafe, load_xontrib: Callable[[str], None]) -> UvoxHandler:
    """uvox Alias function"""

    # Set up enough environment for xonsh to function
    xession_safe.env["PWD"] = str(pathlib.Path.cwd())
    xession_safe.env["DIRSTACK_SIZE"] = 10
    xession_safe.path = []
    xession_safe.env["XONSH_SHOW_TRACEBACK"] = True

    load_xontrib("uvox")
    assert xession_safe.aliases is not None
    uvox = xession_safe.aliases["uvox"]
    return uvox


class Listener:
    def __init__(self, xession_safe: XonshSessionSafe):
        self.xession = xession_safe
        self.last = None

    def listener(self, name):
        def _wrapper(**kwargs):
            self.last = (name,) + tuple(kwargs.values())

        return _wrapper

    def __call__(self, *events: str):
        for name in events:
            event = getattr(cast(types.SimpleNamespace, self.xession.builtins).events, name)
            event(self.listener(name))


@pytest.fixture
def record_events(xession_safe: XonshSessionSafe) -> Listener:
    return Listener(xession_safe)


def test_uvox_flow(
    xession_safe: XonshSessionSafe,
    uvox: UvoxHandler,
    record_events: Listener,
    venv_home: pathlib.Path,
):
    """
    Creates a virtual environment, gets it, enumerates it, and then deletes it.
    """

    record_events("uvox_on_create", "uvox_on_delete", "uvox_on_activate", "uvox_on_deactivate")

    uvox(["create", "spam"])
    assert (venv_home / "spam").is_dir()
    assert record_events.last == ("uvox_on_create", "spam")

    ve = uvox.uvox.get_env("spam")
    assert ve.root == venv_home / "spam"
    assert ve.bin.is_dir()

    assert "spam" in uvox.uvox.list_envs()

    # activate
    uvox(["activate", "spam"])
    virtualenv_path_var = xession_safe.env["VIRTUAL_ENV"]
    assert isinstance(virtualenv_path_var, str)
    assert len(virtualenv_path_var) > 0
    assert pathlib.Path(virtualenv_path_var) == uvox.uvox.get_env("spam").root
    assert record_events.last == ("uvox_on_activate", "spam", ve.root)

    out = io.StringIO()
    # info
    uvox(["info"], stdout=out)
    assert "spam" in out.getvalue()
    out.seek(0)

    # list
    uvox(["list"], stdout=out)
    print(out.getvalue())
    assert "spam" in out.getvalue()
    out.seek(0)

    # # wipe
    # uvox(["wipe"], stdout=out)
    # print(out.getvalue())
    # assert "Nothing to remove" in out.getvalue()
    # out.seek(0)

    # deactivate
    uvox(["deactivate"])
    assert "VIRTUAL_ENV" not in xession_safe.env
    assert record_events.last == ("uvox_on_deactivate", ve.root)

    # runin
    # TODO: check if testing on pip makes sense
    uvox(["runin", "spam", "pip", "--version"], stdout=out)
    print(out.getvalue())
    assert "spam" in out.getvalue()
    out.seek(0)

    # removal
    uvox(["rm", "spam", "--force"])
    assert not (venv_home / "spam").exists()
    assert record_events.last == ("uvox_on_delete", "spam")


def test_activate_non_uv_venv(
    monkeypatch: pytest.MonkeyPatch,
    xession_safe: XonshSessionSafe,
    uvox: UvoxHandler,
    record_events,
    venv_home: pathlib.Path,
):
    """
    Create a virtual environment using Python's built-in venv module
    (not in VIRTUALENV_HOME) and verify that vox can activate it correctly.
    """
    xession_safe.env["PATH"] = []

    record_events("uvox_on_activate", "uvox_on_deactivate")

    with monkeypatch.context() as m:
        m.chdir(venv_home)
        venv_dirname = "venv"
        subprocess.run([sys.executable, "-m", "venv", venv_dirname])  # noqa: S603
        uvox(["activate", venv_dirname])
        vxv = uvox.uvox.get_env(venv_dirname)

    assert vxv.root.is_absolute()
    assert vxv.bin.is_absolute()

    env = xession_safe.env

    assert pathlib.Path(cast(str, xession_safe.path[0])) == vxv.bin

    virtualenv_path_var = env["VIRTUAL_ENV"]
    assert isinstance(virtualenv_path_var, str)
    assert len(virtualenv_path_var) > 0
    assert pathlib.Path(virtualenv_path_var) == vxv.root

    assert record_events.last == (
        "uvox_on_activate",
        venv_dirname,
        venv_home / venv_dirname,
    )

    uvox(["deactivate"])
    assert not xession_safe.path
    assert "VIRTUAL_ENV" not in env
    assert record_events.last == (
        "uvox_on_deactivate",
        pathlib.Path(str(venv_home)) / venv_dirname,
    )


@skip_if_on_msys
@skip_if_on_conda
def test_path(xession_safe: XonshSessionSafe, uvox: UvoxHandler, a_venv: pathlib.Path):
    """
    Test to make sure Vox properly activates and deactivates by examining $PATH
    """
    oldpath = list(xession_safe.path)
    uvox(["activate", str(a_venv)])

    assert oldpath != xession_safe.path

    uvox.deactivate()

    assert oldpath == xession_safe.path


def test_crud_subdir(monkeypatch: pytest.MonkeyPatch, venv_home: pathlib.Path):
    """
    Creates a virtual environment, gets it, enumerates it, and then deletes it.
    """

    with monkeypatch.context() as m:
        m.chdir(venv_home)
        uvox = UvoxHandler()
        uvox(["create", "spam/eggs"])

    assert (venv_home / "spam" / "eggs").is_dir()

    ve = uvox.uvox.get_env("spam/eggs")
    assert ve.root == venv_home / "spam" / "eggs"
    assert ve.bin.is_dir()

    # assert 'spam/eggs' in list(vox)  # This is NOT true on Windows  # FIXME: why?
    assert "spam" not in uvox.uvox.list_envs()

    uvox(["remove", "--force", "spam/eggs"])

    assert not (venv_home / "spam" / "eggs").exists()


def test_crud_path(tmp_path):
    """
    Creates a virtual environment, gets it, enumerates it, and then deletes it.
    """

    uvox = UvoxHandler()
    uvox(["create", str(tmp_path)])
    assert (tmp_path / "lib").is_dir()

    ve = uvox.uvox.get_env(tmp_path)
    assert ve.root == tmp_path
    assert ve.bin.is_dir()

    uvox(["remove", "--force", str(tmp_path)])

    assert not tmp_path.exists()


# TODO: we don't currently disallow them, make sure we mean that (but why wouldn't we?)
# @skip_if_on_msys
# @skip_if_on_conda
# def test_reserved_names(xession_safe: XonshSessionSafe, tmp_path):
#     """
#     Tests that reserved words are disallowed.
#     """
#     xession_safe.env["VIRTUALENV_HOME"] = str(tmp_path)

#     uvox = Uvox()

#     if ON_WINDOWS:
#         with pytest.raises(ValueError):
#             uvox.create("Scripts")
#     else:
#         with pytest.raises(ValueError):
#             uvox.create("bin")

#     if ON_WINDOWS:
#         with pytest.raises(ValueError):
#             uvox.create("spameggs/Scripts")
#     else:
#         with pytest.raises(ValueError):
#             uvox.create("spameggs/bin")


@pytest.fixture
def create_venv() -> Callable[[str], pathlib.Path]:
    uvox = Uvox()

    def wrapped(name):
        uvox.create(name)
        return uvox.get_env(name).root

    return wrapped


@pytest.fixture
def venvs(venv_home: pathlib.Path, create_venv: Callable[[str], pathlib.Path]):
    """Create virtualenv with names venv0, venv1"""
    from xonsh.dirstack import popd, pushd

    pushd([str(venv_home)])
    yield [create_venv(f"venv{idx}") for idx in range(2)]
    popd([])


@pytest.fixture
def a_venv(create_venv: Callable[[str], pathlib.Path]) -> pathlib.Path:
    return create_venv("venv0")


# TODO: figure this out sometimes

# @pytest.fixture
# def patched_cmd_cache(xession_safe: XonshSessionSafe, uvox, monkeypatch):
#     cc = xession_safe.commands_cache

#     def no_change(self, *_):
#         return False, False

#     monkeypatch.setattr(cc, "_check_changes", types.MethodType(no_change, cc))
#     bins = {path: (path, False) for path in _PY_BINS}
#     monkeypatch.setattr(cc, "_cmds_cache", bins)
#     yield cc


# _VENV_NAMES = {"venv1", "venv1/", "venv0/", "venv0"}
# if ON_WINDOWS:
#     _VENV_NAMES = {"venv1\\", "venv0\\", "venv0", "venv1"}

# _HELP_OPTS = {
#     "-h",
#     "--help",
# }
# _PY_BINS = {"/bin/python3"}

# _VOX_RM_OPTS = {"-f", "--force"}.union(_HELP_OPTS)


# class TestVoxCompletions:
#     @pytest.fixture
#     def check(self, check_completer, xession_safe: XonshSessionSafe, uvox):
#         def wrapped(cmd, positionals, options=None):
#             for k in list(xession_safe.completers):
#                 if k != "alias":
#                     xession_safe.completers.pop(k)
#             assert check_completer(cmd) == positionals
#             xession_safe.env["ALIAS_COMPLETIONS_OPTIONS_BY_DEFAULT"] = True
#             if options:
#                 assert check_completer(cmd) == positionals.union(options)

#         return wrapped

#     @pytest.mark.parametrize(
#         "args, positionals, opts",
#         [
#             (
#                 "uvox",
#                 {
#                     "new",
#                     "create",
#                     "activate",
#                     "workon",
#                     "enter",
#                     "deactivate",
#                     "exit",
#                     "list",
#                     "ls",
#                     "remove",
#                     "rm",
#                     "delete",
#                     "del",
#                     "info",
#                     "runin",
#                     "runin-all",
#                     # "upgrade",
#                 },
#                 _HELP_OPTS,
#             ),
#             (
#                 "vox new",
#                 set(),
#                 {
#                     "-p",
#                     "--interpreter",
#                     "--prompt",
#                     "--system_site_packages",
#                     "--ssp",
#                 },
#             ),
#             ("vox activate", _VENV_NAMES, set()),
#             ("vox rm", _VENV_NAMES, _VOX_RM_OPTS),
#             ("vox rm venv1", _VENV_NAMES, _VOX_RM_OPTS),  # pos nargs: one or more
#             ("vox rm venv1 venv2", _VENV_NAMES, _VOX_RM_OPTS),  # pos nargs: two or more
#         ],
#     )
#     def test_vox_commands(self, args, positionals, opts, check, venvs):
#         check(args, positionals, opts)

#     @pytest.mark.parametrize(
#         "args",
#         [
#             "vox new --interpreter",  # "option: first
#             "vox new env1 --interpreter",  # option after pos
#             "vox new env1",
#             "vox new env1 --interpreter=",  # "option: at end with
#         ],
#     )
#     def test_interpreter(self, check, args, patched_cmd_cache):
#         check(args, _PY_BINS)
