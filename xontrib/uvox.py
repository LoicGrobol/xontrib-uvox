"""Python virtual environment manager for xonsh."""

import subprocess
from typing import Annotated, Sequence

import xonsh.cli_utils as xcli
from xonsh.built_ins import XonshSession
from xonsh.completers.path import complete_dir
from xonsh.platform import ON_WINDOWS

import xontrib.uvoxapi as uvoxapi

__all__ = ()


def venv_names_completer(command, alias: "UvoxHandler", **_):
    envs = alias.uvox.list_envs()

    yield from envs

    paths, _ = complete_dir(command)
    yield from paths


def py_interpreter_path_completer(xsh, **_):
    for _, (path, is_alias) in xsh.commands_cache.all_commands.items():
        if not is_alias and ("/python" in path or "/pypy" in path):
            yield path


class UvoxHandler(xcli.ArgParserAlias):
    """UVox is a UV-backed virtual environment manager for xonsh."""

    def build(self):
        """lazily called during dispatch"""
        self.uvox = uvoxapi.Uvox()
        parser = self.create_parser(prog="uvox")

        parser.add_command(self.new, aliases=["create"])
        parser.add_command(self.activate, aliases=["workon", "enter"])
        parser.add_command(self.deactivate, aliases=["exit"])
        parser.add_command(self.list, aliases=["ls"])
        parser.add_command(self.remove, aliases=["rm", "delete", "del"])
        parser.add_command(self.info)
        parser.add_command(self.runin)
        parser.add_command(self.runin_all)

        return parser

    def hook_post_add_argument(self, action, param: str, **_):
        if param == "interpreter":
            action.completer = py_interpreter_path_completer

    def activate(
        self,
        name: Annotated[
            str | None,
            xcli.Arg(metavar="ENV", nargs="?", completer=venv_names_completer),
        ] = None,
    ):
        """Activate a virtual environment.

        Parameters
        ----------
        name
            The environment to activate.
            ENV can be either a name from the venvs shown by `uvox list`
            or the path to an arbitrary venv
        """

        if name is None:
            return self.list()

        try:
            self.uvox.activate(name)
        except ValueError as e:
            raise self.Error(
                f"This environment doesn't exist. Create it with 'uvox new {name}'",
            ) from e

        self.out(f'Activated "{name}".')

    def deactivate(self):
        """Deactivate the active virtual environment."""

        if self.uvox.active() is None:
            raise self.Error(
                'No environment currently active. Activate one with "uvox activate".',
            )
        self.uvox.deactivate()

    def info(
        self,
        name: Annotated[
            str | None,
            xcli.Arg(metavar="ENV", nargs="?", completer=venv_names_completer),
        ] = None,
    ):
        """Prints the path for the supplied env

        Parameters
        ----------
        venv
            name of the venv
        """
        if name is None:
            self.out(self.uvox.active())
        else:
            self.out(self.uvox.get_env(name))

    def list(self):
        """List available virtual environments."""

        envs = sorted(self.uvox.list_envs())

        if not envs:
            raise self.Error(
                'No environments available. Create one with "uvox new".',
            )
        for e in envs:
            self.out(e)

    def new(
        self,
        name: Annotated[str, xcli.Arg(metavar="ENV")],
        interpreter: str | None = None,
        system_site_packages=False,
        prompt: str | None = None,
    ):
        """Create a virtual environment in $VIRTUALENV_HOME with uv.

        Parameters
        ----------
        name : str
            Virtual environment name
        interpreter: -p, --interpreter
            Python interpreter used to create the virtual environment.
            Can be configured via the $VOX_DEFAULT_INTERPRETER environment variable.
        system_site_packages : --system-site-packages, --ssp
            If True, the system (global) site-packages dir is available to
            created environments.
        prompt: --prompt
            Provides an alternative prompt prefix for this environment.
        """

        self.out(f"Creating environment {name}…")

        self.uvox.create(
            name,
            system_site_packages=system_site_packages,
            interpreter=interpreter,
            prompt=prompt,
        )

        self.out(f'Environment {name!r} created. Activate it with "uvox activate {name}"')

    def remove(
        self,
        names: Annotated[
            Sequence[str],
            xcli.Arg(metavar="ENV", nargs="+", completer=venv_names_completer),
        ],
        force=False,
    ):
        """Remove virtual environments.

        Parameters
        ----------
        names
            The environments to remove. ENV can be either a name from the venvs shown by uvox
            list or the path to an arbitrary venv
        force : -f, --force
            Delete virtualenv without prompt
        """
        for name in names:
            try:
                self.uvox.del_env(name, silent=force)
            except uvoxapi.EnvironmentInUseError as e:
                raise self.Error(
                    f'The "{name}" environment is currently active. '
                    'In order to remove it, deactivate it first with "uvox deactivate".',
                ) from e
            except ValueError as e:
                raise self.Error(f'"{name}" environment doesn\'t exist.') from e
            else:
                self.out(f'Environment "{name}" removed.')
        self.out()

    def runin(
        self,
        venv: Annotated[
            str,
            xcli.Arg(completer=venv_names_completer),
        ],
        args: Annotated[Sequence[str], xcli.Arg(nargs="...")],
    ):
        """Run the command in the given environment

        Parameters
        ----------
        venv
            The environment to run the command for
        args
            The actual command to run

        Examples
        --------
            uvox runin venv1 black --check-only
        """
        if not args:
            raise self.Error("No command is passed")
        env = self.uvox.get_venv_env(venv)
        try:
            return subprocess.check_call(args, shell=bool(ON_WINDOWS), env=env)
            # need to have shell=True on windows, otherwise the PYTHONPATH
            # won't inherit the PATH
        except OSError as e:
            if e.errno == 2:
                raise self.Error(f"Unable to find {args[0]}") from e
            raise e

    def runin_all(
        self,
        args: Annotated[Sequence[str], xcli.Arg(nargs="...")],
    ):
        """Run the command in all environments found under $VIRTUALENV_HOME

        Parameters
        ----------
        args
            The actual command to run with arguments
        """
        errors = False
        for env in self.uvox.list_envs():
            try:
                self.runin(env, *args)
            except subprocess.CalledProcessError as e:
                errors = True
                self.err(e)
        self.parser.exit(errors)


def _load_xontrib_(xsh: XonshSession, **_):
    assert xsh.aliases is not None
    xsh.aliases["uvox"] = UvoxHandler(threadable=False)
