from xonsh.environ import Env
from xonsh.built_ins import XonshSession, XSH
from xonsh.tools import EnvPath


def get_env_safe(xession: XonshSession = XSH) -> Env:
    env = xession.env
    if not isinstance(env, Env):
        raise ValueError("Xonsh session has no environemnt")
    return env


def get_str_env_var(name: str, env: Env | None = None) -> str | None:
    if env is None:
        env = get_env_safe()
    if (var := env.get(name)) is None:
        return None

    if not isinstance(var, str):
        raise ValueError(f"str expected for env variable `{name}`, got {var!r}")
    return var


def get_path_var(env: Env | None = None) -> EnvPath:
    if env is None:
        env = get_env_safe()
    res = env["PATH"]
    if not isinstance(res, EnvPath):
        raise ValueError(
            f'Something is wrong with your Xonsh session: `${{...}}["PATH"]` is {res!r}'
        )
    return res
