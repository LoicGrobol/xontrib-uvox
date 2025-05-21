from xonsh.environ import Env
from xonsh.built_ins import XonshSession, XSH


def get_env_safe(xession: XonshSession = XSH) -> Env:
    env = xession.env
    if not isinstance(env, Env):
        raise ValueError("Xonsh session has no environemnt")
    return env
