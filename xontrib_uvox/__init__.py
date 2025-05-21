from xonsh.built_ins import XonshSession
from xontrib_uvox.uvox import UvoxHandler


def _load_xontrib_(xsh: XonshSession, **kwargs):
    assert xsh.aliases is not None
    xsh.aliases["uvox"] = UvoxHandler(threadable=False)
