UVox
====

A [uv](https://github.com/astral-sh/uv)-based Python virtual environment manager for the xonsh shell.

## Installation

To install use pip:

```bash
xpip install xontrib-uvox
```

## Usage

This package contains two xontribs:

- `uvox` - Python virtual environment manager for xonsh.
- `uvoxapi` - API for Uvox

### uvox

Python virtual environment manager for xonsh.

```bash
xontrib load uvox
uvox --help
```

### uvoxapi

```bash
xontrib load uvoxapi
```

UVox defines several events related to the life cycle of virtual environments:

* `uvox_on_create(env: str) -> None`
* `uvox_on_activate(env: str, path: pathlib.Path) -> None`
* `uvox_on_deactivate(env: str, path: pathlib.Path) -> None`
* `uvox_on_delete(env: str) -> None`


## Credits

This package is a fork of [xontrib-vox](https://github.com/xonsh/xontrib-vox).
