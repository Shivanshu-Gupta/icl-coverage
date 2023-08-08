# Modified by Shivanshu Gupta from https://gist.github.com/tbenthompson/9db0452445451767b59f5cb0611ab483

"""
Usage:

A dataclass/YAML/CLI config system:
- write a @dataclass with your config options
- make sure every option has a default value
- include a `config: str = ""` option in the dataclass.
- write a main function that takes a single argument of the dataclass type
- decorate your main function with @dataclass_cli
- make sure your main function has a docstring.

The config will be loaded from a YAML file specified by the --config option,
and CLI options will override the config file.

Example from running this file:

> python edit/config.py --help

 Usage: config.py [OPTIONS]

 test

╭─ Options
│ --config        TEXT
│ --hi            INTEGER  [default: 1]
│ --bye           TEXT     [default: bye]
│ --help                   Show this message and exit.
╰─
"""

import attr
import dataclasses
import inspect
import typing

import typer
import yaml
from tools.param_impl import DictDataClass

app = typer.Typer()

def conf_callback(ctx: typer.Context, param: typer.CallbackParam, value: str) -> str:
    """
    Callback for typer.Option that loads a config file from the first
    argument of a dataclass.

    Based on https://github.com/tiangolo/typer/issues/86#issuecomment-996374166
    """
    if param.name == "config" and value:
        typer.echo(f"Loading config file: {value}")
        try:
            with open(value, "r") as f:
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}
            ctx.default_map.update(conf)
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return value


def dataclass_cli(func):
    """
    Converts a function taking a dataclass as its first argument into a
    dataclass that can be called via `typer` as a CLI.

    Additionally, the --config option will load a yaml configuration before the
    other arguments.

    Modified from:
    - https://github.com/tiangolo/typer/issues/197

    A couple related issues:
    - https://github.com/tiangolo/typer/issues/153
    - https://github.com/tiangolo/typer/issues/154
    """
    # The dataclass type is the first argument of the function.
    sig = inspect.signature(func)
    # breakpoint()
    param = list(sig.parameters.values())[0]
    cls = param.annotation
    assert dataclasses.is_dataclass(cls) or attr.has(cls), f"{cls} is not a dataclass"

    def wrapped(**kwargs):
        # Load the config file if specified.
        if kwargs.get("config", "") != "":
            with open(kwargs["config"], "r") as f:
                conf = yaml.safe_load(f)
        else:
            conf = {}

        # CLI options override the config file.
        conf.update(kwargs)

        # Convert back to the original dataclass type.
        if attr.has(cls):

            arg = cls.from_dict(conf)
        else:
            arg = cls(**conf)

        # Actually call the entry point function.
        return func(arg)

    # To construct the signature, we remove the first argument (self)
    # from the dataclass __init__ signature.
    signature = inspect.signature(cls.__init__)
    parameters = list(signature.parameters.values())
    types = typing.get_type_hints(cls.__init__)
    def get_docstring(key):
        from dataclasses import asdict
        from simple_parsing.docstring import get_attribute_docstring, AttributeDocString
        all_docstrings: AttributeDocString = get_attribute_docstring(cls, key)
        doc_list = [d for d in asdict(all_docstrings).values() if d]
        return '\n'.join(doc_list)

    if len(parameters) > 0 and parameters[0].name == "self":
        del parameters[0]
        from pathlib import Path
        parameters = [inspect.Parameter(
                p.name,
                kind=p.kind,
                default=typer.Option(p.default, help=get_docstring(p.name)),
                annotation=types[p.name],
            ) for p in parameters]
    # Add the --config option to the signature.
    # When called through the CLI, we need to set defaults via the YAML file.
    # Otherwise, every field will get overwritten when the YAML is loaded.
    parameters = [
        inspect.Parameter(
            "config",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=typer.Option("", callback=conf_callback, is_eager=True),
        )
    ] + [p for p in parameters if p.name != "config"]

    # The new signature is compatible with the **kwargs argument.
    wrapped.__signature__ = signature.replace(parameters=parameters)

    # The docstring is used for the explainer text in the CLI.
    if func.__doc__:
        wrapped.__doc__ = func.__doc__ + "\n" + ""
    wrapped.__name__ = func.__name__

    return wrapped


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # @dataclasses.dataclass
    @attr.s(auto_attribs=True)
    class Test(DictDataClass):
        config: str = ""
        hi: int = 1
        bye: str = "bye"

    @app.command()
    @dataclass_cli
    def main(c: Test):
        """test"""
        print(c.hi, c.bye)
        return str(c.hi) + c.bye

    # The function can either be called directly using the dataclass
    # parameters:
    assert main(hi=2, bye="hello") == "2hello"

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        yaml.dump({"hi": 3, "bye": "yummy"}, temp_file)
        # Including a config file:
        assert main(config=temp_file.name) == "3yummy"
        # CLI options override the config file:
        assert main(config=temp_file.name, hi=15) == "15yummy"

    # We can also call directly via the CLI:
    # typer.run(main)
    app()

