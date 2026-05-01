# ruff: noqa
import logging

fstr = "\x1b[30;3m%(asctime)s\x1b[0m %(shortname)-30s %(levelname)-10s <%(sim_time)-.2f> %(message)s"

colors = dict(
    GRAY=90,
    RED=91,
    GREEN=92,
    YELLOW=93,
    BLUE=94,
    MAGENTA=95,
    CYAN=96,
    WHITE=97,
    DARK_GRAY=30,
    DARK_RED=31,
    DARK_GREEN=32,
    DARK_YELLOW=33,
    DARK_BLUE=34,
    DARK_MAGENTA=35,
    DARK_CYAN=36,
    DARK_WHITE=37,
)


def style_string(
    string,
    no_format=False,
    style_spec=None,
    color=None,
    background_color=None,
    bold=False,
    emph=False,
    underline=False,
):
    if no_format:
        return string

    if style_spec is None:
        style_spec = []
    if color is not None:
        style_spec.append(colors[color.upper()])
    if background_color is not None:
        style_spec.append(colors[background_color.upper()] + 10)
    if bold:
        style_spec.append(1)
    if emph:
        style_spec.append(3)
    if underline:
        style_spec.append(4)
    return (
        "\x1b["
        + ";".join([str(style) for style in style_spec])
        + "m"
        + string
        + "\x1b[0m"
    )


level_color = {
    "DEBUG": None,
    "INFO": None,
    "WARNING": "YELLOW",
    "ERROR": "RED",
    "CRITICAL": "RED",
}
sat_color_cycle = [
    "DARK_CYAN",
    "GREEN",
    "DARK_BLUE",
    "MAGENTA",
    "CYAN",
    "DARK_GREEN",
    "BLUE",
    "DARK_MAGENTA",
]


class SimFormatter(logging.Formatter):
    def __init__(self, *args, color_output=True, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        self.color_output = color_output
        self.satellite_colors = {}

    def set_format(self, fmt, defaults=None, style="%"):
        self._style = logging._STYLES[style][0](fmt, defaults=defaults)
        self._fmt = self._style._fmt

    def format(self, record):
        no_format = not self.color_output
        fstr = ""
        fstr += style_string(
            "%(asctime)s ", color="GRAY", emph=True, no_format=no_format
        )

        sat_name = None
        sat_color = None
        try:
            if record.name.split(".")[1] == "sats":
                sat_name = record.name.split(".")[3]
                if sat_name not in self.satellite_colors:
                    self.satellite_colors[sat_name] = len(self.satellite_colors) % len(
                        sat_color_cycle
                    )
                sat_color = sat_color_cycle[self.satellite_colors[sat_name]]
        except IndexError:
            pass

        record.shortname = ".".join(record.name.split(".")[1:])
        fstr += style_string("%(shortname)-30s ", color=sat_color, no_format=no_format)
        fstr += style_string(
            "%(levelname)-10s ",
            color=level_color[record.levelname],
            bold=record.levelname == "CRITICAL",
            no_format=no_format,
        )
        if hasattr(record, "sim_time"):
            fstr += style_string(
                "<%(sim_time)-.2f> ", color="DARK_YELLOW", no_format=no_format
            )
        if sat_name is not None:
            fstr += style_string(
                f"{sat_name}: ",
                color=sat_color,
                no_format=no_format,
            )
        if record.msg.startswith("=== "):
            fstr += style_string(
                "%(message)s",
                bold=True,
                color="YELLOW",
                no_format=no_format,
            )
        else:
            fstr += style_string(
                "%(message)s",
                color=level_color[record.levelname],
                bold=record.levelname == "CRITICAL",
                no_format=no_format,
            )

        self.set_format(fstr)
        return super().format(record)


class ContextFilter(logging.Filter):
    def __init__(self, name: str = "", env=None, proc_id=None) -> None:
        super().__init__(name)
        self.env = env
        self.proc_id = proc_id

    def filter(self, record):
        try:
            record.sim_time = self.env.unwrapped.simulator.sim_time
        except AttributeError:
            pass
        return self.proc_id == record.process
