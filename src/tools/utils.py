import re
import errno
import os
import pandas as pd
from typing import Any
from pathlib import Path

def convert_table_text_to_pandas(table_text):
    _data = {}

    table_text = re.sub(r" ?\| ?", " | ", table_text)
    cells = [row.split(" | ") for row in table_text.split("\n")]

    row_num = len(cells)
    column_num = len(cells[0])

    # for table without a header
    first_row = cells[0]
    matches = re.findall(r"[\d]+", " ".join(first_row))
    if len(matches) > 0:
        header = [f"Column {i+1}" for i in range(column_num)]
        cells.insert(0, header)

    # build DataFrame for the table
    for i in range(column_num):
        _data[cells[0][i]] = [row[i] for row in cells[1:]]

    table_pd = pd.DataFrame.from_dict(_data)

    return table_pd


class Logger:
    def __init__(self, outfile = None):
        from rich.console import Console
        self.std_console = Console()
        self.file_console = Console(file=open(outfile, "w")) if outfile else None

    def log(
        self,
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        stdout = True,
        file = True,
    ):
        if stdout: self.std_console.print(*objects, sep=sep, end=end)
        if file and self.file_console: self.file_console.print(*objects, sep=sep, end=end)

def symlink_force(target: Path, link_name: Path):
    import errno
    try:
        # os.symlink(target, link_name)
        link_name.symlink_to(target)
    except OSError as e:
        if e.errno == errno.EEXIST:
            link_name.unlink()
            link_name.symlink_to(target)
            # os.remove(link_name)
            # os.symlink(target, link_name)
        else:
            raise e
