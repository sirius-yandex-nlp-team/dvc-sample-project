from pathlib import Path

import attr

from dvc_sample_project.utils.fs import find_parent_containing

@attr.s
class Context:
    root_dir = attr.ib(default=None, type=Path)
    def __attrs_post_init__(self):
        self.root_dir = find_parent_containing(".git")

    @property
    def data_dir(self):
        return self.root_dir / "data"

ctx = context = Context()
