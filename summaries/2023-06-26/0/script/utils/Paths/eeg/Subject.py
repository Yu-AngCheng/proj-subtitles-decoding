#!/usr/bin/env python3
"""
Created on 21:37, Dec. 23rd, 2022

@author: Norbert Zheng
"""
import os
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
from utils import DotDict

__all__ = [
    "Subject",
]

class Subject:
    """
    `Subject` class of the specified subject.
    """

    def __init__(self, base):
        """
        Initialize `Subject` object of specified subject. The subject path contains the following files:
         - 202[2,3]...: The path of runs.
         - name: The name of current subject.
        :param base: The base path of specified subject.
        """
        # Initialize parameters.
        self.base = base
        # Initialize variables.
        self._init_metadatas()

    """
    init funcs
    """
    # def _init_metadatas func
    def _init_metadatas(self):
        """
        Initialize the metadata of specified subject. All metadatas are loaded from behavior files.
        """
        # Initialize name of current `Subject` object.
        path_name = os.path.join(self.base, "name")
        if os.path.exists(path_name):
            with open(path_name, "r") as f:
                self.name = f.read().strip()
        else:
            self.name = None
        # Initialize the path of dataset.
        path_runs = [path_i for path_i in os.listdir(self.base) if path_i.isdigit()]; path_runs.sort()
        path_runs = [os.path.join(self.base, path_i) for path_i in path_runs]
        self.dataset = [os.path.join(path_i, "dataset") for path_i in path_runs]

if __name__ == "__main__":
    # macro
    base = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
    path_data = os.path.join(base, "data", "demo")
    path_subject = os.path.join(path_data, "004")

    # Instantiate Subject.
    subject_inst = Subject(path_subject)

