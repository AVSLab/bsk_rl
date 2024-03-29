# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import datetime

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import re
import sys
from pathlib import Path

# sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))
now = datetime.datetime.now()

project = "BSK-RL"
copyright = str(now.year) + ", Autonomous Vehicle Systems (AVS) Laboratory"
author = "Mark Stephenson"
release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = ".rst"
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "style_nav_header_background": "#CFB87C",
    "navigation_depth": -1,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "./_images/static/Basilisk-Logo.png"

add_module_names = False


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


class FileCrawler:
    def __init__(self, base_source_dir, base_doc_dir):
        self.base_source_dir = base_source_dir
        self.base_doc_dir = base_doc_dir

    def grab_files(self, dir_path):
        dirs_in_dir = [x for x in dir_path.iterdir() if x.is_dir()]
        files_in_dir = dir_path.glob("*.py")

        # Remove any directories that shouldn't be added directly to the website
        dir_filters = [
            r".*__pycache__.*",
            r".*\.ruff_cache.*",
            r".*\.egg-info",
            r".*\/simplemaps_worldcities",
        ]
        dirs_in_dir = list(
            filter(
                lambda dir: not any(
                    re.match(filter, str(dir)) for filter in dir_filters
                ),
                dirs_in_dir,
            )
        )

        file_filters = [
            r".*__init__\.py",
            r"(.*\/|)_[a-zA-Z0-9_]*\.py",
        ]
        files_in_dir = list(
            filter(
                lambda file: not any(
                    re.match(filter, str(file)) for filter in file_filters
                ),
                files_in_dir,
            )
        )

        return sorted(list(files_in_dir)), sorted(list(dirs_in_dir))

    def populate_doc_index(self, index_path, file_paths, dir_paths, source_dir):
        name = index_path.stem
        lines = ""

        # if a _default.rst file exists in a folder, then use it to generate the index.rst file
        try:
            docFileName = source_dir / "_default.rst"
            with open(docFileName, "r") as docFile:
                docContents = docFile.read()
            lines += docContents + "\n\n"
        except FileNotFoundError:  # Auto-generate the index.rst file
            # add page tag
            qual_name = str(
                source_dir.relative_to(self.base_source_dir.parent)
            ).replace("/", ".")
            lines += ".. _" + qual_name.replace(" ", "_") + ":\n\n"

            # Title the page
            lines += name + "\n" + "=" * len(name) + "\n\n"
            lines += f"``{qual_name}``\n\n"

            # pull in folder _doc.rst file if it exists
            try:
                docFileName = source_dir / "_doc.rst"
                if os.path.isfile(docFileName):
                    with open(docFileName, "r") as docFile:
                        docContents = docFile.read()
                    lines += docContents + "\n\n"
            except FileNotFoundError:
                pass

            # Also check for docs in the __init__.py file
            lines += (
                """.. automodule:: """
                + qual_name
                + """\n   :members:\n   :show-inheritance:\n\n"""
            )

            # Add a linking point to all local files
            lines += (
                """\n\n.. toctree::\n   :maxdepth: 1\n   :caption: """ + "Files:\n\n"
            )
            added_names = []
            for file_path in sorted(file_paths):
                file_name = os.path.basename(os.path.normpath(file_path))
                file_name = file_name[: file_name.rfind(".")]

                if file_name not in added_names:
                    lines += "   " + file_name + "\n"
                    added_names.append(file_name)
            lines += "\n"

            # Add a linking point to all local directories
            lines += (
                """.. toctree::\n   :maxdepth: 1\n   :caption: """ + "Directories:\n\n"
            )

            for dir_path in sorted(dir_paths):
                dirName = os.path.basename(os.path.normpath(dir_path))
                lines += "   " + dirName + "/index\n"

        with open(os.path.join(index_path, "index.rst"), "w") as f:
            f.write(lines)

    def generate_autodoc(self, doc_path, source_file):
        short_name = source_file.name.replace(".py", "")
        qual_name = (
            str(source_file.relative_to(self.base_source_dir.parent))
            .replace("/", ".")
            .replace(".py", "")
        )

        # Generate the autodoc file
        lines = ".. _" + qual_name + ":\n\n"
        lines += short_name + "\n" + "=" * len(short_name) + "\n\n"
        lines += f"``{qual_name}``\n\n"
        lines += """.. toctree::\n   :maxdepth: 1\n   :caption: """ + "Files" + ":\n\n"
        lines += (
            """.. automodule:: """
            + qual_name
            + """\n   :members:\n   :show-inheritance:\n\n"""
        )

        # Write to file
        with open(doc_path / f"{short_name}.rst", "w") as f:
            f.write(lines)

    def run(self, source_dir=None):
        if source_dir is None:
            source_dir = self.base_source_dir

        file_paths, dir_paths = self.grab_files(source_dir)
        index_path = source_dir.relative_to(self.base_source_dir)

        # Populate the index.rst file of the local directory
        os.makedirs(self.base_doc_dir / index_path, exist_ok=True)
        self.populate_doc_index(
            self.base_doc_dir / index_path, file_paths, dir_paths, source_dir
        )

        # Generate the correct auto-doc function for python modules
        for file in file_paths:
            self.generate_autodoc(
                self.base_doc_dir / index_path,
                file,
            )

        # Recursively go through all directories in source, documenting what is available.
        for dir_path in sorted(dir_paths):
            self.run(
                source_dir=dir_path,
            )

        return


sys.path.append(os.path.abspath("../.."))
FileCrawler(Path("../../src/bsk_rl/"), Path("./API Reference/")).run()
FileCrawler(Path("../../examples"), Path("./Examples/")).run()
