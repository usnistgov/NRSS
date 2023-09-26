# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "NRSS"
author = "Peter Dudenas"
release = "0.1.0"
copyright = (
    ": Official Contribution of the US Government.  Not subject to copyright in the United States"
)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../NRSS/"))


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_design",
]  # numpydoc and google docstrings

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "pydata_sphinx_theme"
html_logo = "_static/Logo_NRSS.svg"
html_theme_options = {
    "logo": {
        "image_light": "_static/Logo_NRSS.svg",
        "image_dark": "_static/Logo_NRSS.svg",
    },
    "github_url": "https://github.com/usnistgov/NRSS",
    "collapse_navigation": True,
    "external_links": [
        {
            "name": "NIST RSoXS project",
            "url": "https://www.nist.gov/programs-projects/resonant-soft-x-ray-scattering-rsoxs",
        },
        {
            "name": "PyHyperScattering",
            "url": "https://pages.nist.gov/PyHyperScattering/en/main/index.html",
        },
    ],
    "header_links_before_dropdown": 6,
    # Add light/dark mode and documentation version switcher:
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
