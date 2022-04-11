# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../cosmic_shapes/for_docs'))
sys.path.insert(0, os.path.abspath('../../example_scripts'))


# -- Project information -----------------------------------------------------

project = 'cosmic_shapes'
copyright = '2022, Tibor Dome'
author = 'Tibor Dome'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.linkcode', # Use .viewcode for links to source in local folder, .linkcode for external source code links (e.g. GitHub).
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram'
]

# -- Inheritance graphs options --
inheritance_graph_attrs = dict(rankdir="TB", size='""') # TB=Top to bottom view

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['code_reference/setup.rst']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Linkcode configuration --

# Resolve function for the linkcode extension.
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        import inspect
        import os
        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(lasagne.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain not in ['py'] or not info['module']:
        return None
    try:
        filename = 'lasagne/%s#L%d-L%d' % find_source()
    except Exception:
        if info['module'] == 'gen_csh_gx_cat' or info['module'] == 'cython_helpers' or info['module'] == 'cosmic_shapes':
            extension = '.pyx'
        else:
            extension = '.py'
        filename = info['module'].replace('.', '/') + extension
    tag = 'master' if 'dev' in release else ('v' + release)
    return "https://github.com/tibordome/cosmic_shapes/%s" % (filename)

# -- Making sure __init__ of classes show up --
def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)
