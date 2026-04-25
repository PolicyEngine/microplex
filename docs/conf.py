"""Sphinx configuration for the microplex documentation."""

project = "microplex"
author = "Cosilico"
copyright = "2026, Cosilico"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
root_doc = "index"

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]

autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
typehints_fully_qualified = False

html_theme = "pydata_sphinx_theme"
html_title = "microplex"
html_theme_options = {
    "github_url": "https://github.com/CosilicoAI/microplex",
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "CosilicoAI",
    "github_repo": "microplex",
    "github_version": "main",
    "doc_path": "docs",
}
