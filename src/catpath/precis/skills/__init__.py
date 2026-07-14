"""Plugin skill root for the catpath `pathway` kind.

precis-mcp's skill handler resolves the ``precis.skills`` entry point
(``catpath = "catpath.precis.skills"``) to this package's directory and serves
every ``*.md`` in it via ``get(kind='skill', id=…)``. Built-ins win slug
collisions, so plugin skills use a distinct ``precis-<name>-help`` slug.
"""
