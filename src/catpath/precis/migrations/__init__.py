"""Plugin migration root for the catpath `pathway` kind.

precis-mcp's migrator resolves the ``precis.migrations`` entry point
(``catpath = "catpath.precis.migrations"``) to *this package's directory*
and applies the ``*.sql`` files in it under the plugin namespace ``catpath``
(ADR 0005: forward-only, idempotent). The entry-point key is the namespace.
"""
