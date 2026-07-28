"""Plugin migration root for the autocatpath `pathway` kind.

precis-mcp's migrator resolves the ``precis.migrations`` entry point
(``autocatpath = "autocatpath.precis.migrations"``) to *this package's directory*
and applies the ``*.sql`` files in it under the plugin namespace ``autocatpath``
(ADR 0005: forward-only, idempotent). The entry-point key is the namespace.
"""
