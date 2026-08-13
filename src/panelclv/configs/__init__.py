"""Declarations the pipeline is configured with — the bottom of the import stack.

Modules:
  - ``panel_config``       ``PanelConfig``: every column role, window date, embedding
                           declaration and engineered-time-feature flag, validated at
                           construction. Also the one table of those flags, which
                           ``data_preparation.add_time_features`` builds against.
  - ``ar_feature_names``   the grammar of ``PanelConfig.ar_features``: the supported
                           names and how one is read. Lives here, not beside the
                           computation, because the config validates the names and
                           nothing down here may import upward.

Nothing in this subpackage imports another ``panelclv`` subpackage, and that is the
point: every other subpackage is free to name it.
"""
