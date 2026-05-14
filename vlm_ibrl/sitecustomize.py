"""Process-level compatibility shims for the demo2reward env (Python 3.10+).

Loaded automatically by Python's ``site.py`` when this directory is on
``PYTHONPATH`` (which ``set_env.sh`` arranges). Lives in repo source so no
package-manager-controlled files are modified.

Currently only fixes one breakage:

  ``from collections import Iterable``

This was removed in Python 3.10 (moved to ``collections.abc.Iterable`` back
in 3.3). The robosuite commit Chris pins
(``de64fa5935f9f30ce01b36a3ef1a3242060b9cdb``) still does the old import in
``robosuite/models/arenas/multi_table_arena.py``. We rebind the legacy name
on the ``collections`` module so the import succeeds. Single localized
attribute set; not a full backport.
"""

import collections
import collections.abc as _abc

for _name in (
    "Iterable", "Iterator", "Container", "Hashable", "Sized",
    "Callable", "Mapping", "MutableMapping", "Sequence", "MutableSequence",
    "Set", "MutableSet",
):
    if not hasattr(collections, _name) and hasattr(_abc, _name):
        setattr(collections, _name, getattr(_abc, _name))
del _name, _abc
