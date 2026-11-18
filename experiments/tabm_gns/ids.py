"""Identity of a TabGNS-TabM result, shared by the runner (controls, saved masks) and the report."""
from __future__ import annotations

import hashlib
import json
from typing import Dict


def source_id(rec: Dict) -> str:
  """Identity of a TabGNS-TabM result as far as its controls depend on it: the search that produced it
  (architecture and loss histories), the found widths and the parameter target. Stored on every control
  and in the saved masks, so that a rerun of the source invalidates them."""
  key = {k: rec.get(k) for k in ('dataset', 'method', 'seed', 'widths')}
  key['params_full'] = rec['params']['full']
  key['arch_history'] = rec['search']['arch_history']
  key['loss_history'] = [list(x) for x in rec['search']['loss_history']]
  return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
