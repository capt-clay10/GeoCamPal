"""
feature_model.py  —  GeoCamPal Feature Identifier label model
=============================================================

Purpose
-------
Small, dependency-free helpers that let each identified feature carry
its own label (for example "shoreline" or "dune") while staying
compatible with every piece of code that treats a feature as a plain
``(feature_type, points)`` 2-tuple.

Feature
    A ``tuple`` subclass whose payload is exactly ``(feature_type,
    points)``.  Unpacking (``for ftype, pts in features``), indexing
    (``f[1]``), ``len(f) == 2`` and equality behave exactly like the
    plain tuples produced by the detectors.  The label is an extra
    attribute.  A plain tuple, or a Feature with an empty label, has
    no label of its own: it takes the Feature ID typed in the
    configuration window at export time, which is how every release
    before per-feature labels behaved.

slug
    The filesystem-safe form of a label used in export file names
    (lower case, spaces to underscores, only letters, digits, ``_``
    and ``-``).  Two labels with the same slug are the same label,
    because they would write the same files.

LabelCatalog
    Ordered mapping of label to category ID, so a label keeps the same
    category ID across features, images and sessions.  Persisted in the
    settings JSON under ``label_catalog`` and, for multi-label exports,
    in ``training dataset/multiclass/categories.json``.
"""

__all__ = ["Feature", "feature_label", "with_label", "slug", "LabelCatalog"]


class Feature(tuple):
    """``(feature_type, points)`` 2-tuple that also carries a label."""

    def __new__(cls, ftype, pts, label=""):
        obj = super().__new__(cls, (ftype, pts))
        obj.label = (label or "").strip()
        return obj

    @property
    def ftype(self):
        return self[0]

    @property
    def pts(self):
        return self[1]

    def replace(self, pts=None, label=None):
        """Return a copy with new points and/or a new label."""
        return Feature(self[0], self[1] if pts is None else pts,
                       self.label if label is None else label)

    def __reduce__(self):
        # copy / deepcopy / pickle keep the label
        return (Feature, (self[0], self[1], self.label))

    def __repr__(self):
        return (f"Feature({self[0]!r}, <{len(self[1])} pts>, "
                f"label={self.label!r})")


def feature_label(feat):
    """Return the feature's own label, or "" for a plain tuple."""
    return (getattr(feat, "label", "") or "").strip()


def with_label(feat, label):
    """Return *feat* as a Feature carrying *label* (points untouched)."""
    return Feature(feat[0], feat[1], label)


def slug(text):
    """Filesystem-safe suffix for a label (same rule as export file names)."""
    if not text:
        return ""
    # runs of blanks count as one, so "sand  bar" and "sand bar" are one label
    s = "_".join(str(text).strip().lower().split())
    return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-"))


class LabelCatalog:
    """
    Ordered ``slug -> (display name, category_id)`` mapping.

    Identity is the slug, so "Sand bar" and "sand_bar" are one label;
    the first spelling seen is kept as the display name.  Category IDs
    are positive integers and each one belongs to at most one label.
    """

    def __init__(self, entries=None):
        self._entries = {}          # slug -> [display_name, category_id]
        if entries:
            self.merge(entries)

    # ── queries ──
    def __len__(self):
        return len(self._entries)

    def __contains__(self, label):
        return slug(label) in self._entries

    def id_for(self, label):
        e = self._entries.get(slug(label))
        return e[1] if e else None

    def name_for(self, label):
        e = self._entries.get(slug(label))
        return e[0] if e else (label or "").strip()

    def owner_of(self, category_id):
        """Display name of the label that owns *category_id*, or None."""
        for name, cid in self._entries.values():
            if cid == category_id:
                return name
        return None

    def names(self):
        return [e[0] for e in self._entries.values()]

    def used_ids(self):
        return {e[1] for e in self._entries.values()}

    def next_free_id(self, reserved=()):
        used = self.used_ids() | {int(r) for r in reserved if r is not None}
        cid = 1
        while cid in used:
            cid += 1
        return cid

    # ── updates ──
    def ensure(self, label, preferred_id=None, reserved=()):
        """
        Return the category ID of *label*, registering it if needed.

        A new label gets *preferred_id* when that ID is free, otherwise
        the smallest free ID that is not in *reserved*.
        """
        key = slug(label)
        if key in self._entries:
            return self._entries[key][1]
        name = (label or "").strip()
        used = self.used_ids() | {int(r) for r in reserved if r is not None}
        if preferred_id is not None and int(preferred_id) >= 1 \
                and int(preferred_id) not in used:
            cid = int(preferred_id)
        else:
            cid = self.next_free_id(reserved)
        self._entries[key] = [name, cid]
        return cid

    def set_id(self, label, category_id):
        """
        Give *label* the ID *category_id* (registering the label if new).
        Raises ValueError if another label already owns that ID.
        """
        cid = int(category_id)
        if cid < 1:
            raise ValueError("Category ID must be 1 or larger.")
        key = slug(label)
        owner = self.owner_of(cid)
        if owner is not None and slug(owner) != key:
            raise ValueError(
                f"Category ID {cid} is already used by '{owner}'.")
        if key in self._entries:
            self._entries[key][1] = cid
        else:
            self._entries[key] = [(label or "").strip(), cid]

    def merge(self, entries):
        """
        Add entries from a list of ``{"feature_id": name, "category_id": id}``
        (or ``{"name": ..., "id": ...}``) dicts.  Entries that conflict
        with what is already known are skipped and reported.

        Returns a list of human-readable conflict strings.
        """
        conflicts = []
        for item in entries or []:
            if not isinstance(item, dict):
                continue
            name = item.get("feature_id", item.get("name", ""))
            cid = item.get("category_id", item.get("id"))
            name = (str(name) if name is not None else "").strip()
            try:
                cid = int(cid)
            except (TypeError, ValueError):
                continue
            if not slug(name) or cid < 1:
                continue
            known = self.id_for(name)
            if known is not None:
                if known != cid:
                    conflicts.append(
                        f"'{self.name_for(name)}' has category ID {known} "
                        f"here but {cid} in the other list")
                continue
            owner = self.owner_of(cid)
            if owner is not None:
                conflicts.append(
                    f"category ID {cid} is '{owner}' here but '{name}' "
                    f"in the other list")
                continue
            self._entries[slug(name)] = [name, cid]
        return conflicts

    def to_list(self):
        """Entries sorted by category ID, in settings-file form."""
        rows = sorted(self._entries.values(), key=lambda e: e[1])
        return [{"feature_id": name, "category_id": cid} for name, cid in rows]

    def copy(self):
        new = LabelCatalog()
        new._entries = {k: list(v) for k, v in self._entries.items()}
        return new
