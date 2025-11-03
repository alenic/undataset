from typing import Dict

from typing import Dict, Tuple

def map_merge_preserving_labels(
    map_1: Dict[int, str],
    map_2: Dict[int, str],
    name_1: str = "1",
    name_2: str = "2",
) -> Tuple[Dict[int, str], Dict[int, int], Dict[int, int]]:
    """
    Merge two label maps (id -> name) keeping all labels.
    If the *name* appears in both datasets, rename them to disambiguate:
      - labels from label_map_1 become "<name>_<name_1>"
      - labels from label_map_2 become "<name>_<name_2>"

    IDs from label_map_1 are preserved.
    IDs from label_map_2 are kept if free, otherwise reassigned to the next free integer.
    
    Returns:
      merged_label_map, id_map_1, id_map_2
        - merged_label_map: Dict[int, str] of the merged labels
        - id_map_1: mapping old_id_from_map1 -> new_id (identity)
        - id_map_2: mapping old_id_from_map2 -> new_id (after reassignment if needed)
    """
    # names that appear in both maps -> need suffixing
    names_1 = set(map_1.values())
    names_2 = set(map_2.values())
    conflict_names = names_1 & names_2

    def rename(name: str, suffix: str) -> str:
        return f"{name}_{suffix}"

    # 1) prepare map1 (rename only conflicting names)
    merged: Dict[int, str] = {}
    id_map_1: Dict[int, int] = {}
    for lid, lname in map_1.items():
        if lname in conflict_names:
            lname = rename(lname, name_1)
        merged[lid] = lname
        id_map_1[lid] = lid  # identity

    used_ids = set(merged.keys())

    # helper to get next free integer id
    def next_free_id(start: int = 0) -> int:
        cid = start
        while cid in used_ids:
            cid += 1
        return cid

    # 2) add map2 (rename conflicting names; avoid id collisions)
    id_map_2: Dict[int, int] = {}
    # sort for deterministic behavior
    for lid in sorted(map_2.keys()):
        lname = map_2[lid]
        if lname in conflict_names:
            lname = rename(lname, name_2)

        new_id = lid if lid not in used_ids else next_free_id(max(used_ids) + 1 if used_ids else 0)
        merged[new_id] = lname
        used_ids.add(new_id)
        id_map_2[lid] = new_id

    return merged, id_map_1, id_map_2
