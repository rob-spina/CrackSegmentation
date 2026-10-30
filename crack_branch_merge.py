"""crack_branch_merge.py
v1.5.0 Merge crack branches, mixed into CrackSegmentation (smart_segmentation.py).
Makes several nearby cracks the branches of one crack:
  - the sidebar [Merge crack branches] button or the Tools menu (code 134) turns the mode on;
  - the FIRST crack clicked (its line or its number) is the main crack;
  - every next crack clicked becomes one of its branches. A branch that stops short of the main crack (or of one
    of its branches) is joined to it: along the real edge when one is found, else with a straight bridge, as LINK
    does. A crack that is already the main crack of other branches brings them along;
  - Shift+click on the button, Shift+C, or the Tools menu exit command (code 135) end the mode. Another command
    (Cut, Join, a tool, Save and next ...) ends it too; view keys, Undo and Redo leave it on.
The cracks stay separate lines, so every edit tool keeps working on them, but they share one LabelMe group_id in
the JSON (flags "branch_main" / "branch"): one crack object with its branches. Their markers read N, N.1, N.2 ...
JSON and binary masks are written after every branch; [U] undoes the last one.
"""
import uuid

import cv2
import numpy as np

from crack_relabel import RELABEL_KEEP_KEYS, RELABEL_TARGET_BY_CODE


MERGE_BRANCH_CODE = 134        # C1 control range, like 128-133: never produced by a typed character
MERGE_BRANCH_EXIT_CODE = 135
MERGE_BRANCH_EXIT_KEY = ord('C')   # Shift+C
# Keys that leave the mode on: view keys (as for the relabel commands), Undo/Redo, Save and the mode's own codes.
MERGE_BRANCH_KEEP_KEYS = (RELABEL_KEEP_KEYS | frozenset(RELABEL_TARGET_BY_CODE)
                          | frozenset([ord('u'), ord('U'), ord('r'), ord('R'), ord('s'), ord('S'),
                                       130, 131, 133, MERGE_BRANCH_CODE, MERGE_BRANCH_EXIT_CODE]))
BRANCH_TOUCH_PX = 3   # a branch this close to the main crack already touches it: no bridge


class CrackBranchMergeMixin:
    """Groups cracks into a main crack with branches. Relies on CrackSegmentation's cfg, crack pickers,
    LINK bridge helpers and save helpers."""

    # --- Group helpers --------------------------------------------------------------------

    @staticmethod
    def _crack_group_id(crack):
        return crack.get('crack_group')

    def _group_members(self, gid):
        """Indices of the cracks of group gid, main crack first."""
        if gid is None:
            return []
        idx = [i for i, c in enumerate(self.cfg.saved_cracks) if c.get('crack_group') == gid]
        idx.sort(key=lambda i: 0 if self.cfg.saved_cracks[i].get('branch_main') else 1)
        return idx

    def _group_main_index(self, gid):
        for i in self._group_members(gid):
            if self.cfg.saved_cracks[i].get('branch_main'):
                return i
        return None

    def _new_crack_group_id(self):
        used = {c.get('crack_group') for c in self.cfg.saved_cracks}
        while True:
            gid = int(uuid.uuid4().int % 1000000)
            if gid not in used:
                return gid

    def _crack_marker_label(self, idx):
        """Marker text of crack idx: its number, or N.k for the k-th branch of main crack N."""
        crack = self.cfg.saved_cracks[idx]
        if not crack.get('branch'):
            return idx + 1
        main = self._group_main_index(crack.get('crack_group'))
        if main is None:
            return idx + 1
        branches = [i for i in self._group_members(crack['crack_group']) if self.cfg.saved_cracks[i].get('branch')]
        return f"{main + 1}.{branches.index(idx) + 1}"

    def _strip_lone_crack_groups(self):
        """A main crack left without branches (mode closed before any branch, or Undo) is a plain crack again."""
        counts = {}
        for c in self.cfg.saved_cracks:
            gid = c.get('crack_group')
            if gid is not None:
                counts[gid] = counts.get(gid, 0) + 1
        for c in self.cfg.saved_cracks:
            if c.get('crack_group') is not None and counts[c['crack_group']] < 2:
                for k in ('crack_group', 'branch', 'branch_main'):
                    c.pop(k, None)

    # --- Mode -------------------------------------------------------------------------------

    def _start_crack_branch_merge(self):
        """Code 134: turns the mode on. While it is on, a plain click on the button only reminds how to exit."""
        state = self.cfg.crack_branch_merge_state
        if state["active"]:
            print(" [MERGE BRANCH] Already on: click the branches; Shift+click on the button or Shift+C to exit.")
            self.refresh_zoom_viewport()
            return
        state["active"], state["main_group"] = True, None
        print(" [MERGE BRANCH] On: click the MAIN crack (its line or its number), then every crack to add as its "
              "branch. Shift+click on the button or Shift+C to exit.")
        self.refresh_zoom_viewport()

    def _exit_crack_branch_merge(self, reason=None):
        state = self.cfg.crack_branch_merge_state
        if not state["active"]:
            return
        state["active"], state["main_group"] = False, None
        self._strip_lone_crack_groups()
        print(f" [MERGE BRANCH] Off{': ' + reason if reason else ''}. Clicks trace again.")
        self.refresh_zoom_viewport()

    def _handle_crack_branch_merge_key(self, key_clean):
        """Shift+C / code 135 end the mode; another command ends it too. True when the key was consumed."""
        if not self.cfg.crack_branch_merge_state["active"]:
            return key_clean == MERGE_BRANCH_EXIT_CODE   # exit with the mode already off: nothing to do
        if key_clean in (MERGE_BRANCH_EXIT_KEY, MERGE_BRANCH_EXIT_CODE):
            self._exit_crack_branch_merge()
            return True
        if key_clean not in MERGE_BRANCH_KEEP_KEYS:
            self._exit_crack_branch_merge("another command was chosen")
        return False

    def _draw_crack_branch_merge_label(self, win_out):
        """Mode line (status bar in the GUI) while the mode is on."""
        state = self.cfg.crack_branch_merge_state
        if not state["active"]:
            return
        large = self.cfg.hud_large_size
        y = (135 if large else 120) + 7 * (45 if large else 30)
        main = self._group_main_index(state["main_group"])
        if main is None:
            text = "MERGE CRACK BRANCHES: click the MAIN crack (Shift+C = exit)"
        else:
            text = f"MERGE CRACK BRANCHES: main crack {main + 1} -- click its branches (Shift+C = exit)"
        self._put_hud_text(win_out, text, (15, y), (0, 200, 0))

    # --- Click ------------------------------------------------------------------------------

    def _handle_crack_branch_merge_mouse(self, event, x, y):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        idx = self._crack_under_click(x, y)
        if idx is None:
            print(" [MERGE BRANCH] No crack there: click ON a crack line or on its number (Shift+C exits).")
            return
        state = self.cfg.crack_branch_merge_state
        main = self._group_main_index(state["main_group"])
        if main is None:
            self._set_branch_main(idx)
        elif self._merge_crack_as_branch(main, idx):
            self.recalculate_masks()
            self._save_annotations_now("MERGE BRANCH", "the last branch")
        self.refresh_zoom_viewport()

    def _set_branch_main(self, idx):
        """First click: crack idx (or the main crack of the group it already belongs to) is the main crack."""
        crack = self.cfg.saved_cracks[idx]
        gid = crack.get('crack_group')
        main = self._group_main_index(gid)
        if main is None:
            gid = self._new_crack_group_id()
            crack['crack_group'], crack['branch_main'] = gid, True
            crack.pop('branch', None)
            main = idx
        self.cfg.crack_branch_merge_state["main_group"] = gid
        extra = f" (crack {idx + 1} is already one of its branches)" if main != idx else ""
        print(f" [MERGE BRANCH] Main crack: {main + 1}{extra}. Now click the cracks to add as branches.")

    def _merge_crack_as_branch(self, main, idx):
        """Adds crack idx -- with its own branches, if it is a main crack -- to the group of crack main,
        bridging the gap when it stops short of the group. False when nothing changed."""
        cracks = self.cfg.saved_cracks
        gid = cracks[main]['crack_group']
        if cracks[idx].get('crack_group') == gid:
            print(f" [MERGE BRANCH] Crack {idx + 1} is already part of main crack {main + 1}.")
            return False
        group_idx = self._group_members(gid)
        bridge = self._branch_bridge_plan(group_idx, idx)
        if bridge is None:
            print(f" [MERGE BRANCH] Crack {idx + 1} is too far from main crack {main + 1} and its branches "
                  f"(more than {self.cfg.BRANCH_MERGE_MAX_GAP_PX} px): click a crack that reaches it.")
            return False
        before = self._snapshot_crack_edit_state()
        moved = self._group_members(cracks[idx].get('crack_group')) or [idx]
        for i in moved:
            cracks[i]['crack_group'], cracks[i]['branch'] = gid, True
            cracks[i].pop('branch_main', None)
        gap = 0.0
        if bridge != 'touch':
            target_idx, at_start, q = bridge
            gap = self._extend_crack_to(target_idx, at_start, q)
        self._commit_crack_edit(before)
        joined = f", {gap:.0f} px gap bridged" if gap else ""
        others = f" (with its {len(moved) - 1} branch(es))" if len(moved) > 1 else ""
        print(f" [MERGE BRANCH] Crack {idx + 1}{others} is now a branch of main crack {main + 1}{joined}.")
        return True

    def _branch_bridge_plan(self, group_idx, idx):
        """How crack idx meets the group: 'touch', or (crack to extend, at its start?, point to reach), or None
        when it is farther than BRANCH_MERGE_MAX_GAP_PX. The branch's own ends are tried first; otherwise a group
        crack's end that stops short of the branch line is extended."""
        cracks = self.cfg.saved_cracks
        branch = np.asarray(cracks[idx]['path'], dtype=np.float64)
        group = [(i, np.asarray(cracks[i]['path'], dtype=np.float64)) for i in group_idx if cracks[i].get('path')]
        if branch.size == 0 or not group:
            return None
        best = None   # (distance, crack to extend, at_start, point to reach)
        for at_start, end in ((True, branch[0]), (False, branch[-1])):
            for _, pts in group:
                d = np.hypot(pts[:, 0] - end[0], pts[:, 1] - end[1])
                k = int(np.argmin(d))
                if best is None or d[k] < best[0]:
                    best = (float(d[k]), idx, at_start, (int(pts[k, 0]), int(pts[k, 1])))
        for i, pts in group:
            for at_start, end in ((True, pts[0]), (False, pts[-1])):
                d = np.hypot(branch[:, 0] - end[0], branch[:, 1] - end[1])
                k = int(np.argmin(d))
                if d[k] < best[0] - 1e-9:
                    best = (float(d[k]), i, at_start, (int(branch[k, 0]), int(branch[k, 1])))
        if best[0] <= BRANCH_TOUCH_PX:
            return 'touch'
        if best[0] > self.cfg.BRANCH_MERGE_MAX_GAP_PX:
            return None
        return best[1], best[2], best[3]

    def _extend_crack_to(self, idx, at_start, q):
        """Lengthens crack idx from its start (or end) to point q, along the edge or straight (LINK bridge);
        the bridge is recorded in 'links'. Returns the gap length."""
        crack = self.cfg.saved_cracks[idx]
        path = [(int(p[0]), int(p[1])) for p in crack['path']]
        p = path[0] if at_start else path[-1]
        inner, method = self._link_bridge(p, q)
        inner = [(int(a[0]), int(a[1])) for a in inner if (int(a[0]), int(a[1])) not in (p, q)]
        segs = crack.get('width_segments')
        if at_start:
            new_part = [q] + inner[::-1]
            new_path = new_part + path
            segs, _ = self._remap_width_segments(segs, 0, len(path) - 1, len(new_part))
        else:
            new_path = path + inner + [q]
            segs, _ = self._remap_width_segments(segs, 0, len(path) - 1)
        new_crack = self._crack_with_path(crack, new_path, segs)
        links = list(crack.get('links') or []) + [{"from": [p[0], p[1]], "to": [q[0], q[1]], "method": method}]
        new_crack['links'] = self._links_on_path(links, new_path)
        self.cfg.saved_cracks[idx] = new_crack
        return float(np.hypot(q[0] - p[0], q[1] - p[1]))
