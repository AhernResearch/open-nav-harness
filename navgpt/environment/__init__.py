"""Navigation Environment service for R2R-CE, RxR-CE and scan development.

Needs only PyYAML, NumPy, numpy-quaternion, Pillow and habitat_sim, all present in
the Habitat environment. The Planner-side dependencies (claude-agent-sdk, mcp)
need Python >= 3.10 and live on the other side of the HTTP seam.
The scan adapter runs in a separate pinned Habitat-GS environment.
"""
