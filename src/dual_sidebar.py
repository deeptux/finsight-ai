"""Mutual-exclusive left (Streamlit) and right (PDF) sidebars."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def sync_right_sidebar_query() -> None:
    """Apply ?right=0 from JS when user expands the left Documents sidebar."""
    val = st.query_params.get("right")
    if val == "0":
        st.session_state.right_sidebar_open = False
        try:
            del st.query_params["right"]
        except Exception:
            pass


def inject_dual_sidebar_sync(*, right_open: bool, collapse_left: bool) -> None:
    """Body class, collapse-left on right open, left-expand closes right."""
    right_js = "true" if right_open else "false"
    collapse_js = "true" if collapse_left else "false"

    components.html(
        f"""
        <script>
        (function () {{
          const doc = window.parent.document;
          const rightOpen = {right_js};
          const collapseLeft = {collapse_js};

          doc.body.classList.toggle("finsight-right-sidebar-open", rightOpen);

          function collapseLeftSidebar() {{
            const btn =
              doc.querySelector('[data-testid="stSidebarCollapseButton"]') ||
              doc.querySelector('section[data-testid="stSidebar"] button[kind="header"]');
            if (btn) btn.click();
          }}

          function clickCloseRightButton() {{
            const marker = doc.getElementById("finsight-right-sidebar-marker");
            if (!marker) return false;
            const col =
              marker.closest('[data-testid="stColumn"]') ||
              marker.closest('[data-testid="column"]');
            if (!col) return false;
            const buttons = col.querySelectorAll("button");
            for (let i = 0; i < buttons.length; i++) {{
              const label = (buttons[i].textContent || "").trim();
              if (label === "\\u203A" || label === ">") {{
                buttons[i].click();
                return true;
              }}
            }}
            return false;
          }}

          if (collapseLeft) {{
            collapseLeftSidebar();
            setTimeout(collapseLeftSidebar, 120);
          }}

          if (rightOpen) {{
            const leftOpen = doc.querySelector(
              'section[data-testid="stSidebar"][aria-expanded="true"]'
            );
            if (leftOpen) collapseLeftSidebar();
          }}

          if (!window.parent.__finsightDualSidebarBound) {{
            window.parent.__finsightDualSidebarBound = true;
            doc.addEventListener("click", function (ev) {{
              const t = ev.target;
              if (!t) return;
              if (t.closest('[data-testid="stSidebarCollapsedControl"]')) {{
                if (!clickCloseRightButton()) {{
                  try {{
                    const url = new URL(window.parent.location.href);
                    url.searchParams.set("right", "0");
                    window.parent.location.href = url.toString();
                  }} catch (e) {{}}
                }}
              }}
            }}, true);

            const left = doc.querySelector('section[data-testid="stSidebar"]');
            if (left) {{
              let leftNavPending = false;
              const mo = new MutationObserver(function () {{
                if (leftNavPending) return;
                if (
                  left.getAttribute("aria-expanded") === "true" &&
                  doc.body.classList.contains("finsight-right-sidebar-open")
                ) {{
                  leftNavPending = true;
                  if (!clickCloseRightButton()) {{
                    try {{
                      const url = new URL(window.parent.location.href);
                      url.searchParams.set("right", "0");
                      window.parent.location.href = url.toString();
                    }} catch (e) {{
                      leftNavPending = false;
                    }}
                  }}
                }}
              }});
              mo.observe(left, {{ attributes: true, attributeFilter: ["aria-expanded"] }});
            }}
          }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )
