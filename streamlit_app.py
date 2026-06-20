import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api")

st.set_page_config(page_title="Blood Report Analyser", layout="wide")

# ── session state defaults ──────────────────────────────────────────────────
for key, default in {
    "session_id": None,
    "detected_panels": [],
    "anomaly_tags": [],
    "extracted_markers": [],
    "messages": [],
    "needs_clarification": False,
    "clarification_reason": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── helpers ─────────────────────────────────────────────────────────────────
def status_color(status: str) -> str:
    return {"HIGH": "🟠", "LOW": "🔵", "CRITICAL_HIGH": "🔴", "CRITICAL_LOW": "🟣"}.get(status, "⚪")


def reset_session():
    for key in ["session_id", "detected_panels", "anomaly_tags", "extracted_markers", "messages", "needs_clarification", "clarification_reason"]:
        st.session_state[key] = [] if isinstance(st.session_state[key], list) else None


# ── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Blood Report Analyser")
    st.caption("Upload a lab report to get started.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Report",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        help="Supports PDFs and images. Multi-page PDFs are handled automatically."
    )

    if uploaded_file and not st.session_state.session_id:
        with st.spinner("Processing report..."):
            response = requests.post(
                f"{API_BASE}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            )

        if response.status_code == 200:
            data = response.json()
            if data.get("needs_clarification"):
                st.session_state.needs_clarification = True
                st.session_state.clarification_reason = data.get("reason")
                st.session_state.session_id = data.get("session_id")
            else:
                st.session_state.session_id = data["session_id"]
                st.session_state.detected_panels = data["detected_panels"]
                st.session_state.anomaly_tags = data["anomaly_tags"]
                st.session_state.extracted_markers = data["extracted_markers"]
                st.success("Report processed.")
        else:
            st.error(f"Upload failed: {response.text}")

    if st.session_state.needs_clarification:
        st.warning("Report unclear")
        st.caption(st.session_state.clarification_reason or "Could not read the report clearly.")
        clarification = st.text_area("Describe the report contents:")
        if st.button("Submit Clarification") and clarification:
            with st.spinner("Re-processing..."):
                response = requests.post(
                    f"{API_BASE}/clarify/{st.session_state.session_id}",
                    params={"clarification": clarification}
                )
            if response.status_code == 200:
                data = response.json()
                st.session_state.needs_clarification = False
                st.session_state.detected_panels = data["detected_panels"]
                st.session_state.anomaly_tags = data["anomaly_tags"]
                st.session_state.extracted_markers = data["extracted_markers"]
                st.rerun()
            else:
                st.error("Clarification failed. Please try again.")

    if st.session_state.session_id and not st.session_state.needs_clarification:
        st.divider()
        st.subheader("Detected Panels")
        if st.session_state.detected_panels:
            for panel in st.session_state.detected_panels:
                st.markdown(f"- `{panel}`")
        else:
            st.caption("None detected.")

        st.divider()
        st.subheader("Anomalies")
        if st.session_state.anomaly_tags:
            for tag in st.session_state.anomaly_tags:
                icon = status_color(tag["status"])
                deviation = f"({tag['deviation_percent']:+.1f}%)" if tag.get("deviation_percent") else ""
                st.markdown(f"{icon} **{tag['marker_name']}** — {tag['value']} {tag['unit']} {deviation}")
        else:
            st.success("All markers within range.")

        st.divider()
        if st.button("Clear Session"):
            reset_session()
            st.rerun()


# ── main chat area ───────────────────────────────────────────────────────────
if not st.session_state.session_id:
    st.markdown("## Upload a blood report on the left to begin.")
    st.stop()

if st.session_state.needs_clarification:
    st.markdown("## Waiting for clarification on the uploaded report.")
    st.stop()

st.markdown("## Ask about your report")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask anything about your report...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_BASE}/chat",
                json={"session_id": st.session_state.session_id, "question": question}
            )

        if response.status_code == 200:
            data = response.json()
            answer = data["response"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            error = "Something went wrong. Please try again."
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error})
