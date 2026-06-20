import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api")

st.set_page_config(page_title="Blood Report Analyser", layout="wide", page_icon="")

# ── session state defaults ───────────────────────────────────────────────────
for key, default in {
    "session_id": None,
    "patient_id": "",
    "detected_panels": [],
    "anomaly_tags": [],
    "extracted_markers": [],
    "pattern_matches": [],
    "trend_results": [],
    "doctor_briefing": None,
    "messages": [],
    "needs_clarification": False,
    "clarification_reason": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── helpers ──────────────────────────────────────────────────────────────────
ANOMALY_ICON = {"HIGH": "🟠", "LOW": "🔵", "CRITICAL_HIGH": "🔴", "CRITICAL_LOW": "🟣"}
TRAJECTORY_ICON = {"IMPROVING": "↗ Improving", "STABLE": "→ Stable", "WORSENING": "↘ Worsening", "CRITICAL_TREND": "↘ Critical Trend"}
URGENCY_COLOR = {"CRITICAL": "red", "HIGH": "orange", "MODERATE": "blue", "LOW": "green"}
CONFIDENCE_ICON = {"HIGH": "●●●", "MODERATE": "●●○"}


def reset_session():
    for key in ["session_id", "patient_id", "detected_panels", "anomaly_tags",
                "extracted_markers", "pattern_matches", "trend_results",
                "doctor_briefing", "messages", "needs_clarification", "clarification_reason"]:
        default = [] if isinstance(st.session_state.get(key), list) else None
        st.session_state[key] = default
    st.session_state["patient_id"] = ""


def store_upload_response(data: dict):
    st.session_state.session_id       = data["session_id"]
    st.session_state.detected_panels  = data.get("detected_panels", [])
    st.session_state.anomaly_tags     = data.get("anomaly_tags", [])
    st.session_state.extracted_markers = data.get("extracted_markers", [])
    st.session_state.pattern_matches  = data.get("pattern_matches", [])
    st.session_state.trend_results    = data.get("trend_results", [])
    st.session_state.doctor_briefing  = data.get("doctor_briefing")


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Blood Report Analyser")
    st.caption("Upload any lab report to analyse it.")
    st.divider()

    patient_id_input = st.text_input(
        "Patient ID (optional)",
        value=st.session_state.patient_id,
        placeholder="e.g. patient_001",
        help="Enter the same ID across multiple uploads to enable longitudinal trend tracking.",
        disabled=bool(st.session_state.session_id),
    )

    uploaded_file = st.file_uploader(
        "Upload Report",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        disabled=bool(st.session_state.session_id),
    )

    if uploaded_file and not st.session_state.session_id:
        st.session_state.patient_id = patient_id_input
        with st.spinner("Processing report — this may take 15–30 seconds..."):
            params = {"patient_id": patient_id_input} if patient_id_input else {}
            response = requests.post(
                f"{API_BASE}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                params=params,
                timeout=120,
            )

        if response.status_code == 200:
            data = response.json()
            if data.get("needs_clarification"):
                st.session_state.needs_clarification = True
                st.session_state.clarification_reason = data.get("reason")
                st.session_state.session_id = data.get("session_id")
            else:
                store_upload_response(data)
                st.success("Report processed successfully.")
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
                    params={"clarification": clarification},
                    timeout=120,
                )
            if response.status_code == 200:
                st.session_state.needs_clarification = False
                store_upload_response(response.json())
                st.rerun()
            else:
                st.error("Clarification failed. Please try again.")

    if st.session_state.session_id and not st.session_state.needs_clarification:
        st.divider()

        if st.session_state.patient_id:
            st.caption(f"Patient: `{st.session_state.patient_id}`")

        st.subheader("Panels Detected")
        for panel in st.session_state.detected_panels:
            st.markdown(f"- `{panel}`")

        st.divider()
        st.subheader("Anomalies")
        if st.session_state.anomaly_tags:
            for tag in st.session_state.anomaly_tags:
                icon = ANOMALY_ICON.get(tag["status"], "⚪")
                dev = f"  ({tag['deviation_percent']:+.1f}%)" if tag.get("deviation_percent") else ""
                st.markdown(f"{icon} **{tag['marker_name']}** {tag['value']} {tag['unit']}{dev}")
        else:
            st.success("All markers within range.")

        st.divider()
        if st.button("Clear Session", use_container_width=True):
            reset_session()
            st.rerun()


# ── main area ─────────────────────────────────────────────────────────────────
if not st.session_state.session_id:
    st.markdown("## Upload a blood report on the left to begin.")
    st.stop()

if st.session_state.needs_clarification:
    st.markdown("## Waiting for clarification on the uploaded report.")
    st.stop()

# ── SECTION 1: Doctor's Briefing ─────────────────────────────────────────────
briefing = st.session_state.doctor_briefing
if briefing:
    urgency = briefing.get("urgency_level", "LOW")
    color = URGENCY_COLOR.get(urgency, "blue")

    st.markdown(f"## Doctor's Briefing  &nbsp; :{color}[{urgency}]")
    st.markdown(f"_{briefing.get('summary', '')}_")

    col1, col2 = st.columns(2)

    with col1:
        if briefing.get("critical_now"):
            st.markdown("**Act Now**")
            for item in briefing["critical_now"]:
                st.error(item, icon="🚨")

        if briefing.get("watch_list"):
            st.markdown("**Watch List**")
            for item in briefing["watch_list"]:
                st.warning(item, icon="👁")

    with col2:
        if briefing.get("positive_findings"):
            st.markdown("**What Looks Good**")
            for item in briefing["positive_findings"]:
                st.success(item, icon="✅")

        if briefing.get("lifestyle_notes"):
            st.markdown("**Lifestyle Actions**")
            for item in briefing["lifestyle_notes"]:
                st.info(item, icon="💡")

    if briefing.get("questions_for_doctor"):
        st.markdown("**Questions to Ask Your Doctor**")
        for i, q in enumerate(briefing["questions_for_doctor"], 1):
            st.markdown(f"{i}. {q}")

    st.divider()

# ── SECTION 2: Pattern Matches ────────────────────────────────────────────────
if st.session_state.pattern_matches:
    st.markdown("## Clinical Patterns Detected")
    cols = st.columns(min(len(st.session_state.pattern_matches), 3))
    for i, pattern in enumerate(st.session_state.pattern_matches):
        with cols[i % 3]:
            confidence = pattern.get("confidence", "")
            badge = CONFIDENCE_ICON.get(confidence, "●○○")
            st.markdown(f"### {pattern['name']}")
            st.caption(f"Confidence: {badge} {confidence}")
            st.markdown(pattern.get("implication", ""))
            with st.expander("Evidence"):
                for e in pattern.get("evidence", []):
                    st.markdown(f"- {e}")
            if pattern.get("differentials"):
                with st.expander("Rule out"):
                    for d in pattern["differentials"]:
                        st.markdown(f"- {d}")
    st.divider()

# ── SECTION 3: Trend Analysis ─────────────────────────────────────────────────
if st.session_state.trend_results:
    st.markdown("## Longitudinal Trends")
    for trend in st.session_state.trend_results:
        trajectory = trend.get("trajectory", "STABLE")
        traj_label = TRAJECTORY_ICON.get(trajectory, "→ Stable")
        change = trend.get("change_percent", 0)
        change_str = f"{change:+.1f}%"

        with st.expander(f"{trend['marker_name']} — {traj_label}  ({change_str})"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Previous", f"{trend['previous_value']} {trend['unit']}")
            c2.metric("Current", f"{trend['current_value']} {trend['unit']}", delta=change_str)
            if trend.get("monthly_rate") is not None:
                c3.metric("Monthly Rate", f"{trend['monthly_rate']:+.3f} {trend['unit']}/mo")

            st.markdown(f"_{trend.get('interpretation', '')}_")

            if trend.get("months_to_threshold"):
                st.warning(f"Projection: {trend['months_to_threshold']}", icon="⏱")

    st.divider()

# ── SECTION 4: Chat ───────────────────────────────────────────────────────────
st.markdown("## Ask About Your Report")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask anything about your results...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = requests.post(
                f"{API_BASE}/chat",
                json={"session_id": st.session_state.session_id, "question": question},
                timeout=60,
            )
        if resp.status_code == 200:
            answer = resp.json()["response"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error("Something went wrong. Please try again.")
