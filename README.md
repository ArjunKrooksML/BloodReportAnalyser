 Blood Report Analyser is a multi-node agentic pipeline built with LangGraph and FastAPI that lets you upload any blood test  report (PDF or image) and chat with it.

  On upload, it automatically extracts text from the PDF, detects what test panels are present, fans out parallel extractors
  for each panel, runs a self-critique loop to validate the results, and flags anomalies — all before you ask a single
  question. The chat layer is a ReAct agent that decides which tools to call (lookup a marker, check its range, list
  anomalies) based on your question, rather than following a fixed path.
