import streamlit as st

from pipeline import PDF_PATH, ingest_pdf, run_query


st.set_page_config(page_title="Advance RAG Chat", page_icon="📚")
st.title("Advance RAG Chat")
st.caption("Ask questions about the PDF document.")

if "active_document" not in st.session_state:
    st.session_state.active_document = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Document")
    with st.form("document_upload"):
        uploaded_pdf = st.file_uploader("Choose a PDF", type=["pdf"])
        ingest_clicked = st.form_submit_button(
            "Upload and ingest",
            type="primary",
        )

    if ingest_clicked and uploaded_pdf is not None:
        PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        PDF_PATH.write_bytes(uploaded_pdf.getvalue())
        st.session_state.active_document = None
        st.session_state.messages = []

        try:
            with st.spinner("Processing the PDF and building its search index..."):
                ingest_pdf()
            st.session_state.active_document = uploaded_pdf.name
            st.success(f"Ready to query: {uploaded_pdf.name}")
        except Exception as error:
            st.error(f"Could not ingest the PDF: {error}")
    elif ingest_clicked:
        st.warning("Choose a PDF before starting ingestion.")

    if st.session_state.active_document:
        st.caption(f"Indexed: {st.session_state.active_document}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.active_document:
    question = st.chat_input("Ask a question about the document...")
else:
    st.info("Upload and ingest a PDF in the sidebar to start asking questions.")
    question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Running the RAG pipeline..."):
        answer = run_query(question)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
