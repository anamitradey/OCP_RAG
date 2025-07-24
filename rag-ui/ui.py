import os
import gradio as gr
import requests

API = os.getenv("BACKEND_URL", "http://localhost:8080")   # FastAPI service


# ---------- backend wrappers -------------------------------------------------
def ingest(document_id, source, text, use_hash_ids):
    r = requests.post(f"{API}/ingest", json={
        "document_id": document_id,
        "text": text,
        "source": source or None,
        "use_content_hash_ids": use_hash_ids
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def chat(prompt, k):
    r = requests.post(f"{API}/chat", json={
        "query": prompt,
        "k": k
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def reset_collection():
    # Thin wrapper around DELETE /collection/reset
    r = requests.delete(f"{API}/collection/reset", timeout=30)
    r.raise_for_status()
    return r.json()


# ---------- Gradio UI --------------------------------------------------------
with gr.Blocks(title="RAG Docs Playground") as demo:
    gr.Markdown("### 📚 Interact with **rag_docs**")

    # -- Ingest tab -----------------------------------------------------------
    with gr.Tab("Ingest"):
        doc_id   = gr.Textbox(label="Document ID", placeholder="e.g. wiki_42")
        source   = gr.Textbox(label="Source (optional)")
        text     = gr.Textbox(lines=10, label="Raw text to ingest")
        hash_ids = gr.Checkbox(label="Use content‑hash IDs", value=False)
        ingest_btn = gr.Button("Ingest")
        ingest_out = gr.JSON()

        ingest_btn.click(
            fn=ingest,
            inputs=[doc_id, source, text, hash_ids],
            outputs=ingest_out,
        )

    # -- Chat tab -------------------------------------------------------------
    with gr.Tab("Chat"):
        prompt   = gr.Textbox(label="Query")
        topk     = gr.Slider(1, 20, value=5, step=1, label="Top‑K")
        chat_btn = gr.Button("Ask")
        chat_out = gr.JSON()

        chat_btn.click(
            fn=chat,
            inputs=[prompt, topk],
            outputs=chat_out,
        )

    # -- Drop‑all tab ---------------------------------------------------------
    with gr.Tab("⚠️ Drop All Docs"):
        gr.Markdown(
            "**Danger zone — this deletes every document in the system.**\n\n"
            "Use when you want a clean slate."
        )
        reset_btn = gr.Button("Delete EVERYTHING", variant="stop")
        reset_out = gr.JSON()

        reset_btn.click(
            fn=lambda: reset_collection(),
            outputs=reset_out,
        )

demo.launch(server_name="0.0.0.0", server_port=7860)
