#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import gradio as gr

# Ensure model is downloaded and cached locally before app starts
if not os.path.exists("./model_cache/all-MiniLM-L6-v2"):
    print("Downloading and caching model for offline pipeline compatibility...")
    subprocess.run(["python", "download_model.py"], check=True)

def run_ranking(file_obj):
    if file_obj is None:
        return "Please upload a candidates JSONL file.", None, None
    
    input_path = file_obj.name
    output_path = "team_cache_Q.csv"
    
    # Run the production ranking pipeline
    try:
        cmd = ["python", "rank.py", "--candidates", input_path, "--out", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logs = result.stdout + "\n" + result.stderr
    except subprocess.CalledProcessError as e:
        return f"Error executing ranker:\n{e.stderr}", None, None
    
    # Load output to display
    if not os.path.exists(output_path):
        return f"Ranker ran but output file was not found.\nLogs:\n{logs}", None, None
        
    df = pd.read_csv(output_path)
    
    # Preview top 15 candidates
    preview_df = df.head(15)
    
    return logs, preview_df, output_path

# Custom CSS for modern visual design (slate, premium, rounded cards)
custom_css = """
body {
    background-color: #0B0F19;
    color: #F3F4F6;
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 40px auto !important;
    border-radius: 16px;
    background: radial-gradient(circle at top, #1E293B, #0F172A) !important;
    border: 1px solid #334155 !important;
    box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.5), 0 8px 10px -6px rgb(0 0 0 / 0.5) !important;
    padding: 30px !important;
}
.title-container {
    text-align: center;
    margin-bottom: 25px;
}
.title-container h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(to right, #60A5FA, #3B82F6, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.title-container p {
    color: #94A3B8;
    font-size: 1.1rem;
}
.card {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}
.btn-primary {
    background: linear-gradient(to right, #2563EB, #4F46E5) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgb(37 99 235 / 0.3) !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=custom_css) as demo:
    gr.HTML(
        """
        <div class="title-container">
            <h1>RedRob AI Talent Search Ranker</h1>
            <p>Production candidate sorting pipeline for the Senior AI Engineer founding role</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### 🛠️ Submission & Configuration Info
                * **Team ID / Name**: `cache_Q`
                * **Primary Contact**: Syed Mohsin Naina S
                * **Model Used**: `all-MiniLM-L6-v2` (SentenceTransformer, 100% Offline execution)
                * **Target Persona**: Senior AI Engineer (Founding team, Series A Talent Intelligence Platform)
                
                ### 📂 Upload Dataset
                Upload the JSONL candidate dataset (`candidates.jsonl` or a smaller subset file). The ranker evaluates candidate skills, experience curves, product company history, and 16 behavioural signals.
                """
            )
            
            input_file = gr.File(label="Upload candidates.jsonl", file_types=[".jsonl", ".jsonl.gz"])
            run_btn = gr.Button("🚀 Run AI Ranker", variant="primary", elem_classes="btn-primary")
            
        with gr.Column(scale=1.5):
            gr.Markdown("### 📊 Status & Output")
            status_logs = gr.Textbox(label="Execution Logs / Status Output", placeholder="Run the ranker to see pipeline logs...", lines=10, max_lines=15)
            
            download_file = gr.File(label="Download Ranked CSV (team_cache_Q.csv)")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🏆 Top 15 Candidates Preview")
            preview_table = gr.DataFrame(
                headers=["candidate_id", "rank", "score", "reasoning"],
                datatype=["str", "number", "number", "str"],
                wrap=True,
                label="Candidate Leaderboard"
            )

    run_btn.click(
        fn=run_ranking,
        inputs=[input_file],
        outputs=[status_logs, preview_table, download_file]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
