import streamlit as st

def apply_custom_css():
    """Applies custom CSS for a modern, dark/light compatible UI."""
    custom_css = """
    <style>
        /* Base styles */
        :root {
            --primary-color: #6366f1;
            --background-color: #f8fafc;
            --card-background: #ffffff;
            --text-color: #1e293b;
            --border-color: #e2e8f0;
        }
        
        @media (prefers-color-scheme: dark) {
            :root {
                --primary-color: #818cf8;
                --background-color: #0f172a;
                --card-background: #1e293b;
                --text-color: #f8fafc;
                --border-color: #334155;
            }
        }

        /* Top padding reduction */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }

        /* Modern Card styling */
        .css-card {
            background-color: var(--card-background);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid var(--border-color);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .css-card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-2px);
        }

        /* File Uploader styling */
        .stFileUploader {
            border: 2px dashed var(--primary-color);
            border-radius: 12px;
            padding: 10px;
            background-color: rgba(99, 102, 241, 0.05);
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton>button:hover {
            transform: scale(1.02);
        }
        
        /* Metric styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            color: var(--primary-color);
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_card(title: str, content: str, icon: str = ""):
    """Renders a beautifully styled card component."""
    html_content = f"""
    <div class="css-card">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 8px;">
            {icon} {title}
        </h3>
        <p style="color: var(--text-color); line-height: 1.6;">
            {content}
        </p>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)
