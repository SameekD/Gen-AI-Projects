import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# ==========================================
# 1. PAGE SETUP & CLIENT INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="Project: Meeting Transcript Summarizer",
    page_icon="📝",
    layout="wide",
)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)

# ==========================================
# 2. HELPER FUNCTIONS & PROMPT LOGIC
# ==========================================
def clean_vtt(vtt_text: str) -> str:
    """Strips WebVTT header tags, cue numbers, and timestamp lines."""
    lines = vtt_text.splitlines()
    cleaned_lines = []
    timestamp_pattern = re.compile(
        r"(\d{2}:)?\d{2}:\d{2}\.\d{3}\s+-->\s+(\d{2}:)?\d{2}:\d{2}\.\d{3}"
    )

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("WEBVTT") or stripped.isdigit():
            continue
        if timestamp_pattern.search(stripped):
            continue
        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


output_instructions = {
    "Bulleted Actions": (
        "Extract only concrete, actionable follow-ups from the transcript. "
        "Format each item as a bullet point using this strict structure: "
        "- [Action Item] | Owner: [Name or 'Unassigned'] | Deadline: [Date/Time or 'Not specified']. "
        "Do not include conversational filler, general chatter, or items not explicitly agreed upon."
    ),
    "Executive Summary": (
        "Provide a high-level strategic overview written for leadership. "
        "Synthesize the core purpose of the meeting, major business decisions agreed upon, "
        "and critical risks or blockers discussed. Keep it within 2 to 3 concise, formal paragraphs. "
        "Omit minor operational details."
    ),
    "Detailed Notes": (
        "Generate comprehensive, structured meeting minutes organized into logical sections: "
        "1. Meeting Objectives, 2. Key Discussion Topics (with major viewpoints noted), "
        "3. Decisions Made, and 4. Next Steps. Capture relevant context, metrics, and "
        "specific nuances discussed by the participants."
    ),
}


def create_system_instruction() -> str:
    return (
        "You are an executive assistant specializing in summarizing meeting transcripts accurately. "
        "Base your output strictly on the provided transcript. "
        "Do not extrapolate, assume, or hallucinate facts not directly stated. "
        "Do not include conversational filler, pleasantries, or preamble in your final output."
    )


def create_user_prompt(transcript: str, output_style: str) -> str:
    style_rule = output_instructions.get(
        output_style, output_instructions["Bulleted Actions"]
    )
    return f"""### Formatting Requirements ({output_style}):
{style_rule}

### Meeting Transcript:
<transcript>
{transcript.strip()}
</transcript>
"""


def generate_response(
    system_instruction: str,
    user_prompt: str,
    model_name: str = "openai/gpt-oss-120b",
) -> tuple[str, dict]:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    content = response.choices[0].message.content or "No response generated."
    usage_stats = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return content.strip(), usage_stats

# ==========================================
# 3. UI LAYOUT & SIDEBAR
# ==========================================
st.title("📝 Meeting Transcript Summarizer")
st.caption(
    "Paste notes or attach a `.txt` / `.vtt` file directly using the clip icon in the input box below."
)

with st.sidebar:
    st.header("⚙️ Configuration")
    output_style = st.selectbox(
        label="Select Output Style",
        options=["Bulleted Actions", "Executive Summary", "Detailed Notes"],
        index=0,
        help="Choose the structure and focus of the generated output.",
    )
    selected_model = st.selectbox(
        label="Select Model",
        options=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
        index=0,
        help="gpt-oss-120b provides deeper synthesis; 20b provides ultra-fast generation.",
    )
# ==========================================
# 4. SESSION STATE INITIALIZATION
# ==========================================
if "summary_result" not in st.session_state:
    st.session_state["summary_result"] = None
if "summary_metrics" not in st.session_state:
    st.session_state["summary_metrics"] = None
if "style_used" not in st.session_state:
    st.session_state["style_used"] = None

# ==========================================
# 5. INTEGRATED CHAT INPUT WITH FILE CLIP
# ==========================================
# Note: accept_file=True requires streamlit >= 1.40.0
prompt_data = st.chat_input(
    placeholder="Paste transcript or click 📎 to upload (.txt, .vtt)...",
    accept_file=True,
    file_type=["txt", "vtt"],
)

if prompt_data:
    transcript = ""

    # Priority 1: Check if a file was attached via the clip icon
    if prompt_data.files:
        attached_file = prompt_data.files[0]
        raw_text = attached_file.read().decode("utf-8", errors="ignore")
        if attached_file.name.endswith(".vtt"):
            transcript = clean_vtt(raw_text)
        else:
            transcript = raw_text

    # Priority 2: Check if text was pasted directly into the input bar
    elif prompt_data.text:
        transcript = prompt_data.text

    if not transcript.strip():
        st.warning("Please provide transcript text or upload a valid file.")
    else:
        system_instruction = create_system_instruction()
        user_prompt = create_user_prompt(transcript, output_style)

        with st.spinner("Analyzing transcript and generating summary..."):
            try:
                summary, metrics = generate_response(
                    system_instruction, user_prompt, model_name=selected_model
                )

                # Word counts
                metrics["input_words"] = len(transcript.strip().split())
                metrics["output_words"] = len(summary.split())

                # Save results to session state
                st.session_state["summary_result"] = summary
                st.session_state["summary_metrics"] = metrics
                st.session_state["style_used"] = output_style

            except Exception as e:
                st.error(f"Error calling model: {e}")

# ==========================================
# 6. RESULTS & METRICS DISPLAY
# ==========================================
if st.session_state["summary_result"]:
    summary = st.session_state["summary_result"]
    metrics = st.session_state["summary_metrics"]
    style = st.session_state["style_used"]

    st.write("---")
    st.subheader(f"Generated Summary ({style})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Input Words", f"{metrics['input_words']:,}")
    with col2:
        st.metric("Summary Words", f"{metrics['output_words']:,}")
    with col3:
        st.metric("Prompt Tokens", f"{metrics['prompt_tokens']:,}")
    with col4:
        st.metric("Completion Tokens", f"{metrics['completion_tokens']:,}")

    st.markdown(summary)

    # st.download_button(
    #     label="📥 Download Summary (.txt)",
    #     data=summary,
    #     file_name=f"meeting_summary_{style.lower().replace(' ', '_')}.txt",
    #     mime="text/plain",
    # )
    col_dl1, col_dl2 = st.columns([0.2, 1.2])

    with col_dl1:
        st.download_button(
            label="📥 Download (.txt)",
            data=summary,
            file_name=f"meeting_summary_{style.lower().replace(' ', '_')}.txt",
            mime="text/plain",
        )

    with col_dl2:
        st.download_button(
            label="📥 Download (.md)",
            data=summary,
            file_name=f"meeting_summary_{style.lower().replace(' ', '_')}.md",
            mime="text/markdown",
        )