import streamlit as st
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
from openai import OpenAI
from ddgs import DDGS
from streamlit_mic_recorder import mic_recorder
import io

# ---------------------------------------------------------
# 1. SETUP & VIDEO DATABASE 
# ---------------------------------------------------------
st.set_page_config(page_title="Sigma Advance AI", page_icon="🎙️")
st.title("🎙️ Sigma Voice & Text Assistant")

# 
ALL_VIDEO_DETAILS = {
    1: "https://youtu.be/tVzUXW6siu0?si=gmFVXmyb9KEbJ_69",
    2: "https://youtu.be/kJEsTjH5mVg?si=4eBr_lrqX8dH2QUk", 
    3: "https://youtu.be/BGeDBfCIqas?si=_D2XCpV-4cPMe34-", 
    4: "https://youtu.be/nXba2-mgn1k?si=cWDrvnT4hCRN4WXO",
    5:"https://youtu.be/1BsVhumGlNc?si=O-ySMQcUd8-wXtGq"
}

# API KEY (Cloud/Local)
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    api_key = "sk-proj-TUMHARI_ASLI_KEY_YAHAN_DAAL" 

client = OpenAI(api_key="A")

# ---------------------------------------------------------
# 2. CORE BRAIN FUNCTIONS
# ---------------------------------------------------------

def speech_to_text(audio_bytes):
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.mp3"
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcript.text
    except: return None

def text_to_speech(text):
    try:
        response = client.audio.speech.create(model="tts-1", voice="alloy", input=text[:4000])
        return response.content
    except: return None

def get_web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=1))
            if results: return results[0]['body']
        return ""
    except: return ""

@st.cache_data
def load_embeddings():
    try: return joblib.load('embeddings.joblib')
    except: return None

def create_embedding(text_list):
    try:
        res = client.embeddings.create(input=text_list, model="text-embedding-3-small")
        return [data.embedding for data in res.data]
    except: return None

def get_rag_prompt(query):
    df = load_embeddings()
    chunks_info = ""
    is_course_related = False
    
    # Live search har baar call hogi latest info ke liye
    web_data = get_web_search(query)

    if df is not None:
        emb = create_embedding([query])
        if emb:
            sims = cosine_similarity(np.vstack(df['embedding']), [emb[0]]).flatten()
            top_indices = sims.argsort()[::-1][:3]
            
            for idx in top_indices:
                # Strict Threshold: 0.4 se niche matlab course ka sawal nahi hai
                if sims[idx] < 0.4: continue
                
                is_course_related = True
                row = df.iloc[idx]
                v_num = int(row['number'])
                base_link = ALL_VIDEO_DETAILS.get(v_num, "")
                if base_link:
                    chunks_info += f"- Video {v_num}: '{row['title']}' | Link: {base_link}&t={int(row['start'])}s\n"

    # AI Instructions (Hallucination Control)
    if is_course_related and chunks_info:
        prompt = f"""
        User Question: "{query}"
        Course Context: {chunks_info}
        Web Data: {web_data}
        Instruction: Answer using course context. Provide video links at the end under 'Video References'.
        """
    else:
        # Strictly for General/Latest News (IPL, Weather, etc.)
        prompt = f"""
        User Question: "{query}"
        Today's Date: April 25, 2026.
        Latest Web Information: {web_data}
        
        Instruction: Provide the most accurate and RECENT answer based ONLY on the Web Information. 
        If it's about a future event in 2026, use the search data. 
        DO NOT mention the Sigma course. DO NOT mention videos. DO NOT give any course links.
        """
    return prompt
# ---------------------------------------------------------
# 3. CHAT INTERFACE
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Voice & Clean-up
with st.sidebar:
    st.header("🎤 Voice Command")
    audio_input = mic_recorder(start_prompt="Bolo (Click to Speak)", stop_prompt="Ruko (Stop)", key='voice_recorder')
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# INPUT LOGIC
voice_text = None
if audio_input:
    with st.spinner("Sun raha hoon..."):
        voice_text = speech_to_text(audio_input['bytes'])

text_query = st.chat_input("Ya yahan type karo...")
final_query = voice_text or text_query

if final_query:
    with st.chat_message("user"): st.markdown(final_query)
    st.session_state.messages.append({"role": "user", "content": final_query})

    with st.chat_message("assistant"):
        # Mr. BIRU Special
        if any(word in final_query.lower() for word in ["boss", "biru", "founder", "made you"]):
            res = "Mujhe **Mr. BIRU** 😎 ne banaya hai! Wo mere boss hain."
            st.markdown(res)
        else:
            full_p = get_rag_prompt(final_query)
            stream = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": full_p}], stream=True)
            
            def stream_gen():
                for chunk in stream:
                    if chunk.choices[0].delta.content: yield chunk.choices[0].delta.content
            
            res = st.write_stream(stream_gen())
        
        # AUTO AUDIO OUTPUT
        with st.empty():
            audio_out = text_to_speech(res)
            if audio_out:
                st.audio(audio_out, format="audio/mp3", autoplay=True)

    st.session_state.messages.append({"role": "assistant", "content": res})