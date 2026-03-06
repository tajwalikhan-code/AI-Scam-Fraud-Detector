# =====================================================  
# AI SCAM & Fraud DETECTOR
# =====================================================  
import os, re, sqlite3, datetime, email  
import gradio as gr  
from transformers import pipeline  
from sentence_transformers import SentenceTransformer, util  
from groq import Groq  
from pypdf import PdfReader  

# Optional multimodal imports  
try:  
    from PIL import Image  
    import pytesseract  
except:  
    Image = None  
    pytesseract = None  

try:  
    import speech_recognition as sr  
except:  
    sr = None  

try:  
    from moviepy.editor import VideoFileClip  
except:  
    VideoFileClip = None  

# =====================================================  
# CONFIG  
# =====================================================  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
DATABASE = "scam_detector.db"  

# =====================================================  
# DATABASE SETUP  
# =====================================================  
conn = sqlite3.connect(DATABASE, check_same_thread=False)  
cursor = conn.cursor()  
cursor.execute("""  
CREATE TABLE IF NOT EXISTS scans (  
    id INTEGER PRIMARY KEY AUTOINCREMENT,  
    content TEXT,  
    risk TEXT,  
    score REAL,  
    date TEXT  
)  
""")  
conn.commit()  

# =====================================================  
# LOAD MODELS  
# =====================================================  
classifier = pipeline(  
    "text-classification",  
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",  
    truncation=True  
)  
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None  

# =====================================================  
# SCAM PATTERNS  
# =====================================================  
known_scams = [  
    "verify your account",  
    "urgent action required",  
    "you won lottery",  
    "claim prize",  
    "send crypto",  
    "account suspended"  
]  
known_embeddings = embedder.encode(known_scams)  
urgency_words = ["urgent", "immediately", "verify", "suspended", "now"]  
sensitive_words = ["password", "otp", "bank", "crypto", "login"]  

# =====================================================  
# ANALYSIS FUNCTIONS  
# =====================================================  
def analyze_rules(text):  
    text_lower = text.lower()  
    urls = re.findall(r'https?://\S+', text)  
    emails = re.findall(r'\S+@\S+', text)  
    urgency = sum(1 for w in urgency_words if w in text_lower)  
    sensitive = sum(1 for w in sensitive_words if w in text_lower)  
    url_score = 0.9 * len(urls)  
    score = urgency*0.2 + sensitive*0.2 + url_score*0.3 + len(emails)*0.3  
    return min(score,1.0), urls, emails, urgency, sensitive  

def similarity_score(text):  
    emb = embedder.encode(text)  
    sim = util.cos_sim(emb, known_embeddings)[0]  
    return float(sim.max())  

def classify(text):  
    if not text.strip():  
        return {"score":0,"risk":"LOW","color":"🟢","urls":[],"emails":[],"urgency":0,"sensitive":0,"similarity":0}  
    hf = classifier(text[:1000])[0]  
    hf_score = hf["score"] if "negative" in hf["label"].lower() else 1 - hf["score"]  
    rule_score, urls, emails, urgency, sensitive = analyze_rules(text)  
    sim_score = similarity_score(text)  
    final_score = hf_score*0.5 + rule_score*0.3 + sim_score*0.2  
    if final_score > 0.8: risk, color = "HIGH", "🔴"  
    elif final_score > 0.5: risk, color = "MEDIUM", "🟠"  
    else: risk, color = "LOW", "🟢"  
    return {"score": round(final_score,3),"risk":risk,"color":color,"urls":urls,"emails":emails,"urgency":urgency,"sensitive":sensitive,"similarity":sim_score}  

def save_scan(text, risk, score):  
    cursor.execute(  
        "INSERT INTO scans (content,risk,score,date) VALUES (?,?,?,?)",  
        (text,risk,score,str(datetime.datetime.now()))  
    )  
    conn.commit()  

def load_history():  
    cursor.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 20")  
    rows = cursor.fetchall()  
    if not rows: return "No scans yet."  
    return "\n".join([f"{r[4]} | {r[2]} | {r[3]}" for r in rows])  

# =====================================================  
# MULTIMODAL ANALYSIS FUNCTIONS  
# =====================================================  
def analyze_pdf(file):  
    if not file: return ""  
    try:  
        reader = PdfReader(file)  
        return "".join(page.extract_text() or "" for page in reader.pages)  
    except: return ""  

def analyze_image(file):  
    if not Image or not pytesseract or not file: return "Image scanning not available."  
    try:  
        img = Image.open(file)  
        return pytesseract.image_to_string(img)  
    except: return "Could not read image."  

def analyze_voice(file):  
    if not sr or not file: return "Voice scanning not available."  
    recognizer = sr.Recognizer()  
    try:  
        with sr.AudioFile(file) as source:  
            audio = recognizer.record(source)  
        return recognizer.recognize_google(audio)  
    except: return "Could not transcribe audio."  

def analyze_video(file):  
    if not VideoFileClip or not file: return "Video scanning not available."  
    try:  
        clip = VideoFileClip(file)  
        audio_file = "temp_audio.wav"  
        clip.audio.write_audiofile(audio_file)  
        return analyze_voice(audio_file)  
    except: return "Could not process video."  

def analyze_email(file):  
    if not file: return ""  
    try:  
        msg = email.message_from_bytes(file.read())  
        content = msg.get_payload(decode=True)  
        return content.decode(errors='ignore') if content else ""  
    except: return ""  

# =====================================================  
# GROQ EXPLANATION  
# =====================================================  
def explain(text, score, risk):  
    if not GROQ_API_KEY: return "GROQ API key not found."  
    try:  
        client = Groq(api_key=GROQ_API_KEY)  
        prompt = f"You are a cybersecurity expert. Explain why this message is {risk} risk with score {score:.2f}/1.00.\n{text}"  
        response = client.chat.completions.create(  
            model="llama-3.1-8b-instant",  
            messages=[{"role":"user","content":prompt}],  
            temperature=0.3,  
            max_tokens=250  
        )  
        return response.choices[0].message.content.strip()  
    except Exception as e: return f"GROQ Error: {str(e)}"  

# =====================================================  
# DETECTION WRAPPERS  
# =====================================================  
def detect(text):  
    if not text.strip(): return "No input provided.","","",0  
    r = classify(text)  
    save_scan(text,r["risk"],r["score"])  
    explanation_text = explain(text,r["score"],r["risk"])  
    technical = f"URLs: {len(r['urls'])}\nEmails: {len(r['emails'])}\nUrgency: {r['urgency']}\nSensitive: {r['sensitive']}\nSimilarity: {round(r['similarity'],2)}"  
    summary = f"{r['color']} {r['risk']} | Score: {r['score']}"  
    return summary, explanation_text, technical, int(r["score"]*100)  

def detect_pdf(f): return detect(analyze_pdf(f))  
def detect_image(f): return detect(analyze_image(f))  
def detect_voice(f): return detect(analyze_voice(f))  
def detect_video(f): return detect(analyze_video(f))  
def detect_email(f): return detect(analyze_email(f))  

# =====================================================  
# URL DETECTOR (NEW FEATURE)  
# =====================================================  
def detect_url(url):  
    if not url.strip(): return "No URL provided.","","",0  
    r = classify(url)  
    save_scan(url, r["risk"], r["score"])  
    explanation_text = explain(url, r["score"], r["risk"])  
    technical = f"Detected URL: {url}\nRisk Level: {r['risk']}\nSimilarity: {round(r['similarity'],2)}"  
    summary = f"{r['color']} {r['risk']} | Score: {r['score']}"  
    return summary, explanation_text, technical, int(r["score"]*100)  

# =====================================================  
# AI CHATBOT  
# =====================================================  
def chatbot_ai(query):  
    if not query.strip(): return "Ask me anything about scams or cybersecurity."  
    try:  
        response = groq_client.chat.completions.create(  
            model="llama-3.1-8b-instant",  
            messages=[{"role":"user","content":query}],  
            temperature=0.3,  
            max_tokens=250  
        )  
        return response.choices[0].message.content.strip()  
    except Exception as e: return f"GROQ Error: {str(e)}"  

# =====================================================  
# GRADIO UI  
# =====================================================  
custom_css = """  
body {background: white; font-family: 'Arial Black', sans-serif;}  
h1 {color:#39ff14; font-weight:900; font-size:48px; text-align:center; text-shadow:0 0 5px #39ff14,0 0 10px #39ff14;}  
.sidebar {background-color:#f0f0f0; color:#000; border-radius:12px; padding:15px;}  
button {background:#00aaff; color:white; border-radius:12px; font-weight:bold; margin:10px 0; width:100%; font-size:20px; padding:15px; box-shadow: 3px 3px 5px #999;}  
button:hover {transform: translateY(-3px); box-shadow:6px 6px 10px #888; cursor:pointer;}  
textarea,input {background-color:white !important; color:black !important; border:2px solid #00aaff !important; border-radius:12px; padding:8px;}  
"""  

with gr.Blocks() as app:  
    gr.Markdown("<h1>AI Scam Detector</h1>")  
    gr.Markdown("<div style='color:#333; font-size:16px; text-align:center; margin-bottom:15px;'>Protect yourself from phishing, scams, and suspicious content</div>")  

    with gr.Row():  
        # Left Sidebar  
        with gr.Column(scale=1, elem_classes="sidebar"):  
            btn_dashboard = gr.Button("🏠 Dashboard")  
            btn_text = gr.Button("📝 Text Detector")  
            btn_pdf = gr.Button("📄 PDF Detector")  
            btn_url = gr.Button("🔗 URL Detector")  
            btn_voice = gr.Button("🎤 Voice")  
            btn_video = gr.Button("🎬 Video")  
            btn_email = gr.Button("📧 Email")  
            btn_image = gr.Button("🖼 Image")  
            btn_chat = gr.Button("🤖 AI Chatbot")  

        # Right panel  
        with gr.Column(scale=3) as right_panel:  
            dashboard_md = gr.Markdown("""  
                ## Dashboard  
                ⚠️ Beware of phishing emails    
                💰 Crypto scams with fake returns    
                🎁 Lottery scams asking upfront payment    
                🛑 Urgent action messages are suspicious  
            """)  

            # Inputs  
            text_input = gr.Textbox(lines=6, placeholder="Paste suspicious text here...", visible=False)  
            pdf_input = gr.File(label="Upload PDF", visible=False)  
            url_input = gr.Textbox(label="Enter URL", placeholder="Paste suspicious URL here...", visible=False)  
            voice_input = gr.Audio(label="Upload Voice", visible=False)  
            video_input = gr.File(label="Upload Video", visible=False)  
            email_input = gr.File(label="Upload Email", visible=False)  
            image_input = gr.File(label="Upload Image", visible=False)  
            chat_input = gr.Textbox(label="Ask AI", placeholder="Type your question...", visible=False)  
            chat_output = gr.Textbox(label="AI Response", visible=False)  
            summary = gr.Textbox(label="Risk Summary", visible=False)  
            explanation = gr.Textbox(lines=6,label="AI Explanation", visible=False)  
            technical = gr.Textbox(lines=5,label="Technical Details", visible=False)  
            risk_bar = gr.Slider(0,100,label="Risk %", visible=False)  
            history_box = gr.Textbox(lines=8,label="Recent Scans", visible=False)  

            # Analyze Buttons  
            btn_analyze_text = gr.Button("Analyze Text", visible=False)  
            btn_analyze_pdf = gr.Button("Analyze PDF", visible=False)  
            btn_analyze_url = gr.Button("Analyze URL", visible=False)  
            btn_analyze_voice = gr.Button("Analyze Voice", visible=False)  
            btn_analyze_video = gr.Button("Analyze Video", visible=False)  
            btn_analyze_email = gr.Button("Analyze Email", visible=False)  
            btn_analyze_image = gr.Button("Analyze Image", visible=False)  
            btn_ask_chat = gr.Button("Ask", visible=False)  

    # Global components list  
    comps = [
        dashboard_md,text_input,pdf_input,url_input,voice_input,video_input,email_input,image_input,
        chat_input,chat_output,summary,explanation,technical,risk_bar,history_box,
        btn_analyze_text,btn_analyze_pdf,btn_analyze_url,
        btn_analyze_voice,btn_analyze_video,btn_analyze_email,btn_analyze_image,btn_ask_chat
    ]  

    # Show/hide function  
    def show_module(module):  
        updates = [gr.update(visible=False) for _ in comps]
        if module=="dashboard": updates[0]=gr.update(visible=True)
        elif module=="text": updates[1]=updates[10]=updates[11]=updates[12]=updates[13]=updates[15]=gr.update(visible=True)
        elif module=="pdf": updates[2]=updates[10]=updates[11]=updates[12]=updates[13]=updates[16]=gr.update(visible=True)
        elif module=="url": updates[3]=updates[10]=updates[11]=updates[12]=updates[13]=updates[17]=gr.update(visible=True)
        elif module=="voice": updates[4]=updates[10]=updates[11]=updates[12]=updates[13]=updates[18]=gr.update(visible=True)
        elif module=="video": updates[5]=updates[10]=updates[11]=updates[12]=updates[13]=updates[19]=gr.update(visible=True)
        elif module=="email": updates[6]=updates[10]=updates[11]=updates[12]=updates[13]=updates[20]=gr.update(visible=True)
        elif module=="image": updates[7]=updates[10]=updates[11]=updates[12]=updates[13]=updates[21]=gr.update(visible=True)
        elif module=="chat": updates[8]=updates[9]=updates[22]=gr.update(visible=True)
        return updates  

    # Button connections  
    for btn, mod in zip([btn_dashboard,btn_text,btn_pdf,btn_url,btn_voice,btn_video,btn_email,btn_image,btn_chat],
                        ["dashboard","text","pdf","url","voice","video","email","image","chat"]):
        btn.click(show_module, inputs=[gr.State(mod)], outputs=comps)

    # Analyze buttons  
    btn_analyze_text.click(detect, inputs=text_input, outputs=[summary,explanation,technical,risk_bar])
    btn_analyze_pdf.click(detect_pdf, inputs=pdf_input, outputs=[summary,explanation,technical,risk_bar])
    btn_analyze_url.click(detect_url, inputs=url_input, outputs=[summary,explanation,technical,risk_bar])
    btn_analyze_voice.click(detect_voice, inputs=voice_input, outputs=[summary,explanation,technical,risk_bar])
    btn_analyze_video.click(detect_video, inputs=video_input, outputs=[summary,explanation,technical,risk_bar])
    btn_analyze_email.click(detect_email, inputs=email_input, outputs=[summary,explanation,technical,risk_bar])
    btn_analyze_image.click(detect_image, inputs=image_input, outputs=[summary,explanation,technical,risk_bar])
    btn_ask_chat.click(chatbot_ai, inputs=chat_input, outputs=chat_output)

# Launch with custom CSS
app.launch(css=custom_css)
