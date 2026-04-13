import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, time, torch

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdSent-VI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUT_DIR = "./vfnd_experiment/output"
DATA_DIR   = "./vfnd_experiment/data"

# ── CSS (Light theme) ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .main { background: #F8F9FB; }
    section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }
    h1,h2,h3,h4 { color: #111827; }

    /* Cards */
    .kpi-card {
        background: #FFFFFF; border-radius: 14px; padding: 20px 16px;
        border: 1px solid #E5E7EB; text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .kpi-val  { font-size: 2.1rem; font-weight: 800; line-height: 1.1; }
    .kpi-lbl  { font-size: 0.82rem; color: #6B7280; margin-top: 5px; }
    .kpi-sub  { font-size: 0.82rem; margin-top: 5px; font-weight: 600; }

    /* Section title */
    .sec-title {
        font-size: 1.15rem; font-weight: 700; color: #111827;
        border-left: 4px solid #2563EB; padding-left: 10px;
        margin: 28px 0 14px 0;
    }

    /* Alert boxes */
    .box-green  { background:#F0FDF4; border-left:4px solid #16A34A; border-radius:10px; padding:14px; margin:6px 0; }
    .box-red    { background:#FFF1F2; border-left:4px solid #DC2626; border-radius:10px; padding:14px; margin:6px 0; }
    .box-blue   { background:#EFF6FF; border-left:4px solid #2563EB; border-radius:10px; padding:14px; margin:6px 0; }
    .box-yellow { background:#FFFBEB; border-left:4px solid #D97706; border-radius:10px; padding:14px; margin:6px 0; }

    /* Result badge */
    .badge-fake { background:#FEE2E2; color:#991B1B; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1rem; }
    .badge-real { background:#D1FAE5; color:#065F46; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1rem; }
    .badge-pos  { background:#FEF9C3; color:#854D0E; padding:4px 12px; border-radius:12px; font-weight:600; font-size:0.85rem; }
    .badge-neg  { background:#FEE2E2; color:#991B1B; padding:4px 12px; border-radius:12px; font-weight:600; font-size:0.85rem; }
    .badge-neu  { background:#F1F5F9; color:#475569; padding:4px 12px; border-radius:12px; font-weight:600; font-size:0.85rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #F3F4F6; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 18px;
        font-weight: 600; color: #6B7280;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important; color: #2563EB !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }

    /* Input area */
    .stTextArea textarea { border-radius: 10px; border: 1.5px solid #D1D5DB; font-size: 0.95rem; }
    .stButton>button {
        border-radius: 10px; font-weight: 600; font-size: 0.95rem;
        padding: 10px 28px; transition: all 0.2s;
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(37,99,235,0.25); }

    /* Progress bar custom */
    .prob-bar-wrap { background:#F3F4F6; border-radius:8px; height:12px; overflow:hidden; margin:4px 0; }
    .prob-bar-fill-fake { background:linear-gradient(90deg,#EF4444,#DC2626); height:100%; border-radius:8px; transition:width 0.5s; }
    .prob-bar-fill-real { background:linear-gradient(90deg,#10B981,#059669); height:100%; border-radius:8px; transition:width 0.5s; }
</style>
""", unsafe_allow_html=True)

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import joblib

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        PHOBERT = "vinai/phobert-base-v2"

        tokenizer = AutoTokenizer.from_pretrained(PHOBERT)

        # Baseline PhoBERT
        baseline = AutoModelForSequenceClassification.from_pretrained(PHOBERT, num_labels=2).to(DEVICE)
        baseline_w = torch.load(os.path.join(OUTPUT_DIR, "phobert_baseline.pt"),
                                map_location=DEVICE, weights_only=True)
        baseline.load_state_dict(baseline_w)
        baseline.eval()

        return {"tokenizer": tokenizer, "baseline": baseline,
                "device": DEVICE, "loaded": True}
    except Exception as e:
        return {"loaded": False, "error": str(e)}

def load_qwen():
    try:
        from openai import OpenAI
        # Đọc key trực tiếp mỗi lần — không cache
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            return {"loaded": False, "error": "No API key"}
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        return {"client": client, "loaded": True}
    except Exception as e:
        return {"loaded": False, "error": str(e)}

# ── Helper functions ──────────────────────────────────────────────────────────
def predict_text(text, models):
    from transformers import AutoTokenizer
    from torch.utils.data import Dataset, DataLoader

    tokenizer = models["tokenizer"]
    model     = models["baseline"]
    device    = models["device"]

    enc = tokenizer(text, max_length=256, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        out    = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device))
        probs  = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
    return {"real": float(probs[0]), "fake": float(probs[1]),
            "label": "FAKE" if probs[1] > probs[0] else "REAL"}

def call_qwen_api(prompt, qwen, max_retries=2):
    for _ in range(max_retries):
        try:
            r = qwen["client"].chat.completions.create(
                model="qwen/qwen3.6-plus:free",
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            time.sleep(3)
    return None

def rewrite_sentiment_api(text, sentiment, qwen):
    prompt = f"""Rewrite the following Vietnamese article with {sentiment} sentiment but do not change any facts!
Do not include the prompt in the response and do not summarize or expand the original article!

Article:
{text[:1200]}

Rewritten article ({sentiment}):"""
    return call_qwen_api(prompt, qwen)

def neutralize_api(text, qwen):
    prompt = f"""Hãy viết lại đoạn văn bản tiếng Việt dưới đây theo giọng hoàn toàn trung lập, khách quan.
Giữ nguyên hoàn toàn các sự kiện, số liệu, tên người, địa điểm. Chỉ trả về đoạn văn đã viết lại.

Văn bản gốc:
{text[:1200]}

Văn bản trung lập:"""
    return call_qwen_api(prompt, qwen)

def detect_sentiment_api(text, qwen):
    prompt = f"""Đoạn văn bản sau mang giọng điệu gì? Chỉ trả lời đúng 1 từ: positive, negative, hoặc neutral.

Văn bản: {text[:600]}

Giọng điệu:"""
    r = call_qwen_api(prompt, qwen)
    if r:
        r = r.lower()
        if "positive" in r: return "positive"
        if "negative" in r: return "negative"
    return "neutral"

# ── Static data ───────────────────────────────────────────────────────────────
@st.cache_data
def get_results():
    multirun = pd.DataFrame([
        {"Model":"Baseline","Test":"Original", "F1":0.8465,"F1_std":0.0182,"Acc":0.8529,"Acc_std":0.0186},
        {"Model":"Baseline","Test":"Neutral",  "F1":0.7254,"F1_std":0.0565,"Acc":0.7588,"Acc_std":0.0432},
        {"Model":"AdSent",  "Test":"Original", "F1":0.7878,"F1_std":0.0179,"Acc":0.7941,"Acc_std":0.0186},
        {"Model":"AdSent",  "Test":"Neutral",  "F1":0.8056,"F1_std":0.0150,"Acc":0.8118,"Acc_std":0.0144},
    ])
    sentiment = pd.DataFrame([
        {"Set":"Original","Base_F1":0.8464,"AdSent_F1":0.7850,"Base_FFR":0,"Ad_FFR":0},
        {"Set":"Positive","Base_F1":0.8464,"AdSent_F1":0.8179,"Base_FFR":0,"Ad_FFR":0},
        {"Set":"Negative","Base_F1":0.8179,"AdSent_F1":0.7896,"Base_FFR":0,"Ad_FFR":0},
        {"Set":"Neutral", "Base_F1":0.6900,"AdSent_F1":0.8179,"Base_FFR":5,"Ad_FFR":0},
    ])
    consistency = pd.DataFrame([
        {"Variant":"Neutral","Base_F1":0.6900,"Base_FFR":5,"Ad_F1":0.8179,"Ad_FFR":0},
        {"Variant":"Pos→Neu","Base_F1":0.6458,"Base_FFR":6,"Ad_F1":0.7896,"Ad_FFR":0},
        {"Variant":"Neg→Neu","Base_F1":0.5983,"Base_FFR":7,"Ad_F1":0.8179,"Ad_FFR":0},
        {"Variant":"Neu→Neu","Base_F1":0.4902,"Base_FFR":9,"Ad_F1":0.8179,"Ad_FFR":0},
    ])
    fact = pd.DataFrame([
        {"Sentiment":"Positive","LLM_Judge":79.4,"Token_Overlap":0.482},
        {"Sentiment":"Negative","LLM_Judge":100.0,"Token_Overlap":0.514},
        {"Sentiment":"Neutral", "LLM_Judge":100.0,"Token_Overlap":0.462},
    ])
    return multirun, sentiment, consistency, fact

multirun_df, sentiment_df, consistency_df, fact_df = get_results()
PLOT_CFG = dict(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(color="#111827", family="Inter, sans-serif"),
                margin=dict(t=36,b=20,l=10,r=10))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.markdown("## AdSent-VI")
    st.markdown("Robust Fake News Detection  \nunder Adversarial Sentiment Attacks")
    st.divider()

    # API Key input
    st.markdown("### 🔑 API Key (Qwen)")
    api_key = st.text_input("OpenRouter API Key", type="password",
                             placeholder="sk-or-...",
                             help="Lấy tại openrouter.ai/keys — cần để dùng Demo & Attack Simulator")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
        st.success("Key đã được set ✓")

    st.divider()
    st.markdown("### 📊 Dataset: VFND")
    m1,m2 = st.columns(2)
    m1.metric("Tổng mẫu","225"); m2.metric("Train","157")
    m3,m4 = st.columns(2)
    m3.metric("Val","34");       m4.metric("Test","34")
    st.divider()
    st.markdown("### 🧠 Models")
    st.markdown("- **Baseline**: PhoBERT (orig data)\n- **AdSent**: PhoBERT (neutral data)")
    st.markdown("### 🔄 LLM")
    st.markdown("- Rewrite/Neutralize: **Qwen3.6**\n- Judge: **Qwen3.6**")
    st.divider()
    st.caption("Tái hiện: *Tahmasebi et al., arXiv:2601.15277*")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ AdSent-VI")
st.markdown("**Robust Fake News Detection under Adversarial Sentiment Attacks** — Vietnamese Adaptation (VFND Dataset)")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_demo, tab_attack, tab_results, tab_consistency, tab_fact, tab_data = st.tabs([
    "🔍 Live Demo",
    "⚔️ Attack Simulator",
    "📊 Experiment Results",
    "🔄 Consistency Check",
    "✅ Fact Preservation",
    "📁 Data Explorer",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DEMO
# ════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown('<div class="sec-title">🔍 Phát hiện Fake News theo thời gian thực</div>',
                unsafe_allow_html=True)
    st.markdown("Nhập một bài báo tiếng Việt để phân tích: nhận dạng fake/real, phát hiện sentiment, và xem bản trung lập hóa.")

    # Sample articles
    samples = {
        "-- Chọn bài mẫu --": "",
        "🟢 Real: Zverev vô địch ATP Finals":
            "Đánh bại Djokovic, Zverev lần đầu vô địch Giải quần vợt ATP Finals. Alexander Zverev đã xuất sắc vượt qua Novak Djokovic với tỷ số 2-0 (6-4, 6-3) trong trận chung kết ATP Finals 2018 tại London. Đây là lần đầu tiên tay vợt người Đức giành được danh hiệu quan trọng này sau nhiều năm nỗ lực. Zverev cho thấy bản lĩnh vượt trội khi liên tục tạo ra những cú đánh chính xác vào hai góc sân, khiến Djokovic không thể phản ứng kịp.",
        "🔴 Fake: Thủ tướng Abe xin lỗi (fake)":
            "Thủ tướng Abe cúi đầu xin lỗi vì hành động phi thể thao của tuyển Nhật. Theo Sankei Sports, sáng nay Thủ tướng Nhật Bản Shinzo Abe công khai gửi lời xin lỗi tới Nhật hoàng và toàn bộ người dân vì tinh thần thi đấu phi thể thao của đội tuyển Nhật tại World Cup 2018. Với tinh thần của những võ sĩ đạo Samurai, nhưng đội tuyển Nhật Bản đã có những hành động thiếu tinh thần thượng võ.",
    }

    col_select, _ = st.columns([2, 1])
    choice = col_select.selectbox("Chọn bài mẫu hoặc tự nhập:", list(samples.keys()))

    input_text = st.text_area(
        "Nội dung bài báo tiếng Việt:",
        value=samples[choice],
        height=160,
        placeholder="Nhập hoặc dán nội dung bài báo tiếng Việt vào đây...",
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    run_basic   = col_btn1.button("🔍 Phân tích", type="primary", use_container_width=True)
    run_full    = col_btn2.button("🧪 Phân tích đầy đủ + Neutralize", use_container_width=True)

    if (run_basic or run_full) and input_text.strip():
        models = load_models()
        qwen   = load_qwen()

        st.divider()

        # ── Prediction ──
        if models["loaded"]:
            with st.spinner("Đang phân tích..."):
                pred = predict_text(input_text, models)

            col_pred, col_prob = st.columns([1, 2])
            with col_pred:
                st.markdown("#### 🎯 Kết quả phân loại")
                badge = "badge-fake" if pred["label"] == "FAKE" else "badge-real"
                icon  = "🔴" if pred["label"] == "FAKE" else "🟢"
                st.markdown(f'<div style="text-align:center;padding:20px">'
                            f'<span class="{badge}">{icon} {pred["label"]}</span>'
                            f'</div>', unsafe_allow_html=True)
                conf = max(pred["fake"], pred["real"])
                st.markdown(f"**Độ tin cậy:** {conf:.1%}")

            with col_prob:
                st.markdown("#### 📊 Xác suất")
                st.markdown(f"**🔴 Fake:** {pred['fake']:.1%}")
                fake_w = int(pred['fake'] * 100)
                st.markdown(f'<div class="prob-bar-wrap"><div class="prob-bar-fill-fake" style="width:{fake_w}%"></div></div>',
                            unsafe_allow_html=True)
                st.markdown(f"**🟢 Real:** {pred['real']:.1%}")
                real_w = int(pred['real'] * 100)
                st.markdown(f'<div class="prob-bar-wrap"><div class="prob-bar-fill-real" style="width:{real_w}%"></div></div>',
                            unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ Không load được PhoBERT model: {models.get('error','')}")

        # ── Sentiment + Neutralize ──
        if run_full:
            if qwen["loaded"]:
                col_sent, col_neut = st.columns(2)

                with col_sent:
                    st.markdown("#### 🎭 Phân tích Sentiment")
                    with st.spinner("Đang phát hiện sentiment..."):
                        sentiment_label = detect_sentiment_api(input_text, qwen)
                    badge_map = {"positive": "badge-pos", "negative": "badge-neg", "neutral": "badge-neu"}
                    icon_map  = {"positive": "😊", "negative": "😠", "neutral": "😐"}
                    badge_cls = badge_map.get(sentiment_label, "badge-neu")
                    icon_s    = icon_map.get(sentiment_label, "😐")
                    st.markdown(f'<div style="padding:12px 0">'
                                f'<span class="{badge_cls}">{icon_s} {sentiment_label.upper()}</span>'
                                f'</div>', unsafe_allow_html=True)

                    bias_msg = {
                        "positive": "⚠️ Sentiment POSITIVE có thể khiến fake news bị nhận nhầm thành real.",
                        "negative": "⚠️ Sentiment NEGATIVE có thể khiến real news bị nhận nhầm thành fake.",
                        "neutral":  "✅ Sentiment NEUTRAL — detector ít bị bias nhất với loại này.",
                    }
                    st.info(bias_msg.get(sentiment_label, ""))

                with col_neut:
                    st.markdown("#### 🔄 Bản Neutralized")
                    with st.spinner("Đang neutralize..."):
                        neutralized = neutralize_api(input_text, qwen)

                    if neutralized:
                        st.text_area("Bản trung lập hóa:", value=neutralized, height=160)

                        if models["loaded"]:
                            pred_neut = predict_text(neutralized, models)
                            col_a, col_b = st.columns(2)
                            col_a.metric("Kết quả gốc", pred["label"],
                                         delta=f"Fake {pred['fake']:.1%}")
                            col_b.metric("Kết quả sau neutralize", pred_neut["label"],
                                         delta=f"Fake {pred_neut['fake']:.1%}")

                            if pred["label"] != pred_neut["label"]:
                                st.markdown("""<div class="box-red">
                                    <b>⚠️ Prediction Flip detected!</b><br>
                                    Kết quả thay đổi sau khi neutralize — bài này nhạy cảm với sentiment manipulation.
                                </div>""", unsafe_allow_html=True)
                            else:
                                st.markdown("""<div class="box-green">
                                    <b>✅ Prediction ổn định</b><br>
                                    Kết quả không thay đổi sau khi neutralize.
                                </div>""", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Cần nhập OpenRouter API Key ở sidebar để dùng tính năng này.")

    elif (run_basic or run_full) and not input_text.strip():
        st.warning("⚠️ Vui lòng nhập nội dung bài báo.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ATTACK SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
with tab_attack:
    st.markdown('<div class="sec-title">⚔️ Attack Simulator — Thử nghiệm tấn công Sentiment</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Công cụ này mô phỏng **adversarial sentiment attack**: viết lại bài báo sang 3 giọng điệu khác nhau
    và xem detector phản ứng như thế nào — đúng theo quy trình của paper AdSent.
    """)

    atk_text = st.text_area(
        "Nhập bài báo cần tấn công:",
        height=140,
        placeholder="Dán nội dung bài báo tiếng Việt vào đây...",
        key="attack_input",
    )

    sentiments_to_run = st.multiselect(
        "Chọn loại sentiment attack:",
        ["positive", "negative", "neutral"],
        default=["positive", "negative", "neutral"],
    )

    run_attack = st.button("🚀 Chạy Attack Simulation", type="primary")

    if run_attack and atk_text.strip():
        models = load_models()
        qwen   = load_qwen()

        if not qwen["loaded"]:
            st.warning("⚠️ Cần nhập OpenRouter API Key ở sidebar để dùng Attack Simulator.")
        else:
            results_attack = {}

            # Original prediction
            if models["loaded"]:
                pred_orig = predict_text(atk_text, models)
                results_attack["Original"] = {"text": atk_text, "pred": pred_orig, "sentiment": "original"}

            progress = st.progress(0, text="Đang chạy attack simulation...")
            for i, s in enumerate(sentiments_to_run):
                progress.progress((i+1)/len(sentiments_to_run),
                                   text=f"Đang rewrite sang {s}...")
                rewritten = rewrite_sentiment_api(atk_text, s, qwen)
                if rewritten and models["loaded"]:
                    pred_rw = predict_text(rewritten, models)
                    results_attack[s.capitalize()] = {
                        "text": rewritten, "pred": pred_rw, "sentiment": s
                    }
            progress.empty()

            # Display results
            st.divider()
            st.markdown("#### 📊 Kết quả Attack Simulation")

            cols = st.columns(len(results_attack))
            for i, (name, res) in enumerate(results_attack.items()):
                with cols[i]:
                    p = res["pred"]
                    badge = "badge-fake" if p["label"]=="FAKE" else "badge-real"
                    icon  = "🔴" if p["label"]=="FAKE" else "🟢"

                    # Check flip
                    orig_label = results_attack.get("Original",{}).get("pred",{}).get("label","")
                    is_flip = (name != "Original") and (p["label"] != orig_label) and orig_label

                    flip_badge = ' <span style="background:#FEF3C7;color:#92400E;padding:2px 8px;border-radius:8px;font-size:0.75rem;font-weight:700">⚡ FLIP</span>' if is_flip else ""

                    st.markdown(f"**{name}**{flip_badge}", unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align:center;padding:10px 0">'
                                f'<span class="{badge}">{icon} {p["label"]}</span></div>',
                                unsafe_allow_html=True)
                    fake_w = int(p["fake"]*100)
                    real_w = int(p["real"]*100)
                    st.markdown(f"🔴 Fake: **{p['fake']:.1%}**")
                    st.markdown(f'<div class="prob-bar-wrap"><div class="prob-bar-fill-fake" style="width:{fake_w}%"></div></div>', unsafe_allow_html=True)
                    st.markdown(f"🟢 Real: **{p['real']:.1%}**")
                    st.markdown(f'<div class="prob-bar-wrap"><div class="prob-bar-fill-real" style="width:{real_w}%"></div></div>', unsafe_allow_html=True)
                    with st.expander("Xem text đã rewrite"):
                        st.write(res["text"][:600] + "..." if len(res["text"])>600 else res["text"])

            # Visualization
            if len(results_attack) > 1:
                st.divider()
                st.markdown("#### 📈 So sánh xác suất Fake theo sentiment")
                fig_atk = go.Figure()
                names   = list(results_attack.keys())
                fakes   = [results_attack[n]["pred"]["fake"] for n in names]
                colors  = ["#6B7280","#F59E0B","#EF4444","#94A3B8"]
                flips   = [n for n in names if n!="Original" and
                           results_attack[n]["pred"]["label"] != results_attack["Original"]["pred"]["label"]]

                fig_atk.add_trace(go.Bar(
                    x=names, y=fakes,
                    marker_color=colors[:len(names)],
                    text=[f"{v:.1%}" for v in fakes],
                    textposition="outside",
                ))
                fig_atk.add_hline(y=0.5, line_dash="dash", line_color="#DC2626",
                                  annotation_text="Decision boundary (0.5)")
                fig_atk.update_layout(
                    height=320, yaxis=dict(range=[0,1.1], title="P(Fake)", gridcolor="#F3F4F6"),
                    xaxis=dict(gridcolor="#F3F4F6"),
                    showlegend=False, **PLOT_CFG
                )
                st.plotly_chart(fig_atk, use_container_width=True)

                if flips:
                    st.markdown(f"""<div class="box-red">
                        <b>⚡ Prediction Flip phát hiện!</b><br>
                        Các sentiment gây flip: <b>{', '.join(flips)}</b><br>
                        Đây là bằng chứng detector dễ bị tấn công bởi sentiment manipulation.
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="box-green">
                        <b>✅ Không có Prediction Flip</b><br>
                        Detector ổn định với bài báo này trên tất cả sentiment variants.
                    </div>""", unsafe_allow_html=True)

    elif run_attack and not atk_text.strip():
        st.warning("⚠️ Vui lòng nhập nội dung bài báo.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPERIMENT RESULTS
# ════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.markdown('<div class="sec-title">📊 Kết quả thực nghiệm (5 runs, mean ± std)</div>',
                unsafe_allow_html=True)

    # KPI row
    c1,c2,c3,c4 = st.columns(4)
    kpis = [
        ("0.8465", "#2563EB", "Baseline F1\n(Original)", "PhoBERT + orig data"),
        ("0.7254", "#DC2626", "Baseline F1\n(Neutral Attack)", "▼ −0.121 drop"),
        ("0.8056", "#16A34A", "AdSent F1\n(Neutral Attack)", "▲ +0.080 vs Baseline"),
        ("−0.018", "#D97706", "AdSent F1 Drop", "Near-zero degradation"),
    ]
    for col, (val, clr, lbl, sub) in zip([c1,c2,c3,c4], kpis):
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-val" style="color:{clr}">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
            <div class="kpi-sub" style="color:{clr}">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col_chart, col_insight = st.columns([3, 2])

    with col_chart:
        fig_main = go.Figure()
        color_map = {("Baseline","Original"):"#3B82F6",("Baseline","Neutral"):"#EF4444",
                     ("AdSent","Original"):"#10B981",  ("AdSent","Neutral"):"#F59E0B"}
        for _, row in multirun_df.iterrows():
            fig_main.add_trace(go.Bar(
                name=f"{row['Model']} / {row['Test']}",
                x=[f"{row['Model']}<br>{row['Test']}"],
                y=[row['F1']],
                error_y=dict(type='data',array=[row['F1_std']],visible=True,thickness=2),
                marker_color=color_map[(row['Model'],row['Test'])],
                text=f"{row['F1']:.4f}",
                textposition='outside', width=0.5,
            ))
        fig_main.update_layout(
            height=380, showlegend=False,
            yaxis=dict(range=[0.55,1.0], title="Macro F1", gridcolor="#F3F4F6"),
            xaxis=dict(gridcolor="#F3F4F6"),
            title="Baseline vs AdSent — Macro F1 (mean ± std, 5 runs)",
            **PLOT_CFG
        )
        st.plotly_chart(fig_main, use_container_width=True)

    with col_insight:
        st.markdown("#### 💡 Phân tích")
        st.markdown("""<div class="box-red">
            <b>⚠️ Baseline dễ bị tấn công</b><br>
            F1 giảm <b>0.121</b> trên neutral set. Std cao (±0.057) = không ổn định.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="box-green">
            <b>✅ AdSent robust hơn rõ rệt</b><br>
            F1 drop chỉ <b>0.018</b>. Std thấp (±0.015) = ổn định.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="box-blue">
            <b>🔄 Đánh đổi hợp lý</b><br>
            AdSent yếu hơn 0.059 trên original, đổi lại robust hơn 0.080 trên adversarial — đúng tinh thần paper.
        </div>""", unsafe_allow_html=True)

    # Sentiment impact
    st.markdown('<div class="sec-title">⚔️ Tác động của từng loại Sentiment (Table 3)</div>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns([3,2])
    with col_a:
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Bar(name="Baseline", x=sentiment_df["Set"],
                                  y=sentiment_df["Base_F1"], marker_color="#3B82F6",
                                  text=[f"{v:.4f}" for v in sentiment_df["Base_F1"]],
                                  textposition="outside"))
        fig_sent.add_trace(go.Bar(name="AdSent", x=sentiment_df["Set"],
                                  y=sentiment_df["AdSent_F1"], marker_color="#10B981",
                                  text=[f"{v:.4f}" for v in sentiment_df["AdSent_F1"]],
                                  textposition="outside"))
        fig_sent.update_layout(
            height=340, barmode="group",
            yaxis=dict(range=[0.5,1.0],title="Macro F1",gridcolor="#F3F4F6"),
            xaxis=dict(gridcolor="#F3F4F6"),
            title="F1 theo sentiment variant",
            legend=dict(bgcolor="#FFFFFF", bordercolor="#E5E7EB", borderwidth=1),
            **PLOT_CFG
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_b:
        st.markdown("#### 🔄 FF→R Flips (Fake → nhầm Real)")
        fig_flip = go.Figure()
        fig_flip.add_trace(go.Bar(name="Baseline", x=sentiment_df["Set"],
                                  y=sentiment_df["Base_FFR"], marker_color="#EF4444",
                                  text=sentiment_df["Base_FFR"], textposition="outside"))
        fig_flip.add_trace(go.Bar(name="AdSent", x=sentiment_df["Set"],
                                  y=sentiment_df["Ad_FFR"], marker_color="#10B981",
                                  text=sentiment_df["Ad_FFR"], textposition="outside"))
        fig_flip.update_layout(
            height=280, barmode="group",
            yaxis=dict(title="Count", gridcolor="#F3F4F6"),
            legend=dict(bgcolor="#FFFFFF", bordercolor="#E5E7EB", borderwidth=1),
            **PLOT_CFG
        )
        st.plotly_chart(fig_flip, use_container_width=True)

        st.markdown("""<div class="box-yellow">
            <b>Neutral gây khó nhất</b><br>
            Baseline FF→R = <b>5</b> trên neutral.<br>
            AdSent FF→R = <b>0</b> — hoàn toàn ổn định.
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — CONSISTENCY CHECK
# ════════════════════════════════════════════════════════════════════════════
with tab_consistency:
    st.markdown('<div class="sec-title">🔄 LLM Consistency Check (Figure 4 của paper)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Kiểm tra neutralization có nhất quán không: dù bắt đầu từ **Positive → Neutral**, **Negative → Neutral**,
    hay **Neutral → Neutral**, detector có cho kết quả như nhau không?
    """)

    col1, col2 = st.columns([3,2])
    with col1:
        fig_con = make_subplots(specs=[[{"secondary_y":True}]])
        fig_con.add_trace(go.Bar(
            name="Baseline F1", x=consistency_df["Variant"], y=consistency_df["Base_F1"],
            marker_color="#EF4444", opacity=0.85,
            text=[f"{v:.4f}" for v in consistency_df["Base_F1"]],
            textposition="outside",
        ))
        fig_con.add_trace(go.Bar(
            name="AdSent F1", x=consistency_df["Variant"], y=consistency_df["Ad_F1"],
            marker_color="#10B981", opacity=0.85,
            text=[f"{v:.4f}" for v in consistency_df["Ad_F1"]],
            textposition="outside",
        ))
        fig_con.add_trace(go.Scatter(
            name="Baseline FF→R", x=consistency_df["Variant"], y=consistency_df["Base_FFR"],
            mode="lines+markers+text", line=dict(color="#F59E0B",width=2.5,dash="dot"),
            marker=dict(size=10,color="#F59E0B"),
            text=consistency_df["Base_FFR"], textposition="top center",
        ), secondary_y=True)
        fig_con.update_layout(
            height=380, barmode="group",
            legend=dict(bgcolor="#FFFFFF",bordercolor="#E5E7EB",borderwidth=1),
            title="F1 và FF→R flips theo neutralization variant",
            **PLOT_CFG
        )
        fig_con.update_yaxes(title_text="Macro F1", range=[0.3,1.0],
                             gridcolor="#F3F4F6", secondary_y=False)
        fig_con.update_yaxes(title_text="FF→R Count", range=[0,14],
                             secondary_y=True)
        st.plotly_chart(fig_con, use_container_width=True)

    with col2:
        st.markdown("#### 📋 Bảng chi tiết")
        disp = consistency_df.rename(columns={
            "Variant":"Variant","Base_F1":"Base F1","Base_FFR":"Base FF→R",
            "Ad_F1":"AdSent F1","Ad_FFR":"AdSent FF→R"
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.markdown("""<div class="box-red">
            <b>⚠️ Baseline ngày càng tệ hơn</b><br>
            Neu→Neu: F1 = 0.490, FF→R = <b>9</b><br>
            Second-level neutralization làm fake khó detect hơn nhiều.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="box-green">
            <b>✅ AdSent hoàn toàn ổn định</b><br>
            FF→R = <b>0</b> trên tất cả variants.<br>
            F1 dao động 0.789–0.818, std thấp.
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — FACT PRESERVATION
# ════════════════════════════════════════════════════════════════════════════
with tab_fact:
    st.markdown('<div class="sec-title">✅ Fact Preservation Check (Section 4.2.3)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Kiểm tra xem bản rewrite có giữ nguyên facts không — dùng 2 phương pháp:
    **LLM-as-a-Judge** (đúng chuẩn paper) và **Token Overlap** (phương pháp đơn giản).
    """)

    col1, col2 = st.columns(2)
    with col1:
        fig_judge = go.Figure(go.Bar(
            x=fact_df["Sentiment"], y=fact_df["LLM_Judge"],
            marker_color=["#F59E0B","#10B981","#10B981"],
            text=[f"{v:.1f}%" for v in fact_df["LLM_Judge"]],
            textposition="outside",
        ))
        fig_judge.add_hline(y=100, line_dash="dot", line_color="#6B7280",
                            annotation_text="100% threshold")
        fig_judge.update_layout(
            height=320, yaxis=dict(range=[0,120],title="Preservation Rate (%)",gridcolor="#F3F4F6"),
            title="🤖 LLM-as-a-Judge (Paper Method)", **PLOT_CFG
        )
        st.plotly_chart(fig_judge, use_container_width=True)

    with col2:
        fig_tok = go.Figure(go.Bar(
            x=fact_df["Sentiment"], y=fact_df["Token_Overlap"],
            marker_color=["#F59E0B","#3B82F6","#6B7280"],
            text=[f"{v:.3f}" for v in fact_df["Token_Overlap"]],
            textposition="outside",
        ))
        fig_tok.update_layout(
            height=320, yaxis=dict(range=[0,0.7],title="Token Overlap Score",gridcolor="#F3F4F6"),
            title="🔤 Token Overlap Score (Simple Method)", **PLOT_CFG
        )
        st.plotly_chart(fig_tok, use_container_width=True)

    comp_df = fact_df.copy()
    comp_df["LLM_Judge"]      = comp_df["LLM_Judge"].apply(lambda x: f"{x:.1f}%")
    comp_df["Token_Overlap"]  = comp_df["Token_Overlap"].apply(lambda x: f"{x:.3f}")
    comp_df.columns = ["Sentiment","LLM Judge (Paper)","Token Overlap (Simple)"]
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("""<div class="box-blue">
        <b>💡 LLM Judge vs Token Overlap</b><br>
        LLM Judge hiểu ngữ nghĩa → negative/neutral đạt 100% fact preserved.
        Token Overlap chỉ đo từng token riêng lẻ nên bị ảnh hưởng bởi cách diễn đạt.
        Positive thấp hơn (79.4%) vì hay thêm từ mô tả mới — đúng với quan sát của paper (Figure 3).
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="sec-title">📁 Data Explorer</div>', unsafe_allow_html=True)

    data_files = {
        "📄 Clean Dataset":        os.path.join(DATA_DIR, "vfnd_clean.csv"),
        "🚂 Train Set":            os.path.join(DATA_DIR, "train.csv"),
        "✅ Test Set":             os.path.join(DATA_DIR, "test.csv"),
        "⚔️ Test Rewritten":      os.path.join(DATA_DIR, "test_rewritten.csv"),
        "🎯 Test Adversarial":     os.path.join(DATA_DIR, "test_adversarial.csv"),
        "🔄 Test Consistency":     os.path.join(DATA_DIR, "test_consistency.csv"),
        "🧹 Train Neutralized":    os.path.join(DATA_DIR, "train_neutralized.csv"),
    }

    selected = st.selectbox("Chọn file dữ liệu để xem:", list(data_files.keys()))
    fpath = data_files[selected]

    if os.path.exists(fpath):
        df_view = pd.read_csv(fpath, encoding="utf-8-sig")
        c1,c2,c3 = st.columns(3)
        c1.metric("Số mẫu", len(df_view))
        c2.metric("Số cột", len(df_view.columns))
        if "label" in df_view.columns:
            c3.metric("Fake / Real",
                      f"{(df_view['label']==1).sum()} / {(df_view['label']==0).sum()}")
        else:
            c3.metric("Columns", ", ".join(df_view.columns[:3]))

        # Label distribution
        if "label" in df_view.columns:
            col_dist, col_table = st.columns([1,2])
            with col_dist:
                fig_dist = go.Figure(go.Pie(
                    labels=["Real","Fake"],
                    values=[( df_view["label"]==0).sum(),(df_view["label"]==1).sum()],
                    marker_colors=["#10B981","#EF4444"],
                    hole=0.5,
                ))
                fig_dist.update_layout(height=220, showlegend=True,
                                       margin=dict(t=20,b=10,l=10,r=10),
                                       **{k:v for k,v in PLOT_CFG.items() if k!="margin"})
                st.plotly_chart(fig_dist, use_container_width=True)
            with col_table:
                st.dataframe(df_view.head(10), use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_view.head(10), use_container_width=True, hide_index=True)

        # Download
        csv_bytes = df_view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(f"⬇️ Download {selected}", csv_bytes,
                           file_name=os.path.basename(fpath),
                           mime="text/csv")
    else:
        st.warning(f"⚠️ File chưa tồn tại: `{fpath}`")

    st.divider()
    st.markdown("#### 📋 Tất cả experiment files")
    all_files = list(data_files.values()) + [
        os.path.join(OUTPUT_DIR, f) for f in
        ["baseline_model.pkl","phobert_baseline.pt","final_table_multirun.csv",
         "llm_judge_results.csv","consistency_results.csv","adversarial_results.csv"]
    ]
    col1, col2, col3 = st.columns(3)
    for i, f in enumerate(all_files):
        exists = os.path.exists(f)
        icon   = "✅" if exists else "❌"
        [col1,col2,col3][i%3].markdown(f"{icon} `{os.path.basename(f)}`")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Reference:** Tahmasebi, S., Müller-Budack, E., & Ewerth, R. (2026). "
    "*Robust Fake News Detection using Large Language Models under Adversarial Sentiment Attacks.* "
    "arXiv:2601.15277 | "
    "**Dataset:** VFND — Vietnamese Fake News Dataset (Ho Quang Thanh, 2019)"
)
# python -m streamlit run C:\Users\npd20\Downloads\NLP_HUIT_v2\app.py