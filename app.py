import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from transformers import AutoTokenizer, AutoModel
import folium
from streamlit_folium import st_folium
from sklearn.metrics.pairwise import cosine_similarity # Добавлено для детектора аномалий


st.set_page_config(page_title="AI Threat Monitor", page_icon="🛡️", layout="wide")

class AIThreatNet(nn.Module):
    def __init__(self, input_size=768, num_classes=4):
        super(AIThreatNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.drop1(self.relu(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


@st.cache_resource
def load_models():
    device = torch.device('cpu') 
    
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embed_model = AutoModel.from_pretrained(model_name).to(device)
    embed_model.eval()

    model = AIThreatNet(input_size=768, num_classes=4).to(device)
    model.load_state_dict(torch.load('ai_threat_model.pth', map_location=device))
    model.eval()
    
    return tokenizer, embed_model, model, device

@st.cache_data
def load_data():
    df_geo = pd.read_csv('geo_threats_data.csv')
    kmeans = joblib.load('kmeans_clusterer.pkl')
    return df_geo, kmeans

try:
    tokenizer, embed_model, model, device = load_models()
    df_geo, kmeans = load_data()
except Exception as e:
    st.error(f"Ошибка загрузки файлов модели: {e}. Убедитесь, что .pth, .pkl и .csv файлы лежат в той же папке, что и app.py")
    st.stop()

CLUSTER_NAMES = {
    0: "Автономные системы и физ. риски",
    1: "Уязвимости генеративного ИИ (LLM)",
    2: "Классические кибератаки и ВПО",
    3: "Deepfake и медиафальсификации"
}


def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_embedding(text):
    cleaned = clean_text(text)
    with torch.no_grad():
        encoded = tokenizer([cleaned], padding=True, truncation=True, return_tensors='pt').to(device)
        model_out = embed_model(**encoded)
        mask = encoded['attention_mask'].unsqueeze(-1).expand(model_out[0].size()).float()
        sum_emb = torch.sum(model_out[0] * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        embedding = sum_emb / sum_mask
    return embedding

def render_metric(label, value):
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; margin-bottom: 1rem;">
            <span style="font-size: 14px; opacity: 0.7;">{label}</span>
            <span style="font-size: 26px; font-weight: bold; line-height: 1.2; word-wrap: break-word; white-space: normal;">{value}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.title("🛡️ AI Threat System")
st.sidebar.info("Информационно-аналитическая система мониторинга злонамеренного использования ИИ")
page = st.sidebar.radio("Навигация:", ["🌍 Глобальный мониторинг", "🔍 Анализ инцидента"])


if page == "🌍 Глобальный мониторинг":
    st.title("Глобальный мониторинг угроз ИИ")
    

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric("Всего инцидентов", f"{len(df_geo)}")
    with col2:
        render_metric("Топ угроза", df_geo['Cluster'].mode()[0])
    with col3:
        render_metric("Активных регионов", df_geo['Country'].nunique())
    
    st.markdown("---")
    

    st.subheader("Карта интенсивности угроз")
    geo_coords = {'Россия': [61.52, 105.31], 'США': [37.09, -95.71], 'Китай': [35.86, 104.19]}
    m = folium.Map(location=[40, 0], zoom_start=2, tiles='CartoDB positron')
    
    for country, coords in geo_coords.items():
        subset = df_geo[df_geo['Country'] == country]
        if not subset.empty:
            top_threats = subset['Cluster'].value_counts().head(3)
            popup_text = f"<b>{country}</b><br><hr>"
            for threat, count in top_threats.items():
                popup_text += f"• {threat}: {count}<br>"
            
            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
    
    st_folium(m, width=1200, height=500)
    

    st.subheader("Статистика по странам")
    threat_counts = df_geo.groupby(['Country', 'Cluster']).size().unstack(fill_value=0)
    st.bar_chart(threat_counts)


elif page == "🔍 Анализ инцидента":
    st.title("Классификация угроз нейросетью")
    st.write("Введите текст новости на английском языке для автоматического определения типа угрозы.")
    
    st.markdown("---")
    st.subheader("⚙️ Настройки детектора аномалий")
    st.info("Настройка детектора аномалий: чем выше процент, тем строже условия для предсказания. Чем ниже процент, тем шире диапазон для отнесения текста к кластерам.")
    threshold = st.slider(
        "Минимальный порог сходства с базой данных (%):", 
        min_value=10, max_value=80, value=30, step=5,
        help="Текст, удаленный от всех кластеров угроз в векторном пространстве (например, рецепт) будет отсеян."
    ) / 100.0
    st.markdown("---")
    
    input_text = st.text_area("Текст новости:", height=200)
    
    if st.button("Проанализировать", type="primary"):
        if input_text:
            with st.spinner("Извлечение семантики и расчет расстояний..."):
                # 1. Получаем эмбеддинг
                emb = get_embedding(input_text)
                emb_np = emb.cpu().numpy()
                
                # 2. Расчет семантического сходства (Косинусное расстояние до центров KMeans)
                cluster_centers = kmeans.cluster_centers_
                sims = cosine_similarity(emb_np, cluster_centers)[0]
                max_sim = np.max(sims) # Максимальное сходство с одним из 4х кластеров
                
                # 3. Предсказание нейросети (для уверенности модели)
                logits = model(emb)
                probs = F.softmax(logits, dim=1)[0]
                pred_idx = torch.argmax(logits, 1).item()
                confidence = probs[pred_idx].item()
                
                st.markdown("### 🎯 Результат классификации:")
                
                # ЛОГИКА ДЕТЕКТОРА:
                if max_sim < threshold:
                    # Текст не похож ни на одну из угроз (Out-of-Distribution)
                    st.error(f"**Статус:** ⚠️ НЕРЕЛЕВАНТНЫЙ ТЕКСТ ИЛИ НЕИЗВЕСТНАЯ АНОМАЛИЯ")
                    st.warning(f"Текст семантически далек от изученных инцидентов.\n\nМаксимальное сходство с базой угроз: **{max_sim*100:.1f}%** (Требуемый порог: {threshold*100:.0f}%).")
                    
                    # Показываем, что нейросеть при этом могла ошибочно быть "уверена"
                    st.caption(f"Справка: Классификатор попытался отнести текст к «{CLUSTER_NAMES[pred_idx]}», но детектор аномалий заблокировал вывод.")
                else:
                    # Текст прошел проверку сходства
                    if confidence > 0.8:
                        st.success(f"**Категория:** {CLUSTER_NAMES[pred_idx]}")
                    else:
                        st.info(f"**Категория:** {CLUSTER_NAMES[pred_idx]}")
                        
                    st.progress(confidence, text=f"Уверенность нейросети: {confidence*100:.1f}% | Сходство с кластером: {max_sim*100:.1f}%")
                    
                st.markdown("#### Подробное распределение вероятностей:")
                prob_dict = {CLUSTER_NAMES[i]: float(probs[i])*100 for i in range(4)}
                
                # Сортируем по убыванию вероятности
                sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
                
                for threat, prob in sorted_probs.items():
                    col1, col2 = st.columns([3, 1])
                    col1.write(threat)
                    col2.write(f"**{prob:.1f}%**")
        else:
            st.warning("Пожалуйста, введите текст.")
