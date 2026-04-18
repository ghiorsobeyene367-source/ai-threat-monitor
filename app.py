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
    # На сайте используем CPU, так как GPU может не быть
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

tokenizer, embed_model, model, device = load_models()
df_geo, kmeans = load_data()

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


st.sidebar.title("🛡️ AI Threat System")
st.sidebar.info("Информационно-аналитическая система мониторинга злонамеренного использования ИИ.")
page = st.sidebar.radio("Навигация:", ["🌍 Глобальный мониторинг", "🔍 Анализ инцидента (Нейросеть)"])


if page == "🌍 Глобальный мониторинг":
    st.title("Глобальный мониторинг угроз ИИ")
    st.markdown("Аналитика распределения инцидентов по ключевым регионам: **Россия, США, Китай**.")
    

    col1, col2, col3 = st.columns(3)
    col1.metric("Всего зафиксировано инцидентов", f"{len(df_geo)}")
    col2.metric("Преобладающая угроза", df_geo['Cluster'].mode()[0])
    col3.metric("Активных регионов", df_geo['Country'].nunique())
    
    st.markdown("---")
    

    st.subheader("Интерактивная карта инцидентов")
    geo_coords = {'Россия': [61.52, 105.31], 'США': [37.09, -95.71], 'Китай': [35.86, 104.19]}
    m = folium.Map(location=[40, 0], zoom_start=2, tiles='CartoDB positron')
    
    for country in geo_coords.keys():
        top_threats = df_geo[df_geo['Country'] == country]['Cluster'].value_counts().head(3)
        popup_html = f"<b>{country}</b><br><hr><b>ТОП-3 Угрозы ИИ:</b><br>"
        for threat, count in top_threats.items():
            popup_html += f"- {threat}: <i>{count} шт.</i><br>"
            
        folium.Marker(
            location=geo_coords[country],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
    st_folium(m, width=1000, height=400)
    

    st.subheader("Статистика по странам")
    threat_counts = df_geo.groupby(['Country', 'Cluster']).size().unstack(fill_value=0)
    st.bar_chart(threat_counts)


elif page == "🔍 Анализ инцидента":
    st.title("Классификация угроз нейросетью")
    st.write("Введите текст новости на английском языке для автоматического определения типа угрозы.")
    
    st.markdown("---")
    st.subheader("⚙️ Настройки детектора нулевого дня (Zero-Day)")
    threshold = st.slider(
        "Порог уверенности для фильтрации аномалий (%):", 
        min_value=30, max_value=95, value=55, step=5,
        help="Если максимальная уверенность нейросети ниже этого порога, угроза будет помечена как неизвестная (подозрение на угрозу нулевого дня)."
    ) / 100.0
    st.markdown("---")
    
    input_text = st.text_area("Текст новости:", height=200)
    
    if st.button("Проанализировать", type="primary"):
        if input_text:
            with st.spinner("Обработка данных..."):
                emb = get_embedding(input_text)
                logits = model(emb)
                probs = F.softmax(logits, dim=1)[0]
                pred_idx = torch.argmax(logits, 1).item()
                confidence = probs[pred_idx].item()
                
                st.markdown("### 🎯 Результат классификации:")
                
                if confidence < threshold:
                    # Если модель не уверена — бьем тревогу
                    st.error(f"**Категория:** ⚠️ НЕИЗВЕСТНАЯ УГРОЗА / АНОМАЛИЯ (Zero-Day)")
                    st.warning(f"Модель сомневается! Максимальная вероятность ({confidence*100:.1f}%) ниже заданного порога безопасности ({threshold*100:.0f}%). Система рекомендует передать инцидент на ручной анализ ИБ-специалистам.")
                else:
                    if confidence > 0.8:
                        st.success(f"**Категория:** {CLUSTER_NAMES[pred_idx]}")
                    else:
                        st.info(f"**Категория:** {CLUSTER_NAMES[pred_idx]}")
                        
                st.progress(confidence, text=f"Максимальная уверенность модели: {confidence*100:.1f}%")
                
                st.markdown("#### Подробное распределение вероятностей:")
                chart_data = pd.DataFrame({
                    'Тип угрозы': [CLUSTER_NAMES[i] for i in range(4)],
                    'Вероятность (%)': [p.item()*100 for p in probs]
                })
                st.bar_chart(chart_data.set_index('Тип угрозы'))
        else:
            st.warning("Пожалуйста, введите текст.")
