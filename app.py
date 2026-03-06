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


elif page == "🔍 Анализ инцидента (Нейросеть)":
    st.title("Модератор: Анализ новых инцидентов")
    st.markdown("Вставьте текст новости или отчета, и обученная нейронная сеть (PyTorch) определит, к какому кластеру злонамеренного использования ИИ он относится.")
    
    user_input = st.text_area("Текст инцидента (на английском):", height=200, 
                              placeholder="Например: Hackers used deepfake video technology to impersonate the CEO and steal funds...")
    
    if st.button("КЛАССИФИЦИРОВАТЬ УГРОЗУ", type="primary"):
        if len(user_input.split()) < 5:
            st.warning("⚠️ Пожалуйста, введите более подробный текст (минимум 5 слов).")
        else:
            with st.spinner("🧠 Нейросеть анализирует семантику текста..."):

                embedding = get_embedding(user_input)
                

                with torch.no_grad():
                    logits = model(embedding)
                    probs = F.softmax(logits, dim=1)[0]
                    pred_class = torch.argmax(logits, 1).item()
                    confidence = probs[pred_class].item()
                

                st.markdown("### 🎯 Результат классификации:")
                
                # Цветные плашки в зависимости от уверенности
                if confidence > 0.8:
                    st.success(f"**Категория:** {CLUSTER_NAMES[pred_class]}")
                elif confidence > 0.5:
                    st.info(f"**Категория:** {CLUSTER_NAMES[pred_class]}")
                else:
                    st.warning(f"**Категория (Низкая уверенность):** {CLUSTER_NAMES[pred_class]}")
                
                st.progress(confidence, text=f"Уверенность модели: {confidence*100:.1f}%")
                

                st.markdown("#### Подробное распределение вероятностей:")
                prob_dict = {CLUSTER_NAMES[i]: float(probs[i])*100 for i in range(4)}
                
                # Сортируем по убыванию вероятности
                sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
                
                for threat, prob in sorted_probs.items():
                    col1, col2 = st.columns([3, 1])
                    col1.write(threat)
                    col2.write(f"**{prob:.1f}%**")