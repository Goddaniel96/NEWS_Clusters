import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np
import umap
import plotly.express as px
import nltk
nltk.download('punkt_tab')
from tqdm import tqdm 

# --- 0.use nltk (seperate text)---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: 
    print("ไม่พบโมเดล 'punkt' ของ NLTK, กำลังดาวน์โหลด...")
    nltk.download('punkt')

# 1.setting 
FILE_PATH = 'cnbc_news.parquet'
TEXT_COLUMN = 'text'
MODEL_NAME = 'all-mpnet-base-v2' #for EN

#2.Read
print(f"กำลังอ่านข้อมูลจากไฟล์: {FILE_PATH}...")
try:
    df = pd.read_parquet(FILE_PATH)
    df.dropna(subset=[TEXT_COLUMN], inplace=True)
    # !!! เพื่อความรวดเร็วในการทดลอง สามารถลดขนาดข้อมูลลงได้ !!!
    # df = df.head(1000) # <--- ลองเอาคอมเมนต์ออกเพื่อทดสอบกับข้อมูลแค่ 1000 ข่าว
    corpus = df[TEXT_COLUMN].tolist()
    print(f"อ่านข้อมูลสำเร็จ พบข่าวทั้งหมด {len(corpus)} ข่าว")
except FileNotFoundError:
    print(f"!!! ไม่พบไฟล์ '{FILE_PATH}'. กรุณาตรวจสอบชื่อไฟล์และตำแหน่งที่ถูกต้อง")
    exit()

#3.text to Vector like  Late Chunking ---
print(f"กำลังโหลดโมเดล '{MODEL_NAME}'...")
embedder = SentenceTransformer(MODEL_NAME)

print(f"กำลังแปลงข้อความ {len(corpus)} ข่าวให้เป็นเวกเตอร์ (ด้วยวิธีแบ่งประโยค)...")

# create list for keep text
final_article_embeddings = []

# for loop  seperated news text
for article in tqdm(corpus, desc="Processing Articles"):
    # 1. แบ่งข่าวออกเป็นประโยค
    sentences = nltk.sent_tokenize(article)

    # จัดการกรณีที่ข่าวไม่มีข้อความหรือแบ่งประโยคไม่ได้
    if not sentences:
        embedding_dim = embedder.get_sentence_embedding_dimension()
        final_article_embeddings.append(np.zeros(embedding_dim))
        continue

    # 2. สร้าง embedding ของทุกประโยคในข่าวนี้
    sentence_embeddings = embedder.encode(sentences, show_progress_bar=False) 

    # 3. รวม embedding ของทุกประโยคให้เป็น embedding เดียว (หาค่าเฉลี่ย)
    # Main point 
    article_embedding = np.mean(sentence_embeddings, axis=0)

    # 4. add embedding of news to  list 
    final_article_embeddings.append(article_embedding)

# change list of embeddings to numpy array 
corpus_embeddings = np.array(final_article_embeddings)

#4.cluster by  HDBSCAN
print("\nกำลังจัดกลุ่มข่าวด้วย HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=10,
                            metric='euclidean',
                            cluster_selection_method='eom')
clusters = clusterer.fit_predict(corpus_embeddings)

# 5.output 
df['cluster'] = clusters
OUTPUT_FILE = 'news_with_clusters_late_chunking.csv' 
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print("\n--- การจัดกลุ่มเสร็จสมบูรณ์! ---")
print(f"ผลลัพธ์ถูกบันทึกไว้ที่ไฟล์: {OUTPUT_FILE}")
num_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
num_outliers = np.sum(clusters == -1)
print(f"ค้นพบกลุ่มข่าวทั้งหมด: {num_clusters} กลุ่ม")
print(f"พบข่าวที่ไม่เข้าพวก (Outliers): {num_outliers} ข่าว")

#6.deminish dimension for UMAP 
print("\nกำลังลดมิติของเวกเตอร์ด้วย UMAP เพื่อสร้างกราฟ...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(corpus_embeddings)
df['x'] = embedding_2d[:, 0]
df['y'] = embedding_2d[:, 1]

#7. Scatter plot interactive
print("กำลังสร้างกราฟ Scatter Plot...")
df['cluster_str'] = df['cluster'].astype(str)
fig = px.scatter(
    df, x='x', y='y', color='cluster_str',
    hover_name=df.index,
    hover_data={'text': True, 'cluster': True, 'x': False, 'y': False},
    title='2D Visualization of News Clusters (Late Chunking Approach)',
    color_discrete_map={'-1': 'lightgrey'}
)
fig.update_traces(marker=dict(size=5, opacity=0.7))
fig.update_layout(legend_title_text='Cluster ID')

# 8. export file html
OUTPUT_HTML_FILE = 'interactive_cluster_plot_late_chunking.html' 
fig.write_html(OUTPUT_HTML_FILE)
print(f"\nกราฟ Cluster แบบ Interactive ถูกบันทึกแล้วที่ไฟล์: {OUTPUT_HTML_FILE}")