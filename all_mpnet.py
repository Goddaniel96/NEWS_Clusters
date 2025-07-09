import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np
import umap
import plotly.express as px

# --- 1. setting ---

FILE_PATH = 'cnbc_news.parquet' 
TEXT_COLUMN = 'text'    
MODEL_NAME = 'all-mpnet-base-v2'    

# --- 2. read file ---
print(f"กำลังอ่านข้อมูลจากไฟล์: {FILE_PATH}...")
try:
    df = pd.read_parquet(FILE_PATH)
    # ลบแถวที่ไม่มีข้อมูลในคอลัมน์ข่าว
    df.dropna(subset=[TEXT_COLUMN], inplace=True)
    # เก็บข้อความข่าวไว้ในลิสต์
    corpus = df[TEXT_COLUMN].tolist()
    print(f"อ่านข้อมูลสำเร็จ พบข่าวทั้งหมด {len(corpus)} ข่าว")
except FileNotFoundError:
    print(f"!!! ไม่พบไฟล์ '{FILE_PATH}'. กรุณาตรวจสอบชื่อไฟล์และตำแหน่งที่ถูกต้อง")
    exit()


# --- 3. text to Vector (Embedding) ---
#long time
print(f"กำลังโหลดโมเดล '{MODEL_NAME}'...")
embedder = SentenceTransformer(MODEL_NAME)

print(f"กำลังแปลงข้อความ {len(corpus)} ข่าวให้เป็นเวกเตอร์... (ขั้นตอนนี้อาจใช้เวลาหลายนาที)")
corpus_embeddings = embedder.encode(corpus, show_progress_bar=True)


# --- 4. cluster by HDBSCAN ---
print("กำลังจัดกลุ่มข่าวด้วย HDBSCAN...")
# min_cluster_size คือ ขนาดของกลุ่มที่เล็กที่สุดที่จะถูกพิจารณาว่าเป็นกลุ่มจริงๆ
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, 
                            metric='euclidean',
                            cluster_selection_method='eom')

clusters = clusterer.fit_predict(corpus_embeddings)


# --- 5. record output  ---
# เพิ่มคอลัมน์ 'cluster' เข้าไปใน DataFrame เดิม
df['cluster'] = clusters

# export csv
OUTPUT_FILE = 'news_with_clusters.csv'
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print("\n--- การจัดกลุ่มเสร็จสมบูรณ์! ---")
print(f"ผลลัพธ์ถูกบันทึกไว้ที่ไฟล์: {OUTPUT_FILE}")

# output
num_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
num_outliers = np.sum(clusters == -1)
print(f"ค้นพบกลุ่มข่าวทั้งหมด: {num_clusters} กลุ่ม")
print(f"พบข่าวที่ไม่เข้าพวก (Outliers): {num_outliers} ข่าว (ถูกกำกับด้วย cluster = -1)")

# --- 6. down dimension  for use  UMAP ---
# ใช้ตัวแปร corpus_embeddings และ df ที่คำนวณไว้แล้วจากข้างบน
print("\nกำลังลดมิติของเวกเตอร์ด้วย UMAP เพื่อสร้างกราฟ...")

reducer = umap.UMAP(n_neighbors=15, 
                    min_dist=0.1,   
                    metric='cosine',
                    random_state=42) 

embedding_2d = reducer.fit_transform(corpus_embeddings)

# เพิ่มข้อมูล 2 มิติเข้าไปใน DataFrame
df['x'] = embedding_2d[:, 0]
df['y'] = embedding_2d[:, 1]


# --- 7. สร้างกราฟ Scatter Plot แบบ Interactive ---
print("กำลังสร้างกราฟ Scatter Plot...")

df['cluster_str'] = df['cluster'].astype(str)

fig = px.scatter(
    df,
    x='x',
    y='y',
    color='cluster_str',
    hover_name=df.index,
    hover_data={'text': True, 'cluster': True, 'x': False, 'y': False},
    title='2D Visualization of News Clusters (UMAP + HDBSCAN)',
    color_discrete_map={'-1': 'lightgrey'}
)

fig.update_traces(marker=dict(size=5, opacity=0.7))
fig.update_layout(legend_title_text='Cluster ID')


# --- 8. export  HTML ---
OUTPUT_HTML_FILE = 'interactive_cluster_plot.html'
fig.write_html(OUTPUT_HTML_FILE)

print(f"\nกราฟ Cluster แบบ Interactive ถูกบันทึกแล้วที่ไฟล์: {OUTPUT_HTML_FILE}")