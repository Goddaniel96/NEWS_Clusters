import pandas as pd
import requests
import json
import numpy as np
import hdbscan 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

#1.embedding
def get_embedding(text, model_name="nomic-embed-text"):
    """
    ส่งข้อความไปยัง Ollama Server เพื่อรับ Vector Embedding กลับมา
    """
    url = "http://localhost:11434/api/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": text
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=120)
        response.raise_for_status() # status HTTP error
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        # พิมพ์ข้อความแสดงข้อผิดพลาดแบบย่อๆ เพื่อไม่ให้เยอะเกินไป
        # print(f"Error getting embedding for text (first 50 chars): '{text[:50]}...': {e}")
        return None


if __name__ == "__main__":
    print("--- เริ่มต้นกระบวนการ Clustering ข่าวด้วย Ollama และ HDBSCAN ---")

    # --- 1. อ่านไฟล์ ---
    file_path = '/Users/wish.cha/Desktop/Week1_learning/NEWS_clusters2/cnbc_news.parquet'
    try:
        df = pd.read_parquet(file_path)
        print(f"\n[Step 1/6] อ่านไฟล์ '{file_path}' สำเร็จ. จำนวนแถว: {len(df)}")
        print("ตัวอย่าง 5 แถวแรกของ DataFrame:\n", df.head())
        print("\nชื่อคอลัมน์ทั้งหมดใน DataFrame:\n", df.columns.tolist())
    except FileNotFoundError:
        print(f"**ข้อผิดพลาด: ไม่พบไฟล์ '{file_path}' โปรดตรวจสอบพาธไฟล์ให้ถูกต้อง**")
        exit()
    except Exception as e:
        print(f"**ข้อผิดพลาดในการอ่านไฟล์ .parquet: {e}**")
        exit()

    # --- 2. ดึงข้อความข่าว ---
    news_column_name = 'text'

    if news_column_name not in df.columns:
        print(f"**ข้อผิดพลาด: ไม่พบคอลัมน์ '{news_column_name}' ในไฟล์ของคุณ**")
        print(f"โปรดเปลี่ยน '{news_column_name}' ในโค้ดให้เป็นชื่อคอลัมน์ที่ถูกต้องตามที่แสดงใน df.columns.tolist()")
        exit()

    news_articles = df[news_column_name].tolist()
    print(f"\n[Step 2/6] ดึงข้อความข่าวจากคอลัมน์ '{news_column_name}' ได้ {len(news_articles)} รายการ")

    # --- 3. สร้าง Embedding ---
    embeddings = []
    processed_news_articles = [] # เก็บเฉพาะข่าวที่สร้าง embedding ได้สำเร็จ
    original_indices = [] # เก็บ index เดิมของข่าวที่ถูกประมวลผล (จาก df)

    print(f"\n[Step 3/6] กำลังสร้าง Embedding สำหรับ {len(news_articles)} ข่าว (อาจใช้เวลานานมาก ขึ้นอยู่กับสเปกเครื่องและโมเดล)...")
    for i, article in enumerate(news_articles):
        # ข้ามข้อความที่เป็นค่าว่าง (NaN), ไม่ใช่ string, หรือเป็น string ว่างเปล่า
        if pd.isna(article) or not isinstance(article, str) or not article.strip():
            continue

        embedding = get_embedding(article, model_name="nomic-embed-text")
        if embedding:
            embeddings.append(embedding)
            processed_news_articles.append(article)
            original_indices.append(i)
        else:
            print(f"**คำเตือน: ไม่สามารถสร้าง Embedding สำหรับข่าวที่ original index {i} ได้**")

        if (i + 1) % 100 == 0:
            print(f"  > สร้าง Embedding ไปแล้ว {i + 1}/{len(news_articles)} รายการ")

    if not embeddings:
        print("**ข้อผิดพลาด: ไม่สามารถสร้าง Embedding ได้เลย โปรดตรวจสอบว่า Ollama server ทำงาน, โมเดลถูก pull และพารามิเตอร์ถูกต้อง**")
        exit()

    X = np.array(embeddings)
    print(f"\n[Step 3/6] สร้าง Embedding สำเร็จ {len(embeddings)} รายการ (จาก {len(news_articles)} รายการที่พยายามประมวลผล)")
    print(f"ขนาดของ Embedding (จำนวนข่าว x มิติ): {X.shape}")

    # --- 4. Clustering by HDBSCAN ---
    print(f"\n[Step 4/6] กำลังทำ Clustering ข่าวด้วย HDBSCAN...")

    # --- ส่วนที่เปลี่ยนแปลง: ลบการหาค่า eps ของ DBSCAN และใช้ HDBSCAN แทน ---
    # HDBSCAN ไม่ต้องการค่า eps แต่จะใช้พารามิเตอร์อื่นแทน
    # min_cluster_size: จำนวนข่าวขั้นต่ำที่ต้องมีเพื่อสร้างเป็น 1 กลุ่ม (พารามิเตอร์ที่สำคัญที่สุด)
    # min_samples: ใช้ควบคุมว่า clustering จะเป็นแบบ conservative แค่ไหน (ค่ามากจะทำให้มี noise มากขึ้น) ถ้าไม่กำหนดจะใช้ค่าเดียวกับ min_cluster_size
    hdbscan_min_cluster_size = 5
    hdbscan_min_samples = 5 # สามารถลองปรับค่านี้ได้

    print(f"  > รัน HDBSCAN Clustering ด้วย min_cluster_size={hdbscan_min_cluster_size}, min_samples={hdbscan_min_samples}...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric='euclidean',
        allow_single_cluster=True, # อนุญาตให้ผลลัพธ์มีแค่กลุ่มเดียวได้
        gen_min_span_tree=True # จำเป็นสำหรับการวิเคราะห์เพิ่มเติมบางอย่าง
    )
    clusters_hdbscan = clusterer.fit_predict(X)

    df_clustered = pd.DataFrame({
        'original_index': original_indices,
        'article': processed_news_articles,
        'cluster': clusters_hdbscan # ใช้ผลลัพธ์จาก HDBSCAN
    })

    print(f"\n[Step 4/6] ผลลัพธ์ HDBSCAN Clustering:")
    print(df_clustered['cluster'].value_counts().sort_index())
    print("หมายเหตุ: Cluster -1 คือ Noise (ข่าวที่ไม่ได้จัดอยู่ในกลุ่มใดๆ)")


    # --- 5. Analysis & sample NEWS---
    print("\n[Step 5/6] แสดงตัวอย่างข่าวในแต่ละกลุ่ม:")
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        print(f"\n### กลุ่มที่ {cluster_id} ### (มี {len(df_clustered[df_clustered['cluster'] == cluster_id])} ข่าว)")
        # random 3 NEWS from clusters
        sample_articles = df_clustered[df_clustered['cluster'] == cluster_id]['article'].sample(
            min(3, len(df_clustered[df_clustered['cluster'] == cluster_id])), random_state=42
        )
        for i, article_text in enumerate(sample_articles):
            print(f"  {i+1}. {article_text[:200]}...") #Text 200 word from NEWS

    # --- 6.Visualisation by t-SNE ---
    print("\n[Step 6/6] กำลังลดมิติด้วย t-SNE เพื่อการแสดงผล (อาจใช้เวลาสักครู่)...")
    #  PCA and t-SNE for accelerate
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate='auto', init='random')
    X_tsne = tsne.fit_transform(X_pca)

    # Plot Output
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=df_clustered['cluster'], # ใช้ผลลัพธ์จาก HDBSCAN
        palette=sns.color_palette("deep", len(np.unique(df_clustered['cluster']))), # ปรับ palette
        legend="full",
        alpha=0.7,
        s=50
    )
    # visualize
    plt.title('News Article Clusters (HDBSCAN - t-SNE visualization)', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    print("\n--- The Process : Clustering Successfully ---")
