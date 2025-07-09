import pandas as pd
import requests
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score # [ใหม่] เพิ่ม Library สำหรับวัดคุณภาพ Cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# 1. ฟังก์ชันสำหรับเรียกใช้ Ollama Embedding
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
        response.raise_for_status() # ตรวจสอบ status HTTP error
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"**คำเตือน: เกิดข้อผิดพลาดในการเชื่อมต่อ Ollama -> {e}**")
        return None

if __name__ == "__main__":
    print("--- เริ่มต้นกระบวนการ Clustering ข่าวด้วย Ollama (เวอร์ชันปรับปรุง) ---")

    # --- Step 1: อ่านข้อมูล ---
    file_path = '/Users/wish.cha/Desktop/Week1_learning/NEWS_clusters2/cnbc_news.parquet'  
    try:
        df = pd.read_parquet(file_path)
        print(f"\n[Step 1/7] อ่านไฟล์ '{file_path}' สำเร็จ. จำนวนแถว: {len(df)}")
    except FileNotFoundError:
        print(f"**ข้อผิดพลาด: ไม่พบไฟล์ '{file_path}' โปรดตรวจสอบพาธไฟล์ให้ถูกต้อง**")
        exit()
    except Exception as e:
        print(f"**ข้อผิดพลาดในการอ่านไฟล์ .parquet: {e}**")
        exit()

    # --- Step 2:Use colum Text ---
    news_column_name = 'text' 
    if news_column_name not in df.columns:
        print(f"**ข้อผิดพลาด: ไม่พบคอลัมน์ '{news_column_name}' ในไฟล์ของคุณ**")
        exit()

    news_articles = df[news_column_name].tolist()
    print(f"\n[Step 2/7] ดึงข้อความข่าวจากคอลัมน์ '{news_column_name}' ได้ {len(news_articles)} รายการ")

    # --- Step 3:Embedding ---
    embeddings = []
    processed_news_articles = []
    original_indices = []

    print(f"\n[Step 3/7] กำลังสร้าง Embedding สำหรับ {len(news_articles)} ข่าว...")
    for i, article in enumerate(news_articles):
        # [ปรับปรุง] เพิ่มการตรวจสอบและทำความสะอาดข้อมูลเบื้องต้น
        if pd.isna(article) or not isinstance(article, str) or not article.strip():
            continue

        clean_article = " ".join(article.split()) # ลบช่องว่าง/ขึ้นบรรทัดใหม่ที่เกินมา
        embedding = get_embedding(clean_article, model_name="nomic-embed-text") 
        
        if embedding:
            embeddings.append(embedding)
            processed_news_articles.append(clean_article)
            original_indices.append(i)
        else:
            print(f"**คำเตือน: ไม่สามารถสร้าง Embedding สำหรับข่าวที่ original index {i} ได้**")

        if (i + 1) % 100 == 0:
            print(f"  > สร้าง Embedding ไปแล้ว {i + 1}/{len(news_articles)} รายการ")

    if not embeddings:
        print("**ข้อผิดพลาด: ไม่สามารถสร้าง Embedding ได้เลย โปรดตรวจสอบว่า Ollama server ทำงาน, โมเดลถูก pull และพารามิเตอร์ถูกต้อง**")
        exit()

    X = np.array(embeddings)
    print(f"\n[Step 3/7] สร้าง Embedding สำเร็จ {len(embeddings)} รายการ")
    print(f"ขนาดของ Embedding (จำนวนข่าว x มิติ): {X.shape}")

    # --- Step 4: ทำ Clustering ---
    print(f"\n[Step 4/7] กำลังทำ Clustering ข่าว...")
    
    # [ปรับปรุง] เปลี่ยนมาใช้ n_clusters เพื่อควบคุมจำนวนกลุ่มโดยตรง
    # ลองเปลี่ยนค่านี้เพื่อดูผลลัพธ์ที่แตกต่าง (เช่น 8, 10, 12, 15)
    N_CLUSTERS = 40
    
    print(f"  > กำลังทำ Agglomerative Clustering โดยกำหนดจำนวนกลุ่ม = {N_CLUSTERS}...")
    agg_clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS)
    clusters_agg = agg_clustering.fit_predict(X)

    df_clustered = pd.DataFrame({
        'original_index': original_indices,
        'article': processed_news_articles,
        'cluster': clusters_agg
    })
    
    print(f"\n[Step 4/7] ผลลัพธ์ Agglomerative Clustering:")
    print(df_clustered['cluster'].value_counts().sort_index())

    # --- [ใหม่] Step 5: ประเมินคุณภาพของ Cluster ---
    print(f"\n[Step 5/7] กำลังประเมินคุณภาพของ Cluster ด้วย Silhouette Score...")
    try:
        silhouette_avg = silhouette_score(X, clusters_agg)
        print(f"  > Silhouette Score: {silhouette_avg:.4f}")
        print("  > (ค่าเข้าใกล้ 1 = ดี, ค่าเข้าใกล้ 0 = กลุ่มซ้อนทับ, ค่าติดลบ = แย่)")
    except Exception as e:
        print(f"  > ไม่สามารถคำนวณ Silhouette Score ได้: {e}")


    # --- Step 6: Analysis NEWS ---
    print("\n[Step 6/7] แสดงตัวอย่างข่าวในแต่ละกลุ่ม:")
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_size = len(df_clustered[df_clustered['cluster'] == cluster_id])
        print(f"\n### กลุ่มที่ {cluster_id} ### (มี {cluster_size} ข่าว)")
        sample_articles = df_clustered[df_clustered['cluster'] == cluster_id]['article'].sample(
            min(3, cluster_size), random_state=42
        )
        for i, article_text in enumerate(sample_articles):
            print(f"  {i+1}. {article_text[:200]}...")

    # --- Step 7: visualize t-SNE ---
    print("\n[Step 7/7] กำลังลดมิติด้วย t-SNE เพื่อการแสดงผล...")
    # การทำ PCA ก่อนช่วยให้ t-SNE ทำงานได้เร็วขึ้นและมีประสิทธิภาพขึ้นในข้อมูลมิติสูง
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(14, 12))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=df_clustered['cluster'],
        # [ปรับปรุง] ใช้ palette ที่เหมาะกับจำนวนกลุ่มที่ไม่แน่นอนมากขึ้น
        palette=sns.color_palette("viridis", n_colors=N_CLUSTERS),
        legend="full",
        alpha=0.8,
        s=60
    )
    plt.title(f'News Article Clusters (k={N_CLUSTERS}, Agglomerative) - t-SNE', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout() 
    plt.show()

    print("\n--- กระบวนการ Clustering สำเร็จ ---")
