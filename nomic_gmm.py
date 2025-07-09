import pandas as pd
import requests
import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# 1. OLLMA
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
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"**คำเตือน: เกิดข้อผิดพลาดในการเชื่อมต่อ Ollama -> {e}**")
        return None

if __name__ == "__main__":
    print("--- เริ่มต้นกระบวนการ Clustering ข่าวด้วย Gaussian Mixture Model (GMM) ---")

   
    file_path = '/Users/wish.cha/Desktop/Week1_learning/NEWS_clusters2/cnbc_news.parquet'  
    try:
        df = pd.read_parquet(file_path)
        print(f"\n[Step 1/7] อ่านไฟล์ '{file_path}' สำเร็จ. จำนวนแถว: {len(df)}")
    except FileNotFoundError:
        print(f"**ข้อผิดพลาด: ไม่พบไฟล์ '{file_path}'**")
        exit()
    except Exception as e:
        print(f"**ข้อผิดพลาดในการอ่านไฟล์ .parquet: {e}**")
        exit()

    news_column_name = 'text' 
    if news_column_name not in df.columns:
        print(f"**ข้อผิดพลาด: ไม่พบคอลัมน์ '{news_column_name}'**")
        exit()
    news_articles = df[news_column_name].tolist()
    print(f"\n[Step 2/7] ดึงข้อความข่าวได้ {len(news_articles)} รายการ")

    embeddings = []
    processed_news_articles = []
    original_indices = []
    print(f"\n[Step 3/7] กำลังสร้าง Embedding...")
    for i, article in enumerate(news_articles):
        if pd.isna(article) or not isinstance(article, str) or not article.strip():
            continue
        clean_article = " ".join(article.split())
        embedding = get_embedding(clean_article, model_name="nomic-embed-text") 
        if embedding:
            embeddings.append(embedding)
            processed_news_articles.append(clean_article)
            original_indices.append(i)
        if (i + 1) % 100 == 0:
            print(f"  > สร้าง Embedding ไปแล้ว {i + 1}/{len(news_articles)} รายการ")

    if not embeddings:
        print("**ข้อผิดพลาด: ไม่สามารถสร้าง Embedding ได้เลย**")
        exit()
    X = np.array(embeddings)
    print(f"\n[Step 3/7] สร้าง Embedding สำเร็จ {len(embeddings)} รายการ")
    print(f"ขนาดของ Embedding (จำนวนข่าว x มิติ): {X.shape}")

    # Cluster by GMM
    print(f"\n[Step 4/7] กำลังทำ Clustering ข่าวด้วย GMM...")
    
   
    N_COMPONENTS = 10
    
    print(f"  > กำลังเทรน GMM โดยกำหนดจำนวนกลุ่ม (components) = {N_COMPONENTS}...")
    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=42, n_init=5)
    gmm.fit(X)
    
    # การทำนายกลุ่ม (Hard Clustering) เพื่อนำไปพล็อตและคำนวณ Score
    clusters_gmm = gmm.predict(X)

    df_clustered = pd.DataFrame({
        'original_index': original_indices,
        'article': processed_news_articles,
        'cluster': clusters_gmm
    })
    
    print(f"\n[Step 4/7] ผลลัพธ์ GMM Clustering:")
    print(df_clustered['cluster'].value_counts().sort_index())

    # --- Step 5: ประเมินคุณภาพ (เหมือนเดิม) ---
    print(f"\n[Step 5/7] กำลังประเมินคุณภาพของ Cluster ด้วย Silhouette Score...")
    try:
        silhouette_avg = silhouette_score(X, clusters_gmm)
        print(f"  > Silhouette Score: {silhouette_avg:.4f}")
    except Exception as e:
        print(f"  > ไม่สามารถคำนวณ Silhouette Score ได้: {e}")

    # --- [ใหม่] Step 6: แสดงผลลัพธ์แบบ Soft Clustering ---
    print("\n[Step 6/7] แสดงตัวอย่างผลลัพธ์แบบ Soft Clustering (ความน่าจะเป็น)")
    probabilities = gmm.predict_proba(X)
    prob_df = pd.DataFrame(probabilities, columns=[f'prob_cluster_{i}' for i in range(N_COMPONENTS)])
    
    # แสดง 5 ข่าวแรกพร้อมความน่าจะเป็นของแต่ละกลุ่ม
    print("ตัวอย่าง 5 แถวแรก พร้อมความน่าจะเป็นในการอยู่ในแต่ละกลุ่ม:")
    # รวม DataFrame หลักกับ DataFrame ความน่าจะเป็น
    df_full_results = pd.concat([df_clustered.reset_index(drop=True), prob_df], axis=1)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 50) # แสดงคอลัมน์ได้มากขึ้น
    print(df_full_results[['article', 'cluster'] + prob_df.columns.tolist()].head())

    # --- Step 7: แสดงผลด้วยภาพ t-SNE (เหมือนเดิม) ---
    print("\n[Step 7/7] กำลังลดมิติด้วย t-SNE เพื่อการแสดงผล...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(14, 12))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=df_clustered['cluster'],
        palette=sns.color_palette("viridis", n_colors=N_COMPONENTS),
        legend="full",
        alpha=0.8,
        s=60
    )
    plt.title(f'News Article Clusters (k={N_COMPONENTS}, GMM) - t-SNE', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    print("\n--- กระบวนการ Clustering สำเร็จ ---")
