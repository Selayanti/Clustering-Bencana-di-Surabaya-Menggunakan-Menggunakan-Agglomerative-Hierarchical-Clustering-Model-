import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Clustering Kejadian Bencana",
    page_icon=Image.open("BPBD.png"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for background, font, and button styling
st.markdown("""
    <style>
        body {
            background-color: #fdf4e3;
        }
        .title {
            font-family: 'Trebuchet MS', sans-serif;
            font-size: 36px;
            color: #d35400;
            text-align: center;
        }
        .subtitle {
            font-family: 'Trebuchet MS', sans-serif;
            font-size: 18px;
            color: #a04e00;
            text-align: center;
        }
        .sidebar .sidebar-content {
            background-color: #f9e5d3;
        }
        .stButton>button {
            background-color: #e67e22;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #cf6f19;
        }
        .result-table th {
            font-weight: bold;
            text-align: center;
            background-color: #e67e22;
            color: white;
        }
        .result-table td {
            text-align: center;
            color: #333333;
        }
    </style>
    """, unsafe_allow_html=True)

# Add logos on the left side
col1, col2, col3, col4 = st.columns([1, 1, 1, 6])
with col1:
    st.image("UPN.png", width=75)
with col2:
    st.image("BPBD.png", width=75)
with col3:
    st.image("KM.png", width=75)
with col4:
    pass

# Display title with appealing font
st.markdown("<h1 style='text-align: center; font-family: 'Trebuchet MS', sans-serif; font-size: 36px; color: #333333;'>Clustering Kejadian Bencana di Surabaya Selama Musim Hujan</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-family: 'Trebuchet MS', sans-serif; font-size: 15px; color: #666666;'>Jenis Kejadian Darurat Medis, Genangan, Kebakaran, Kecelakaan, Pohon Tumbang, Rumah Roboh.<br> Analisis clustering ini digunakan untuk mengidentifikasi pola kejadian bencana di wilayah Kelurahan Surabaya selama musim hujan,  fokus pada periode November hingga Maret. Hasil clustering memberikan wawasan kepada BPBD Surabaya mengenai kelurahan yang memerlukan prioritas penanganan dan membantu dalam alokasi sumber daya yang lebih efektif.</p>", unsafe_allow_html=True)

# Load data from Excel file
try:
    df_combined = pd.read_excel('data untuk clustering.xlsx', sheet_name='Data Clustering')
except FileNotFoundError:
    st.error("File 'data untuk clustering.xlsx' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    st.stop()  # Stop execution if file is not found

# Select number of clusters
num_clusters = st.slider("Pilih Jumlah Cluster :", min_value=2, max_value=5, value=4)

# Button to start predictions
if st.button("PREDICT CLUSTER"):
    # Step 1: Prepare features by dropping 'KELURAHAN'
    features = df_combined.drop(columns=['KELURAHAN'], errors='ignore')

    # Step 2: Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    norm_all = pd.DataFrame(scaled_data, columns=features.columns)

    # Step 3: Perform hierarchical clustering using average linkage
    Z_average = linkage(norm_all, method='average')

    # Plot dendrogram with selected number of clusters
    st.subheader(f"📊 Dendrogram dengan {num_clusters} Cluster")
    plt.figure(figsize=(12, 6))
    dendrogram(
        Z_average,
        color_threshold=Z_average[-num_clusters, 2],  # Threshold to get selected clusters
        above_threshold_color='k'
    )
    plt.axhline(y=Z_average[-num_clusters, 2], color='r', linestyle='--')
    plt.xlabel('Kelurahan')
    plt.ylabel('Distance')
    plt.title(f'Dendrogram dengan Average Linkage untuk {num_clusters} Cluster')
    st.pyplot(plt)

    # Step 4: Use AgglomerativeClustering to get cluster labels
    model_average = AgglomerativeClustering(n_clusters=num_clusters, linkage='average')
    cluster_labels = model_average.fit_predict(norm_all)

    # Add cluster column to the original dataframe
    df_combined['Cluster'] = cluster_labels

    # Display results
    st.subheader("🌐 Hasil Cluster")
    kelurahan_per_cluster = df_combined.groupby('Cluster')['KELURAHAN'].agg(lambda x: ', '.join(x)).reset_index()
    kelurahan_per_cluster['Jumlah Anggota'] = df_combined.groupby('Cluster')['KELURAHAN'].count().values  # Hitung jumlah anggota
    kelurahan_per_cluster.columns = ['CLUSTER', 'KELURAHAN', 'JUMLAH']  
    kelurahan_per_cluster_html = kelurahan_per_cluster.to_html(index=False)
    kelurahan_per_cluster_html = kelurahan_per_cluster_html.replace('<th>', '<th style="font-weight: bold; text-align: center;">')
    kelurahan_per_cluster_html = kelurahan_per_cluster_html.replace('<td>', '<td style="text-align: center;">')
    st.markdown(kelurahan_per_cluster_html, unsafe_allow_html=True)

    # Tambahkan jarak antara hasil cluster dan rata-rata variabel
    st.write("")  # Menambahkan jarak kosong
    st.write("")  # Menambahkan jarak kosong

    # Step 5: Calculate average variables per cluster
    df_norm = pd.DataFrame(scaled_data, columns=features.columns)  # Mengambil kolom numerik yang dinormalisasi
    df_norm['Cluster'] = cluster_labels  # Menambahkan kolom cluster

    # Remove 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df_norm.columns:
        df_norm = df_norm.drop(columns=['Unnamed: 0'])

    # Menghitung rata-rata setiap variabel berdasarkan Cluster hanya untuk kolom numerik
    cluster_summary = df_norm.groupby('Cluster').mean()  # Hitung rata-rata setiap kolom berdasarkan cluster

    # Menghitung rata-rata total untuk setiap cluster dari data yang dinormalisasi
    cluster_summary['Rata-Rata Total'] = cluster_summary.mean(axis=1)  # Menghitung rata-rata dari semua variabel

    # Format untuk membuat jenis kejadian bold
    if 'JENIS KEJADIAN' in df_combined.columns:
        jenis_keadaan = df_combined.groupby('Cluster')['JENIS KEJADIAN'].first()  # Ambil jenis kejadian untuk setiap cluster
        cluster_summary['Jenis Kejadian'] = jenis_keadaan.apply(lambda x: f"<b>{x}</b>")  # Format bold

    st.subheader("🔎 Rata-Rata Variabel per Cluster")
    st.write(cluster_summary.to_html(escape=False), unsafe_allow_html=True)

    # Menambahkan jarak
    st.write("<br>", unsafe_allow_html=True) 

    # Step 6: Generate Interpretation for Each Cluster
    st.subheader("📋 Interpretasi dan Starategi Penanganan di Setiap Cluster")

    # Iterate through each cluster and display interpretation
    kejadian_vars = ['Darurat Medis', 'Genangan', 'Kebakaran', 'Kecelakaan', 'Pohon Tumbang', 'Rumah Roboh']
    korban_vars = ['KORBAN TERDAMPAK', 'KORBAN LUKA', 'KORBAN MENINGGAL']

    # Interpretasi tiap cluster dalam Bahasa Indonesia dengan detail korban dan kejadian terendah
    for cluster in cluster_summary.index:
        st.write(f"**CLUSTER {cluster}**:")
        
        # Cari dua kejadian tertinggi dan terendah di setiap cluster
        top_kejadian = cluster_summary.loc[cluster, kejadian_vars].nlargest(2)
        low_kejadian = cluster_summary.loc[cluster, kejadian_vars].nsmallest(2)
        
        # Cari variabel korban dengan nilai tertinggi di setiap cluster
        top_korban = cluster_summary.loc[cluster, korban_vars].idxmax()
        top_korban_value = cluster_summary.loc[cluster, top_korban]

        # Interpretasi dalam Bahasa Indonesia
        st.write(f"- **Dua Kejadian Utama**: {top_kejadian.index[0]} dengan rata-rata {top_kejadian.iloc[0]:.2f} kejadian dan "
                f"{top_kejadian.index[1]} dengan rata-rata {top_kejadian.iloc[1]:.2f} kejadian.")
        
        st.write(f"- **Dua Kejadian Terendah**: {low_kejadian.index[0]} dengan rata-rata {low_kejadian.iloc[0]:.2f} kejadian dan "
                f"{low_kejadian.index[1]} dengan rata-rata {low_kejadian.iloc[1]:.2f} kejadian.")
        
        # Menampilkan variabel korban dengan angka tertinggi
        st.write(f"- **Korban Tertinggi** : {top_korban} dengan rata-rata {top_korban_value:.2f}")
        
        # Insight berdasarkan variabel korban tertinggi
        if top_korban == 'KORBAN MENINGGAL':
            st.write("  **Insight Korban**: Tingginya rata-rata korban meninggal di cluster ini mungkin mengindikasikan bahwa "
                    "kejadian di wilayah ini cenderung fatal atau memerlukan tindakan respons darurat yang lebih cepat.")
        elif top_korban == 'KORBAN LUKA':
            st.write("  **Insight Korban**: Rata-rata korban luka yang tinggi menunjukkan bahwa kejadian di wilayah ini sering "
                    "mengakibatkan cedera, sehingga BPBD dapat mempertimbangkan untuk meningkatkan pelatihan pertolongan pertama "
                    "dan fasilitas medis lokal.")
        elif top_korban == 'KORBAN TERDAMPAK':
            st.write(" **Insight Korban**: Tingginya jumlah korban terdampak menandakan bahwa kejadian di cluster ini memiliki "
                    "dampak luas pada masyarakat. BPBD dapat berfokus pada upaya mitigasi yang dapat menekan dampak luas ini, "
                    "misalnya dengan memberikan sosialisasi atau memperkuat infrastruktur.")
        
        # Insight untuk dua kejadian terendah
        st.write(f" **Insight Kejadian Terendah**: Rendahnya frekuensi {low_kejadian.index[0]} dan {low_kejadian.index[1]} di cluster ini mungkin "
                f"menunjukkan bahwa jenis kejadian ini jarang terjadi di wilayah tersebut, atau bahwa wilayah ini mungkin "
                f"memiliki kondisi yang lebih aman terhadap jenis kejadian ini.")
        
        # Rekomendasi strategis untuk BPBD berdasarkan analisis kejadian, korban, dan variabel terendah
        st.write(f"📌 **Rekomendasi Strategi**:")
        st.write(f"  Mengingat kejadian utama seperti {top_kejadian.index[0]} dan {top_kejadian.index[1]}, disertai tingginya angka "
                f"{top_korban.lower()}, BPBD sebaiknya fokus pada langkah pencegahan khusus dan kesiapsiagaan dalam menghadapi "
                f"jenis kejadian ini. Sementara itu, pengawasan terhadap kejadian {low_kejadian.index[0]} dan {low_kejadian.index[1]} "
                f"dapat dilakukan secara berkala, mengingat tingkat kejadiannya yang rendah.")
        
        # Garis pemisah antar cluster
        st.write("<hr style='border:1px dashed gray;'>", unsafe_allow_html=True) 
        
