from Img2Vec import ImageVectorizer
import pickle
from sklearn.metrics import f1_score
import numpy as np
from PIL import Image
import streamlit as st
st.title(":blue[IMAGE RETRIEVAL]")    
st.subheader(":green[Lưu ý khi sử dụng: ]")
st.write("**Đưa vào 1 hình ảnh, hệ thống sẽ đưa ra danh sách các hình ảnh tương tự**")
st.write("**Hệ thống chỉ hỗ trợ đề xuất ảnh với dữ liệu thuộc Animal(...)**" )
uploaded_file = st.file_uploader("Chọn hình ảnh", type=['png', 'jpg', 'jpeg', 'jfif'])
y_true = np.ones(16, dtype=int)
y_pred = []
if uploaded_file is not None:
    # Đọc và hiển thị hình ảnh tải lên
    image = Image.open(uploaded_file)
   
    st.image(image, caption='Hình ảnh đã tải lên', use_column_width=True)
    # Định nghĩa ảnh cần tìm kiếm

    # Khởi tạo model
    model = ImageVectorizer()

    # Trích đặc trưng từ ảnh cần tìm kiếm
    search_vector = model.extract_vector(uploaded_file)
    
    # Load data ra biến
    vectors = pickle.load(open("vectors.pkl","rb"))
    paths = pickle.load(open("paths.pkl","rb"))

    # Tính khoảng cách từ vector tìm kiếm đến tất cả các vector
    distance = np.linalg.norm(vectors - search_vector, axis=1)

    # Sắp xếp và lấy ra K vector có khoảng cách ngắn nhất
    K = 16
    ids = np.argsort(distance)[:K]
    threshold = 0.8
    # Lưu 
    nearest_image = [paths[id] for id in ids]
    distance_image = [distance[id] for id in ids]
    folder = 'C:/Users/MSI/OneDrive/CV/ImageRetrieval/'
    st.subheader(":blue[KẾT QUẢ TÌM KIẾM]") 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        for item in [0,1,2,3]:
            st.image(Image.open(folder + nearest_image[item]))
            if distance_image[item] < threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
    with col2:
        for item in [4,5,6,7]:
            st.image(Image.open(folder + nearest_image[item]))
            if distance_image[item] < threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
    with col3:
        for item in [8,9,10,11]:
            st.image(Image.open(folder + nearest_image[item]))
            if distance_image[item] < threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
    with col4:
        for item in [12,13,14,15]:
            st.image(Image.open(folder + nearest_image[item]))  
            if distance_image[item] < threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)    
    f1 = f1_score(y_true, y_pred)
    st.write("Độ chính xác: ", f1)
