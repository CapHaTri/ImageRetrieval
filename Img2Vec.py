import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import  Model
from RemoveBackground import RemoveBackground

from PIL import Image
import pickle
import numpy as np
class ImageVectorizer:
    def __init__(self):
        self.model = self.get_extract_model()
    # Tạo ra một mô hình trích xuất đặc trưng từ VGG16
    def get_extract_model(self): 
        vgg16_model = VGG16(weights="imagenet") #Tạo mô hình VGG16 từ keras và tải trọng số được huấn luyện trước trên tập dữ liệu Imagenet
        extract_model = Model(inputs=vgg16_model.inputs, outputs = vgg16_model.get_layer("fc1").output) #Tạo mô hình mới với đầu ra sẽ là đầu ra của tầng fc1
        return extract_model

    # Hàm tiền xử lý
    def image_preprocess(self, img):
        img = img.resize((224,224)) #resize
        img = img.convert("RGB")    # chuyển đổi hình ảnh sang chế độ màu RGB
        x = image.img_to_array(img) # chuyển đổi hình ảnh thành mảng numpy
        x = np.expand_dims(x, axis=0) # thêm chiều mới vào mảng
        x = preprocess_input(x) # tiền xử lý 
        return x
    # Chuyển đổi hình ảnh thành tensor
    def extract_vector(self, image_path):
        print("Xu ly : ", image_path)
        img = Image.open(image_path)
        img_rembg = RemoveBackground(img)      #Remove BackGround
        img_tensor = self.image_preprocess(img_rembg)
        vector = self.model.predict(img_tensor)[0]   # trích xuất đặc trưng
        vector = vector / np.linalg.norm(vector)    # chuẩn hóa vector : chia vector cho độ dài Euclid của nó
        return vector



def main():
    # Khởi tạo model

    model = ImageVectorizer()

    vectors = [] # Lưu trữ vector
    paths = [] # Lưu trữ path
    # Định nghĩa thư mục data
    list = ['dataset\dataset', 'dataset\dataset1\cane', 'dataset\dataset1\cavallo', 'dataset\dataset1\elefante', 'dataset\dataset1\gallina', 'dataset\dataset1\gatto', 'dataset\dataset1\mucca', 'dataset\dataset1\pecora']
    list1 = ['dataset/dataset2/animals/animals']
    for data_folder in list1:
        for folder in os.listdir(data_folder):
            for image_path in os.listdir(os.path.join(data_folder,folder)):
                image_path_full = os.path.join(data_folder,folder, image_path) # Nối full path
                image_vector = model.extract_vector(image_path_full)    # Trích đặc trưng
                
                vectors.append(image_vector)    #Thêm vector vào list
                paths.append(image_path_full)   #Thêm full_path vào list
    for data_folder in list:
    
        for image_path in os.listdir(data_folder):

            image_path_full = os.path.join(data_folder, image_path) # Nối full path

            image_vector = model.extract_vector(image_path_full)    # Trích đặc trưng

            vectors.append(image_vector)    #Thêm vector vào list
            paths.append(image_path_full)   #Thêm full_path vào list
    
    # Lưu vào file
    vector_file = "vectors.pkl"
    path_file = "paths.pkl"

    pickle.dump(vectors, open(vector_file, "wb"))
    pickle.dump(paths, open(path_file, "wb"))
if __name__ == "__main__":
    main()