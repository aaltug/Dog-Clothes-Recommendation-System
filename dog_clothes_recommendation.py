import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import numpy as np
import os
import matplotlib.pyplot as plt

# Önceden eğitilmiş VGG16 modelini yükle
base_model = VGG16(weights='imagenet', include_top=False)     # VGG16'nın ImageNet veri kümesi üzerinde eğitilmiş ağırlıklarını kullanacağını belirtir.
                                                              # Modelin üzerindeki tam bağlantılı katmanları dahil etmemeyi belirtir. Sadece özellik çıkarma.
model = Model(inputs=base_model.input, outputs=base_model.output)     # VGG16 modelinin giriş ve çıkışını yeni modelimize ('model') eşitliyoruz. 

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))     # 224'e 224 ölçeklendirme
    img_array = image.img_to_array(img)     # Görüntü, bir NumPy dizisine dönüştürülür.
    img_array = np.expand_dims(img_array, axis=0)      #Görüntü dizisi, bir boyut arttırılarak (axis=0), modele tek bir örnek olarak beslenebilecek şekle getirilir.
    img_array = preprocess_input(img_array)      # Modelin eğitildiği şekilde normalleştirme / veri ön işleme
    return img_array

def get_vgg16_features(img_array):
    features = model.predict(img_array)     # Görüntü model üzerinden geçirilir ve özellik vektörleri elde edilir.
    return features.flatten()      # Özellik vektörü tek boyutlu hale getirilir.

def calculate_color_histogram(image_path):
    img = cv2.imread(image_path)     # Görüntüyü bir NumPy dizisine dönüştürür.

    # Renk histogramını hesapla
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])     # 3 kanallı (RGB) olduğu  ve bu kanalların değer aralıkları belirtilir
                                            # Maske (belirli bir renk aralığı veya belirli bir bölgeyi işleme) olmadığı (none) 
    hist = cv2.normalize(hist, hist).flatten()      # Normalize edilir (0 ile 1 arasında) ve tek boyutlu hale getirilir
    return hist

def calculate_histogram_similarity(hist1, hist2):
    # Histogram benzerliğini hesapla
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    return similarity

def find_most_similar_clothing(input_clothing_path, clothing_folder):
    input_clothing_img = preprocess_image(input_clothing_path)     # Kullanıcının girdiği kıyafet görüntüsü yüklenir ve yukarıda tanımladığımı preprocess fonksiyonu ile işlenir.
    input_clothing_vgg16_features = get_vgg16_features(input_clothing_img)     # get_vgg16_features fonksiyonu kullanılarak özellik vektörleri elde edilir.
    input_clothing_hist = calculate_color_histogram(input_clothing_path)     # calculate_color_histogram fonksiyonu kullanılarak renk histogramı hesaplanır.

    most_similar_clothing = None     # En benzer kıyafeti tutan 
    max_similarity = -1      # En düşük benzerlik başlangıç değeri

    for clothing_img_path in os.listdir(clothing_folder):     # Hedef dosyadaki kıyafetler alınıp listelenir ve içerisinde gezinmek için for döngüsüne sokulur.
        if clothing_img_path.endswith('.jpg'):     # Sadece jpg uzantısı işleme alınır.
            clothing_img_path = os.path.join(clothing_folder, clothing_img_path)     # Hedef fotoğrafın tam yolu alınır.
            clothing_img = preprocess_image(clothing_img_path)     # Hedef fotoğraf preprocess fonkiyonu ile işlenir.
            clothing_vgg16_features = get_vgg16_features(clothing_img)     # Hedef fotoğrafın özellik vektörü çıkarılır.
            clothing_hist = calculate_color_histogram(clothing_img_path)enk     # Hedef fotoğrafın histogramı hesaplanır.

            vgg16_similarity = np.dot(input_clothing_vgg16_features, clothing_vgg16_features) / (
                    np.linalg.norm(input_clothing_vgg16_features) * np.linalg.norm(clothing_vgg16_features))
          # 2 vektör arasında nokta çarpımı, vektörlerin normlarının çarpımına bölünerek kosinüs benzerlik değeri bulunur ve [0,1] değerleri aralığında ölçeklendirilir. 
            
            hist_similarity = calculate_histogram_similarity(input_clothing_hist, clothing_hist)     # Renk benzerliğine bakar.

            overall_similarity = (vgg16_similarity + hist_similarity) / 2     # Özellik benzerliği ve renk benzerliğinin ortalaması alınır.

            if overall_similarity > max_similarity:       # En yüksek benzerliği bulmak için basit bir for döngüsü.
                max_similarity = overall_similarity
                most_similar_clothing = clothing_img_path

    return most_similar_clothing, max_similarity      # En benzer fotoğrafı ve benzerlik değerini döndürür

# Verilen bir örnek kıyafet ve benzerlik ölçmek istediğiniz kıyafet dizini
input_clothing_path = '/content/10732.jpg'
clothing_folder = '/content/ss'

# En benzer kıyafeti bul
most_similar_clothing, similarity_score = find_most_similar_clothing(input_clothing_path, clothing_folder)     # Yukarıda yazdığımız en benzer kıyafeti bulma fonksiiyonunu çağırır

# Kullanıcının girdiği kıyafeti göster
input_clothing_img = cv2.imread(input_clothing_path)     # cv2 ile fotoğrafı okur
input_clothing_rgb = cv2.cvtColor(input_clothing_img, cv2.COLOR_BGR2RGB)      #BGR formatından RGB formatına döndürür.

# En benzer kıyafeti göster
most_similar_img = cv2.imread(most_similar_clothing)     # cv2 ile fotoğrafı okur
most_similar_rgb = cv2.cvtColor(most_similar_img, cv2.COLOR_BGR2RGB)     #BGR formatından RGB formatına döndürür.

# Matplotlib ile görselleştirme
plt.subplot(1, 2, 1)     # 1 satır 2 sütünluk alan oluşturur ve ilk sutuna görseli yerleştirmesi gerektiğini söyler
plt.imshow(input_clothing_rgb)     # İlk sutuna görseli yerleştirir
plt.title("Kullanıcının Girdiği Kıyafet")     # Başlık

plt.subplot(1, 2, 2)     # 2. sütuna görseli yerleştirmesi gerektiğini söyler
plt.imshow(most_similar_rgb)     # İkinci sutuna görseli yerleştirir
plt.title("En Benzer Kıyafet")     # Başlık

plt.show()     # Oluşturulan subplotları gösterir.
