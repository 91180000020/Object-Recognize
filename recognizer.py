from PIL import Image
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def normalize(hist):
	hist_np = np.array(hist, dtype=np.float32)
	hist_sum = sum(hist)
	hist_norm = hist_np / hist_sum
	return hist_norm.tolist()


image1tv = Image.open('televizyon/tv1.jpg').resize((500,325))         
image2tv = Image.open('televizyon/tv2.jpg').resize((500,325)) 
image3tv = Image.open('televizyon/tv3.jpg').resize((500,325))
image4tv = Image.open('televizyon/tv4.jpg').resize((500,325))
image5tv = Image.open('televizyon/tv5.jpg').resize((500,325))
image1tv_hist = normalize(image1tv.histogram())     
image2tv_hist = normalize(image2tv.histogram()) 
image3tv_hist = normalize(image3tv.histogram())
image4tv_hist = normalize(image4tv.histogram())
image5tv_hist = normalize(image5tv.histogram())
image1p = Image.open('pirinc/p1.jpg').resize((500,325))         
image2p = Image.open('pirinc/p2.jpg').resize((500,325)) 
image3p = Image.open('pirinc/p3.jpg').resize((500,325))
image4p = Image.open('pirinc/p4.jpg').resize((500,325))
image5p = Image.open('pirinc/p5.jpg').resize((500,325))
image1p_hist = normalize(image1p.histogram())     
image2p_hist = normalize(image2p.histogram()) 
image3p_hist = normalize(image3p.histogram())
image4p_hist = normalize(image4p.histogram())
image5p_hist = normalize(image5p.histogram())
image1d = Image.open('deterjan/d1.jpg').resize((500,325))         
image2d = Image.open('deterjan/d2.jpg').resize((500,325)) 
image3d = Image.open('deterjan/d3.jpg').resize((500,325))
image4d = Image.open('deterjan/d4.jpg').resize((500,325))
image5d = Image.open('deterjan/d5.jpg').resize((500,325))
image1d_hist = normalize(image1d.histogram())     
image2d_hist = normalize(image2d.histogram()) 
image3d_hist = normalize(image3d.histogram())
image4d_hist = normalize(image4d.histogram())
image5d_hist = normalize(image5d.histogram())
image1g = Image.open('gozluk/g1.jpg').resize((500,325))         
image2g = Image.open('gozluk/g2.jpg').resize((500,325)) 
image3g = Image.open('gozluk/g3.jpg').resize((500,325))
image4g = Image.open('gozluk/g4.jpg').resize((500,325))
image5g = Image.open('gozluk/g5.jpg').resize((500,325))
image1g_hist = normalize(image1g.histogram())     
image2g_hist = normalize(image2g.histogram()) 
image3g_hist = normalize(image3g.histogram())
image4g_hist = normalize(image4g.histogram())
image5g_hist = normalize(image5g.histogram())
image1k = Image.open('kitaplik/k1.jpg').resize((500,325))         
image2k = Image.open('kitaplik/k2.jpg').resize((500,325)) 
image3k = Image.open('kitaplik/k3.jpg').resize((500,325))
image4k = Image.open('kitaplik/k4.jpg').resize((500,325))
image5k = Image.open('kitaplik/k5.jpg').resize((500,325))
image1k_hist = normalize(image1k.histogram())     
image2k_hist = normalize(image2k.histogram()) 
image3k_hist = normalize(image3k.histogram())
image4k_hist = normalize(image4k.histogram())
image5k_hist = normalize(image5k.histogram())
image1mk = Image.open('makarna/mk1.jpg').resize((500,325))         
image2mk = Image.open('makarna/mk2.jpg').resize((500,325)) 
image3mk = Image.open('makarna/mk3.jpg').resize((500,325))
image4mk = Image.open('makarna/mk4.jpg').resize((500,325))
image5mk = Image.open('makarna/mk5.jpg').resize((500,325))
image1mk_hist = normalize(image1mk.histogram())     
image2mk_hist = normalize(image2mk.histogram()) 
image3mk_hist = normalize(image3mk.histogram())
image4mk_hist = normalize(image4mk.histogram())
image5mk_hist = normalize(image5mk.histogram())
image1m = Image.open('masa/m1.jpg').resize((500,325))         
image2m = Image.open('masa/m2.jpg').resize((500,325)) 
image3m = Image.open('masa/m3.jpg').resize((500,325))
image4m = Image.open('masa/m4.jpg').resize((500,325))
image5m = Image.open('masa/m5.jpg').resize((500,325))
image1m_hist = normalize(image1m.histogram())     
image2m_hist = normalize(image2m.histogram()) 
image3m_hist = normalize(image3m.histogram())
image4m_hist = normalize(image4m.histogram())
image5m_hist = normalize(image5m.histogram())
image1s = Image.open('sut/s1.jpg').resize((500,325))         
image2s = Image.open('sut/s2.jpg').resize((500,325)) 
image3s = Image.open('sut/s3.jpg').resize((500,325))
image4s = Image.open('sut/s4.jpg').resize((500,325))
image5s = Image.open('sut/s5.jpg').resize((500,325))
image1s_hist = normalize(image1s.histogram())     
image2s_hist = normalize(image2s.histogram()) 
image3s_hist = normalize(image3s.histogram())
image4s_hist = normalize(image4s.histogram())
image5s_hist = normalize(image5s.histogram())
image1sd = Image.open('sandalye/s1.jpg').resize((500,325))         
image2sd = Image.open('sandalye/s2.jpg').resize((500,325)) 
image3sd = Image.open('sandalye/s3.jpg').resize((500,325))
image4sd = Image.open('sandalye/s4.jpg').resize((500,325))
image5sd = Image.open('sandalye/s5.jpg').resize((500,325))
image1sd_hist = normalize(image1sd.histogram())     
image2sd_hist = normalize(image2sd.histogram()) 
image3sd_hist = normalize(image3sd.histogram())
image4sd_hist = normalize(image4sd.histogram())
image5sd_hist = normalize(image5sd.histogram())
image1u = Image.open('utu/u1.jpg').resize((500,325))         
image2u = Image.open('utu/u2.jpg').resize((500,325)) 
image3u = Image.open('utu/u3.jpg').resize((500,325))
image4u = Image.open('utu/u4.jpg').resize((500,325))
image5u = Image.open('utu/u5.jpg').resize((500,325))
image1u_hist = normalize(image1u.histogram())     
image2u_hist = normalize(image2u.histogram()) 
image3u_hist = normalize(image3u.histogram())
image4u_hist = normalize(image4u.histogram())
image5u_hist = normalize(image5u.histogram())

dataset = [
    (image1tv_hist),                     
    (image2tv_hist),
    (image3tv_hist),
    (image4tv_hist),
    (image5tv_hist),                     
    (image1p_hist),
    (image2p_hist),
    (image3p_hist),                     
    (image4p_hist),
    (image5p_hist),
    (image1d_hist),
    (image2d_hist),
    (image3d_hist),
    (image4d_hist), 
    (image5d_hist),                    
    (image1g_hist),
    (image2g_hist),
    (image3g_hist),                     
    (image4g_hist),
    (image5g_hist),
    (image1k_hist),
    (image2k_hist),                     
    (image3k_hist),
    (image4k_hist),
    (image5k_hist),
    (image1mk_hist),                     
    (image2mk_hist),
    (image3mk_hist),
    (image4mk_hist),
    (image5mk_hist),
    (image1m_hist),
    (image2m_hist),
    (image3m_hist),
    (image4m_hist),
    (image5m_hist),
    (image1s_hist),
    (image2s_hist),
    (image3s_hist),
    (image4s_hist),
    (image5s_hist),
    (image1sd_hist),
    (image2sd_hist),
    (image3sd_hist),
    (image4sd_hist),
    (image5sd_hist),
    (image1u_hist),
    (image2u_hist),
    (image3u_hist),
    (image4u_hist),
    (image5u_hist)


]

labels = [
   0,              
   0,
   0,
   0,
   0,
   1,
   1,
   1,
   1,
   1,
   2,
   2,
   2,
   2,
   2,
   3,
   3,
   3,
   3,
   3,
   4,
   4,
   4,
   4,
   4,
   5,
   5,
   5,
   5,
   5,
   6,
   6,
   6,
   6,
   6,
   7,
   7,
   7,
   7,
   7,
   8,
   8,
   8,
   8,
   8,
   9,
   9,
   9,
   9,
   9
   ]


dataset_np = np.array(dataset, dtype=np.float32)
labels_np = np.array(labels, dtype=np.float32)

print("\n\n") 
print("K Nearest Neighbors")
knn =cv2.ml.KNearest_create()
knn = KNeighborsClassifier(n_neighbors=3)
 
kf = KFold(n_splits=10) # Define the split - into 2 folds 
kf.get_n_splits(dataset_np) # returns the number of splitting iterations in the cross-validator
print(kf) 
KFold(n_splits=10, random_state=None, shuffle=False)
for train_index, test_index in kf.split(dataset_np):
     print("train:",train_index,"test:",test_index)
     dataset_np_train, dataset_np_test = dataset_np[train_index], dataset_np[test_index]
     labels_np_train, labels_np_test = labels_np[train_index], labels_np[test_index]
     knn.fit(dataset_np_train, labels_np_train)
     sonuc=knn.predict(dataset_np_test)
     print(sonuc)
     cm=confusion_matrix(labels_np_test,sonuc)
     print("Confusion Matrix: \n",cm)
     plt.matshow(cm)
     plt.title('Confusion Matrix')
     plt.colorbar()
     plt.ylabel('True Label')
     plt.xlabel('Predicted Label')
     plt.show()
 
    
print("\n\n\n")   
print("Decision Tree Classifier")
clf=DecisionTreeClassifier(random_state=0)

kf = KFold(n_splits=10) # Define the split - into 2 folds 
kf.get_n_splits(dataset_np) # returns the number of splitting iterations in the cross-validator
print(kf) 
KFold(n_splits=10, random_state=None, shuffle=False)
for train_index, test_index in kf.split(dataset_np):
     print("train:",train_index,"test:",test_index)
     dataset_np_train, dataset_np_test = dataset_np[train_index], dataset_np[test_index]
     labels_np_train, labels_np_test = labels_np[train_index], labels_np[test_index]
     clf.fit(dataset_np_train, labels_np_train)
     sonuc=clf.predict(dataset_np_test)
     print(sonuc)
     cm=confusion_matrix(labels_np_test,sonuc)
    print("Confusion Matrix: \n",cm)
     plt.matshow(cm)
     plt.title('Confusion Matrix')
     plt.colorbar()
     plt.ylabel('True Label')
     plt.xlabel('Predicted Label')
     plt.show()

 
     