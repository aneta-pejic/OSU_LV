import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()


# ----------------------------
# ZADATAK 7.5.2 - 1
# ----------------------------

unique_colors = len(set([tuple(pixel) for pixel in img_array]))
print("Broj različitih boja u slici:", unique_colors)


# ----------------------------
# ZADATAK 7.5.2 - 2, 3 & 4
# ----------------------------
for K in [2, 5, 10, 20]:
    kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto')
    kmeans.fit(img_array)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    img_array_aprox = centers[labels]
    img_aprox = np.reshape(img_array_aprox, (w, h, d))

    plt.figure()
    plt.title(f"Kvantizacija boja (K={K})")
    plt.imshow(img_aprox)
    plt.tight_layout()
    plt.show()

    ### --- Razlika između dobivene slike i originala je ta da je dobivena slika kvantizirana, tj. boje su svedene na K različitih boja.
    ### --- Što je veći K, to je slika bliža originalu. Međutim, s povećanjem K raste i veličina slike, pa je potrebno naći optimalan broj K.


# ----------------------------
# ZADATAK 7.5.2 - 6
# ----------------------------
inertias = []
k_values = range(1, 10)

for K in k_values:
    kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto')
    kmeans.fit(img_array)
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(k_values, inertias, marker='o')
plt.xlabel("K")
plt.ylabel("J")
plt.title("Ovisnost J o broju grupa K")
plt.show()

# ----------------------------
# ZADATAK 7.5.2 - 7
# ----------------------------
K = 3 
kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto')
kmeans.fit(img_array)
labels = kmeans.labels_

for i in range(K):
    mask = (labels == i)
    cluster_img = np.zeros_like(img_array)
    cluster_img[mask] = img_array[mask]
    cluster_img = np.reshape(cluster_img, (w, h, d))

    plt.figure()
    plt.title(f"Grupa {i+1}")
    plt.imshow(cluster_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

