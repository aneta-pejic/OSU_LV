import numpy as np

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

visina = data[:, 1]  
spol = data[:, 0] 

ind_muskarci = (spol == 1)
ind_zene = (spol == 0) 

min_visina_muskarci = np.min(visina[ind_muskarci])
max_visina_muskarci = np.max(visina[ind_muskarci])
mean_visina_muskarci = np.mean(visina[ind_muskarci])

min_visina_zene = np.min(visina[ind_zene])
max_visina_zene = np.max(visina[ind_zene])
mean_visina_zene = np.mean(visina[ind_zene])

print("Podaci za muškarce:")
print("Minimalna visina:", min_visina_muskarci, "cm.")
print("Maksimalna visina:", max_visina_muskarci, "cm.")
print("Srednja visina:", mean_visina_muskarci, "cm.")

print("\nPodaci za žene:")
print("Minimalna visina:", min_visina_zene, "cm.")
print("Maksimalna visina:", max_visina_zene, "cm.")
print("Srednja visina:", mean_visina_zene, "cm.")
