import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from  sklearn.metrics import adjusted_rand_score


class K_Means:
	def __init__(self, df) -> None:
		self.dataset = np.array(df)
		self.cluster_count = int(input("K clusters: "))

		self.centroids = [self.dataset[np.random.randint(len(self.dataset))] for _ in range(self.cluster_count)]
		
		self.cluster_points = {}
		for k in range(self.cluster_count): self.cluster_points[k] = []

		self.cluster_points_indx = {}
		for k in range(self.cluster_count): self.cluster_points_indx[k] = []

	
	def distance(self, p1, p2):
		return np.sqrt(np.sum((p1 - p2)**2))
	

	def k_means(self):
		# epocas = int(input("Epocas: "))


		while True:
			# Asignar los elementos
			for k in range(len(self.centroids)):
				self.cluster_points[k] = []
				self.cluster_points_indx[k] = []

				for i in range(len(self.dataset)):
					distancias = []
					
					for centroid in self.centroids: distancias.append(self.distance(centroid, self.dataset[i]))

					classificar = np.argmin(distancias)
					self.cluster_points[classificar].append(self.dataset[i])
					if i not in self.cluster_points_indx[classificar]: self.cluster_points_indx[classificar].append(i)

			# Calcular el promedio de los valores que pertenecen a un cluster (todos los elementos por ahora)
			new_centroids = [0] * len(self.centroids)
			for k in range(len(self.cluster_points)):
				total_sum = [0] * len(self.centroids[0])

				temp_cluster_points = np.transpose(self.cluster_points[k])
				for i in range(len(temp_cluster_points)):
					total_sum[i] += np.mean(temp_cluster_points[i])
				
				new_centroids[k] = total_sum
			

			# Reemplazar los nuevos centroides si es necesario
			centroid_counter = 0
			for k in range(len(self.centroids)):
				if np.sum(self.centroids[k]) == np.sum(self.centroids[k]): centroid_counter += 1
			
			# Terminar si no hay cambios en los centroides
			if np.sum(self.centroids) == np.sum(new_centroids): break
			else: self.centroids = new_centroids



def gs_labels():
	x={'kidney': 0, 'hippocampus': 1, 'cerebellum': 2, 'colon': 3, 'liver': 4, 'endometrium': 5, 'placenta': 6} 
	etiquetas=[]
	with open('db\clase.txt', 'r') as read_obj:
		csv_reader = csv.reader(read_obj)
		nai = -1
		for i in csv_reader:
			nai +=1 
			if nai==0: continue
			for j in range(1,len(i)): 
				etiquetas.append(x[i[j]])
	print("Labels")
	print(etiquetas)
	return etiquetas



def main():
	file = open("db\dataset_tissue.txt")
	reader = csv.reader(file, delimiter=",")

	values = []
	for row in reader:
		values.append(row)


	values = np.array(values[1:])
	values = np.transpose(values)
	values = np.array(values[1:], dtype=float)

	values = StandardScaler().fit_transform(values)
	pca = PCA(.95)
	principalComponents = pca.fit_transform(values)

	values = np.array(principalComponents)


	test = K_Means(values)
	test.k_means()


	# aea
	file2 = open("db\clase.txt")
	new_file = csv.reader(file2, delimiter=",")

	vals = []
	
	for v in new_file: vals.append(v)

	res = []
	for i in range(1, len(vals)):
		res.append(vals[i][1])

	
	x = {'kidney': 0, 'hippocampus': 1, 'cerebellum': 2, 'colon': 3, 'liver': 4, 'endometrium': 5, 'placenta': 6} 
	gg = {}

	for i in test.cluster_points_indx.keys():
		tasx = []
		for j in test.cluster_points_indx[i]:
			tasx.append(x[res[j]])
		gg[i] = tasx
		
	
	print("ok")
	for i in gg.keys():
		print("Cluster", i)
		for j in range(len(gg[i])):
			print(gg[i][j], end=" ")
		print()

	solucion_lista = []
	for i in range(len(values)):
		for j in test.cluster_points_indx.keys():
			if i in test.cluster_points_indx[j]: solucion_lista.append(j)
	
	label_final = gs_labels()

	valor_indx_rand = adjusted_rand_score(label_final, solucion_lista)
	print(valor_indx_rand)
	
	
	
	



main()
