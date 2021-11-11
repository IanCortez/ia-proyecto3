import numpy as np
from queue import Queue
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DBScan:
    def __init__(self, df, radius=1, min_points=5) -> None:
        self.dataset = np.array(df)
        self.radius = radius	# epsilon
        self.min_points = min_points
        self.noise = 0
        self.cluster_label = 0


    def distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))


    def range_query(self, x):
        neighbors = []

        for y in range(len(self.dataset)):
            q = self.dataset[y, :len(self.dataset[y])]
            if self.distance(x, q) <= self.radius: neighbors.append(y)
        
        return neighbors

    
    def predict(self, x):
        preds = []
        
        for point in x:
            neighbors = self.range_query(point)
            label = self.dataset[neighbors[0], -1]
            print(label)
            preds.append(label)
        
        return preds


    def fit(self):
        self.dataset = np.append(self.dataset, np.array([[-1]*len(self.dataset)]).reshape(-1, 1), axis=1)

        for i in range(len(self.dataset)):
            if self.dataset[i][-1] != -1: continue

            p = self.dataset[i, :len(self.dataset[i])]
            
            neighbours = self.range_query(p)

            if len(neighbours) < self.radius:
                self.dataset[i][-1] = self.noise
                continue
                
            self.cluster_label += 1
            self.dataset[i][-1] = self.cluster_label

            encontrados = neighbours

            q = Queue()

            for i in neighbours:
                q.put(i)
            
            while not q.empty():
                curr = q.get()

                if self.dataset[curr][-1] == self.noise: self.dataset[curr][-1] = self.cluster_label

                if self.dataset[curr][-1] != -1: continue

                point = self.dataset[curr][: len(self.dataset[curr])]
                new_neighbours = self.range_query(point)

                self.dataset[curr][-1] = self.cluster_label

                if len(new_neighbours) < self.min_points: continue

                for i in new_neighbours:
                    if i not in encontrados:
                        q.put(i)
                        encontrados.append(i)


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
    print(len(principalComponents)) # cantidad de datos
    print(len(principalComponents[0]))  # cantidad de caracteristicas

    values = np.array(principalComponents)


    radius = float(input("Radio de busqueda: "))
    min_points = int(input("Minimo de puntos: "))
    test = DBScan(values, radius, min_points)

    test.fit()


    # aea
    file2 = open("db\clase.txt")
    new_file = csv.reader(file2, delimiter=",")

    vals = []
    res = {}
    for v in new_file: vals.append(v)
    
    for i in range(1, len(vals)):
        if vals[i][1] not in res: res[vals[i][1]] = 1
        else: res[vals[i][1]] += 1

    for i in res.keys(): print(f"{i}: {res[i]}")

    print()
    indexes = {}
    for i in range(1, len(vals)):
        indexes[vals[i][0]] = vals[i][1]


    labels = {}
    contador = {}
    for i in range(len(test.dataset)):
        if test.dataset[i][-1] not in labels:
            labels[test.dataset[i][-1]] = [i]
            contador[test.dataset[i][-1]] = 1
        else:
            labels[test.dataset[i][-1]].append(i)
            contador[test.dataset[i][-1]] += 1
    
    for i in labels.keys():
        print(str(i) + ":", end=" ")
        for j in labels[i]:
            print(indexes[str(j+1)], end=" ")
        print()
        

main()