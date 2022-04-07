

map1={
	0:"VIT Chennai",
	1:"Vandalur",
	7:"Kelambakkam",
	3:"Tambaram",
	5:"Anna Nagar",
	6:"Velacherry",
	2:"Irandankattalai",
	8:"Chrompet",
	4:"Airport"
}

import cv2
import numpy as np

input_size = 320
confThreshold =0.2
nmsThreshold= 0.2
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
required_class_index = [2, 3, 5, 7]

detected_classNames = []
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)



net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)#cuda is a GPU Model
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)



def predict(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        


def read(image):
    img = cv2.imread(image)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    predict(outputs,img)
    return (len(detected_classNames))



















from collections import defaultdict


class Graph:


	def minDistance(self,dist,queue):

		minimum = float("Inf")
		min_index = -1
		for i in range(len(dist)):
			if dist[i] < minimum and i in queue:
				minimum = dist[i]
				min_index = i
		return min_index
	def printPath(self, parent, j):

		if parent[j] == -1 :
			print(map1[j],end="-->")
			return
		self.printPath(parent , parent[j])
		print (map1[j],end="-->")
	def printSolution(self, dist, parent):
		src = 0
		print("Vertex \t\t Traffic from Source\tPath")
		for i in range(1, len(dist)):
			print("\n%s --> %s \t%d\t" % (map1[src], map1[i],dist[i]),end=" ")
			self.printPath(parent,i)
	def dijkstra(self, graph, src):

		row = len(graph)
		col = len(graph[0])
		dist = [float("Inf")] * row
		parent = [-1] * row
		dist[src] = 0
		queue = []
		for i in range(row):
			queue.append(i)
		while queue:

			u = self.minDistance(dist,queue)
			queue.remove(u)
			for i in range(col):
				if graph[u][i] and i in queue:
					if dist[u] + graph[u][i] < dist[i]:
						dist[i] = dist[u] + graph[u][i]
						parent[i] = u
		self.printSolution(dist,parent)

g= Graph()

c1=read("images/1.jpeg")
c2=read("images/2.jpeg")
c3=read("images/3.jpeg")
c4=read("images/4.jpeg")
c5=read("images/5.jpeg")
c6=read("images/6.jpeg")
c7=read("images/7.jpeg")+1000
c8=read("images/8.jpeg")
c9=read("images/1.jpeg")
c10=read("images/2.jpeg")
c11=read("images/3.jpeg")
c12=read("images/4.jpeg")
c13=read("images/5.jpeg")
c14=read("images/6.jpeg")


graph = [[0, c1, 0, 0, 0, 0, 0, c2, 0],
        [c1, 0, c4, 0, 0, 0, 0, c3, 0],
        [0, c4, 0, c8, 0, c9, 0, 0, c6],
        [0, 0, c8, 0, c12, c11, 0, 0, 0],
        [0, 0, 0, c12, 0, c10, 0, 0, 0],
        [0, 0, c9, c11, c10, 0, c13, 0, 0],
        [0, 0, 0, 0, 0, c13, 0, c14, c7],
        [c2, c3, 0, 0, 0, 0, c14, 0, c5],
        [0, 0, c6, 0, 0, 0, c7, c5, 0]
        ]

print(graph)
g.dijkstra(graph,0)

image_file = 'images/8.jpeg'


