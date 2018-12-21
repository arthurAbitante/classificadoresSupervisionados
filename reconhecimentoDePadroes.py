from sklearn import svm
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split  

def pca_func(X, m):
	pca = decomposition.PCA(n_components=m)
	pca.fit(X)
	Y = pca.transform(X)
	
	
	return pca.explained_variance_ratio_.astype(np.float32), Y.astype(np.float32)
	
def pca_func2(X, m, var):
	pca = decomposition.PCA(n_components=m)
	pca.fit(X)
	Y = pca.transform(X)
	
	pca.explained_variance_ratio_ = var
		
	return pca.explained_variance_ratio_.astype(np.float32), Y.astype(np.float32)	
	
def normalization(X):
	#normalizacao
	for i in range(X.shape[1]):
		X[...,i] = (X[...,i] - np.min(X[...,i])) / (np.max(X[...,i]) - np.min(X[...,i]))
	
	return X

def NaiveBayes(dadosTreino, labelsTreino, dadosTeste):
	
	gnb = GaussianNB()
	gnb.fit(dadosTreino, labelsTreino) 
	nb = gnb.predict(dadosTeste) 	
	
	return nb
	
def SvmLinear(dadosTreino, labelsTreino, dadosTeste):
	
	models = (svm.SVC(kernel='linear', C=1.0))
	models = (models.fit(dadosTreino, labelsTreino.ravel()))
	svmLinear = models.predict(dadosTeste)
	
	return svmLinear
	
def SvmNLinear(dadosTreino, labelsTreino, dadosTeste):
	
	rbf = (svm.SVC(kernel='rbf', gamma=0.7, C=1.0))
	rbf = (rbf.fit(dadosTreino, labelsTreino.ravel()))
	
	svmn = rbf.predict(dadosTeste)
	
	return svmn
	
def Cart(dadosTreino, labelsTreino, dadosTeste):
	
	cart = tree.DecisionTreeRegressor()
	cart = cart.fit(dadosTreino, labelsTreino)
	
	return cart.predict(dadosTeste)	
	
def variarDados(dados, var):
	colunaDados = dados.shape[1]
	dad = dados
	for i in range(0, colunaDados):
		if(var == 1):
			variarColuna = (colunaDados * 0.75)
			variarColuna = int(variarColuna)
			#for j in range(0, variarColuna):		
			dadosVariados = dad[:, 0:variarColuna]
				#print("Variando em 75\n\n")
		if(var == 2):
			variarColuna = (colunaDados * 0.90)
			variarColuna = int(variarColuna)
			#for j in range(0, variarColuna):
			dadosVariados = dad[:, 0:variarColuna]
				#print("Variando em 90\n\n")
		if(var == 3):
			variarColuna = (colunaDados * 0.99)
			variarColuna = int(variarColuna)
			#for j in range(0, variarColuna):
				
			dadosVariados = np.asarray(dad[:,0:variarColuna])
				
				#print("Variando em 99\n\n")
	
	return dadosVariados

#mudadas por causa da variabilidade
#muda 2/3 de dados
def particionar(dados, labels):
	linhasDados = dados.shape[0]
	linhasLabels = labels.shape[0]
	d = dados
	lab = labels
	jun = np.concatenate((d, lab), axis=1)


	np.random.shuffle(jun)
#	np.random.shuffle(lab)
	

	
	ldTeste = int(linhasDados / 3)
	ldTreino = int((linhasDados * 2) / 3)
	llTeste = int(linhasLabels / 3)
	llTreino = int((linhasLabels * 2) / 3)
	


	labelsTreino = lab[0:llTreino,...]
	labelsTeste = lab[0:llTeste,...]
	dadosTreino = d[0:ldTreino,...]
	dadosTeste = d[0:ldTeste,...]


	
	return dadosTreino, labelsTreino, dadosTeste, labelsTeste

def svmLinear(dadosTreino, labelsTreino, dadosTeste, labelsTeste):
	y_pred = SvmLinear(dadosTreino, labelsTreino, dadosTeste)
	acuracia = accuracy_score(labelsTeste, y_pred)
	confusao = confusion_matrix(labelsTeste.ravel(), y_pred)
	especificidade = precision_score(labelsTeste, y_pred)
	sensibilidade = recall_score(labelsTeste, y_pred)
	
	
	print("\n\n\nAcuracia: ", acuracia)
	print("Matriz Confusao: \n", confusao)
	print("Especificidade_score:  ", especificidade)
	print("Sensibilidade_score: ", sensibilidade)

	
def svmnLinear(dadosTreino, labelsTreino, dadosTeste, labelsTeste):
	y_pred = SvmNLinear(dadosTreino, labelsTreino, dadosTeste)
	acuracia = accuracy_score(labelsTeste.ravel(), y_pred)
	confusao = confusion_matrix(labelsTeste.ravel(), y_pred)
	especificidade = precision_score(labelsTeste, y_pred)
	sensibilidade = recall_score(labelsTeste, y_pred)
	

	print("\n\nAcuracia: ", acuracia)
	print("Matriz Confusao: \n", confusao)
	print("Especificidade:  ", especificidade)
	print("Sensibilidade: ", sensibilidade)	

	
	
def nBayes(dadosTreino, labelsTreino, dadosTeste, labelsTeste):
	y_pred = NaiveBayes(dadosTreino, labelsTreino, dadosTeste)
	acuracia = accuracy_score(labelsTeste.ravel(), y_pred)
	confusao = confusion_matrix(labelsTeste.ravel(), y_pred)

	
	especificidade = precision_score(labelsTeste, y_pred)
	sensibilidade = recall_score(labelsTeste, y_pred)
	
	print("\n\nAcuracia: ", acuracia)
	print("Matriz Confusao: \n", confusao)
	print("Especificidade:  ", especificidade)
	print("Sensibilidade: ", sensibilidade)	

def cart(dadosTreino, labelsTreino, dadosTeste, labelsTeste):	
	y_pred = Cart(dadosTreino, labelsTreino, dadosTeste)
	acuracia = accuracy_score(labelsTeste.ravel(), y_pred)
	confusao = confusion_matrix(labelsTeste.ravel(), y_pred)
	especificidade = precision_score(labelsTeste, y_pred)
	sensibilidade = recall_score(labelsTeste, y_pred)
	
	print("\n\nAcuracia: ", acuracia)
	print("Matriz Confusao: \n", confusao)
	print("Especificidade:  ", especificidade)
	print("Sensibilidade: ", sensibilidade)	
	
	
	
def variancia(explain, var):
	exp = explain
	soma = 0

	if(var == 1):
		soma = exp[0] + exp[1] + exp[2] 
	if(var == 2):
		soma = exp[0] + exp[1] + exp[2] + exp[3] + exp[4] + exp[5] 
	if(var == 3):				
		soma = exp[0] + exp[1] + exp[2] + exp[3] + exp[4] + exp[5] + exp[6] + exp[7] + exp[8] + exp[9] + exp[10] + exp[11] + exp[12] + exp[13] + exp[14] + exp[15]		
					
	return soma

		
	
def main():
	np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)}) # Para imprimir em decimal
	X = np.loadtxt("cancer.data", delimiter=",") # pega o dataset
	label = open("label.data", 'r')
	
	#cria um array de 569 começando com 0 e o transforma em 569 linhas
	lab = np.zeros(569).reshape((569, 1))
	c = 0
	#percorre as linhas do dataset do label
	for l in label:
		#se tal linha tiver o valor M , a matriz label recebe 1 e se for B recebe 0
		if(l == "M\n"):
			lab[c] = 1
		elif(l == "B\n"): 
			lab[c] = 0
		c = c + 1
	lab = lab.astype(int)
      
#	print(lab)  
######################################
	
	normalizar = normalization(X) 
			
	explain, normalizado = pca_func(normalizar, normalizar.shape[1])
#	print(explain.shape)
	
	var1 = variancia(explain, 1)
	var2 = variancia(explain, 2)
	var3 = variancia(explain, 3)	
	
	explain75, normalizado75 = pca_func2(normalizar, normalizar.shape[1], var1)
	explain90, normalizado90 = pca_func2(normalizar, normalizar.shape[1], var2)
	explain99, normalizado99 = pca_func2(normalizar, normalizar.shape[1], var3)

	print("Variancia 75: ", explain75)
	print("Variancia 90: ", explain90)
	print(" Variancia 99: ", explain99)
	



#	integrada = np.concatenate((normalizado, lab), axis=1)
	dTreino, dTeste, lTreino, lTeste = train_test_split(normalizado, lab, test_size = 0.30)
	dTreino75, dTeste75, lTreino75, lTeste75 = train_test_split(normalizado75, lab, test_size=0.30) 
	dTreino90, dTeste90, lTreino90, lTeste90 = train_test_split(normalizado90, lab, test_size=0.30) 
	dTreino99, dTeste99, lTreino99, lTeste99 = train_test_split(normalizado99, lab, test_size=0.30) 	
	

#--------
#VP | FP
#--------
#FN | VN
#--------
#	
	

	print("\n---------------------Suport Vector Machine Linear------------------------\n")
	
	print("-----------------------Normal--------------------------")
	svmLinear(dTreino, lTreino, dTeste, lTeste)
	print("------------75-------------------------------")
	svmLinear(dTreino75, lTreino75, dTeste75, lTeste75)
	print("-----------------------90-------------------")
	svmLinear(dTreino90, lTreino90, dTeste90, lTeste90)
	print("-----------------------99--------------------")
	svmLinear(dTreino99, lTreino99, dTeste99, lTeste99)

	print("\n---------------------Suport Vector Machine Non-Linear------------------------\n")
	
	print("-----------------------Normal--------------------------")
	svmnLinear(dTreino, lTreino, dTeste, lTeste)
	print("------------75-------------------------------")
	svmnLinear(dTreino75, lTreino75, dTeste75, lTeste75)
	print("-----------------------90-------------------")
	svmnLinear(dTreino90, lTreino90, dTeste90, lTeste90)
	print("-----------------------99--------------------")
	svmnLinear(dTreino99, lTreino99, dTeste99, lTeste99)

	
#	print("\n\n\n----Naive Bayes----\n")
#	nBayes(dTreino, lTreino, dTeste, lTeste)

	print("\n---------------------Nayve Bayes------------------------\n")
	
	print("-----------------------Normal--------------------------")
	nBayes(dTreino, lTreino, dTeste, lTeste)
	print("------------75-------------------------------")
	nBayes(dTreino75, lTreino75, dTeste75, lTeste75)
	print("-----------------------90-------------------")
	nBayes(dTreino90, lTreino90, dTeste90, lTeste90)
	print("-----------------------99--------------------")
	nBayes(dTreino99, lTreino99, dTeste99, lTeste99)

	
#	print("\n\n\n----Cart----\n")
#	cart(dTreino, lTreino, dTeste, lTeste)
	print("\n-------------------------Cart----------------------------\n")
	
	print("-----------------------Normal--------------------------")
	cart(dTreino, lTreino, dTeste, lTeste)
	print("------------75-------------------------------")
	cart(dTreino75, lTreino75, dTeste75, lTeste75)
	print("-----------------------90-------------------")
	cart(dTreino90, lTreino90, dTeste90, lTeste90)
	print("-----------------------99--------------------")
	cart(dTreino99, lTreino99, dTeste99, lTeste99)
	
#	print("\n-----------------------------\n")

	
	
	
main()
	

