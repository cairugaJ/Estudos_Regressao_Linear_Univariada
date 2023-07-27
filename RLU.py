import numpy as np

class RLUnivariada():
    '''Classe para algoritmo de Regressão Linear Univariada usando Equação Normal'''
    def __init__(self):
       self.w = 0
       self.b = 0  
    
    def predicao (self,x):
        '''Função que calcula valores estimados para y'''
        m = x.shape
        f_wb = np.zeros(m)
        for i in range(len(x)):
            f_wb[i] = self.w*x[i] + self.b
        return f_wb 
    
    def fCusto(self,x,y):
        '''Cálculo da função perda: Erro Médio Quadrado'''
        soma = 0
        for i in range(len(y)):
            soma+= (x[i]*self.w+self.b - y[i])**2
        j = soma/2 *len(y)
        return j

    def gd_w(self,x,y):
        '''Gradiente Descendente de w'''
        soma =0
        for i in range(len(y)):
            soma+= (x[i]*self.w+self.b  - y[i])*x[i]
        jw = soma/len(y)
        return self.w - self.alpha * jw

    def treino(self, x, target):
        #Adicionando uma coluna de 1's para a matriz X para incluir o termo de polarização (bias)
        X = np.c_[x, np.ones(x.shape[0])]

        #Utilizando a equação normal para encontrar os valores de w e b
        XT_X_inv = np.linalg.inv(X.T.dot(X)) #Inversa da multiplicação da matriz X por sua transposta
        theta = XT_X_inv.dot(X.T).dot(target) #vetor com w e b

        self.w = theta[0]
        self.b = theta[1]
