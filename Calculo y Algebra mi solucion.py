#CALCULO


#EJERCICIO 1
#Let's say, in my office, it takes me 10 seconds (time) to travel 25 meters (distance) to the coffee machine. If we want to express the above situation as a function, then it would be:

#distance = speed * time

#So for this case, speed is the first derivative of the distance function above. As speed describes the rate of change of distance over time,
#  when people say taking the first derivative of a certain function, they mean finding out the rate of change of a function.

#1 Encuentra la velocidad y construye la funcion lineal de la distancia (d) con respecto al tiempo (t), cuando (t va entre [0,10])

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from operator import add

#La velicodad seria 25metros sobre 10 segundos lo cual deberia dar un resultado de 2.5m/seg
V = 25/10


#Definiendo la funcion

def d(t):
    d=V*t
    return d

t= np.linspace(0,10,11)

#Graficando la funcion de distancia en el dominio (t)

plt.plot(t, d(t))
plt.xlabel('tiempo')
plt.ylabel('Distancia')
plt.title('Grafico de la distancia en funcion del tiempo')

# Mostramos el gráfico
plt.show()

#Creando un Data Frame
df= pd.DataFrame({'Tiempo': t, 'Distancia': d(t)})
print(df.head)





#EJERCICIO 2 
# It turned out that I wasn't walking a constant speed towards getting my coffee, but I was accelerating (my speed increased over time). If my
#  initial speed = 0, it still took me 10 seconds to travel from my seat to my coffee, but I was walking faster and faster.

#Vo = 0 initial speed
#t2 = 10 time
# a = acceleration

#d2 = Vo * t + 0.5*a*(t**2)
#Vf= Vo + a*t
#despejando se obtiene que la aceleracion es de 0.5 m/s**2 OJO AQUI <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#The first derivative of the speed function is acceleration. I realize that the speed function is closely related to the distance function.
#Find the acceleration value and build the quadratic function where t goes from [0,10]. Also, create a graph and a table

#definiendo aceleracion donde d son 25m y t es un array que va de 0 a 10
def d2(t):
    d2 = 0.5*0.5*(t**2)
    return d2


plt.plot(t,d2(t))
plt.xlabel('tiempo')
plt.ylabel('Aceleracion')
plt.title('Grafico de la distancia en funcion del tiempo')

# Mostramos el gráfico
plt.show()

#Creando un Data Frame
df2= pd.DataFrame({'Tiempo': t, 'Distancia': d2(t)})
print(df2.head)



#When I arrive to the coffee machine, I hear my colleague talking about the per-unit costs of producing 'product B' for the company. 
# As the company produces more units, the per-unit costs continue to decrease until a point where they start to increase.

#To optimize the per-unit production cost at its minimum to optimize efficiency, the company would need to find the number of units to be 
# produced where the per-unit production costs begin to change from decreasing to increasing.

#Build a quadratic function f(x) = 0.1(x**2) - 9x + 4500 on x[0,100] to create the per-unit cost function, and make a conclusion

# Define and plot the function
def Production_cost(x): 
    cost = 0.1*(x)**2 -9*x + 4500
    return cost
x = np.linspace(0,100)
plt.plot(x, Production_cost(x))
plt.xlabel('Unidades')
plt.ylabel('Costo')
plt.title('Costo por unidad')

# Mostramos el gráfico
plt.show()

#Con el grafico podemos ver que el costo por unidad baja hasta alcanzar las 40 unidades y de ese punto empieza a aumentar alcanzano su punto maximo en 100


#ALGEBRA LINEAR

#EJERCICIO 1 SUMA DE MATRICES

#Suppose we have two matrices A and B.

A = [[1,2],[3,4]]
B = [[4,5],[6,7]]

#then we get
#A + B = [[5,7],[9,11]]
#A - B = [[-3,-3],[-3,-3]]
#Make the sum of two matrices using Python with NumPy

# Creating first matrix
A = np.array([[1, 2], [3, 4]])
 
# Creating second matrix
B = np.array([[4, 5], [6, 7]])

# Print elements
print("Printing elements of first matrix")
print(A)
print("Printing elements of second matrix")
print(B)
 
# Adding both matrices
print("Addition of two matrices")
print(np.add(A, B)) #El metodo add crea la adicion de matrices directamente con Numpy.and

#Exercise 2: Sum of two lists
#There will be many situations in which we'll have to find an index-wise summation of two different lists. 
# This can have possible applications in day-to-day programming. In this exercise, 
# we will solve the same problem in various ways in which this task can be performed.

#We have the following two lists:

list1 = [2, 5, 4, 7, 3]
list2 = [1, 4, 6, 9, 10]
#Now let's use Python code to demonstrate addition of two lists.

# Naive method

# Initializing lists
list1 = [2, 5, 4, 7, 3]
list2 = [1, 4, 6, 9, 10]
 
# Printing original lists
print ("Original list 1 : " + str(list1))
print ("Original list 2 : " + str(list2))
 
# Using naive method to add two lists 
res_list = []
for i in range(0, len(list1)):
    res_list.append(list1[i] + list2[i])
 
# Printing resulting list 
print ("Resulting list is : " + str(res_list))

#Now use the following three different methods to make the same calculation: sum of two lists

# Use list comprehension to perform addition of the two lists:

# Initializing lists
list1 = [1, 3, 4, 6, 8]
list2 = [4, 5, 6, 2, 10]
 
# Printing original lists
print ("Original list 1 : " + str(list1))
print ("Original list 2 : " + str(list2))
 
# Using list comprehension to add two lists
res_list = [list1[i] + list2[i] for i in range(len(list1))]
 
# Printing resulting list 
print ("Resulting list is : " + str(res_list))

# Use map() + add():

from operator import add
 
# Initializing lists
list1 = [1, 3, 4, 6, 8]
list2 = [4, 5, 6, 2, 10]
 
# Printing original lists
print ("Original list 1 : " + str(list1))
print ("Original list 2 : " + str(list2))
 
# Using map() + add() to add two lists
res_list = list(map(add, list1, list2))
 
# Printing resulting list  
print ("Resulting list is : " + str(res_list))

# Initializing lists
list1 = [1, 3, 4, 6, 8]
list2 = [4, 5, 6, 2, 10]
 
# Printing original lists
print ("Original list 1 : " + str(list1))
print ("Original list 2 : " + str(list2))
 
# Using zip() + sum() to add two lists
res_list = [sum(i) for i in zip(list1, list2)]
 
# Printing resulting list  
print ("Resulting list is : " + str(res_list))


#Exercise 3: Dot multiplication
#We have two matrices:

matrix1 = [[1,7,3],
 [4,5,2],
 [3,6,1]]
matrix2 = [[5,4,1],
 [1,2,3],
 [4,5,2]]


#A simple technique but expensive method for larger input datasets is using for loops. In this exercise, we will first use nested for loops to iterate through each row and column of the matrices, 
#and then we will perform the same multiplication using NumPy.

# Using a for loop input two matrices of size n x m
matrix1 = [[1,7,3],
 [4,5,2],
 [3,6,1]]
matrix2 = [[5,4,1],
 [1,2,3],
 [4,5,2]]
 
res = [[0 for x in range(3)] for y in range(3)]
 
# Explicit for loops
for i in range(len(matrix1)):
    for j in range(len(matrix2[0])):
        for k in range(len(matrix2)):
 
            # Resulting matrix
            res[i][j] += matrix1[i][k] * matrix2[k][j]
 
print(res)

# Input two matrices
mat1 = ([1,7,3],[ 4,5,2],[ 3,6,1])
mat2 = ([5,4,1],[ 1,2,3],[ 4,5,2])
 
# This will return dot product
res = np.dot(mat1,mat2)
 
# Print resulting matrix
print(res)