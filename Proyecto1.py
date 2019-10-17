"""-------------------------------------------------------------------
Universidad: Universidad del Valle de Guatemala
Curso: Gráficas por Computadora
Fecha: 15-4-2019
Nombre: Alejandro Tejada 
Carnet: 17584
Nombre programa: Proyecto1.py
Propósito: Este programa es para el proyecto
Auxiliar: Gadhi Marcucci u otro
----------------------------------------------------------------------
"""
# Importamos valores de librerías
import struct
from random import uniform
from random import randint
from math import sqrt
from math import ceil
from math import *
import pywavefront
from collections import namedtuple
from numpy import *

# Definimos valores de dword y colores


def char(c):
    return struct.pack("=c", c.encode('ascii'))
# Definición de word


def word(c):
    return struct.pack("=h", c)
# definicion de dword


def dword(c):
    return struct.pack("=l", c)
# definicion de color


def color(r, g, b):
    return bytes([b, g, r])

V1 = namedtuple('Vertex1',['x'])
V2 = namedtuple('Vertex2', ['x', 'y'])
V3 = namedtuple('Vertex3', ['x', 'y', 'z'])
V4 = namedtuple('Vertex4', ['x', 'y', 'z', 'w'])

Modelo=[]

# Ahora la clase Texture aunque no está terminada, en esta tarea no se usa
class Texture(object):
    
    def __init__(self, path):
        self.path = path
        self.read()

    def read(self):
        img = open(self.path, 'rb')
        img.seek(2+4+4)
        header_size = struct.unpack("=l", img.read(4))[0]
        img.seek(2+4+4+4+4)
        self.width = float(struct.unpack("=l", img.read(4))[0])
        self.heigth =float( struct.unpack("=l", img.read(4))[0])
        self.pixels = []
        img.seek(header_size)

        for y in range(int(self.heigth)):
            self.pixels.append([])
            for x in range(int(self.width)):
                b = ord(img.read(1))
                g = ord(img.read(1))
                r = ord(img.read(1))
                self.pixels[y].append(color(r, g, b))

        img.close()

    def get_color(self, tx, ty,intensity=1):
        x = int(tx*self.width)
        y = int(ty*self.heigth)
        try:
            return self.pixels[y][x]
        except:
            return bytes(
                map(
                    lambda b: round(b*intensity)
                    if b * intensity > 0 else 0,
                    self.pixels[y-1][x-1]
                )
            )
            


# La clase object, donde cargamos nuestro modelo
class Obj(object):
    ROJO = ""
    GREEN = ""
    BLUE = ""
    
    def __init__(self, filename,mtl):
        with open(filename) as f:
            self.lines = f.read().splitlines()
        # Creamos las listas
        self.vertices = []
        self.faces = []
        self.colores = []
        self.normales=[]
        self.texturas = []
        self.VariableMTLO=mtl
        self.read()
        self.Color = ""
        # Método inicial
        

    def colorG(self, ColorMaterial):
        list1 = ColorMaterial
        str1 = ''.join(list1)
        self.GREEN = ((self.materiales[str1]['Kd'][1]))

    def colorR(self, ColorMaterial):
        list1 = ColorMaterial
        str1 = ''.join(list1)
        self.ROJO = ((self.materiales[str1]['Kd'][0]))

    def colorB(self, ColorMaterial):
        list1 = ColorMaterial
        str1 = ''.join(list1)
        self.BLUE = ((self.materiales[str1]['Kd'][2]))

    # método para retornar materiales
    # Este método de leer materiales fue extraido de: https://github.com/ratcave/wavefront_reader
    def read_mtlfile(self, fname):
        materials = {}
        with open(fname) as f:
            lines = f.read().splitlines()

        for line in lines:
            if line:
                split_line = line.strip().split(' ', 1)
                if len(split_line) < 2:
                    continue

                prefix, data = split_line[0], split_line[1]
                if 'newmtl' in prefix:
                    material = {}
                    materials[data] = material
                elif materials:
                    if data:
                        split_data = data.strip().split(' ')

                        if len(split_data) > 1:
                            material[prefix] = tuple(
                                float(d) for d in split_data)
                        else:
                            try:
                                material[prefix] = int(data)
                            except ValueError:
                                material[prefix] = float(data)

        return materials

    # Esta función quita la linea doble
    def Parseo_Entero(self, caras):
        # Creamos una lista de caras con split del SLASH
        listaDeCaras = caras.split('/')
        # Ahora hacemos una condicion de que si las comillas están en la lista
        if ('' in listaDeCaras):
            # Le removiemos la lista de caras
            listaDeCaras.remove('')
        # Y retornamos un mapa entero sin comillas
        return map(int, listaDeCaras)
    
    #método para arregar que no sea de base correcta, By Dennis
    def try_int(self,s, base=10, val=None):
        try:
            return int(s, base)
        except ValueError:
            return val

    # Declaramos un método para leer
    def read(self):
        # By Dennis :D
        # Hacemos un ciclo for
        self.materiales = self.read_mtlfile(self.VariableMTLO)
        for line in self.lines:
            if line:
                # Creamos un Prefix y un Split
                prefix, value = line.split(' ', 1)
                # Ahora hacemos una condición
                if prefix == 'v':
                    # Llenamos una lista con nuestros vértices
                    self.vertices.append(list(map(float, value.split(' '))))
                    # Ahora vemos la condición del Prefix para llenar un array con nuestras caras
                elif prefix == 'f':
                    listaTotal=([list(map(self.try_int, face.split('/'))) for face in value.split(' ')])
                    listaTotal.append(self.ROJO)
                    listaTotal.append(self.GREEN)
                    listaTotal.append(self.BLUE)
                    self.faces.append(listaTotal)
                elif prefix == 'usemtl':
                    Color = value.split(' ')
                    self.colorR(Color)
                    self.colorG(Color)
                    self.colorB(Color)
                elif prefix == 'vt':
                    self.texturas.append(list(map(float, value.split(' '))))
                elif prefix == 'vn':
                    self.normales.append(list(map(float, value.split(' '))))


class Bitmap(object):
    # Variables de la pantalla, por defecto esta es negra
    RedScreen = 0
    BlueScreen = 0
    GreenScreen = 0

    # Creamos las variables para el color del Vertex
    RedVertex = 0
    GreenVertex = 0
    BlueVertex = 0

    # Creamos variables para el cubo
    CuboXInicial = 0.0
    CuboYInicial = 0.0

    # Variables para correra
    corrimientoX = 0
    corrimientoY = 0

    # variables para color de point
    RedPoint = 255
    BluePoint = 255
    GreenPoint = 255

    zbuffer = []

    Rojo1 = 0
    Green1 = 0
    Blue1 = 0

    light2=V3(0,0,0)

    def __init__(self):
        self.glinit()

    # Acá iniciamos lo que necesitemos en la clase, como objetos internos
    def glinit(self):
        return None
        # Declaramos el zbuffer
        zbuffer = []

    # este método es para dibujar las lineas, extraído de lo que Dennis nos enseñó en clase.
    def glLine(self, x0, y0, x1, y1):
        offset = 0
        ListaPasos = []
        # Definimos los valores absolutos de la diferencia entre los parámetros
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        # declaramos un steep entre ambos diferenciales
        paso = dy > dx
        # se hace una condición
        if paso:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        # se calculan los diferenciales de nuevo
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        # Se iguala a
        threshold = dx
        # La variable y=y0 inicial
        yInicial = y0
        # se hace un ciclo while, si mientras la variable x0 es decir nuestro paso inicial es menor a x1
        while x0 <= x1:
            ListaPasos.append(x0)  # le agregamos el valor a una lista
            x0 += 0.001  # este iterados es lo que se agregar a la lista
        # luegog hacemos un for
        for pasito in ListaPasos:  # si el for en la lsita
            if paso:
                self.glVertex(yInicial, pasito)  # dibujamos iterando la lista
            else:
                self.glVertex(pasito, yInicial)
            offset += dy
            # acumulamos en el offset
            # si el offset es mayor que threshold entonces regresamos un valor pequeño decimal o aumentamos.
            if offset >= threshold:
                yInicial += 0.001 if y0 < y1 else -0.001
                threshold += 1 * dx

    # este método es para dibujar las lineas es con c ciclo While
    def glLine2(self, xx1, yy1, xx2, yy2):
            # los valores que nos ingresen serán entre -1 y 1, por lo cual llamamos a decodificadorX para pasarlo a pixel y que lo reconozca Point()
        s = 0
        x1 = int(self.decodificadorX(xx1))
        x2 = int(self.decodificadorX(xx2))
        y1 = int(self.decodificadorX(yy1))
        y2 = int(self.decodificadorX(yy2))
        # creamos un ciclo para imprimir
        while (s <= 1):
            # se calculan las ecuaciones de la recta
            x = x1+(x2-x1)*s
            y = y1+(y2-y1)*s
            # se invoca a point
            self.point(round(x), round(y))
            # se da un step mpequeño para que no se vean hoyos
            s = s+0.00001

    #este transform es el original
    def transform(self, vertex, translate=(0, 0, 0), scale=(1, 1, 1)):
        return V3(
            round(self.decodificadorX((vertex[0] + translate[0]) * scale[0])),
            round(self.decodificadorY((vertex[1] + translate[1]) * scale[1])),
            round(self.decodificadorX((vertex[2] + translate[2]) * scale[2]))
        )

    #definimos el segundo transform
    def transform2(self, vertex, translate=(0, 0, 0), scale=(1, 1, 1)):
        #Tomamos el vector que entra
        VectorOriginal = [vertex[0], vertex[1], vertex[2], 1]
        #se multiplica por la mega matriz
        VectorTransformado=  (self.MegaMatriz @ VectorOriginal ).tolist()[0]
        #se retorna normalizado
        return V3(
        round(VectorTransformado[0]/VectorTransformado[3]),
        round(VectorTransformado[1]/VectorTransformado[3]),
        round(VectorTransformado[2]/VectorTransformado[3])
        )
    
    #creamos el coso para pasar de textura a fondo
    def background_texture(self,texture):
        t = Texture(texture)
        for x in range(self.width):
            for y in range(self.height):
                self.framebuffer[x][y] = t.pixels[x][y]
        
# creamos un método para pintar
    def CambiarColorMesh(self, R, G, B):
        global Rojo1
        Rojo1 = (float(R)*255)
        global Green1
        Green1 = (float(G)*255)
        global Blue1
        Blue1 = (float(B)*255)
    
    #Creamos el load de los obj
    def loadTriangulos(self, filename, translate, scale,texture=None,rotate=(0,0,0),mtl=""):
        #llamamos al método que calcula la matriz de View
        self.loadTriangulosMatrix(translate,scale,rotate)
        # declaramos un objeto con el nombre del archivo
        modelo = Obj(filename,mtl)
        #Calculamos la Mega matriz destructora de mundos
        self.MatricitadePipeline()
        light = V3(0, 0,1)
        self.light2=V3(1,0,0)
        # colocamos la luz en ese eje
        for face in modelo.faces:
            vcount = len(face)
           
            # Contamos la longitud
            # contamos cada cara
            f1 = face[0][0] - 1
            f2 = face[1][0] - 1 
            f3 = face[2][0] - 1
            # en cada variable guardamos en una pareja el vértice del triángulo

            a2 = self.transform2(modelo.vertices[f1], translate, scale)
            b2 = self.transform2(modelo.vertices[f2], translate, scale)
            c2 = self.transform2(modelo.vertices[f3], translate, scale)

            #extraemos las normales
            n1=  face[0][2] - 1
            n2 = face[1][2] - 1
            n3 = face[2][2] - 1      
            nA = V3(*modelo.normales[n1])
            nB = V3(*modelo.normales[n2])
            nC = V3(*modelo.normales[n3])

            # se calcula la normal para el gary que no lleva textura
            normal = self.norm(self.cross(self.sub(b2, a2), self.sub(c2, a2)))
            # se calcula la intensidad
            intensidad = self.dot(normal, light)
            # ahora tomamos los parámetros de la lista y extramos los colores y le damos intensidad
            list1 = str((face[3]))
            str1 = ''.join(str(e) for e in list1)
            if(str1 != '[]'):
                color1 = int(round((float(str1)*255)*intensidad))
            list2 = str((face[4]))
            str2 = ''.join(str(e) for e in list2)
            color2 = int(round((float(str2)*255)*intensidad))
            list3 = str((face[5]))
            str3 = ''.join(str(e) for e in list3)
            color3 = int(round((float(str3)*255)*intensidad))

            # si la intensidad es 0 se salta por el momento no la usamos
            if intensidad < 0:
                continue
            if not texture:
                self.triangleBarycentric(a2, b2, c2, color(color1, color2, color3),mtl=mtl)
                
            else:
                #si lleva textura entonces jalamos las texturas
                t1 = face[0][1] - 1
                t2 = face[1][1] - 1
                t3 = face[2][1] - 1
                #estos if son para que no se salga de la longitud
                if(t1>=len(modelo.texturas)):
                   
                    t1=(len(modelo.texturas))-1
              
                if(t2>=len(modelo.texturas)):
                    
                    t2=(len(modelo.texturas))-1
                if(t3>=len(modelo.texturas)):
                    
                    t3=(len(modelo.texturas))-1

                tA = V2(*modelo.texturas[t1])
                tB = V2(*modelo.texturas[t2])
                tC = V2(*modelo.texturas[t3])
                #enviamos los parámetros a triangle barycentric
                self.triangleBarycentric(a2,b2,c2,color(color1, color2, color3),texture=texture,texture_coords=(tA,tB,tC),intensity=intensidad, normales=(nA,nC,nB),mtl=mtl)

    #Matricita
    def MatricitadePipeline(self):
        #primera muliplicacion: Modelo * View
        ModelView=self.Model @ self.View 
        #segunda multiplicacion: ModelView * Proyeccion
        ModelViewProjection= ModelView @ self.Projection 
        #tercera, ModelViewProjection por la de viewport
        self.MegaMatriz=ModelViewProjection @ self.ViewPortMatrix 
        #tenemos la mega matriz

    #Acá se calcula la matriz de vista
    def loadTriangulosMatrix(self, translate=(0,0,0), scale=(1,1,1),rotate=(0,0,0)):
        translate=V3(*translate)
        scale=V3(*scale)
        rotate=V3(*rotate)
        #se pasan a vectores de tres los parámetros de entrada
        #se calcula la matriz de traslacion
        translate_matrix=matrix([
            [1,0,0,translate.x],
            [0,1,0,translate.y],
            [0,0,1,translate.z],
            [0,0,0,1]
        ])
        #se implementa la matriz de rotacion en X, Y y Z, cada una con sus valores de sin y  cos
        rotation_matrix_x=matrix([
            [1,0,0,0],
            [0,cos(rotate.x),-sin(rotate.x),0],
            [0,sin(rotate.x),cos(rotate.x),0],
            [0,0,0,1]
        ])

        rotation_matrix_y=matrix([
            [cos(rotate.y),0,sin(rotate.y),0],
            [0,1,0,0],
            [-sin(rotate.y),0,cos(rotate.y),0],
            [0,0,0,1]
        ])
        
        rotarion_matrix_z=matrix([
            [cos(rotate.z),-sin(rotate.z),0,0],
            [sin(rotate.z),cos(rotate.z),0,0],
            [0,0,1,0],
            [0,0,0,1]
        ])
        #matriz de escala, en 0 por default, o identidad
        scale_matrix=matrix([
            [scale.x,0,0,0],
            [0,scale.y,0,0],
            [0,0,scale.z,0],
            [0,0,0,1]
        ])
        #se calcula la matriz de rotacion
        rotation_matrix=rotation_matrix_x @ rotation_matrix_y @ rotarion_matrix_z
        #calculamos la model
        self.Model=translate_matrix @ rotation_matrix @ scale_matrix
    
    #calculamos la matriz de View
    def loadViewMatrix(self,x,y,z,center):
        M=matrix([
            [x.x,x.y,x.z,0],
            [y.x,y.y,y.z,0],
            [z.x,z.y,z.z,0],
            [0,0,0,1]
        ])

        O=matrix([
            [1,0,0,-center.x],
            [0,1,0,-center.y],
            [0,0,1,-center.z],
            [0,0,0,1]
        ])
        #se multiplican la matriz M y O de la vista
        self.View= M @ O

    #método de la cámara, con Eye,center,up y Translate de parámetros
    def lookAt(self,eye,center,up,translate=(0,0,0)):
        translate= V3(*translate)
        #se calculan las coordenadas
        z=self.norm(self.sub(eye,center))
        x=self.norm(self.cross(up,z))
        y=self.norm(self.cross(z,x))
        
        #calculamos la matriz de view
        self.loadViewMatrix(x,y,z,center)
        #calculamos el coeficiente
        self.coeff=eye.z/self.length(self.sub(eye,center))
        #enviamos el coceff a la matriz de proyección
        self.loadProjectionMatrix(self.coeff)
        #calculamos la matriz de viewport
        self.loadViewPortMatrix(translate)
        
    #Calculo de la matriz de proyección
    def loadProjectionMatrix(self,coeff):
        self.Projection=matrix([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,coeff,1]
        ])
    
    #se calcula la matriz de viewport
    def loadViewPortMatrix(self,translate):
     
        self.ViewPortMatrix= matrix([
            [self.ViewpWidth/2,0,0,translate.x+ (self.ViewpWidth/2)],
            [0 ,self.ViewpHeight/2,0,translate.y+(self.ViewpHeight/2)],
            [0,0,500,500],
            [0,0,0,1]
        ])
           
    #se crea la ventana, se cambió el color a gris
    def glCreateWindow(self, width, height):
        self.width = width
        self.height = height
        self.framebuffer = [
            [
                color(155, 155, 155) for x in range(self.width)
            ]
            for y in range(self.height)
        ]
    # Esta función llena el mapa de bits(toda la ventana) de un solo color, en este caso es negro.
    # Los valores de las variables las tomamos por default en negro
    def glClear(self):
        self.framebuffer = [
            [
                color(self.RedScreen, self.GreenScreen, self.BlueScreen) for x in range(self.width)
            ]
            for y in range(self.height)
        ]

        self.zbuffer = [
            [-float('inf')for x in range(self.width)]
            for y in range(self.height)
        ]

    # Este método le cambia el color al glclear()
    def glClearColor(self, red, green, blue):
        if (green <= 1 and red <= 1 and blue <= 1) and (green >= 0 and red >= 0 and blue >= 0):
                # iniciamos la conversión, es una regla de tres
                # 1=255 entonces (r,g,b)=Conversion entre 255
            self.RedScreen = int((red*255)/1)
            self.GreenScreen = int((green*255)/1)
            self.BlueScreen = int((blue*255)/1)
            # le cambiamos valores a las variables y las parseamos
        else:
            # Imprimimos un error por si eso pasa
            print("ERROR, Ingrese valores entre 0 y 1 para el glClearColor(r,g,b)")

    # Método para colocar los valores del Viewport
    def glCreateViewPort(self, Centerx, CenterY, Width, Height):
        try:
            if(Width > self.width) or (Height > self.height):
                print("No puede crear un ViewPort mas grande que la pantalla")
            else:
                # Estas serán las variables que tendremos para usar en el Viewport
                self.ViewPX = Centerx
                self.ViewPY = CenterY
                # Dividimos dentro de 2 el valor de la altura, puesto que usaremos esos valores para movernos
                self.ViewpWidth = Width
                self.ViewpHeight = Height
        # Creamos una excepción por cualquier asunto.
        except:
            print("Hubo un error, revisa las coordenadas del Viewport, su centro o que el tamaño no exceda la pantalla.")

    # Ahora creamos el Vertex()
    def glVertex(self, x, y):
        self.VertexX = x*self.ViewpWidth/2
        self.VertexY = y*self.ViewpHeight/2
        self.framebuffer[self.ViewPY+int(self.ViewpHeight/2)+int(self.VertexY)][self.ViewPX+int(
            self.ViewpWidth/2)+int(self.VertexX)] = color(self.RedVertex, self.GreenVertex, self.BlueVertex)

    # Método para cambiar el color del glVertex()
    def glColor(self, red, green, blue):
        if (green <= 1 and red <= 1 and blue <= 1) and (green >= 0 and red >= 0 and blue >= 0):
            # iniciamos la conversión, es una regla de tres
            # 1=255 entonces (r,g,b)=Conversion entre 255
            self.RedVertex = int((red*255)/1)
            self.GreenVertex = int((green*255)/1)
            self.BlueVertex = int((blue*255)/1)
            # le cambiamos valores a las variables y las parseamos
        else:
            # Imprimimos un error por si eso pasa
            print("ERROR, Ingrese valores entre 0 y 1 para el glColor(r,g,b)")

    # algoritmo para pintar una figura
    def pintarFigura(self, Poligono):
        # declaramos un map
        mapeado = []
        ingreso = 0
        # convertimos el vector entrante en un vector de coordenadas X,Y
        for x in range(int(len(Poligono)/2)):
            mapeado.append((Poligono[ingreso], Poligono[ingreso+1]))
            ingreso = ingreso+2
        # ahora iteramos el ancho o el alto del viewport
        for x in range(self.ViewpWidth):
            for y in range(self.ViewpHeight):
                # si el algoritmo de poly retorna que está dentro del área, entonces...
                if(self.cn_PnPoly((x, y), mapeado) == 1):
                    # pintamos ese lugar.
                    self.glVertex(self.traductorx(x), self.traductory(y))

        # cn_PnPoly(): crossing number test for a point in a polygon
    #     Input:  P = a point,
    #             V[] = vertex points of a polygon
    #     Return: 0 = outside, 1 = inside
    # This code is patterned after [Franklin, 2000]

    def cn_PnPoly(self, P, V):
        cn = 0    # the crossing number counter

        # repeat the first vertex at end
        V = tuple(V[:])+(V[0],)

        # loop through all edges of the polygon
        for i in range(len(V)-1):   # edge from V[i] to V[i+1]
            if ((V[i][1] <= P[1] and V[i+1][1] > P[1])   # an upward crossing
                    or (V[i][1] > P[1] and V[i+1][1] <= P[1])):  # a downward crossing
                # compute the actual edge-ray intersect x-coordinate
                vt = (P[1] - V[i][1]) / float(V[i+1][1] - V[i][1])
                if P[0] < V[i][0] + vt * (V[i+1][0] - V[i][0]):  # P[0] < intersect
                    cn += 1  # a valid crossing of y=P[1] right of P[0]

        return cn % 2   # 0 if even (out), and 1 if odd (in)

    # ===================================================================

    def sum(self, v0, v1):
        return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

    def sub(self, v0, v1):
        return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

    def mul(self, v0, k):
        return V3(v0.x * k, v0.y * k, v0.z * k)

    def dot(self, v0, v1):
        return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

    # este método calcula el producto cruz
    def cross(self, v0, v1):
        return V3(
            v0.y * v1.z - v0.z * v1.y,
            v0.z * v1.x - v0.x * v1.z,
            v0.x * v1.y - v0.y * v1.x
        )

    def length(self, v0):
        return (v0.x**2 + v0.y**2 + v0.z**2)**0.5

    def norm(self, v0):
        # variable de la longitud
        l = self.length(v0)
        # si no tenemos longitud
        if not l:
            return V3(0, 0, 0)
        # se retorna
        return V3(v0.x/l, v0.y/l, v0.z/l)

    # método de LineSweeping
    def triangle(self, A, B, C, color=None):
        # se reordenan los valores
        if A.y > B.y:
            A, B = B, A
        if A.y > C.y:
            A, C = C, A
        if B.y > C.y:
            B, C = C, B

        dx_ac = C.x - A.x
        dy_ac = C.y - A.y
        # Esta parte es importante, si es 0 el denominador se sale
        if dy_ac == 0:
            return
            # ahora si se calcula la pendiente
        M_ac = dx_ac/dy_ac
        # se calcula el otro triángulo
        dx_ab = B.x - A.x
        dy_ab = B.y - A.y
        # de nuevo, si el denominador es cero, ni entra al if
        if (dy_ab != 0):
            # se calcula la pendiente
            M_ab = dx_ab/dy_ab
            for y in range(A.y, B.y + 1):
                # se usa la ecuación de la recta
                xi = round(A.x - M_ac * (A.y - y))
                xf = round(A.x - M_ab * (A.y - y))
                # operación más costosa, el ordenamiento
                if xi > xf:
                    xi, xf = xf, xi
                    # Acá se recorre la longitd
                for x in range(xi, xf + 1):
                    # se llama a point color y se pinta
                    self.pointColor(x, y, color)
        # Se repite el proceso con el otro triángulo
        dx_bc = C.x - B.x
        dy_bc = C.y - B.y
        if (dy_bc != 0):
            M_bc = dx_bc/dy_bc
            for y in range(B.y, C.y + 1):
                xi = round(A.x - M_ac * (A.y - y))
                xf = round(B.x - M_bc * (B.y - y))

                if xi > xf:
                    xi, xf = xf, xi
                for x in range(xi, xf + 1):
                    self.pointColor(x, y, color)

    # Creamos un método bbox() para encerrar el triángulo
    def bbox(self, A, B, C):
        xs = sorted([A.x, B.x, C.x])
        ys = sorted([A.y, B.y, C.y])

        return V2(xs[0], ys[0]), V2(xs[2], ys[2])

    #método de barycentric para poder usarse en el cálculo del triangulo
    def barycentric(self, A, B, C, P):
        cx, cy, cz = self.cross(
            V3(B.x - A.x, C.x - A.x, A.x-P.x),
            V3(B.y - A.y, C.y - A.y, A.y-P.y)
        )
        # si el triángulo no tiene área
        if cz == 0:
            return -1, -1, -1
        # se calculan las coordenadas
        u = cx / cz
        v = cy/cz
        w = 1 - (u + v)

        return w, v, u
    # este es el método de barycentric

    def triangleBarycentric(self, A, B, C, color=color, texture=None, texture_coords=(), intensity=1,normales=(),mtl=""):
        bbox_min, bbox_max = self.bbox(A, B, C)
        #Para cada obj, se tiene una luz diferente, para no tener solo una, se envía el nombre del MTL y se lee y se coloca la luz para ese.
        if(mtl=="patrick.mtl"):
            light=V3(0,0,1)
        elif(mtl=="Spongebob.mtl"):
            light=V3(1,0,0)
        elif(mtl=="squidward.mtl"):
            light=V3(0,0,1)
        elif(mtl=="mrkrabsnude.mtl"):
            light=V3(0,0,1)
 
        # este for recorre eel bbox osea la cajita
        for x in range(bbox_min.x, bbox_max.x + 1):
            for y in range(bbox_min.y, bbox_max.y+1):
                w, v, u = self.barycentric(A, B, C, V2(x, y))
                # condición para las coordenaas
                if w < 0 or v < 0 or u < 0:
                    continue
                if texture:
                    #si existe textura
                    if(mtl=="mrkrabsnude.mtl"):
                        #en este if, le colocamos shader al obj de don cangrejo, está desnudo, por lo que le pondremos pantalón azul, camisa verde
                        #y un random para que cubra el resto del cuerpo
                        #Acá, hacemos un if con las coordenadas del triangulo, es decir con la altura que es Y
                        if 450 < (y + (4* sin(120 + x/50))) and (y+ (2 * sin(140 + x/50))) < 480:
                                tcolor = (255,25,24) 
    
                        elif 481 < (y + (2* sin(140 + x/50))) and (y + (2 * sin(160 + x/50))) < 500:
                                tcolor = (6,255,24)
                            
                        elif 501 < (y + (2* sin(160 + x/50))) and (y + (4 * sin(180 + x/50))) < 502:
                                tcolor = (randint(0,255),randint(0,255),randint(0,255))
                            
                        elif 503 < (y + (4* sin(180 + x/50))) and (y + (2 * sin(200 + x/50))) < 510:
                                tcolor = (randint(0,255),randint(0,255),randint(0,255))
                            
                        elif 511 < (y + (2* sin(200 + x/50))) and y < 520:
                                tcolor = (randint(0,255),randint(0,255),randint(0,255))
                        else:
                            tcolor=(randint(0,255),randint(0,255),randint(0,255))  
                        #después, jalamos las normales, y las multiplicamos por la luz que le corresponde, se colocan en tres coordenadas
                        iA, iB, iC = [ dot(n, light) for n in normales]
                        #la intensidad es la multiplicación de esas tres coordenadas
                        intensity = w*iA + v*iB + u*iC
                        #el color es la intensidad por lambda b que adquiere el valor de tcolor
                        color=bytes(map(lambda b: int(round(b*(intensity))) if b*intensity > 0 else 0, tcolor))
                    else:
                        #si no es don cangrejo, entonces hacemos shaders coon las texturas
                        ta, tb, tc = texture_coords
                        #desúés de obtener las corrdenadas, obtenemos tx y ty
                        tx = ta.x*w + tb.x*v + tc.x*u
                        ty = ta.y*w + tb.y*v + tc.y*u                
                        #invocamos a get color para obtener el color en la textura
                        color=texture.get_color(tx,ty)
                        #con las normales hallamos las tres coordenadas
                        iA, iB, iC = [ dot(n, light) for n in normales]
                        intensity = w*iA + v*iB + u*iC
                        #tenemos la intensidad y luego multiplicamos por el color
                        color=bytes(map(lambda b: int(round(b*intensity)) if b*intensity > 0 else 0, color))
                #el z es la multiplicacion y suma de las coordenadas
                z = A.z * w + B.z*v + C.z * u
                #un continue
                if x<0 or y<0:
                    continue
                #if para ver que pinte correctamente
                if x < len(self.zbuffer) and y < len(self.zbuffer[x]) and z > self.zbuffer[x][y]:
                    self.pointColor(x, y, color)
                    self.zbuffer[x][y] = z

        ########################################################################################################
    # FUNCIONES EXTRA INTERESANTES"""""""""""""""""""""
    # crearemos un método para pintar los polígonos

    # este point es para pintar

    def point(self, x, y):
        self.framebuffer[y][x] = color(
            self.RedPoint, self.GreenPoint, self.BluePoint)

    def pointColor(self, x, y, color):
        self.framebuffer[y][x] = color

    # Este traductor es para cambiar
    def traductorx(self, x):
        resultado = 2*((x-self.ViewPX)/self.ViewpWidth)-1
        return resultado
    # este es un traductor y

    def traductory(self, y):
        resultado = 2*((y-self.ViewPY)/self.ViewpHeight)-1
        return resultado
    # este decodificador codifica

    def decodificadorX(self, x):
        resultado = (((x+1)/2)*self.ViewpWidth)+self.ViewPX
        return resultado

    def decodificadorY(self, y):
        resultado = (((y+1)/2)*self.ViewpHeight)+self.ViewPY
        return resultado

    # este método cambia el color del Point

    def glColorPoint(self, R, G, B):
        self.RedPoint = R
        self.BluePoint = B
        self.GreenPoint = G

    # ESte método es para hacer una línea horizontal a la derecha
    def lineHD(self, x, y, width, color):
        # Ahora empezamos en la coordenada que querramos
        for i in range(width):
            # iteramos la coordenada inicial y le sumamos el width
            self.point(x+i, y, color)

    # ESte método es para hacer una línea horizontal a la izquierda
    def lineHI(self, x, y, width, color):
        # Ahora empezamos en la coordenada que querramos
        for i in range(width):
            # iteramos la coordenada inicial y le restamos el width
            self.point(x-i, y, color)

    # Este método es para hacer una linea vertical hacia arriba.
    def lineVUp(self, x, y, height, color):
        # ahora empezamos en la coordenada que querramos
        for i in range(height):
            self.point(x, y+i, color)

    # Este método es para hacer una linea vertical hacia abajo.
    def lineVDown(self, x, y, height, color):
        # ahora empezamos en la coordenada que querramos
        for i in range(height):
            self.point(x, y-i, color)

    # Este método coloca un cuadrado genérico de X y Y dimensiones
    def rectangulo(self, xx, yy, width, height, color):
        for x in range(height):
            self.lineHD(xx, yy-x, width, color)

    # Esta funcion es un range(start, stop, step) para floats.
    # Método adquirido de Internet
    # Link:
    def frange(self, start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

    # Funcion para hacer un cubitronix
    def Cubo_center(self, size):
        # Este será  el tamaño del cubo, le sacamos raiz cuadrada
        tamaño = sqrt(size)
        corrimiento = tamaño/2
        coordenada = corrimiento/(self.ViewpWidth/2)
        # Pintamos la linea de arriba y abajo
        for x in range(int(corrimiento*2)):
            movimiento = x/(self.ViewpWidth/2)
            self.glVertex(coordenada-movimiento, coordenada)
            self.glVertex(coordenada-movimiento, -1*coordenada)

        # Pintamos la linea de izquierda y derecha
        for y in range(int(corrimiento*2)):
            movimiento = y/(self.ViewpWidth/2)
            self.glVertex(-1*coordenada, (-1*coordenada)+movimiento)
            self.glVertex(coordenada, (-1*coordenada)+movimiento)

    # método para pintar en las orillas
    def pintar_orillas(self):
        # Pintamos la linea de arriba y abajo
        for x in range(int(self.ViewpWidth*2)):
            movimiento = x/(self.ViewpWidth)
            self.glVertex(1-movimiento, 1)
            self.glVertex(1.0-movimiento, -1.0)

        # Pintamos la linea de izquierda y derecha
        for y in range(int(self.ViewpHeight*2)):
            movimiento = y/(self.ViewpHeight)
            self.glVertex(-1, -1+movimiento)
            self.glVertex(1, -1.0+movimiento)

    # método para pintar la diagonal en la mitad de la pantalla
    def diagonal_center(self):
        # Pintamos en los puntos
        for x in range(int(self.ViewpWidth*2)):
            corrimiento = x/(self.ViewpWidth)
            self.glVertex(-1.0+corrimiento, 1.0-corrimiento)

    # metodo para hacer un conjunto de puntos aleatorios
    def pantalla_gris(self):
        for x in (self.frange(-1, 1, 1/(self.ViewpWidth))):
            for y in (self.frange(-1, 1, 1/(self.ViewpHeight))):
                if randint(0, 1) == 1:
                    self.glColor(1, 1, 1)
                elif(randint(0, 1) == 0):
                    self.glColor(0, 0, 0)
                self.glVertex(x, y)

    # método para hacer una pantalla llea de colores random
    def pantalla_random(self):
        for x in (self.frange(-1, 1, 1/(self.ViewpWidth))):
            for y in (self.frange(-1, 1, 1/(self.ViewpHeight))):
                self.glColor(uniform(0, 1), uniform(0, 1), uniform(0, 1))
                self.glVertex(x, y)

    # hacemos el cielo de estrellas
    def pantalla_estrellada(self):
        for x in range(int(randint(500, 525))):
            random1 = uniform(-1, 1)
            random2 = uniform(-1, 1)
            random3 = randint(0, 9)
            if(random3 == 3):
                self.glVertex(random1, random2)
                for w in (self.frange(0, 0.003, 0.0001)):
                    self.glVertex(random1+w, random2)
                    self.glVertex(random1-w, random2)
                    self.glVertex(random1, random2-w)
                    self.glVertex(random1, random2+w)
                    self.glVertex(random1+w, random2+w)
                    self.glVertex(random1-w, random2+w)
                    self.glVertex(random1+w, random2-w)
                    self.glVertex(random1-w, random2-w)
            else:
                self.glVertex(random1, random2)

    ########################################################################################################
    # Funcion de finish, estos parámetros fueron vistos en clase

    def glFinish(self, filename):
        f = open(filename, "wb")
        # estandar
        f.write(char('B'))
        f.write(char('M'))
        # file size0
        f.write(dword(14 + 40 + self.width * self.height * 3))
        # reserved
        f.write(dword(0))
        # data offset
        f.write(dword(54))
        # header size
        f.write(dword(40))
        # width
        f.write(dword(self.width))
        # height
        f.write(dword(self.height))
        # planes
        f.write(word(1))
        # bits per pixelxdf
        f.write(word(24))
        # compression
        f.write(dword(0))
        # image size
        f.write(dword(self.width * self.height * 3))
        # x pixels per meter
        f.write(dword(0))
        # y pixels per meter
        f.write(dword(0))
        # number of colors
        f.write(dword(0))
        # important colors
        f.write(dword(0))
        # image data
        for x in range(self.height):
            for y in range(self.width):
                    f.write(self.framebuffer[x][y])
        # close file
        f.close()


bitmap = Bitmap()
# Creamos una imagen de 1000x10000 por default negra
bitmap.glCreateWindow(1000, 1000)
# le damos clear para que se vuelva blanca
bitmap.glClear()
# Cambiamos los valroes de Clear()
bitmap.glClearColor(0.5, 0.5, 0.5)
# Volvemos a pintar la pantalla
bitmap.glClear()
# Ahora cambiamos el color de los puntos del vertex() a blanco
bitmap.glColor(1, 1, 1)
bitmap.glColorPoint(255, 255, 255)
# Creamos un Viewporte de 999x999
bitmap.glCreateViewPort(0, 0, 1000, 1000)
#definicion de lookAt
bitmap.lookAt(V3(0,0,0), V3(0,0,-100 ), V3(0, 1, 0),translate=(0,0,0))
#definicion de la textura y de las demás
texturitaPatricio = Texture("Char_Patrick.bmp")
texturaBob=Texture("SpongeSkin_0.bmp")
bitmap.background_texture("BikiniBottom.bmp")
texturaCalamardo=Texture("calamardo.bmp")
texturaDonCangrejo=Texture("donCangrejo.bmp")
#defincion de triángulos
bitmap.loadTriangulos("patrick.obj", translate=(50, 150, 0), scale=(0.3, 0.3,0.3),texture=texturitaPatricio,rotate=(0,-0.5,0),mtl="patrick.mtl")
bitmap.loadTriangulos("Spongebob.obj", translate=(900, 150, 0), scale=(0.3, 0.3,0.3),texture=texturaBob,rotate=(0,-1.8,0),mtl="Spongebob.mtl")
bitmap.loadTriangulos("squidward.obj", translate=(250, 100, 0), scale=(0.3, 0.3,0.3),texture=texturaCalamardo,rotate=(0,-0.3,0),mtl="squidward.mtl")
bitmap.loadTriangulos("gary.obj", translate=(500, 50, 0), scale=(0.12, 0.12,0.12),texture=None,rotate=(0,-0.2,0),mtl="gary.mtl")
bitmap.loadTriangulos("mrkrabsnude.obj", translate=(570, 375, 0), scale=(0.11, 0.11,0.11),texture=texturaDonCangrejo,rotate=(0,-0.2,0),mtl="mrkrabsnude.mtl")
#creamos el BMP
bitmap.glFinish("escena.bmp") 