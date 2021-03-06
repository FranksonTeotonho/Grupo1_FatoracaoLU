import pygame, os, sys, time
import numpy as np

np.seterr(all='raise')
import warnings
#warnings.filterwarnings("error")

# Inicializando pygame
from numpy.testing.nose_tools.parameterized import param

pygame.init()


# Faz o pivoteamento recebendo a matriz A, o vetor resposta V
# a matriz de pivoteamento P e o índice da coluna
def Pivoting(A, V, P, L, iter):
    Maior = abs(A[iter, iter])
    MaiorI = iter

    for i in range(iter, int(np.sqrt(A.size))):  # Encontra o maior elemento da coluna
        if (Maior < abs(A[i, iter])):
            Maior = abs(A[i, iter])
            MaiorI = i

    P[MaiorI, iter] = 1  # Coloca 1 no local correto da matriz de permutação

    auxiliary_vector = V[iter]  # Permuta o vetor resposta
    V[iter] = V[MaiorI]
    V[MaiorI] = auxiliary_vector

    auxiliary_lower = L[iter, :iter].copy()  # permuta matriz L
    L[iter, :iter] = L[MaiorI, :iter]
    L[MaiorI, :iter] = auxiliary_lower

    auxiliary_matrix = A[iter, iter:].copy()  # Permuta a matriz A
    A[iter, iter:] = A[MaiorI, iter:]
    A[MaiorI, iter:] = auxiliary_matrix

    return A, V, P, L  # Retorna a matriz A, o vetor resposta e a matriz de permutação


# Recebe a matriz, o vetor resposta V, a matriz de coeficientes L e a coluna desejada
def LUGaussElimination(A, L, iter, passosA, passosL,DZ):
    for i in range(iter + 1, int(np.sqrt(A.size))):
        try:  # Verifica se o pivot é zero
            Aux = A[i, iter] / A[iter, iter]
            L[i, iter] = Aux  # Muda a matriz L com o modificador
            A[i, iter] = 0  # Elimina elementos da coluna menos a diagonal
        except FloatingPointError:
            print("Matriz inválida")  # Joga exceção se for zero
            DZ = -1
            return A,L, passosA, passosL, DZ

        for j in range(iter + 1, int(np.sqrt(A.size))):  # Calcula o multiplicador sobre as linhas
            A[i, j] = A[i, j] - A[iter, j] * L[i, iter]  # Multiplica as linhas

        passosA.append(A.copy())
        passosL.append(L.copy())

    return A, L, passosA, passosL, DZ  # Retorna  matriz atual, vetor resposta e matriz L


# Cria uma matriz de zeros
def ZeroMatrix(Tam):
    # Cria uma string que inicializa uma matri zero
    Zero = ""
    for i in range(0, Tam):
        for j in range(0, Tam):
            Zero = Zero + "0 "
        if (i != Tam - 1):
            Zero = Zero + " ; "

    return np.matrix(Zero).astype(np.float64)  # Retorna a matriz zerada


# Cria uma matriz identidade
def IdMatrix(Tam):
    M = ZeroMatrix(Tam)
    for i in range(0, Tam):
        M[i, i] = 1

    return M  # Retorna a matriz identidade


# Faz a Fatoração LU
def LUDecomposition(A, V, L, P, passosA, passosL,DZ):
    for i in range(0, int(np.sqrt(A.size))):  # Para cada coluna da matriz nxn
        A, V, P, L = Pivoting(A, V, P, L, i)  # Faz o pivoteamento de linhas
        A, L, passosA, passosL, DZ = LUGaussElimination(A, L, i, passosA, passosL,DZ)  # Faz a fatoração LU na Coluna

    return A, V, L, P, passosA, passosL, DZ  # Retorna a fatoração


# Verifica se uma matriz não possui linhas zeradas
def MatrixSolvability(A):
    Tam = int(np.sqrt(A.size))  # Pega o tamanho n da matriz A
    Zeros = 0  # Número de zeros acumulados em uma linha
    for i in range(0, Tam):  # Para cada linha
        Zeros = 0  # Reseta o contador
        for j in range(0, Tam):  # Para cada coluna
            if (A[i, j] == 0):  # Verifica se valor é 0
                Zeros = Zeros + 1  # Se for, aumenta o contador +1
        if (Zeros == Tam):  # Se tiver zeros em toda a linha
            return False  # Retorna que a matriz não tem solução única

    return True  # Retorna que existe solução única


def SolveLower(L, V, Tamanho):
    Respostas = []
    for i in range(0, Tamanho):  # para cada linha da matriz L
        subtract_auxiliary = 0
        for x in range(0, i):  # subtrai y passados * os valores a esquerda da coluna da linha sendo iterada
            subtract_auxiliary = subtract_auxiliary + Respostas[x] * L[i, x]
        Respostas.append((V[i] - subtract_auxiliary) / L[i, i])
    return Respostas


def SolveUpper(A, Respostas, Tamanho):
    Final = [0] * Tamanho
    for i in reversed(range(0, Tamanho)):  # para cada linha da matriz A
        subtract_auxiliary = 0
        for x in reversed(range(i + 1, Tamanho)):
            subtract_auxiliary = subtract_auxiliary + Final[x] * A[i, x]
        Final[i] = (Respostas[i] - subtract_auxiliary) / A[i, i]
    return Final


def SolveSystem(A, L, V, SU):
    if (not MatrixSolvability(A)):  # Verifica se matriz possui solução única
        print("Matriz A não possui solução única")  # Mostra mensagem de erro
        SU = -1
        return  None, None, SU
    Tamanho = int(np.sqrt(A.size))  # Pega o tamanho n da matriz A
    Respostas = SolveLower(L, V, Tamanho)
    return Respostas ,SolveUpper(A, Respostas, Tamanho), SU


########################################### INTERFACE GRAFICA ################################

# Função responsavel por carregar imagens
def load_png(name):
    # Resource
    fullname = os.path.join('asserts', name)

    try:
        image = pygame.image.load(fullname)
        if image.get_alpha() is None:
            image = image.convert()
        else:
            image = image.convert_alpha()
    except pygame.error as message:
        print('Cannot load image:', fullname)
        pass

    return image, image.get_rect()


class FatoracaoLU:
    # Fontes
    # Tela inicial
    msgTelaInicial = pygame.font.Font(None, 25)

    # Loading Imagens
    # Imagem de background inicial
    backgroundInicialImage, backgroundInicialImageRect = load_png("Background_inicial.jpg")
    # Imagem de backgroud default
    backgroundImage, backgroundImageRect = load_png("BG.png")
    # Imagem para exibirResultados
    resultadoImage, resultadoImageRect = load_png("Background_default_sb.jpg")
    # Imagem para passos
    passosImage, passosImageRect = load_png("passos.png")
    # Imagem para exibirResultadosFinais
    resultadoFinalImage, resultadoFinalImageRect = load_png("ResultadoFinal.png")

    def __init__(self):
        # Inicia screen
        self.screen = pygame.display.set_mode((800, 600))
        # Titulo da screen
        pygame.display.set_caption('Fatoração LU')

        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.startScreen()
        self.resetScreen()

        # Blit everything to the screen
        self.render()

    def render(self):
        pygame.display.flip()

    def resetScreen(self):
        self.screen.blit(self.background, (0, 0))

    def startScreen(self):
        # Tela background_inicial
        self.background.blit(self.backgroundInicialImage, self.backgroundInicialImageRect)
        # Texto matriz 3x3
        text = self.msgTelaInicial.render("Pressione 3 para iniciar matriz 3x3", 1, (10, 10, 10))
        self.background.blit(text, (180, 460))
        # Texto matriz 4x4
        text = self.msgTelaInicial.render("Pressione 4 para iniciar matriz 4x4", 1, (10, 10, 10))
        self.background.blit(text, (150, 505))

    def defaultScreen(self, string):
        self.background.blit(self.backgroundImage, self.backgroundImageRect)
        numero = pygame.font.Font(None, 50)
        label = numero.render(string, 1, (0, 0, 0))
        self.background.blit(label, (315, 90))

    def addLabel(self, string, posicao, tam):
        lb = pygame.font.Font(None, tam)
        label = lb.render(string, 1, (105, 105, 105))
        self.background.blit(label, posicao)

    def resultadoScreen(self):
        self.background.blit(self.resultadoImage, self.resultadoImageRect)
        self.addLabel("Resultados",(220,300),100)

    def warningDZScreen(self):
        self.background.blit(self.resultadoImage, self.resultadoImageRect)
        self.addLabel("Erro: Divisão por zero",(220,300),50)

    def warningSUScreen(self):
        self.background.blit(self.resultadoImage, self.resultadoImageRect)
        self.addLabel("Eroo: A solução não é única",(220,300),50)

    def passosScreen(self):
        self.background.blit(self.passosImage, self.passosImageRect)
        numero = pygame.font.Font(None, 50)
        label = numero.render("Passos", 1, (255, 215, 0))
        self.background.blit(label, (335, 90))

    def finalScreen(self):
        self.background.blit(self.resultadoFinalImage, self.resultadoFinalImageRect)
        numero = pygame.font.Font(None, 50)
        label = numero.render("Resultado", 1, (0, 128, 0))
        self.background.blit(label, (320, 90))

    def exibeMatriz(self, matriz, xi, yi, xe, ye, tam):

        x = xi
        y = yi

        for i in matriz:
            i = np.array(i)
            for j in i[0]:
                self.addLabel(str(round(j,3)), (x, y), tam)
                x = x + xe

            x = xi
            y = y + ye

    def exibeVetor(self, vetor, xi, yi, ye, tam):

        y = yi
        for i in vetor:
            self.addLabel(str(round(i,3)), (xi, y), tam)
            y = y +ye

    def getDigito(self, n, xi, yi, xe, ye, tam, ordem):
        A = []
        V = []
        string = []

        i = 0
        x = xi
        y = yi
        # Ordem da matriz + V
        l = ordem + 1

        while i < n:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_0:
                        string.append("0")
                    if event.key == pygame.K_1:
                        string.append("1")
                    if event.key == pygame.K_2:
                        string.append("2")
                    if event.key == pygame.K_3:
                        string.append("3")
                    if event.key == pygame.K_4:
                        string.append("4")
                    if event.key == pygame.K_5:
                        string.append("5")
                    if event.key == pygame.K_6:
                        string.append("6")
                    if event.key == pygame.K_7:
                        string.append("7")
                    if event.key == pygame.K_8:
                        string.append("8")
                    if event.key == pygame.K_9:
                        string.append("9")
                    if event.key == pygame.K_PERIOD:
                        string.append(".")
                    if event.key == pygame.K_MINUS:
                        string.append("-")
                    if event.key == pygame.K_RETURN:
                        # Transformando em string e adicionando na matriz
                        string = ''.join(string)
                        if (i % l == l - 1):
                            V.append(string)
                        else:
                            A.append(string)

                        # Desenhando o novo numero na tela
                        self.addLabel(string, (x, y), tam)
                        self.resetScreen()
                        self.render()
                        # Limpando cache
                        string = []

                        # Atualizando posições
                        x = x + xe
                        if (i + 1) % l == 0:
                            x = xi
                            y = y + ye

                        # Contador
                        i = i + 1

                    if event.key == pygame.K_ESCAPE:
                        return A, V
        return A, V

    def matriz33(self):
        passosA = []
        passosL = []
        SU = 1
        DZ = 1
        A, V = self.getDigito(12, 210, 200, 110, 100, 50, 3)

        if (len(A) == 9) and (len(V) == 3):
            print("Todos os inputs adicionados")
            #Calcula resultados
            A = np.matrix(A).astype(np.float64)  # Aceita a matriz de entrada
            A = A.reshape((3, 3))
            Tam = int(np.sqrt(A.size))  # Calcula o n da matrix nxn

            V = [np.float64(x) for x in V]  # Transforma a entrada de String para Float

            L = IdMatrix(Tam)  # Faz a matriz Identidade em L
            P = ZeroMatrix(Tam)  # Inicia a matriz de Permutações

            A, V, L, P, passosA, passosL, DZ = LUDecomposition(A, V, L, P, passosA, passosL,DZ)  # Faz a fatoração LU
            b = V #Mudança de nome

            if(DZ != -1):

                y, x, SU = SolveSystem(A, L, V,SU)  # Resolve o sistema
                # Timer
                time.sleep(1)

                if (SU == 1):

                    #Tela de apresentação de resultados
                    self.resultadoScreen()
                    self.resetScreen()
                    self.render()

                    time.sleep(1)
                    for i in range(len(passosA)):

                        #Exibindo passos
                        self.passosScreen()
                        #Matriz A (U)
                        self.exibeMatriz(passosA[i], 110, 270,100,50,50)
                        #Matriz L
                        self.exibeMatriz(passosL[i], 450, 270,100,50,50)
                        self.resetScreen()
                        self.render()
                        #Timer
                        time.sleep(1.5)

                    #Resultado final
                    self.finalScreen()
                    # Matriz A (U)
                    self.exibeMatriz(passosA[len(passosA)-1], 140, 210,75,40,35)
                    # Matriz L
                    self.exibeMatriz(passosL[len(passosL)-1], 460, 210,75,40,35)
                    #vetor b
                    self.exibeVetor(b,165,375,40,45)
                    #vetor y
                    self.exibeVetor(y, 390, 375, 40, 45)
                    #vetor x
                    self.exibeVetor(x, 600, 375, 40, 45)
                    self.resetScreen()
                    self.render()
                else:
                    self.warningSUScreen()
                    self.resetScreen()
                    self.render()
            else:
                self.warningDZScreen()
                self.resetScreen()
                self.render()
        else:
            print("Erro ao adicionar todos os inputs")



    def matriz44(self):
        passosA = []
        passosL = []
        SU = 1
        DZ = 1
        A, V = self.getDigito(20, 210, 200, 80, 70, 40, 4)

        if (len(A) == 16) and (len(V) == 4):
            print("Matriz completa")

            print("Todos os inputs adicionados")
            # Calcula resultados
            A = np.matrix(A).astype(np.float64)  # Aceita a matriz de entrada
            A = A.reshape((4, 4))
            Tam = int(np.sqrt(A.size))  # Calcula o n da matrix nxn

            V = [np.float64(x) for x in V]  # Transforma a entrada de String para Float

            L = IdMatrix(Tam)  # Faz a matriz Identidade em L
            P = ZeroMatrix(Tam)  # Inicia a matriz de Permutações

            A, V, L, P, passosA, passosL, DZ = LUDecomposition(A, V, L, P, passosA, passosL,DZ)  # Faz a fatoração LU
            b = V  # Mudança de nome
            if(DZ == 1):
                y, x, SU = SolveSystem(A, L, V, SU)  # Resolve o sistema
                # Timer
                time.sleep(1)

                if(SU == 1):


                    # Tela de apresentação de resultados
                    self.resultadoScreen()
                    self.resetScreen()
                    self.render()

                    time.sleep(1)
                    for i in range(len(passosA)):
                        # Exibindo passos
                        self.passosScreen()
                        # Matriz A (U)
                        self.exibeMatriz(passosA[i], 120, 255, 70, 45, 25)
                        # Matriz L
                        self.exibeMatriz(passosL[i], 455, 255, 70, 45, 25)
                        self.resetScreen()
                        self.render()
                        # Timer
                        time.sleep(1.5)

                    # Resultado final
                    self.finalScreen()
                    # Matriz A (U)
                    self.exibeMatriz(passosA[len(passosA) - 1], 140, 205, 55, 30, 25)
                    # Matriz L
                    self.exibeMatriz(passosL[len(passosL) - 1], 460, 205, 55, 30, 25)
                    # vetor b
                    self.exibeVetor(b, 165, 375, 30, 30)
                    # vetor y
                    self.exibeVetor(y, 390, 375, 30, 30)
                    # vetor x
                    self.exibeVetor(x, 600, 375, 30, 30)
                    self.resetScreen()
                    self.render()

                else:
                    self.warningSUScreen()
                    self.resetScreen()
                    self.render()

            else:
                self.warningDZScreen()
                self.resetScreen()
                self.render()

        else:
            print("Matriz incompleta")


def main():
    # Objeto grafico
    f = FatoracaoLU()

    # event loop
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_3:
                    f.defaultScreen("Matriz 3x3")
                    f.resetScreen()
                    f.render()
                    f.matriz33()
                if event.key == pygame.K_4:
                    f.defaultScreen("Matriz 4x4")
                    f.resetScreen()
                    f.render()
                    f.matriz44()
                if event.key == pygame.K_ESCAPE:
                    f.startScreen()
                    f.resetScreen()
                    f.render()


if __name__ == '__main__':
    main()