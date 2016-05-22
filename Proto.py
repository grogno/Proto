import pygame
import numpy as np
import numba as nb
import math as math

from pygame.locals import *
import matplotlib.pyplot as plt

#Каждая частица принадлежит классу Particle.
#Здесь содержится ее местоположение и скорость (вместе они называются coordinates),
#а также цвет и размер частиц для визуализации
class Particle:
    def __init__(self, coordinates, size=2, mass=1):
        self.colour = (0, 0, 255)
        self.size = size
        self.mass = 1
        self.coordinates = coordinates           
    def get_coordinates(self):
        return self.coordinates    
    def set_coordinates(self, values):
        self.coordinates = values

#Эта функция вычисляет массив действующих на частицы сил, исходя из массива их местоположения.
#На частицу действуют сила гравитации (F = G/r) и межмолекулярные силы, равные производной потенциала Морзе по r.
#Потенциал Морзе характеризуется параметрами a("жесткость" потенциала) и равновесным расстоянием r0.
@nb.jit(nopython=True) #Декоратор из модуля numba комилирует эту функцию на ходу, обеспечивая прирост производительности.
def get_forces(positions):
    n_particles = positions.shape[0]
    forces =  np.zeros(positions.shape)
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            dx = positions[j,0] - positions[i,0]
            dy = positions[j,1] - positions[i,1]             
            dr = (dx * dx + dy * dy)**0.5
            ex = dx/dr
            ey = dy/dr
            if dr>0.5: #Межмолекулярные взаимодействия на больших расстояниях обрезаются
                force = G/dr
            else:
                force = G/dr**2-2*a*math.exp(a*(r0-dr))*(math.exp(a*(r0-dr))-1)
            forces[i, 0] += ex * force
            forces[i, 1] += ey * force
            forces[j, 0] -= ex * force
            forces[j, 1] -= ey * force
    return(forces)

#Эта функция вычисляет кинетическую и потенциальную энергию системы в данной конфигурации.
@nb.jit(nopython=True)
def get_energy(positions, velocities):
    n_particles = positions.shape[0]
    kinetic_energy = 0
    gravitational_energy = 0
    intermolecular_energy = 0
    for i in range(n_particles):
        kinetic_energy += 0.5*np.sum(velocities[i]**2)
        for j in range(i+1, n_particles):
            dx = positions[j,0] - positions[i,0]
            dy = positions[j,1] - positions[i,1]             
            dr = (dx * dx + dy * dy)**0.5
            gravitational_energy += G*math.log(dr)
            intermolecular_energy += (1-math.exp(-a*(dr-r0)))**2-1
    return (kinetic_energy, gravitational_energy, intermolecular_energy)

@nb.jit
#Позволяет получить распределение расстояний от каждой точки до ее соседей от rmin до rmax.
#Хотел получить пик около r=r0, но не вышло.
def get_distances_distribiton(positions, step=0.1, rmin=0, rmax=2):
    n_particles = positions.shape[0]
    number_of_baskets = int((rmax-rmin)/step)
    baskets = np.zeros(number_of_baskets)
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            dx = positions[j,0] - positions[i,0]
            dy = positions[j,1] - positions[i,1]             
            dr = (dx * dx + dy * dy)**0.5
            if dr>rmin and dr<rmax:
                basket = int((dr-rmin)/step)
                baskets[basket] += dr
    return baskets

#Вычисляет шаг методом Эйлера. Чтобы энергия системы сохранялась, пользуюсь симплектным методом.
#Берет список частиц, размер шага, вычисляет следующее положение и прописывает его частицам.
def euler(particles, dt=0.001):
    coordinates = np.asanyarray([particle.get_coordinates() for particle in particles])
    positions, velocities = np.split(coordinates, 2, 1)
    
    next_velocities = velocities + get_forces(positions)*dt
    next_positions = positions + next_velocities*dt
    next_coordinates = np.append(next_positions, next_velocities, 1)
    
    for i in range(len(particles)):
        particles[i].set_coordinates(next_coordinates[i])  
    return None

#То же самое с методом Рунге-Кутта.
#Не уверен насчет правильности имплементации.
def rungeKutta(particles, dt=0.001):
    coordinates = np.asanyarray([particle.get_coordinates() for particle in particles])
    positions, velocities = np.split(coordinates, 2, 1)
    
    k1 = get_forces(positions)
    k2 = get_forces(positions+0.5*k1*dt)
    k3 = get_forces(positions+0.5*k2*dt)
    k4 = get_forces(positions+k3*dt)
    
    next_velocities = velocities + (k1+2*k2+2*k3+k4)/6*dt
    next_positions = positions + next_velocities*dt
    next_coordinates = np.append(next_positions, next_velocities, 1)
    
    for i in range(len(particles)):
        particles[i].set_coordinates(next_coordinates[i])
    return None

#Список констант
G = 1e-4 #Гравитационная постоянная
a = 10 #Жесткость потенциала Морзе
r0 = 0.1 #Равновесное расстояние в потенциала Морза
world_size = 2 #Размер мира симуляции
dt = 0.001 #Размер шага по времени

number_of_particles = 80 #Число частиц
particles = []

for i in range(number_of_particles):
    x,y = (np.random.rand(2))*world_size #Задает положение частиц
    vx, vy = np.zeros(2) #Задает скорости частиц
    particle = Particle(np.asarray([x,y,vx,vy])) #Создает частицу
    particles.append(particle) #Прикрепляет ее к списку частиц
    

#Ниже идет сама симуляция и ее визуализация.
#Если PyGame вылетает, то надо удалить отмеченные ниже строки.
background_colour = (255,255,255) 
screen_size = 300

screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption('Proto')
screen.fill(background_colour)
pygame.init() #Удалить, если вылетает
font = pygame.font.Font(None, 32) #Удалить, если вылетает


i = True
while i:
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            i=False
    screen.fill(background_colour)
    pygame.display.flip()
    
step = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            
    screen.fill(background_colour)
    time_text = 'Time: {0:.1f}'.format(step*dt) #Удалить, если вылетает
    text = font.render(time_text, True, (0, 0, 0)) #Удалить, если вылетает
    screen.blit(text, (5, 5)) #Удалить, если вылетает
    
    rungeKutta(particles, dt) #Делает шаг симуляции
    for i, particle in enumerate(particles): #Прорисовывает каждую частицу
        pygame.draw.circle(screen, particle.colour, [int((i*0.8/world_size+0.1)*screen_size)
                                                     for i in particle.get_coordinates()[0:2]], 2, 1)
        
    pygame.display.flip()
    step += 1

