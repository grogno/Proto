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

#Ýòà ôóíêöèÿ âû÷èñëÿåò ìàññèâ äåéñòâóþùèõ íà ÷àñòèöû ñèë, èñõîäÿ èç ìàññèâà èõ ìåñòîïîëîæåíèÿ.
#Íà ÷àñòèöó äåéñòâóþò ñèëà ãðàâèòàöèè (F = G/r) è ìåæìîëåêóëÿðíûå ñèëû, ðàâíûå ïðîèçâîäíîé ïîòåíöèàëà Ìîðçå ïî r.
#Ïîòåíöèàë Ìîðçå õàðàêòåðèçóåòñÿ ïàðàìåòðàìè a("æåñòêîñòü" ïîòåíöèàëà) è ðàâíîâåñíûì ðàññòîÿíèåì r0.
@nb.jit(nopython=True) #Äåêîðàòîð èç ìîäóëÿ numba êîìèëèðóåò ýòó ôóíêöèþ íà õîäó, îáåñïå÷èâàÿ ïðèðîñò ïðîèçâîäèòåëüíîñòè.
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
            if dr>0.5: #Ìåæìîëåêóëÿðíûå âçàèìîäåéñòâèÿ íà áîëüøèõ ðàññòîÿíèÿõ îáðåçàþòñÿ
                force = G/dr
            else:
                force = G/dr**2-2*a*math.exp(a*(r0-dr))*(math.exp(a*(r0-dr))-1)
            forces[i, 0] += ex * force
            forces[i, 1] += ey * force
            forces[j, 0] -= ex * force
            forces[j, 1] -= ey * force
    return(forces)

#Ýòà ôóíêöèÿ âû÷èñëÿåò êèíåòè÷åñêóþ è ïîòåíöèàëüíóþ ýíåðãèþ ñèñòåìû â äàííîé êîíôèãóðàöèè.
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
#Ïîçâîëÿåò ïîëó÷èòü ðàñïðåäåëåíèå ðàññòîÿíèé îò êàæäîé òî÷êè äî åå ñîñåäåé îò rmin äî rmax.
#Õîòåë ïîëó÷èòü ïèê îêîëî r=r0, íî íå âûøëî.
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

#Âû÷èñëÿåò øàã ìåòîäîì Ýéëåðà. ×òîáû ýíåðãèÿ ñèñòåìû ñîõðàíÿëàñü, ïîëüçóþñü ñèìïëåêòíûì ìåòîäîì.
#Áåðåò ñïèñîê ÷àñòèö, ðàçìåð øàãà, âû÷èñëÿåò ñëåäóþùåå ïîëîæåíèå è ïðîïèñûâàåò åãî ÷àñòèöàì.
def euler(particles, dt=0.001):
    coordinates = np.asanyarray([particle.get_coordinates() for particle in particles])
    positions, velocities = np.split(coordinates, 2, 1)
    
    next_velocities = velocities + get_forces(positions)*dt
    next_positions = positions + next_velocities*dt
    next_coordinates = np.append(next_positions, next_velocities, 1)
    
    for i in range(len(particles)):
        particles[i].set_coordinates(next_coordinates[i])  
    return None

#Òî æå ñàìîå ñ ìåòîäîì Ðóíãå-Êóòòà.
#Íå óâåðåí íàñ÷åò ïðàâèëüíîñòè èìïëåìåíòàöèè.
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

#Ñïèñîê êîíñòàíò
G = 1e-4 #Ãðàâèòàöèîííàÿ ïîñòîÿííàÿ
a = 10 #Æåñòêîñòü ïîòåíöèàëà Ìîðçå
r0 = 0.1 #Ðàâíîâåñíîå ðàññòîÿíèå â ïîòåíöèàëà Ìîðçà
world_size = 2 #Ðàçìåð ìèðà ñèìóëÿöèè
dt = 0.001 #Ðàçìåð øàãà ïî âðåìåíè

number_of_particles = 80 #×èñëî ÷àñòèö
particles = []

for i in range(number_of_particles):
    x,y = (np.random.rand(2))*world_size #Çàäàåò ïîëîæåíèå ÷àñòèö
    vx, vy = np.zeros(2) #Çàäàåò ñêîðîñòè ÷àñòèö
    particle = Particle(np.asarray([x,y,vx,vy])) #Ñîçäàåò ÷àñòèöó
    particles.append(particle) #Ïðèêðåïëÿåò åå ê ñïèñêó ÷àñòèö
    

#Íèæå èäåò ñàìà ñèìóëÿöèÿ è åå âèçóàëèçàöèÿ.
#Åñëè PyGame âûëåòàåò, òî íàäî óäàëèòü îòìå÷åííûå íèæå ñòðîêè.
background_colour = (255,255,255) 
screen_size = 300

screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption('Proto')
screen.fill(background_colour)
pygame.init() #Óäàëèòü, åñëè âûëåòàåò
font = pygame.font.Font(None, 32) #Óäàëèòü, åñëè âûëåòàåò


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
    time_text = 'Time: {0:.1f}'.format(step*dt) #Óäàëèòü, åñëè âûëåòàåò
    text = font.render(time_text, True, (0, 0, 0)) #Óäàëèòü, åñëè âûëåòàåò
    screen.blit(text, (5, 5)) #Óäàëèòü, åñëè âûëåòàåò
    
    rungeKutta(particles, dt) #Äåëàåò øàã ñèìóëÿöèè
    for i, particle in enumerate(particles): #Ïðîðèñîâûâàåò êàæäóþ ÷àñòèöó
        pygame.draw.circle(screen, particle.colour, [int((i*0.8/world_size+0.1)*screen_size)
                                                     for i in particle.get_coordinates()[0:2]], 2, 1)
        
    pygame.display.flip()
    step += 1

