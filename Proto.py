import pygame
import numpy as np
import numba as nb
import math as math

from pygame.locals import *
import matplotlib.pyplot as plt

#������ ������� ����������� ������ Particle.
#����� ���������� �� �������������� � �������� (������ ��� ���������� coordinates),
#� ����� ���� � ������ ������ ��� ������������
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

#��� ������� ��������� ������ ����������� �� ������� ���, ������ �� ������� �� ��������������.
#�� ������� ��������� ���� ���������� (F = G/r) � ��������������� ����, ������ ����������� ���������� ����� �� r.
#��������� ����� ��������������� ����������� a("���������" ����������) � ����������� ����������� r0.
@nb.jit(nopython=True) #��������� �� ������ numba ���������� ��� ������� �� ����, ����������� ������� ������������������.
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
            if dr>0.5: #��������������� �������������� �� ������� ����������� ����������
                force = G/dr
            else:
                force = G/dr**2-2*a*math.exp(a*(r0-dr))*(math.exp(a*(r0-dr))-1)
            forces[i, 0] += ex * force
            forces[i, 1] += ey * force
            forces[j, 0] -= ex * force
            forces[j, 1] -= ey * force
    return(forces)

#��� ������� ��������� ������������ � ������������� ������� ������� � ������ ������������.
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
#��������� �������� ������������� ���������� �� ������ ����� �� �� ������� �� rmin �� rmax.
#����� �������� ��� ����� r=r0, �� �� �����.
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

#��������� ��� ������� ������. ����� ������� ������� �����������, ��������� ����������� �������.
#����� ������ ������, ������ ����, ��������� ��������� ��������� � ����������� ��� ��������.
def euler(particles, dt=0.001):
    coordinates = np.asanyarray([particle.get_coordinates() for particle in particles])
    positions, velocities = np.split(coordinates, 2, 1)
    
    next_velocities = velocities + get_forces(positions)*dt
    next_positions = positions + next_velocities*dt
    next_coordinates = np.append(next_positions, next_velocities, 1)
    
    for i in range(len(particles)):
        particles[i].set_coordinates(next_coordinates[i])  
    return None

#�� �� ����� � ������� �����-�����.
#�� ������ ������ ������������ �������������.
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

#������ ��������
G = 1e-4 #�������������� ����������
a = 10 #��������� ���������� �����
r0 = 0.1 #����������� ���������� � ���������� �����
world_size = 2 #������ ���� ���������
dt = 0.001 #������ ���� �� �������

number_of_particles = 80 #����� ������
particles = []

for i in range(number_of_particles):
    x,y = (np.random.rand(2))*world_size #������ ��������� ������
    vx, vy = np.zeros(2) #������ �������� ������
    particle = Particle(np.asarray([x,y,vx,vy])) #������� �������
    particles.append(particle) #����������� �� � ������ ������
    

#���� ���� ���� ��������� � �� ������������.
#���� PyGame ��������, �� ���� ������� ���������� ���� ������.
background_colour = (255,255,255) 
screen_size = 300

screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption('Proto')
screen.fill(background_colour)
pygame.init() #�������, ���� ��������
font = pygame.font.Font(None, 32) #�������, ���� ��������


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
    time_text = 'Time: {0:.1f}'.format(step*dt) #�������, ���� ��������
    text = font.render(time_text, True, (0, 0, 0)) #�������, ���� ��������
    screen.blit(text, (5, 5)) #�������, ���� ��������
    
    rungeKutta(particles, dt) #������ ��� ���������
    for i, particle in enumerate(particles): #������������� ������ �������
        pygame.draw.circle(screen, particle.colour, [int((i*0.8/world_size+0.1)*screen_size)
                                                     for i in particle.get_coordinates()[0:2]], 2, 1)
        
    pygame.display.flip()
    step += 1

