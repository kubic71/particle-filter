# Simple pygame demo with bouncing ball
import math
from turtle import window_height
import pygame
import itertools
from copy import deepcopy
import random
from typing import List
import sys
import numpy as np

pygame.init()

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Particle Filter")


NUM_PARTICLES = 200
# INIT_POS = (random.randint(0, 1920), random.randint(0, 1080))
INIT_POS = (1920/2, 1080/2)
POS_STD = 0.5
VEL_STD = 0.5

WHITE = (255, 255, 255)
RED = (255, 0, 0)
PURPLE = (255, 0, 255)
BLUE = (0, 0, 255)


# w is gaussian noise added to the motion model
VEL_STD = 0.1

# measurement noise
MES_STD = 40

def gaussian2d_pdf(x, y, std):
    # compute the probability density
    p = 1 / (2 * math.pi * std ** 2) * math.exp(-(x ** 2 + y ** 2) / (2 * std ** 2))
    # print(x, y, std, p)
    return p


def gaussian2d_sample(std):
    # return the sample and the probability density at that point

    # sample from a gaussian distribution
    x = random.gauss(0, std)
    y = random.gauss(0, std)

    # compute the probability density
    return x, y, gaussian2d_pdf(x, y, std)

def color_intensified(color, intensity):
    return tuple(map(lambda c: min(int(c * intensity), 255), color))

class Particle:

    def __init__(self) -> None:
        # Particles are normally distributed around the initial position
        self.x = INIT_POS[0] + random.gauss(0, POS_STD)
        self.y = INIT_POS[1] + random.gauss(0, POS_STD)

        self.vx = random.gauss(0, VEL_STD)
        self.vy = random.gauss(0, VEL_STD) 

        self.weight = 1.0
        self.color = (255, 255, 0)

        self.radius = 5
    

    def draw(self):

        intensity = self.weight 
        pygame.draw.circle(screen, color_intensified(self.color, intensity), (self.x, self.y), self.radius)
        # pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

        arr_len = 4
        ###  draw velocity vector ###
        pygame.draw.line(screen, color_intensified(WHITE, intensity) , (self.x, self.y), (self.x + self.vx*arr_len, self.y + self.vy*arr_len), 1)

        ## end arrows of the velocity vector ##
        pygame.draw.circle(screen, color_intensified(RED, intensity), (self.x + self.vx*arr_len, self.y + self.vy*arr_len), 3)


    def motion_update(self):
        # x_t = x_t-1 + v_t-1 + w_t-1
        # where w_t-1 is gaussian noise

        # sample from 2D normal distribution
        wx, wy, prob = gaussian2d_sample(POS_STD)

        # update the particle
        self.x += self.vx + wx
        self.y += self.vy + wy


        self.vx += random.gauss(0, VEL_STD)
        self.vy += random.gauss(0, VEL_STD)

        # print(self.weight)

    def measurement_update(self, zx, zy):
        self.weight *= gaussian2d_pdf(self.x - zx, self.y - zy, MES_STD*2)


class MockMeasurements:
    def __init__(self) -> None:
        self.x, self.y = (INIT_POS[0], INIT_POS[1])
        self.vx, self.vy = (0, 0)

        self.movement_speed_change = VEL_STD


    def get_next(self):
        self.x += self.vx
        self.y += self.vy 

        self.vx += random.gauss(0, self.movement_speed_change)
        self.vy += random.gauss(0, self.movement_speed_change)


        bounce_slowdown = 0.2

        # bounce off the walls
        if self.x < 0:
            self.x = 0
            self.vx *= -bounce_slowdown
        if self.x > WINDOW_WIDTH:
            self.x = WINDOW_WIDTH
            self.vx *= -bounce_slowdown
        if self.y < 0:
            self.y = 0
            self.vy *= -bounce_slowdown
        if self.y > WINDOW_HEIGHT:
            self.y = WINDOW_HEIGHT
            self.vy *= -bounce_slowdown

        zx, zy = random.gauss(self.x, MES_STD), random.gauss(self.y, MES_STD)
        return zx, zy

def normalize(particles):
    # normalize the weights
    max_weight = max(particles, key=lambda p: p.weight).weight
    # print(max_weight)
    for p in particles:
        p.weight /= max_weight
        p.weight = max(p.weight, 0.00001)
        print(p.weight)

def resample(particles) -> List[Particle]:
    # resample the particles
    # print("resampling")
    # normalize the weights
    normalize(particles)

    percent_to_keep = 0.8

    particles = sorted(particles, key=lambda p: p.weight, reverse=True)

    # remove the lowest percent_to_keep of the particles
    particles = particles[:int(len(particles) * percent_to_keep)]
    to_add = NUM_PARTICLES - len(particles)

    # add new particles
    # weigthed random sampling
    sum_weight = sum(p.weight for p in particles)
    probs = [p.weight / sum_weight for p in particles]
    parents = np.random.choice(particles, size=to_add, p=probs)

    for p in parents:
        particles.append(deepcopy(p))

    return particles



mock_measurements = MockMeasurements()
particles = []
for i in range(NUM_PARTICLES):
    particles.append(Particle())


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((0, 0, 0))

    #### Update particles ##### 
    for particle in particles:
        particle.motion_update()
    
    #### Measurement update #####
    zx, zy = mock_measurements.get_next()

    measurement = False
    if random.random() < 0.3:
        measurement = True
        for particle in particles:
            particle.measurement_update(zx, zy)

        # normalize(particles)
        particles = resample(particles)

    for particle in particles:
        particle.draw()

    # draw the true position
    pygame.draw.circle(screen, PURPLE, (mock_measurements.x, mock_measurements.y), 10)

    # draw measurement
    if measurement:
        pygame.draw.circle(screen, BLUE, (zx, zy), 15)


    pygame.display.update()
    pygame.time.delay(10)
        