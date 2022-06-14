# Simple pygame demo with bouncing ball
import math
import pygame
import itertools
import random
import sys

pygame.init()

screen = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption("Particle Filter")

NUM_PARTICLES = 100
# INIT_POS = (random.randint(0, 1920), random.randint(0, 1080))
INIT_POS = (1920/2, 1080/2)
INIT_POS_STD = 0.5
INIT_VEL_STD = 0.5

WHITE = (255, 255, 255)
RED = (255, 0, 0)
PURPLE = (255, 0, 255)


# w is gaussian noise added to the motion model
VEL_STD = 0.1

# measurement noise
MES_STD = 10

def gaussian2d_pdf(x, y, std):
    # compute the probability density
    p = 1 / (2 * math.pi * std ** 2) * math.exp(-(x ** 2 + y ** 2) / (2 * std ** 2))
    print(p)
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
        self.x = INIT_POS[0] + random.gauss(0, INIT_POS_STD)
        self.y = INIT_POS[1] + random.gauss(0, INIT_POS_STD)

        self.vx = random.gauss(0, INIT_VEL_STD)
        self.vy = random.gauss(0, INIT_VEL_STD) 

        self.weight = 1.0
        self.color = (255, 255, 0)

        self.radius = 5

    def draw(self):

        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

        ###  draw velocity vector ###
        pygame.draw.line(screen, color_intensified(WHITE, self.weight) , (self.x, self.y), (self.x + self.vx, self.y + self.vy), 1)

        ## end arrows of the velocity vector ##
        arr_len = 10
        pygame.draw.circle(screen, color_intensified(RED, self.weight), (self.x + self.vx*arr_len, self.y + self.vy*arr_len), 3)


    def motion_update(self):
        # x_t = x_t-1 + v_t-1 + w_t-1
        # where w_t-1 is gaussian noise

        # sample from 2D normal distribution
        wx, wy, prob = gaussian2d_sample(VEL_STD)

        # update the particle
        self.x += self.vx + wx
        self.y += self.vy + wy

        # update the weight
        self.weight *= prob

        # print(self.weight)

    def measurement_update(self, zx, zy):
        self.weight *= gaussian2d_pdf(self.x - zx, self.y - zy, MES_STD)


class MockMeasurements:
    def __init__(self) -> None:
        self.x, self.y = (INIT_POS[0], INIT_POS[1])
        self.vx, self.vy = (0, 0)

        self.movement_speed_change = 0.1


    def get_next(self):
        self.x += self.vx
        self.y += self.vy 

        self.vx += random.gauss(0, self.movement_speed_change)
        self.vy += random.gauss(0, self.movement_speed_change)

        zx, zy = random.gauss(self.x, MES_STD), random.gauss(self.y, MES_STD)
        return zx, zy

def normalize(particles):
    # normalize the weights
    max_weight = max(particles, key=lambda p: p.weight).weight
    # print(max_weight)
    for p in particles:
        p.weight /= max_weight

    new_max = max(particles, key=lambda p: p.weight).weight
    # print(new_max)


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

    zx, zy = mock_measurements.get_next()

    #### Update particles ##### 
    for particle in particles:
        particle.motion_update()
        particle.measurement_update(zx, zy)
        particle.draw()
    
    # draw the true position
    pygame.draw.circle(screen, PURPLE, (mock_measurements.x, mock_measurements.y), 5)

    normalize(particles)
    pygame.display.update()
    pygame.time.delay(10)
        