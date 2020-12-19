import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
from examples import Point
from examples import Examples
from nn import Neural_Network

arm_length = 50.0
pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Open Sans', 24)
screen = pygame.display.set_mode([500, 500])

def translate(center, angle):
    return Point(center.x + arm_length * np.sin(angle), center.y - arm_length * np.cos(angle))

def alpha_point(alpha, beta, center):
    first_point = translate(center, alpha)
    second_joint = translate(first_point, np.pi - beta + alpha)
    return first_point, second_joint

def unstandarise(ang):
    #ang_x= np.array(ang)/np.pi*0.8 - 0.1
    ang_x= np.array(ang)*np.pi
    return ang_x

def main():
    
    ex = Examples(arm_length)
    e = ex.generate(1000)

    #fig, ax = plt.subplots()
    #ax.axis('equal')
    #for (x, y) in e[0]:
    #    plt.scatter(x, y, marker='o')
    #plt.show()

    np.max(e[0]), np.max(e[1])
    x_train = (np.array(e[0]) + arm_length*2.0)/(arm_length*4.0)*0.8 + 0.1
    y_train = np.array(e[1])/np.pi*0.8 + 0.1
    NN = Neural_Network()
    for i in range(5000):
        NN.train(x_train, y_train)
    
    err = NN.errors
    plt.plot(range(len(err)), err)
    plt.show()

    screen.fill((255, 255, 255))
    image = pygame.image.load('robot.png')
    screen.blit(image, (0, 0))
    pygame.display.flip()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if pos[0] > 250 or True:
                    position = np.array(pos)
                    position[0] =  position[0] - 250
                    position[1] =  250 - position[1]
                    position = (np.array(position) + arm_length*2.0)/(arm_length*4.0)*0.8 + 0.1
                    ang = NN.forward( (position[0],position[1]) )
                    alpha, beta = unstandarise( ang )
                    print(alpha, beta)
                    points = alpha_point( np.pi - alpha, -beta,Point(250,250))
                    
                    print(points[0].x,points[0].y)
                    screen.fill((255, 255, 255))
                    image = pygame.image.load('robot.png')
                    screen.blit(image, (0, 0))
                    pygame.draw.line(screen, (255, 128, 0),
                                     (250, 250), (points[0].x, points[0].y), 5)
                    pygame.draw.line(
                        screen, (255, 128, 0), (points[0].x ,  points[0].y), (points[1].x , points[1].y ), 5)
                    pygame.display.flip()
                
    pygame.quit()

if __name__ == "__main__":
    main()