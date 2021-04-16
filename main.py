import numpy as np
import math
import time
import random
import pygame
from random import randint
pygame.init()
font = pygame.font.Font('freesansbold.ttf', 32)


class Player():
  def __init__(self, inputs, hiddenlayer, outputs):
    self.weights_1 = 0.10 * np.random.randn(inputs, hiddenlayer)
    self.biases_1 = np.zeros((1, hiddenlayer))

    self.weights_2 = 0.10 * np.random.randn(hiddenlayer, hiddenlayer)
    self.biases_2 = np.zeros((1, hiddenlayer))

    self.weights_3 = 0.10 * np.random.randn(hiddenlayer, outputs)
    self.biases_3 = np.zeros((1, outputs))

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def think(self, inputs):
    self.output_layer1 = self.sigmoid(np.dot(inputs, self.weights_1) + self.biases_1)
    # self.output_layer2 = self.sigmoid(np.dot(self.output_layer1, self.weights_2) + self.biases_2)
    # self.output_layer3 = self.sigmoid(np.dot(self.output_layer2, self.weights_3) + self.biases_3)
    self.output_layer3 = self.sigmoid(np.dot(self.output_layer1, self.weights_3) + self.biases_3)    
  def convert_output(self):
    self.highest_ouput_from_neuron = 0            
    for i in range(len(self.output_layer3[0])):
      if (self.output_layer3[0][i] > self.highest_ouput_from_neuron):
        self.highest_output_neuron_index = i
        self.highest_ouput_from_neuron = self.output_layer3[0][i]
        # print('neouron outputs are: ', self.output_layer3)
        # print('highest output neuron is: ', self.highest_ouput_from_neuron)
    self.output =  self.highest_output_neuron_index

  def mutate(self, chance, mutate_rate):
    for i in range(len(self.weights_1)):
      for k in range(len(self.weights_1[i])):
        if (randint(0,chance) == 1):
          self.weights_1[i][k] += random.uniform(-1, 1) * mutate_rate
    for i in range(len(self.biases_1[0])):
      if (randint(0,chance) == 1):
        self.biases_1[0][i] += random.uniform(-1, 1) * mutate_rate

    for i in range(len(self.weights_2)):
      for k in range(len(self.weights_2[i])):
        if (randint(0,chance) == 1):
          self.weights_2[i][k] += random.uniform(-1, 1) * mutate_rate
    for i in range(len(self.biases_2[0])):
      if (randint(0,chance) == 1):
        self.biases_2[0][i] += random.uniform(-1, 1) * mutate_rate

    for i in range(len(self.weights_3)):
      for k in range(len(self.weights_3[i])):
        if (randint(0,chance) == 1):
          self.weights_3[i][k] += random.uniform(-1, 1) * mutate_rate
    for i in range(len(self.biases_3[0])):
      if (randint(0,chance) == 1):
        self.biases_3[0][i] += random.uniform(-1, 1) * mutate_rate
  
class Game():
  def __init__(self):
    self.tiles = np.array([[2, 0, 2, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
    self.score = 0
  def reset(self):
     self.tiles = np.array([[2, 0, 2, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])

  def move(self, direction):
    self.already_moved = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])

    if (direction == 0):
      self.move_offset = [0, 1]

      self.x_move_range = [0, 4, 1]
      self.y_move_range = [0, 3, 1]
      self.new_tile_range = [random.randint(0,3), 3]
    if (direction == 1):
      self.move_offset = [-1, 0]

      self.x_move_range = [3, 0, -1]
      self.y_move_range = [0, 4, 1]
      self.new_tile_range = [0, random.randint(0,3)]
    if (direction == 2):
      self.move_offset = [0, -1]

      self.x_move_range = [0, 4, 1]
      self.y_move_range = [3, 0, -1]
      self.new_tile_range = [random.randint(0,3), 0]
    if (direction == 3):
      self.move_offset = [-1, 0]

      self.x_move_range = [0, 3, 1]
      self.y_move_range = [0, 4, 1]
      self.new_tile_range = [3, random.randint(0,3)]

    
    for swipe in range(3):
      for i in range(self.x_move_range[0], self.x_move_range[1], self.x_move_range[2]):
        for k in range(self.y_move_range[0], self.y_move_range[1], self.y_move_range[2]):
          if (self.tiles[i][k] == 0):
            self.tiles[i][k] = self.tiles[i+self.move_offset[0]][k+self.move_offset[1]]
            self.tiles[i+self.move_offset[0]][k+self.move_offset[1]] = 0
            
    for i in range(self.x_move_range[0], self.x_move_range[1], self.x_move_range[2]):
      for k in range(self.y_move_range[0], self.y_move_range[1], self.y_move_range[2]):
        if (self.tiles[i][k] == self.tiles[i+self.move_offset[0]][k+self.move_offset[1]]) and    (self.already_moved[i+self.move_offset[0]][k+self.move_offset[1]] == 0):
          self.tiles[i][k] = self.tiles[i][k] * 2
          self.already_moved[i][k] = 1
          self.tiles[i+self.move_offset[0]][k+self.move_offset[1]] = 0

    for swipe in range(3):
      for i in range(self.x_move_range[0], self.x_move_range[1], self.x_move_range[2]):
        for k in range(self.y_move_range[0], self.y_move_range[1], self.y_move_range[2]):
          if (self.tiles[i][k] == 0):
            self.tiles[i][k] = self.tiles[i+self.move_offset[0]][k+self.move_offset[1]]
            self.tiles[i+self.move_offset[0]][k+self.move_offset[1]] = 0
    self.add_new_tile(direction)
    

  def end_of_game(self):
    self.tiles_chache = np.array([[2, 2, 2, 0],
                                  [0, 4, 2, 2],
                                  [0, 2, 0, 2],
                                  [2, 2, 4, 0]])
    self.tiles_chache[:] = self.tiles[:]
    for i in range(0,3):
      self.move(i)

    if (np.array_equal(self.tiles_chache, self.tiles)):
      return True
    else:
      return False
    self.tiles = self.tiles_chache[:]
    
  def add_new_tile(self, direction):
    self.new_tile_avaliable_spots = []

    if (direction == 0):
      for i in range(4):
        if (self.tiles[i][3] == 0):
          self.new_tile_avaliable_spots.append(i)
          self.new_tile_spot = [random.choice(self.new_tile_avaliable_spots), 3]

    if (direction == 1):
      for i in range(4):
        if (self.tiles[0][i] == 0):
          self.new_tile_avaliable_spots.append(i)
          self.new_tile_spot = [0, random.choice(self.new_tile_avaliable_spots)]

    if (direction == 2):
      for i in range(4):
        if (self.tiles[i][0] == 0):
          self.new_tile_avaliable_spots.append(i)
          self.new_tile_spot = [random.choice(self.new_tile_avaliable_spots), 0]

    if (direction == 3):
      for i in range(4):
        if (self.tiles[3][i] == 0):
          self.new_tile_avaliable_spots.append(i)
          self.new_tile_spot = [3, random.choice(self.new_tile_avaliable_spots)]
    
    if (random.randint(0, 10) == 0):
      self.tiles[self.new_tile_spot[0]][self.new_tile_spot[1]] = 4
    else:
      self.tiles[self.new_tile_spot[0]][self.new_tile_spot[1]] = 2
      
    

  def calculate_score(self):
      
      self.score = np.sum(self.tiles) 
      for i in range(3):
        for k in range(3):
          if (self.tiles[i][k] == self.tiles[i+1][k] * 2 or self.tiles[i][k] == self.tiles[i+1][k] / 2):
            self.score += self.tiles[i][k] + self.tiles[i+1][k]
          else:
            self.score -= self.tiles[i][k] - self.tiles[i+1][k]
          if (self.tiles[i][k] == self.tiles[i][k+1] * 2 or self.tiles[i][k] == self.tiles[i][k+1] / 2):
            self.score += self.tiles[i][k] + self.tiles[i][k+1]
          else:
            self.score -= self.tiles[i][k] - self.tiles[i][k+1]
          

      for i in range(3):
        if (self.tiles[i][3] == self.tiles[i+1][3] * 2 or self.tiles[i][3] == self.tiles[i+1][3] / 2):
            self.score += self.tiles[i][3] + self.tiles[i+1][k]
        else:
          self.score -= self.tiles[i][3] - self.tiles[i+1][k]
      for k in range(3):
        if (self.tiles[3][k] == self.tiles[3][k+1] * 2 or self.tiles[3][k] == self.tiles[3][k+1] / 2):
            self.score += self.tiles[3][k] + self.tiles[i][k+1]
        else:
          self.score -= self.tiles[3][k] - self.tiles[i][k+1]
          
    
  def determine_color(self, number):
      if (number == 2):
        return tile_color_2
      if (number == 4):
        return tile_color_4
      if (number == 8):
        return tile_color_8
      if (number == 16):
        return tile_color_16
      if (number == 32):
        return tile_color_32
      if (number == 64):
        return tile_color_64
      if (number == 128):
        return tile_color_128
      if (number == 256):
        return tile_color_256
      if (number == 512):
        return tile_color_512
      else:
        return tile_color_0
    
  def drawBoard(self):
    screen.fill(background_color)
    
    tile_width = 70
    tile_offset = 24

    
    for i in range(4):
      for k in range(4):
        tile_position = [i*tile_width + tile_offset*(i+1), k*tile_width + tile_offset*(k+1)]
        
        tile_color = self.determine_color(self.tiles[i][k])
        self.drawTile(tile_position, tile_width, self.tiles[i][k], tile_color)
    
    pygame.display.flip()

  def drawTile(self, position, width, value, color):
    pygame.draw.rect(screen, color, (position[0], position[1], width, width))
    textsurface = myfont.render(str(value), False, BLACK)
    screen.blit(textsurface,(position[0]+width/2.5, position[1]+width/2.5))

class Population():
  def __init__(self,size):
    self.population = size
    self.players = list()
    self.games = list()
    self.highest_fitness_index = 0
    for i in range(size):
      self.players.append(Player(16, 16, 4))
      self.games.append(Game())
  

  def run_generation(self):
    self.players_alive = [True] * self.population
    for i in range(self.population):
      self.games[i].reset()
      self.players_alive[i] = True
    while(any(self.players_alive)):
      
      for i in range(self.population):
        self.players[i].think(self.games[i].tiles.flatten())
        self.players[i].convert_output()
        # print('direction of player', i, ' is ', self.players[i].output)
            
        # print(self.games[i].tiles)
        if (self.games[i].end_of_game() == False):
          self.games[i].move(self.players[i].output)
          # print('move was made for player #', i)
          
        else:
          self.players_alive[i] = False
          # print('player died #', i)
      

      self.games[self.highest_fitness_index].drawBoard()
      # time.sleep(1)


    # after they all die

    for i in range(self.population):
      self.games[i].calculate_score()
      self.players[i].fitness = self.games[i].score

    self.best_fitness = 0
    self.added_fitness = 0
    self.highest_fitness_index = 0
    for i in range(self.population):
      self.added_fitness += self.players[i].fitness
      if (self.players[i].fitness > self.best_fitness):
        self.best_fitness = self.players[i].fitness
        self.highest_fitness_index = i

    for i in range(self.population):
      if (self.players[i].fitness < self.added_fitness / (self.population)):
        self.players[i].weights_1[:] = self.players[self.highest_fitness_index].weights_1[:]
        self.players[i].biases_1[:] = self.players[self.highest_fitness_index].biases_1[:]

        self.players[i].weights_2[:] = self.players[self.highest_fitness_index].weights_2[:]
        self.players[i].biases_2[:] = self.players[self.highest_fitness_index].biases_2[:]

        self.players[i].weights_3[:] = self.players[self.highest_fitness_index].weights_3[:]
        self.players[i].biases_3[:] = self.players[self.highest_fitness_index].biases_3[:]

      if (self.players[i].fitness != self.best_fitness):
        self.players[i].mutate(8, 0.2)

      
    print(self.games[self.highest_fitness_index].tiles)

    for i in range(self.population):
      self.games[i].reset()
    
    # print(self.players[0].weights_1)
    # print(self.players[0].weights_2)
    
    print('highest fitness/score: ', self.best_fitness)
    # print(self.players[self.highest_fitness_index].output_layer3)
    


screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption('2048')
clock = pygame.time.Clock()
myfont = pygame.font.SysFont('Comic Sans MS', 25)


background_color = 130, 110, 91
RED = 255, 0, 0
BLACK = 0, 0, 0
tile_color_0 = BLACK
tile_color_2 = 213, 222, 235
tile_color_4 = 222, 124, 71
tile_color_8 = 59, 126, 227
tile_color_16 = 39, 138, 72
tile_color_32 = 207, 219, 94
tile_color_64 = 245, 77, 59
tile_color_128 = 176, 35, 136
tile_color_256 = 89, 174, 240
tile_color_512 = 42, 209, 50


pop = Population(70)

crash = False
while crash == False:
  for i in range(1000):
    pop.generation = i
    print('-New generation-: ', pop.generation)
    pop.run_generation()

    # time.sleep(0.2)
  crash = True


