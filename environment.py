import pygame
import random
import copy
import numpy as np
import tensorflow as tf

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

board = ["*   g",
         " www ",
         " w*  ",
         " www ",
         "p    "]

clock = pygame.time.Clock()

class Pacman:
    def __init__(self, board):
        self.player_image = pygame.transform.scale(pygame.image.load("assets/pacman.png"), (30, 30))
        self.ghost_image = pygame.transform.scale(pygame.image.load("assets/red_ghost.png"), (30, 30))
        self.display_mode_on = True
        self.board = board
        self.cell_size = 60
        pygame.init()
        self.screen = pygame.display.set_mode((len(board[0]) * self.cell_size, (len(board) * self.cell_size)))
        self.player_pos = dict()
        self.ghosts = []
        self.foods = []
        self.score = 0
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                if self.board[y][x] == 'p':
                    self.player_pos['x'] = x
                    self.player_pos['y'] = y
                    self.init_player_pos = self.player_pos.copy()
                elif self.board[y][x] == 'g':
                    ghost = dict()
                    ghost['x'] = x
                    ghost['y'] = y
                    ghost['direction'] = random.choice([LEFT, DOWN])
                    self.ghosts.append(ghost)
                elif self.board[y][x] == '*':
                    food = dict()
                    food['x'] = x
                    food['y'] = y
                    self.foods.append(food)

        self.init_foods = copy.deepcopy(self.foods)
        self.init_ghosts = copy.deepcopy(self.ghosts)
        self.states={}
        self.actions={}
        self.nextStates={}
        self.makeStates()
        self.beginingState=self.__get_state()
        self.__draw_board()

    def reset(self):
        self.foods = copy.deepcopy(self.init_foods)
        self.ghosts = copy.deepcopy(self.init_ghosts)
        self.player_pos = self.init_player_pos.copy()
        self.score = 0
        return self.__get_state()

    def nextPos(self, state, objectNum, action):
        pos = dict()
        if action == 0:
            pos['x'] = state[objectNum]['x'] - 1
            pos['y'] = state[objectNum]['y']
        if action == 1:
            pos['x'] = state[objectNum]['x']
            pos['y'] = state[objectNum]['y'] + 1
        if action == 2:
            pos['x'] = state[objectNum]['x'] + 1
            pos['y'] = state[objectNum]['y']
        if action == 3:
            pos['x'] = state[objectNum]['x']
            pos['y'] = state[objectNum]['y'] - 1
        return pos

    def makeStates(self):
        possiblePlayer = []
        possibleGhost = []
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                if self.board[y][x] != 'w':
                    possiblePlayer.append({'x': x, 'y': y})
                    possibleGhost.append({'x': x, 'y': y})
        possibleFood = [[{'x':0, 'y':0}, {'x':2, 'y':2}], [{'x':0, 'y':0}], [{'x':2, 'y':2}], []]
        stateNum=0

        for player in possiblePlayer:
            for ghost in possibleGhost:
                for food in possibleFood:
                    self.states[stateNum]=[player, ghost, food]
                    stateNum+=1

        for key, state in self.states.items():
            x=state[0]['x']
            y=state[0]['y']
            self.actions[key]=[]
            if(x!=0):
                if(self.board[y][x-1] != 'w'):
                    self.actions[key].append(LEFT)
            if(y!=0):
                if(self.board[y-1][x] != 'w'):
                    self.actions[key].append(UP)
            if(x!=len(self.board[0])-1):
                if(self.board[y][x+1] != 'w'):
                    self.actions[key].append(RIGHT)
            if(y!=len(self.board)-1):
                if(self.board[y+1][x] != 'w'):
                    self.actions[key].append(DOWN)
        
        for key, state in self.states.items():
            self.nextStates[key]={}
            for action in self.actions[key]:
                self.nextStates[key][action]={}
                for nsKey, ns in self.states.items():
                    if ns[0] == self.nextPos(state,0,action) and ns[2] == state[2]:
                    # for ghost in s.ghost list:
                        for i in range(4):
                            if ns[1] == self.nextPos(state,1,i):
                                self.nextStates[key][action][nsKey]=1

    def get_all_states(self):
        return self.states.keys()

    def is_terminal(self, state):
        if (self.states[state][0] == self.states[state][1]) or self.states[state][2] == False:
            return True
        return False

    def get_possible_actions(self, state):
        return self.actions[state]

    def get_next_states(self, state, action):
        return self.nextStates[state][action]

    def get_number_of_states(self):
        return len(self.states)

    def get_reward(self, state, action, next_state):
        reward = 0
        for food_pos in self.states[state][2]:
            if self.nextPos(self.states[state],0,action) == food_pos:
                reward += 10
        if self.states[next_state][2] == False:
            reward += 500
        if self.nextPos(self.states[state],0,action) == self.states[next_state][1]:
            reward -= 500
        reward -= 1
        return reward

    def step(self, action):
        width = len(self.board[0])
        height = len(self.board)

        if action == LEFT and self.player_pos['x'] > 0:
            if self.board[self.player_pos['y']][self.player_pos['x'] - 1] != 'w':
                self.player_pos['x'] -= 1
        if action == RIGHT and self.player_pos['x'] + 1 < width:
            if self.board[self.player_pos['y']][self.player_pos['x'] + 1] != 'w':
                self.player_pos['x'] += 1
        if action == UP and self.player_pos['y'] > 0:
            if self.board[self.player_pos['y'] - 1][self.player_pos['x']] != 'w':
                self.player_pos['y'] -= 1
        if action == DOWN and self.player_pos['y'] + 1 < height:
            if self.board[self.player_pos['y'] + 1][self.player_pos['x']] != 'w':
                self.player_pos['y'] += 1

        for ghost in self.ghosts:
            if ghost['x'] == self.player_pos['x'] and ghost['y'] == self.player_pos['y']:
                self.score -= 500
                reward = -500
                self.__draw_board()
                return  self.__get_state(), reward, True, self.score

        for food in self.foods:
            if food['x'] == self.player_pos['x'] and food['y'] == self.player_pos['y']:
                self.score += 10
                reward = 10
                self.foods.remove(food)
                break
        else:
            self.score -= 1
            reward = -1

        for ghost in self.ghosts:
            moved = False
            ghost_moves = [LEFT, RIGHT, UP, DOWN]
            if ghost['x'] > 0 and self.board[ghost['y']][ghost['x'] - 1] != 'w':
                if ghost['direction'] == LEFT:
                    if RIGHT in ghost_moves:
                        ghost_moves.remove(RIGHT)
            else:
                if LEFT in ghost_moves:
                    ghost_moves.remove(LEFT)

            if ghost['x'] + 1 < width and self.board[ghost['y']][ghost['x'] + 1] != 'w':
                if ghost['direction'] == RIGHT:
                    if LEFT in ghost_moves:
                        ghost_moves.remove(LEFT)
            else:
                if RIGHT in ghost_moves:
                    ghost_moves.remove(RIGHT)

            if ghost['y'] > 0 and self.board[ghost['y'] - 1][ghost['x']] != 'w':
                if ghost['direction'] == UP:
                    if DOWN in ghost_moves:
                        ghost_moves.remove(DOWN)
            else:
                if UP in ghost_moves:
                    ghost_moves.remove(UP)

            if ghost['y'] + 1 < height and self.board[ghost['y'] + 1][ghost['x']] != 'w':
                if ghost['direction'] == DOWN:
                    if UP in ghost_moves:
                        ghost_moves.remove(UP)
            else:
                if DOWN in ghost_moves:
                    ghost_moves.remove(DOWN)

            ghost['direction'] = random.choice(ghost_moves)

            if ghost['direction'] == LEFT and ghost['x'] > 0:
                if self.board[ghost['y']][ghost['x'] - 1] != 'w':
                    ghost['x'] -= 1
            if ghost['direction'] == RIGHT and ghost['x'] + 1 < width:
                if self.board[ghost['y']][ghost['x'] + 1] != 'w':
                    ghost['x'] += 1
            if ghost['direction'] == UP and ghost['y'] > 0:
                if self.board[ghost['y'] - 1][ghost['x']] != 'w':
                    ghost['y'] -= 1
            if ghost['direction'] == DOWN and ghost['y'] + 1 < height:
                if self.board[ghost['y'] + 1][ghost['x']] != 'w':
                    ghost['y'] += 1

        for ghost in self.ghosts:
            if ghost['x'] == self.player_pos['x'] and ghost['y'] == self.player_pos['y']:
                self.score -= 500
                reward = -500
                self.__draw_board()
                return  self.__get_state(), reward, True, self.score

        self.__draw_board()

        if len(self.foods) == 0:
            reward = 500
            self.score += 500

        return self.__get_state(), reward, len(self.foods) == 0, self.score

    def __draw_board(self):
        if self.display_mode_on:
            self.screen.fill((0, 0, 0))

            y = 0

            for line in board:
                x = 0
                for obj in line:
                    if obj == 'w':
                        color = (0, 255, 255)
                        pygame.draw.rect(self.screen, color, pygame.Rect(x, y, 60, 60))
                    x += 60
                y += 60

            color = (255, 255, 0)
            self.screen.blit(self.player_image, (self.player_pos['x'] * self.cell_size + 15, self.player_pos['y'] * self.cell_size + 15))

            color = (255, 0, 0)
            for ghost in self.ghosts:
                self.screen.blit(self.ghost_image,
                                 (ghost['x'] * self.cell_size + 15, ghost['y'] * self.cell_size + 15))

            color = (255, 255, 255)

            for food in self.foods:
                pygame.draw.ellipse(self.screen, color, pygame.Rect(food['x'] * self.cell_size + 25, food['y'] * self.cell_size + 25, 10, 10))

            pygame.display.flip()

    def __get_state(self):
        for key, state in self.states.items():
            if state[0] == self.player_pos and state[1]['x'] == self.ghosts[0]['x'] and state[1]['y'] == self.ghosts[0]['y'] and state[2] == self.foods:
                return key

    def turn_off_display(self):
        self.display_mode_on = False

    def turn_on_display(self):
        self.display_mode_on = True