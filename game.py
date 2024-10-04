from agent import *
from environment import *

class Game:
    def __init__(self):
        # Setup model
        self.model = tf.keras.models.load_model('model.h5')
        # Setup env
        self.env = Pacman(board)
        # Setup agent
        self.agent = DQNAgent(4, 0.001, self.model, get_legal_actions=self.env.get_possible_actions)
        self.agent.epsilon = 0.0

        self.done = False
        self.total_reward = 0.0
        self.state = self.env.reset()

    def step(self):
        action = self.agent.get_action(self.state)
        self.state, reward, self.done, _ = self.env.step(action)
        self.total_reward += reward

    def run(self):     
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
            self.step()    
            pygame.time.wait(300)

if __name__ == "__main__":
    game = Game()
    game.run()
    print(f'Points: {game.total_reward}')
