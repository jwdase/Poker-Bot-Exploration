import os
from typing import runtime_checkable
os.makedirs('results/20-Epochs', exist_ok=True)

import numpy as np
import random
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from modules import WeightedMSELoss, WeightedDataset
from cursor import CursorPlayer

class Game:
    def __init__(self):
        self.cards = ["11", "11", "12", "12", "13", "13"]

    def play_game(self, player1, player2):
        """Game code for the round - we assume player1 is exporatory and player2 isn't"""

        card0, card1, card2 = random.sample(self.cards, k=3)

        # Define the history
        history = tuple()

        player1.new_round()
        player2.new_round()

        # Fill the pot
        player1.pay()
        player2.pay()
        pot = 2

        # Players make first move
        history += player1.move(card0, None, history)
        history += player2.move(card1, None, history)

        # Additional move if check --> bet
        if history == ("c", "b"):
            history += player1.move(card0, None, history)
        else:
            history += ('*',)

        # Game logic
        if history == ("c", "b", "c"):
            player2.win(pot)
            player1.loss()
            return
        elif (history == ("b", "c", '*')):
            player1.win(pot)
            player2.loss()
            return
        
        # Update the pot
        if (history == ("b", "b", "*") or history == ("c", "b", "b")):
            player1.pay()
            player2.pay()
            pot += 2

        # Next move
        history += player1.move(card0, card2, history)
        history += player2.move(card1, card2, history)

        # Addiiton move if check --> bet
        if history[-2:] == ("c", "b"):
            history += player1.move(card0, card2, history)
        else:
            history += ("*",)

        # Pot amount
        if history[-3:] == ("b", "b", "*") or history[-3:] == ("c", "b", "b"):
            player1.pay()
            player2.pay()
            pot += 2

        # game logic
        if (
            history == ("b", "b", "*", "b", "c", "*")
            or history == ("c", "c", "*", "b", "c", "*")
            or history == ("c", "b", "b", "b", "c", "*")
        ):
            player1.win(pot)
            player2.loss()

        elif (
            history == ("b", "b", "*", "c", "b", "c")
            or history == ("c", "c", "*", "c", "b", "c")
            or history == ("c", "b", "b", "c", "b", "c")
        ):
            player2.win(pot)
            player1.loss()

        elif (
            history == ("b", "b", "*", "b", "b", "*")
            or history == ("b", "b", "*", "c", "c", "*")
            or history == ("b", "b", "*", "c", "b", "b")
            or history == ("c", "b", "b", "b", "b", "*")
            or history == ("c", "b", "b", "c", "c", "*")
            or history == ("c", "b", "b", "c", "b", "b")
            or history == ("c", "c", "*", "b", "b", "*") 
            or history == ("c", "c", "*", "c", "c", "*")
            or history == ("c", "c", "*", "c", "b", "b")
        ):
            if card0 == card2 or (card0 > card1 and card1 != card2):
                player1.win(pot)
                player2.loss()
            elif card1 == card2 or (card1 > card0 and card0 != card2):
                player2.win(pot)
                player1.loss()
            else:
                player1.win(int(pot/2))
                player2.win(int(pot/2))

        # Esnures all histories are accounted for:
        else:
            raise AssertionError(f"history: {history} does not exist")

    def train(self, player1, player2, K=5_000, epochs=100):
        """Code to train the two bots"""
        print("Beginning training ------- ")

        player1.train_mode()
        player2.train_mode()

        player1.begin_training()
        player2.begin_training()

        for epoch in range(epochs):
            for round in range(K):
                self.play_game(player1, player2)

            player1.train_on_memory()
            player2.train_on_memory()

            if epoch < epochs - 1:
                player1.epoch_update()
                player2.epoch_update()

            print(f"Completed epoch: {epoch + 1}")

        print("Completed training -------- ")

    def compare_bot(self, player1, player2, iterations=10_000, amount=10_000):
        """Code to compare two robots"""

        print(f"Comparison of {player1} vs {player2}")

        player1.test_mode()
        player2.test_mode()

        player1.wealth = amount
        player2.wealth = amount

        for i in range(iterations):
            self.play_game(player1, player2)

        print(f"{player1} wealth: {player1.wealth}")
        print(f"{player2} wealth: {player2.wealth}")

        print("Comparison Complete ------ ")

class Player(CursorPlayer):
    def __init__(self, name='bot'):
        # Constants
        self.wealth = 10_000
        self.mode = 'test'
        self.spent_this_round = 0
        self.name = name

        # One-Hot values
        self.one_hot_priv_card = {"11" : 0, "12" : 1, "13" : 2}
        self.one_hot_pub_card = {None : 3, "11" : 4, "12" : 5, "13" : 6}
        self.actions = {'b' : torch.tensor([1, 0]), 'c' : torch.tensor([0, 1])}

        # Generate our NN
        self.network = self.gen_network()
        self.loss_f = WeightedMSELoss()
        self.criterion = Adam(self.network.parameters(), lr=0.0001)
        self.epochs = 75
        self.gamma = 0.95

        # Memory
        self.train_memory = []
        self.round_memory = []
        self.probability_memory = []

        # Constants for training NN
        self.e_greedy = 0.2
        self.dynamic_e_greedy = None
        self.current_epoch = None

    def state_tensor(self, priv_card, pub_card, history):
        """We encode each state with one-hot encoding"""
        one_hot = torch.zeros(19)

        # Embedd the card values
        one_hot[self.one_hot_priv_card[priv_card]] = 1
        one_hot[self.one_hot_pub_card[pub_card]] = 1

        # Insert the values for state
        i = 7
        for move in history:
            if move == "b":
                one_hot[i] = 1
            elif move == "c":
                one_hot[i+1] = 1

            i += 2

        return one_hot

    def new_round(self):
        self.network = self.gen_network()
        self.spent_this_round = 0
        self.round_memory = []

    def pay(self, amount=1):
        self.wealth -= amount
        self.spent_this_round += amount

    def move(self, priv_card, pub_card, history):
        """Class used for move"""

        input = self.state_tensor(priv_card, pub_card, history)

        if self.mode == 'train':
            return self.epsilon_gen_move(input)

        return self.gen_move(input)

    def gen_move(self, input):
        """Uses a NN to generate the next move"""

        with torch.no_grad():
            q_bet = self.network(torch.cat([input, self.actions['b']]))
            q_check = self.network(torch.cat([input, self.actions['c']]))
        
        # Save probabilities of actions
        q_bet = max(0, q_bet)
        q_check = max(0, q_check)
        total = q_bet + q_check

        self.probability_memory.append((
            input, (q_bet, q_check), self.current_epoch
        ))

        if total == 0:
            return (random.choice(['b', 'c']), )

        if random.random() < q_bet / total:
            self.round_memory.append((input, self.actions['b']))
            return ("b", )
        else:
            self.round_memory.append((input, self.actions['c']))
            return ("c", )

    def epsilon_gen_move(self, input):
        """Uses epsilon greedy to generate moves"""

        if random.random() < self.dynamic_e_greedy:
            a = random.choice(['b', 'c'])
            self.round_memory.append((input, self.actions[a]))
            return (a, )
        else:
            return self.gen_move(input)

    def begin_training(self):
        """ Setup constants for begiing of training"""

        # Reset Data collection
        self.train_memory = []
        self.round_memory = []
        self.probability_memory = []

        # Setup e-greedy and epoch counting - Create a network
        self.dynamic_e_greedy = self.e_greedy
        self.current_epoch = 1
        self.network = self.gen_network()

    def epoch_update(self, decay_rate=.8):
        # Updates the greedy value
        self.dynamic_e_greedy = (self.dynamic_e_greedy * decay_rate)

        # Updates the epoch number
        self.current_epoch += self.current_epoch
        
    def gen_network(self):
        return nn.Sequential(
            nn.Linear(21, 60),
            nn.ReLU(),
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Linear(60, 1),
        )

    def win(self, value):
        self.wealth += value
        self.apply_reward(value - self.spent_this_round)
        
    def apply_reward(self, reward):
        """Applies reward to each state"""

        for i, (state, action_vec) in enumerate(reversed(self.round_memory)):
            discounted_reward = reward * (self.gamma ** i)
            self.train_memory.append((state, action_vec, discounted_reward, self.current_epoch))

        self.round_memory = []
        self.spent_this_round = 0

    def loss(self):
        self.apply_reward(-self.spent_this_round)

    def train_mode(self):
        self.mode = 'train'

    def test_mode(self):
        self.mode = 'test'

    def train_on_memory(self):
        """Code to train out network"""

        if not self.train_memory:
            return None

        states, actions, rewards, weights = zip(*self.train_memory)

        input = torch.hstack((
            torch.vstack(states),
            torch.vstack(actions)
        ))

        targets = torch.tensor(rewards, dtype=torch.float32)[:, None]

        dataset = WeightedDataset(input, targets, weights)
        train_loader = DataLoader(dataset, batch_size=250, shuffle=True)
        
        # Training Loop
        self.network.train()

        for i in range(self.epochs):
            epoch_loss = 0
            for x_input, y_label, weights in train_loader:
                self.criterion.zero_grad()

                outputs = self.network(x_input)
                loss = self.loss_f(outputs, y_label, weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                self.criterion.step()
                epoch_loss += loss.item()

        # Set Network back to evaluate
        self.network.eval()

    def __str__(self):
        return self.name


class Dum_Player:
    def __init__(self):
        self.wealth = 10_000

    def new_round(self):
        pass

    def pay(self):
        self.wealth -= 1

    def win(self, value):
        self.wealth += value

    def move(self, priv_card, pub_card, history):
        return ('b', )

    def loss(self):
        pass

    def test_mode(self):
        pass

    def __str__(self):
        return "Dum Bot"

if __name__ == "__main__":

    # Create the players
    player1 = Player('Player 1')
    player2 = Player('Player 2')
    player3 = Dum_Player()

    # Initilize the game
    game = Game()

    # Initial comparison
    game.compare_bot(player1, player3)
    game.compare_bot(player2, player3)

    # Train bots against each other
    game.train(player1, player2, K=5_000, epochs=3)

    # Print out q-values
    fig, ax = player1.plot_q_values_round_1()
    fig.savefig('results/20-Epochs/player1-plot1.png')

    fig, ax = player1.plot_q_values_round_2()
    fig.savefig('results/20-Epochs/player1-plot2.png')

    fig, ax = player2.plot_q_values_round_1()
    fig.savefig('results/20-Epochs/player2-plot1.png')

    fig, ax = player2.plot_q_values_round_2()
    fig.savefig('results/20-Epochs/player2-plot2.png')

    # Evaluate bots
    game.compare_bot(player1, player3)
    game.compare_bot(player2, player3)