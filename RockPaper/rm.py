import random
import copy

# Index w/ opponent 1, then second
WINS = {
        'rock' : {'rock' : 0, 'paper' : 1, 'scissors' : -1},
        'paper' : {'rock' : -1, 'paper' : 0, 'scissors' : 1},
        'scissors' : {'rock' : 1, 'paper' : -1, 'scissors' : 0}
    }

class Agent:
    moves = ['rock', 'paper', 'scissors']

    # Index w/ Opponent 1, then Oppenent 2
    # Reversed from usual order
    game_utility = { 
        'rock' : {'rock' : 0, 'paper' : 1, 'scissors' : -1},
        'paper' : {'rock' : -1, 'paper' : 0, 'scissors' : 1},
        'scissors' : {'rock' : 1, 'paper' : -1, 'scissors' : 0}
    }

    def make_move(self):
        pass

    def update_regret(self, move):
        pass

    def __str__(self):
        return "Random Agent"

class RM_Agent(Agent):
    """
    Class that comes up with regret matchin
    algorithm
    """
    def __init__(self):
        self.rock_regret = 0
        self.paper_regret = 0
        self.scissors_regret = 0

        # Initilize with no previous move
        self.move = None

    def make_move(self):

        # Ensures all values greater than 0
        rock_regret = max(self.rock_regret, 0)
        paper_regret = max(self.paper_regret, 0)
        scissors_regret = max(self.scissors_regret, 0)

        # Get Probabilities
        total = rock_regret + paper_regret + scissors_regret

        if total == 0:
            probs = [1/3, 1/3, 1/3] 
        else:
            probs = [
                rock_regret / total,
                paper_regret / total, 
                scissors_regret / total,
            ]

        # Make choice
        self.move = random.choices(Agent.moves, weights=probs)[0]
        return self.move

    def update_regret(self, opponent_move):
        prev_utility = Agent.game_utility[opponent_move][self.move]

        # Update utility w/ change (utility_[other move] - utility_[choice])
        self.rock_regret += Agent.game_utility[opponent_move]['rock'] - prev_utility
        self.paper_regret += Agent.game_utility[opponent_move]['paper'] - prev_utility
        self.scissors_regret += Agent.game_utility[opponent_move]['scissors'] - prev_utility

    def get_regret(self):
        print(f"Rock has regret: {self.rock_regret}")
        print(f"Paper has regret: {self.paper_regret}")
        print(f"Scissors has regret: {self.scissors_regret}")

    def __str__(self):
        return "RM Agent"


class Biased_Agent(Agent):
    def __init__(self, weights=None):

        if weights is None:
            self.weights = [1/3, 1/3, 1/3]
        else:
            self.weights = weights

    def make_move(self):
        return random.choices(Agent.moves, weights=self.weights, k=1)[0]
    
    def __str__(self):
        return "Biased Agent"


def run_rounds(Agent1, Agent2, n = 1000):

    # Define how you can win - gives points to player 1 
    # Assumes player1 moves go first
    player_1_wins = 0
    player_2_wins = 0

    for i in range(n):
        # Update moves
        player1_move = Agent1.make_move()
        player2_move = Agent2.make_move()

        # Update regret
        Agent1.update_regret(player2_move)
        Agent2.update_regret(player1_move)

        # Count wins
        if WINS[player1_move][player2_move] == 0:
            pass
        elif WINS[player1_move][player2_move] == 1:
            player_2_wins += 1
        elif WINS[player1_move][player2_move] == -1:
            player_1_wins += 1

    print(f"{Agent1} won: {player_1_wins} and {Agent2} won: {player_2_wins} times")

if __name__ == '__main__':
    rm_agent = RM_Agent()
    biased_agent = Biased_Agent([1/4, 1/4, 1/2])
    exp_agent = Exploitative_Agent(1)


    run_rounds(rm_agent, biased_agent, 1000)
    run_rounds(rm_agent, exp_agent, 1000)

    # print('Regret Matching: ')
    # rm_agent.get_regret()

    # print('Explotative: ')
    # exp_agent.get_regret()









