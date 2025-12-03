import random

# Defines payoff for player 1, if want to get opposite take -1
# expects player1 to be first
PAYOFF = {
    ('', 'p', 'p') : lambda x, y: 1 if x > y else -1,
    ('', 'p', 'b', 'p') : lambda x, y: -1,
    ('', 'p', 'b', 'b') : lambda x, y: 2 if x > y else -2,
    ('', 'b', 'p') : lambda x, y: +1,
    ('', 'b', 'b') : lambda x, y: +2 if x > y else -2
}

class Player:
    def __init__(self):
        pass

    def valid_actions(self, h):
        """
        Given history (h) returns possible moves

        args:
            string with each charecter denoting prev move
        """
        return ("p", "b")

    def payoff(self, history, cards):
        """
        returns payoff how it was defined above
        """
        func = PAYOFF.get(history, None)

        if func is None:
            return None

        return func(cards[0], cards[1])

class Game_Node:
    """
    This class stores information about each game state
    
    args:
        self.info_key:
            contains all the information from before
        self.actions:
            lists all the possible actions
    """

    def __init__(self, info_key, actions):
        self.info_key = info_key
        self.actions = actions
        self.regret = {a : 0.0 for a in actions}

        # probability of player 0 selecting it multiplied by
        self.strategy_sum = {a : 0.0 for a in actions}

        # Special way to index to select correct regret alg
        self.regret_calc = {
            0 : lambda p0, p1, util, node_util: (p1 * (util - node_util)),
            1 : lambda p0, p1, util, node_util: (p0 * -(util - node_util)),
        }

        # Special way to index to select correct 
        self.strategy_sum_calc = {
            0 : lambda p0, p1, strategy: (p1 * strategy),
            1 : lambda p0, p1, strategy: (p0 * strategy),
        }

        # Records final_strategy
        self.final_strategy = None

    def regret_matching(self):
        """
        Returns a dict w/ {move : probability} 
        """
        options = {a : max(0, val) for a, val in self.regret.items()}
        s = sum(options.values())

        # If no optimal regret does nothing
        if s == 0:
            return {a : 1 / len(self.actions) for a in self.actions}

        # Returns the probability of each
        return {a : options[a] / s for a in self.actions}

    def update_values(self, player, actions, util, node_util, p0, p1):
        """
        Code for updating the values
        """
        current_strategy = self.regret_matching()

        for a in actions:
            self.regret[a] += self.regret_calc[player](p0, p1, util[a], node_util)
            self.strategy_sum[a] += self.strategy_sum_calc[player](p0, p1, current_strategy[a])

    def calc_final_strategy(self):
        # Explicit strategy if we learn something
        total = sum(self.strategy_sum.values())
        if (total > 0):
            self.final_strategy = {a : val / total for a, val in self.strategy_sum.items()}

        # Uniform strategy if we learn nothing
        else:
            self.final_strategy = {a : 1 / len(self.actions) for a in self.actions}

class CFR_Bot(Player):
    def __init__(self):
        super().__init__()
        self.node_list = {}

    def cfr_train(self, history, cards, p0, p1):
        """
        This runs cfr by playing with another player
        to figure out regret in each scenario and then 
        make the correct move off of that
        """

        # Checks if game enters a terminal state
        x = self.payoff(history, cards)
        if x is not None:
            return x

        # Figure out the player
        player = len(history) % 2

        # Retrive Saved information on each action
        card = cards[player]
        info_key = (card,) + history

        actions = self.valid_actions(history)
        node = self.get_node(info_key, actions)

        # get strategy from current regret sums
        strategy = node.regret_matching()

        # Get strategy
        util = {}
        node_util = 0
        
        # perform cfr updates
        for a in actions:
            next_hist = history + (a,)
            
            # Updates for player 0
            if player == 0:
                util[a] = self.cfr_train(
                    history=next_hist,
                    cards=cards,
                    p0=p0 * strategy[a],
                    p1=p1
                )

            # Updates for player 1
            else:
                util[a] = self.cfr_train(
                    history=next_hist,
                    cards=cards,
                    p0=p0,
                    p1=p1 * strategy[a]
                )

            node_util += strategy[a] * util[a]

        # Updates regrets
        node.update_values(player, actions, util, node_util, p0, p1)

        return node_util

    def get_node(self, info_key, actions):
        # Figures out if node is in out list
        if info_key not in self.node_list:
            self.node_list[info_key] = Game_Node(info_key, actions)

        # Returns the node
        return self.node_list[info_key]

    def get_node_play(self, info_key):
        return self.node_list[info_key]


    def play_game(self, history, card):
        """
        Code for playing the game with an opponent
        """

        # Get node for given state
        info_key = ((card,) + history)
        node = self.get_node_play(info_key)

        # Decide move
        return random.choices(
            list(node.final_strategy.keys()),
            weights=list(node.final_strategy.values()),
            k=1
        )[0]

    def finalize_strategy(self):
        """
        Get final strategy from each after training
        """
        for node in self.node_list.values():
            node.calc_final_strategy()

    def train(self, n=1_000, readout=100):
        """
        Calls CFR train for the game
        """
        main_cards = [1, 2, 3]

        # Training Loop
        for i in range(n):
            history = ("",)
            cards = random.sample(main_cards, k=2)
            self.cfr_train(history, cards, 1.0, 1.0)

            # Print out training rounds
            if i % readout == 0:
                print(f"Loop {i} completed")

        self.finalize_strategy()

class Heuristic_Bot:
    def __init__(self):
        # defining random preference for betting (b, p)
        
        self.game_preference = {
            # you start
            (3, "")      : (0.90, 0.10),
            (2, "")      : (0.50, 0.50),
            (1, "")      : (0.10, 0.90),
            
            # Other player passed
            (3, "", "p") : (0.97, 0.03),
            (2, "", "p") : (0.60, 0.40),
            (1, "", "p") : (0.03, 0.97),
            
            # Other player bet
            (3, "", "b") : (0.97, 0.03),
            (2, "", "b") : (0.50, 0.50),
            (1, "", "b") : (0.03, 0.97),

            # Opponent passed AFTER peek
            (3, "", "p", "b") : (0.98, 0.02),
            (2, "", "p", "b") : (0.65, 0.35),
            (1, "", "p", "b") : (0.05, 0.95),
        }

    def play_game(self, history, card):
        """
        We're going to encode moves with probabilities
        of betting at each point
        arg:
            info_set: tells whether previous player held or not
            cars: list indexed by player_num that says what card player has
        """
        return random.choices(('p', 'b'), k=1, weights=self.game_preference[(card,) + history])[0]

def play_game(player1, player2, rounds, balance):
    """
    takes in two players and then gives winning average
    of player 1
    """
    # Design array to hold players
    players = {
        1 : player1,
        2 : player2,
    }

    # Player 1 payoff
    for i in range(rounds):
        # select player to start at random
        start = random.choice([1, 2])
        cards = random.sample([1, 2, 3], k=2)

        # Gives first player card 0
        card0 = cards[0]
        card1 = cards[1]

        history = ('',)
        other = 3 - start
        move1 = players[start].play_game(history, card0)

        history = history + (move1,)
        move2 = players[other].play_game(history, card1)

        # Update history
        history = history + (move2,)

        # calculate payoff if we terminate after state
        payoff_func = PAYOFF.get(history, None)
        if payoff_func is not None:
            if start == 1:
                balance += payoff_func(move1, move2)
            else:
                balance += -payoff_func(move1, move2)
            continue

        # calculate 3rd move
        move3 = players[start].play_game(history, card0)

        # Update history
        history = history + (move3,)

        payoff_func = PAYOFF[history]

        if start == 1:
            balance += payoff_func(move1, move2)
        else:
            balance += -payoff_func(move1, move2)

    return balance

if __name__ == "__main__":
    bot1 = CFR_Bot()
    bot1.train(n=100_000, readout=10_000)

    bot2 = CFR_Bot()
    bot2.train(n=100_000, readout=10_000)

    initial_money = 1_000
    rounds = 10_000
    heuristic_bot = Heuristic_Bot()
    money = play_game(bot1, heuristic_bot, rounds, initial_money)

    print(f'Resulting money against heuristic bot: {money}')

    initial_money = 10_000
    rounds = 10_000
    money = play_game(bot1, bot2, rounds, initial_money)
    print(f'Resulting money against trained bot: {money}')
