import random

class Player:
    def __init__(self):
        pass

    def valid_actions(self, h):
        """
        Given history (h) returns possible moves

        args:
            string with each charecter denoting prev move
        """
        if h == "":
            return ["pass", "bet"]
        elif h == "pass":
            return ["pass", "bet"]
        elif h == "bet":
            return ["pass", "bet"]
        elif h == "passbet":
            return ["pass", "bet"]

        raise ValueError(f"Unable to reach state w/ {h}")

    def payoff(self, history, cards):
        """
        Returns positive if we make money
        And returns negative if we fail to make
        any money - returns false if not in a terminal
        state
        """
        if history == "passpass":
            return 1 if cards[0] > cards[1] else -1
        elif history == "passbetbet" or history == "betbet":
            return 2 if cards[0] > cards[1] else -2
        elif history == "passbetpass":
            return -1
        elif history == "betpass":
            return 1
        else:
            return False

class Game_Node:
    """
    This class stores information about each game state
    
    args:
        self.info_key: 
            contains all the information from before
        self.actions: 
            lists all the possible actions
        self.regret: 
            says how much we regret a certain move when we've been in this game state 
            before. It essentially asks how much more utility would I have gotten if 
            I had always chosen this action compared to the strategy I always use
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

    def final_strategy(self):
        # Explicit strategy if we learn something
        total = sum(self.strategy_sum.values())
        if (total > 0):
            return {a : val / total for a, val in self.strategy_sum.items()}

        # Uniform strategy if we learn nothing
        return {a : 1 / len(self.actions) for a in self.actions}

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
        if (x := self.payoff(history, cards)):
            return x

        # Figure out the player
        player = len(history) % 2

        # Retrive Saved information on each action
        card = cards[player]
        info_key = (card, history)

        actions = self.valid_actions(history)
        node = self.get_node(info_key, actions)

        # get strategy from current regret sums
        strategy = node.regret_matching()

        # Get strategy
        util = {}
        node_util = 0
        
        # perform cfr updates
        for a in actions:
            next_hist = history + a
            
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

def train(player1, n = 100):
    """
    Code for training the cfr on an individual player
    """
    cards = [1, 2, 3]

    # Training Loop
    for i in range(n):
        history = ""
        cards = random.sample(cards, k=2)
        player1.cfr_train(history, cards, 1.0, 1.0)

        # Print out training rounds
        if i % 50 == 0:
            print(f"Loop {i} completed")

    return player1


class Bot(Player):
    def __init__(self):
        super().__init__()
        self.player_num = 1

    def game_logic(self, info_set, cards):
        """
        We're going to encode moves with probabilities
        of betting at each point
        arg:
            info_set: tells whether previous player held or not
            cars: list indexed by player_num that says what card player has
        """
        
        if cards[self.player_num] == 3:
            return 1.0
        elif cards[self.player_num] == 2 and info_set == "bet":
            return 0.25
        elif cards[self.player_num] == 2 and info_set == "pass":
            return 0.50
        elif cards[self.player_num] == 1 and info_set == "bet":
            return 0.10
        elif cards[self.player_num] == 1 and info_set == "pass":
            return 0.30

    def make_move(self, info_set, cards):
        """
        Returns the players move
        """
        return "bet" if random.random() < self.game_logic(info_set, cards) else "pass"

if __name__ == "__main__":
    bot = CFR_Bot()
    train(bot)
