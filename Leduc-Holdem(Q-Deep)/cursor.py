import matplotlib.pyplot as plt
import torch
import numpy as np
import random

class CursorPlayer:
    def __init__(self):
        pass

    def plot_q_values_round_1(self, figsize=(14, 4)):
        """
        Creates a heatmap showing Q-values for 'c' (check) and 'b' (bet) actions
        across Round 1 game states. The neural network is queried for each state-action pair.
        """
        # Define all possible game states
        priv_cards = ["11", "12", "13"]
        
        # Round 1 histories (before public card is revealed)
        round1_histories = [
            (),                    # Player 1's first move
            ("c",),                # After P1 checks
            ("b",),                # After P1 bets
            ("c", "b"),            # After P1 checks, P2 bets
        ]
        
        # Build all state labels and corresponding tensors
        states = []
        labels = []
        
        # Round 1 states (pub_card = None)
        for priv in priv_cards:
            for hist in round1_histories:
                hist_str = "".join(hist) if hist else "start"
                label = f"{priv} | {hist_str}"
                state_tensor = self.state_tensor(priv, None, hist)
                states.append(state_tensor)
                labels.append(label)
        
        # Query the network for Q-values
        q_values = []
        self.network.eval()
        
        with torch.no_grad():
            for state in states:
                q_check = self.network(torch.cat([state, self.actions['c']])).item()
                q_bet = self.network(torch.cat([state, self.actions['b']])).item()
                q_values.append([q_check, q_bet])
        
        q_values = np.array(q_values).T  # Transpose: rows=actions, cols=states
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(q_values, aspect='auto', cmap='RdYlGn')
        
        # Set axis labels
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Check (c)', 'Bet (b)'])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Q-Value (Expected Reward)')
        
        # Add value annotations
        for i in range(2):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{q_values[i, j]:.2f}',
                              ha='center', va='center', color='black', fontsize=8)
        
        ax.set_ylabel('Action')
        ax.set_xlabel('Game State (Card | History)')
        ax.set_title('Round 1: Neural Network Q-Values by Game State and Action')
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax

    def plot_q_values_round_2(self, figsize=(16, 5), histories_per_card=3):
        """
        Creates a heatmap showing Q-values for 'c' (check) and 'b' (bet) actions
        across Round 2 game states. The neural network is queried for each state-action pair.
        
        Args:
            figsize: Figure size tuple
            histories_per_card: Number of random histories to sample per priv/pub card combo
        """
        priv_cards = ["11", "12", "13"]
        pub_cards = ["11", "12", "13"]
        
        # Round 2 histories (after public card is revealed)
        round2_histories = [
            ("c", "c", "*"),           # Both checked R1
            ("b", "b", "*"),           # Both bet R1
            ("c", "b", "b"),           # P1 check, P2 bet, P1 call
            ("c", "c", "*", "c"),      # R1: cc, R2: P1 checks
            ("c", "c", "*", "b"),      # R1: cc, R2: P1 bets
            ("b", "b", "*", "c"),      # R1: bb, R2: P1 checks
            ("b", "b", "*", "b"),      # R1: bb, R2: P1 bets
            ("c", "b", "b", "c"),      # R1: cbb, R2: P1 checks
            ("c", "b", "b", "b"),      # R1: cbb, R2: P1 bets
            ("c", "c", "*", "c", "b"), # R1: cc, R2: cb (P1 decides)
            ("b", "b", "*", "c", "b"), # R1: bb, R2: cb (P1 decides)
            ("c", "b", "b", "c", "b"), # R1: cbb, R2: cb (P1 decides)
        ]
        
        # Build all state labels and corresponding tensors
        states = []
        labels = []
        
        # Round 2 states (pub_card revealed) - sample random histories per card combo
        for priv in priv_cards:
            for pub in pub_cards:
                sampled_histories = random.sample(round2_histories, min(histories_per_card, len(round2_histories)))
                for hist in sampled_histories:
                    hist_str = "".join(hist).replace("*", ".")
                    label = f"{priv}/{pub} | {hist_str}"
                    state_tensor = self.state_tensor(priv, pub, hist)
                    states.append(state_tensor)
                    labels.append(label)
        
        # Query the network for Q-values
        q_values = []
        self.network.eval()
        
        with torch.no_grad():
            for state in states:
                q_check = self.network(torch.cat([state, self.actions['c']])).item()
                q_bet = self.network(torch.cat([state, self.actions['b']])).item()
                q_values.append([q_check, q_bet])
        
        q_values = np.array(q_values).T  # Transpose: rows=actions, cols=states
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(q_values, aspect='auto', cmap='RdYlGn')
        
        # Set axis labels
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Check (c)', 'Bet (b)'])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.35)
        cbar.set_label('Q-Value (Expected Reward)')
        
        # Add value annotations
        for i in range(2):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{q_values[i, j]:.2f}',
                              ha='center', va='center', color='black', fontsize=5)
        
        ax.set_ylabel('Action')
        ax.set_xlabel('Game State (Priv/Pub Card | History)')
        ax.set_title('Round 2: Neural Network Q-Values by Game State and Action')
        
        plt.tight_layout()
        
        return fig, ax