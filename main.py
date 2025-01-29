from math import factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
import random
from typing import Tuple, Union

def comb(n: int, r: int) -> float:
    return factorial(n) / (factorial(n - r) * factorial(r))

def probab_game(p: float) -> float:
    p_4_0 = comb(4,4) * p**4
    p_4_1 = (comb(5,4) - comb(4,4)) * p**4 * (1-p)
    p_4_2 = (comb(6,4) - comb(5,4)) * p**4 * (1-p)**2
    deuce = comb(6,3) * (p**5 * (1-p)**3) / (1 - 2*p*(1-p))
    return p_4_0 + p_4_1 + p_4_2 + deuce

def chance_of_wining_point(
    p_first_is_good: float, 
    p_win_with_first_serve: float, 
    p_win_with_second_serve: float
    ) -> float:
    p = (p_first_is_good * p_win_with_first_serve) + (1 - p_first_is_good) * p_win_with_second_serve
    return p

def sim_point(p_s: float) -> bool:
    """Simulate point in tennis by drawing from uni dist

    Args:
        p_s (float): probability server wins point

    Returns:
        bool: True if server won, False if not
    """
    return random.uniform(0, 1) <= p_s

def sim_game(p_s: float, ppg: int = 4) -> Tuple[bool, list]:
    """Simulate game of tennis using just prob server wins point

    Args:
        p_s (float): probability server wins point
        ppg (int): points per game in case want to play with longer game length

    Returns:
        Tuple[bool, list]: tuple of True if server won game and list
        of tuples of points as the game progressed. E.g. if server won all
        points would return (True, [(1,0),(2,0),(3,0),(4,0)])
    """
    scores = []
    # s and r are points scored by server and returner
    s = 0
    r = 0
    # while game still going
    while (s < ppg) and (r < ppg):
        # simulate the point
        if sim_point(p_s):
            s += 1
        else:
            r += 1

        # add score tuple to the score list
        scores.append((s, r))

        # we need a catcher here if we get to 3-3
        # so that we can handle deuce
        if (s == (ppg - 1)) and (r == (ppg - 1)):
            # give a bit more space
            while (s < (ppg + 1)) and (r < (ppg + 1)):
                # simulate the point
                if sim_point(p_s):
                    s += 1
                else:
                    r += 1
                # add score tuple to the score list
                scores.append((s, r))

                # if we're at 4 all then bring us back to 3 all
                if (r == ppg) and (s == ppg):
                    s = ppg - 1
                    r = ppg - 1
            # if we've excited then must be game over after deuce
            if s == ppg + 1:
                return (True, scores)
            elif r == ppg + 1:
                return (False, scores)

    # if here then must have finished game pre-deuce
    # return True if server wins, false if returner
    if s == ppg:
        return (True, scores)
    else:
        return (False, scores)
    
def sim_set(p_s1: float, p_s2:float, player_to_start_serve: int, ppg: int = 4, gps: int = 6) -> Tuple[bool, list]:
    """Simulate set of tennis using just prob server wins game

    Args:
        p_s1 (float): probability player1 wins a point when serving
        p_s2 (float): probability player2 wins a point when serving
        ppg (int): points per game in case want to play with longer game length
        gps (int): games per set in case want to play with longer set length

    Returns:
        Tuple[bool, list]: tuple of True if server won set and list
        of tuples of games as the set progressed. E.g. if server won all
        games would return (True, [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)])
    """
    scores = []
    # s and r are games scored by server and returner
    s = 0
    r = 0
    total_points_played = 0
    # while set still going
    while ((s < gps) and (r < gps)) or (s==gps and r == gps-1) or (s == gps - 1 and r == gps):
    # or (s >= gps and abs(s - r) < 2) or (r >= gps and abs(s - r) < 2):
        if(player_to_start_serve == 1):
            p_s = p_s1 if (s + r) % 2 == 0 else (1 - p_s2)
        else:
            p_s = p_s1 if (s + r) % 2 == 1 else (1 - p_s2)
        # simulate the game
        game_result = sim_game(p_s, ppg)
        if game_result[0]:
            s += 1
        else:
            r += 1

        # add score tuple to the score list
        scores.append((s, r))
        total_points_played += len(game_result[1])

    # if here then must have finished set
    # return True if server wins, false if returner

    if s == gps and r == gps:
        player_to_start_serve_on_tiebreak = player_to_start_serve if (s + r) % 2 == 0 else (3 - player_to_start_serve)
        tiebreak_result = sim_tiebreak(p_s1, p_s2, player_to_start_serve_on_tiebreak)
        if tiebreak_result[0]:
            s += 1
        else:
            r += 1
        scores.append((s, r))
        
        total_points_played += len(tiebreak_result[1])
        return (tiebreak_result[0], scores, tiebreak_result[1], total_points_played, tiebreak_result[2])
    elif s >= gps:
        return (True, scores, [], total_points_played, 2)
    else:
        return (False, scores, [], total_points_played, 1)
    
def sim_tiebreak(p_s1: float, p_s2: float, player_to_start_serve: int, ppg: int = 7) -> Tuple[bool, list]:
    """Simulate tiebreak of tennis using just prob server wins point

    Args:
        p_s (float): probability server wins point
        ppg (int): points per game - the length of the tiebreak - usually 7

    Returns:
        Tuple[bool, list]: tuple of True if server won tiebreak and list
        of tuples of points as the tiebreak progressed. E.g. if server won all
        points would return (True, [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0)])
    """
    scores = []
    # s and r are points scored by server and returner
    s = 0
    r = 0
    # while tiebreak still going
    while ((s < ppg) and (r < ppg)) or (abs(s - r) < 2):
        res = (s + r) % 4
        # switch service every two points after first point
        if(player_to_start_serve == 1):
            p_s = p_s1 if res == 0 or res == 3 else (1 - p_s2)
        else:
            p_s = (1 - p_s2) if res == 0 or res == 3 else p_s1

        # simulate the point
        if sim_point(p_s):
            s += 1
        else:
            r += 1

        # add score tuple to the score list
        scores.append((s, r))

    # if here then must have finished tiebreak
    # return True if server wins, false if returner
    if s == ppg:
        return (True, scores, 2)
    else:
        return (False, scores, 1)
    
def sim_match(p_s1: float, p_s2:float, ppg: int = 4, gps: int = 6, sets: int = 3) -> bool:
    """Simulate match of tennis using just prob server wins game

    Args:
        p_s (float): probability server wins game
        ppg (int): points per game in case want to play with longer game length
        gps (int): games per set in case want to play with longer set length
        sets (int): number of sets

    Returns:
        bool: True if server won match, False if not
    """
    s = 0
    r = 0
    total_games_played = 0
    simulated_sets = []
    next_server = 1
    sets_needs_to_win = (sets // 2) + 1
    while ((s < sets_needs_to_win) and (r < sets_needs_to_win)):
        sim_set_result = sim_set(p_s1, p_s2, next_server, ppg, gps)
        next_server = sim_set_result[4]
        simulated_sets.append(sim_set_result)
        total_games_played += sim_set_result[3]
        if sim_set_result[0]:
            s += 1
        else:
            r += 1
    if (s == sets_needs_to_win):
        return (True, simulated_sets)
    else:
        return (False, simulated_sets)
     
# What is the average length of a game as a function of the probability of the server winning a point?
def average_game_length():
    probabilities = [i * 0.05 for i in range(21)]
    average_lengths = []
    num_simulations_per_game = 10000

    for p in probabilities:
        total_length = 0
        for _ in range(num_simulations_per_game):
            scores = sim_game(p)[1]
            total_length += len(scores)
        average_length = total_length / num_simulations_per_game
        average_lengths.append((p, average_length))

    return average_lengths

def plot_game_length():
    average_game_lengths = average_game_length()
    print(average_game_lengths)
    
    # Separate the list into x and y values
    x_values, y_values = zip(*average_game_lengths)

    # Create the plot
    plt.plot(x_values, y_values, marker='o')

    # Add labels and title
    plt.xlabel('Probability of Server Winning a Point')
    plt.ylabel('Average Game Length')
    plt.title('Average Game Length vs. Probability of Server Winning a Point')

    # Show the plot
    plt.show()

# What is the probability of winning a game as a function of the probability of the server winning a point?
def prob_of_winning_game():
    probabilities = [i * 0.05 for i in range(21)]
    prob_winning_game = []

    for p in probabilities:
        num_simulations = 10000
        num_wins = 0
        for _ in range(num_simulations):
            if sim_game(p)[0]:
                num_wins += 1
        prob_winning_game.append((p, num_wins / num_simulations))

    return prob_winning_game

def plot_prob_winning_game():
    prob_winning_game = prob_of_winning_game()
    print(prob_winning_game)
    
    # Separate the list into x and y values
    x_values, y_values = zip(*prob_winning_game)

    # Create the plot
    plt.plot(x_values, y_values, marker='o')

    # Add labels and title
    plt.xlabel('Probability of Server Winning a Point')
    plt.ylabel('Probability of Winning a Game')
    plt.title('Probability of Winning a Game vs. Probability of Server Winning a Point')

    # Show the plot
    plt.show()

# What is the average length of a set (total points played) and set winning percentage as a function of the probability of the server winning a point?
def set_averages():
    '''Returns a list of tuples containing the probability of the server winning a point, the average win percentage of the server, and the average length of a set'''
    
    probabilities = [i * 0.05 for i in range(21)]

    averages = []
    num_simulations_per_set = 500
    total_points = 0

    for p1 in probabilities:
        for p2 in probabilities:
            total_points = 0
            win_count = 0
            for _ in range(num_simulations_per_set):
                set_result = sim_set(p1, p2)
                if(set_result[0]):
                    win_count += 1
                # scores = set_result[1]
                # tiebreak_scores = set_result[2]
                total_points += set_result[3]
                # total_length += (len(scores) + len(tiebreak_scores))
            average_length = total_points / (num_simulations_per_set)
            average_wins = win_count / (num_simulations_per_set)
            averages.append((p1, p2, average_wins, average_length))

    # average number of points played in a set
    return averages

def set_averages_fixed_p2(p2):
    '''Returns a list of tuples containing the probability of the server winning a point, the average win percentage of the server, and the average length of a set'''
    
    probabilities = [i * 0.05 for i in range(21)]

    averages = []
    num_simulations_per_set = 500

    for p1 in probabilities:
        total_points = 0
        win_count = 0
        for _ in range(num_simulations_per_set):
            set_result = sim_set(p1, p2, 1)
            if(set_result[0]):
                win_count += 1
            # scores = set_result[1]
            # tiebreak_scores = set_result[2]
            total_points += set_result[3]
            # total_length += (len(scores) + len(tiebreak_scores))
        average_length = total_points / (num_simulations_per_set)
        average_wins = win_count / (num_simulations_per_set)
        averages.append((p1, average_wins, average_length))

    # average number of points played in a set
    return averages

def plot_3d(x_values, y_values, z_values, x_label, y_label, z_label, title):
    '''Plots a 3D plot of the given x, y, and z values'''
    # Create a grid of x and y values
    x = np.linspace(min(x_values), max(x_values), 100)
    y = np.linspace(min(y_values), max(y_values), 100)
    x, y = np.meshgrid(x, y)

    # Interpolate z values on the grid
    z = griddata((x_values, y_values), z_values, (x, y), method='cubic')

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Plot the surface
    ax.plot_surface(x, y, z, cmap='viridis')

    # Add labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)

    # Show the plot
    plt.show()
    
def plot_set_length_3d():
    '''Plots the average length of a set as a function of the probability of the server winning a point'''
    set_averages_result = set_averages()
    # print(set_averages_result)
    
    # Separate the list into x and y values
    #x_values, y_win_values, y_length_values = zip(*set_averages_result)
    x_p1_values, y_p2_values, z_win_values, z_length_values = zip(*set_averages_result)

    plot_3d(x_p1_values, y_p2_values, z_length_values, 
            'Probability of Server 1 Winning a Point', 
            'Probability of Server 2 Winning a Point', 
            'Average Set Length', 'Average Set Length')

    # Show the plot
    plt.show()

    # # Create the plot
    # plt.plot(x_values, y_length_values, marker='o')

    # # Add labels and title
    # plt.xlabel('Probability of Server Winning a Point')
    # plt.ylabel('Average Set Length')
    # plt.title('Average Set Length vs. Probability of Server Winning a Point')

    # # Show the plot
    # plt.show()

def plot_set_win_percentage_3d():
    '''Plots the average set win percentage as a function of the probability of the server winning a point'''
    set_averages_result = set_averages()
    # print(set_averages_result)
    
    # Separate the list into x and y values
    #x_values, y_win_values, y_length_values = zip(*set_averages_result)
    x_p1_values, y_p2_values, z_win_values, z_length_values = zip(*set_averages_result)

    plot_3d(x_p1_values, y_p2_values, z_win_values, 
            'Probability of Server 1 Winning a Point', 
            'Probability of Server 2 Winning a Point', 
            'Player1 Set Win Percentage', 
            'Player1 Set Win Percentage')
    
    # set_averages_result = set_averages()
    # print(set_averages_result)
    
    # # Separate the list into x and y values
    # x_values, y_win_values, y_length_values = zip(*set_averages_result)

    # # Create the plot
    # plt.plot(x_values, y_win_values, marker='o')

    # # Add labels and title
    # plt.xlabel('Probability of Server Winning a Point')
    # plt.ylabel('Average Set Win Percentage')
    # plt.title('Average Set Win Percentage vs. Probability of Server Winning a Point')

    # # Show the plot
    #plt.show()
    
def plot_set_win_percentage_2d():
    '''Plots the average set win percentage as a function of the probability of the server winning a point'''
    
    p2_values = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    for p2 in p2_values:
        set_averages_result = set_averages_fixed_p2(p2)
        x_p1_values, y_win_values, y_length_values = zip(*set_averages_result)
        plt.plot(x_p1_values, y_win_values, label='p2='+ str(p2), marker='o')

    # Add labels and title
    plt.xlabel('player 1 point win percentage when holding serve')
    plt.ylabel('player 1 set win percentage')
    
    plt.title('set win percentage')
    plt.legend()

    # Show the plot
    plt.show()

def plot_set_length_2d():
    '''Plots the average set length as a function of the probability of the server winning a point'''
    
    p2_values = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    for p2 in p2_values:
        set_averages_result = set_averages_fixed_p2(p2)
        x_p1_values, y_win_values, y_length_values = zip(*set_averages_result)
        plt.plot(x_p1_values, y_length_values, label='p2='+ str(p2), marker='o')

    # Add labels and title
    plt.xlabel('player 1 point win percentage when holding serve')
    plt.ylabel('average set length')
    
    plt.title('set length (total points played)')
    plt.legend()

    # Show the plot
    plt.show()

# plot_set_win_percentage_2d()
# plot_set_win_percentage_3d()

# plot_set_length_2d()


#Sinner vs Zverev
#player 1 stats
p_first_serve_is_good1:float = 0.6 
p_win_with_first_serve1:float = 0.84 
p_win_with_second_serve1:float = 0.63

p_first_serve_is_good2:float = 0.68 
p_win_with_first_serve2:float = 0.69 
p_win_with_second_serve2:float = 0.5

p1 = p_first_serve_is_good1 * p_win_with_first_serve1 + (1 - p_first_serve_is_good1) * p_win_with_second_serve1
p2 = p_first_serve_is_good2 * p_win_with_first_serve2 + (1 - p_first_serve_is_good2) * p_win_with_second_serve2

# chance_of_wining_point(p_first_serve_is_good1, p_win_with_first_serve1, p_win_with_second_serve1)
# p2 = chance_of_wining_point(p_first_serve_is_good2, p_win_with_first_serve2, p_win_with_second_serve2)

print(p1, p2)

match_result = sim_match(p1, p2) # ppg: int = 4, gps: int = 6, sets: int = 3
for set_result in match_result[1]:
    print(set_result[0], set_result[1])

