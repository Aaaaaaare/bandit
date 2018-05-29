import Casino
import Ghostwriter
import players
import pro_players
from datetime import datetime
import matplotlib.pyplot as plt
import pylab as pyl

import numpy as np
#reload(Casino)

def main():

    documented = True

    #========.. Create the report file ..========
    test_name = 'bandit'
    _timestamp = datetime.now().strftime("_%m%d_%H-%M-%S")
    report_name = test_name + _timestamp + '.txt'
    
    if documented:
        ghost = Ghostwriter.Ghostwriter(report_name)

    #========.. Run specs ..========
    type_game = 'finite'
    number_arms = 5
    num_plays = 100000
    budget = 1000
    type_reward = 'bernoulli'
    is_cost = True
    is_infinity = False
    type_cost = 'normal'
    specs = {
            'type_game' : type_game,
            'number_plays' : num_plays,
            'budget' : budget,
            'number_arms' : number_arms,
            'type_reward' : type_reward,
            'is_cost' : is_cost,
            'is_infinity' : is_infinity,
            'type_cost' : type_cost}
    
    if documented:
        ghost.write_(specs)

    #========.. Create the bandit ..========
    # If is infinity, the number of arms is undefined... but 
    # i will approximate it as 2 times the budget
    # Then, each player will decide the number of arms
    if is_infinity:
        number_arms_ = (int)(2 * budget)
    else:
        number_arms_ = number_arms
    casino = Casino.bandit(num_arms=number_arms_, type_b='bernoulli', is_cost=is_cost, infinity=is_infinity)

    #========.. Create the players ..========
    player_one = players.random_player(number_arms, budget, is_infinity)
    player_two = players.eps_greedy_player(number_arms, budget, is_infinity)
    player_three = players.softmax_player(number_arms, budget, is_infinity)
    player_four = players.ucb1(number_arms, budget, is_infinity)
    player_five = players.ucb_v(number_arms, budget, is_infinity)
    player_six = players.kl_ucb(number_arms, budget, is_infinity)
    pro_player_one = pro_players.RCB_I(number_arms, budget, is_infinity)
    #pro_player_two = pro_players.RCB_AIR(number_arms, budget, is_infinity)

    all_players = [ player_one,
                    player_two,
                    player_three,
                    player_four,
                    player_five,
                    player_six,
                    pro_player_one ]

    r = np.zeros(len(all_players))
    c = np.zeros(len(all_players))

    scale = 10
    plotting_info = np.zeros([len(all_players), (int)(num_plays/scale)])
    indexes = np.zeros(len(all_players))

    #========.. play with all the players ..========
    for i, p in enumerate(all_players):
        remaining_budget = budget
        for j in range(num_plays):
            p.play(casino)
            # If after playing the budget is exhausted, the play does not count
            if p.remaining_budget() < 0:
                double_print (ghost, documented, '--Player {} has finished his turn--'.format(p.get_id()))
                break
            if (j % scale) == 0:
                plotting_info[i, (int)(j/scale)] = p.regret(casino)
                indexes[i] = (int)(indexes[i]+1)
        r[i], c[i] = p.get_prize()
    

    #=======.. Print the result ..========
    #=======.. Casino info ..========
    double_print (ghost, documented, 'Reward type: \t{}'.format(casino.get_type_b()))
    double_print (ghost, documented, 'Casino best reward: \t%.5f' % casino.best_arm_reward())
    double_print (ghost, documented, 'Best arm: \t\t{}'.format(casino.best_arm()))


    #=======.. Results ..========
    for i, p in enumerate(all_players):
        double_print (ghost, documented, '========.. Player {} ..========'.format(p.get_id()))
        if is_infinity:
            double_print ((ghost, documented, 'Arms played: \t{}'.format(p.get_num_arms())))
        double_print (ghost, documented, 'Total reward: \t{}'.format(r[i]))
        double_print (ghost, documented, 'Total cost: \t{}'.format(c[i]))
        double_print (ghost, documented, 'Total plays: \t{}'.format(p.get_total_plays()))
        double_print (ghost, documented, 'Regret: \t%.5f' % p.regret(casino))
        double_print (ghost, documented, 'Final budget: \t%.5f' % p.remaining_valid_budget())
        double_print (ghost, documented, 'Best arm: \t{}'.format(p.best_arm_casino()))

    # casino.print_all_arms()

    for i, p in enumerate(all_players):
        double_print (ghost, documented, '========.. Player {} ..========'.format(p.get_id()))
        double_print (ghost, documented, 'reward \tpulls')
        for j in range(number_arms):
            double_print (ghost, documented, '{}\t{}'.format(p.rewards[j], p.pulls[j]))


    #========.. Arms who succedd ..========
    double_print (ghost, documented, '========.. Arms who got it right ..========')
    for p in all_players:
        if p.best_arm_casino() == casino.best_arm():
            double_print (ghost, documented, p.get_id())
    double_print (ghost, documented, '========.. Players Ranking ..========')
    for p_ in sorted(all_players, key=lambda player: player.regret(casino)):
        double_print(ghost, documented, p_.get_id())
    #double_print (ghost, documented,  sorted(all_players, key=lambda player: player.regret(casino))[0].get_id() )

    #========.. Plot ..========
    for j, info in enumerate(plotting_info):
        lim = (int)(indexes[j])
        plt.plot(info[:lim], label=all_players[j].get_id())
    plt.legend()
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('regret')
    plt.savefig(report_name[:-4] + '.png')
    plt.show()

    double_print (ghost, documented, 'END')

    #double_print (ghost, documented, pro_player_two.get_number_arms_played())

    if documented:
        end_timestamp = datetime.now().strftime("_%m%d_%H-%M-%S")
        ghost.write('ENDED SUCCESsFULLY at {}'.format(end_timestamp))
        print ('-Log in {}-'.format(report_name))

def double_print(g, d, s):
    print (s)
    if d:
        g.write(s)

if __name__ == "__main__":
    main()







