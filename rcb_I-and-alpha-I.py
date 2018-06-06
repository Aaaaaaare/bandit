import Casino
import Ghostwriter

import players
import pro_players
import arm_search

from datetime import datetime
import matplotlib.pyplot as plt
import pylab as pyl


import numpy as np
#reload(Casino)

def main():

    documented = True

    #========.. Create the report file ..========
    folder = './Output/'
    test_name = 'bandit-IMB_KB-'
    _timestamp = datetime.now().strftime("_%m%d_%H-%M-%S")
    report_name = folder + test_name + _timestamp + '.txt'
    
    if documented:
        ghost = Ghostwriter.Ghostwriter(report_name)

    #========.. Run specs ..========
    type_game = 'infinite arms, known budget'
    number_arms = 20
    num_plays = 100000
    budget = 3000
    type_reward = 'bernoulli'
    is_cost = True
    type_cost = 'normal'
    is_infinity = True
    if is_infinity:
        number_arms_s = str(number_arms) + '\tDeprecated'
    else:
        number_arms_s = str(number_arms)

    specs = {
            'type_game' : type_game,
            'number_plays' : num_plays,
            'budget' : budget,
            'number_arms' : number_arms_s,
            'type_reward' : type_reward,
            'is_cost' : is_cost,
            'is_infinity' : is_infinity,
            'type_cost' : type_cost}
    
    speech = '\nTesting the algoriths for the case of infinite arms and know budget\n'
    speech = speech + 'My algorithm, alpha-I and the RCB-I algorithm\n'

    if documented:
        ghost.write_(specs)
        ghost.write(speech)

    #========.. Create the bandit ..========
    # If is infinity, the number of arms is undefined... but 
    # i will approximate it as 2 times the budget for the creation 
    # of the casino... 
    # Then, each player will deal with the number of arms
    if is_infinity:
        number_arms_ = (int)(2 * budget)
    else:
        number_arms_ = number_arms
    casino = Casino.bandit(num_arms=number_arms_, type_b='bernoulli', is_cost=is_cost, infinity=is_infinity)

    #========.. Create the players ..========
    player_one = players.random_player(number_arms_, budget, is_infinity)
    pro_player_one = pro_players.RCB_I(number_arms_, budget, is_infinity)
    pro_player_two = arm_search.alpha_I(budget, {'is_infinity': True})

    all_players = [ player_one,
                    pro_player_one,
                    pro_player_two ]

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
                plotting_info[i, (int)(j/scale)] = p.get_prize()[0]
                indexes[i] = (int)(indexes[i]+1)
        r[i], c[i] = p.get_prize()
    

    #=======.. Print the result ..========
    #=======.. Casino info ..========
    double_print (ghost, documented, 'Reward type: \t{}'.format(casino.get_type_b()))
    double_print (ghost, documented, 'Casino number of arms: \t\t{}'.format(casino.get_number_arms()))
    double_print (ghost, documented, 'Casino best reward: \t%.5f' % casino.best_arm_reward())
    double_print (ghost, documented, 'Best arm: \t\t{}'.format(casino.best_arm()))


    #=======.. Results ..========
    for i, p in enumerate(all_players):
        double_print (ghost, documented, '========.. Player {} ..========'.format(p.get_id()))
        if is_infinity:
            double_print (ghost, documented, 'Arms played: \t{}'.format(p.get_num_arms()))
        double_print (ghost, documented, 'Total reward: \t{}'.format(r[i]))
        double_print (ghost, documented, 'Total cost: \t{}'.format(c[i]))
        double_print (ghost, documented, 'Total plays: \t{}'.format(p.get_total_plays()))
        double_print (ghost, documented, 'Final budget: \t%.5f' % p.remaining_valid_budget())
        double_print (ghost, documented, 'Best arm reward: \t{}'.format(p.best_arm_reward()))

    # for i, p in enumerate(all_players):
    #     double_print (ghost, documented, '========.. Player {} ..========'.format(p.get_id()))
    #     double_print (ghost, documented, 'reward \tpulls')
    #     for j in range(number_arms):
    #         double_print (ghost, documented, '{}\t{}'.format(p.rewards[j], p.pulls[j]))


    #========.. Arms who succedd ..========
    # double_print (ghost, documented, '========.. Arms who got it right ..========')
    # for p in all_players:
    #     if p.best_arm_casino() == casino.best_arm():
    #         double_print (ghost, documented, p.get_id())
    double_print (ghost, documented, '========.. Players Ranking ..========')
    for p_ in sorted(all_players, key=lambda player: player.get_prize()[0], reverse=True):
        double_print(ghost, documented, '{}\t\t{}'.format(p_.get_id(), p_.get_prize()[0]))
    #double_print (ghost, documented,  sorted(all_players, key=lambda player: player.regret(casino))[0].get_id() )

    #========.. Plot ..========
    for j, info in enumerate(plotting_info):
        lim = (int)(indexes[j])
        plt.plot(info[:lim], label=all_players[j].get_id())
    plt.legend()
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('reward')
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