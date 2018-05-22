import Casino
import Ghostwriter
import players
from datetime import datetime

import numpy as np
#reload(Casino)

def main():

    #========.. Create the report file ..========
    test_name = 'bandit'
    _timestamp = datetime.now().strftime("%m%d_%H-%M-%S")
    report_name = test_name + _timestamp + '.txt'
    #ghost = Ghostwriter.Ghostwriter(report_name)

    #========.. Run specs ..========
    type_game = 'finite'
    number_arms = 4
    num_plays = 50000
    type_reward = 'bernoulli'
    is_cost = False
    type_cost = 'normal'
    specs = {
            'type_game' : type_game,
            'number_plays' : num_plays,
            'number_arms' : number_arms,
            'type_reward' : type_reward,
            'is_cost' : is_cost,
            'type_cost' : type_cost}
    #ghost.write_(specs)

    #========.. Create the bandit ..========
    casino = Casino.bandit(num_arms=number_arms, type_b='bernoulli', is_cost=False, infinity=False)

    #========.. Create the players ..========
    player_one = players.random_player(number_arms)
    player_two = players.eps_greedy_player(number_arms)
    player_three = players.softmax_player(number_arms)
    player_four = players.ucb1(number_arms)
    player_five = players.ucb_v(number_arms)

    all_players = [ player_one,
                    player_two,
                    player_three,
                    player_four,
                    player_five ]

    r = np.zeros(len(all_players))
    c = np.zeros(len(all_players))

    #========.. play with all the players ..========
    for i, p in enumerate(all_players):
        for j in range(num_plays):
            p.play(casino)
        r[i], c[i] = p.get_prize()
    

    #=======.. Print the result ..========
    #=======.. Casino info ..========
    print ('Reward type: {}'.format(casino.get_type_b()))
    print ('Casino best reward: %.5f' % casino.best_arm_reward())
    print ('Best arm: {}'.format(casino.best_arm()))


    #=======.. Results ..========
    for i, p in enumerate(all_players):
        print ('========.. Player {} ..========'.format(p.get_id()))
        print ('Total reward: {}'.format(r[i]))
        #print ('Total cost: {}'.format(c[i]))
        #print ('Regret: %.5f' % p.regret(casino))
        print ('Best arm: {}'.format(p.best_arm()))

    casino.print_all_arms()

    for i, p in enumerate(all_players):
        print ('======================')
        print ('reward \tpulls')
        for j in range(number_arms):
            print ('{}\t{}'.format(p.rewards[j], p.pulls[j]))

    print ('END')


    #ghost.write('hola')

if __name__ == "__main__":
    main()







