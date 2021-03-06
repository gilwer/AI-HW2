from layout import getLayout
from pacman import *
from ghostAgents import *
from submission import  ReflexAgent, MinimaxAgent, AlphaBetaAgent, RandomExpectimaxAgent, CompetitionAgent
from textDisplay import *
from multiprocessing import Process, Lock, Pool
from itertools import product
import traceback

withDepthplayers = [CompetitionAgent]
withoutDepthplayers = []
depths = [2, 3, 4]
layouts = ['capsuleClassic', 'contestClassic', 'mediumClassic',
           'minimaxClassic', 'openClassic', 'originalClassic',
           'smallClassic', 'testClassic', 'trappedClassic', 'trickyClassic']
ghosts = [RandomGhost(1), RandomGhost(2)]


def processesFunction(run):
    # run = player, layout_name, filename, depth=1
    player = run[0]
    layout_name = run[1]
    filename = run[2]

    depth = 1
    runsNum = run[3]
    if len(run) == 5:
        depth = run[4]

    layout = getLayout(layout_name)
    if depth > 1:
        player.depth = depth

    try:
        games = runGames(layout, player, ghosts, NullGraphics(), 7, False, 0, False, 30)
    except: # catch *all* exceptions
        e = sys.exc_info()[0]
        lock.acquire()
        print ("ERROR!!!!!!!")
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        print("layout ", layout)
        print("player ", player)
        print("depth ", depth)
        traceback.print_exc()
        exit()
    scores = [game.state.getScore() for game in games]
    times = [game.my_avg_time for game in games]
    avg_score = sum(scores) / float(len(scores))
    avg_time = sum(times) / float(len(times))
    line = (player.__class__.__name__ + ',' +
            str(depth) + ',' +
            layout_name + ',' +
            '%.2f' % avg_score + ',' +
            '%.2f' % (avg_time * 1e6) + 'E-06\n')

    # Begin of critical code
    lock.acquire()
    try:
        with open('experiments.csv', 'a') as file_ptr:
            file_ptr.write(line)
        file_ptr.close()

        with open(filename, 'a') as file_ptr:
            file_ptr.write(line)
        file_ptr.close()

        print("we finished another run, yeayy!!! the number of run is: ", runsNum,"")
        print( "run line:", line, "")

    finally:
        lock.release()

    # End of critical code

    return

def init(l):
    global lock
    lock = l

if __name__ == '__main__':
    l = Lock()
    base = time.time()
    runsNum = 0
    runs = []

    filename = 'experiments.csv'
    if os.path.exists(filename):
        os.remove(filename)

    file_ptr = open(filename, 'w+')
    file_ptr.close()

    for layout in layouts:
        filename = 'results_' + layout + '.csv'
        if os.path.exists(filename):
            os.remove(filename)

    for layout in layouts:
        filename = 'results_' + layout + '.csv'
        file_ptr = open(filename, 'w+')
        file_ptr.close()

    for layout in layouts:
        filename = 'results_' + layout + '.csv'
        for player in withoutDepthplayers:
            runs.append((player(), layout, filename, runsNum))
            runsNum = runsNum + 1
    for d in depths:
        for player in withDepthplayers:
            for layout in layouts:
                runs.append((player(), layout, filename, runsNum, d))
                runsNum = runsNum + 1

    print(runsNum, len(runs))
    assert runsNum == len(runs)
    runsNum = len(runs)
    print("total number of runs we are about to do: ", runsNum)
    runsNum = 0
    numOfCPUs = os.cpu_count()
    print("numOfCPUs: ", numOfCPUs)

    pool = Pool(numOfCPUs, initializer=init, initargs=(l,))

    print("Be patient its gona take a while, I love coffie so meanwhile I recommend you to drink coffie")

    pool.map(processesFunction, runs)
    pool.close()
    pool.join()
    print('experiments time: ', (time.time() - base)/60, 'min')
    
