import pacman
import layout
import graphicsDisplay
import sys
from io import StringIO
import csv

if __name__ == '__main__':
    """
       test OBJ
       """
    pacman_p = ['RandomExpectimaxAgent', 'DirectionalExpectimaxAgent']
    depth_p = '4'

    layouts = ['trickyClassic']
    depths = ['4']
    ghosts = ['DirectionalGhost','RandomGhost']
    layout = 'trickyClassic'
    std_out = sys.stdout
    fieldnames = ['Agent', 'layout', 'ghost', 'average score', 'average turn time']
    with open('ghost.csv', mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    for pacman_i in pacman_p:
        for ghost in ghosts:
            sys.argv.append('-p')
            sys.argv.append(pacman_i)
            sys.argv.append('-q')
            sys.argv.append('-k')
            sys.argv.append(depth_p)
            sys.argv.append('-l')
            sys.argv.append(layout)
            sys.argv.append('-n')
            sys.argv.append('2')
            sys.argv.append('-g')
            sys.argv.append(ghost)

            print(sys.argv)
            stream = StringIO()
            sys.stdout = stream
            print(sys.argv)
            # args_i = ['run.py', '-p', 'ReflexAgent', '-q']
            pacman.main()
            a = stream.getvalue()
            del sys.argv[:9]
            average_score = float(stream.getvalue().split("\n")[8].split(":")[1])
            average_time = float(stream.getvalue().split("\n")[12].split(":")[1])
            with open('ghost.csv', mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({'Agent': pacman_i, 'layout': layout, 'ghost': ghost, 'average score': average_score,
                                 'average turn time': average_time})
            stream.close()
            sys.stdout = std_out
        # sys.argv.pop()
        # sys.argv.pop()
