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
    pacman_p = 'RandomExpectimaxAgent'
    depth_p = '2'
    layouts = ['minimaxClassic', 'trappedClassic', 'testClassic', 'smallClassic',
               'originalClassic', 'openClassic',  'mediumClassic',
               'contestClassic', 'trickyClassic', 'capsuleClassic']
    depths = ['2' ,'3' ,'4']


    std_out = sys.stdout
    fieldnames = ['Agent', 'layout', 'depth', 'average score', 'average turn time']
    with open('ExpectimaxAgent.csv', mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    for layout in layouts:
        for depth in depths:
            sys.argv.append('-p')
            sys.argv.append(pacman_p)
            sys.argv.append('-q')
            sys.argv.append('-k')
            sys.argv.append(depth)
            sys.argv.append('-l')
            sys.argv.append(layout)
            sys.argv.append('-n')
            sys.argv.append('7')

            print(sys.argv)
            stream = StringIO()
            sys.stdout = stream
            print(sys.argv)
            # args_i = ['run.py', '-p', 'ReflexAgent', '-q']
            pacman.main()

            del sys.argv[:9]
            average_score = float(stream.getvalue().split("\n")[8].split(":")[1])
            average_time = float(stream.getvalue().split("\n")[12].split(":")[1])
            with open('ExpectimaxAgent.csv', mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({'Agent': pacman_p, 'layout': layout, 'depth': depth, 'average score': average_score,
                                 'average turn time': average_time})
            stream.close()
            sys.stdout = std_out
        # sys.argv.pop()
        # sys.argv.pop()
