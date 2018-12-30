import random, util, math, timeit
from game import Agent
from game import Actions
from game import Directions

#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None


    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        """
        # Collect legal moves and successor states
        start = timeit.default_timer()
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current GameState (pacman.py) and the proposed action
        and returns a number, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
    """
    return gameState.getScore()


######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
    """

    The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

    A GameState specifies the full game state, including the food, capsules, agent configurations and more.
    Following are a few of the helper methods that you can use to query a GameState object to gather information about
    the present state of Pac-Man, the ghosts and the maze:

    gameState.getLegalActions():
    gameState.getPacmanState():
    gameState.getGhostStates():
    gameState.getNumAgents():
    gameState.getScore():
    The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
    """

    def bonusCalc(state):
        if state.scaredTimer > 3:
            return 15/util.manhattanDistance(state.getPosition(), gameState.getPacmanPosition())
        return 0

    def minDistList(grid,pos):
        minDist = float('inf')
        for food in grid:
            minDist = min(minDist,util.manhattanDistance(food,pos))
        return minDist

    def minDistGrid(grid,pos):
        minDist = float('inf')
        for w in range(grid.width):
            for h in range(grid.height):
                if grid[w][h]:
                    minDist = min(minDist,util.manhattanDistance((w,h),pos))
        return minDist

    bonusList = list(map(bonusCalc, gameState.getGhostStates()))
    return gameState.getScore() + sum(bonusList)+ 1/min(minDistGrid(gameState.getFood(),gameState.getPacmanPosition()),minDistList(gameState.getCapsules(), gameState.getPacmanPosition()))








#  ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent
    """

    def minMax(self, gameState, agent, depth):
        if gameState.isLose() or gameState.isWin() or len(gameState.getLegalActions()) == 0:
            return (gameState.getScore(),0)
        if depth == self.depth:
            return (self.evaluationFunction(gameState),0)
        actions = gameState.getLegalActions(agent)
        nextAgent = agent + 1
        if agent == gameState.getNumAgents() - 1:
            nextAgent = 0
            depth = depth + 1
        if agent == 0:
            curMax = (-float('inf'),0)
            currentActions = [curMax]
            for action in actions:
                temp = self.minMax(gameState.generateSuccessor(agent, action), nextAgent, depth)
                if temp[0] == curMax[0]:
                    currentActions.append((temp[0],action))
                elif temp[0] > curMax[0]:
                    curMax = (temp[0], action)
                    currentActions = [curMax]
            return random.choice(currentActions)
        else:
            curMin = (float('inf'),0)
            for action in actions:
                temp = self.minMax(gameState.generateSuccessor(agent, action), nextAgent, depth)
                curMin = (temp[0],action) if temp[0]<curMin[0] else curMin
            return curMin

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.getScore():
            Returns the score corresponding to the current state of the game

          gameState.isWin():
            Returns True if it's a winning state

          gameState.isLose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue

        """

        # BEGIN_YOUR_CODE
        return self.minMax(gameState,0,0)[1]
        # END_YOUR_CODE


######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning
    """

    def alpha_beta(self, gameState, agent, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin() or len(gameState.getLegalActions()) == 0:
            return (gameState.getScore(), 0)
        if depth == self.depth:
            return (self.evaluationFunction(gameState), 0)
        actions = gameState.getLegalActions(agent)
        nextAgent = agent + 1
        if agent == gameState.getNumAgents() - 1:
            nextAgent = 0
            depth = depth + 1
        if agent == 0:
            curMax = (-float('inf'), 0)
            currentActions = [curMax]
            for action in actions:
                temp = self.alpha_beta(gameState.generateSuccessor(agent, action), nextAgent, depth,alpha,beta)
                if temp[0] == curMax[0]:
                    currentActions.append((temp[0], action))
                elif temp[0] > curMax[0]:
                    curMax = (temp[0], action)
                    currentActions = [curMax]
                alpha = max(alpha, curMax[0])
                if curMax[0] >= beta:
                    return float('inf'), 0
            return random.choice(currentActions)
        else:
            curMin = (float('inf'), 0)
            for action in actions:
                temp = self.alpha_beta(gameState.generateSuccessor(agent, action), nextAgent, depth, alpha ,beta)
                curMin = (temp[0], action) if temp[0] < curMin[0] else curMin
                beta = min(curMin[0],beta)
                if curMin[0] <= alpha:
                    return -float('inf'), 0
            return curMin





    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        # BEGIN_YOUR_CODE
        return self.alpha_beta(gameState, 0, 0, -float('inf'), float('inf'))[1]
        # END_YOUR_CODE


######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def expectimax(self, gameState, agent, depth):
        if gameState.isLose() or gameState.isWin() or len(gameState.getLegalActions()) == 0:
            return (gameState.getScore(), 0)
        if depth == self.depth:
            return (self.evaluationFunction(gameState), 0)
        actions = gameState.getLegalActions(agent)
        nextAgent = agent + 1
        if agent == gameState.getNumAgents() - 1:
            nextAgent = 0
            depth = depth + 1
        if agent == 0:
            curMax = (-float('inf'), 0)
            currentActions = [curMax]
            for action in actions:
                temp = self.expectimax(gameState.generateSuccessor(agent, action), nextAgent, depth)
                if temp[0] == curMax[0]:
                    currentActions.append((temp[0], action))
                elif temp[0] > curMax[0]:
                    curMax = (temp[0], action)
                    currentActions = [curMax]
            return random.choice(currentActions)
        else:
            legalActions = []
            for action in actions:
                legalActions.append(self.expectimax(gameState.generateSuccessor(agent, action), nextAgent, depth))
            return random.choice(legalActions)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their legal moves.
        """

        return self.expectimax(gameState, 0, 0)[1]
        # END_YOUR_CODE


######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """
    def direcional_expectimax(self, gameState, agent, depth):
        if gameState.isLose() or gameState.isWin() or len(gameState.getLegalActions()) == 0:
            return (gameState.getScore(), 0)
        if depth == self.depth:
            return (self.evaluationFunction(gameState), 0)
        actions = gameState.getLegalActions(agent)
        nextAgent = agent + 1
        if agent == gameState.getNumAgents() - 1:
            nextAgent = 0
            depth = depth + 1
        if agent == 0:
            curMax = (-float('inf'), 0)
            currentActions = [curMax]
            for action in actions:
                temp = self.direcional_expectimax(gameState.generateSuccessor(agent, action), nextAgent, depth)
                if temp[0] == curMax[0]:
                    currentActions.append((temp[0], action))
                elif temp[0] > curMax[0]:
                    curMax = (temp[0], action)
                    currentActions = [curMax]
            return random.choice(currentActions)
        else:
            ghostState = gameState.getGhostState(agent)
            isScared = ghostState.scaredTimer > 0
            pacmanPosition = gameState.getPacmanPosition()
            pos = gameState.getGhostPosition(agent)
            speed = 1
            if isScared: speed = 0.5

            actionVectors = [Actions.directionToVector(a, speed) for a in actions]
            newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
            distancesToPacman = [util.manhattanDistance(pos, pacmanPosition) for pos in newPositions]
            if isScared:
                bestScore = max(distancesToPacman)
                bestProb = 0.8
            else:
                bestScore = min(distancesToPacman)
                bestProb = 0.8
            succ_actions = []
            for action in actions:
                succ_actions.append(self.direcional_expectimax(gameState.generateSuccessor(agent, action), nextAgent, depth))
            bestActions = [action for action, distance in zip(succ_actions, distancesToPacman) if distance == bestScore]
            # Construct distribution
            dist = util.Counter()
            for a in bestActions: dist[a] = bestProb / len(bestActions)
            for a in succ_actions: dist[a] += (1 - bestProb) / len(succ_actions)

            dist.normalize()
            return util.chooseFromDistribution(dist)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
        """

        # BEGIN_YOUR_CODE
        return self.direcional_expectimax(gameState, 0, 0)[1]
        # END_YOUR_CODE


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
    """
      Your competition agent
    """
    def alpha_beta_c(self, gameState, agent, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin() or len(gameState.getLegalActions()) == 0:
            return (gameState.getScore(), 0)
        if depth == self.depth:
            return (self.evaluationFunction(gameState), 0)
        actions = gameState.getLegalActions(agent)
        nextAgent = agent + 1
        if agent == gameState.getNumAgents() - 1:
            nextAgent = 0
            depth = depth + 1
        if agent == 0:
            curMax = (-float('inf'), 0)
            currentActions = [curMax]
            for action in actions:
                temp = self.alpha_beta_c(gameState.generateSuccessor(agent, action), nextAgent, depth,alpha,beta)
                if temp[1] == Directions.REVERSE[gameState.getPacmanState().getDirection()]:
                   temp = temp[0]-20, temp[1]
                if temp[0] == curMax[0]:
                    currentActions.append((temp[0], action))
                elif temp[0] > curMax[0]:
                    curMax = (temp[0], action)
                    currentActions = [curMax]
                alpha = max(alpha, curMax[0])
                if curMax[0] >= beta:
                    return float('inf'), 0
            return random.choice(currentActions)
        else:
            curMin = (float('inf'), 0)
            for action in actions:
                temp = self.alpha_beta_c(gameState.generateSuccessor(agent, action), nextAgent, depth, alpha ,beta)
                curMin = (temp[0], action) if temp[0] < curMin[0] else curMin
                beta = min(curMin[0],beta)
                if curMin[0] <= alpha:
                    return -float('inf'), 0
            return curMin


    def getAction(self, gameState):
        """
          Returns the action using self.depth and self.evaluationFunction

        """

        # BEGIN_YOUR_CODE
        return self.alpha_beta_c(gameState, 0, 0, -float('inf'), float('inf'))[1]
        # END_YOUR_CODE
