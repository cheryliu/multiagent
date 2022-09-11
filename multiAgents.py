# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from cmath import inf
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodScore = 0
        ghostScore = 0

        newFoodList = newFood.asList()
        #print(newFoodList)
        for food in newFoodList:
            foodDist = util.manhattanDistance(food, newPos)
            #if foodDist < minFoodDist:
            #    minFoodDist = foodDist
            foodScore += 1 / foodDist # sum up all the distances to food

        #print(minFoodDist)
        i = 0
        for ghostPos in successorGameState.getGhostPositions():
            scareTime = newScaredTimes[i]
            ghostDist = util.manhattanDistance(ghostPos, newPos)
            if scareTime == 0 and ghostDist <= 1:  # when the ghost is not scared and gets too close to pacman!!!
                ghostScore -= 800
            else:
                ghostScore = 1/ ghostDist  # when the pacman is trying so hard to eat
            i += 1
        
        #return successorGameState.getScore() + 1 / minFoodDist + ghostScore
        return successorGameState.getScore() + foodScore + ghostScore


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def value(gameState, depth, agentIndex):
            if depth == 0:                                                  # when in terminal state
                return self.evaluationFunction(gameState)
            elif agentIndex == 0:                                           # when the agent is pacman 
                return maximize(gameState, agentIndex, depth)
            else:                                                           # when the agent is ghost
                return minimize(gameState, agentIndex, depth)
        
        def maximize(gameState, agentIndex, depth):
            v = -1E9
            
            if agentIndex == gameState.getNumAgents() - 1:
                nextAgentIndex = 0
                depth -= 1
            else:
                nextAgentIndex = agentIndex + 1

            actions = gameState.getLegalActions(agentIndex)

            if gameState.isWin() or gameState.isLose() or len(actions) == 0:
                return self.evaluationFunction(gameState) # base case

            for action in actions: # recursion on the successors
                successor = gameState.generateSuccessor(agentIndex, action)
                newScore = value(successor, depth, nextAgentIndex)
                v = max(v, newScore)
            return v

        def minimize(gameState, agentIndex, depth):
            v = 1E9

            if agentIndex == gameState.getNumAgents() - 1:
                nextAgentIndex = 0
                depth -= 1
            else:
                nextAgentIndex = agentIndex + 1

            actions = gameState.getLegalActions(agentIndex)

            if gameState.isWin() or gameState.isLose() or len(actions) == 0:
                return self.evaluationFunction(gameState) # base case

            for action in actions: # recursion on the successors
                successor = gameState.generateSuccessor(agentIndex, action)
                newScore = value(successor, depth, nextAgentIndex)
                v = min(v, newScore)
            return v

        depth = self.depth
        agentIndex = 0

        actions = gameState.getLegalActions(agentIndex)
        minimax_scores = [value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1) for action in actions] # generate all minimax scores

        best_indices = [] # looks for the best score 
        best_score = max(minimax_scores)
        for i, score in enumerate(minimax_scores):
            if score == best_score:
                best_indices.append(i)

        return actions[best_indices[0]]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(gameState, depth, agentIndex, alpha, beta):
            if depth == 0:                                                  # when in terminal state
                return self.evaluationFunction(gameState)
            elif agentIndex == 0:                                           # when the agent is pacman 
                return maximize(gameState, agentIndex, depth, alpha, beta)
            else:                                                           # when the agent is ghost
                return minimize(gameState, agentIndex, depth, alpha, beta)
        
        def maximize(gameState, agentIndex, depth, alpha, beta):
            v = -1E9

            if agentIndex == gameState.getNumAgents() - 1:
                nextAgentIndex = 0
                depth -= 1
            else:
                nextAgentIndex = agentIndex + 1

            actions = gameState.getLegalActions(agentIndex)

            if gameState.isWin() or gameState.isLose() or len(actions) == 0:
                v = self.evaluationFunction(gameState) # base case comparision
                if v > beta: 
                    return v
                alpha = max(alpha, v)
                return v

            for action in actions: # recursion on the successors
                successor = gameState.generateSuccessor(agentIndex, action)
                newScore = value(successor, depth, nextAgentIndex, alpha, beta)
                v = max(v, newScore)
                if v > beta: 
                    return v
                alpha = max(alpha, v)
            return v

        def minimize(gameState, agentIndex, depth, alpha, beta):
            v = 1E9

            if agentIndex == gameState.getNumAgents() - 1:
                nextAgentIndex = 0
                depth -= 1
            else:
                nextAgentIndex = agentIndex + 1

            actions = gameState.getLegalActions(agentIndex)

            if gameState.isWin() or gameState.isLose() or len(actions) == 0:
                v = self.evaluationFunction(gameState) # base case comparision
                if v < alpha:
                    return v
                beta = min(beta, v)
                return v

                
            for action in actions: # recursion on the successors
                successor = gameState.generateSuccessor(agentIndex, action)
                newScore = value(successor, depth, nextAgentIndex, alpha, beta)
                v = min(v, newScore)
                if v < alpha: 
                    return v
                beta = min(beta, v)
            return v

        depth = self.depth
        agentIndex = 0
        alpha = -1E9    # initial alpha
        beta = 1E9      # initial beta

        actions = gameState.getLegalActions(agentIndex)
        alphaBeta =[]

        for action in actions:
            alphaBetaScore = value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1, alpha, beta) # pass the pruning function down to get the recursion
            if alphaBetaScore > beta:           # first layer is always maximize
                alphaBeta.append(alphaBetaScore)
                break
            alpha = max(alpha, alphaBetaScore)  # update the values in first layer
            alphaBeta.append(alphaBetaScore)


        best_score = max(alphaBeta)             #traverse through the alpha beta pruning to find the index of best score
        best_indices = []
        for i, score in enumerate(alphaBeta):
            if score == best_score:
               best_indices.append(i)
        return actions[best_indices[0]]

        
       

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(gameState, depth, agentIndex):
            if depth == 0:                                                  # when in terminal state
                return self.evaluationFunction(gameState)
            elif agentIndex == 0:                                           # when the agent is pacman 
                return maximize(gameState, agentIndex, depth)
            else:                                                           # when the agent is ghost
                return expected(gameState, agentIndex, depth)
        
        def maximize(gameState, agentIndex, depth):
            v = -1E9
            
            if agentIndex == gameState.getNumAgents() - 1:
                nextAgentIndex = 0
                depth -= 1
            else:
                nextAgentIndex = agentIndex + 1

            actions = gameState.getLegalActions(agentIndex)

            if gameState.isWin() or gameState.isLose() or len(actions) == 0:
                return self.evaluationFunction(gameState) # base case

            for action in actions: # recursion on the successors
                successor = gameState.generateSuccessor(agentIndex, action)
                newScore = value(successor, depth, nextAgentIndex)
                v = max(v, newScore)
            return v

        def expected(gameState, agentIndex, depth):
            v = 0

            if agentIndex == gameState.getNumAgents() - 1:
                nextAgentIndex = 0
                depth -= 1
            else:
                nextAgentIndex = agentIndex + 1

            actions = gameState.getLegalActions(agentIndex)

            if gameState.isWin() or gameState.isLose() or len(actions) == 0:
                return self.evaluationFunction(gameState) # base case

            for action in actions: # recursion on the successors
                p = 1 / len(actions) # probability of choosing one node
                successor = gameState.generateSuccessor(agentIndex, action)
                newScore = value(successor, depth, nextAgentIndex)
                v += p * newScore
            return v

        depth = self.depth
        agentIndex = 0

        actions = gameState.getLegalActions(agentIndex)
        expectimax_scores = [value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1) for action in actions] # generate all minimax scores

        best_indices = [] # looks for the best score 
        best_score = max(expectimax_scores)
        for i, score in enumerate(expectimax_scores):
            if score == best_score:
                best_indices.append(i)

        return actions[best_indices[0]]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Score calculating components:
    (1) 1 / minimum distance towards food ( name: foodScore )
    (2) the distance between pacman and ghost (ghostScore decreases when approaches ghosts)
    (3) current state score ( currentGameState.getScore() )
    (4) I also add weights in front of each varaible (0.2 for (1), 0.2 for (2) and 0.3 for (3))
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    foodScore = 0
    ghostScore = 0

    newFoodList = newFood.asList()
    
    minFoodDist = float("inf")
    for food in newFoodList:
        minFoodDist = min(util.manhattanDistance(food, newPos), minFoodDist)
    foodScore += 1 / minFoodDist # sum up all the distances to food

    i = 0
    for ghostPos in currentGameState.getGhostPositions():
        scareTime = newScaredTimes[i]
        ghostDist = util.manhattanDistance(ghostPos, newPos)
        if scareTime == 0 and ghostDist <= 1:  # when the ghost is not scared and gets too close to pacman!!!
            ghostScore -= 1000
        else:
            ghostScore = 1/ ghostDist  # when the pacman is trying so hard to eat
        i += 1
    return 0.3 * currentGameState.getScore() + 0.2 * foodScore + 0.2 * ghostScore

# Abbreviation
better = betterEvaluationFunction
