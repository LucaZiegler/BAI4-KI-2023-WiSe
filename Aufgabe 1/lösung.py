import copy
import time


def treeSearch(helper, strategy):
    rootNode = Node(helper, None, None, None)
    openList = [rootNode]   # Unprocessed nodes
    closedList = []         # Processed nodes

    startTime = time.perf_counter() # Start time of search

    while openList:         # Search until openList is empty
        currentNode = strategy.getNext(openList)    # 
        closedList.append(currentNode)

        if currentNode.value.isValidResult():
            # Found a valid solution, finishing search
            endTime = time.perf_counter()
            return SearchResult(True, endTime - startTime, closedList)
        else:
            # Continue
            strategy.expand(currentNode, openList)

    endTime = time.time_ns()
    # Did not found a valid solution and processed all ways
    return SearchResult(False, endTime - startTime, closedList)


class BFS:
    def expand(self, node, openList):
        """
        Expands the Helper within the given node and expands them according to the strategy
        :param node: the current node to be expanded
        :param openList: the open list, where new nodes are added
        :return: None
        """
        newMatrices = node.value.move()
        for m in newMatrices:
            createdNode = Node(Helper(m), node, 0, None)  # Create new node for the openlist
            node.addChildren(createdNode)  # Add the new node as children to the parent node
            openList.append(createdNode)  # Add the new node to the openlist

    def getNext(self, openList):
        """
        Returns the next node to be expanded, according to the strategy
        :param openList: the open list with all open nodes
        :return: the next node to be expanded
        """
        return openList.pop(0) # FIFO


class DFS:
    def expand(self, node, openList):
        """
        Expands the Helper within the given node and expands them according to the strategy
        :param node: the current node to be expanded
        :param openList: the open list, where new nodes are added
        :return: None
        """
        newMatrices = node.value.move()
        for m in newMatrices:
            createdNode = Node(Helper(m), node, 0, None)  # create new node for the openlist
            node.addChildren(createdNode)  # add the new node as children to the parent node
            openList.append(createdNode)  # add the new node to the openlist

    def getNext(self, openList):
        """
        Returns the next node to be expanded, according to the strategy
        :param openList: the open list with all open nodes
        :return: the next node to be expanded
        """
        return openList.pop() # FILO


class IDS:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def expand(self, node, openList):
        """
        Expands the Helper within the given node and expands them according to the strategy
        :param node: the current node to be expanded
        :param openList: the open list, where new nodes are added
        :return: None
        """
        if node.depth < self.max_depth:
            newMatrices = node.value.move()
            for m in newMatrices:
                createdNode = Node(Helper(m), node, 0, None)  # create new node for the openlist
                node.addChildren(createdNode)  # add the new node as children to the parent node
                openList.append(createdNode)  # add the new node to the openlist

    def getNext(self, openList):
        """
        Returns the next node to be expanded, according to the strategy
        :param openList: the open list with all open nodes
        :return: the next node to be expanded
        """
        return openList.pop()


class AStar:
    def expand(self, node, openList):
        """
        Expands the Helper within the given node and expands them according to the strategy
        :param node: the current node to be expanded
        :param openList: the open list, where new nodes are added
        :return: None
        """
        newMatrices = node.value.move()
        for m in newMatrices:
            newHelper = Helper(m)
            expected_rest_cost = self.heuristicIdea(newHelper) # 0
            total_cost = expected_rest_cost + node.depth + 1
            createdNode = Node(newHelper, node, total_cost, None)  # create new node for the openlist
            node.addChildren(createdNode)  # add the new node as children to the parent node

            i = 0

            while i < len(openList) or len(openList) == 0:
                if len(openList) == 0 or openList[i].weight < total_cost or i == len(openList) - 1:
                    openList.insert(i, createdNode)  # add the new node to the openlist
                    i = len(openList) + 1
                else:
                    i += 1

    def getNext(self, openList):
        """
        Returns the next node to be expanded, according to the strategy
        :param openList: the open list with all open nodes
        :return: the next node to be expanded
        """
        return openList.pop()

    def heuristicIdea(self, helper):
        return 15 - helper.getCurPathLength()


class SearchResult:
    def __init__(self, success, runtime, solution):
        self.success: bool = success
        self.runtime = runtime
        self.solution: list = solution

    def __str__(self):
        return (f"------------\n"
                f"Solution found? {self.success}\n"
                f"Runtime: {self.runtime}\n"
                f"Visited nodes: {len(self.solution)}\n"
                f"Final matrix: {self.solution[-1].value.matrix}")


class Node:
    def __init__(self, value, parent, weight, children):
        """
        Initialises the new Node
        :param value: the value stored inside the node
        :param parent: reference to the parent node
        :param weight: the weight of the node
        :param children: the child nodes of this node
        """
        self.value = value
        self.parent = parent
        self.weight = weight
        self.children = children if children is not None else []
        self.depth = self.parent.depth + 1 if self.parent is not None else 0

    def addChildren(self, node):
        """Ads the given node as a child
        :param node: the node to be added
        """
        self.children.append(node)


class Helper:
    def __init__(self, start_matrix):
        """
        :param start_matrix: the matrix as an array [[][]]
        """
        self.matrix: list = start_matrix
        self.empty_pos: list = self.findPos(0)
        self.start_pos: list = self.findPos(1)

    def move(self):
        """
        Moves the blank space through the Helper array in all possible direction
        :return: a list over all newly created arrays
        """
        empty_pos_x: int = self.empty_pos[0]
        empty_pos_y: int = self.empty_pos[1]
        newMatrices: list = []

        if empty_pos_x - 1 >= 0:
            new_matrix = copy.deepcopy(self.matrix)
            new_matrix[empty_pos_x][empty_pos_y] = new_matrix[empty_pos_x - 1][empty_pos_y]
            new_matrix[empty_pos_x - 1][empty_pos_y] = 0
            newMatrices.append(new_matrix)
        if empty_pos_y - 1 >= 0:
            new_matrix = copy.deepcopy(self.matrix)
            new_matrix[empty_pos_x][empty_pos_y] = new_matrix[empty_pos_x][empty_pos_y - 1]
            new_matrix[empty_pos_x][empty_pos_y - 1] = 0
            newMatrices.append(new_matrix)
        if empty_pos_x + 1 <= len(self.matrix) - 1:
            new_matrix = copy.deepcopy(self.matrix)
            new_matrix[empty_pos_x][empty_pos_y] = new_matrix[empty_pos_x + 1][empty_pos_y]
            new_matrix[empty_pos_x + 1][empty_pos_y] = 0
            newMatrices.append(new_matrix)
        if empty_pos_y + 1 <= len(self.matrix[empty_pos_x]) - 1:
            new_matrix = copy.deepcopy(self.matrix)
            new_matrix[empty_pos_x][empty_pos_y] = new_matrix[empty_pos_x][empty_pos_y + 1]
            new_matrix[empty_pos_x][empty_pos_y + 1] = 0
            newMatrices.append(new_matrix)

        return newMatrices

    def findPos(self, target):
        """
        Finds and returns the position of any given number within the Helper array
        :param target: the number to be found
        :return: an array with the coordinates of the number: [x , y]
        :raises NotInMatrixException
        """
        for i in range(0, len(self.matrix)):
            for j in range(0, len(self.matrix[i])):
                if target == self.matrix[i][j]:
                    return [i, j]
        raise NotInMatrixException

    def isValidResult(self):
        """
        Checks if the current configuration of the array is a valid solution for the Helper: i.e. it is possible
        for the chess knight to go from one to 15 within the array
        :return: true if the configuration is valid, false if not
        """
        reachable = True  # path with only one always exists
        next_target = 2  # starting from one the number two is the next to be reached
        current_pos = self.start_pos  # pos of the one in the array

        while reachable and next_target <= 15:
            reachable = self.isReachable(current_pos, next_target)  # check if path is continuing
            current_pos = self.findPos(next_target)  # set current position to the right number
            next_target += 1  # set next target

        return reachable

    def getCurPathLength(self):
        """
        Returns the current max path length for the chess knight starting from the number one
        :return: the number of possible steps
        """
        next_target = 2
        current_pos = self.start_pos
        reachable = True
        while reachable:
            reachable = self.isReachable(current_pos, next_target)
            if not reachable:
                return next_target - 1
            current_pos = self.findPos(next_target)
            next_target += 1
        return next_target - 1  # evtl. exception

    def isReachable(self, source_pos, target_value):
        """
        Checks if the given number is reachable from the defined source, according to the
        possible movements of the chess knight

        :param source_pos: the starting point in the Helper array, pos 0 is the source_pos_x coordinate and pos 1 the source_pos_y coordinate
        :param target_value: the number to be reached from the source point
        :return: true if the point is reachable, false if not
        """
        source_pos_x = source_pos[0]
        source_pos_y = source_pos[1]

        if source_pos_x - 1 >= 0 and source_pos_y + 2 < len(self.matrix) and self.matrix[source_pos_x - 1][source_pos_y + 2] == target_value:
            return True
        elif source_pos_x - 1 >= 0 and source_pos_y - 2 >= 0 and self.matrix[source_pos_x - 1][source_pos_y - 2] == target_value:
            return True
        elif source_pos_x + 1 < len(self.matrix) and source_pos_y + 2 < len(self.matrix) and self.matrix[source_pos_x + 1][source_pos_y + 2] == target_value:
            return True
        elif source_pos_x + 1 < len(self.matrix) and source_pos_y - 2 >= 0 and self.matrix[source_pos_x + 1][source_pos_y - 2] == target_value:
            return True
        elif source_pos_x - 2 >= 0 and source_pos_y + 1 < len(self.matrix) and self.matrix[source_pos_x - 2][source_pos_y + 1] == target_value:
            return True
        elif source_pos_x - 2 >= 0 and source_pos_y - 1 >= 0 and self.matrix[source_pos_x - 2][source_pos_y - 1] == target_value:
            return True
        elif source_pos_x + 2 < len(self.matrix) and source_pos_y + 1 < len(self.matrix) and self.matrix[source_pos_x + 2][source_pos_y + 1] == target_value:
            return True
        elif source_pos_x + 2 < len(self.matrix) and source_pos_y - 1 >= 0 and self.matrix[source_pos_x + 2][source_pos_y - 1] == target_value:
            return True
        else:
            return False
        # An array with possible steps would be better


class NotInMatrixException(BaseException):
    pass


example_1 = [[15, 10, 3, 6], [4, 7, 14, 11], [0, 12, 5, 2], [9, 1, 8, 13]]
example_2 = [[15, 10, 3, 6], [4, 7, 14, 11], [12, 0, 5, 2], [9, 1, 8, 13]]
example_3 = [[15, 10, 3, 6], [4, 14, 0, 11], [12, 7, 5, 2], [9, 1, 8, 13]]

print("--BFS Breitensuche--")
print(treeSearch(Helper(example_1), BFS()))
print(treeSearch(Helper(example_2), BFS()))
print(treeSearch(Helper(example_3), BFS()))

print("\n\n--IDS Iterative Tiefensuche--")
print(treeSearch(Helper(example_1), IDS(4)))
print(treeSearch(Helper(example_2), IDS(4)))
print(treeSearch(Helper(example_3), IDS(4)))

print("\n\n--A-Star--")
print(treeSearch(Helper(example_1), AStar()))
print(treeSearch(Helper(example_2), AStar()))
print(treeSearch(Helper(example_3), AStar()))

print("\n\n--DFS Tiefensuche--")
print(treeSearch(Helper(example_1), DFS()))
print(treeSearch(Helper(example_2), DFS()))
print(treeSearch(Helper(example_3), DFS()))