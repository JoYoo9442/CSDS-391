import random
import math


class ProgrammingAssignment1:
    """
    Puzzle class stores all the appropriate functions for
    Programming Assignment 1, as well as initializes space for
    the Puzzle state and the max nodes to find the solution
    """

    def __init__(self, start_state):
        self.start_puzzle = self.Puzzle(start_state, start_state.index(0))
        self.max_nodes = None
        self.goal_puzzle = self.Puzzle([0, 1, 2, 3, 4, 5, 6, 7, 8], 0)

    def set_state(self, state):
        self.start_puzzle = self.Puzzle(state, state.index(0))

    def print_state(self):
        print(self.start_puzzle.state)

    def move(self, old_puzzle, direction):
        """Move the blank tile in the state
        up, down, left, or right.

        Keyword arguments:
        puzzle - puzzle to move
        direction -- string representation of direction
        """
        puzzle = self.Puzzle(old_puzzle.state.copy(), old_puzzle.blank)

        match direction:
            case "up":
                if puzzle.blank < 3:
                    # print("Invalid Move")
                    pass
                else:
                    puzzle.state[puzzle.blank] = puzzle.state[puzzle.blank - 3]
                    puzzle.blank -= 3
                    puzzle.state[puzzle.blank] = 0

            case "down":
                if puzzle.blank > 5:
                    # print("Invalid Move")
                    pass
                else:
                    puzzle.state[puzzle.blank] = puzzle.state[puzzle.blank + 3]
                    puzzle.blank += 3
                    puzzle.state[puzzle.blank] = 0

            case "left":
                if puzzle.blank % 3 == 0:
                    # print("Invalid Move")
                    pass
                else:
                    puzzle.state[puzzle.blank] = puzzle.state[puzzle.blank - 1]
                    puzzle.blank -= 1
                    puzzle.state[puzzle.blank] = 0

            case "right":
                if (puzzle.blank) % 3 == 2:
                    # print("Invalid Move")
                    pass
                else:
                    puzzle.state[puzzle.blank] = puzzle.state[puzzle.blank + 1]
                    puzzle.blank += 1
                    puzzle.state[puzzle.blank] = 0

            case _:
                print("Invalid Direction...")
                return -1

        return puzzle

    def randomize_state(self, puzzle, n):
        for i in range(n):
            puzzle = self.move(puzzle, random.choice(
                self.list_valid_move(puzzle)))

        return puzzle

    def solve_A_star(self, heuristic):
        if self.start_puzzle.state == self.goal_puzzle.state:
            print("start state is goal state")
            return True

        def f(node):
            return node.cost + heuristic(node)
        generated_nodes = 1

        node = self.Node(
                self.Puzzle(self.start_puzzle.state, self.start_puzzle.blank),
                [])
        frontier = [node]
        reached = {str(node): node}

        while frontier:
            if generated_nodes > self.max_nodes:
                print(f'Exceeded the maximum node count of {self.max_nodes}.')
                return False
            
            node = frontier.pop(0)

            # if goal state is found
            if node.value.state == self.goal_puzzle.state:
                print("-----A-star-----")
                print("Number of moves: " + str(node.cost))
                print("----------------")
                print("Path taken:", node.path)
                print("----------------")
                print("Nodes generated:", generated_nodes)
                print("----------------")
                return True

            # Else keep going
            for child in self.expand(node):
                s = str(child)
                if (s not in reached or
                        child.cost < reached[s].cost):
                    reached[s] = child
                    frontier.append(child)
                    frontier.sort(key=f)
                generated_nodes += 1

        print("No solution found")

    def expand(self, node):
        valid_moves = self.list_valid_move(node.value)
        child_list = []

        for move in valid_moves:
            child = self.Node(self.move(node.value, move), node.path + [move])
            child_list.append(child)

        return child_list

    def list_valid_move(self, puzzle):
        b = puzzle.blank
        valid_moves = []

        if not b < 3:
            valid_moves.append("up")
        if not b > 5:
            valid_moves.append("down")
        if not b % 3 == 0:
            valid_moves.append("left")
        if not b % 3 == 2:
            valid_moves.append("right")

        return valid_moves

    def solve_beam(self, k):
        if self.start_puzzle.state == self.goal_puzzle.state:
            print("start state is goal state")
            return

        generated_nodes = 1

        node = self.Node(
                self.Puzzle(self.start_puzzle.state, self.start_puzzle.blank),
                [])
        considered_nodes = [node]

        no_solution = False

        while not no_solution:
            if generated_nodes > self.max_nodes:
                print(f'Exceeded the maximum node count of {self.max_nodes}.')
                return False

            for i in range(len(considered_nodes)):
                child_nodes = []

                for child in self.expand(considered_nodes.pop(0)):
                    if child.value.state == self.goal_puzzle.state:
                        print("------Beam------")
                        print("Number of moves: " + str(child.cost))
                        print("----------------")
                        print("Path taken:", child.path)
                        print("----------------")
                        print("Nodes generated:", generated_nodes)
                        print("----------------")
                        return True
                    child_nodes.append(child)
                    generated_nodes += 1
                considered_nodes += child_nodes
            considered_nodes.sort(key=self.h_2)

            if len(considered_nodes) > k:
                considered_nodes = considered_nodes[:k]

    def set_max_nodes(self, num):
        """set_max_nodes(num) -> None"""
        self.max_nodes = num

    def h_1(self, node):
        """h_1() -> int"""
        count = 0

        for i, val in enumerate(node.value.state):
            if not i == val:
                count += 1

        return count

    def h_2(self, node):
        """h_2() -> int"""
        count = 0

        for i, val in enumerate(node.value.state):
            # How many vertical moves
            count += abs(math.floor(i/3)-math.floor(val/3))
            # How many horizontal moves
            count += abs((i % 3)-(val % 3))

        return count

    class Node:
        def __init__(self, value, path):
            self.value = value
            self.path = path
            self.cost = len(self.path)

        def __str__(self):
            return f'{self.value.state}'

    class Puzzle:
        def __init__(self, state, blank):
            self.state = state
            self.blank = blank


def main():

    random.seed(9098098231)
    valid_commands = [
            "setState",
            "printState",
            "move",
            "randomizeState",
            "solve A-star",
            "solve beam",
            "maxNodes"
            ]

    tester = ProgrammingAssignment1([0, 1, 2, 3, 4, 5, 6, 7, 8])

    with open("./command_file.txt") as f:
        contents = f.readlines()
        for line in contents:
            for command in valid_commands:
                if command in line:
                    match command:
                        case "setState":
                            arg = line[9:]
                            state = []
                            for char in arg:
                                if char == "b":
                                    state.append(0)
                                elif char in [
                                        "1", "2", "3", "4", "5", "6", "7", "8"
                                        ]:
                                    state.append(int(char))
                            tester.set_state(state)

                        case "printState":
                            tester.print_state()

                        case "move":
                            tester.set_state(
                                    tester.move(
                                        tester.start_puzzle,
                                        line[5:]
                                        ).state
                                    )

                        case "randomizeState":
                            tester.set_state(tester.randomize_state(
                                tester.start_puzzle, int(line[15:-1])).state)

                        case "solve A-star":
                            match line[13:-1]:
                                case "h1":
                                    print("Below is h1 heuristic:")
                                    tester.solve_A_star(tester.h_1)
                                case "h2":
                                    print("Below is h2 heuristic:")
                                    tester.solve_A_star(tester.h_2)

                        case "solve beam":
                            print(f'h2 eval beam search k={line[11:-1]}:')
                            tester.solve_beam(int(line[11:-1]))

                        case "maxNodes":
                            tester.set_max_nodes(int(line[9:-1]))


if __name__ == "__main__":
    main()
