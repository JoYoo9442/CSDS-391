import pa_1
import random


random.seed(9098098231)

tester = pa_1.ProgrammingAssignment1([0, 1, 2, 3, 4, 5, 6, 7, 8])
tester.set_max_nodes(20000)
# tester.set_state([5, 3, 2, 0, 7, 1, 6, 8, 4])
# tester.set_state([7, 2, 4, 5, 0, 6, 8, 3, 1])
# tester.set_state([0, 1, 2, 3, 4, 5, 6, 7, 8])

# Code Corectness section
tester.set_state(tester.randomize_state(tester.start_puzzle, 18).state)
tester.print_state()
print("Below is h1 heuristic:")
tester.solve_A_star(tester.h_1)

print("Below is h2 heuristic:")
tester.solve_A_star(tester.h_2)

print("Below is h2 evaluation beam search k=10:")
tester.solve_beam(10)

print("Below is h2 evaluation beam search k=20:")
tester.solve_beam(20)

print("Below is h2 evaluation beam search k=30:")
tester.solve_beam(30)

tester.set_state(tester.randomize_state(tester.start_puzzle, 57).state)
tester.print_state()
print("Below is h1 heuristic:")
tester.solve_A_star(tester.h_1)

print("Below is h2 heuristic:")
tester.solve_A_star(tester.h_2)


# tester.start_puzzle = tester.randomizeState(tester.start_puzzle, 2)
# print("Start state: " + str(tester.start_puzzle.state))
# tester.solve_A_star(tester.h_1)
# tester.solve_A_star(tester.h_2)
# tester.solve_beam(10)
