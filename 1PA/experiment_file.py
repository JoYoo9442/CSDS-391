import pa_1
import random


random.seed(9098098231)

tester = pa_1.ProgrammingAssignment1([0, 1, 2, 3, 4, 5, 6, 7, 8])
tester.set_max_nodes(20000)
# tester.set_state([5, 3, 2, 0, 7, 1, 6, 8, 4])
# tester.set_state([7, 2, 4, 5, 0, 6, 8, 3, 1])
# tester.set_state([0, 1, 2, 3, 4, 5, 6, 7, 8])

# Experiment section 1

for i in range(1, 51):
    tester.set_state([0, 1, 2, 3, 4, 5, 6, 7, 8])
    tester.set_max_nodes(i*1000)
    print(f"Max Nodes Value: {tester.max_nodes}")

    h_1_not_maxed = True
    h_2_not_maxed = True
    beam_not_maxed = True

    random_n = 10
    while h_1_not_maxed:
        tester.start_puzzle = tester.randomize_state(
                tester.start_puzzle, random_n)
        random_n += 10
        h_1_not_maxed = tester.solve_A_star(tester.h_1)
    random_n -= 10
    print("h1 Randomized moves max: " + str(random_n))

    random_n = 10
    while h_2_not_maxed:
        tester.start_puzzle = tester.randomize_state(
                tester.start_puzzle, random_n)
        random_n += 10
        h_2_not_maxed = tester.solve_A_star(tester.h_2)
    random_n -= 10
    print("h2 Randomized moves max: " + str(random_n))

    random_n = 10
    while beam_not_maxed:
        tester.start_puzzle = tester.randomize_state(
                tester.start_puzzle, random_n)
        random_n += 10
        beam_not_maxed = tester.solve_beam(25)
    print("beam Randomized moves max: " + str(random_n))

# print("Start state: " + str(tester.start_puzzle.state))
# tester.solve_A_star(tester.h_1)
# tester.solve_A_star(tester.h_2)
# tester.solve_beam(10)
