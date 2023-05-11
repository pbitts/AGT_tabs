from heapq import heappush, heappop
import numpy as np

#from fretboard_map import fretboard_map



# Euclidean distance
def heuristic(node1, node2):
    return np.sqrt(((node1.position[0] - node2.position[0]) ** 2) +
                   ((node1.position[1] - node2.position[1]) ** 2))


class Node:

    def __init__(self, name, position, parent=None):
        self.name = name
        self.position = position
        self.parent = parent
        self.neighbors = []
        self.g = 0
        self.h = 0
        self.f = 0

    def add_neighbor(self, v):
        self.neighbors.append(v)

    # we compare the nodes based on the f(x) values
    # f = g + h
    def __lt__(self, other_node):
        return self.f < other_node.f

    def __repr__(self):
        return self.name


class Edge:

    def __init__(self, target, weight):
        self.target = target
        self.weight = weight


class SearchAlgorithm:

    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.explored = set()
        self.heap = [source]

    def run(self):
        # we keep iterating while the heap is not empty
        while self.heap:
            # we always get the node with the lowest f value possible
            current = heappop(self.heap)
            # we add the node to the visited set
            self.explored.add(current)
            # if we reach the destination - this is the end of the algorithm
            if current == self.destination:
                break

            # consider all the neighbors (adjacent nodes)
            for edge in current.neighbors:
                child = edge.target
                temp_g = current.g + edge.weight
                temp_f = temp_g + heuristic(current, self.destination)

                # if we have considered the child and the f(x) is higher
                if child in self.explored and temp_f >= child.f:
                    continue

                # else if we have not visited OR the f(x) score is lower
                if child not in self.heap or temp_f < child.f:
                    child.parent = current
                    child.g = temp_g
                    child.f = temp_f

                    # we should update the heap
                    if child in self.heap:
                        self.heap.remove(child)

                    heappush(self.heap, child)

    def show_solution(self):
        solution = []
        node = self.destination

        while node:
            solution.append(node.position)
            node = node.parent

        return(solution[::-1])

if __name__ == '__main__':
    #notes_original = ['D♯3','E3','D4','C4','G♯3', 'A2', 'E3']
    #notes = ['D♯3','E3','D4','C4','G♯3', 'A2', 'E3']
    notes_original =   ['G4', 'F4', 'A♯4', 'C5', 'F5', 'D5', 'C5', 'F5', 'D5', 'A♯4', 'C5', 'G4', 'F4']
    notes =   ['G4', 'F4', 'A♯4', 'C5', 'F5', 'D5', 'C5', 'F5', 'D5', 'A♯4', 'C5', 'G4', 'F4']
    order = {}
    from iteration_utilities import unique_everseen
    from iteration_utilities import duplicates
    duplicated = list(unique_everseen(duplicates(notes)))
    print(duplicated)
    index = 0
    for i in range(len(notes)):
        if notes[i] in duplicated:
            notes[i] = notes[i] + '_' + str(index)
            index += 1

    print(notes)



    nodes = {'init': [Node('init',(0,0))], 'end': [Node('end',(0,0))]}
    for note in notes:
        nodes[note] = []
    
    for note_index in range(len(notes)):
        positions = fretboard_map[notes_original[note_index]] 
        for position in positions:
            nodes[notes[note_index]].append(Node(notes[note_index], position))
    print(nodes)

    notes.insert(0, "init")
    notes.insert(len(notes), "end")
    print(notes)

    for note_index in range(len(notes)):
        for node in nodes[notes[note_index]]:
            if note_index != len(notes) -1:
                for next_node in nodes[notes[note_index+1]]:
                    node.add_neighbor(Edge(next_node, 1))
                    #print(f'Node {notes[note_index]} added {notes[note_index+1]}')
    




    # n1 = Node("A", (0, 0))
    # n2 = Node("B", (10, 20))
    # n3 = Node("C", (20, 40))
    # n4 = Node("D", (30, 10))
    # n5 = Node("E", (40, 30))
    # n6 = Node("F", (50, 10))
    # n7 = Node("G", (50, 40))

    # n1.add_neighbor(Edge(n2, 10))
    # n1.add_neighbor(Edge(n4, 50))

    # n2.add_neighbor(Edge(n3, 10))
    # n2.add_neighbor(Edge(n4, 20))

    # n3.add_neighbor(Edge(n5, 10))
    # n3.add_neighbor(Edge(n7, 30))

    # n4.add_neighbor(Edge(n6, 80))

    # n5.add_neighbor(Edge(n6, 50))
    # n5.add_neighbor(Edge(n7, 10))

    # n7.add_neighbor(Edge(n6, 10))

    algorithm = SearchAlgorithm(nodes['init'][0], nodes['end'][0])
    algorithm.run()
    solution = algorithm.show_solution()
    print(notes_original)
    print(solution)
