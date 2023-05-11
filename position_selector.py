from iteration_utilities import unique_everseen
from iteration_utilities import duplicates
import logging
    
from utils import AStarSearch 
from utils.fretboard_map import fretboard_map

class Position_Selector:
    def __init__(self, notes: list, init:tuple = (0,0), end:tuple=(0,0)):
        logger = logging.getLogger(Position_Selector.__qualname__)

        self.notes_original = notes.copy()
        self.notes = self.modify_duplicates(notes)
        self.nodes = self.create_nodes(self.notes, init, end)
        
        logger.info(f'Received notes: {self.notes_original}')

    def modify_duplicates(self, notes) -> list:
        '''
        This method differentiate duplicates in note list
        :param
        notes: a list of string containing notes
        :return
        list of string with symbols differenciating duplicated notes.
        '''
        duplicated = list(unique_everseen(duplicates(notes)))
        index = 0
        for i in range(len(notes)):
            if notes[i] in duplicated:
                notes[i] = notes[i] + '_' + str(index)
                index += 1
        return notes

    def create_nodes(self, notes, init, end):
        nodes = {'init': [AStarSearch.Node('init',init)], 'end': [AStarSearch.Node('end',end)]}
        for note in notes:
            nodes[note] = []
        return nodes

    def add_init_end_to_notes(self):
        self.notes.insert(0, "init")
        self.notes.insert(len(self.notes), "end")

    def get_notes_positions(self):
        for note_index in range(len(self.notes)):
            positions = fretboard_map[self.notes_original[note_index]] 
            for position in positions:
                self.nodes[self.notes[note_index]].append(AStarSearch.Node(self.notes[note_index], position))

    def add_neighbor(self):
        for note_index in range(len(self.notes)):
            for node in self.nodes[self.notes[note_index]]:
                if note_index != len(self.notes) -1:
                    for next_node in self.nodes[self.notes[note_index+1]]:
                        node.add_neighbor(AStarSearch.Edge(next_node, 1))
                        #print(f'Node {self.notes[note_index]} added {self.notes[note_index+1]}')

    def get_solution(self):
        self.get_notes_positions()
        self.add_init_end_to_notes()
        self.add_neighbor()
        algorithm = AStarSearch.SearchAlgorithm(self.nodes['init'][0], self.nodes['end'][0])
        algorithm.run()
        return algorithm.show_solution()

# if __name__ == '__main__':
#     notes = ['G4', 'F4', 'A♯4', 'C5', 'F5', 'D5', 'C5', 'F5', 'D5', 'A♯4', 'C5', 'G4', 'F4']
    
#     positions = Position_Selector(notes)
#     solution = positions.get_solution()
