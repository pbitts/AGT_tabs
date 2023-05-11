import logging

class Tablature_Transcription:

    def __init__(self, positions, save_path):
        logger = logging.getLogger(Tablature_Transcription.__qualname__)

        self.positions = positions
        self.strings = {
                '1': '|--',
                '2': '|--',
                '3': '|--',
                '4': '|--',
                '5': '|--',
                '6': '|--',
            }
        self.string_to_note = {'1': 'E','2':'B', '3':'G', '4':'D', '5':'A', '6':'E'}
        self.save_path = save_path

    def transcribe(self, filename= ''):
        logger = logging.getLogger(Tablature_Transcription.transcribe.__qualname__)

        tabs_filename = filename.split('.')[0] + '_tabs.txt'
        add = '---'
        for position in self.positions:
            string_position, fret_position = position
            if len(str(fret_position)) == 2:
                add = '--'
            else:
                add = '---'
            self.strings[str(string_position)] += str(fret_position)+add
            for string in self.strings.keys():
                if string != str(string_position):
                    self.strings[string] += '----'

        for string in self.strings.keys():
            self.strings[string] += '|'

        with open(self.save_path + tabs_filename, "w") as file:
            for key, value in self.strings.items():
                file.write(f"{self.string_to_note[key]}: {value}\n")
        
        logger.info('Tabs saved at ' + self.save_path + tabs_filename)
        
# if __name__ == '__main__':
#     positions = [(1, 3), (1, 1), (1, 6), (1, 8), (1, 13), (1, 10), (1, 8), (1, 13), (1, 10), (1, 6), (1, 8), (1, 3), (1, 1)]
#     tabs = Tablature_Transcription(positions)
#     tabs.transcribe()
