import librosa

from f0_detection import f0Detection
from onset_detection import OnsetDetection
from position_selector import Position_Selector
from tablature_transcription import Tablature_Transcription
from audio_sample_test import audio_list

import logging
import numpy as np
import json


def run(f0_detection_parameters, 
        onset_detection_parameters, 
        position_selector_parameters, 
        tablature_transcription_parameters):

    logging.basicConfig(
                        format='%(asctime)s\t[%(name)s]\t[%(levelname)s]\t%(message)s',
                        datefmt ="%Y-%m-%d %H:%M:%S%z",
                        level=logging.INFO,  encoding='utf-8',
                        handlers=[
                            logging.FileHandler('logs.log', 'w', 'utf-8'),
                            logging.StreamHandler()
                            ])

    logger = logging.getLogger(__name__)

    for audio in audio_list:   
        path_to_file = audio['path'] + audio['filename']
        filename = audio['filename']
        duration = audio['duration']
        offset = audio['offset']
        audio['predicted_onset'] = []
        audio['predicted_f0'] = []
        audio['results_onset']  = []
        sample_rate = audio['sample_rate']

        f0_result = []
        
        audio_data,sample_rate = librosa.load(path_to_file, 
                                                sr=sample_rate,
                                                duration=duration,
                                                offset=offset, 
                                                res_type='kaiser_best')
        audio_duration = librosa.get_duration(y=audio_data, 
                                                sr=sample_rate)
        logger.info(f'Audio: {filename}'
                    f' | Sample_rate: {sample_rate}'
                    f' | Samples: {len(audio_data)}'
                    f' | Audio duration: {audio_duration} s')
        # ONSET DETECTION
        onset_detection_step = OnsetDetection(audio_data=audio_data,
                                                sample_rate=sample_rate,
                                                parameters=onset_detection_parameters)
        logger.info(f'Onset Final Result: {onset_detection_step.final_result}')
        logger.info('\n')
        onset_times_result = onset_detection_step.final_result.get('onset_times', [])
        audio['n_predicted_onsets'] = len(onset_detection_step.final_result.get('onset_samples', []))
        for onset,_ in onset_times_result:
            audio['predicted_onset'].append(onset)

        if onset_detection_step.final_result.get('onset_samples', []):
            # SEGMENT AUDIO USING ONSET POINTS
            for onset_position in range(len(onset_detection_step.final_result.get('onset_samples'))):
                time_starts, time_ends = onset_detection_step.final_result.get('onset_times')[onset_position]
                position_starts, position_ends = onset_detection_step.final_result.get('onset_samples')[onset_position]
                
                logger.info(f'Onset time: {time_starts} to {time_ends} '
                            f'| Onset Sample: {position_starts} to {position_ends}')
                # F0 DETECTION 
                f0_detection_step = f0Detection(audio_data=audio_data[position_starts:position_ends],
                                                sample_rate=sample_rate,
                                                parameters=f0_detection_parameters)
                if f0_detection_step.final_result != []:
                    f0_result = f0_result + [f0_detection_step.final_result]
                logger.info(f'F0  Result: {f0_detection_step.final_result}')
                logger.info('\n')
        audio['predicted_f0'] = audio['predicted_f0'] + f0_result
        audio['n_predicted_onsets'] = len(onset_detection_step.final_result.get('onset_samples', []))

    for audio in audio_list: 
        # POSITION SELECTOR
        logger.info(f"Audio:{audio['filename']}")
        notes = audio['predicted_f0'].copy()
        positions = Position_Selector(notes, 
                                        init=position_selector_parameters.get('init'),
                                        end=position_selector_parameters.get('end'))
        final_position = positions.get_solution()
        # TABLATURE TRANSCRIPTION
        tabs = Tablature_Transcription(positions=final_position[1:len(final_position)-1],
                                        save_path=tablature_transcription_parameters.get('save_path'))
        tabs.transcribe(audio['filename'])

if __name__ == '__main__':
    from setup import f0_detection_parameters
    from setup import onset_detection_parameters
    from setup import position_selector_parameters
    from setup import tablature_transcription_parameters

    run(f0_detection_parameters, onset_detection_parameters, position_selector_parameters, tablature_transcription_parameters)
