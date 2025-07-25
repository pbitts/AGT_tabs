import librosa
import logging

from f0_detection import f0Detection
from onset_detection import OnsetDetection
from position_selector import Position_Selector
from tablature_transcription import Tablature_Transcription


def run(audio: dict,
        onset_detection_parameters: dict, 
        f0_detection_parameters: dict, 
        position_selector_parameters: dict, 
        tablature_transcription_parameters: dict,
        ):

    logging.basicConfig(
                        format='%(asctime)s\t[%(name)s]\t[%(levelname)s]\t%(message)s',
                        datefmt ="%Y-%m-%d %H:%M:%S%z",
                        level=logging.INFO,  encoding='utf-8',
                        handlers=[
                            logging.FileHandler('logs.log', 'w', 'utf-8'),
                            logging.StreamHandler()
                            ])

    logger = logging.getLogger(__name__)


    acquisition_result = {}
    f0_result = []
    onset_detection_result = {}
    f0_detection_result = {}
    tablature_transcription_result = {}

    # Audio Acquisition Stage
    try:
        path_to_file = 'static/audios/' + audio.get('filename')
        filename = audio.get('filename')
        duration = audio['duration']
        offset = audio['offset']
        sample_rate = audio['sample_rate']

        audio_data,sample_rate = librosa.load(path_to_file, 
                                                sr=sample_rate,
                                                duration=duration,
                                                offset=offset, 
                                                res_type='kaiser_best')
        
        audio_duration = librosa.get_duration(y=audio_data, 
                                                sr=sample_rate)
        acquisition_result = {
                "audio": filename,
                "sample_rate": sample_rate,
                "n_samples": len(audio_data),
                'audio_durantion_secs': audio_duration}

    except Exception as err_msg:
        logger.fatal(f'Acquisition Error: {str(err_msg)}')
        acquisition_result = {
            "error": {
                "err_msg": 'Error on Acquisiton Stage: ' + str(err_msg)
            }
        }
        return acquisition_result, onset_detection_result, f0_detection_result, tablature_transcription_result

    logger.info(f'Audio: {filename}'
                f' | Sample_rate: {sample_rate}'
                f' | Samples: {len(audio_data)}'
                f' | Audio duration: {audio_duration} s')
    
    
    # ONSET DETECTION STAGE
    try:
        onset_detection_step = OnsetDetection(audio_data=audio_data,
                                                sample_rate=sample_rate,
                                                parameters=onset_detection_parameters)
        
        logger.info(f'Onset Final Result: {onset_detection_step.final_result}')
        onset_detection_result = onset_detection_step.final_result.copy()
        onset_detection_result['plot_path'] = onset_detection_step.plot_path
    except Exception as err_msg:
        logger.fatal(f'Onset Detection Error: {str(err_msg)}')
        onset_detection_result = { "error": 'Error on Onset Detection Stage: ' + str(err_msg)}
        return acquisition_result, onset_detection_result, f0_detection_result, tablature_transcription_result

    logger.info('\n')
    # onset_times_result = onset_detection_step.final_result.get('onset_times', [])
    # audio['n_predicted_onsets'] = len(onset_detection_step.final_result.get('onset_samples', []))

    # for onset,_ in onset_times_result:
    #     audio['predicted_onset'].append(onset)

    if onset_detection_step.final_result.get('onset_samples', []):
        # SEGMENT AUDIO USING ONSET POINTS
        for onset_position in range(len(onset_detection_step.final_result.get('onset_samples'))):
            time_starts, time_ends = onset_detection_step.final_result.get('onset_times')[onset_position]
            position_starts, position_ends = onset_detection_step.final_result.get('onset_samples')[onset_position]
            
            logger.info(f'Onset time: {time_starts} to {time_ends} '
                        f'| Onset Sample: {position_starts} to {position_ends}')
            
    # F0 DETECTION STAGE
            try:
                f0_detection_step = f0Detection(audio_data=audio_data[position_starts:position_ends],
                                                sample_rate=sample_rate,
                                                parameters=f0_detection_parameters)
                logger.info('Appending f0....')
                
                if f0_detection_step.final_result:
                    f0_result.append(f0_detection_step.final_result)
                logger.info(f'F0  Result: {f0_detection_step.final_result}')
                logger.info('\n')

            except Exception as err_msg:
                logger.fatal(f'F0 Detection Error: {str(err_msg)}')
                f0_detection_result = { "error": 'Error on F0 Detection Stage: ' + str(err_msg)}
                return acquisition_result, onset_detection_result, f0_detection_result, tablature_transcription_result
        
    f0_detection_result = {"predicted_f0": f0_result}
    # audio['predicted_f0'] = audio['predicted_f0'] + f0_result
    # audio['n_predicted_onsets'] = len(onset_detection_step.final_result.get('onset_samples', []))


    # POSITION SELECTOR STAGE

    try:

        logger.info(f"Audio:{audio['filename']}")
        notes = f0_detection_result['predicted_f0']
        positions = Position_Selector(notes, 
                                        init=position_selector_parameters.get('init'),
                                        end=position_selector_parameters.get('end'))
        final_position = positions.get_solution()
        # TABLATURE TRANSCRIPTION
        tabs = Tablature_Transcription(positions=final_position[1:len(final_position)-1],
                                        save_path=tablature_transcription_parameters.get('save_path'))
        tabs_path = tabs.transcribe(audio['filename'])
        tablature_transcription_result = {"tabs_path": tabs_path}
    except Exception as err_msg:
        tablature_transcription_result = { "error": 'Error on F0 Detection Stage: ' + str(err_msg)}
    
    return acquisition_result, onset_detection_result, f0_detection_result, tablature_transcription_result

