import librosa


from f0_detection import f0Detection
from onset_detection import OnsetDetection
from audio_sample_licks import audio_list
from segment_silence import Segmentation
from filters import Filter
from position_selector import Position_Selector
from tablature_transcription import Tablature_Transcription

import logging
import numpy as np
import json
from Levenshtein import distance

def main(f0Detection_parameters, OnsetDetection_parameters, Transcription_parameters):
    logging.basicConfig(
                            format='%(asctime)s\t[%(name)s]\t[%(levelname)s]\t%(message)s',datefmt ="%Y-%m-%d %H:%M:%S%z",
                            level=logging.INFO,  encoding='utf-8',
                            handlers=[
                                logging.FileHandler('logs.log', 'w', 'utf-8'),
                                logging.StreamHandler()
                                ]
                            )
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
        
        audio_data,sample_rate = librosa.load(path_to_file, sr=sample_rate,duration=duration,
                                                offset=offset, res_type='kaiser_best')
        audio_duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        logger.info(f'Audio: {filename}'
                    f' | Sample_rate: {sample_rate}'
                    f' | Samples: {len(audio_data)}'
                    f' | Audio duration: {audio_duration} s')
        #ONSET DETECTION
        OnsetDetection_step = OnsetDetection(audio_data=audio_data,
                                                sample_rate=sample_rate,
                                                parameters=OnsetDetection_parameters)
        logger.info(f'Onset Final Result: {OnsetDetection_step.final_result}')
        logger.info('\n')
        onset_times_result = OnsetDetection_step.final_result.get('onset_times', [])
        audio['n_predicted_onsets'] = len(OnsetDetection_step.final_result.get('onset_samples', []))
        for onset,_ in onset_times_result:
            audio['predicted_onset'].append(onset)
    # return check_onsets_results(audio_list)
        if OnsetDetection_step.final_result.get('onset_samples', []):
            for onset_position in range(len(OnsetDetection_step.final_result.get('onset_samples'))):
                time_starts, time_ends = OnsetDetection_step.final_result.get('onset_times')[onset_position]
                position_starts, position_ends = OnsetDetection_step.final_result.get('onset_samples')[onset_position]
                logger.info(f'Onset time: {time_starts} to {time_ends} '
                            f'| Onset Sample: {position_starts} to {position_ends}')
                #f0 detection
                f0Detection_step = f0Detection(audio_data=audio_data[position_starts:position_ends],
                                        sample_rate=sample_rate,
                                        parameters=f0Detection_parameters)
                if f0Detection_step.final_result != []:
                    f0_result = f0_result + [f0Detection_step.final_result]
                logger.info(f'F0  Result: {f0Detection_step.final_result}')
                logger.info('\n')
        audio['predicted_f0'] = audio['predicted_f0'] + f0_result
        audio['n_predicted_onsets'] = len(OnsetDetection_step.final_result.get('onset_samples', []))
        #check_f0_results(audio_list)
    for audio in audio_list: 
        #get positions
        logger.info(f"Audio:{audio['filename']}")
        notes = audio['predicted_f0'].copy()
        positions = Position_Selector(notes, 
                                        init=Transcription_parameters.get('init'),
                                        end=Transcription_parameters.get('end'))
        final_position = positions.get_solution()
        #transcribe into tabs
        tabs = Tablature_Transcription(positions=final_position[1:len(final_position)-1],
                                        save_path=Transcription_parameters.get('save_path'))
        tabs.transcribe(audio['filename'])

    return check_f0_results(audio_list)



def get_predicted_onset(onset_times, audio):
    for onset,_ in onset_times:
        audio['predicted_onset'].append(onset)

def check_onsets_results(audio_list):
    from Levenshtein import distance
    logger = logging.getLogger(__name__)
    final_result = []
    final_result = []
    final_accuracy = []
    final_precision = []
    final_recall = []
    final_f = []
    final_distance = []

    for audio in audio_list:  
        logger.info(f"\n\n####{audio['filename']}")
        onset = audio['onset']
        predicted_onset = audio['predicted_onset']
        onset_ones = convert_to_ones(onset)
        onset_ranges = convert_to_range(onset, 0.01)
        predicted_onset_ones = get_values_in_range(predicted_onset, onset_ranges)
        accuracy, precision, recall, f_measure, levenshtein_distance = get_metrics(onset_ones, predicted_onset_ones)
        logger.info(f'Accuracy: {accuracy}')
        logger.info(f"Precisão: {precision}")
        logger.info(f"Revocação: {recall}")
        logger.info(f"F-measure: {f_measure}")
        logger.info(f'Levenshtein distance: {levenshtein_distance}')
        final_accuracy.append(accuracy)
        final_precision.append(precision)
        final_recall.append(recall)
        final_f.append(f_measure)
        final_distance.append(levenshtein_distance) 
        logger.info(f'onset: {onset}')
        logger.info(f'Predicted onset: {predicted_onset}')
        logger.info(f'Predicted onset ranges: {onset_ranges}')
        logger.info(f'Length onset: {len(predicted_onset)}')
        logger.info(f'True Onset ones: {onset_ones}')
        logger.info(f'Predicted Onset ones: {predicted_onset_ones}')
    logger.info(f'General Accuracy: {sum(final_accuracy)/len(final_accuracy)}')
    logger.info(f'General Precision: {sum(final_precision)/len(final_precision)}')
    logger.info(f'General Recall: {sum(final_recall)/len(final_recall)}')
    logger.info(f'General F-Measure: {sum(final_f)/len(final_f)}')
    logger.info(f'General Distance: {sum(final_distance)/len(final_distance)}')
    return sum(final_accuracy)/len(final_accuracy),sum(final_precision)/len(final_precision),sum(final_recall)/len(final_recall),sum(final_f)/len(final_f),sum(final_distance)/len(final_distance)


def check_f0_results(audio_list):
    logger = logging.getLogger(__name__)
    final_result = []
    final_accuracy = []
    final_precision = []
    final_recall = []
    final_f = []
    final_distance = []
    for audio in audio_list: 
        logger.info(f"\n\n####{audio['filename']}") 
        f0 = audio['f0']
        predicted_f0 = audio['predicted_f0']
        accuracy, precision, recall, f_measure, levenshtein_distance = get_metrics(f0, predicted_f0)
        logger.info(f'Accuracy: {accuracy}')
        logger.info(f"Precisão: {precision}")
        logger.info(f"Revocação: {recall}")
        logger.info(f"F-measure: {f_measure}")
        logger.info(f'Levenshtein distance: {levenshtein_distance}')
        final_accuracy.append(accuracy)
        final_precision.append(precision)
        final_recall.append(recall)
        final_f.append(f_measure)
        final_distance.append(levenshtein_distance)   
        logger.info(f'F0: {f0}')
        logger.info(f'Predicted f0: {predicted_f0}')
        logger.info(f'Length f0: {len(predicted_f0)}')
    logger.info('###### FINAL RESULTS')
    logger.info(f'General Accuracy: {sum(final_accuracy)/len(final_accuracy)}')
    logger.info(f'General Precision: {sum(final_precision)/len(final_precision)}')
    logger.info(f'General Recall: {sum(final_recall)/len(final_recall)}')
    logger.info(f'General F-Measure: {sum(final_f)/len(final_f)}')
    logger.info(f'General Distance: {sum(final_distance)/len(final_distance)}')
    return sum(final_accuracy)/len(final_accuracy),sum(final_precision)/len(final_precision),sum(final_recall)/len(final_recall),sum(final_f)/len(final_f),sum(final_distance)/len(final_distance)

def get_metrics(y_true:list, y_pred:list):
    logger = logging.getLogger(__name__)
    y_true_length = len(y_true)
    y_pred_length = len(y_pred)
    accuraccy = 0.0
    precision = 0.0
    recall = 0.0
    f_measure = 0.0
    try:
        levenshtein_distance = distance(y_true, y_pred)
        accuracy = 1 - (levenshtein_distance/y_true_length)
        if y_pred_length <= 0:
            precision = 0.000
        else:
            precision = (y_true_length-levenshtein_distance)/len(y_pred)
        recall = (y_true_length-levenshtein_distance)/y_true_length
        f_measure = 2*(precision*recall)/float(precision+recall)
        return accuracy, precision, recall, f_measure, levenshtein_distance
    except Exception as error_msg:
        logger.info(f'Failed to calculate metrics: {str(error_msg)}, setting to 0.0.'
        f'y_true length {y_true_length}, y_pred length {y_pred_length},'
        f'accuraccy: {accuraccy}, precision {precision}, recall {recall}, f_measureme {f_measure}')
        return accuracy, precision, recall, f_measure, levenshtein_distance


def get_values_in_range(values, range_values):
    within_range = []
    for value in values:
        is_within_range = False
        for range_value in range_values:
            if value >= range_value[0] and value <= range_value[1]:
                is_within_range = True
                break
        within_range.append(1 if is_within_range else 0)
    return within_range

def convert_to_range(y_values, tolerance):
  true_ranges = []
  for y in y_values:
    true_ranges.append((y*(1-tolerance),y*1.0 + tolerance))
  return true_ranges

def convert_to_ones(y_values):
  length = len(y_values)
  return [1 for i in range(length)]

def get_values_in_range(values, range_values):
    within_range = []
    for value in values:
        is_within_range = False
        for range_value in range_values:
            if value >= range_value[0] and value <= range_value[1]:
                is_within_range = True
                break
        within_range.append(1 if is_within_range else 0)
    return within_range

def automate_test_f0(method):
    logger = logging.getLogger(__name__)
    from setup import f0Detection_parameters, OnsetDetection_parameters, Transcription_parameters
    best = 80.95
    results =[]
    n = 1
    frame_list = [0.2,0.4,0.6,0.8,1.0,2.0] #8
    probs_list = [0.1,0.3,0.5,0.7,0.9] #5
    hop_list = [0.02,0.03,0.04, 0.06] #3
    win_list = [0.6,0.8,1.0] #5
    step_list = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    for probs in probs_list:
        for step in step_list:
            try:
                #print(f'####Testing {n}, Frame {frame}, Win {win}, Hop {hop}, probs {probs}')
                print(f'Testing {n}, step {step}, probs: {probs}')
                f0Detection_parameters['methods'][method]['dinamic']['confidence_filter'] = probs
                f0Detection_parameters['methods'][method]['dinamic']['step_size'] = step
                #f0Detection_parameters['methods'][method]['dinamic']['hop_length'] = hop
                #f0Detection_parameters['methods'][method]['dinamic']['voiced_probs_filter'] = probs
                logger.info(f'Testing {n}, step {step}, probs: {probs}')
                #logger.info(f'Testing {n}, frame: {frame}, win: {win}, hop: {hop}, probs: {probs}')
                accuracy, precision, recall, f_measure, levenshtein_distance = main(f0Detection_parameters, 
                                OnsetDetection_parameters, Transcription_parameters)
                results.append({
                                'n': n,
                                'accuracy':accuracy,
                                'precision':precision,
                                'recall':recall,
                                'f_measure':f_measure,
                                'levenshtein_distance':levenshtein_distance,
                                'voiced_probs_filter':probs,
                                'step_size': step,
                                'success': True
                            })
                with open("crepe_tests.txt", "w") as f:
                    f.write(str(results))
                n += 1
            except Exception as e:
                results.append({
                        'n': n,
                        'accuracy':0.0,
                        'precision':0.0,
                        'recall':0.0,
                        'f_measure':0.0,
                        'levenshtein_distance':10000,
                        'voiced_probs_filter':probs,
                        'step_size': step,
                        'success': False
                    })
                with open("crepe_tests.txt", "w") as f:
                    f.write(str(results))
                n +=1
                logger.error(f'FAILED, IGNORING: {str(e)}')
def automate_test_pyin(method):
    logger = logging.getLogger(__name__)
    from setup import f0Detection_parameters, OnsetDetection_parameters, Transcription_parameters
    best = 80.95
    results =[]
    n = 1
    frame_list = [0.2,0.4,0.6,0.8,1.0,2.0] #6
    probs_list = [0.1,0.3,0.5,0.7,0.9] #5
    hop_list = [0.02,0.04, 0.06,0.08] #4
    win_list = [0.6,0.8,1.0] #3
    for probs in probs_list:
        for frame in frame_list:
            for hop in hop_list:
                for win in win_list:
                    try:
                        print(f'####Testing {n}, Frame {frame}, Win {win}, Hop {hop}, probs {probs}')
                        #print(f'Testing {n}, step {step}, probs: {probs}')
                        #f0Detection_parameters['methods'][method]['dinamic']['confidence_filter'] = probs
                        #f0Detection_parameters['methods'][method]['dinamic']['step_size'] = step
                        f0Detection_parameters['methods'][method]['dinamic']['hop_length'] = hop
                        f0Detection_parameters['methods'][method]['dinamic']['voiced_probs_filter'] = probs
                        f0Detection_parameters['methods'][method]['dinamic']['frame_length'] = frame
                        f0Detection_parameters['methods'][method]['dinamic']['win_length'] = win
                        #logger.info(f'Testing {n}, step {step}, probs: {probs}')
                        logger.info(f'Testing {n}, frame: {frame}, win: {win}, hop: {hop}, probs: {probs}')
                        accuracy, precision, recall, f_measure, levenshtein_distance = main(f0Detection_parameters, 
                                        OnsetDetection_parameters,Transcription_parameters)
                        results.append({
                                        'n': n,
                                        'accuracy':accuracy,
                                        'precision':precision,
                                        'recall':recall,
                                        'f_measure':f_measure,
                                        'levenshtein_distance':levenshtein_distance,
                                        'voiced_probs_filter':probs,
                                        'hop_length': hop,
                                        'frame_length':frame,
                                        'win_length':win,
                                        'success': True
                                    })
                        with open("pyin_tests.txt", "w") as f:
                            f.write(str(results))
                        n += 1
                    except Exception as e:
                        results.append({
                                'n': n,
                                'accuracy':0.0,
                                'precision':0.0,
                                'recall':0.0,
                                'f_measure':0.0,
                                'levenshtein_distance':10000,
                                'voiced_probs_filter':probs,
                                'hop_length': hop,
                                'frame_length':frame,
                                'win_length':win,
                                'success': False
                            })
                        with open("pyin_tests.txt", "w") as f:
                            f.write(str(results))
                        n +=1
                        logger.error(f'FAILED, IGNORING: {str(e)}')
def automate_test_onset(method='super_flux'):
    logger = logging.getLogger(__name__)
    from setup import f0Detection_parameters, OnsetDetection_parameters, Transcription_parameters
    ######## BEST SO FAR: 62.16931216931217,  Max size: 90,Win: 7040, lag: 2, Hop: 256, mel: 425, FFT 7040

    ## Last {'n': 467, 'accuracy': 0.23280423280423282, 'precision': 0.24679542202675375, 
    #'recall': 0.23280423280423276, 'f_measure': 0.24403986085155058, 'levenshtein_distance': 16.11111111111111, 
    #'max_size': 10, 'lag': 1, 'hop_length': 256, 'n_mels': 511, 'n_fft': 11264, 'win_length': 10240, 
    #'fmax': 2093.0, 'fmin': 82, 'success': True}]
    results = []
    n=1
    best =  62.16931216931217
    with open('onset_tests_dinamic.txt', 'w') as f:
        f.write(str(best))
    '''    max_size_list = [200] #1
    lag_list = [2] #1
    hop_list = [0.000001,0.00001, 0.0001, 0.001] #4
    mel_list = [200,600,700] #3
    fft_list = [ 0.001, 0.01,0.05, 0.1] #4
    win_list = [0.8,0.9,1.0] #3'''
    max_size_list = [100] #3
    lag_list = [2] #1
    hop_list = [0.0005] #3
    mel_list = [200] #4
    fft_list = [ 0.01] #1
    win_list = [0.8] #1
    #win_list = [1024, 4096, 7040, 8192] #4
    fmax = 2093.
    fmin = 82
    for lag in lag_list:
        for max_size in max_size_list:
            for mel in mel_list:
                for hop in hop_list:
                    for win in win_list:
                        for fft in fft_list:
    
                            print(f'######## Testing {n},Max size: {max_size},Win: {win}, lag: {lag}, Hop: {hop}, mel: {mel}, FFT {fft}')

                            try:
                                OnsetDetection_parameters['methods'][method]['dinamic']['n_fft'] = fft
                                OnsetDetection_parameters['methods'][method]['dinamic']['win_length'] = win
                                OnsetDetection_parameters['methods'][method]['dinamic']['n_mels'] = mel
                                OnsetDetection_parameters['methods'][method]['dinamic']['lag'] = lag
                                OnsetDetection_parameters['methods'][method]['dinamic']['max_size'] = max_size
                                OnsetDetection_parameters['methods'][method]['dinamic']['hop_length'] = hop

                                accuracy, precision, recall, f_measure, levenshtein_distance = main(f0Detection_parameters, 
                                                                            OnsetDetection_parameters, Transcription_parameters
                                                                            )
                                print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F Measure: {f_measure}, levenshtein_distance: {levenshtein_distance}')
                                results.append({
                                                'n': n,
                                                'accuracy':accuracy,
                                                'precision':precision,
                                                'recall':recall,
                                                'f_measure':f_measure,
                                                'levenshtein_distance':levenshtein_distance,
                                                'max_size':max_size,
                                                'lag':lag,
                                                'hop_length':hop,
                                                'n_mels': mel,
                                                'n_fft':fft,
                                                'win_length':win,
                                                'fmax':fmax,
                                                'fmin':fmin,
                                                'success': True
                                            })

                                #logger.info(f'######## BEST SO FAR: {best},  Max size: {max_size},Win: {win}, lag: {lag}, Hop: {hop}, mel: {mel}, FFT {fft}')
                                #print(f'######## BEST SO FAR: {best},  Max size: {max_size},Win: {win}, lag: {lag}, Hop: {hop}, mel: {mel}, FFT {fft}')
                                with open("onset_tests_dinamic.txt", "w") as f:
                                    f.write(str(results))
                                n += 1
                            except Exception as e:
                                results.append({
                                        'n': n,
                                        'accuracy':0.0,
                                        'precision':0.0,
                                        'recall':0.0,
                                        'f_measure':0.0,
                                        'levenshtein_distance':10000,
                                        'max_size':max_size,
                                        'lag':lag,
                                        'hop_length':hop,
                                        'n_mels': mel,
                                        'n_fft':fft,
                                        'win_length':win,
                                        'fmax':fmax,
                                        'fmin':fmin,
                                        'success': False,
                                    })
                                with open("onset_tests_dinamic.txt", "w") as f:
                                    f.write(str(results))
                                n += 1
                                logger.error(f'FAILED, IGNORING: {str(e)}')


if __name__ == '__main__':
    from setup import f0Detection_parameters, OnsetDetection_parameters, Transcription_parameters
    accuracy, precision, recall, f_measure, levenshtein_distance = main(f0Detection_parameters, 
                                OnsetDetection_parameters, Transcription_parameters
                                )
    # automate_test_onset('super_flux')
    # automate_test_f0(method='crepe_pitch_tracker')
    # automate_test_pyin(method='probabilistic_yin')