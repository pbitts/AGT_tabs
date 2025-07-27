import logging

import librosa
import crepe
import numpy as np

#from interfaces import Method

class f0Detection:
    def __init__(self,audio_data,sample_rate,parameters):
        logger = logging.getLogger(f0Detection.__qualname__)
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.show = parameters.get('show', False)
        self.delete_min = parameters.get('delete_min', False)
        self.verbose = parameters.get('verbose', False)
        self.plot_path = None
        result_type = parameters.get('result_type', 'max')
        method = parameters.get('method', 'bypass')
        tactic = parameters.get('tactic', 'static' )

    
        logger.info(f'Method "{method}" | Tactic "{tactic}" | Result type: "{result_type}".')

        if method  == 'crepe_pitch_tracker':
            crepe_pitch_tracker_parameters = parameters.get('methods', {}).get('crepe_pitch_tracker', {})
            result = self.crepe_pitch_tracker(tactic, crepe_pitch_tracker_parameters)
            self.crepe_pitch_tracker_result = result
        elif method == 'probabilistic_yin':
            logger.info('Processing Probabilistic YIN...')
            probabilistic_yin_parameters = parameters.get('methods', {}).get('probabilistic_yin', {})
            result = self.probabilistic_yin(tactic,probabilistic_yin_parameters)
            logger.info(f'Probabilistic Yin Result: {result}')
            self.probabilistic_yin_result = result
        elif method == 'crepe_and_yin':
            result = {'f0':[], 'times': []}
            probabilistic_yin_parameters = parameters.get('methods', {}).get('probabilistic_yin', {})
            yin_result = self.probabilistic_yin(tactic,probabilistic_yin_parameters)
            self.probabilistic_yin_result = yin_result
            crepe_pitch_tracker_parameters = parameters.get('methods', {}).get('crepe_pitch_tracker', {})
            crepe_result = self.crepe_pitch_tracker(tactic, crepe_pitch_tracker_parameters)
            self.crepe_pitch_tracker_result = crepe_result
            result['f0'] += yin_result.get('f0',[])
            result['f0'] += crepe_result.get('f0', [])
        else:
            result = self.bypass()
        logger.info("Setting final result...")
        self.final_result = self.get_results(result, result_type)
        
    
    def get_results(self, results, result_type):
        logger = logging.getLogger(f0Detection.get_results.__qualname__)

        logger.info(f'Result strategy: {result_type}')

        f0 = results['f0']
        logger.info(f'F0: {f0}, {type(f0)}, {len(f0)}')


        if type(f0) == np.ndarray:
            if f0.size == 0:
                logger.warning('No notes identified in array f0')
                return f0
        elif type(f0) == list:
            if not f0:
                logger.warning('No notes identified in list f0')
                return f0
            
        if result_type == 'max':
            max_note = librosa.hz_to_note(max(f0))
            logger.info(f'Max note: {max_note}')
            return max_note
        
        elif result_type == 'max_count':
            from collections import Counter
            logger.info('Processing "max_count"...')
            max_count_note = Counter(f0).most_common(1)[0][0]
            logger.info(f'Max count note: {max_count_note}')
            return str(max_count_note)
        
        else:
            logger.error(f'Result type "{result_type}" not found, returning None.')
            return ''


    def crepe_pitch_tracker(self,tactic,parameters):
        logger = logging.getLogger(f0Detection.crepe_pitch_tracker.__qualname__)
        logger.info('####### CREPE PITCH TRACKER')

        if tactic == 'static':
            parameters = parameters.get('static')
            step_size = parameters.get('step_size', 40)
            confidence_filter = parameters.get('confidence_filter',None)
        elif tactic == 'dinamic':
            parameters = parameters.get('dinamic')
            audio_data_length = len(self.audio_data)
            audio_duration = librosa.get_duration(y=self.audio_data,sr=self.sample_rate)
            step_size_parameter = parameters.get('step_size', 1)
            step_size = audio_duration*step_size_parameter*1000
            confidence_filter = parameters.get('confidence_filter',None)
            
        logger.info(f'Step size: {step_size}')
        try:
            times, f0, confidences, activations = crepe.predict(
                                                                self.audio_data, 
                                                                self.sample_rate, 
                                                                step_size = step_size,
                                                                viterbi=True,
                                                                model_capacity='small'
                                                                )
        except Exception as error_msg:
            logger.error(f'Could not proceed with crepe pitch tracker: {str(error_msg)}')
            raise Exception(error_msg)

        if self.verbose:
            logger.info(f'times  -  f0  -  confidence')
            for i in range(len(times)):
                logger.info(f'{times[i]}  -  {librosa.hz_to_note(f0[i])}  -  {confidences[i]}')
        
        if self.show:
            logger.warning(f'Plotting for CREPE not implemented yet.')

        if self.delete_min:
            logger.warning(f'There is no Fmin for crepe')

        if confidence_filter:
            logger.info(f'Confidence filter set to {confidence_filter}')
            #logger.info('Not implemented yet')
            f0_filtered = []
            times_filtered = []
            confidences_filtered = []
            activations_filtered = []
            for i in range(len(f0)):
                if confidences[i] >= confidence_filter:
                    f0_filtered.append(f0[i])
                    times_filtered.append(times[i])
                    confidences_filtered.append(confidences[i])
                    activations_filtered.append(activations[i])
            #logger.info(f'Filtered f0 would be: {f0_filtered}')
            f0 = f0_filtered
            times = times_filtered
            confidences = confidences_filtered
            activations = activations_filtered

        if not type(f0) == list:
            f0 = f0.tolist()
        if not type(times) == list:
            times = times.tolist()

        if f0 != []:
            f0 = librosa.hz_to_note(f0) 
        return {'f0':f0,'times':times, 'confidences': confidences, 'activation':activations}


    def probabilistic_yin(self,tactic,parameters):
        logger = logging.getLogger(f0Detection.probabilistic_yin.__qualname__)
        logger.info('####### PROBABILISTIC YIN')


        if tactic == 'static':
            parameters = parameters.get('static')
            frame_length = parameters.get('frame_length', 2048)
            win_length = parameters.get('win_length', 1024)
            hop_length = parameters.get('hop_length', 512)
            fmin = parameters.get('fmin', 82)
            fmax = parameters.get('fmax', 2093)
            n_thresholds = parameters.get('n_thresholds', 100)
            beta_parameters = parameters.get('beta_parameters', (2,18))
            boltzmann_parameter = parameters.get('boltzmann_parameter', 2)
            resolution = parameters.get('resolution', 0.1)
            max_transition_rate = parameters.get('max_transition_rate', 36.92)
            switch_prob = parameters.get('switch_prob', 0.01)
            no_trough_prob = parameters.get('no_trough_prob', 0.01)
            center = parameters.get('center', True)
            pad_mode = parameters.get('pad_mode', 'constant')
            fill_na = parameters.get('fill_na', None)
            voiced_probs_filter = parameters.get('voiced_probs_filter', None)
        elif tactic == 'dinamic':
            parameters = parameters.get('dinamic')
            audio_data_length = len(self.audio_data)

            frame_length_parameter = parameters.get('frame_length', 1)
            frame_length = int(audio_data_length*frame_length_parameter)
            win_length_parameter = parameters.get('win_length', 1000)
            win_length = int(win_length_parameter*frame_length)
            #win_length = int(audio_data_length/1000)*win_length_parameter
            hop_length_parameter = parameters.get('hop_length', 0.032)
            hop_length = int(audio_data_length*hop_length_parameter)

            fmin = parameters.get('fmin', 0.0)
            fmax = parameters.get('fmax', 2093)
            n_thresholds = parameters.get('n_thresholds', 100)
            beta_parameters = parameters.get('beta_parameters', (2,18))
            boltzmann_parameter = parameters.get('boltzmann_parameter', 2)
            resolution = parameters.get('resolution', 0.1)
            max_transition_rate = parameters.get('max_transition_rate', 36.92)
            switch_prob = parameters.get('switch_prob', 0.01)
            no_trough_prob = parameters.get('no_trough_prob', 0.01)
            center = parameters.get('center', True)
            pad_mode = parameters.get('pad_mode', 'constant')
            fill_na = parameters.get('fill_na', None)
            voiced_probs_filter = parameters.get('voiced_probs_filter', None)

        logger.info(f'Frame Length: {frame_length}'
                    f' | Window Length: {win_length}'
                    f' | Hop length: {hop_length}'
                    f' | Audio Length: {len(self.audio_data)}')
        try:
            f0, voiced_flags, voiced_probs = librosa.pyin(
                                y=self.audio_data, fmin=fmin, fmax=fmax, sr=self.sample_rate, 
                                frame_length=frame_length, win_length=win_length, 
                                hop_length=hop_length, n_thresholds=n_thresholds, 
                                beta_parameters=beta_parameters, boltzmann_parameter=boltzmann_parameter, 
                                resolution=resolution, max_transition_rate=max_transition_rate, 
                                switch_prob=switch_prob,no_trough_prob=no_trough_prob, 
                                center=center, pad_mode=pad_mode, fill_na=fill_na
                                )
            times = librosa.times_like(f0)
        except Exception as error_msg:  
                logger.error(f'Could not proceed with probabilistic yin: {str(error_msg)}')
                raise Exception(error_msg)
        if self.verbose:
            logger.info(f'times  -  f0  -  voiced_flags  -  voiced_probs  ')
            for i in range(len(times)):
                logger.info(f'{times[i]}  -  {librosa.hz_to_note(f0[i])}  -  {voiced_flags[i]}  -  {voiced_probs[i]}')
        
        if self.show:
            from datetime import datetime
            import matplotlib.pyplot as plt
            try:
                D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data, 
                                                                n_fft=frame_length,
                                                                hop_length=hop_length,
                                                                win_length=win_length)), 
                                                                ref=np.max)
                fig, ax = plt.subplots()
                img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
                ax.set(title='pYIN fundamental frequency estimation')
                fig.colorbar(img, ax=ax, format="%+2.f dB")
                ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
                ax.legend(loc='upper right')
                self.plot_path = f'static/plots/f0_detection__probabilistic_yin_result-{datetime.now()}.png'
                plt.savefig(self.plot_path)
            except Exception as error_msg:
                logger.error(f'An error ocurred when plotting: {str(error_msg)}')
                
        if not type(f0) == list:
            f0 = f0.tolist()
        if not type(times) == list:
            times = times.tolist()

        if not self.delete_min:
            logger.warning(f'Fmin deletion not implemented yet.')
            fmin = 0.0
        
        if voiced_probs_filter:
            logger.info(f'Voiced Probs filter set to {voiced_probs_filter}')
            #logger.info('Not implemented yet')
            f0_filtered = []
            times_filtered = []
            voiced_flags_filtered =[]
            voiced_probs_filtered = []
            for i in range(len(f0)):
                if voiced_probs[i] >= voiced_probs_filter and f0[i] != fmin:
                    f0_filtered.append(f0[i])
                    times_filtered.append(times[i])
                    voiced_flags_filtered.append(voiced_flags[i])
                    voiced_probs_filtered.append(voiced_probs[i])
            #logger.info(f'Filtered f0 would be: {f0_filtered}')
            f0 = f0_filtered
            times = times_filtered
            voiced_flags = voiced_flags_filtered
            voiced_probs = voiced_probs_filtered
        logger.info(f' New F0: {f0}')

        if f0 != []:
            try:
                f0 = librosa.hz_to_note(f0)
                logger.info(f'F0 NOTE: {f0}')
            except Exception as e:
                logger.fatal(f'Failed to convert hz to note: {str(e)}')
        return {'f0':f0, 'times':times, 'voiced_flags':voiced_flags, 'voiced_probs':voiced_probs}
    
    def bypass(self):
        return {'f0':[], 'times':[]}