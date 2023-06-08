import logging

import librosa
import numpy as np

#from interfaces import Method

class OnsetDetection:
    #[(i,f),(i,f)]
    def __init__(self, audio_data, sample_rate, parameters):
        logger = logging.getLogger(OnsetDetection.__qualname__)
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.show = parameters.get('show', False)
        self.delete_min = parameters.get('delete_min', False)
        result_type = parameters.get('result_type', 'max')
        method = parameters.get('method', 'bypass')
        tactic = parameters.get('tactic', 'static' )
        logger.info(f'Method "{method}" | Tactic "{tactic}" ')
        if method == 'spectral_flush':
            spectral_flush_parameters = parameters.get('methods', {}).get('spectral_flush', {})
            result = self.spectral_flush(tactic,spectral_flush_parameters)
            self.spectral_flush_result = result
        elif method == 'super_flux':
            super_flux_parameters = parameters.get('methods', {}).get('super_flux', {})
            result = self.super_flux(tactic,super_flux_parameters)
            self.super_flux_result = result
        elif method == 'bypass':
            result = self.bypass()
        self.final_result = result

    def spectral_flush(self, tatic, parameters):
        logger = logging.getLogger(OnsetDetection.spectral_flush.__qualname__)
        
        wait = parameters.get('wait', 1)
        pre_avg = parameters.get('pre_avg', 1)
        post_avg = parameters.get('post_avg', 1)
        pre_max = parameters.get('pre_max', 1)
        post_max = parameters.get('post_max', 1)
        backtrack = parameters.get('backtrack', True)
        n_mels = parameters.get('n_mels', 480)
        fmax = parameters.get('fmax', 2093)
        aggregate = parameters.get('aggregate', 'median')
        detrend = parameters.get('detrend', False)

        if aggregate == 'median':
            aggregate = np.median
        elif aggregate == 'mean':
            aggregate = np.mean
        elif aggregate == 'max':
            aggregate = np.max
        elif aggregate == 'min':
            aggregate = np.min
        else:
            logger.warning(f'Invalid value for {aggregate}, setting it to None')
            aggregate = None

        # CUSTOM MEL
        onset_samples = []
        onset_frames = []
        onset_times = []
        onset_env = librosa.onset.onset_strength(y=self.audio_data, sr=self.sample_rate,
                                                aggregate=aggregate, detrend=detrend,
                                                fmax=fmax, n_mels=n_mels)

        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sample_rate, 
                                            wait=wait, pre_avg=pre_avg, 
                                            post_avg=post_avg, pre_max=pre_max, 
                                            post_max=post_max, backtrack=backtrack,
                                            normalize=True, units='frames').tolist()
        onset_times = librosa.frames_to_time(onset_frames).tolist()
        onset_samples = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sample_rate, 
                                            wait=wait, pre_avg=pre_avg, 
                                            post_avg=post_avg, pre_max=pre_max, 
                                            post_max=post_max, backtrack=backtrack,
                                            normalize=True, units='samples').tolist()
        onset_amplitudes = [self.audio_data[sample] for sample in onset_samples ]
        audio_duration = librosa.get_duration(y=self.audio_data, sr=self.sample_rate)
        onset_samples = self.group_onsets(np.append(onset_samples, [len(self.audio_data)]))
        onset_times = self.group_onsets(np.append(onset_times, audio_duration))
        
        
        return {'onset_samples':onset_samples,
                'onset_times':onset_times,
                'onset_amplitudes':onset_amplitudes,
                'onset_frames':onset_frames}

    def super_flux(self,tactic,parameters):
        logger = logging.getLogger(OnsetDetection.super_flux.__qualname__)

        if tactic == 'static':
            #Default parameters are taken directly from Super flux paper
            parameters = parameters.get('static')
            n_fft = parameters.get('n_fft',1024)
            hop_length = parameters.get('hop_lenght', 
                                        int(librosa.time_to_samples(1./200, sr=self.sample_rate)))
            #hop_length = int(librosa.time_to_samples(1./200, sr=sr))
            lag = parameters.get('lag', 2)
            n_mels = parameters.get('n_mels', 138)
            fmin = parameters.get('fmin', 27.5)
            fmax = parameters.get('fmax',16000.)
            max_size = parameters.get('max_size',3)
            spectogram = parameters.get('spectogram', 'mel')
            win_length= parameters.get('win_length', None)
            window= parameters.get('window', 'hann') 
            center= parameters.get('center', True) 
            pad_mode= parameters.get('pad_mode','constant') 
            power= parameters.get('power',2.0)
        elif tactic == 'dinamic':
            parameters = parameters.get('dinamic')
            audio_data_length = len(self.audio_data)
            n_fft = parameters.get('n_fft',0.2)
            n_fft = int(n_fft*audio_data_length)
            hop_length = parameters.get('hop_length', 0.032)
            hop_length = int(hop_length*audio_data_length)
            #hop_length = int(librosa.time_to_samples(1./200, sr=sr))
            lag = parameters.get('lag', 2)
            n_mels = parameters.get('n_mels', 138)
            fmin = parameters.get('fmin', 27.5)
            fmax = parameters.get('fmax',16000.)
            max_size = parameters.get('max_size',3)
            spectogram = parameters.get('spectogram', 'mel')
            win_length= parameters.get('win_length', 0.6)
            win_length = int(win_length*n_fft)
            window= parameters.get('window', 'hann') 
            center= parameters.get('center', True) 
            pad_mode= parameters.get('pad_mode','constant') 
            power= parameters.get('power',2.0)

        
        representation = self.get_spectogram(spectogram, parameters, tactic)
        onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(representation, ref=np.mean),
                                                sr=self.sample_rate,
                                                hop_length=hop_length,
                                                lag=lag, max_size=max_size)
        onset_samples = librosa.onset.onset_detect(onset_envelope=onset_env,
                                            sr=self.sample_rate,
                                            hop_length=hop_length,
                                            units='samples')
        onset_times = librosa.onset.onset_detect(onset_envelope=onset_env,
                                            sr=self.sample_rate,
                                            hop_length=hop_length,
                                            units='time')
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env,
                                            sr=self.sample_rate,
                                            hop_length=hop_length,
                                            units='frames')
        onset_amplitudes = [self.audio_data[sample] for sample in onset_samples ]
        audio_duration = librosa.get_duration(y=self.audio_data, sr=self.sample_rate)
        onset_samples = self.group_onsets(np.append(onset_samples, [len(self.audio_data)]))
        onset_times = self.group_onsets(np.append(onset_times, audio_duration))

        if self.show:
            import matplotlib.pyplot as plt
            try:
                audio_duration = librosa.get_duration(y=self.audio_data, sr=self.sample_rate)
                audio_time = np.arange(0.0, audio_duration, (1/(self.sample_rate)))
                plt.plot(audio_time, self.audio_data, color='darkblue' )
                plt.vlines(onset_times[:len(onset_times)-1],ymin=self.audio_data.min(), ymax=self.audio_data.max(), color='tab:orange', alpha=0.9,
                        linestyle='--', label='Onsets')
                plt.title(label='Super Flux Onset detection')
                plt.show()
                cmap = plt.colormaps['plasma']
                fig, ax = plt.subplots(nrows=2, sharex=True)
                img = librosa.display.specshow(representation,
                          y_axis='mel', x_axis='time', ax=ax[0],cmap=cmap)
                ax[0].set(title='Power spectrogram')
                ax[0].label_outer()
                D = librosa.feature.melspectrogram(y=self.audio_data,sr=self.sample_rate)
                times = librosa.times_like(librosa.power_to_db(representation))
                ax[1].plot(times,onset_env, alpha=0.8,
                    label='Mean (mel)')
                # fig.colorbar(img, ax=ax[0], format="%+2.f dB")
                plt.show()
            except Exception as error_msg:
                logger.error(f'An error ocurred when plotiing: {str(error_msg)}')
        
        
        return {'onset_samples':onset_samples,
                'onset_times':onset_times,
                'onset_amplitudes':onset_amplitudes,
                'onset_frames':onset_frames}

    def get_spectogram(self, spectogram, parameters, tactic):
        if spectogram == 'mel':
            if tactic == 'static':
                n_fft = parameters.get('n_fft',1024)
                hop_length = parameters.get('hop_lenght', 
                                            int(librosa.time_to_samples(1./200, sr=self.sample_rate)))
                lag = parameters.get('lag', 2)
                n_mels = parameters.get('n_mels', 138)
                fmin = parameters.get('fmin', 27.5)
                fmax = parameters.get('fmax',16000.)
                max_size = parameters.get('max_size',3)
                win_length= parameters.get('win_length', None)
                window= parameters.get('window', 'hann') 
                center= parameters.get('center', True) 
                pad_mode= parameters.get('pad_mode','constant') 
                power= parameters.get('power',2.0)
            if tactic == 'dinamic':
                audio_data_length = len(self.audio_data)
                n_fft = parameters.get('n_fft',0.2)
                n_fft = int(n_fft*audio_data_length)
                hop_length = parameters.get('hop_length', 0.032)
                hop_length = int(hop_length*audio_data_length)
                #hop_length = int(librosa.time_to_samples(1./200, sr=sr))
                lag = parameters.get('lag', 2)
                n_mels = parameters.get('n_mels', 138)
                fmin = parameters.get('fmin', 27.5)
                fmax = parameters.get('fmax',16000.)
                max_size = parameters.get('max_size',3)
                spectogram = parameters.get('spectogram', 'mel')
                win_length= parameters.get('win_length', 0.6)
                win_length = int(win_length*n_fft)
                window= parameters.get('window', 'hann') 
                center= parameters.get('center', True) 
                pad_mode= parameters.get('pad_mode','constant') 
                power= parameters.get('power',2.0)

            mel_spec = librosa.feature.melspectrogram(y=self.audio_data,sr=self.sample_rate, 
                                                    n_fft=n_fft,hop_length=hop_length,
                                                    fmin=fmin,fmax=fmax,
                                                    n_mels=n_mels,win_length=win_length,
                                                    window=window,center=center,
                                                    pad_mode=pad_mode,power=power)
            return librosa.power_to_db(mel_spec)
        elif spectogram == 'cqt':
            #raise NotImplementedError()
            hop_length = parameters.get('hop_length',512)
            fmin = parameters.get('fmin',None)
            n_bins = parameters.get('n_bins',84)
            bins_per_octave = parameters.get('bins_per_octave',12)
            tuning = parameters.get('tuning', 0.0)
            filter_scale = parameters.get('filter_scale',1)
            norm = parameters.get('norm',1)
            sparsity = parameters.get('sparsity', 0.01)
            window= parameters.get('window', 'hann') 
            scale = parameters.get('scale',True)
            pad_mode= parameters.get('pad_mode','constant')
            res_type = parameters.get('res_type', 'kaiser_best')
            dtype = parameters.get('dtype',None)
            return librosa.cqt(y=self.audio_data, sr=self.sample_rate)
            '''n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            tuning=tuning,filter_scale=filter_scale,
            norm=norm,sparsity=sparsity,
            window=window,scale=scale,
            pad_mode=pad_mode,res_type=res_type,
            dtype=dtype'''
        elif spectogram == 'fft':
            return 0 

    def group_onsets(self,onsets):
        if len(onsets) >= 2:
            return [    (onsets[i], onsets[i+1]) for i in range(len(onsets)-1)]
        else:
            return onsets


    def bypass(self):
        return []