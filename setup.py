

OnsetDetection_parameters = {
                        'method': 'super_flux',
                        'tactic': 'dinamic',
                        'show': True,
                        'verbose':False,
                        'methods':{
                            'spectral_flush':{
                                    'backtrack': True,
                                    'n_mels': 100,
                                    'fmax': 2093,
                                    'aggregate': 'mean',
                                    'detrend': True,
                            },
                            'super_flux':{
                                'static':{
                                    ######## BEST SO FAR: 62.16931216931217,  Max size: 90,Win: 7040, lag: 2, Hop: 256, mel: 425, FFT 7040
                                    'n_fft' :8192,
                                    'hop_length' : 128, 
                                    'win_length': 8192,
                                    'lag' : 2,
                                    'n_mels' : 600,
                                    'fmin' : 82,
                                    'fmax' : 2093.,
                                    'max_size' : 180,
                                    'spectogram':'mel',
                                    },
                                'dinamic':{
                                    'n_fft' :0.01,
                                    'hop_length' : 0.0005, 
                                    'win_length': 0.8,
                                    'lag' : 2,
                                    'n_mels' : 200,
                                    'fmin' : 82,
                                    'fmax' : 2093.,
                                    'max_size' : 100,
                                    'spectogram':'mel',
                                },

                            },
                            'bypass':{

                            },
                        }


}


f0Detection_parameters = {
                        'method':'probabilistic_yin',
                        'tactic': 'dinamic',
                        'result_type': 'max_count',
                        'verbose':True,
                        'show': False,
                        'delete_min' : False,
                        'methods':{
                            'probabilistic_yin':{
                                'static':{
                                'voiced_probs_filter':0.5,
                                'frame_length':27097,
                                'win_length': 27000 ,
                                'hop_length': 256,
                                },
                                'dinamic':{
                                'voiced_probs_filter':0.5,
                                'frame_length':2.0,
                                'win_length': 0.6,
                                'hop_length': 0.02,
                                'fmin': 82.0,
                                'n_thresholds':100,
                                },
                                
                            },
                            'crepe_pitch_tracker':{
                                'static':{
                                    'confidence_filter':0.7,
                                    'step_size': 0.01,
                                },
                                'dinamic':
                                {
                                    'confidence_filter':0.7,
                                    'step_size':0.01,
                                },
                            },
                            'bypass':{

                            },
                        }
}

Position_Selector_parameters = {
        'init': (0,0),
    'end': (0,0),
}
Transcription_parameters = {
    'save_path':'C:\\Users\\home\\Desktop\\Projetos\\AGT_tabs\\tabs\\',

}