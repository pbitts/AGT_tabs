# Automatic Transcription of monophonic guitar audios to tablature

The objective of this work is to use methods
of digital signal processing for feature extraction and
recognition of musical signals from monophonic guitar
audio for automatic tablature transcription. The Super
Flux, Probabilistic Yin, CREPE Pitch Tracker and A-star
methods are employed.
This project was built for a undergraduated final thesis.
The original paper is avaiable at: https://repositorio.ufu.br/handle/123456789/38851 

Keywords – audio, digital, engineering, guitar,
processing, signal, songs, transcription

# Topics

- [How it works?](#how-it-works)
- [Stages](#stages)
- [Results](#results)

# How it works?

Initially, the setup.py file is configured with the parameters of the chosen methods. For example, it starts with the Onset identification method. In the first parameter, the method is chosen (among those implemented, in this case, Super Flux). The tactic used in the example below is relative dynamic variation, named 'dinamic'. The 'show' parameter enables visual graphics, and 'verbose' enables verbose terminal output. Within the 'methods' key, the parameters used are configured. If any parameter is not specified, the algorithm will use the default values of the library that implements it. In the example below, for Super Flux, the parameters are listed using the relative dynamic variation tactic. If no method is selected, the application uses the default bypass method in which, at this stage, nothing is done.

~~~yaml

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

~~~ 

For the fundamental frequency identification, the configuration scheme is similar to the one explained above. In the example below, there are configurations for both the CREPE and PYIN methods. The method to be effectively used is configured in the 'method' parameter. If no method is selected, the application uses the default bypass method, in which nothing is done at this stage.

~~~yaml
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

~~~

Some brief configurations for position optimization and tablature transcription are also included, which include the 'INIT' and 'END' parameters - to adjust the region of the guitar fretboard where the tablature will be generated - and the directory to save the generated tablature. The audio for analysis is included in the 'audio_samples' file, specifying the directory, sampling frequency, offset, and duration if needed. The program runs with the 'main.py' file, which retrieves the configurations from the 'setup.py' file.

~~~yaml

Position_Selector_parameters = {
        'init': (0,0),
    'end': (0,0),
}
Transcription_parameters = {
    'save_path':'C:\\Users\\home\\Desktop\\Projetos\\AGT_tabs\\tabs\\',

}
~~~ 




# Stages

## Acquistion
![Acquisition](images/acquisition.png)

The first step focuses on loading this audio file into quantized amplitude values over time as a sequence of 32-bit floating-point numbers converted by the `load` method of the Librosa library. This method takes, along with the audio file, its sampling frequency and other optional parameters such as offset and duration, in case you want to extract only a portion of the audio for analysis.

## Analysis
![Analysis](images/analysis.png)

After loading the file, the respective data is passed to the Onset stage, which will return start and end time values for sound events identified in the audio. In the scope of this work, these events are the musical notes played by the instrument in a monophonic manner.


![Analysis](images/analysis2.png)

With these values in hand, the audio data is segmented, corresponding to the start and end of these notes. These segments are then passed to the next stage of fundamental frequency identification. In this identification stage, the algorithm utilizes the time-domain data and returns the identified note, indicating its octave in textual form (A3, B4, C5, etc.). For each note segment, a note value is found and added to a list, which is returned in the end, containing all the notes, in sequence, found in the input audio file.

## Classification
![Classification](images/classification.png)

With the final list containing the identified notes in the audio and a mapping of existing notes on the instrument's fretboard, an optimization algorithm, A Star, is used. It seeks to logically find the best positions for playing the identified notes, with the deciding factor being the proximity between them.

## Transcription
![Transcription](images/transcription.png)

The final stage returns, in ASCII text form, the tablature, containing finger positions for playing the identified notes on the instrument.

# Results

With the parameter that deliver the best performance (choices made in the tests from the original paper written by this author avaiable at https://repositorio.ufu.br/handle/123456789/38851), testing was carried out on various audio sets from datasets 2 and 3 of the IDMT-SMT-Guitar database, which contain 6 monophonic excerpts played on three different guitar brands in 2 versions (dataset 2) and 2 monophonic songs (dataset 3) played on the Ibanez RG 2820 guitar. The final results were as follows:

- Accuracy: 0.9662007583060216
- Precision: 0.9909774436090226
- Recall: 0.9662007583060216
- F-Measure: 0.9773000388079276
- Levenshtein Distance: 0.5

For a second test, the algorithm is put into practice in real home scenarios. It makes use of 10 recordings made by the author with their own instrument, including excerpts of songs recorded on a cellphone using a lapel microphone. The following results were obtained:

- Accuracy: 0.878409090909091
- Precision: 0.8529817404817404
- Recall: 0.878409090909091
- F-Measure: 0.865111494451153
- Levenshtein Distance: 1.5

The results demonstrate excellent performance for home recordings and practical scenarios, paving the way for future improvements.