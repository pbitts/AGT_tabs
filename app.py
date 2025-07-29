from flask import Flask, render_template, request
import os
import json

from process import run

app = Flask(__name__)

# Pasta de áudios
AUDIO_FOLDER = os.path.join('static', 'audios')

@app.route('/')
def index():
    audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.wav') or f.endswith('.ogg')]
    return render_template('index.html', audio_files=audio_files)

@app.route('/transcribe', methods=['POST'])
def transcribe():


    audio_params = {
         "filename": request.form['audio_file'],
        "offset": float(request.form['audio_start']),
        "duration": float(request.form['audio_end']) - float(request.form['audio_start']),
        "sample_rate": float(request.form['sample_rate']),  
    }

    onset_detection_params = {
            "method": "super_flux",
            "tactic": 'dinamic',
            'show': True,
            'verbose':False,
            "methods": {
            "super_flux": {
                 "dinamic":{
                    "n_fft": float(request.form['n_fft']),
                    "hop_length": float(request.form['hop_length']),
                    "win_length": float(request.form['win_length']),
                    "lag": int(request.form['lag']),
                    "n_mels": int(request.form['n_mels']),
                    "fmin": int(request.form['fmin']),
                    "fmax": float(request.form['fmax']),
                    "max_size": int(request.form['max_size']),
                    "spectogram": request.form['spectogram']
                        }
                    }
                }
            }
        
        
    f0_detection_params = {
            "method": request.form['f0_method'],
            'tactic': 'dinamic',
            'result_type': 'max_count',
            'verbose':False,
            'show': False,
            'delete_min' : False,
            "methods": {
                "probabilistic_yin": {
                     "dinamic": {
                    "voiced_probs_filter": float(request.form['voiced_probs_filter']),
                    "frame_length": float(request.form['frame_length']),
                    "win_length": float(request.form['yin_win_length']),
                    "hop_length": float(request.form['yin_hop_length']),
                    "fmin": float(request.form['yin_fmin']),
                    "n_thresholds": int(request.form['n_thresholds'])
                    }
                },
                "crepe_pitch_tracker": {
                     "dinamic": {
                    "confidence_filter": float(request.form['confidence_filter']),
                    "step_size": float(request.form['step_size'])
                    }
                },
                
            }
        }

    position_selector_params= {
            "init": (float(request.form['init_x']), float(request.form['init_y'])),
            "end": (float(request.form['end_x']), float(request.form['end_y']))
            }

    acquisition_result, \
    onset_detection_result, \
    f0_detection_result, \
    tablature_transcription_result = run(audio=audio_params,
                                        onset_detection_parameters=onset_detection_params,
                                        f0_detection_parameters=f0_detection_params,
                                        position_selector_parameters=position_selector_params,
                                        tablature_transcription_parameters={'save_path':"static/tabs/"})
   
    if 'tabs_path' in tablature_transcription_result:
        with open(tablature_transcription_result['tabs_path'], 'r') as f:
            tablature_transcription_result['content'] = f.read()

    return render_template(
                    "transcribe.html",
                    acquisition_result=acquisition_result,
                    onset_detection_result=onset_detection_result,
                    f0_detection_result=f0_detection_result,
                    tablature_transcription_result=tablature_transcription_result
    )

if __name__ == '__main__':
    app.run(debug=True)
