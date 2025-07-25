from flask import Flask, render_template, request
import os
import json

app = Flask(__name__)

# Pasta de áudios
AUDIO_FOLDER = os.path.join('static', 'audios')

@app.route('/')
def index():
    audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.wav')]
    return render_template('index.html', audio_files=audio_files)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = {
        "audio_file": request.form['audio_file'],
        "position_selector": {
            "init": float(request.form['start']),
            "end": float(request.form['end'])
        },
        "onset_detection": {
            "method": "super_flux",
            "methods": {
            "super_flux": {
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
        },
        "f0_detection": {
            "method": request.form['f0_method'],
            "methods": {
                "probabilistic_yin": {
                    "voiced_probs_filter": float(request.form['voiced_probs_filter']),
                    "frame_length": float(request.form['frame_length']),
                    "win_length": float(request.form['yin_win_length']),
                    "hop_length": float(request.form['yin_hop_length']),
                    "fmin": float(request.form['yin_fmin']),
                    "n_thresholds": int(request.form['n_thresholds'])
                },
                "crepe_pitch_tracker": {
                    "confidence_filter": float(request.form['confidence_filter']),
                    "step_size": float(request.form['step_size'])
                },
                "bypass": {}
            }
        },

        "position_selector": {
            "init": (float(request.form['init_x']), float(request.form['init_y'])),
            "end": (float(request.form['end_x']), float(request.form['end_y']))
},

    }

    return render_template('transcribe.html', result=json.dumps(data, indent=4))

if __name__ == '__main__':
    app.run(debug=True)
