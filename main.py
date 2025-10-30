from flask import Flask, request, jsonify

from prediction import predict, class_names_from_csv, model

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)
# wav_file_name = 'example_sounds/ambulance_siren_us.wav'

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def main():
    wav_file_name = request.files["audio"]
    predicted_class = predict(file_name=wav_file_name, class_names=class_names)
    # print(predicted_class)
    if predicted_class.lower() == "alarm":
        is_siren = {"is_siren": True}
    elif "siren" in predicted_class.lower():
        is_siren = {"is_siren": True}
    else:
        is_siren = {"is_siren": False}
    res = jsonify(is_siren)
    # print(res)
    return res

if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask, request, jsonify
# import tempfile
# import os

# # Assuming prediction.py contains predict, class_names_from_csv, and model
# from prediction import predict, class_names_from_csv, model

# # Load model/class names once globally
# class_map_path = model.class_map_path().numpy()
# class_names = class_names_from_csv(class_map_path)

# app = Flask(__name__)
# @app.route("/health", methods=["GET"])
# def checking_server():
#     return "Server chal raha hai ", 200


# @app.route("/predict", methods=["POST"])
# def main():
#     if not request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     uploaded_file = next(iter(request.files.values()))
#     temp_dir = tempfile.gettempdir()
#     temp_wav_path = os.path.join(temp_dir, "temp_audio_check.wav")
#     uploaded_file.save(temp_wav_path)
#     predicted_class = predict(file_name=temp_wav_path, class_names=class_names)

#     os.remove(temp_wav_path)

#     if predicted_class.lower() == "alarm" or "siren" in predicted_class.lower():
#         is_siren = {"is_siren": True}
#     else:
#         is_siren = {"is_siren": False}

#     return jsonify(is_siren)

# if __name__ == "__main__":
#     app.run(debug=True)
