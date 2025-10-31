from flask import Flask, request, jsonify

from prediction import predict, class_names_from_csv, model

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)
# wav_file_name = 'example_sounds/ambulance_siren_us.wav'

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def main():
    # print("i am here hello")
    wav_file_name = request.form.to_dict()
    wav_file_name = list(wav_file_name.keys())[0]
    # print(wav_file_name)
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
