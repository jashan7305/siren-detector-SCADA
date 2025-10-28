from prediction import predict, class_names_from_csv, model

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)
wav_file_name = 'example_sounds/ambulance_siren_us.wav'

def main():
    predicted_class = predict(file_name=wav_file_name, class_names=class_names)
    print(predicted_class)

if __name__ == "__main__":
    main()