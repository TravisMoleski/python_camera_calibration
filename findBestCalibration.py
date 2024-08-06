import yaml
import glob
import os
import matplotlib.pyplot as plt

def get_calibration_data_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        # Replace these keys with the actual keys used in your YAML files
        error = data.get('reprojectionError', float('inf'))
        image_count = data.get('images_used', 0)
        print(error, image_count)
    return error, image_count

def main():
    calibration_dir = 'caliRandom'  # Directory containing the YAML files
    min_error = float('inf')
    best_file = None
    errors = []
    image_counts = []

    yaml_files = glob.glob(os.path.join(calibration_dir, '*.yaml'))

    for yaml_file in yaml_files:
        reprojection_error, image_count = get_calibration_data_from_yaml(yaml_file)

        print(f"File: {yaml_file}, Reprojection Error: {reprojection_error}, Image Count: {image_count}")

        if reprojection_error< 1000:
            errors.append(reprojection_error)
            image_counts.append(image_count)


        if reprojection_error < min_error:
            min_error = reprojection_error
            best_file = yaml_file

    if best_file:
        print(f"Best calibration file: {best_file} with Reprojection Error: {min_error}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(image_counts, errors, color='blue', alpha=0.7)
    plt.title('Reprojection Error vs. Number of Calibration Images')
    plt.xlabel('Number of Calibration Images')
    plt.ylabel('Reprojection Error')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
