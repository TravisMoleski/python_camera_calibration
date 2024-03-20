import yaml

with open('calibrate_kb.yaml', 'r') as file:
    dat = yaml.safe_load(file)

print(dat['camera_matrix']['data'])