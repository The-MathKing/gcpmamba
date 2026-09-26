from gears import PertData
import os

print("Initializing GEARS PertData...")
pert_data = PertData('./data')

print("Loading 'replogle_rpe1_essential'...")
pert_data.load(data_name = 'replogle_rpe1_essential')

print("Data successfully loaded!")
print("Checking for downloaded files:")
os.system("ls -lh data/replogle_rpe1_essential/")
