import os
import shutil
import multiprocessing

MAIN_PATH = r"C:\tithi\Term 2\Data science and decision making"

# Unpack the main Stress_dataset.zip file
shutil.unpack_archive(os.path.join(MAIN_PATH, "Stress_data.zip"), MAIN_PATH)

# Define the path to the unpacked Stress_dataset directory
stress_data_path = os.path.join(MAIN_PATH, "Stress_data")

cpu_count = int(multiprocessing.cpu_count()/2)
print(f'Using {cpu_count} CPUs')

new_list = [
    (file, sub_file) 
    for file in os.listdir(stress_data_path) 
    for sub_file in os.listdir(os.path.join(stress_data_path, file))
]

def unzip_parallel(file, sub_file):
    shutil.unpack_archive(
        os.path.join(stress_data_path, file, sub_file), 
        os.path.join(stress_data_path, file, sub_file[:-4])
    )


if __name__ == '__main__':
    pool = multiprocessing.Pool(cpu_count)
    results = pool.starmap(unzip_parallel, new_list)
    pool.close()

    num_extracted = len(results)
    print(f"{num_extracted} files extracted.")
