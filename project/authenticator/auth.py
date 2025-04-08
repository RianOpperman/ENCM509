import os
import time
from typing import Optional

def watch(directory: str = "../data_collector", filename: str = "data1.csv", interval: int = 1) -> Optional[str]:
    while True:
        files = os.scandir(directory)
        for entry in files:
            # print(f"{entry.name}: is_file? {entry.is_file()}")
            if entry.is_file() and entry.name == filename:
                data = None
                with open(os.fsdecode(entry.path)) as file:
                    data = file.read()
                os.remove(os.fsdecode(entry.path))
                return data
        time.sleep(interval)
        

def main():
    while True:
        data = watch()
        if data is None:
            continue

        print("Found a new file!")

if __name__ == "__main__":
    main()
