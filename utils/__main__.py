from typing import List


class MyString(str):
    def __sub__(self, other: str) -> str:
        return self.split(other)[0]

def rmrear(file_name : str, target : str) -> None:
    with open(file_name, "r") as f:
        lines : List[str] = f.readlines()
    with open(file_name, "w") as f:
        f.writelines([MyString(line) - target + "\n" if target in line else line for line in lines])

# main()
def main() -> None:
    rmrear("requirements.txt", "@")

# Main()
if __name__ == "__main__":
    main()