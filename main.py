import sys

def main():
    if len(sys.argv) != 3:
        print("FORMAT: rlbm <case> <geometry>")

    case_file = sys.argv[1]
    geom_file = sys.argv[2]

if __name__ == "__main__":
    main()
