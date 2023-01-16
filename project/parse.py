import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Name of image - located in mosaic dir")
    args = vars(ap.parse_args())

    print(args)
