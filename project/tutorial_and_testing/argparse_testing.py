import argparse
import os

MOSAIC_DIR_TESTING = "/Users/nicolasbancel/git/perso/mamie/data/mosaic/"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="Name of image - located in mosaic dir")
    ap.add_argument("-n", "--num_mosaics", required=False, type=int, help="Number of mosaics to process")
    args = vars(ap.parse_args())

    print(args)

    if args["image"] is not None:
        print(f"Will process image : {args['image']}")
    else:
        if args["num_mosaics"] is None:
            mosaics_to_process = sorted(os.listdir(MOSAIC_DIR_TESTING))
            print(f"Will process all images")
        else:
            print(f"Will process images from index 0 until {args['num_mosaics'] - 1}")
            mosaics_to_process = sorted(os.listdir(MOSAIC_DIR_TESTING))[: args["num_mosaics"]]
