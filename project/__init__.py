import sys, os
from utils import *
import transfo

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

print("__init__ ran")

ROTATION_METADATA = load_metadata(filename="rotation_metadata.csv")
