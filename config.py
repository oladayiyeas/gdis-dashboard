import os
BASE_DIR = r"C:\Users\Oladayiye A S\PHD_Files\NPC\analysis_reports\gdis_data"

MODEL_DIR = os.path.join(BASE_DIR, "model_files")
SOCIOECON_DIR = os.path.join(BASE_DIR, "input_socioecon_files")
OTHER_LAYERS_DIR = os.path.join(BASE_DIR, "other_layers")

# Directories for each LGA
ETIOSA_MODEL_DIR = os.path.join(MODEL_DIR, "etiosa")
SURULERE_MODEL_DIR = os.path.join(MODEL_DIR, "surulere")

ETIOSA_SOCIOECON_DIR = os.path.join(SOCIOECON_DIR, "etiosa")
SURULERE_SOCIOECON_DIR = os.path.join(SOCIOECON_DIR, "surulere")