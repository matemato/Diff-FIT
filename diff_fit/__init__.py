from pathlib import Path

DIFF_FIT_BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
DATA_DIR = Path(__file__).parent.parent / "data"
FACE_LANDMARKS_WEIGHTS = (
    WEIGHTS_DIR / "face_landmarks" / "shape_predictor_68_face_landmarks.dat"
)
RESNET_WEIGHTS = WEIGHTS_DIR / "resnet" / "resnet18.pt"

LIGHTNING_DRAG_WEIGHTS = WEIGHTS_DIR / "lightning_drag"
