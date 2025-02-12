from typing import Generator, Iterable, List, TypeVar
import numpy as np
import supervision as sv
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from torchvision.models import efficientnet_b0
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
import cv2

V = TypeVar("V")

class TeamClassifier:
    def __init__(self, device: str = 'cuda', batch_size: int = 256):
        self.device = device
        self.batch_size = batch_size

        self.model = efficientnet_b0(pretrained=True)
        self.model.classifier = torch.nn.Identity()
        self.model = self.model.eval().to(device)

        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.reducer = PCA(n_components=64)
        self.cluster_model = MiniBatchKMeans(
            n_clusters=2,
            batch_size=2048,
            init='k-means++',
            n_init=3,
            max_iter=100
        )

    @staticmethod
    def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
        return (sequence[i:i + batch_size] for i in range(0, len(sequence), batch_size))

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extraction combinée de features profondes et de couleur"""
        deep_features = []
        color_features = []
        
        with torch.no_grad():
            for batch in self.create_batches(crops, self.batch_size):
                tensor_batch = torch.stack(
                    [self.transform(sv.cv2_to_pillow(img)) for img in batch]
                ).to(self.device, non_blocking=True)
                deep_features.append(
                    self.model(tensor_batch).cpu().numpy()
                )

                for img in batch:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
                    color_features.append(cv2.normalize(hist, None).flatten())
        
        return np.hstack([
            np.concatenate(deep_features),
            np.array(color_features)
        ])

    def fit(self, crops: List[np.ndarray]) -> None:
        """Entraînement avec normalisation L2"""
        data = self.extract_features(crops)
        data = F.normalize(torch.Tensor(data)).numpy()
        data = self.reducer.fit_transform(data)
        self.cluster_model.fit(data)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Prédiction optimisée avec half-precision"""
        if not crops:
            return np.array([])
            
        with torch.cuda.amp.autocast():
            data = self.extract_features(crops)
            data = F.normalize(torch.Tensor(data)).numpy()
            data = self.reducer.transform(data)
            return self.cluster_model.predict(data)
