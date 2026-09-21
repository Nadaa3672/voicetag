"""
Tagger acustico: CNN su spettrogrammi log-Mel.

Tre blocchi convoluzionali (conv 3x3 - batch norm - ReLU - max pooling), global
average pooling, dropout e strato lineare finale. L'ingresso ha tre canali:
log-Mel, derivata prima e derivata seconda lungo il tempo, perche' l'emozione
sta soprattutto nella dinamica della voce e non nello spettro statico.

La rete e' volutamente piccola: il corpus di addestramento conta 960 clip, e
una rete piu' capiente memorizzerebbe invece di generalizzare.
"""
import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self, n_classes: int = 8, dropout: float = 0.3,
                 width: int = 32, in_channels: int = 3):
        super().__init__()
        w = int(width)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, w),
            block(w, w * 2),
            block(w * 2, w * 4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(w * 4, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_checkpoint(path, device=None):
    """
    Ricostruisce la rete da un checkpoint che contiene pesi, iperparametri,
    numero di canali e statistiche di normalizzazione (media, deviazione
    standard del log-Mel calcolate sul training set).
    """
    device = device or torch.device("cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    model = EmotionCNN(
        n_classes=8,
        dropout=float(ck["hp"].get("dropout", 0.3)),
        width=int(ck["hp"].get("width", 32)),
        in_channels=int(ck.get("in_channels", 3)),
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mean, std = ck["norm"]
    return model, float(mean), float(std)
