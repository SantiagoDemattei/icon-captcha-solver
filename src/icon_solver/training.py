import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from icon_solver.challenges import VAL_SPLIT, split_manifest
from icon_solver.dataset.build import MANIFEST_FILE, META_FILE, PROTOTYPES_FILE
from icon_solver.dataset.torch_dataset import IconMaskDataset
from icon_solver.evaluation import accuracy, balanced_accuracy, confusion_matrix
from icon_solver.legend import load_prototypes
from icon_solver.model import IconClassifier, IconModel, default_device
from icon_solver.paths import PROCESSED_DIR
from icon_solver.solver import DEFAULT_MODEL


def _batch_loss(logits, y, exclude, criterion) -> torch.Tensor:
    positive = y >= 0
    loss = logits.new_zeros(())
    if positive.any():
        loss = loss + criterion(logits[positive], y[positive]) * positive.sum() / y.size(0)
    if (~positive).any():
        # -log(1 - P(alguna clase pedida)) = -logsumexp de las log-probs de las clases permitidas.
        log_probs = torch.log_softmax(logits[~positive], dim=1).masked_fill(exclude[~positive], float("-inf"))
        loss = loss - torch.logsumexp(log_probs, dim=1).sum() / y.size(0)
    return loss


def _run_train_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    correct, total, running_loss, seen = 0, 0, 0.0, 0
    for x, y, exclude in loader:
        x, y, exclude = x.to(device), y.to(device), exclude.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = _batch_loss(logits, y, exclude, criterion)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
        seen += y.size(0)
        positive = y >= 0
        correct += (logits[positive].argmax(1) == y[positive]).sum().item()
        total += positive.sum().item()
    return running_loss / seen, correct / max(1, total)


def seed_everything(seed: int) -> None:
    # El shuffle de los DataLoader usa el generador global de torch; la muestra de negativos y la
    # rotacion de la augmentacion, el de random.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = VAL_SPLIT,
    negative_ratio: float = 1.0,
    out_path: Path = DEFAULT_MODEL,
    processed_dir: Path = PROCESSED_DIR,
    seed: int | None = None,
) -> IconModel:
    if seed is not None:
        seed_everything(seed)
    device = default_device()
    num_classes = json.loads((processed_dir / META_FILE).read_text())["num_classes"]
    prototypes = load_prototypes(processed_dir / PROTOTYPES_FILE)
    if prototypes.num_classes != num_classes:
        raise ValueError(
            f"el dataset tiene {num_classes} clases y los prototipos {prototypes.num_classes}: reconstruir el dataset"
        )
    manifest = json.loads((processed_dir / MANIFEST_FILE).read_text())
    train_indices, val_indices = split_manifest(manifest, val_split)
    positive_indices = [i for i in train_indices if manifest[i]["class_id"] >= 0]
    negative_indices = [i for i in train_indices if manifest[i]["class_id"] < 0] if negative_ratio > 0 else []
    # Con negativos se agrega una salida extra "fondo": las formas que no son ningun icono
    # necesitan un lugar donde poner la probabilidad que el softmax las obliga a repartir.
    num_outputs = num_classes + (1 if negative_indices else 0)

    def dataset(indices, augment):
        return IconMaskDataset(processed_dir, augment=augment, indices=indices, num_outputs=num_outputs)

    positive_set = dataset(positive_indices, True)
    negative_set = dataset(negative_indices, True)
    val_loader = DataLoader(dataset(val_indices, False), batch_size=batch_size)
    positive_loader = DataLoader(positive_set, batch_size=batch_size, shuffle=True)

    def epoch_loader():
        if not negative_indices:
            return positive_loader
        # Hay muchos mas negativos que positivos: cada epoch usa una muestra distinta.
        k = min(len(negative_indices), int(len(positive_indices) * negative_ratio))
        sample = Subset(negative_set, random.sample(range(len(negative_indices)), k))
        return DataLoader(ConcatDataset([positive_set, sample]), batch_size=batch_size, shuffle=True)

    classifier = IconClassifier(num_outputs).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_acc, best_balanced = 0.0, 0.0
    for epoch in range(epochs):
        train_loss, train_acc = _run_train_epoch(classifier, epoch_loader(), optimizer, criterion, device)
        scheduler.step()
        confusion = confusion_matrix(classifier, val_loader, num_classes, num_outputs, device)
        val_acc, val_balanced = accuracy(confusion), balanced_accuracy(confusion)
        print(
            f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} val_bal_acc={val_balanced:.3f}"
        )
        if val_acc >= best_val_acc:
            best_val_acc, best_balanced = val_acc, val_balanced

    # El ultimo epoch, no el de mejor val_acc: con schedule coseno el final es estable, y el maximo
    # sobre 200 evaluaciones de un val set chico es ruido que el solver end-to-end no premia.
    classifier.eval()
    model = IconModel(classifier, prototypes, num_classes)
    model.save(out_path)
    print(f"mejor val_acc: {best_val_acc:.3f} (bal_acc {best_balanced:.3f}) -> {out_path}")
    return model
