import torch
import torch.nn as nn


class ROIPre(nn.Module):
    def __init__(self, num_classes):
        super(ROIPre, self).__init__()

        self.roi_pooling = nn.AdaptiveMaxPool2d((7, 7))
        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc_class = nn.Linear(1024, num_classes)
        self.fc_bbox = nn.Linear(1024, 4 * num_classes)

    def forward(self, x):
        # ROI pooling
        roi_features = self.roi_pooling(x)

        # Reshape
        roi_features = roi_features.view(roi_features.size(0), -1)

        # Classification and bounding box regression
        fc1_out = self.fc1(roi_features)
        fc1_out = self.relu1(fc1_out)
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.relu2(fc2_out)
        class_scores = self.fc_class(fc2_out)
        bbox_preds = self.fc_bbox(fc2_out)

        return class_scores, bbox_preds
