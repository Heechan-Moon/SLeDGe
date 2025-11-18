import torch
import torch.nn.functional as F
import random

class Window_time_decaying:
    def __init__(self, buffer_size, device, isLabel=True):
        self.buffer_size = buffer_size
        self.device = device
        self.isLabel=isLabel
        
        self.buffer = []
        self.time_buffer_label = []
        if self.isLabel:
            self.labels = []

    def get_centroids(self):
        if self.isLabel:
            if len(self.buffer)==0:
                return torch.empty(0, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)
            return torch.stack(self.buffer), torch.stack(self.labels)
        else:
            if len(self.buffer)==0:
                return torch.empty(0, device=self.device)
            return torch.stack(self.buffer)

    def get_time_decaying(self):
        if len(self.time_buffer_label)==0:
            return torch.empty(0, device=self.device)
        else:
            return torch.tensor(self.time_buffer_label, device=self.device)

    def partial_fit(self, feat, label=None):
        self.buffer.append(feat)
        if len(self.time_buffer_label)!=0:
            self.time_buffer_label = [x - 1 for x in self.time_buffer_label]
        self.time_buffer_label.append(0)
        if self.isLabel:
            self.labels.append(label)

        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
            self.time_buffer_label = self.time_buffer_label[-self.buffer_size:]
            if self.isLabel:
                self.labels = self.labels[-self.buffer_size:]


class Update:
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        
        self.centroids = None
        self.counts = torch.ones(self.buffer_size, device=self.device)

    def partial_fit(self, new_point):
        new_point = F.normalize(new_point.unsqueeze(0), p=2, dim=1)
        
        if self.centroids is None:
            self.centroids = new_point
        elif self.centroids.shape[0] < self.buffer_size:
            self.centroids = torch.cat((self.centroids, new_point), dim=0)
        else:
            # Cosine similarity
            sim = self.centroids @ new_point.T
            cluster_idx = torch.argmax(sim).item()
    
            # Update count
            self.counts[cluster_idx] += 1
            eta = 1.0 / self.counts[cluster_idx]
    
            # Incremental update of centroid
            #updated = (1 - eta) * self.centroids[cluster_idx] + eta * new_point
            updated = self.centroids[cluster_idx] + new_point
            self.centroids[cluster_idx] = F.normalize(updated, p=2, dim=1) 

    def partial_fit_ver2(self, new_point, new_point_embs, embs):
        #new_point = F.normalize(new_point.unsqueeze(0), p=2, dim=1)
        new_point = F.normalize(new_point, p=2, dim=1)
        
        new_point_embs = F.normalize(new_point_embs, p=2, dim=1)
        centroid_embs = F.normalize(embs, p=2, dim=1)
        if self.centroids is None:
            self.centroids = new_point
            return
        elif centroid_embs.shape[0] != self.buffer_size:
            self.centroids = torch.cat((self.centroids, new_point), dim=0)
            return
        
        # Cosine similarity
        sim = centroid_embs @ new_point_embs.T
        cluster_idx = torch.argmax(sim).item()

        # Update count
        self.counts[cluster_idx] += 1
        eta = 1.0 / self.counts[cluster_idx]

        # Incremental update of centroid
        #updated = (1 - eta) * self.centroids[cluster_idx] + eta * new_point
        updated = self.centroids[cluster_idx] + new_point
        self.centroids[cluster_idx] = F.normalize(updated, p=2, dim=1)

    def get_centroids(self):
        return self.centroids if self.centroids is not None else None
