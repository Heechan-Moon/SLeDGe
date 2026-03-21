import torch
import torch.nn.functional as F

class Update_adaptive_EMA:
    def __init__(self, n_classes, feat_dim, labeled_buffer_size, unlabeled_buffer_size, device):
        self.device = device

        self.n_classes = n_classes
        self.feat_dim = feat_dim
        
        self.buffer_per_label = int(labeled_buffer_size/n_classes)
        self.labeled_buffer_size = self.buffer_per_label * n_classes 
        self.unlabeled_buffer_size = unlabeled_buffer_size 

        self.labeled_bank = torch.zeros((self.labeled_buffer_size, feat_dim), device=device)
        self.labeled_counts = torch.zeros(self.labeled_buffer_size, device=device)
        self.labeled_filled = torch.zeros(self.labeled_buffer_size, dtype=torch.bool, device=device)
        self.static_labels = torch.arange(self.n_classes, device=device).repeat_interleave(self.buffer_per_label)
        
        self.labeled_age = torch.zeros(self.labeled_buffer_size, device=device)
        
        self.unlabeled_bank = torch.zeros((self.unlabeled_buffer_size, feat_dim), device=device)
        self.unlabeled_counts = torch.zeros(self.unlabeled_buffer_size, device=device)
        self.unlabeled_filled = torch.zeros(self.unlabeled_buffer_size, dtype=torch.bool, device=device)
        self.current_count = 0

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.n_classes}, "
            f"feat_dim={self.feat_dim}, "
            f"labeled_buffer_size={self.labeled_buffer_size}, "
            f"buffer_per_label={self.buffer_per_label}, "
            f"unlabeled_buffer_size={self.unlabeled_buffer_size}, "
            f"current_count={self.current_count}, "
            f"labeled_counts={self.labeled_counts.shape}, "
            f"unlabeled_counts={self.unlabeled_counts.shape}"
            f")"
        )

    def partial_fit(self, feat, feat_embs, embs, label=None):
        if label is not None: # Labeled
            self.labeled_age[self.labeled_filled] += 1
            
            cls_idx = label.item()

            start_idx = cls_idx * self.buffer_per_label
            end_idx = start_idx + self.buffer_per_label

            cls_filled = self.labeled_filled[start_idx:end_idx]
            empty_rel_indices = torch.where(~cls_filled)[0]

            if len(empty_rel_indices) > 0:
                slot_idx = start_idx + empty_rel_indices[0].item()
                self.labeled_bank[slot_idx] = feat
                self.labeled_counts[slot_idx] = 1
                self.labeled_filled[slot_idx] = True
                self.labeled_age[slot_idx] = 0
            else:
                feat = feat.unsqueeze(0)

                if embs[start_idx:end_idx].shape[0] != self.buffer_per_label:
                    print("Error")
                sims = embs[start_idx:end_idx] @ feat_embs.T
                
                rel_target = torch.argmax(sims).item()
                slot_idx = start_idx + rel_target

                count_weight = 1 - (1.0 / (self.labeled_counts[slot_idx] + 1))
                alpha = count_weight * max(0, sims[rel_target].item())
                
                updated = (1 - alpha) * self.labeled_bank[slot_idx] + alpha * feat
                self.labeled_bank[slot_idx] = F.normalize(updated, p=2, dim=1)
                self.labeled_counts[slot_idx] += 1
                self.labeled_age[slot_idx] = 0
        else: # Unlabeled
            if self.current_count < self.unlabeled_buffer_size:
                slot_idx = self.current_count
                self.unlabeled_bank[slot_idx] = feat
                self.unlabeled_counts[slot_idx] = 1
                self.unlabeled_filled[slot_idx] = True
                self.current_count += 1
            else:
                feat = feat.unsqueeze(0)

                if embs.shape[0] != self.unlabeled_buffer_size:
                    print("Error")
                sim = embs @ feat_embs.T
                slot_idx = torch.argmax(sim).item()

                max_sim = sim[slot_idx].item()
                count_weight = 1 - (1.0 / (self.unlabeled_counts[slot_idx] + 1))
                beta = count_weight * max(0, max_sim)

                updated = beta * self.unlabeled_bank[slot_idx] + (1 - beta) * feat
                self.unlabeled_bank[slot_idx] = F.normalize(updated, p=2, dim=1)
                self.unlabeled_counts[slot_idx] += 1

        activated_labeled_bank = self.labeled_bank[self.labeled_filled]
        activated_label = self.static_labels[self.labeled_filled]

        activated_unlabeled_bank = self.unlabeled_bank[self.unlabeled_filled]
        
        X_combined = torch.cat([activated_labeled_bank, activated_unlabeled_bank], dim=0)
        
        return X_combined, activated_label
        
    def get_test(self, input):
        activated_labeled_bank = self.labeled_bank[self.labeled_filled]
        activated_unlabeled_bank = self.unlabeled_bank[self.unlabeled_filled]
        X_combined = torch.cat([activated_labeled_bank, activated_unlabeled_bank, input], dim=0)
        
        return X_combined
