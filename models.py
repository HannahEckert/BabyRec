
import torch
import torch.nn as nn
import numpy as np
from metrics import compute_metric, check_metrics
import torch.nn.functional as F

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors, bpr=False):
        """
        Hybrid model: Matrix Factorization (MF) + Multi-Layer Perceptron (MLP)
        """
        super(MatrixFactorization, self).__init__()

        # User and item embeddings for MF branch
        self.u_emb = nn.Embedding(n_users, n_factors)
        self.i_emb = nn.Embedding(n_items, n_factors)

        self.bpr = bpr

        # Numerical feature (age) transformation
        self.u_age_fc = nn.Linear(1, n_factors // 2)

        # MLP to learn interactions from explicit features
        mlp_input_dim = (
            n_factors  # User embedding
            + n_factors  # Item embedding
        )

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_input_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output a single prediction score
        )


        # Initialize embeddings
        nn.init.xavier_normal_(self.u_emb.weight)
        nn.init.xavier_normal_(self.i_emb.weight)

    #def forward(self, u_idx, i_idx, u_feat, i_feat):
    def forward(self, user, item):

        u_idx = user[:, 0].long()
        i_idx = item[:, 0].long()


        # MF part: Latent embeddings dot product
        u_emb = self.u_emb(u_idx)
        i_emb = self.i_emb(i_idx)

        # for embedings that come from LTN variables
        if len(u_emb.shape) == 3:
            u_emb = u_emb.squeeze()
        if len(i_emb.shape) == 3:
            i_emb = i_emb.squeeze()

        mf_pred = torch.sum(u_emb * i_emb, dim=1)

        

        # Concatenate all features
        user_features = u_emb
        item_features = i_emb
        mlp_input = torch.cat([user_features, item_features], dim=1)

        # MLP output
        mlp_pred = self.mlp(mlp_input).squeeze()

        if self.bpr:
            pred = mf_pred
        else:
            # Final prediction: MF + MLP
            pred = mf_pred + mlp_pred


        if self.bpr:
            return pred
        else:
            return torch.sigmoid(pred)  # Output in range [0,1]
        


    



class Trainer:
    """
    Abstract base class that any trainer must inherit from.
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, train_loader, val_loader, val_metric, n_epochs=200, early=None, verbose=10, save_path=None):
        """
        Method for the train of the model.

        :param train_loader: data loader for training data
        :param val_loader: data loader for validation data
        :param val_metric: validation metric name
        :param n_epochs: number of epochs of training, default to 200
        :param early: threshold for early stopping, default to None
        :param verbose: number of epochs to wait for printing training details (every 'verbose' epochs)
        :param save_path: path where to save the best model, default to None
        """
        best_val_score = 0.0
        early_counter = 0
        check_metrics(val_metric)

        for epoch in range(n_epochs):
            # training step
            train_loss = self.train_epoch(train_loader)
            # validation step
            val_score = self.validate(val_loader, val_metric)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                print("Epoch %d - Train loss %.3f - Validation %s %.3f"
                      % (epoch + 1, train_loss, val_metric, val_score))
            # save best model and update early stop counter, if necessary
            if val_score > best_val_score:
                best_val_score = val_score
                early_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    print("Training interrupted due to early stopping")
                    break

    def train_epoch(self, train_loader):
        """
        Method for the training of one single epoch.

        :param train_loader: data loader for training data
        :return: training loss value averaged across training batches
        """
        raise NotImplementedError()

    def predict(self, x, *args, **kwargs):
        """
        Method for performing a prediction of the model.

        :param x: input for which the prediction has to be performed
        :param args: these are the potential additional parameters useful to the model for performing the prediction
        :param kwargs: these are the potential additional parameters useful to the model for performing the prediction
        :return: prediction of the model for the given input
        """

    def validate(self, val_loader, val_metric):
        """
        Method for validating the model.

        :param val_loader: data loader for validation data
        :param val_metric: validation metric name
        :return: validation score based on the given metric averaged across validation batches
        """
        raise NotImplementedError()

    def save_model(self, path):
        """
        Method for saving the model.

        :param path: path where to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        """
        Method for loading the model.

        :param path: path from which the model has to be loaded.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def test(self, test_loader, metrics):
        """
        Method for performing the test of the model based on the given test data and test metrics.

        :param test_loader: data loader for test data
        :param metrics: metric name or list of metrics' names that have to be computed
        :return: a dictionary containing the value of each metric average across the test batches
        """
        raise NotImplementedError()
    

class MFTrainer(Trainer):
    def __init__(self, mf_model, optimizer, item_features):
        self.item_features = item_features
        super().__init__(mf_model, optimizer)
    
   
    #define bpr loss
    def bpr_loss(self, pos_pred, neg_pred):
        if pos_pred is None or neg_pred is None:
            raise ValueError("BPR loss received None predictions")

        return -torch.mean(F.logsigmoid(pos_pred.squeeze() - neg_pred.squeeze()))
    
    #bce loss
    bce_loss = torch.nn.BCELoss()
    


    def train_epoch(self, train_loader, loss_function="bce", factor_axiom_vs_task=1):
        train_loss = 0.0
        for batch_idx, (user, positive_item, negative_item) in enumerate(train_loader):
            self.optimizer.zero_grad()

            # Move all inputs to the same device as the model
            device = next(self.model.parameters()).device
            user = user.to(device)
            positive_item = positive_item.to(device)
            negative_item = negative_item.to(device)

            pos_pred = self.model(user, positive_item)
            neg_pred = self.model(user, negative_item)

            if loss_function == "bpr":
                loss = self.bpr_loss(pos_pred, neg_pred)
            elif loss_function == "bce":
                # Move targets to the same device as predictions
                pos_target = torch.ones_like(pos_pred, device=device)
                neg_target = torch.zeros_like(neg_pred, device=device)
                loss = self.bce_loss(pos_pred, pos_target) + self.bce_loss(neg_pred, neg_target)
            
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss / len(train_loader)
    
    def validate(self, val_loader, val_metric, give_userwise = False, give_everything = False):
        """
        Validation logic using NDCG/Recall/Hit.
        Optimized for GPU usage and batch processing.
        
        :param val_loader: validation data loader
        :param val_metric: validation metric, e.g., 'ndcg@10'
        """
        self.model.eval()
        device = next(self.model.parameters()).device  # Get device once
        
        all_scores = []
        all_ground_truth = []
        all_candidates = []

        with torch.no_grad():
            for user, positive_item, negative_items in val_loader:
                # Move entire batch to device at once
                user = user.to(device)
                positive_item = positive_item.to(device)
                negative_items = negative_items.to(device)
                
                batch_size = user.size(0)
                n_candidates = 1 + negative_items.size(1)  # 1 positive + n negatives
                
                # Combine all candidates (positive first, then negatives)
                candidates = torch.cat([
                    positive_item.unsqueeze(1),  # [batch_size, 1, 1 + item_feat_dim]
                    negative_items
                ], dim=1)  # shape: [batch_size, n_candidates, 1 + item_feat_dim]
                
                # Prepare user tensor for batch processing
                user_repeated = user.unsqueeze(1).expand(-1, n_candidates, -1)
                
                # Reshape for batch processing
                user_flat = user_repeated.reshape(-1, user_repeated.size(-1))
                candidates_flat = candidates.reshape(-1, candidates.size(-1))
                
                # Compute all scores in one forward pass
                scores_flat = self.model(user_flat, candidates_flat)
                scores = scores_flat.reshape(batch_size, n_candidates)
                
                # Create ground truth matrix
                ground_truth = torch.zeros(batch_size, n_candidates, device=device)
                ground_truth[:, 0] = 1  # First item is positive
                
                all_scores.append(scores.cpu())
                all_ground_truth.append(ground_truth.cpu())
                all_candidates.append(candidates.cpu())

        # Concatenate all batches using torch (faster than numpy for GPU tensors)
        all_scores = torch.cat(all_scores, dim=0).numpy()
        all_ground_truth = torch.cat(all_ground_truth, dim=0).numpy()
        all_candidates = torch.cat(all_candidates, dim=0).numpy()

        # Compute metric
        result = compute_metric(val_metric, all_scores, all_ground_truth)

        if give_everything:
            return result, all_candidates, all_scores

        if give_userwise:
            return result
        else:
            return np.mean(result)  # Average over users
    


    
    def compute_test_loss(self, test_loader, loss_function="bce", give_userwise=False):
        """
        Compute average test loss without updating gradients.
        """
        self.model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        userwise_loss = []
        users = []

        with torch.no_grad():  # No gradients for test
            for batch_idx, (user, positive_item, negative_item) in enumerate(test_loader):

                # Move all inputs to the same device as the model
                device = next(self.model.parameters()).device
                user = user.to(device)
                positive_item = positive_item.to(device)
                negative_item = negative_item.to(device)

                pos_pred = self.model(user, positive_item)
                neg_pred = self.model(user, negative_item)

                if loss_function == "bpr":
                    loss = self.bpr_loss(pos_pred, neg_pred)
                elif loss_function == "bce":
                    loss = self.bce_loss(pos_pred, torch.ones_like(pos_pred)) + self.bce_loss(neg_pred, torch.zeros_like(neg_pred))

                test_loss += loss.item()
                if give_userwise:
                    userwise_loss.append(loss)
                    users.append(user.cpu().numpy())  # Store user IDs for userwise loss
        if give_userwise:
            return np.array(users), torch.tensor(userwise_loss)
        else:
            return test_loss / len(test_loader)
 

