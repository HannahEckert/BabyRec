
import numpy as np
import torch
import ltn


class TrainingDataLoader:
    """
    data loader for implicit feedback. Creates batches with user-item interactions and negative samples.
    """

    def __init__(self, data, user_features, item_features, n_items, batch_size=1, shuffle=True, LTN=False, M_user=None, f1=False, f2=False, f3=False):
        """
        :param data: list of tuples (user, item) where interaction == 1
        :param user_features: dictionary {user_id: feature_vector}
        :param item_features: dictionary {item_id: feature_vector}
        :param n_items: total number of items
        :param batch_size: batch size
        :param shuffle: whether to shuffle data
        """
        self.data = np.array(data)
        self.user_features = user_features
        self.item_features = item_features
        self.n_items = n_items
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.LTN = LTN
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.M_user = M_user

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            batch = self.data[idxlist[start_idx:end_idx]]

            u_idx = batch[:, 0]
            i_pos_idx = batch[:, 1]

            i_neg_idx = [np.inf]
            while not(i_neg_idx[0] in self.item_features.keys()):
                i_neg_idx = np.random.randint(0, self.n_items, size=len(u_idx))  # negative sampling

            u_feat = torch.tensor(np.array([self.user_features[u] for u in u_idx]))
            i_pos_feat = torch.tensor(np.array([self.item_features[i] for i in i_pos_idx]))
            i_neg_feat = torch.tensor(np.array([self.item_features[i] for i in i_neg_idx]))

            user = torch.cat([torch.tensor(u_idx).unsqueeze(1), u_feat], dim=1)
            positive_item = torch.cat([torch.tensor(i_pos_idx).unsqueeze(1), i_pos_feat], dim=1)
            negative_item = torch.cat([torch.tensor(i_neg_idx).unsqueeze(1), i_neg_feat], dim=1)

            if self.f1 and self.f2:
                M_global = torch.tensor(self.M_user["M_global_D_APC"])
                M_local = torch.tensor(self.M_user["M_local_D_APC"])
                u_idx_f1 = M_global[u_idx] < 0.33
                u_idx_f2 = M_local[u_idx] > 0.66
                user_f1 = user[u_idx_f1]
                user_f2 = user[u_idx_f2]
                positive_item_f1 = positive_item[u_idx_f1]
                positive_item_f2 = positive_item[u_idx_f2]
                negative_item_f1 = negative_item[u_idx_f1]
                negative_item_f2 = negative_item[u_idx_f2]
                yield ltn.Variable("user_f1",user_f1), ltn.Variable("positive item_f1",positive_item_f1), ltn.Variable("negative item_f1",negative_item_f1), ltn.Variable("user_f2",user_f2), ltn.Variable("positive item_f2",positive_item_f2), ltn.Variable("negative item_f2",negative_item_f2), ltn.Variable("user",user), ltn.Variable("positive item",positive_item), ltn.Variable("negative item",negative_item)



            elif self.f1:
                M = torch.tensor(self.M_user["M_global_D_APC"])
                u_idx_f1 = M[u_idx] < 0.33
                user_f1 = user[u_idx_f1]
                positive_item_f1 = positive_item[u_idx_f1]
                negative_item_f1 = negative_item[u_idx_f1]
                
                yield ltn.Variable("user_f1",user_f1), ltn.Variable("positive item_f1",positive_item_f1), ltn.Variable("negative item_f1",negative_item_f1), ltn.Variable("user",user), ltn.Variable("positive item",positive_item), ltn.Variable("negative item",negative_item)

            elif self.f2: 
                M = torch.tensor(self.M_user["M_local_D_APC"])
                u_idx_f2 = M[u_idx] > 0.66
                user_f2 = user[u_idx_f2]
                positive_item_f2 = positive_item[u_idx_f2]
                negative_item_f2 = negative_item[u_idx_f2]

                yield ltn.Variable("user_f2",user_f2), ltn.Variable("positive item_f2",positive_item_f2), ltn.Variable("negative item_f2",negative_item_f2), ltn.Variable("user",user), ltn.Variable("positive item",positive_item), ltn.Variable("negative item",negative_item)

            elif self.f3:
                M = torch.tensor(self.M_user["M_local_D_APC"])
                u_idx_f3 = M[u_idx] > 0.66
                user_f3 = user[u_idx_f3]
                positive_item_f3 = positive_item[u_idx_f3]
                negative_item_f3 = negative_item[u_idx_f3]

                yield ltn.Variable("user_f3",user_f3), ltn.Variable("positive item_f3",positive_item_f3), ltn.Variable("negative item_f3",negative_item_f3), ltn.Variable("user",user), ltn.Variable("positive item",positive_item), ltn.Variable("negative item",negative_item)
            else:
                if self.LTN:
                    #yield ltn.Variable("users",torch.tensor(u_idx)), ltn.Variable("positive items",torch.tensor(i_pos_idx)), ltn.Variable("negative items",torch.tensor(i_neg_idx)), ltn.Variable("user features",u_feat.type(torch.LongTensor)), ltn.Variable("positive item features",i_pos_feat.type(torch.LongTensor)), ltn.Variable("negative item features",i_neg_feat.type(torch.LongTensor))
                    yield ltn.Variable("user",user), ltn.Variable("positive item",positive_item), ltn.Variable("negative item",negative_item)
                else:
                    #yield torch.tensor(u_idx), torch.tensor(i_pos_idx), torch.tensor(i_neg_idx), u_feat, i_pos_feat, i_neg_feat
                    yield user, positive_item, negative_item


class TrainingDataLoaderMultVAE:
    """
    Data loader for implicit feedback. 
    Creates batches with user features and interactions."""
    
    def __init__(self, user_features, all_user_items, n_users, n_items, batch_size=1, shuffle=True, LTN=False):
        """
        :param batch_size: batch size
        :param shuffle: whether to shuffle data
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.user_features = user_features
        self.all_user_items = all_user_items
        self.n_users = n_users
        self.n_items = n_items
        self.LTN = LTN


    def __len__(self):
        return int(np.ceil(self.n_users / self.batch_size))

    def __iter__(self):
        n = self.n_users
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            batch = idxlist[start_idx:end_idx]

            u_feat = torch.tensor(np.array([self.user_features[u] for u in batch]))
            interaction_matrix = []
            
            for user in batch:
                user_items = self.all_user_items[user]
                interactions = np.zeros((self.n_items,), dtype=np.float32)
                interactions[list(user_items)] = 1.0
                interaction_matrix.append(interactions)
            interaction_matrix = torch.tensor(np.array(interaction_matrix))

            if self.LTN:
                yield ltn.Variable("user",u_feat.long()), ltn.Variable("interactions",interaction_matrix)
            else:
                yield u_feat, interaction_matrix





class ValidationDataLoader:
    """
    Data loader for validation/testing, generating candidate sets of 1 positive and multiple negatives.
    """

    def __init__(self, data, user_features, item_features, all_user_items, n_items, n_negatives=99, batch_size=1,LTN=False):
        """
        :param data: list of tuples (user, positive_item)
        :param user_features: dictionary {user_id: feature_vector}
        :param item_features: dictionary {item_id: feature_vector}
        :param all_user_items: dictionary {user_id: set of interacted items}
        :param n_items: total number of items
        :param n_negatives: number of negatives per user
        :param batch_size: batch size
        """
        self.data = data
        self.user_features = user_features
        self.item_features = item_features
        self.all_user_items = all_user_items
        self.n_items = n_items
        self.n_negatives = n_negatives
        self.batch_size = batch_size
        self.LTN = LTN

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __iter__(self):
        for start_idx in range(0, len(self.data), self.batch_size):
            batch = self.data[start_idx:start_idx + self.batch_size]

            u_idx, i_pos_idx, candidates, candidate_feats, u_feats = [], [], [], [], []

            for u, pos_item in batch:
                # Negative sampling ensuring we don't sample interacted items
                negatives = []
                while len(negatives) < self.n_negatives:
                    neg = np.random.randint(0, self.n_items)
                    if neg not in self.all_user_items[u]:
                        negatives.append(neg)

                candidate_items = [pos_item] + negatives  # First is positive, others negatives

                u_idx.append(u)
                i_pos_idx.append(pos_item)
                candidates.append(candidate_items)
                u_feats.append(self.user_features[u])

                # Collect item features
                candidate_feat = [self.item_features[i] for i in candidate_items]
                candidate_feats.append(candidate_feat)

            # Convert to tensors
            u_idx = torch.tensor(u_idx).unsqueeze(1)  # Shape: [batch_size, 1]
            i_pos_idx = torch.tensor(i_pos_idx).unsqueeze(1)  # Shape: [batch_size, 1]
            candidates = torch.tensor(candidates)  # Shape: [batch_size, n_negatives + 1]
            u_feats = torch.tensor(np.array(u_feats)) # Shape: [batch_size, user_feat_dim]
            candidate_feats = torch.tensor(np.array(candidate_feats))  # Shape: [batch_size, n_negatives + 1, item_feat_dim]

            # Combine user ID with user features
            user = torch.cat([u_idx, u_feats], dim=1)  # Shape: [batch_size, 1 + user_feat_dim]
            
            # Get positive item features (first in candidate set)
            positive_item_feats = candidate_feats[:, 0, :]  # Shape: [batch_size, item_feat_dim]
            positive_item = torch.cat([i_pos_idx, positive_item_feats], dim=1)  # Shape: [batch_size, 1 + item_feat_dim]
            
            # Get negative items (all except first in candidate set)
            negative_item_ids = candidates[:, 1:]  # Shape: [batch_size, n_negatives]
            negative_item_feats = candidate_feats[:, 1:, :]  # Shape: [batch_size, n_negatives, item_feat_dim]
            
            # Reshape to combine IDs and features
            negative_item_ids = negative_item_ids.unsqueeze(2)  # Shape: [batch_size, n_negatives, 1]
            negative_items = torch.cat([negative_item_ids, negative_item_feats], dim=2)  # Shape: [batch_size, n_negatives, 1 + item_feat_dim]


            if self.LTN:
                yield (
                    ltn.Variable("user",user),
                    ltn.Variable("positive item",positive_item),
                    ltn.Variable("negative items",negative_items)
                )
            else:
                yield (
                    user,
                    positive_item,
                    negative_items
                )

class ValidationDataLoaderMultVAE:
    """
    Data loader for validation/testing, generating candidate sets of 1 positive and multiple negatives.

    """

    def __init__(self, data, user_features, all_user_items, n_users, n_items, n_negatives=99, batch_size=1,LTN=False):

        self.data = data
        self.user_features = user_features
        self.all_user_items = all_user_items
        self.n_users = n_users
        self.n_items = n_items
        self.n_negatives = n_negatives
        self.batch_size = batch_size
        self.LTN = LTN

    def __len__(self):
        return int(np.ceil(self.n_users / self.batch_size))
    
    def __iter__(self):
        for start_idx in range(0, len(self.data), self.batch_size):
            batch = self.data[start_idx:start_idx + self.batch_size]

            u_idx, i_pos_idx, candidates, u_feats, user_interactions = [], [], [], [], []

            for u, pos_item in batch:
                # Negative sampling ensuring we don't sample interacted items
                negatives = []
                while len(negatives) < self.n_negatives:
                    neg = np.random.randint(0, self.n_items)
                    if neg not in self.all_user_items[u]:
                        negatives.append(neg)

                candidate_items = [pos_item] + negatives  # First is positive, others negatives

                u_idx.append(u)
                i_pos_idx.append(pos_item)
                candidates.append(candidate_items)

                # user features
                u_feats.append(self.user_features[u])

                #user interactions
                user_items = self.all_user_items[u]
                interactions = np.zeros((self.n_items,), dtype=np.float32)
                interactions[list(user_items)] = 1.0
                user_interactions.append(interactions)

            # Convert to tensors
            u_idx = torch.tensor(u_idx).unsqueeze(1)  # Shape: [batch_size, 1]
            i_pos_idx = torch.tensor(i_pos_idx).unsqueeze(1)
            candidates = torch.tensor(candidates)  # Shape: [batch_size, n_negatives + 1]
            u_feats = torch.tensor(np.array(u_feats))
            user_interactions = torch.tensor(np.array(user_interactions))

            yield u_idx, u_feats, candidates, user_interactions