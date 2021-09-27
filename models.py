"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.cuda.current_device()
    torch.cuda.get_device_name(0)
    torch.cuda.set_device(0)
    

class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
    
class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
        
class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64], 
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        self.embedding_sharing = embedding_sharing
        
        # implement embedding sharing
        self.user_embedding = ScaledEmbedding(num_users, embedding_dim)
        self.item_embedding = ScaledEmbedding(num_items, embedding_dim)

        self.user_bias_embedding = ZeroEmbedding(num_users, 1)
        self.item_bias_embedding = ZeroEmbedding(num_items, 1)
        
        # implement MLP network of input 96, ouput 1
        self.layers = nn.ModuleList([
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ])
        
        # implement separated embedding, we create another embedding vectors to use for another task 
        if embedding_sharing is False:
            self.user_embedding_prime = ScaledEmbedding(num_users, embedding_dim)
            self.item_embedding_prime = ScaledEmbedding(num_items, embedding_dim)

            self.user_bias_embedding_prime = ZeroEmbedding(num_users, 1)
            self.item_bias_embedding_prime = ZeroEmbedding(num_items, 1)
        
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        
    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of 
            shape (batch,). This corresponds to p_ij in the 
            assignment.
        score: tensor
            Tensor of user-item score predictions of shape 
            (batch,). This corresponds to r_ij in the 
            assignment.
        """
        
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        # matrix multiplication - probability that user would watch the movie
        # using batch for matrix multilication will extend another dimension, so we need to sum the columns dimension                
        if self.embedding_sharing is False:
            # for separated embedding, we use the "prime" embedding vectors for matrix factorization
            predictions = ((self.item_embedding_prime(item_ids)) @ self.user_embedding_prime(user_ids).T).sum(dim=1, keepdim=False) 
            + self.user_bias_embedding_prime(user_ids) + self.item_bias_embedding_prime(item_ids)
        else:
            # for shared embedding, we use the same embedding vectors for both matrix factorization and regression
            predictions = ((self.item_embedding(item_ids)) @ self.user_embedding(user_ids).T).sum(dim=1, keepdim=False) 
            + self.user_bias_embedding(user_ids) + self.item_bias_embedding(item_ids)
        
        # element wise multiplication - score that user would assign the movie
        # concat input to size 96
        score = torch.cat([self.user_embedding(user_ids), self.item_embedding(item_ids), self.user_embedding(user_ids) * self.item_embedding(item_ids)], dim = 1)

        # feed input to the network
        for layer in self.layers:
            score = layer(score)
            
        predictions = predictions.squeeze()
        score = score.squeeze()
        #********************************************************
        #********************************************************
        #********************************************************
        return predictions, score
