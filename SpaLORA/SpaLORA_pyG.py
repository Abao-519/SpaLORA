import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing


class Train_SpaLORA:
    def __init__(self,
                 data,
                 datatype='10x Genomics Visium',
                 device=torch.device('cpu'),
                 random_seed=2022,
                 learning_rate=0.0001,
                 weight_decay=0.00,
                 epochs=200,
                 dim_input=3000,
                 dim_output=128
                 ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports '10x Genomics Visium', 'Slide-tags', and 'spatial ATAC–RNA-seq'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 200.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 128.

        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''

        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output

        if self.datatype == 'Slide-tags':
            self.weight_factors = [5, 6, 1, 10]

        elif self.datatype == '10x Genomics Visium':
            self.dim_output = 64
            self.weight_factors = [1.9, 2.5, 1.5, 10]

        elif self.datatype == 'spatial ATAC–RNA-seq':
            self.epochs = 1600
            self.weight_factors = [1.5, 5, 1.5, 1]

        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)

        # Features for omics1 (RNA)
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['raw_feat'].copy()).to(self.device)

        # Calculate the global average expression for each gene in RNA modality
        avg_expr = self.features_omics1.mean(dim=0)

        # Sort genes by expression level in ascending order to get gene indices
        sorted_indices = torch.argsort(avg_expr, descending=False)

        expression_threshold = 0.25  # Default: lowest 25% of genes are considered low-expression
        amplification_factor = 6     # The factor to amplify loss weighting for low-expression genes
        n_genes = self.features_omics1.shape[1]
        threshold_index = int(n_genes * expression_threshold)

        # Calculate the expression percentile of each gene (0 to 1)
        expr_percentile = torch.argsort(avg_expr, descending=False) / n_genes

        # Compute the weight vector for genes: low-expression genes receive higher weights
        self.weight_vector_omics1 = 1 + (amplification_factor - 1) * torch.sigmoid(-10 * (expr_percentile - expression_threshold))
        self.weight_vector_omics1 = self.weight_vector_omics1.to(self.device)

        # Features for omics2 (second modality)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        # Number of cells/spots for each omics modality
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs


        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(
            self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

            # weighted reconstruction loss
            diff = self.features_omics1 - results['emb_recon_omics1']
            weight = self.weight_vector_omics1.unsqueeze(0)
            self.loss_recon_omics1 = torch.mean((diff ** 2) * weight)

            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])

            # correspondence loss
            self.loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
            self.loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])

            loss = self.weight_factors[0] * self.loss_recon_omics1 + self.weight_factors[1] * self.loss_recon_omics2 + \
                   self.weight_factors[2] * self.loss_corr_omics1 + self.weight_factors[3] * self.loss_corr_omics2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Model training finished!\n")

        with torch.no_grad():
            self.model.eval()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'SpaLORA': emb_combined.detach().cpu().numpy(),
                  'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
                  'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy()}

        return output