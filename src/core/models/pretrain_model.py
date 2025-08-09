import torch
import torch.nn as nn

from src.common import AUGMENTATION_NUM_VIEWS, DOMAIN_DIMENSIONS, GNN_HIDDEN_DIM, NODE_FEATURE_MASKING_MASK_RATE
from src.core.layers import GradientReversalLayer
from src.core.models.gnn import InputEncoder, GIN_Backbone
from src.core.models.heads import MLPHead, DotProductDecoder, BilinearDiscriminator
from src.training.augmentations import GraphAugmentor


class PretrainableGNN(nn.Module):
    """
    Meta-model for multi-domain GNN pre-training.

    This model assembles all components needed for pre-training:
    - Domain-specific input encoders
    - Shared GNN backbone
    - Task-specific prediction heads
    - Gradient reversal layer for domain-adversarial training
    - Learnable [MASK] tokens for node feature masking
    - Graph augmentation capabilities for contrastive learning
    """

    def __init__(self, device: torch.device, domain_names: list[str], task_names: list[str], enable_augmentations: bool):
        """
        Initialize the pretrainable GNN model.

        Args:
            domain_dimensions: Dict mapping domain names to their input dimensions
            device: Device to place the model on ('cpu', 'cuda', or torch.device)
            enable_augmentations: Whether to enable graph augmentations
        """
        super(PretrainableGNN, self).__init__()

        self.device = device
        self.num_domains = len(domain_names)

        # --- Domain-specific Input Encoders ---
        self.input_encoders = nn.ModuleDict()
        for domain_name in domain_names:
            dim_in = DOMAIN_DIMENSIONS[domain_name]
            self.input_encoders[domain_name] = InputEncoder(dim_in=dim_in)

        # --- Domain-specific [MASK] tokens for node feature masking ---
        self.mask_tokens = nn.ParameterDict()
        for domain_name in domain_names:
            dim_in = DOMAIN_DIMENSIONS[domain_name]
            # Each domain needs its own [MASK] token with the correct input dimension
            # Use normal initialization (0 mean, small std)
            mask_token = torch.zeros(dim_in)
            # BERT-style initialization
            nn.init.normal_(mask_token, mean=0.0, std=0.02)
            self.mask_tokens[domain_name] = nn.Parameter(mask_token)

        # --- Shared GNN Backbone ---
        self.gnn_backbone = GIN_Backbone()

        # --- Task-specific Prediction Heads ---
        self.heads = nn.ModuleDict()
        for task_name in task_names:
            if task_name == 'node_feat_mask':
                # Node feature masking head (reconstruction)
                self.heads[task_name] = MLPHead()
            elif task_name == 'link_pred':
                # Link prediction head (non-parametric)
                self.heads[task_name] = DotProductDecoder()
            elif task_name == 'node_contrast':
                # Node contrastive learning projection head
                self.heads[task_name] = MLPHead(dim_out=GNN_HIDDEN_DIM // 2)
            elif task_name == 'graph_contrast':
                # Graph contrastive learning discriminator
                self.heads[task_name] = BilinearDiscriminator()
            elif task_name == 'graph_prop':
                # Graph property prediction head
                self.heads[task_name] = MLPHead(dim_hidden=GNN_HIDDEN_DIM * 2, dim_out=15)
            elif task_name == 'domain_adv':
                # Domain adversarial classifier
                self.heads[task_name] = MLPHead(dim_hidden=GNN_HIDDEN_DIM // 2, dim_out=self.num_domains)

        # --- Gradient Reversal Layer ---
        self.grl = GradientReversalLayer()

        # --- Graph Augmentation ---
        if enable_augmentations:
            self.augmentor = GraphAugmentor()
        else:
            self.augmentor = None

        # Move to device
        self.to(self.device)

    def apply_node_masking(self, data, domain_name: str):
        """
        Apply node feature masking for the node feature masking pre-training task.

        Args:
            data: PyTorch Geometric Data object
            domain_name: Name of the domain

        Returns:
            Tuple of (masked_data, mask_indices, target_h0):
                - masked_data: Data object with masked node features
                - mask_indices: Indices of masked nodes
                - target_h0: Original h_0 embeddings for masked nodes
        """
        # Move data to the correct device
        data = data.to(self.device)

        # Compute original h_0 embeddings
        encoder = self.input_encoders[domain_name]
        with torch.no_grad():
            original_h0 = encoder(data.x)

        # Clone the data to avoid modifying the original
        masked_data = data.clone()
        num_nodes = data.x.shape[0]

        # Ensure at least 1 node is masked
        num_mask = max(1, int(num_nodes * NODE_FEATURE_MASKING_MASK_RATE))

        # Randomly select nodes to mask
        mask_indices = torch.randperm(num_nodes, device=self.device)[:num_mask]

        # Store original h_0 embeddings for reconstruction targets
        target_h0 = original_h0[mask_indices].clone()

        # Replace selected node features with the learnable [MASK] token
        mask_token = self.mask_tokens[domain_name]
        masked_data.x[mask_indices] = mask_token.unsqueeze(0).expand(num_mask, -1)

        return masked_data, mask_indices, target_h0

    def create_augmented_views(self, data):
        """
        Create multiple augmented views of the graph for contrastive learning.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            List of augmented data objects
        """
        return [self.augmentor(data) for _ in range(AUGMENTATION_NUM_VIEWS)]

    def forward(self, data, domain_name: str):
        """
        Forward pass through the model.

        Args:
            data: PyTorch Geometric Data object
            domain_name: Name of the domain

        Returns:
            Final node embeddings (num_nodes, hidden_dim)
        """
        # Move data to the correct device
        data = data.to(self.device)

        # 1. Select domain-specific encoder
        encoder = self.input_encoders[domain_name]

        # 2. Encode domain-specific features to shared representation
        h_0 = encoder(data.x)

        # 3. Process with shared GNN backbone
        final_node_embeddings = self.gnn_backbone(h_0, data.edge_index)

        return final_node_embeddings

    def get_head(self, head_name: str):
        """
        Get a specific prediction head.

        Args:
            head_name: Name of the head to retrieve

        Returns:
            The requested prediction head module
        """
        return self.heads[head_name]

    def apply_gradient_reversal(self, embeddings, lambda_val):
        """
        Apply gradient reversal to embeddings for domain-adversarial training.

        Args:
            embeddings: Input embeddings tensor
            lambda_val: Scaling factor for gradient reversal

        Returns:
            Embeddings with gradient reversal applied
        """
        return self.grl(embeddings, lambda_val)
