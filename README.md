# ResearchLens - GNNs for Node Classification and Influence Prediction in Research Citation Networks

## Team members
1. Nishanth Chockalingam Veerapandian
2. Sai Nithish Mahadeva Rao

## Abstract
This project implements and analyzes an enhanced Graph Convolutional Network (GCN) for citation network analysis. The system is designed to perform node classification and influence analysis on academic citation networks, specifically tested on the Cora and ogbn-arxiv datasets. The implementation includes advanced features such as batch normalization, multiple hidden layers, and comprehensive influence analysis tools. The project demonstrates significant improvements in node classification accuracy and provides novel insights into citation network dynamics through various centrality measures.

## Overview

### Problem Statement
Citation networks represent a complex web of academic relationships where papers (nodes) are connected through citations (edges). Understanding and analyzing these networks is crucial for:
- Classifying papers into research domains
- Identifying influential papers and research trends
- Understanding how information and influence propagate through academic communities

### Significance
This problem is particularly relevant in today's academic landscape for several reasons:
- The exponential growth of academic publications makes automated analysis essential
- Understanding influence patterns can help researchers identify important works in their field
- Network analysis can reveal emerging research trends and cross-disciplinary connections
- The methods developed can be applied to other network-based problems in social networks, recommendation systems, and knowledge graphs

### Proposed Approach
The project implements a multi-layered solution:
1. An Enhanced GCN with batch normalization and flexible architecture
2. A comprehensive training framework with early stopping and learning rate scheduling
3. A node influence analysis system combining multiple centrality measures
4. Visualization tools for network embeddings and influence propagation

### Rationale and Prior Work
The approach builds upon traditional GCN architectures but also incorporates other improvements:
- Addition of batch normalization layers to stabilize training
- Flexible hidden layer architecture to handle different network sizes
- Integration of multiple centrality measures for more robust influence analysis
- Implementation of early stopping and learning rate scheduling for better convergence

### Key Components and Limitations
Key components:
- EnhancedGCN class with configurable architecture
- CitationNetworkTrainer for model training
- NodeInfluenceAnalyzer for comprehensive network analysis

Limitations:
- Scalability challenges with very large networks
- Memory constraints when computing full similarity matrices
- Computational intensity of certain centrality measures
- Dependency on quality of initial citation network data

## Experiment Setup

### Dataset Description
The project utilizes two primary datasets:

1. Cora Dataset:
- Number of nodes: 2,708
- Number of features: 1,433
- Number of classes: 7
- Number of edges: 5,278

2. ogbn-arxiv Dataset:
- Number of nodes: 169,343
- Number of features: 128
- Number of classes: 40
- Number of edges: 583,121

### Implementation Details
Model Parameters:
- Cora: Hidden dimensions [128, 64], dropout 0.5
- ogbn-arxiv: Hidden dimensions [256, 128], dropout 0.5

Training Environment:
- PyTorch Geometric framework
- GPU acceleration when available
- Early stopping patience: 20 epochs
- Maximum epochs: 200
- Adam optimizer with initial learning rate 0.01

### Model Architecture
The Enhanced GCN architecture consists of:
1. Input layer: Maps node features to first hidden dimension
2. Multiple hidden layers with:
   - Graph convolution operations
   - Batch normalization
   - ReLU activation
   - Dropout regularization
3. Output layer with log-softmax activation

## Experiment Results

### Training Dynamics and Performance

For the Cora dataset:
```
Dataset Statistics:
- Nodes    : 2,708
- Features : 1,433
- Classes  : 7
- Edges    : 5,278

Final Performance:
- Train accuracy      : 0.9929 (99.29%)
- Validation accuracy : 0.7760 (77.60%)
- Test accuracy       : 0.7660 (76.60%)
```

The training process showed several key characteristics:
- Rapid initial convergence (within first 5 epochs)
- Early stopping triggered at epoch 20
- Clear signs of overfitting with training accuracy reaching nearly 100%

For the ogbn-arxiv dataset:
```
Dataset Statistics:
- Nodes    : 169,343
- Features : 128
- Classes  : 40
- Edges    : 583,121

Final Performance:
- Train accuracy      : 0.6430 (64.30%)
- Validation accuracy : 0.6017 (60.17%)
- Test accuracy       : 0.5427 (54.27%)
```

Training progression showed:
- Slower convergence compared to Cora
- More stable learning curve
- Early stopping triggered at epoch 149
- Better generalization characteristics

### Visualization Analysis

#### For Cora
#### Training Curves
![Cora Training Curves](./Results/Cora%20Training%20Loss.png)
*Figure 1: Cora dataset training curves showing loss and accuracy over time. Left: Training loss progression. Right: Training and validation accuracy comparison.*
The training visualizations reveal:
- Cora's loss curve shows rapid descent in the first 5 epochs, followed by stabilization
- Training accuracy quickly reaches near-perfect performance while validation plateaus around 77%
- Clear indication of overfitting despite regularization measures

#### Node Embeddings
![Cora Node Embeddings](./Results/Cora%20-%20t-SNE.png)
*Figure 2: t-SNE visualization of Cora node embeddings. Different colors represent different paper categories, showing clear cluster formation and separation between research domains.*
The t-SNE visualization of node embeddings shows:
- Clear cluster formation corresponding to different paper categories
- Well-separated communities in the embedding space
- Some overlap between related research areas

#### Network Influence Analysis

Centrality Correlation Analysis:
![Cora Centrality Correlations](./Results/Cora%20Centrality%20Measure.png)
*Figure 3: Correlation matrix of different centrality measures for the Cora dataset. The heatmap shows relationships between degree centrality, eigenvector centrality, betweenness centrality, and PageRank.*
- Strong correlation (0.98) between degree centrality and PageRank
- Moderate correlation (0.52) between eigenvector and degree centrality
- High correlation (0.87) between betweenness and degree centrality
- Indicates that different influence measures capture distinct aspects of node importance

Node Influence Distribution:
![Cora Influence Distribution](./Results/Cora%20Node%20Influence.png)
*Figure 4: Distribution of node influence scores in the Cora dataset, showing a highly skewed distribution with few highly influential nodes.*
- Highly skewed distribution of influence scores
- Small number of highly influential nodes
- Large majority of nodes with relatively low influence
- Top influential nodes identified:
  ```
  Cora Top-5:
  1. Node 1358 : Score = 1.0000
  2. Node 306  : Score = 0.2205
  3. Node 1701 : Score = 0.1438
  4. Node 1986 : Score = 0.0952
  5. Node 1623 : Score = 0.0766
  ```

Influence Propagation Analysis:
![Cora Influence Propagation](./Results/Cora%20Influence%20Propagation.png)
*Figure 5: Influence propagation patterns for the top 5 most influential nodes in the Cora dataset, showing how their influence spreads through the network over multiple steps.*
- Different propagation patterns for various influential nodes
- Some nodes show rapid initial influence spread
- Others demonstrate more gradual but sustained influence
- Maximum propagation depth reached within 2-3 steps

Analysis: The significant gap between training accuracy (99.29%) and test accuracy (76.60%) indicates clear overfitting on the Cora dataset. This suggests that despite our regularization efforts (dropout and batch normalization), the model is memorizing training data rather than learning generalizable patterns. This overfitting might be due to:
   - The relatively small size of the Cora dataset
   - Potentially too complex model architecture for this dataset
   - Need for stronger regularization techniques

#### For obgn-arxiv

#### Training Curves
![ogbn-arxiv Training Curves](./Results/Arxiv%20Training%20Loss.png)
*Figure 1: ogbn-arxiv dataset training curves showing loss and accuracy over time. Left: Training loss progression from ~4.0 to ~1.4. Right: Training and validation accuracy comparison showing convergence around 60%.*

The training process demonstrated several key characteristics:
- Initial rapid loss descent from 4.0 to approximately 2.0 in first 10 epochs
- Gradual loss improvement afterwards, stabilizing around 1.4
- Training accuracy showed steady increase to approximately 64%
- Validation accuracy closely tracked training accuracy, reaching about 60%
- Early stopping triggered at epoch 149, indicating good convergence
- Notably smaller gap between training and validation accuracy compared to Cora

#### Node Embeddings
![ogbn-arxiv Node Embeddings](./Results/Arxiv%20-%20t-SNE.png)
*Figure 2: t-SNE visualization of ogbn-arxiv node embeddings. Different colors represent different paper categories (40 classes), showing clear cluster formation and relationships between research domains.*

The t-SNE visualization of node embeddings shows:
- More complex clustering structure compared to Cora dataset
- Multiple interconnected research communities
- Clear separation between major research areas
- Significant overlap between related fields, showing interdisciplinary papers
- Higher dimensional complexity due to 40 distinct classes

#### Network Influence Analysis

Centrality Correlation Analysis:
![ogbn-arxiv Centrality Correlations](./Results/Arxiv%20Centrality%20Measure.png)
*Figure 3: Correlation matrix of different centrality measures for the ogbn-arxiv dataset. The heatmap shows relationships between degree centrality, eigenvector centrality, betweenness centrality, and PageRank.*

Key correlations:
- Very strong correlation (0.97) between degree centrality and PageRank
- Strong correlation (0.82) between betweenness and PageRank
- Moderate correlation (0.76) between eigenvector and degree centrality
- Lower correlation (0.51) between eigenvector and betweenness centrality

Node Influence Distribution:
![ogbn-arxiv Influence Distribution](./Results/Arxiv%20Node%20Influence.png)
*Figure 4: Distribution of node influence scores in the ogbn-arxiv dataset, showing a highly skewed distribution with few highly influential nodes.*

Distribution characteristics:
- Extremely skewed power-law-like distribution
- Vast majority of nodes have very low influence scores
- Small number of highly influential nodes
- Top influential nodes identified:
  ```
  ogbn-arxiv Top-5:
  1. Node 1353  : Score = 1.0000
  2. Node 67166 : Score = 0.8919
  3. Node 25208 : Score = 0.5622
  4. Node 69794 : Score = 0.2078
  5. Node 93649 : Score = 0.1837
  ```

Influence Propagation Analysis:
![ogbn-arxiv Influence Propagation](./Results/Arxiv%20Influence%20Propagation.png)
*Figure 5: Influence propagation patterns for the top 5 most influential nodes in the ogbn-arxiv dataset, showing how their influence spreads through the network over multiple steps.*

The influence propagation analysis reveals:
- More gradual influence spread patterns compared to Cora
- Longer propagation chains due to larger network size
- Complex multi-hop dependency structures
- Higher variance in propagation patterns between different influential nodes


For ogbn-arxiv dataset:
   - Train accuracy      : 0.6430 (64.30%)
   - Validation accuracy : 0.6017 (60.17%)
   - Test accuracy       : 0.5427 (54.27%)
   
Analysis: The ogbn-arxiv results show better generalization characteristics with smaller gaps between train and test performance, though the overall accuracy is lower. This suggests:
   - More balanced learning with less overfitting
   - Higher task difficulty due to larger number of classes and dataset size
   - Potential need for model architecture optimization

## Inference

The ogbn-arxiv results demonstrate several key insights:

1. Model Behavior:
   - Lower absolute accuracy (54.27% vs 76.60%) reflects increased task complexity
   - Better generalization with smaller train-validation gap
   - More stable learning progression
   - Longer training requirement before convergence

2. Network Structure:
   - Complex topology with clear hierarchical patterns
   - Strong correlations between different centrality measures
   - Highly skewed influence distribution suggesting scale-free properties
   - Sophisticated inter-class relationships and interdisciplinary connections

3. Scalability:
   - Successfully handles significantly larger network (169,343 nodes vs 2,708)
   - Maintains reasonable performance despite 40 classes
   - Effective capture of complex citation patterns


## Discussion

The experimental results provide several key insights into both the model's performance and the structure of academic citation networks:

### Model Performance Analysis
1. Dataset-Specific Behavior:
   - Cora shows classic signs of overfitting despite regularization
   - ogbn-arxiv demonstrates better generalization but lower overall performance
   - Early stopping proved effective for both datasets, preventing further divergence

2. Architecture Implications:
   - Current architecture may be oversized for Cora's relatively simple structure
   - May need additional capacity or architectural modifications for ogbn-arxiv
   - Batch normalization and dropout showed mixed effectiveness

### Network Structure Insights
1. Node Embedding Quality:
   - Clear cluster formation in t-SNE visualizations
   - Effective capture of research domain relationships
   - Meaningful separation of different paper categories

2. Influence Dynamics:
   - Highly skewed influence distribution suggests a scale-free network structure
   - Strong correlation between different centrality measures indicates robust identification of influential nodes
   - Influence propagation patterns reveal multi-hop dependency structures

### Limitations and Future Directions
1. Model Improvements:
   - Need for better regularization strategies for small datasets
   - Potential for adaptive architecture sizing
   - Opportunity for meta-learning approaches

2. Analysis Extensions:
   - Temporal analysis of influence propagation
   - Cross-domain influence patterns
   - Community structure impact on information flow
The experimental results reveal several important insights about the GCN architecture:

1. Model Behavior:
   - Clear overfitting on the Cora dataset despite regularization techniques
   - More stable but lower performance on the larger ogbn-arxiv dataset
   - Batch normalization and dropout showing mixed effectiveness

2. Dataset-Specific Challenges:
   - Cora: High training accuracy (99.29%) but limited generalization
   - ogbn-arxiv: More consistent performance across splits but lower overall accuracy

3. Architecture Considerations:
   - Current architecture may be oversized for Cora
   - May need additional capacity for ogbn-arxiv
   - Trade-off between model expressiveness and generalization

Future improvements could include:
- Implementation of attention mechanisms
- Integration of edge features
- More sophisticated influence propagation models
- Distributed computing for larger networks

## Conclusion
The project successfully developed and implemented an enhanced GCN framework for citation network analysis. The system demonstrates robust performance in both node classification and influence analysis tasks, while providing valuable insights into network dynamics. The modular architecture and comprehensive analysis tools make it a valuable resource for citation network research.

## References
1. Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks
2. ogbn-arxiv dataset: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
3. PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
4. NetworkX: https://networkx.org/
5. Page, Lawrence et al. (1999). The PageRank Citation Ranking: Bringing Order to the Web