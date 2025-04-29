import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Union, Tuple, Optional, Any


class Visualizer:
    """
    Class for creating interactive visualizations for multi-omics data and AI model results
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        self.color_palette = px.colors.qualitative.Plotly
        self.theme = 'plotly_white'
    
    def plot_gene_expression(self, data: pd.DataFrame, gene: str, 
                             group_by: str = None, log_scale: bool = True,
                             height: int = 500, width: int = 700) -> go.Figure:
        """
        Create a boxplot or violin plot of gene expression data
        
        Args:
            data: DataFrame with expression data
            gene: Gene to plot
            group_by: Column to group by (e.g., 'cancer_type', 'stage')
            log_scale: Whether to use log scale for expression values
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        if gene not in data.columns and gene not in data.index:
            st.error(f"Gene '{gene}' not found in the data")
            return None
        
        # Extract gene expression data
        if gene in data.columns:
            # Assume samples are rows, genes are columns
            expression_data = data[gene]
            if group_by and group_by in data.columns:
                plot_data = pd.DataFrame({
                    'Expression': expression_data,
                    group_by: data[group_by]
                })
            else:
                plot_data = pd.DataFrame({
                    'Expression': expression_data,
                    'Sample': data.index
                })
        else:
            # Assume genes are rows, samples are columns
            expression_data = data.loc[gene]
            plot_data = pd.DataFrame({
                'Expression': expression_data.values,
                'Sample': expression_data.index
            })
            if group_by and group_by in data.columns:
                # Need to merge with group information
                groups = data[group_by]
                plot_data[group_by] = [groups.get(sample, 'Unknown') for sample in plot_data['Sample']]
        
        # Apply log transformation if requested
        if log_scale:
            plot_data['Expression'] = np.log2(plot_data['Expression'] + 1)
            y_title = 'Expression (log2(TPM+1))'
        else:
            y_title = 'Expression (TPM)'
        
        # Create the plot
        if group_by and group_by in plot_data.columns:
            fig = px.violin(plot_data, x=group_by, y='Expression', 
                           color=group_by, box=True, points="all",
                           title=f'{gene} Expression by {group_by}',
                           height=height, width=width)
        else:
            fig = px.box(plot_data, y='Expression',
                        title=f'{gene} Expression',
                        height=height, width=width)
        
        fig.update_layout(
            yaxis_title=y_title,
            template=self.theme
        )
        
        return fig
    
    def plot_mutation_heatmap(self, data: pd.DataFrame, genes: List[str] = None, 
                             samples: List[str] = None, sample_col: str = 'sample_id',
                             gene_col: str = 'gene', height: int = 600, width: int = 900) -> go.Figure:
        """
        Create a mutation heatmap
        
        Args:
            data: DataFrame with mutation data
            genes: List of genes to include (if None, use top mutated genes)
            samples: List of samples to include (if None, use all samples)
            sample_col: Column containing sample identifiers
            gene_col: Column containing gene names
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        if data.empty:
            st.error("Empty mutation data provided")
            return None
        
        if sample_col not in data.columns:
            st.error(f"Sample column '{sample_col}' not found in mutation data")
            return None
            
        if gene_col not in data.columns:
            st.error(f"Gene column '{gene_col}' not found in mutation data")
            return None
        
        # Filter samples if provided
        if samples:
            data = data[data[sample_col].isin(samples)]
        
        # Get top mutated genes if not provided
        if not genes:
            gene_counts = data[gene_col].value_counts()
            genes = gene_counts.head(20).index.tolist()
        
        # Filter for selected genes
        data = data[data[gene_col].isin(genes)]
        
        # Create mutation matrix
        mutation_matrix = pd.crosstab(data[sample_col], data[gene_col])
        mutation_matrix = mutation_matrix.reindex(columns=genes)
        
        # Fill missing values with 0
        mutation_matrix = mutation_matrix.fillna(0)
        
        # Create heatmap
        fig = px.imshow(
            mutation_matrix,
            labels=dict(x="Gene", y="Sample", color="Mutation"),
            x=genes,
            title="Mutation Heatmap",
            color_continuous_scale="Reds",
            height=height,
            width=width
        )
        
        fig.update_layout(
            xaxis={'tickangle': 45},
            template=self.theme
        )
        
        return fig
    
    def plot_survival_curve(self, data: pd.DataFrame, time_col: str, event_col: str, 
                           group_col: str = None, height: int = 500, width: int = 700) -> go.Figure:
        """
        Create a Kaplan-Meier survival curve
        
        Args:
            data: DataFrame with survival data
            time_col: Column containing survival times
            event_col: Column containing event indicators (1 = event, 0 = censored)
            group_col: Column to group by (e.g., 'treatment', 'stage')
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        # Check for lifelines package
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test
        except ImportError:
            st.error("lifelines package not available. Cannot create survival curve.")
            return None
        
        if time_col not in data.columns:
            st.error(f"Time column '{time_col}' not found in data")
            return None
            
        if event_col not in data.columns:
            st.error(f"Event column '{event_col}' not found in data")
            return None
        
        # Create the plot
        fig = go.Figure()
        
        # Fit Kaplan-Meier model with or without groups
        if group_col and group_col in data.columns:
            groups = data[group_col].unique()
            
            # Calculate p-value if there are exactly 2 groups
            p_value = None
            if len(groups) == 2:
                group1 = data[data[group_col] == groups[0]]
                group2 = data[data[group_col] == groups[1]]
                
                results = logrank_test(
                    group1[time_col], group2[time_col],
                    group1[event_col], group2[event_col]
                )
                p_value = results.p_value
            
            for i, group in enumerate(groups):
                group_data = data[data[group_col] == group]
                
                kmf = KaplanMeierFitter()
                kmf.fit(group_data[time_col], group_data[event_col], label=str(group))
                
                # Get survival curve data
                survival_df = kmf.survival_function_
                
                # Add trace to plot
                fig.add_trace(go.Scatter(
                    x=survival_df.index,
                    y=survival_df.values.flatten(),
                    mode='lines',
                    name=str(group),
                    line=dict(color=self.color_palette[i % len(self.color_palette)]),
                    hovertemplate=(
                        "Time: %{x:.1f}<br>"
                        "Survival Probability: %{y:.3f}<br>"
                        "<extra></extra>"
                    )
                ))
            
            # Add p-value annotation if available
            if p_value is not None:
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.5, y=0.05,
                    text=f"Log-rank p-value: {p_value:.3f}",
                    showarrow=False,
                    font=dict(size=12)
                )
                
            title = f"Kaplan-Meier Survival Curve by {group_col}"
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(data[time_col], data[event_col])
            
            # Get survival curve data
            survival_df = kmf.survival_function_
            
            # Add trace to plot
            fig.add_trace(go.Scatter(
                x=survival_df.index,
                y=survival_df.values.flatten(),
                mode='lines',
                name='Overall Survival',
                line=dict(color=self.color_palette[0]),
                hovertemplate=(
                    "Time: %{x:.1f}<br>"
                    "Survival Probability: %{y:.3f}<br>"
                    "<extra></extra>"
                )
            ))
            
            title = "Kaplan-Meier Survival Curve"
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Survival Probability",
            yaxis=dict(range=[0, 1.05]),
            height=height,
            width=width,
            template=self.theme,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_volcano_plot(self, data: pd.DataFrame, logfc_col: str, pvalue_col: str,
                         gene_col: str = None, highlight_genes: List[str] = None,
                         logfc_threshold: float = 1.0, pvalue_threshold: float = 0.05,
                         height: int = 600, width: int = 800) -> go.Figure:
        """
        Create a volcano plot for differential expression analysis
        
        Args:
            data: DataFrame with differential expression results
            logfc_col: Column containing log fold change values
            pvalue_col: Column containing p-values
            gene_col: Column containing gene names
            highlight_genes: List of genes to highlight
            logfc_threshold: Threshold for log fold change significance
            pvalue_threshold: Threshold for p-value significance
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        if logfc_col not in data.columns:
            st.error(f"Log fold change column '{logfc_col}' not found in data")
            return None
            
        if pvalue_col not in data.columns:
            st.error(f"P-value column '{pvalue_col}' not found in data")
            return None
        
        # Compute -log10(p-value)
        data['neg_log_pvalue'] = -np.log10(data[pvalue_col])
        
        # Determine significance
        data['significant'] = (
            (data[logfc_col].abs() >= logfc_threshold) & 
            (data[pvalue_col] < pvalue_threshold)
        )
        
        # Determine direction of change
        data['regulation'] = 'Not Significant'
        data.loc[(data[logfc_col] >= logfc_threshold) & (data[pvalue_col] < pvalue_threshold), 'regulation'] = 'Up-regulated'
        data.loc[(data[logfc_col] <= -logfc_threshold) & (data[pvalue_col] < pvalue_threshold), 'regulation'] = 'Down-regulated'
        
        # Create color mapping
        color_map = {
            'Up-regulated': 'red',
            'Down-regulated': 'blue',
            'Not Significant': 'gray'
        }
        
        # Prepare hover text
        hover_data = {
            logfc_col: ':.3f',
            pvalue_col: ':.2e',
            'neg_log_pvalue': ':.2f'
        }
        
        if gene_col and gene_col in data.columns:
            hover_data[gene_col] = True
        
        # Create volcano plot
        fig = px.scatter(
            data,
            x=logfc_col,
            y='neg_log_pvalue',
            color='regulation',
            color_discrete_map=color_map,
            hover_name=gene_col if gene_col and gene_col in data.columns else None,
            hover_data=hover_data,
            title="Volcano Plot of Differential Expression",
            height=height,
            width=width
        )
        
        # Add threshold lines
        fig.add_hline(y=-np.log10(pvalue_threshold), line_dash="dash", line_color="gray")
        fig.add_vline(x=logfc_threshold, line_dash="dash", line_color="gray")
        fig.add_vline(x=-logfc_threshold, line_dash="dash", line_color="gray")
        
        # Highlight specific genes if provided
        if gene_col and highlight_genes and gene_col in data.columns:
            highlight_data = data[data[gene_col].isin(highlight_genes)]
            
            if not highlight_data.empty:
                fig.add_trace(go.Scatter(
                    x=highlight_data[logfc_col],
                    y=highlight_data['neg_log_pvalue'],
                    mode='markers+text',
                    marker=dict(
                        size=10,
                        color='black',
                        line=dict(width=2, color='black')
                    ),
                    text=highlight_data[gene_col],
                    textposition="top center",
                    name="Highlighted Genes",
                    hoverinfo='text',
                    hovertext=[
                        f"{gene}<br>{logfc_col}: {fc:.3f}<br>p-value: {p:.2e}"
                        for gene, fc, p in zip(
                            highlight_data[gene_col],
                            highlight_data[logfc_col],
                            highlight_data[pvalue_col]
                        )
                    ]
                ))
        
        # Update layout
        fig.update_layout(
            xaxis_title="Log2 Fold Change",
            yaxis_title="-Log10 P-value",
            template=self.theme,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                               top_n: int = 20, height: int = 500, width: int = 700) -> go.Figure:
        """
        Create a bar plot of feature importance from a trained model
        
        Args:
            feature_importance: Dictionary with feature names and importance values
            top_n: Number of top features to show
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        if not feature_importance:
            st.error("Empty feature importance dictionary provided")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        })
        
        # Sort by importance and get top N features
        df = df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create bar plot
        fig = px.bar(
            df,
            y='Feature',
            x='Importance',
            orientation='h',
            title=f"Top {top_n} Feature Importance",
            height=height,
            width=width,
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        # Update layout
        fig.update_layout(
            yaxis=dict(title=''),
            xaxis=dict(title='Importance'),
            template=self.theme,
            yaxis_categoryorder='total ascending'
        )
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: List[str] = None, normalize: bool = False,
                             height: int = 500, width: int = 500) -> go.Figure:
        """
        Create a confusion matrix for classification results
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of the classes
            normalize: Whether to normalize the confusion matrix
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        from sklearn.metrics import confusion_matrix
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.round(cm, 2)
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Get class names if not provided
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        
        # Create heatmap
        fig = px.imshow(
            cm,
            x=class_names,
            y=class_names,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="True", color="Count"),
            title="Confusion Matrix",
            height=height,
            width=width
        )
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{cm[i, j]:{fmt}}",
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > 0.5*cm.max() else "black")
                )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            template=self.theme
        )
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray, 
                      class_names: List[str] = None, multi_class: bool = False,
                      height: int = 500, width: int = 700) -> go.Figure:
        """
        Create a ROC curve for classification results
        
        Args:
            y_true: True labels
            y_score: Predicted probabilities
            class_names: Names of the classes
            multi_class: Whether to plot ROC curves for multi-class problems
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        from sklearn.metrics import roc_curve, auc, roc_auc_score
        
        # Prepare figure
        fig = go.Figure()
        
        if multi_class:
            # For multi-class problems
            from sklearn.preprocessing import label_binarize
            
            # Get unique classes
            classes = np.unique(y_true)
            n_classes = len(classes)
            
            # Get class names if not provided
            if class_names is None:
                class_names = [str(i) for i in classes]
            
            # Binarize the output
            y_true_bin = label_binarize(y_true, classes=classes)
            
            # Compute ROC curve and ROC area for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                
                # Add trace for each class
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{class_names[i]} (AUC = {roc_auc:.3f})',
                    line=dict(color=self.color_palette[i % len(self.color_palette)])
                ))
        else:
            # For binary classification
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Add ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC curve (AUC = {roc_auc:.3f})',
                line=dict(color=self.color_palette[0])
            ))
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=width,
            height=height,
            template=self.theme,
            legend=dict(x=0.7, y=0.05)
        )
        
        return fig
    
    def plot_learning_curves(self, history: Dict[str, List[float]], 
                            height: int = 500, width: int = 700) -> go.Figure:
        """
        Plot learning curves from training history
        
        Args:
            history: Dictionary with training metrics
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        # Check if history is provided
        if not history:
            st.error("Empty training history provided")
            return None
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces for each metric
        for i, (metric, values) in enumerate(history.items()):
            # Skip validation metrics for now
            if metric.startswith('val_'):
                continue
            
            # Get training and validation values
            train_values = values
            val_metric = f'val_{metric}'
            val_values = history.get(val_metric, [])
            
            # Use secondary y-axis for metrics that aren't loss
            use_secondary_y = metric != 'loss' and not metric.endswith('loss')
            
            # Add training metric
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(train_values) + 1)),
                    y=train_values,
                    mode='lines',
                    name=f'Training {metric}',
                    line=dict(color=self.color_palette[i * 2 % len(self.color_palette)])
                ),
                secondary_y=use_secondary_y
            )
            
            # Add validation metric if available
            if val_values:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(val_values) + 1)),
                        y=val_values,
                        mode='lines',
                        name=f'Validation {metric}',
                        line=dict(color=self.color_palette[(i * 2 + 1) % len(self.color_palette)], dash='dash')
                    ),
                    secondary_y=use_secondary_y
                )
        
        # Update layout
        fig.update_layout(
            title="Learning Curves",
            xaxis_title="Epoch",
            template=self.theme,
            height=height,
            width=width,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Loss", secondary_y=False)
        fig.update_yaxes(title_text="Metric Value", secondary_y=True)
        
        return fig
    
    def plot_pca(self, data: pd.DataFrame, n_components: int = 2, 
                group_col: str = None, label_col: str = None,
                height: int = 600, width: int = 800) -> go.Figure:
        """
        Perform PCA and plot the results
        
        Args:
            data: DataFrame with features
            n_components: Number of PCA components to compute
            group_col: Column to use for coloring points
            label_col: Column to use for point labels
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Select only numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        X = data[numeric_cols]
        
        # Standardize the data
        X_scaled = StandardScaler().fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_scaled)
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(
            data=principal_components,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Add group and label columns if provided
        if group_col and group_col in data.columns:
            pca_df[group_col] = data[group_col].values
        
        if label_col and label_col in data.columns:
            pca_df[label_col] = data[label_col].values
        
        # Create the plot
        if n_components >= 3:
            # 3D PCA plot
            fig = px.scatter_3d(
                pca_df,
                x='PC1',
                y='PC2',
                z='PC3',
                color=group_col if group_col and group_col in data.columns else None,
                hover_name=label_col if label_col and label_col in data.columns else None,
                title="3D PCA Plot",
                height=height,
                width=width
            )
        else:
            # 2D PCA plot
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color=group_col if group_col and group_col in data.columns else None,
                hover_name=label_col if label_col and label_col in data.columns else None,
                title="PCA Plot",
                height=height,
                width=width
            )
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        
        # Update axis titles with explained variance
        if n_components >= 3:
            fig.update_layout(
                scene=dict(
                    xaxis_title=f"PC1 ({explained_variance[0]:.2%})",
                    yaxis_title=f"PC2 ({explained_variance[1]:.2%})",
                    zaxis_title=f"PC3 ({explained_variance[2]:.2%})"
                )
            )
        else:
            fig.update_layout(
                xaxis_title=f"PC1 ({explained_variance[0]:.2%})",
                yaxis_title=f"PC2 ({explained_variance[1]:.2%})"
            )
        
        # Update layout
        fig.update_layout(
            template=self.theme,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_clustermap(self, data: pd.DataFrame, method: str = 'ward', 
                       metric: str = 'euclidean', z_score: bool = True,
                       height: int = 800, width: int = 1000) -> go.Figure:
        """
        Create a clustered heatmap (clustermap)
        
        Args:
            data: DataFrame with features as columns and samples as rows
            method: Linkage method for hierarchical clustering
            metric: Distance metric for hierarchical clustering
            z_score: Whether to standardize the data
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import pdist
        
        # Select only numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        X = data[numeric_cols]
        
        # Standardize the data if requested
        if z_score:
            X = (X - X.mean()) / X.std()
        
        # Compute linkage for samples (rows)
        row_linkage = linkage(pdist(X, metric=metric), method=method)
        
        # Compute linkage for features (columns)
        col_linkage = linkage(pdist(X.T, metric=metric), method=method)
        
        # Get row and column dendrograms
        row_dendrogram = dendrogram(row_linkage, no_plot=True)
        col_dendrogram = dendrogram(col_linkage, no_plot=True)
        
        # Reorder data based on clustering
        row_order = row_dendrogram['leaves']
        col_order = col_dendrogram['leaves']
        
        reordered_data = X.iloc[row_order, :].iloc[:, col_order]
        
        # Create the heatmap
        fig = px.imshow(
            reordered_data,
            title="Clustered Heatmap",
            height=height,
            width=width,
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        
        # Update layout
        fig.update_layout(
            template=self.theme,
            xaxis=dict(tickangle=45)
        )
        
        return fig
