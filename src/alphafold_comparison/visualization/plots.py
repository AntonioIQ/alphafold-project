"""
Funciones de visualización reutilizables para el proyecto.

Todas las funciones generan figuras matplotlib que pueden
guardarse o mostrarse en notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from alphafold_comparison.config import Config

# Estilo global
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_rmsd_distribution(results_df: pd.DataFrame, output_path=None, title=None):
    """Histograma de distribución de RMSD global."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(results_df["global_rmsd"], bins=100, edgecolor="black", alpha=0.7, color="#2196F3")
    ax.axvline(results_df["global_rmsd"].median(), color="red", linestyle="--",
               label=f"Mediana: {results_df['global_rmsd'].median():.2f} A")
    ax.axvline(2.0, color="green", linestyle="--", label="Umbral 2 A")

    ax.set_xlabel("RMSD Global (A)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title or "Distribucion de RMSD: PDB vs AlphaFold")
    ax.legend()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_rmsd_by_size(results_df: pd.DataFrame, output_path=None):
    """Boxplot de RMSD por categoría de tamaño de proteína."""
    size_bins = [0, 100, 200, 300, 400, 500, 1000, float("inf")]
    size_labels = ["<100", "100-200", "200-300", "300-400", "400-500", "500-1000", ">1000"]

    df = results_df.copy()
    df["size_category"] = pd.cut(df["protein_length"], bins=size_bins, labels=size_labels)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="size_category", y="global_rmsd", ax=ax,
                showfliers=False, palette="viridis")
    ax.axhline(2.0, color="red", linestyle="--", alpha=0.5, label="Umbral 2 A")

    ax.set_xlabel("Tamano de proteina (residuos)")
    ax.set_ylabel("RMSD Global (A)")
    ax.set_title("RMSD por tamano de proteina")
    ax.legend()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_sweet_spot_comparison(results_df: pd.DataFrame, output_path=None):
    """Comparación sweet spot (250-300) vs resto."""
    sweet = results_df[
        (results_df["protein_length"] >= 250) & (results_df["protein_length"] <= 300)
    ]
    rest = results_df[
        ~((results_df["protein_length"] >= 250) & (results_df["protein_length"] <= 300))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogramas
    axes[0].hist(rest["global_rmsd"], bins=50, alpha=0.5, label="Resto", color="#FF5722")
    axes[0].hist(sweet["global_rmsd"], bins=50, alpha=0.7, label="Sweet Spot (250-300)",
                 color="#4CAF50")
    axes[0].set_xlabel("RMSD Global (A)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribucion de RMSD")
    axes[0].legend()

    # Barras comparativas
    categories = ["Media", "Mediana", "% < 2A"]
    sweet_vals = [
        sweet["global_rmsd"].mean(),
        sweet["global_rmsd"].median(),
        (sweet["global_rmsd"] < 2).mean() * 100,
    ]
    rest_vals = [
        rest["global_rmsd"].mean(),
        rest["global_rmsd"].median(),
        (rest["global_rmsd"] < 2).mean() * 100,
    ]

    x = np.arange(len(categories))
    width = 0.35
    axes[1].bar(x - width / 2, rest_vals, width, label="Resto", color="#FF5722", alpha=0.7)
    axes[1].bar(x + width / 2, sweet_vals, width, label="Sweet Spot", color="#4CAF50", alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_title("Comparacion Sweet Spot vs Resto")
    axes[1].legend()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_rmsd_per_residue(rmsd_values, residue_ids=None, output_path=None, title=None):
    """Gráfico de RMSD por residuo individual."""
    fig, ax = plt.subplots(figsize=(14, 5))

    x = residue_ids if residue_ids is not None else range(len(rmsd_values))
    colors = ["#4CAF50" if v < 2 else "#FF9800" if v < 5 else "#F44336" for v in rmsd_values]

    ax.bar(x, rmsd_values, color=colors, width=1.0, edgecolor="none")
    ax.axhline(2.0, color="black", linestyle="--", alpha=0.5, label="2 A")
    ax.set_xlabel("Posicion del residuo")
    ax.set_ylabel("RMSD (A)")
    ax.set_title(title or "RMSD por residuo")
    ax.legend()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_validation_summary(validation_df: pd.DataFrame, output_path=None):
    """Resumen visual de resultados de validación."""
    status_counts = validation_df["validation_status"].value_counts()

    colors = {
        "EXCELLENT": "#4CAF50",
        "VALIDATED": "#8BC34A",
        "REVIEW": "#FFC107",
        "SUSPICIOUS": "#FF9800",
        "REJECTED": "#F44336",
        "FAILED": "#9E9E9E",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    pie_colors = [colors.get(s, "#9E9E9E") for s in status_counts.index]
    axes[0].pie(status_counts.values, labels=status_counts.index, colors=pie_colors,
                autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Distribucion de estados de validacion")

    # Bar chart
    axes[1].barh(status_counts.index, status_counts.values,
                 color=[colors.get(s, "#9E9E9E") for s in status_counts.index])
    axes[1].set_xlabel("Cantidad de pares")
    axes[1].set_title("Conteo por estado")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
