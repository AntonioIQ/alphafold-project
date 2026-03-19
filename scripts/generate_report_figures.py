#!/usr/bin/env python3
"""
Generador de figuras para el reporte AlphaFold — Maestría en Ciencia de Datos, ITAM
Análisis jerárquico: longitud → aminoácidos → grupos funcionales → átomos (CHONSP)
+ Inferencia bayesiana (Beta-Binomial + HDI 95%)
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
from scipy.stats import beta as beta_dist
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings('ignore', category=PDBConstructionWarning)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ─── Rutas ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / 'data'
FIGURES    = ROOT / 'reports' / 'latex' / 'figures'
CACHE_DIR  = ROOT / 'data' / 'processed' / 'residue_cache'
QUALITY_CSV = DATA_DIR / 'processed' / 'quality_structures_index.csv'
SIFTS_CSV   = DATA_DIR / 'raw' / 'mappings' / 'pdb_chain_uniprot.csv'
PDB_DIR     = DATA_DIR / 'raw' / 'pdb_files'
AF_DIR      = DATA_DIR / 'raw' / 'alphafold_models'

FIGURES.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Estilo visual ITAM / ciencia de datos ───────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
})
PALETTE_CAT = sns.color_palette('Set2', 8)
PALETTE_SEQ = 'cividis'

# ─── Clasificaciones ─────────────────────────────────────────────────────────
AA_THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}
FUNCTIONAL_GROUPS = {
    'Alifático':  list('GAVLIPMS'),
    'Aromático':  list('FWY'),
    'Polar':      list('STCNQ'),
    'Básico':     list('KRH'),
    'Ácido':      list('DE'),
}
AA_TO_FG = {aa: fg for fg, aas in FUNCTIONAL_GROUPS.items() for aa in aas}

LENGTH_BINS   = [0, 100, 200, 300, 400, 500, 1000, 10000]
LENGTH_LABELS = ['<100','100-200','200-300','300-400','400-500','500-1000','>1000']

QUALITY_THRESHOLDS = {
    'Excelente': (0,    1.0),
    'Buena':     (1.0,  2.0),
    'Moderada':  (2.0,  5.0),
    'Pobre':     (5.0,  np.inf),
}
QUALITY_ORDER = ['Excelente','Buena','Moderada','Pobre']

def rmsd_to_quality(rmsd):
    if   rmsd < 1.0: return 'Excelente'
    elif rmsd < 2.0: return 'Buena'
    elif rmsd < 5.0: return 'Moderada'
    else:            return 'Pobre'

# ─── SECCIÓN 0: Cargar datos base ────────────────────────────────────────────

def load_base_data():
    df = pd.read_csv(QUALITY_CSV)
    df['length_cat'] = pd.cut(
        df['matched_residues'],
        bins=LENGTH_BINS, labels=LENGTH_LABELS, right=False
    ).astype(str)
    df['quality'] = df['rmsd'].apply(rmsd_to_quality)
    return df

# ─── SECCIÓN 1: Por longitud de cadena proteica ──────────────────────────────

def fig_01_length(df):
    log.info("Generando figuras 1: análisis por longitud de cadena")

    # 1a. Violin plot RMSD por categoría de longitud
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_cats = [c for c in LENGTH_LABELS if c in df['length_cat'].values]
    data_by_cat = [df[df['length_cat'] == c]['rmsd'].clip(upper=20).values for c in valid_cats]
    counts = [len(d) for d in data_by_cat]

    parts = ax.violinplot(data_by_cat, positions=range(len(valid_cats)),
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE_CAT[i % len(PALETTE_CAT)])
        pc.set_alpha(0.75)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    ax.set_xticks(range(len(valid_cats)))
    ax.set_xticklabels([f'{c}\n(n={counts[i]:,})' for i, c in enumerate(valid_cats)], fontsize=9)
    ax.set_xlabel('Longitud de cadena proteica (residuos)')
    ax.set_ylabel('RMSD (Å) [truncado en 20 Å]')
    ax.set_title('Distribución de RMSD por longitud de cadena\n(10,432 proteínas únicas, mapeo SIFTS correcto)')
    ax.axhline(1.0, color='steelblue', linestyle='--', alpha=0.6, label='Umbral excelente (1 Å)')
    ax.axhline(2.0, color='orange',    linestyle='--', alpha=0.6, label='Umbral buena (2 Å)')
    ax.legend(loc='upper left')

    fig.savefig(FIGURES / 'fig_01_length_violin.png')
    plt.close(fig)
    log.info("  → fig_01_length_violin.png")

    # 1b. Hexbin RMSD vs longitud con LOWESS
    fig, ax = plt.subplots(figsize=(10, 6))
    x = df['matched_residues'].clip(upper=1200)
    y = df['rmsd'].clip(upper=20)
    hb = ax.hexbin(x, y, gridsize=50, cmap=PALETTE_SEQ, mincnt=1, bins='log')
    plt.colorbar(hb, ax=ax, label='log10(n proteínas)')

    # LOWESS smoother
    from statsmodels.nonparametric.smoothers_lowess import lowess
    df_sorted = df[['matched_residues','rmsd']].sort_values('matched_residues')
    df_clip = df_sorted[df_sorted['matched_residues'] <= 1200]
    smooth = lowess(df_clip['rmsd'].clip(upper=20), df_clip['matched_residues'], frac=0.1)
    ax.plot(smooth[:,0], smooth[:,1], color='tomato', linewidth=2.5, label='Tendencia LOWESS')

    ax.axhline(1.0, color='steelblue', linestyle='--', alpha=0.7, label='1 Å')
    ax.axhline(2.0, color='orange',    linestyle='--', alpha=0.7, label='2 Å')
    ax.set_xlabel('Longitud proteica (residuos)')
    ax.set_ylabel('RMSD (Å) [truncado en 20 Å]')
    ax.set_title('Densidad de predicciones: RMSD vs longitud proteica\nColor: escala logarítmica del número de proteínas')
    ax.legend()

    fig.savefig(FIGURES / 'fig_01_length_scatter.png')
    plt.close(fig)
    log.info("  → fig_01_length_scatter.png")

    # 1c. Matriz de confusión calidad × longitud (ordenada de mayor→menor RMSD)
    # "confusión" aquí = quién predice qué calidad, cruzado con la categoría de longitud
    confusion = pd.crosstab(
        df['quality'],
        df['length_cat'],
        normalize='index'
    ).loc[QUALITY_ORDER, valid_cats] * 100

    confusion_abs = pd.crosstab(df['quality'], df['length_cat']).loc[QUALITY_ORDER, valid_cats]

    # --- Split: Heatmap proporcional ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(confusion, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': '% dentro de categoría RMSD'},
                linewidths=0.5, vmin=0, vmax=60)
    ax.set_title('Distribución por longitud dado calidad\n(% por fila)')
    ax.set_xlabel('Categoría de longitud (residuos)')
    ax.set_ylabel('Calidad RMSD (mayor→menor precisión)')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_01_length_confmat_prop.png')
    plt.close(fig)
    log.info("  → fig_01_length_confmat_prop.png")

    # --- Split: Conteos absolutos ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(confusion_abs, annot=True, fmt='d', cmap='Blues',
                ax=ax, cbar_kws={'label': 'N proteínas'},
                linewidths=0.5)
    ax.set_title('Conteos absolutos por longitud dado calidad')
    ax.set_xlabel('Categoría de longitud (residuos)')
    ax.set_ylabel('Calidad RMSD (mayor→menor precisión)')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_01_length_confmat_count.png')
    plt.close(fig)
    log.info("  → fig_01_length_confmat_count.png")

    # --- Legacy: combined panel (kept for backwards compatibility) ---
    _legacy_fig_01_length_confmat(confusion, confusion_abs, valid_cats)

def _legacy_fig_01_length_confmat(confusion, confusion_abs, valid_cats):
    """Legacy combined 1x2 confmat panel — kept for backwards compatibility."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.heatmap(confusion, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=axes[0], cbar_kws={'label': '% dentro de categoría RMSD'},
                linewidths=0.5, vmin=0, vmax=60)
    axes[0].set_title('Distribución por longitud dado calidad\n(% por fila)')
    axes[0].set_xlabel('Categoría de longitud (residuos)')
    axes[0].set_ylabel('Calidad RMSD (mayor→menor precisión)')
    sns.heatmap(confusion_abs, annot=True, fmt='d', cmap='Blues',
                ax=axes[1], cbar_kws={'label': 'N proteínas'},
                linewidths=0.5)
    axes[1].set_title('Conteos absolutos por longitud dado calidad')
    axes[1].set_xlabel('Categoría de longitud (residuos)')
    axes[1].set_ylabel('Calidad RMSD (mayor→menor precisión)')
    fig.suptitle('Análisis de calidad por longitud de cadena proteica', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_01_length_confmat.png')
    plt.close(fig)
    log.info("  → fig_01_length_confmat.png (legacy)")


# ─── SECCIÓN 2–3: Parseo per-residuo ─────────────────────────────────────────

RESIDUE_CACHE = CACHE_DIR / 'per_residue_rmsd.parquet'

def _kabsch_align(P, Q):
    """
    Alinea Q sobre P usando el algoritmo de Kabsch (SVD).
    P, Q: (N,3) arrays de coordenadas.
    Retorna Q_aligned: (N,3) con la misma orientación que P.
    """
    p_cent = P.mean(axis=0)
    q_cent = Q.mean(axis=0)
    P_c = P - p_cent
    Q_c = Q - q_cent
    H = Q_c.T @ P_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return (Q_c @ R.T) + p_cent


def _parse_one_protein(args):
    """Worker: calcula RMSD por residuo para una proteína con alineamiento Kabsch."""
    row, sifts_sub = args
    pdb_id   = row['pdb_id'].lower()
    uniprot  = row['uniprot_id']
    chain_id = row['chain_id']

    pdb_path = PDB_DIR / f'pdb{pdb_id}.ent'
    af_path  = AF_DIR / f'{uniprot}.pdb'
    if not pdb_path.exists() or not af_path.exists():
        return []

    parser = PDBParser(QUIET=True)
    try:
        pdb_struct = parser.get_structure('pdb', str(pdb_path))
        af_struct  = parser.get_structure('af',  str(af_path))
    except Exception:
        return []

    # Seleccionar cadena correcta en PDB
    try:
        pdb_chain = pdb_struct[0][chain_id]
        af_chain  = af_struct[0]['A']  # AlphaFold siempre cadena A
    except Exception:
        return []

    # Construir mapeo PDB_pos → SP_pos usando SIFTS
    # Replicar la lógica exacta del procesador original (processor.py)
    import re as _re
    pos_map = {}  # pdb_res_id (int) → af_res_id (int)
    for _, srow in sifts_sub.iterrows():
        try:
            sp_beg = int(srow['SP_BEG'])
            sp_end = int(srow['SP_END'])
        except (ValueError, TypeError):
            continue

        # Parsear PDB_BEG (puede tener insertion codes como "1H")
        pdb_beg = None
        raw_beg = srow['PDB_BEG']
        if pd.notna(raw_beg):
            try:
                pdb_beg = int(raw_beg)
            except (ValueError, TypeError):
                m = _re.match(r'^(-?\d+)', str(raw_beg).strip())
                if m:
                    pdb_beg = int(m.group(1))

        # Parsear PDB_END
        pdb_end = None
        raw_end = srow['PDB_END']
        if pd.notna(raw_end):
            try:
                pdb_end = int(raw_end)
            except (ValueError, TypeError):
                m = _re.match(r'^(-?\d+)', str(raw_end).strip())
                if m:
                    pdb_end = int(m.group(1))

        # Fallbacks idénticos al procesador original
        if pdb_beg is None:
            pdb_beg = sp_beg
        if pdb_end is None:
            pdb_end = pdb_beg + (sp_end - sp_beg)

        offset = sp_beg - pdb_beg
        for p in range(pdb_beg, pdb_end + 1):
            pos_map[p] = p + offset

    # --- Paso 1: Recolectar todos los pares de CA con el mapeo SIFTS ---
    paired_pdb = []   # lista de (seq_num, aa1, ca_coord_pdb, bb_coords_pdb)
    paired_af_ca = [] # lista de coordenadas CA de AF en el orden correspondiente

    af_res_dict = {res.get_id()[1]: res for res in af_chain.get_residues()
                   if res.get_id()[0] == ' '}

    for pdb_res in pdb_chain.get_residues():
        res_id = pdb_res.get_id()
        if res_id[0] != ' ':
            continue
        pdb_seq_num = res_id[1]
        res_name = pdb_res.get_resname().strip()
        aa1 = AA_THREE_TO_ONE.get(res_name)
        if aa1 is None or 'CA' not in pdb_res:
            continue

        af_seq_num = pos_map.get(pdb_seq_num, pdb_seq_num)
        af_res = af_res_dict.get(af_seq_num)
        if af_res is None or 'CA' not in af_res:
            continue

        ca_pdb = pdb_res['CA'].get_vector().get_array()
        ca_af  = af_res['CA'].get_vector().get_array()

        # Backbone coords
        bb_pdb = []
        bb_af  = []
        for atom_name in ('N', 'CA', 'C', 'O'):
            if atom_name in pdb_res and atom_name in af_res:
                bb_pdb.append(pdb_res[atom_name].get_vector().get_array())
                bb_af.append(af_res[atom_name].get_vector().get_array())

        paired_pdb.append((pdb_seq_num, aa1, ca_pdb, bb_pdb))
        paired_af_ca.append((ca_af, bb_af))

    if len(paired_pdb) < 5:
        return []

    # --- Paso 2: Kabsch sobre todos los CA ---
    P = np.array([t[2] for t in paired_pdb])          # (N,3) PDB CA
    Q = np.array([t[0] for t in paired_af_ca])         # (N,3) AF CA
    Q_aligned = _kabsch_align(P, Q)

    # Rotation matrix from Q_orig to Q_aligned (needed for backbone too)
    q_cent = Q.mean(axis=0)
    p_cent = P.mean(axis=0)
    Q_c = Q - q_cent
    P_c = P - p_cent
    H = Q_c.T @ P_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T  # rotation matrix

    # --- Paso 3: Distancias per-residuo tras alineamiento ---
    records = []
    for i, (pdb_seq_num, aa1, ca_pdb, bb_pdb) in enumerate(paired_pdb):
        ca_rmsd = float(np.linalg.norm(P[i] - Q_aligned[i]))

        # Backbone RMSD con la misma rotación
        bb_af_orig = paired_af_ca[i][1]
        if bb_pdb and bb_af_orig:
            bb_pdb_arr = np.array(bb_pdb)
            bb_af_arr  = np.array(bb_af_orig)
            bb_af_aln  = (bb_af_arr - q_cent) @ R.T + p_cent
            bb_rmsd = float(np.sqrt(np.mean(np.sum((bb_pdb_arr - bb_af_aln)**2, axis=1))))
        else:
            bb_rmsd = np.nan

        records.append({
            'pdb_id':   pdb_id,
            'uniprot':  uniprot,
            'aa':       aa1,
            'fg':       AA_TO_FG.get(aa1, 'Otro'),
            'ca_rmsd':  ca_rmsd,
            'bb_rmsd':  bb_rmsd,
            'length':   row['matched_residues'],
        })
    return records


def build_residue_cache(df, max_workers=8):
    """Parsea todos los archivos PDB/AF para obtener RMSD per-residuo. Usa caché."""
    if RESIDUE_CACHE.exists():
        log.info(f"Cargando caché de residuos: {RESIDUE_CACHE}")
        return pd.read_parquet(RESIDUE_CACHE)

    log.info(f"Construyendo caché per-residuo para {len(df)} proteínas (workers={max_workers})")

    # Cargar SIFTS (solo columnas necesarias)
    log.info("Cargando SIFTS...")
    sifts = pd.read_csv(
        SIFTS_CSV, comment='#', low_memory=False,
        dtype={'PDB_BEG': str, 'PDB_END': str, 'SP_BEG': str, 'SP_END': str}
    )
    sifts['PDB'] = sifts['PDB'].str.lower()

    # Construir índice SIFTS por (pdb, chain)
    sifts_idx = {}
    for (pdb, chain), grp in sifts.groupby(['PDB','CHAIN']):
        sifts_idx[(pdb, chain)] = grp

    args = []
    for _, row in df.iterrows():
        key = (row['pdb_id'].lower(), row['chain_id'])
        sub = sifts_idx.get(key, pd.DataFrame())
        args.append((row, sub))

    all_records = []
    completed = 0
    log.info(f"Iniciando procesamiento con {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_parse_one_protein, a): i for i, a in enumerate(args)}
        for fut in as_completed(futures):
            completed += 1
            try:
                records = fut.result()
                if records:
                    all_records.extend(records)
            except Exception as e:
                pass
            if completed % 1000 == 0:
                log.info(f"  Procesadas {completed}/{len(args)} proteínas, {len(all_records):,} residuos")
                # Checkpoint parcial cada 1000
                if all_records:
                    pd.DataFrame(all_records).to_parquet(
                        RESIDUE_CACHE.with_suffix('.tmp.parquet'), index=False
                    )

    log.info(f"Total residuos parseados: {len(all_records):,}")
    result = pd.DataFrame(all_records)
    result.to_parquet(RESIDUE_CACHE, index=False)
    # Limpiar temporal
    tmp = RESIDUE_CACHE.with_suffix('.tmp.parquet')
    if tmp.exists():
        tmp.unlink()
    log.info(f"Caché guardado en {RESIDUE_CACHE}")
    return result

# ─── SECCIÓN 2: Por tipo de aminoácido ───────────────────────────────────────

def fig_02_aminoacid(res_df):
    log.info("Generando figuras 2: análisis por aminoácido")

    df = res_df.dropna(subset=['ca_rmsd']).copy()
    df['ca_rmsd_clip'] = df['ca_rmsd'].clip(upper=10)
    df['quality'] = df['ca_rmsd'].apply(rmsd_to_quality)

    # Orden de AAs por media de RMSD (mayor → menor)
    aa_stats = df.groupby('aa')['ca_rmsd_clip'].agg(['mean','median','count']).reset_index()
    aa_stats.columns = ['aa','mean_rmsd','median_rmsd','n_residues']
    aa_stats = aa_stats.sort_values('mean_rmsd', ascending=False)
    aa_order = aa_stats['aa'].tolist()

    # 2a. Bar chart ordenado mayor → menor con color por grupo funcional
    fg_colors = {'Alifático':'#4C72B0','Aromático':'#DD8452','Polar':'#55A868',
                 'Básico':'#C44E52','Ácido':'#8172B3','Otro':'gray'}
    bar_colors = [fg_colors.get(AA_TO_FG.get(aa, 'Otro'), 'gray') for aa in aa_order]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(aa_order, aa_stats.set_index('aa').loc[aa_order]['mean_rmsd'],
                  color=bar_colors, edgecolor='white', linewidth=0.5, alpha=0.85)

    # Error bars: bootstrap IC 95%
    err_lo, err_hi = [], []
    for aa in aa_order:
        vals = df[df['aa'] == aa]['ca_rmsd_clip'].values
        if len(vals) > 1:
            bs = np.array([np.mean(np.random.choice(vals, len(vals), replace=True))
                           for _ in range(200)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            err_lo.append(aa_stats.set_index('aa').loc[aa,'mean_rmsd'] - lo)
            err_hi.append(hi - aa_stats.set_index('aa').loc[aa,'mean_rmsd'])
        else:
            err_lo.append(0); err_hi.append(0)

    ax.errorbar(range(len(aa_order)), aa_stats.set_index('aa').loc[aa_order]['mean_rmsd'],
                yerr=[err_lo, err_hi], fmt='none', color='black', capsize=3, linewidth=1.2)

    ax.axhline(1.0, color='steelblue', linestyle='--', alpha=0.6, linewidth=1.2)
    ax.axhline(2.0, color='orange',    linestyle='--', alpha=0.6, linewidth=1.2)
    ax.set_xlabel('Aminoácido (ordenado de mayor a menor RMSD Cα)')
    ax.set_ylabel('RMSD Cα medio (Å) [±IC 95% bootstrap]')
    ax.set_title('Precisión de AlphaFold2 por tipo de aminoácido')

    # Leyenda de grupos funcionales
    handles = [mpatches.Patch(color=c, label=fg) for fg, c in fg_colors.items() if fg != 'Otro']
    ax.legend(handles=handles, title='Grupo funcional', loc='upper right', ncol=2)

    fig.savefig(FIGURES / 'fig_02_aa_rmsd_sorted.png')
    plt.close(fig)
    log.info("  → fig_02_aa_rmsd_sorted.png")

    # 2b. Violin plot por AA, agrupado por grupo funcional — SPLIT into individual panels
    fg_order = ['Ácido','Básico','Polar','Aromático','Alifático']
    fg_name_to_file = {'Ácido': 'acido', 'Básico': 'basico', 'Polar': 'polar',
                       'Aromático': 'aromatico', 'Alifático': 'alifatico'}

    for i, fg in enumerate(fg_order):
        fg_aas = [aa for aa in FUNCTIONAL_GROUPS.get(fg, []) if aa in df['aa'].values]
        fg_aas_sorted = sorted(fg_aas, key=lambda a: df[df['aa']==a]['ca_rmsd_clip'].mean(), reverse=True)
        if not fg_aas_sorted:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        data = [df[df['aa'] == aa]['ca_rmsd_clip'].values for aa in fg_aas_sorted]
        parts = ax.violinplot(data, positions=range(len(fg_aas_sorted)),
                              showmedians=True, showextrema=False)
        c = fg_colors[fg]
        for pc in parts['bodies']:
            pc.set_facecolor(c); pc.set_alpha(0.7)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
        ax.set_xticks(range(len(fg_aas_sorted)))
        ax.set_xticklabels(fg_aas_sorted)
        ax.set_title(f'Distribución de RMSD — Grupo funcional: {fg}', color=c, fontweight='bold')
        ax.axhline(1.0, color='steelblue', linestyle='--', alpha=0.5)
        ax.axhline(2.0, color='orange',    linestyle='--', alpha=0.5)
        ax.set_ylabel('RMSD Cα (Å) [truncado en 10 Å]')
        ax.set_xlabel('Aminoácido')
        fig.tight_layout()
        fname = f'fig_02_aa_violin_{fg_name_to_file[fg]}.png'
        fig.savefig(FIGURES / fname)
        plt.close(fig)
        log.info(f"  → {fname}")

    # Legacy combined panel
    _legacy_fig_02_aa_violin(df, fg_order, fg_colors)

    # 2c. Matriz de confusión calidad × aminoácido
    top_aas = aa_stats.head(20)['aa'].tolist()  # todos los 20 AA estándar
    aa_order_mat = [aa for aa in aa_order if aa in top_aas]

    conf_df = df[df['aa'].isin(top_aas)].copy()
    confusion = pd.crosstab(
        conf_df['quality'], conf_df['aa'],
        normalize='index'
    ).reindex(index=QUALITY_ORDER, columns=aa_order_mat, fill_value=0) * 100

    confusion_abs = pd.crosstab(
        conf_df['quality'], conf_df['aa']
    ).reindex(index=QUALITY_ORDER, columns=aa_order_mat, fill_value=0)

    # --- Split: Proportion heatmap ---
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(confusion, annot=True, fmt='.0f', cmap='RdYlGn_r',
                ax=ax, vmin=0, vmax=25,
                cbar_kws={'label': '% por fila (calidad RMSD)'}, linewidths=0.3)
    ax.set_title('Distribución de aminoácidos por calidad de predicción\n(ordenados mayor→menor RMSD)')
    ax.set_xlabel('Aminoácido (ordenado: mayor RMSD → derecha = menor RMSD)')
    ax.set_ylabel('Calidad RMSD')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_02_aa_confmat_prop.png')
    plt.close(fig)
    log.info("  → fig_02_aa_confmat_prop.png")

    # --- Split: Absolute counts heatmap ---
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(confusion_abs, annot=True, fmt='d', cmap='Blues',
                ax=ax, cbar_kws={'label': 'N residuos'},
                linewidths=0.3)
    ax.set_title('Conteos absolutos de residuos por calidad')
    ax.set_xlabel('Aminoácido')
    ax.set_ylabel('Calidad RMSD')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_02_aa_confmat_count.png')
    plt.close(fig)
    log.info("  → fig_02_aa_confmat_count.png")

    # Legacy combined panel
    _legacy_fig_02_aa_confmat(confusion, confusion_abs, aa_order_mat)
    return aa_stats

def _legacy_fig_02_aa_violin(df, fg_order, fg_colors):
    """Legacy combined 1x5 violin panel — kept for backwards compatibility."""
    fig, axes = plt.subplots(1, len(fg_order), figsize=(18, 6),
                             sharey=True, gridspec_kw={'wspace': 0.08})
    for i, (fg, ax) in enumerate(zip(fg_order, axes)):
        fg_aas = [aa for aa in FUNCTIONAL_GROUPS.get(fg, []) if aa in df['aa'].values]
        fg_aas_sorted = sorted(fg_aas, key=lambda a: df[df['aa']==a]['ca_rmsd_clip'].mean(), reverse=True)
        if not fg_aas_sorted:
            continue
        data = [df[df['aa'] == aa]['ca_rmsd_clip'].values for aa in fg_aas_sorted]
        parts = ax.violinplot(data, positions=range(len(fg_aas_sorted)),
                              showmedians=True, showextrema=False)
        c = fg_colors[fg]
        for pc in parts['bodies']:
            pc.set_facecolor(c); pc.set_alpha(0.7)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
        ax.set_xticks(range(len(fg_aas_sorted)))
        ax.set_xticklabels(fg_aas_sorted)
        ax.set_title(fg, color=c, fontweight='bold')
        ax.axhline(1.0, color='steelblue', linestyle='--', alpha=0.5)
        ax.axhline(2.0, color='orange',    linestyle='--', alpha=0.5)
        if i == 0:
            ax.set_ylabel('RMSD Cα (Å) [truncado en 10 Å]')
    fig.suptitle('Distribución de RMSD por aminoácido y grupo funcional', fontsize=14)
    fig.savefig(FIGURES / 'fig_02_aa_violin.png')
    plt.close(fig)
    log.info("  → fig_02_aa_violin.png (legacy)")


def _legacy_fig_02_aa_confmat(confusion, confusion_abs, aa_order_mat):
    """Legacy combined 1x2 confmat panel — kept for backwards compatibility."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    sns.heatmap(confusion, annot=True, fmt='.0f', cmap='RdYlGn_r',
                ax=axes[0], vmin=0, vmax=25,
                cbar_kws={'label': '% por fila (calidad RMSD)'}, linewidths=0.3)
    axes[0].set_title('Distribución de aminoácidos por calidad de predicción\n(ordenados mayor→menor RMSD)')
    axes[0].set_xlabel('Aminoácido (ordenado: mayor RMSD → derecha = menor RMSD)')
    axes[0].set_ylabel('Calidad RMSD')
    sns.heatmap(confusion_abs, annot=True, fmt='d', cmap='Blues',
                ax=axes[1], cbar_kws={'label': 'N residuos'},
                linewidths=0.3)
    axes[1].set_title('Conteos absolutos de residuos por calidad')
    axes[1].set_xlabel('Aminoácido')
    axes[1].set_ylabel('Calidad RMSD')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_02_aa_confmat.png')
    plt.close(fig)
    log.info("  → fig_02_aa_confmat.png (legacy)")


# ─── SECCIÓN 3: Por grupo funcional ──────────────────────────────────────────

def fig_03_functional_groups(res_df, base_df):
    log.info("Generando figuras 3: análisis por grupo funcional")

    df = res_df.dropna(subset=['ca_rmsd']).copy()
    df['ca_rmsd_clip'] = df['ca_rmsd'].clip(upper=10)
    df['quality'] = df['ca_rmsd'].apply(rmsd_to_quality)
    df = df[df['fg'] != 'Otro']

    # Orden grupos por media RMSD (mayor → menor)
    fg_stats = df.groupby('fg')['ca_rmsd_clip'].mean().sort_values(ascending=False)
    fg_order_sorted = fg_stats.index.tolist()
    fg_colors = {'Alifático':'#4C72B0','Aromático':'#DD8452','Polar':'#55A868',
                 'Básico':'#C44E52','Ácido':'#8172B3'}

    # 3a. Boxplot RMSD por grupo funcional
    fig, ax = plt.subplots(figsize=(9, 6))
    data_fg = [df[df['fg'] == fg]['ca_rmsd_clip'].values for fg in fg_order_sorted]
    bp = ax.boxplot(data_fg, labels=fg_order_sorted, patch_artist=True,
                    medianprops={'color':'black','linewidth':2},
                    whiskerprops={'linewidth':1.2},
                    capprops={'linewidth':1.2},
                    flierprops={'marker':'o','markersize':2,'alpha':0.3})
    for patch, fg in zip(bp['boxes'], fg_order_sorted):
        patch.set_facecolor(fg_colors.get(fg, 'gray'))
        patch.set_alpha(0.75)

    # Overlay media
    for i, fg in enumerate(fg_order_sorted):
        m = df[df['fg'] == fg]['ca_rmsd_clip'].mean()
        ax.plot(i+1, m, 'D', color='black', markersize=7, zorder=5, label='Media' if i==0 else '')

    ax.axhline(1.0, color='steelblue', linestyle='--', alpha=0.6, label='1 Å')
    ax.axhline(2.0, color='orange',    linestyle='--', alpha=0.6, label='2 Å')
    ax.set_xlabel('Grupo funcional (ordenado de mayor a menor RMSD)')
    ax.set_ylabel('RMSD Cα (Å) [truncado en 10 Å]')
    ax.set_title('Precisión de AlphaFold2 por grupo funcional de aminoácidos')
    ax.legend()

    # Añadir n= bajo cada categoría
    n_counts = {fg: len(df[df['fg']==fg]) for fg in fg_order_sorted}
    for i, fg in enumerate(fg_order_sorted):
        ax.text(i+1, -0.5, f'n={n_counts[fg]:,}', ha='center', va='top', fontsize=9, color='gray')

    fig.savefig(FIGURES / 'fig_03_fg_boxplot.png')
    plt.close(fig)
    log.info("  → fig_03_fg_boxplot.png")

    # 3b. Heatmap accuracy% por grupo funcional × longitud
    df['length_cat'] = pd.cut(
        df['length'], bins=LENGTH_BINS, labels=LENGTH_LABELS, right=False
    ).astype(str)

    # % Excelente (RMSD < 1Å) por fg × length_cat
    pct_excellent = df.groupby(['fg', 'length_cat']).apply(
        lambda g: (g['ca_rmsd'] < 1.0).mean() * 100
    ).unstack('length_cat')
    pct_excellent = pct_excellent.reindex(index=fg_order_sorted)
    valid_len_cats = [c for c in LENGTH_LABELS if c in pct_excellent.columns]
    pct_excellent = pct_excellent[valid_len_cats]

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pct_excellent, annot=True, fmt='.0f', cmap='RdYlGn',
                ax=ax, vmin=0, vmax=80,
                cbar_kws={'label': '% residuos con RMSD Cα < 1 Å'},
                linewidths=0.5)
    ax.set_title('Porcentaje de predicciones excelentes (RMSD < 1 Å)\npor grupo funcional y longitud de cadena')
    ax.set_xlabel('Categoría de longitud (residuos)')
    ax.set_ylabel('Grupo funcional (mayor→menor RMSD)')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_03_fg_heatmap.png')
    plt.close(fig)
    log.info("  → fig_03_fg_heatmap.png")

    # 3c. Matriz de confusión calidad × grupo funcional
    confusion = pd.crosstab(
        df['quality'], df['fg'],
        normalize='index'
    ).reindex(index=QUALITY_ORDER, columns=fg_order_sorted, fill_value=0) * 100

    confusion_abs = pd.crosstab(
        df['quality'], df['fg']
    ).reindex(index=QUALITY_ORDER, columns=fg_order_sorted, fill_value=0)

    # --- Split: Proportion heatmap ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(confusion, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=ax, vmin=0, vmax=50,
                cbar_kws={'label': '% por fila'}, linewidths=0.5)
    ax.set_title('Distribución de grupos funcionales por calidad\n(% por fila, ordenado mayor→menor RMSD)')
    ax.set_xlabel('Grupo funcional')
    ax.set_ylabel('Calidad RMSD')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_03_fg_confmat_prop.png')
    plt.close(fig)
    log.info("  → fig_03_fg_confmat_prop.png")

    # --- Split: Absolute counts heatmap ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(confusion_abs, annot=True, fmt='d', cmap='Blues',
                ax=ax, cbar_kws={'label': 'N residuos'}, linewidths=0.5)
    ax.set_title('Conteos absolutos por calidad y grupo funcional')
    ax.set_xlabel('Grupo funcional')
    ax.set_ylabel('Calidad RMSD')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_03_fg_confmat_count.png')
    plt.close(fig)
    log.info("  → fig_03_fg_confmat_count.png")

    # Legacy combined panel
    _legacy_fig_03_fg_confmat(confusion, confusion_abs, fg_order_sorted)

def _legacy_fig_03_fg_confmat(confusion, confusion_abs, fg_order_sorted):
    """Legacy combined 1x2 confmat panel — kept for backwards compatibility."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(confusion, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=axes[0], vmin=0, vmax=50,
                cbar_kws={'label': '% por fila'}, linewidths=0.5)
    axes[0].set_title('Distribución de grupos funcionales por calidad\n(% por fila, ordenado mayor→menor RMSD)')
    axes[0].set_xlabel('Grupo funcional')
    axes[0].set_ylabel('Calidad RMSD')
    sns.heatmap(confusion_abs, annot=True, fmt='d', cmap='Blues',
                ax=axes[1], cbar_kws={'label': 'N residuos'}, linewidths=0.5)
    axes[1].set_title('Conteos absolutos por calidad y grupo funcional')
    axes[1].set_xlabel('Grupo funcional')
    axes[1].set_ylabel('Calidad RMSD')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_03_fg_confmat.png')
    plt.close(fig)
    log.info("  → fig_03_fg_confmat.png (legacy)")


# ─── SECCIÓN 4: Por elemento atómico CHONSP ──────────────────────────────────

def _per_atom_rmsd(args):
    """
    Worker: RMSD por átomo individual (CHONSP) con alineamiento Kabsch global.
    Backbone vs cadena lateral determinado por nombre de átomo.
    """
    import re as _re2
    BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}
    CHONSP = {'C', 'H', 'O', 'N', 'S', 'P'}

    row, sifts_sub = args
    pdb_id   = row['pdb_id'].lower()
    uniprot  = row['uniprot_id']
    chain_id = row['chain_id']

    pdb_path = PDB_DIR / f'pdb{pdb_id}.ent'
    af_path  = AF_DIR / f'{uniprot}.pdb'
    if not pdb_path.exists() or not af_path.exists():
        return []

    parser = PDBParser(QUIET=True)
    try:
        pdb_struct = parser.get_structure('pdb', str(pdb_path))
        af_struct  = parser.get_structure('af',  str(af_path))
        pdb_chain  = pdb_struct[0][chain_id]
        af_chain   = af_struct[0]['A']
    except Exception:
        return []

    # Mapeo SIFTS con fallback NaN idéntico al procesador original
    pos_map = {}
    for _, srow in sifts_sub.iterrows():
        try:
            sp_beg = int(srow['SP_BEG']); sp_end = int(srow['SP_END'])
        except (ValueError, TypeError):
            continue
        pdb_beg = None
        raw = srow['PDB_BEG']
        if pd.notna(raw):
            try: pdb_beg = int(raw)
            except:
                m = _re2.match(r'^(-?\d+)', str(raw).strip())
                if m: pdb_beg = int(m.group(1))
        pdb_end = None
        raw = srow['PDB_END']
        if pd.notna(raw):
            try: pdb_end = int(raw)
            except:
                m = _re2.match(r'^(-?\d+)', str(raw).strip())
                if m: pdb_end = int(m.group(1))
        if pdb_beg is None: pdb_beg = sp_beg
        if pdb_end is None: pdb_end = pdb_beg + (sp_end - sp_beg)
        off = sp_beg - pdb_beg
        for p in range(pdb_beg, pdb_end + 1):
            pos_map[p] = p + off

    # Paso 1: Recolectar pares CA para alineamiento Kabsch global
    af_rd = {r.get_id()[1]: r for r in af_chain.get_residues() if r.get_id()[0] == ' '}
    pdb_ca_list, af_ca_list = [], []
    paired_residues = []   # (pdb_res, af_res)
    for pdb_res in pdb_chain.get_residues():
        if pdb_res.get_id()[0] != ' ' or 'CA' not in pdb_res:
            continue
        pdb_seq = pdb_res.get_id()[1]
        af_seq  = pos_map.get(pdb_seq, pdb_seq)
        af_res  = af_rd.get(af_seq)
        if af_res is None or 'CA' not in af_res:
            continue
        pdb_ca_list.append(pdb_res['CA'].get_vector().get_array())
        af_ca_list.append(af_res['CA'].get_vector().get_array())
        paired_residues.append((pdb_res, af_res))

    if len(pdb_ca_list) < 5:
        return []

    # Paso 2: Kabsch sobre CA
    P = np.array(pdb_ca_list)
    Q = np.array(af_ca_list)
    p_cent = P.mean(0); q_cent = Q.mean(0)
    H = (Q - q_cent).T @ (P - p_cent)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T   # rotation matrix

    # Paso 3: Per-atom distances using same rigid transform
    records = []
    for pdb_res, af_res in paired_residues:
        for atom in pdb_res.get_atoms():
            atom_name = atom.get_name().strip()
            element   = (atom.element.strip() if atom.element else atom_name[0]).upper()
            if element not in CHONSP:
                continue
            if atom_name not in af_res:
                continue
            af_atom = af_res[atom_name]
            af_coord_aligned = (af_atom.get_vector().get_array() - q_cent) @ R.T + p_cent
            dist = float(np.linalg.norm(atom.get_vector().get_array() - af_coord_aligned))
            records.append({
                'element':     element,
                'is_backbone': int(atom_name in BACKBONE_ATOMS),
                'atom_rmsd':   dist,
            })
    return records

ATOM_CACHE = CACHE_DIR / 'per_atom_chonsp.parquet'

def build_atom_cache(df, max_workers=8):
    if ATOM_CACHE.exists():
        log.info(f"Cargando caché CHONSP: {ATOM_CACHE}")
        return pd.read_parquet(ATOM_CACHE)

    log.info(f"Construyendo caché CHONSP para {len(df)} proteínas")
    sifts = pd.read_csv(
        SIFTS_CSV, comment='#', low_memory=False,
        dtype={'PDB_BEG': str, 'PDB_END': str, 'SP_BEG': str, 'SP_END': str}
    )
    sifts['PDB'] = sifts['PDB'].str.lower()
    sifts_idx = {(p, c): g for (p, c), g in sifts.groupby(['PDB','CHAIN'])}

    args = [(row, sifts_idx.get((row['pdb_id'].lower(), row['chain_id']), pd.DataFrame()))
            for _, row in df.iterrows()]

    all_records = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_per_atom_rmsd, a): i for i, a in enumerate(args)}
        for fut in as_completed(futures):
            completed += 1
            try:
                r = fut.result()
                if r:
                    all_records.extend(r)
            except Exception:
                pass
            if completed % 1000 == 0:
                log.info(f"  CHONSP: {completed}/{len(args)} proteínas, {len(all_records):,} átomos")

    result = pd.DataFrame(all_records)
    result.to_parquet(ATOM_CACHE, index=False)
    log.info(f"Caché CHONSP guardado: {len(result):,} átomos")
    return result


def fig_04_chonsp(atom_df):
    log.info("Generando figuras 4: análisis CHONSP")

    df = atom_df[atom_df['atom_rmsd'] < 20].copy()  # truncar outliers extremos
    ELEMENTS = ['C','H','O','N','S','P']

    # Orden por media RMSD (mayor → menor)
    el_stats = df.groupby('element')['atom_rmsd'].agg(['mean','median','count']).reset_index()
    el_stats.columns = ['element','mean_rmsd','median_rmsd','n_atoms']
    el_stats = el_stats[el_stats['element'].isin(ELEMENTS)].sort_values('mean_rmsd', ascending=False)
    el_order = el_stats['element'].tolist()

    el_palette = {'C':'#4C72B0','H':'#C8C8C8','O':'#DD8452','N':'#55A868','S':'#FFD700','P':'#C44E52'}

    # 4a. Violin plot por elemento (ordenado mayor→menor)
    fig, ax = plt.subplots(figsize=(10, 6))
    data_el = [df[df['element'] == el]['atom_rmsd'].clip(upper=8).values for el in el_order]
    counts  = [el_stats[el_stats['element'] == el]['n_atoms'].values[0] for el in el_order]

    parts = ax.violinplot(data_el, positions=range(len(el_order)),
                          showmedians=True, showextrema=False, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(el_palette.get(el_order[i], 'gray'))
        pc.set_alpha(0.80)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2.5)

    ax.set_xticks(range(len(el_order)))
    ax.set_xticklabels([f'{el}\n(n={c/1e6:.1f}M)' if c > 1e6 else f'{el}\n(n={c:,})'
                        for el, c in zip(el_order, counts)], fontsize=10)
    ax.set_xlabel('Elemento químico CHONSP (ordenado: mayor RMSD → menor)')
    ax.set_ylabel('RMSD atómico (Å) [truncado en 8 Å]')
    ax.set_title('Precisión de AlphaFold2 por elemento atómico (CHONSP)\nTodos los átomos de proteína, mapeo SIFTS correcto')
    ax.axhline(1.0, color='steelblue', linestyle='--', alpha=0.6, label='1 Å')
    ax.axhline(2.0, color='orange',    linestyle='--', alpha=0.6, label='2 Å')
    ax.legend()

    fig.savefig(FIGURES / 'comprehensive_element_analysis.png')  # reemplaza figura vieja
    fig.savefig(FIGURES / 'fig_04_chonsp_violin.png')
    plt.close(fig)
    log.info("  → fig_04_chonsp_violin.png / comprehensive_element_analysis.png")

    # 4b. Grouped bar: backbone vs cadena lateral por elemento
    bb_df  = df[df['is_backbone'] == 1].groupby('element')['atom_rmsd'].median().rename('Backbone')
    sc_df  = df[df['is_backbone'] == 0].groupby('element')['atom_rmsd'].median().rename('Cadena lateral')
    compare = pd.concat([bb_df, sc_df], axis=1).reindex(el_order).fillna(0)

    x = np.arange(len(el_order))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, compare['Backbone'],     width=w, label='Backbone (N,Cα,C,O)',
                color='#4C72B0', alpha=0.85, edgecolor='white')
    b2 = ax.bar(x + w/2, compare['Cadena lateral'], width=w, label='Cadena lateral',
                color='#DD8452', alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{el}\n({el_palette.get(el,"")})' for el in el_order])
    ax.set_xlabel('Elemento CHONSP (ordenado: mayor RMSD mediano → menor)')
    ax.set_ylabel('RMSD mediano (Å)')
    ax.set_title('Backbone vs cadena lateral: RMSD mediano por elemento atómico')
    ax.legend()
    ax.axhline(1.0, color='steelblue', linestyle='--', alpha=0.5)

    # Anotaciones de ratio backbone/sidechain
    for i, el in enumerate(el_order):
        bb_val = compare.loc[el, 'Backbone']
        sc_val = compare.loc[el, 'Cadena lateral']
        if sc_val > 0:
            ratio = sc_val / bb_val if bb_val > 0 else 0
            ax.text(i, max(bb_val, sc_val) + 0.05, f'×{ratio:.1f}',
                    ha='center', va='bottom', fontsize=9, color='dimgray')

    fig.savefig(FIGURES / 'atom_type_analysis.png')  # reemplaza figura vieja
    fig.savefig(FIGURES / 'fig_04_chonsp_backbone.png')
    plt.close(fig)
    log.info("  → fig_04_chonsp_backbone.png / atom_type_analysis.png")

    # 4c. Matriz de confusión calidad × elemento CHONSP
    df['quality'] = df['atom_rmsd'].apply(rmsd_to_quality)
    confusion = pd.crosstab(
        df['quality'], df['element'],
        normalize='index'
    ).reindex(index=QUALITY_ORDER, columns=el_order, fill_value=0) * 100

    confusion_abs = pd.crosstab(df['quality'], df['element']).reindex(
        index=QUALITY_ORDER, columns=el_order, fill_value=0)

    # --- Split: proporcional ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(confusion, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=ax, vmin=0, vmax=40,
                cbar_kws={'label': '% por fila'}, linewidths=0.5)
    ax.set_title('Distribución de elementos CHONSP por calidad atómica\n(% por fila)')
    ax.set_xlabel('Elemento (mayor RMSD → menor RMSD)')
    ax.set_ylabel('Calidad RMSD atómico')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_04_chonsp_confmat_prop.png')
    plt.close(fig)
    log.info("  → fig_04_chonsp_confmat_prop.png")

    # --- Split: conteos absolutos ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(confusion_abs, annot=True, fmt=',d', cmap='Blues',
                ax=ax, cbar_kws={'label': 'N átomos'}, linewidths=0.5)
    ax.set_title('Conteos absolutos de átomos por calidad y elemento')
    ax.set_xlabel('Elemento')
    ax.set_ylabel('Calidad RMSD atómico')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_04_chonsp_confmat_count.png')
    plt.close(fig)
    log.info("  → fig_04_chonsp_confmat_count.png")

    # --- Legacy: combined ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(confusion, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=axes[0], vmin=0, vmax=40,
                cbar_kws={'label': '% por fila'}, linewidths=0.5)
    axes[0].set_title('Distribución CHONSP por calidad\n(% por fila)')
    axes[0].set_xlabel('Elemento')
    axes[0].set_ylabel('Calidad RMSD atómico')
    sns.heatmap(confusion_abs, annot=True, fmt=',d', cmap='Blues',
                ax=axes[1], cbar_kws={'label': 'N átomos'}, linewidths=0.5)
    axes[1].set_title('Conteos absolutos')
    axes[1].set_xlabel('Elemento')
    axes[1].set_ylabel('Calidad RMSD atómico')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_04_chonsp_confmat.png')
    plt.close(fig)
    log.info("  → fig_04_chonsp_confmat.png")

# ─── SECCIÓN 5: Inferencia Bayesiana Beta-Binomial ───────────────────────────

def hdi_from_beta(a, b, credible_mass=0.95):
    """Calcula el HDI (highest density interval) de una distribución Beta(a,b)."""
    from scipy.optimize import minimize_scalar
    from scipy.stats import beta as bdist
    if a <= 0 or b <= 0:
        return (0, 1)

    def interval_width(lo):
        hi = bdist.ppf(bdist.cdf(lo, a, b) + credible_mass, a, b)
        return hi - lo

    result = minimize_scalar(interval_width, bounds=(0, 1 - credible_mass), method='bounded')
    lo = result.x
    hi = float(beta_dist.ppf(beta_dist.cdf(lo, a, b) + credible_mass, a, b))
    return float(lo), float(hi)


def fig_05_bayesian(base_df):
    log.info("Generando figuras 5: inferencia bayesiana")

    df = base_df.copy()
    df['excellent'] = (df['rmsd'] < 1.0).astype(int)

    # ──────────────────────────────────────────────────────────────────────────
    # 5a. Credible intervals vs CI frecuentista por categoría de longitud
    fig, ax = plt.subplots(figsize=(12, 6))

    valid_cats = [c for c in LENGTH_LABELS if c in df['length_cat'].values]
    y_pos = range(len(valid_cats))

    freq_lo_list, freq_hi_list = [], []
    bayes_lo_list, bayes_hi_list = [], []
    point_estimates = []

    # Prior no informativo: Beta(1,1) = Uniforme
    alpha0, beta0 = 1.0, 1.0

    for cat in valid_cats:
        sub = df[df['length_cat'] == cat]
        n   = len(sub)
        k   = sub['excellent'].sum()
        p_hat = k / n if n > 0 else 0

        # Frecuentista: IC Wilson
        z = 1.96
        center = (k + z**2/2) / (n + z**2)
        margin = z * np.sqrt(n * p_hat*(1-p_hat) + z**2/4) / (n + z**2)
        freq_lo_list.append(max(0, center - margin))
        freq_hi_list.append(min(1, center + margin))

        # Bayesiano: posterior Beta(alpha0+k, beta0+n-k) → HDI 95%
        a_post = alpha0 + k
        b_post = beta0  + n - k
        lo, hi = hdi_from_beta(a_post, b_post)
        bayes_lo_list.append(lo)
        bayes_hi_list.append(hi)
        point_estimates.append(p_hat)

    # Plot: Bayesiano (azul) vs Frecuentista (naranja)
    offset = 0.18
    for i, (cat, p, flo, fhi, blo, bhi) in enumerate(
            zip(valid_cats, point_estimates, freq_lo_list, freq_hi_list, bayes_lo_list, bayes_hi_list)):
        ax.errorbar(p, i + offset, xerr=[[p - flo],[fhi - p]],
                    fmt='o', color='#DD8452', capsize=5, markersize=7, linewidth=2,
                    label='IC 95% frecuentista (Wilson)' if i == 0 else '')
        ax.errorbar(p, i - offset, xerr=[[p - blo],[bhi - p]],
                    fmt='D', color='#4C72B0', capsize=5, markersize=7, linewidth=2,
                    label='HDI 95% bayesiano Beta(1,1)' if i == 0 else '')
        ax.text(fhi + 0.005, i + offset, f'{p*100:.1f}%', va='center', fontsize=8.5, color='#DD8452')

    ax.set_yticks(range(len(valid_cats)))
    ax.set_yticklabels([f'{c}\n(n={len(df[df["length_cat"]==c]):,})' for c in valid_cats])
    ax.set_xlabel('P(RMSD < 1 Å) — Proporción de predicciones excelentes')
    ax.set_ylabel('Categoría de longitud proteica (residuos)')
    ax.set_title(
        'Inferencia bayesiana: proporción de predicciones excelentes por longitud\n'
        'Prior Beta(1,1) [uniforme no informativo] → Posterior Beta(1+k, 1+n−k)'
    )
    ax.legend(loc='lower right')
    ax.set_xlim(0, 0.85)
    ax.axvline(df['excellent'].mean(), color='gray', linestyle=':', alpha=0.7, label='Media global')

    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_05_bayesian_ci.png')
    plt.close(fig)
    log.info("  → fig_05_bayesian_ci.png")

    # ──────────────────────────────────────────────────────────────────────────
    # 5b. Posterior predictive: distribución posterior para categorías clave
    # SPLIT into individual panels

    key_cats = ['200-300', '100-200', '>1000']
    key_cats = [c for c in key_cats if c in df['length_cat'].values]
    cat_file_names = {k: f'cat{i+1}' for i, k in enumerate(key_cats)}

    x_range = np.linspace(0, 1, 500)
    for i, cat in enumerate(key_cats):
        sub = df[df['length_cat'] == cat]
        n, k = len(sub), sub['excellent'].sum()
        p_hat = k / n

        # Posterior
        a_post = alpha0 + k
        b_post = beta0 + n - k
        posterior = beta_dist.pdf(x_range, a_post, b_post)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.fill_between(x_range, posterior, alpha=0.35, color='#4C72B0', label='Posterior')
        ax.plot(x_range, posterior, color='#4C72B0', linewidth=2)

        # HDI
        lo, hi = hdi_from_beta(a_post, b_post)
        hdi_mask = (x_range >= lo) & (x_range <= hi)
        ax.fill_between(x_range[hdi_mask], posterior[hdi_mask], alpha=0.65, color='#4C72B0')

        ax.axvline(p_hat, color='tomato', linestyle='--', linewidth=2, label=f'MLE: {p_hat:.3f}')
        ax.axvline(a_post / (a_post + b_post), color='navy', linestyle=':',
                   linewidth=1.5, label=f'Media posterior: {a_post/(a_post+b_post):.3f}')

        ax.set_title(f'Distribución posterior Beta-Binomial — Longitud: {cat}\n'
                     f'(n={n:,}, k={k:,}) — Prior Beta(1,1)', fontsize=11)
        ax.set_xlabel('P(RMSD < 1 Å)')
        ax.set_ylabel('Densidad posterior')
        ax.text(0.05, 0.95, f'HDI 95%:\n[{lo:.3f}, {hi:.3f}]',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fname = f'fig_05_bayesian_post_{cat_file_names[cat]}.png'
        fig.savefig(FIGURES / fname)
        plt.close(fig)
        log.info(f"  → {fname}")

    # Legacy combined panel
    _legacy_fig_05_bayesian_posterior(df, key_cats, alpha0, beta0)

def _legacy_fig_05_bayesian_posterior(df, key_cats, alpha0, beta0):
    """Legacy combined 1x3 posterior panel — kept for backwards compatibility."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x_range = np.linspace(0, 1, 500)
    for i, (cat, ax) in enumerate(zip(key_cats, axes)):
        sub = df[df['length_cat'] == cat]
        n, k = len(sub), sub['excellent'].sum()
        p_hat = k / n
        a_post = alpha0 + k
        b_post = beta0 + n - k
        posterior = beta_dist.pdf(x_range, a_post, b_post)
        ax.fill_between(x_range, posterior, alpha=0.35, color='#4C72B0', label='Posterior')
        ax.plot(x_range, posterior, color='#4C72B0', linewidth=2)
        lo, hi = hdi_from_beta(a_post, b_post)
        hdi_mask = (x_range >= lo) & (x_range <= hi)
        ax.fill_between(x_range[hdi_mask], posterior[hdi_mask], alpha=0.65, color='#4C72B0')
        ax.axvline(p_hat, color='tomato', linestyle='--', linewidth=2, label=f'MLE: {p_hat:.3f}')
        ax.axvline(a_post / (a_post + b_post), color='navy', linestyle=':',
                   linewidth=1.5, label=f'Media posterior: {a_post/(a_post+b_post):.3f}')
        ax.set_title(f'Longitud: {cat}\n(n={n:,}, k={k:,})', fontsize=11)
        ax.set_xlabel('P(RMSD < 1 Å)')
        ax.set_ylabel('Densidad posterior' if i == 0 else '')
        ax.text(0.05, 0.95, f'HDI 95%:\n[{lo:.3f}, {hi:.3f}]',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend(fontsize=8)
    fig.suptitle('Distribución posterior Beta-Binomial de P(predicción excelente)\npor categoría de longitud — Prior Beta(1,1)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_05_bayesian_posterior.png')
    plt.close(fig)
    log.info("  → fig_05_bayesian_posterior.png (legacy)")


# ─── SECCIÓN 6: Figura 1 renovada (panel 2×2) ───────────────────────────────

def fig_00_panel_main(base_df):
    """Generates individual standalone panels and legacy combined 2x2 panel."""
    log.info("Generando figura principal — paneles individuales + panel combinado")

    df = base_df.copy()

    # ── Panel A: Pipeline SIFTS (texto esquema) ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    pipeline_text = (
        "Pipeline de análisis\n\n"
        "1. SIFTS (EBI)\n"
        "   PDB_BEG → SP_BEG\n"
        "   (mapeo de numeración)\n\n"
        "2. PDB (RCSB)\n"
        "   estructura experimental\n\n"
        "3. AlphaFold2 (EBI)\n"
        "   estructura predicha\n\n"
        "4. Kabsch SVD\n"
        "   superposición óptima\n\n"
        "5. RMSD por residuo,\n"
        "   aminoácido, grupo\n"
        "   funcional y elemento"
    )
    ax.text(0.5, 0.95, pipeline_text, transform=ax.transAxes,
            ha='center', va='top', fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#EEF4FF', alpha=0.8))
    ax.set_title('Metodología: SIFTS como puente', fontweight='bold', fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_00a_pipeline.png')
    plt.close(fig)
    log.info("  → fig_00a_pipeline.png")

    # ── Panel B: Impacto SIFTS (bar chart) ──
    fig, ax = plt.subplots(figsize=(10, 7))
    metrics = ['RMSD\nmedio (Å)', 'RMSD\nmediana (Å)', '% con\nRMSD<2Å']
    sin_sifts = [29.94, 22.54, 13.9]
    con_sifts = [ 1.05,  0.64, 87.7]
    x = np.arange(3)
    w = 0.35
    b1 = ax.bar(x - w/2, sin_sifts, w, label='Sin mapeo SIFTS', color='#DD8452', alpha=0.85)
    b2 = ax.bar(x + w/2, con_sifts, w, label='Con mapeo SIFTS', color='#4C72B0', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title('Impacto del mapeo SIFTS correcto', fontweight='bold', fontsize=13)
    ax.legend()
    ax.set_ylabel('Valor')
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, color='#4C72B0')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_00b_sifts_comparison.png')
    plt.close(fig)
    log.info("  → fig_00b_sifts_comparison.png")

    # ── Panel C: Distribución RMSD del dataset diversificado ──
    fig, ax = plt.subplots(figsize=(10, 7))
    rmsd_clip = df['rmsd'].clip(upper=10)
    ax.hist(rmsd_clip, bins=60, color='#4C72B0', alpha=0.75, edgecolor='white', linewidth=0.3)
    med = df['rmsd'].median()
    ax.axvline(med, color='tomato',  linewidth=2, label=f'Mediana: {med:.2f} Å')
    ax.axvline(2.0, color='orange',  linewidth=1.5, linestyle='--', label='2 Å (buena)')
    ax.axvline(1.0, color='steelblue', linewidth=1.5, linestyle='--', label='1 Å (excelente)')
    ax.set_xlabel('RMSD Cα (Å) [truncado en 10 Å]')
    ax.set_ylabel('Número de proteínas')
    ax.set_title(f'Distribución de RMSD — {len(df):,} proteínas únicas', fontweight='bold', fontsize=13)
    pct_exc = (df['rmsd'] < 1.0).mean() * 100
    pct_bue = (df['rmsd'] < 2.0).mean() * 100
    ax.text(0.98, 0.95, f'<1 Å: {pct_exc:.1f}%\n<2 Å: {pct_bue:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_00c_rmsd_histogram.png')
    plt.close(fig)
    log.info("  → fig_00c_rmsd_histogram.png")

    # ── Panel D: % excelente por categoría de longitud ──
    fig, ax = plt.subplots(figsize=(10, 7))
    valid_cats = [c for c in LENGTH_LABELS if c in df['length_cat'].values]
    pct_by_len = [(df[df['length_cat']==c]['rmsd'] < 1.0).mean()*100 for c in valid_cats]
    n_by_len   = [len(df[df['length_cat']==c]) for c in valid_cats]
    colors_bar  = [PALETTE_CAT[i % len(PALETTE_CAT)] for i in range(len(valid_cats))]

    bars = ax.bar(valid_cats, pct_by_len, color=colors_bar, alpha=0.85, edgecolor='white')
    ax.axhline(pct_exc, color='gray', linestyle=':', linewidth=1.5, label=f'Global: {pct_exc:.1f}%')
    ax.set_xlabel('Longitud de cadena (residuos)')
    ax.set_ylabel('% predicciones excelentes (RMSD < 1 Å)')
    ax.set_title('Precisión excelente por longitud de cadena', fontweight='bold', fontsize=13)
    ax.set_xticklabels(valid_cats, rotation=30, ha='right', fontsize=9)
    ax.legend()
    for bar, n in zip(bars, n_by_len):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'n={n:,}', ha='center', va='bottom', fontsize=8, color='gray', rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_00d_excellence_by_length.png')
    plt.close(fig)
    log.info("  → fig_00d_excellence_by_length.png")

    # ── Legacy combined 2x2 panel ──
    _legacy_fig_00_panel_main(df)

def _legacy_fig_00_panel_main(df):
    """Legacy combined 2x2 panel — kept for backwards compatibility."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A
    ax = axes[0, 0]
    ax.axis('off')
    pipeline_text = (
        "Pipeline de análisis\n\n"
        "1. SIFTS (EBI)\n"
        "   PDB_BEG → SP_BEG\n"
        "   (mapeo de numeración)\n\n"
        "2. PDB (RCSB)\n"
        "   estructura experimental\n\n"
        "3. AlphaFold2 (EBI)\n"
        "   estructura predicha\n\n"
        "4. Kabsch SVD\n"
        "   superposición óptima\n\n"
        "5. RMSD por residuo,\n"
        "   aminoácido, grupo\n"
        "   funcional y elemento"
    )
    ax.text(0.5, 0.95, pipeline_text, transform=ax.transAxes,
            ha='center', va='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#EEF4FF', alpha=0.8))
    ax.set_title('(A) Metodología: SIFTS como puente', fontweight='bold')

    # Panel B
    ax = axes[0, 1]
    metrics = ['RMSD\nmedio (Å)', 'RMSD\nmediana (Å)', '% con\nRMSD<2Å']
    sin_sifts = [29.94, 22.54, 13.9]
    con_sifts = [ 1.05,  0.64, 87.7]
    x = np.arange(3)
    w = 0.35
    b1 = ax.bar(x - w/2, sin_sifts, w, label='Sin mapeo SIFTS', color='#DD8452', alpha=0.85)
    b2 = ax.bar(x + w/2, con_sifts, w, label='Con mapeo SIFTS', color='#4C72B0', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title('(B) Impacto del mapeo SIFTS correcto', fontweight='bold')
    ax.legend()
    ax.set_ylabel('Valor')
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8.5)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8.5, color='#4C72B0')

    # Panel C
    ax = axes[1, 0]
    rmsd_clip = df['rmsd'].clip(upper=10)
    ax.hist(rmsd_clip, bins=60, color='#4C72B0', alpha=0.75, edgecolor='white', linewidth=0.3)
    med = df['rmsd'].median()
    ax.axvline(med, color='tomato',  linewidth=2, label=f'Mediana: {med:.2f} Å')
    ax.axvline(2.0, color='orange',  linewidth=1.5, linestyle='--', label='2 Å (buena)')
    ax.axvline(1.0, color='steelblue', linewidth=1.5, linestyle='--', label='1 Å (excelente)')
    ax.set_xlabel('RMSD Cα (Å) [truncado en 10 Å]')
    ax.set_ylabel('Número de proteínas')
    ax.set_title(f'(C) Distribución de RMSD — {len(df):,} proteínas únicas', fontweight='bold')
    pct_exc = (df['rmsd'] < 1.0).mean() * 100
    pct_bue = (df['rmsd'] < 2.0).mean() * 100
    ax.text(0.98, 0.95, f'<1 Å: {pct_exc:.1f}%\n<2 Å: {pct_bue:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.legend(fontsize=9)

    # Panel D
    ax = axes[1, 1]
    valid_cats = [c for c in LENGTH_LABELS if c in df['length_cat'].values]
    pct_by_len = [(df[df['length_cat']==c]['rmsd'] < 1.0).mean()*100 for c in valid_cats]
    n_by_len   = [len(df[df['length_cat']==c]) for c in valid_cats]
    colors_bar  = [PALETTE_CAT[i % len(PALETTE_CAT)] for i in range(len(valid_cats))]
    bars = ax.bar(valid_cats, pct_by_len, color=colors_bar, alpha=0.85, edgecolor='white')
    ax.axhline(pct_exc, color='gray', linestyle=':', linewidth=1.5, label=f'Global: {pct_exc:.1f}%')
    ax.set_xlabel('Longitud de cadena (residuos)')
    ax.set_ylabel('% predicciones excelentes (RMSD < 1 Å)')
    ax.set_title('(D) Precisión excelente por longitud de cadena', fontweight='bold')
    ax.set_xticklabels(valid_cats, rotation=30, ha='right', fontsize=9)
    ax.legend()
    for bar, n in zip(bars, n_by_len):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'n={n:,}', ha='center', va='bottom', fontsize=7, color='gray', rotation=0)

    fig.suptitle(
        'Evaluación sistemática de AlphaFold2: análisis comparativo con 10,432 proteínas\n'
        'RMSD mediano con mapeo SIFTS correcto: 1.26 Å | 65.3% con RMSD < 2 Å',
        fontsize=12, y=1.01
    )
    fig.tight_layout()
    fig.savefig(FIGURES / 'Figure1_publication.png', bbox_inches='tight')
    plt.close(fig)
    log.info("  → Figure1_publication.png (legacy combined panel)")


# ─── SECCIÓN 6: Estadística descriptiva global — figuras separadas ────────────

def fig_06_global_stats(df):
    """Genera figuras de estadística descriptiva global, cada una separada."""
    log.info("Generando figuras 6: estadística descriptiva global")

    # 6a. Histograma RMSD global (0–10 Å)
    fig, ax = plt.subplots(figsize=(10, 6))
    rmsd_clip = df['rmsd'].clip(upper=10)
    counts_h, edges, patches = ax.hist(rmsd_clip, bins=80, color='#4C72B0',
                                        alpha=0.8, edgecolor='white', linewidth=0.3)
    med = df['rmsd'].median()
    mean = df['rmsd'].mean()
    ax.axvline(med, color='tomato', linewidth=2.5, linestyle='-',
               label=f'Mediana: {med:.2f} Å')
    ax.axvline(mean, color='darkred', linewidth=2, linestyle='--',
               label=f'Media: {mean:.2f} Å')
    ax.axvline(1.0, color='steelblue', linewidth=1.5, linestyle='--', alpha=0.7,
               label='Umbral excelente (1 Å)')
    ax.axvline(2.0, color='orange', linewidth=1.5, linestyle='--', alpha=0.7,
               label='Umbral buena (2 Å)')
    ax.set_xlabel('RMSD Cα (Å) [truncado en 10 Å]')
    ax.set_ylabel('Frecuencia (número de proteínas)')
    ax.set_title(f'Distribución de RMSD — {len(df):,} proteínas únicas\n'
                 f'Media: {mean:.2f} Å | Mediana: {med:.2f} Å | '
                 f'Desv. est.: {df["rmsd"].std():.2f} Å')
    ax.legend(fontsize=10)
    # Annotate percentages
    pct1 = (df['rmsd'] < 1).mean() * 100
    pct2 = (df['rmsd'] < 2).mean() * 100
    pct5 = (df['rmsd'] < 5).mean() * 100
    ax.text(0.97, 0.95,
            f'RMSD < 1 Å: {pct1:.1f}%\nRMSD < 2 Å: {pct2:.1f}%\nRMSD < 5 Å: {pct5:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    fig.savefig(FIGURES / 'fig_06a_histogram.png')
    plt.close(fig)
    log.info("  → fig_06a_histogram.png")

    # 6b. Histograma RMSD zoomed (0–5 Å) con KDE
    fig, ax = plt.subplots(figsize=(10, 6))
    sub = df[df['rmsd'] < 5]['rmsd']
    ax.hist(sub, bins=100, density=True, color='#4C72B0', alpha=0.6,
            edgecolor='white', linewidth=0.3, label='Histograma')
    # KDE
    kde_x = np.linspace(0, 5, 500)
    kde = stats.gaussian_kde(sub, bw_method=0.05)
    ax.plot(kde_x, kde(kde_x), color='tomato', linewidth=2.5, label='KDE (estimación de densidad)')
    ax.axvline(sub.median(), color='navy', linewidth=2, linestyle='--',
               label=f'Mediana: {sub.median():.2f} Å')
    # Mode from KDE
    mode_idx = np.argmax(kde(kde_x))
    mode_val = kde_x[mode_idx]
    ax.axvline(mode_val, color='green', linewidth=1.5, linestyle=':',
               label=f'Moda (KDE): {mode_val:.2f} Å')
    ax.axvline(1.0, color='steelblue', linewidth=1.2, linestyle='--', alpha=0.6)
    ax.axvline(2.0, color='orange', linewidth=1.2, linestyle='--', alpha=0.6)
    ax.set_xlabel('RMSD Cα (Å)')
    ax.set_ylabel('Densidad de probabilidad')
    ax.set_title(f'Distribución detallada de RMSD (rango 0–5 Å)\n'
                 f'{len(sub):,} proteínas ({len(sub)/len(df)*100:.1f}% del total) '
                 f'con RMSD < 5 Å')
    ax.legend(fontsize=10)
    fig.savefig(FIGURES / 'fig_06b_histogram_zoom.png')
    plt.close(fig)
    log.info("  → fig_06b_histogram_zoom.png")

    # 6c. Log-RMSD distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    log_rmsd = np.log10(df['rmsd'].clip(lower=0.01))
    ax.hist(log_rmsd, bins=80, color='#55A868', alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.axvline(np.log10(med), color='tomato', linewidth=2.5,
               label=f'Mediana: {med:.2f} Å (log₁₀={np.log10(med):.2f})')
    ax.axvline(0, color='steelblue', linewidth=1.5, linestyle='--', alpha=0.7,
               label='1 Å (log₁₀=0)')
    ax.axvline(np.log10(2), color='orange', linewidth=1.5, linestyle='--', alpha=0.7,
               label='2 Å (log₁₀=0.30)')
    ax.set_xlabel('log₁₀(RMSD Cα / Å)')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Distribución logarítmica de RMSD — {len(df):,} proteínas\n'
                 'La transformación log revela la estructura bimodal de la distribución')
    ax.legend(fontsize=10)
    # Annotate skewness and kurtosis
    sk = df['rmsd'].skew()
    ku = df['rmsd'].kurtosis()
    sk_log = log_rmsd.skew()
    ku_log = log_rmsd.kurtosis()
    ax.text(0.03, 0.95,
            f'Asimetría (RMSD): {sk:.2f}\nCurtosis (RMSD): {ku:.2f}\n'
            f'Asimetría (log): {sk_log:.2f}\nCurtosis (log): {ku_log:.2f}',
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    fig.savefig(FIGURES / 'fig_06c_log_histogram.png')
    plt.close(fig)
    log.info("  → fig_06c_log_histogram.png")

    # 6d. CDF (Función de distribución acumulada)
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_rmsd = np.sort(df['rmsd'].values)
    cdf = np.arange(1, len(sorted_rmsd)+1) / len(sorted_rmsd)
    ax.plot(sorted_rmsd, cdf, color='#4C72B0', linewidth=2)
    # Mark key thresholds
    for thresh, color, name in [(0.5, '#2ca02c', '0.5'), (1.0, 'steelblue', '1.0'),
                                 (2.0, 'orange', '2.0'), (5.0, '#C44E52', '5.0'),
                                 (10.0, 'darkred', '10.0')]:
        pct = (df['rmsd'] < thresh).mean()
        ax.axvline(thresh, color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(pct, color=color, linestyle=':', alpha=0.3, linewidth=0.8)
        ax.plot(thresh, pct, 'o', color=color, markersize=8, zorder=5)
        ax.annotate(f'{name} Å\n{pct*100:.1f}%', xy=(thresh, pct),
                    xytext=(thresh+0.5, pct-0.05), fontsize=9, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))
    ax.set_xlabel('RMSD Cα (Å)')
    ax.set_ylabel('Proporción acumulada')
    ax.set_title(f'Función de distribución acumulada (CDF) de RMSD\n{len(df):,} proteínas')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.02)
    fig.savefig(FIGURES / 'fig_06d_cdf.png')
    plt.close(fig)
    log.info("  → fig_06d_cdf.png")

    # 6e. Q-Q plot (vs normal y vs log-normal) — SPLIT into individual panels

    # --- Split: Q-Q vs Normal ---
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(df['rmsd'].clip(upper=50), dist='norm', plot=ax)
    ax.set_title('Q-Q Plot: RMSD vs distribución Normal\n'
                 '(Desviación severa — RMSD no es normal)')
    ax.get_lines()[0].set(color='#4C72B0', markersize=3, alpha=0.5)
    ax.get_lines()[1].set(color='tomato', linewidth=2)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_06e_qqplot_normal.png')
    plt.close(fig)
    log.info("  → fig_06e_qqplot_normal.png")

    # --- Split: Q-Q vs Log-Normal ---
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(np.log10(df['rmsd'].clip(lower=0.01)), dist='norm', plot=ax)
    ax.set_title('Q-Q Plot: log10(RMSD) vs distribución Normal\n'
                 '(Mejor ajuste — distribución log-normal)')
    ax.get_lines()[0].set(color='#55A868', markersize=3, alpha=0.5)
    ax.get_lines()[1].set(color='tomato', linewidth=2)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_06e_qqplot_lognormal.png')
    plt.close(fig)
    log.info("  → fig_06e_qqplot_lognormal.png")

    # Legacy combined panel
    _legacy_fig_06e_qqplot(df)
    log.info("  → fig_06e_qqplot.png (legacy)")

    # 6f. Box plot global con anotaciones
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([df['rmsd'].clip(upper=20).values], vert=True, patch_artist=True,
                    widths=0.5, medianprops={'color': 'tomato', 'linewidth': 2.5},
                    whiskerprops={'linewidth': 1.5}, capprops={'linewidth': 1.5},
                    flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.2})
    bp['boxes'][0].set_facecolor('#4C72B0')
    bp['boxes'][0].set_alpha(0.6)
    # Annotate quartiles
    q1 = df['rmsd'].quantile(0.25)
    q3 = df['rmsd'].quantile(0.75)
    iqr = q3 - q1
    ax.text(1.35, med, f'Mediana: {med:.2f} Å', va='center', fontsize=10, color='tomato')
    ax.text(1.35, q1, f'Q1: {q1:.2f} Å', va='center', fontsize=9, color='gray')
    ax.text(1.35, q3, f'Q3: {q3:.2f} Å', va='center', fontsize=9, color='gray')
    ax.text(1.35, min(q3 + 1.5*iqr, 20), f'IQR: {iqr:.2f} Å', va='center', fontsize=9,
            color='dimgray')
    ax.axhline(1.0, color='steelblue', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(2.0, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel('RMSD Cα (Å) [truncado en 20 Å]')
    ax.set_xticklabels([f'{len(df):,}\nproteínas'])
    ax.set_title(f'Box plot de RMSD global\n'
                 f'IQR: [{q1:.2f}, {q3:.2f}] Å — Rango intercuartílico: {iqr:.2f} Å')
    fig.savefig(FIGURES / 'fig_06f_boxplot_global.png')
    plt.close(fig)
    log.info("  → fig_06f_boxplot_global.png")

    # 6g. Donut chart de calidad
    fig, ax = plt.subplots(figsize=(8, 8))
    quality_counts = df['quality'].value_counts().reindex(QUALITY_ORDER)
    colors_q = ['#2ca02c', '#4C72B0', '#DD8452', '#C44E52']
    wedges, texts, autotexts = ax.pie(
        quality_counts, labels=None, autopct='%1.1f%%',
        startangle=90, colors=colors_q, pctdistance=0.78,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    # Inner circle for donut
    centre_circle = plt.Circle((0,0), 0.55, fc='white')
    ax.add_patch(centre_circle)
    ax.text(0, 0, f'{len(df):,}\nproteínas', ha='center', va='center',
            fontsize=14, fontweight='bold')
    # Legend
    legend_labels = [f'{q} (<{t[1]} Å): {quality_counts[q]:,}'
                     if t[1] != np.inf else f'{q} (≥5 Å): {quality_counts[q]:,}'
                     for q, t in zip(QUALITY_ORDER, QUALITY_THRESHOLDS.values())]
    ax.legend(wedges, legend_labels, title='Calidad RMSD',
              loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=10)
    ax.set_title('Distribución de calidad de predicciones AlphaFold2\n'
                 '(clasificación por umbral de RMSD Cα)', fontsize=13, pad=20)
    fig.savefig(FIGURES / 'fig_06g_quality_donut.png')
    plt.close(fig)
    log.info("  → fig_06g_quality_donut.png")

    # 6h. ECDF por categoría de longitud (superpuestas)
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_cats = [c for c in LENGTH_LABELS if c in df['length_cat'].values]
    for i, cat in enumerate(valid_cats):
        sub = df[df['length_cat'] == cat]['rmsd'].clip(upper=15)
        sorted_v = np.sort(sub.values)
        ecdf = np.arange(1, len(sorted_v)+1) / len(sorted_v)
        ax.plot(sorted_v, ecdf, linewidth=2, color=PALETTE_CAT[i % len(PALETTE_CAT)],
                label=f'{cat} (n={len(sub):,})')
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(2.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('RMSD Cα (Å)')
    ax.set_ylabel('Proporción acumulada')
    ax.set_title('ECDF de RMSD por categoría de longitud proteica\n'
                 'Curvas más a la izquierda = mejor precisión')
    ax.legend(title='Longitud (residuos)', fontsize=9)
    ax.set_xlim(0, 12)
    fig.savefig(FIGURES / 'fig_06h_ecdf_by_length.png')
    plt.close(fig)
    log.info("  → fig_06h_ecdf_by_length.png")

    # 6i. Tabla estadísticas descriptivas como figura
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
    table_data = [
        ['N proteínas', f'{len(df):,}'],
        ['Media', f'{df["rmsd"].mean():.2f} Å'],
        ['Mediana', f'{df["rmsd"].median():.2f} Å'],
        ['Desviación estándar', f'{df["rmsd"].std():.2f} Å'],
        ['Mínimo', f'{df["rmsd"].min():.2f} Å'],
        ['Máximo', f'{df["rmsd"].max():.2f} Å'],
        ['Q1 (percentil 25)', f'{df["rmsd"].quantile(0.25):.2f} Å'],
        ['Q3 (percentil 75)', f'{df["rmsd"].quantile(0.75):.2f} Å'],
        ['IQR', f'{df["rmsd"].quantile(0.75) - df["rmsd"].quantile(0.25):.2f} Å'],
        ['Percentil 5', f'{df["rmsd"].quantile(0.05):.2f} Å'],
        ['Percentil 95', f'{df["rmsd"].quantile(0.95):.2f} Å'],
        ['Asimetría (skewness)', f'{df["rmsd"].skew():.2f}'],
        ['Curtosis (exceso)', f'{df["rmsd"].kurtosis():.2f}'],
        ['Coef. variación', f'{df["rmsd"].std()/df["rmsd"].mean()*100:.1f}%'],
        ['< 0.5 Å', f'{(df["rmsd"]<0.5).sum():,} ({(df["rmsd"]<0.5).mean()*100:.1f}%)'],
        ['< 1.0 Å', f'{(df["rmsd"]<1).sum():,} ({(df["rmsd"]<1).mean()*100:.1f}%)'],
        ['< 2.0 Å', f'{(df["rmsd"]<2).sum():,} ({(df["rmsd"]<2).mean()*100:.1f}%)'],
        ['< 5.0 Å', f'{(df["rmsd"]<5).sum():,} ({(df["rmsd"]<5).mean()*100:.1f}%)'],
        ['> 10.0 Å', f'{(df["rmsd"]>10).sum():,} ({(df["rmsd"]>10).mean()*100:.1f}%)'],
    ]
    table = ax.table(cellText=table_data,
                     colLabels=['Estadístico', 'Valor'],
                     loc='center', cellLoc='left',
                     colWidths=[0.45, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)
    # Style header
    for j in range(2):
        table[0, j].set_facecolor('#4C72B0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Alternate row colors
    for i in range(1, len(table_data)+1):
        for j in range(2):
            table[i, j].set_facecolor('#F0F4F8' if i % 2 == 0 else 'white')
    ax.set_title('Estadísticas descriptivas completas del RMSD\n'
                 f'Dataset: {len(df):,} proteínas únicas con mapeo SIFTS',
                 fontsize=13, pad=20)
    fig.savefig(FIGURES / 'fig_06i_stats_table.png')
    plt.close(fig)
    log.info("  → fig_06i_stats_table.png")

    # 6j. Violin + strip (raincloud-style) por longitud
    fig, ax = plt.subplots(figsize=(14, 7))
    valid_cats = [c for c in LENGTH_LABELS if c in df['length_cat'].values]
    data_by_cat = [df[df['length_cat'] == c]['rmsd'].clip(upper=15).values for c in valid_cats]

    # Violin (half)
    parts = ax.violinplot(data_by_cat, positions=range(len(valid_cats)),
                          showmedians=False, showextrema=False, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE_CAT[i % len(PALETTE_CAT)])
        pc.set_alpha(0.6)
        # Make it half-violin by clipping
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)

    # Box plots (slim)
    bp = ax.boxplot(data_by_cat, positions=range(len(valid_cats)),
                    widths=0.15, patch_artist=True,
                    medianprops={'color': 'white', 'linewidth': 2},
                    whiskerprops={'linewidth': 1.2},
                    capprops={'linewidth': 1.2},
                    flierprops={'marker': '.', 'markersize': 1, 'alpha': 0.1},
                    showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('dimgray')
        patch.set_alpha(0.8)

    # Jitter scatter on right side
    for i, (cat, data) in enumerate(zip(valid_cats, data_by_cat)):
        if len(data) > 300:
            sample = np.random.choice(data, 300, replace=False)
        else:
            sample = data
        jitter = np.random.uniform(0.12, 0.35, len(sample))
        ax.scatter(i + jitter, sample, s=3, alpha=0.15,
                   color=PALETTE_CAT[i % len(PALETTE_CAT)])

    counts = [len(d) for d in data_by_cat]
    medians = [np.median(d) for d in data_by_cat]
    ax.set_xticks(range(len(valid_cats)))
    ax.set_xticklabels([f'{c}\n(n={counts[i]:,})\nmed={medians[i]:.2f}Å'
                        for i, c in enumerate(valid_cats)], fontsize=9)
    ax.axhline(1.0, color='steelblue', linestyle='--', alpha=0.5)
    ax.axhline(2.0, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Categoría de longitud (residuos)')
    ax.set_ylabel('RMSD Cα (Å) [truncado en 15 Å]')
    ax.set_title('Distribución completa de RMSD por longitud: violin + box + jitter\n'
                 '(Raincloud plot — visualiza densidad, cuartiles y datos individuales)')
    fig.savefig(FIGURES / 'fig_06j_raincloud.png')
    plt.close(fig)
    log.info("  → fig_06j_raincloud.png")

    # 6k. Top 50 outliers (highest RMSD)
    fig, ax = plt.subplots(figsize=(14, 7))
    top50 = df.nlargest(50, 'rmsd')[['uniprot_id', 'pdb_id', 'rmsd', 'matched_residues']].copy()
    top50 = top50.sort_values('rmsd', ascending=True).reset_index(drop=True)
    colors_out = ['#C44E52' if r > 50 else '#DD8452' if r > 20 else '#FFD700'
                  for r in top50['rmsd']]
    bars = ax.barh(range(len(top50)), top50['rmsd'], color=colors_out, alpha=0.85,
                   edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(top50)))
    ax.set_yticklabels([f"{row['uniprot_id']} ({row['pdb_id'].upper()}, {row['matched_residues']} res)"
                        for _, row in top50.iterrows()], fontsize=7)
    ax.set_xlabel('RMSD Cα (Å)')
    ax.set_title('Top 50 proteínas con mayor RMSD\n'
                 '(posibles errores de mapeo, proteínas desordenadas o cambios conformacionales)')
    for i, row in top50.iterrows():
        ax.text(row['rmsd'] + 1, i, f'{row["rmsd"]:.1f} Å', va='center', fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_06k_outliers.png')
    plt.close(fig)
    log.info("  → fig_06k_outliers.png")

    # 6l. Correlation: RMSD vs matched residues (with marginal distributions)
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 4, figure=fig)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    x = df['matched_residues'].clip(upper=800)
    y = df['rmsd'].clip(upper=15)

    # Main scatter (hexbin)
    hb = ax_main.hexbin(x, y, gridsize=40, cmap='cividis', mincnt=1, bins='log')
    ax_main.axhline(1.0, color='steelblue', linestyle='--', alpha=0.6)
    ax_main.axhline(2.0, color='orange', linestyle='--', alpha=0.6)
    ax_main.set_xlabel('Residuos mapeados')
    ax_main.set_ylabel('RMSD Cα (Å)')

    # Top marginal
    ax_top.hist(x, bins=60, color='#4C72B0', alpha=0.7, edgecolor='white', linewidth=0.3)
    ax_top.set_ylabel('Frecuencia')
    ax_top.tick_params(labelbottom=False)
    ax_top.set_title('RMSD vs longitud con distribuciones marginales')

    # Right marginal
    ax_right.hist(y, bins=60, orientation='horizontal', color='#55A868',
                  alpha=0.7, edgecolor='white', linewidth=0.3)
    ax_right.set_xlabel('Frecuencia')
    ax_right.tick_params(labelleft=False)

    # Correlation stats
    from scipy.stats import spearmanr, pearsonr
    r_p, p_p = pearsonr(df['matched_residues'], df['rmsd'])
    r_s, p_s = spearmanr(df['matched_residues'], df['rmsd'])
    ax_main.text(0.03, 0.97,
                 f'Pearson r = {r_p:.3f} (p = {p_p:.2e})\n'
                 f'Spearman ρ = {r_s:.3f} (p = {p_s:.2e})',
                 transform=ax_main.transAxes, ha='left', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.colorbar(hb, ax=ax_right, label='log₁₀(n)', shrink=0.5, pad=0.15)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_06l_correlation.png')
    plt.close(fig)
    log.info("  → fig_06l_correlation.png")


def _legacy_fig_06e_qqplot(df):
    """Legacy combined 1x2 Q-Q plot panel — kept for backwards compatibility."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    stats.probplot(df['rmsd'].clip(upper=50), dist='norm', plot=ax)
    ax.set_title('Q-Q Plot: RMSD vs distribución Normal\n'
                 '(Desviación severa → RMSD no es normal)')
    ax.get_lines()[0].set(color='#4C72B0', markersize=3, alpha=0.5)
    ax.get_lines()[1].set(color='tomato', linewidth=2)
    ax = axes[1]
    stats.probplot(np.log10(df['rmsd'].clip(lower=0.01)), dist='norm', plot=ax)
    ax.set_title('Q-Q Plot: log₁₀(RMSD) vs distribución Normal\n'
                 '(Mejor ajuste → distribución log-normal)')
    ax.get_lines()[0].set(color='#55A868', markersize=3, alpha=0.5)
    ax.get_lines()[1].set(color='tomato', linewidth=2)
    fig.suptitle('Evaluación de normalidad: RMSD vs log₁₀(RMSD)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_06e_qqplot.png')
    plt.close(fig)


def fig_07_length_detailed(df):
    """Análisis detallado por longitud — estadísticas por categoría."""
    log.info("Generando figuras 7: análisis detallado por longitud")

    valid_cats = [c for c in LENGTH_LABELS if c in df['length_cat'].values]

    # 7a. Estadísticas detalladas por categoría (tabla visual)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    headers = ['Categoría', 'N', 'Media', 'Mediana', 'Std', 'Q1', 'Q3',
               '<1Å', '<2Å', '>5Å']
    table_data = []
    for cat in valid_cats:
        sub = df[df['length_cat'] == cat]['rmsd']
        table_data.append([
            cat, f'{len(sub):,}', f'{sub.mean():.2f}', f'{sub.median():.2f}',
            f'{sub.std():.2f}', f'{sub.quantile(0.25):.2f}', f'{sub.quantile(0.75):.2f}',
            f'{(sub<1).mean()*100:.1f}%', f'{(sub<2).mean()*100:.1f}%',
            f'{(sub>5).mean()*100:.1f}%',
        ])
    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for j in range(len(headers)):
        table[0, j].set_facecolor('#4C72B0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(table_data)+1):
        for j in range(len(headers)):
            table[i, j].set_facecolor('#F0F4F8' if i % 2 == 0 else 'white')
    ax.set_title('Estadísticas descriptivas de RMSD por categoría de longitud proteica',
                 fontsize=13, pad=20)
    fig.savefig(FIGURES / 'fig_07a_length_table.png')
    plt.close(fig)
    log.info("  → fig_07a_length_table.png")

    # 7b. KDE overlay by length
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cat in enumerate(valid_cats):
        sub = df[df['length_cat'] == cat]['rmsd'].clip(upper=10)
        if len(sub) > 10:
            kde = stats.gaussian_kde(sub, bw_method=0.15)
            x = np.linspace(0, 10, 300)
            ax.plot(x, kde(x), linewidth=2.5, color=PALETTE_CAT[i % len(PALETTE_CAT)],
                    label=f'{cat} (n={len(sub):,}, med={sub.median():.2f}Å)')
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(2.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('RMSD Cα (Å)')
    ax.set_ylabel('Densidad')
    ax.set_title('Estimación de densidad kernel (KDE) de RMSD por categoría de longitud\n'
                 'Comparación directa de distribuciones')
    ax.legend(title='Longitud (residuos)', fontsize=9)
    fig.savefig(FIGURES / 'fig_07b_kde_by_length.png')
    plt.close(fig)
    log.info("  → fig_07b_kde_by_length.png")

    # 7c. % excellent & good stacked bar by length
    fig, ax = plt.subplots(figsize=(10, 6))
    pcts = []
    for cat in valid_cats:
        sub = df[df['length_cat'] == cat]
        n = len(sub)
        pcts.append({
            'cat': cat,
            'Excelente': (sub['rmsd'] < 1).sum() / n * 100,
            'Buena': ((sub['rmsd'] >= 1) & (sub['rmsd'] < 2)).sum() / n * 100,
            'Moderada': ((sub['rmsd'] >= 2) & (sub['rmsd'] < 5)).sum() / n * 100,
            'Pobre': (sub['rmsd'] >= 5).sum() / n * 100,
        })
    pct_df = pd.DataFrame(pcts).set_index('cat')
    colors_stack = ['#2ca02c', '#4C72B0', '#DD8452', '#C44E52']
    pct_df.plot(kind='bar', stacked=True, color=colors_stack, ax=ax, alpha=0.85,
                edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Categoría de longitud (residuos)')
    ax.set_ylabel('Porcentaje (%)')
    ax.set_title('Distribución de calidad por categoría de longitud\n'
                 '(barras apiladas: proporción de cada nivel de calidad)')
    ax.legend(title='Calidad RMSD', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xticklabels(valid_cats, rotation=30, ha='right')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_07c_stacked_bar.png')
    plt.close(fig)
    log.info("  → fig_07c_stacked_bar.png")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Generador de figuras para reporte AlphaFold — ITAM MDS")
    log.info("=" * 60)

    # 0. Cargar datos base
    log.info("Cargando dataset base (10,432 proteínas)...")
    base_df = load_base_data()
    log.info(f"Dataset cargado: {len(base_df)} proteínas, RMSD mediana={base_df['rmsd'].median():.3f} Å")

    # 1. Figuras de estadística descriptiva global (NUEVAS)
    fig_06_global_stats(base_df)

    # 2. Figuras por longitud (originales)
    fig_01_length(base_df)

    # 3. Figuras detalladas por longitud (NUEVAS)
    fig_07_length_detailed(base_df)

    # 4. Figura principal (panel 2×2)
    fig_00_panel_main(base_df)

    # 5. Caché per-residuo (lento — parsea todos los PDB/AF)
    log.info("Cargando/construyendo caché per-residuo...")
    res_df = build_residue_cache(base_df, max_workers=8)

    # 6. Figuras por aminoácido
    fig_02_aminoacid(res_df)

    # 7. Figuras por grupo funcional
    fig_03_functional_groups(res_df, base_df)

    # 8. Caché per-átomo CHONSP
    log.info("Cargando/construyendo caché CHONSP...")
    atom_df = build_atom_cache(base_df, max_workers=8)

    # 9. Figuras CHONSP
    fig_04_chonsp(atom_df)

    # 10. Figuras bayesianas
    fig_05_bayesian(base_df)

    log.info("=" * 60)
    log.info("¡Todas las figuras generadas exitosamente!")
    log.info(f"Directorio: {FIGURES}")
    log.info("Figuras generadas:")
    for f in sorted(FIGURES.glob('fig_0*.png')) + [
        FIGURES / 'Figure1_publication.png',
        FIGURES / 'comprehensive_element_analysis.png',
        FIGURES / 'atom_type_analysis.png',
    ]:
        if f.exists():
            log.info(f"  ✓ {f.name}")


if __name__ == '__main__':
    main()
