
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Dict, List, Optional

plt.switch_backend('Agg')  # headless-safe


@dataclass
class RQ4Config:
    index_data_dir: str = 'rq4_complete_results/index_data'
    output_dir: str = 'rq4_complete_results'
    plots_dir: str = field(init=False)
    exports_dir: str = field(init=False)
    # You can adjust weights to match your study
    estimated_weights: Dict[str, float] = field(default_factory=lambda: {
        'NIFTYBANK': 0.35,
        'NIFTYIT': 0.18,
        'NIFTYENERGY': 0.12,
        'NIFTYFMCG': 0.08,
        'NIFTYAUTO': 0.06,
        'NIFTYDEFENSE': 0.03,
        'Other': 0.18,
    })

    def __post_init__(self):
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.exports_dir = os.path.join(self.output_dir, 'dashboard_exports')
        os.makedirs(self.index_data_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)


class NIFTYSectorAttributionAnalyzer:
    """
    RQ4: Which stocks/sectors drive daily NIFTY movements and why?

    Streamlit-ready version (v3): saves clean CSV/JSON artifacts and plots to
    rq4_complete_results/ so the Streamlit page can render them.
    """

    def __init__(self, cfg: Optional[RQ4Config] = None):
        self.cfg = cfg or RQ4Config()
        self.sector_data: Dict[str, pd.DataFrame] = {}
        self.nifty_data: Optional[pd.DataFrame] = None
        self.contribution_df: Optional[pd.DataFrame] = None
        self.driver_analysis: Optional[pd.DataFrame] = None
        self.summary_stats: Optional[dict] = None

    # ------------------------ Data I/O ------------------------
    def _expected_files(self) -> Dict[str, str]:
        # Update this mapping to the indices you maintain
        return {
            'NIFTY50': os.path.join(self.cfg.index_data_dir, 'NIFTY50.csv'),
            'NIFTYBANK': os.path.join(self.cfg.index_data_dir, 'NIFTYBANK.csv'),
            'NIFTYIT': os.path.join(self.cfg.index_data_dir, 'NIFTYIT.csv'),
            'NIFTYAUTO': os.path.join(self.cfg.index_data_dir, 'NIFTYAUTO.csv'),
            'NIFTYDEFENSE': os.path.join(self.cfg.index_data_dir, 'NIFTYDEFENSE.csv'),
            'NIFTYENERGY': os.path.join(self.cfg.index_data_dir, 'NIFTYENERGY.csv'),
            'NIFTYFMCG': os.path.join(self.cfg.index_data_dir, 'NIFTYFMCG.csv'),
        }

    def load_sector_data(self, file_paths_dict: Optional[Dict[str, str]] = None):
        paths = file_paths_dict or self._expected_files()
        for sector, file_path in paths.items():
            if not os.path.exists(file_path):
                print(f'[WARN] Missing: {file_path}')
                continue
            try:
                df = pd.read_csv(file_path)
                # tolerate a few common schemas
                cols = {c.lower(): c for c in df.columns}
                # normalize columns -> 'date','close','volume'
                date_col = cols.get('date') or cols.get('datetime') or list(df.columns)[0]
                close_col = cols.get('close') or cols.get('adj close') or cols.get('close_price')
                volume_col = cols.get('volume') or cols.get('vol')

                df.rename(columns={date_col: 'date', close_col: 'close'}, inplace=True)
                if volume_col and volume_col in df.columns:
                    df.rename(columns={volume_col: 'volume'}, inplace=True)
                if 'volume' not in df.columns:
                    df['volume'] = np.nan

                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

                df['daily_return'] = df['close'].pct_change()
                df['log_return'] = np.log(df['close'] / df['close'].shift(1))
                df['volatility_5d'] = df['daily_return'].rolling(5).std()
                df['volatility_20d'] = df['daily_return'].rolling(20).std()
                df['volume_ma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']

                self.sector_data[sector] = df
                print(f'Loaded {sector}: {len(df)} rows')
            except Exception as e:
                print(f'[ERR] {sector}: {e}')

        if 'NIFTY50' in self.sector_data:
            self.nifty_data = self.sector_data['NIFTY50'].copy()
        else:
            raise ValueError('NIFTY50 data is required. Missing in provided paths.')

    # ------------------------ Core Computations ------------------------
    def calculate_sector_contributions(self) -> pd.DataFrame:
        if not self.sector_data or self.nifty_data is None:
            raise ValueError('Data not loaded. Call load_sector_data().')

        common_dates = self.nifty_data['date'].tolist()
        rows: List[dict] = []

        for date in common_dates:
            drow = {'date': date}
            nifty_ret = self.nifty_data.loc[self.nifty_data['date'] == date, 'daily_return'].iloc[0]
            drow['nifty_return'] = nifty_ret
            total_weighted = 0.0

            for sector, weight in self.cfg.estimated_weights.items():
                if sector == 'Other':
                    continue
                sdf = self.sector_data.get(sector)
                if sdf is None:
                    drow[f'{sector}_return'] = np.nan
                    drow[f'{sector}_contribution'] = 0.0
                    continue
                sret = sdf.loc[sdf['date'] == date, 'daily_return']
                if len(sret) and not pd.isna(sret.iloc[0]):
                    val = sret.iloc[0]
                    contrib = weight * val
                    drow[f'{sector}_return'] = val
                    drow[f'{sector}_contribution'] = contrib
                    total_weighted += contrib
                else:
                    drow[f'{sector}_return'] = np.nan
                    drow[f'{sector}_contribution'] = 0.0

            drow['Other_contribution'] = nifty_ret - total_weighted
            drow['explained_return'] = total_weighted
            drow['unexplained_return'] = drow['Other_contribution']
            rows.append(drow)

        self.contribution_df = pd.DataFrame(rows)
        return self.contribution_df

    def identify_key_drivers(self, percentile: float = 90.0) -> pd.DataFrame:
        if self.contribution_df is None:
            self.calculate_sector_contributions()

        df = self.contribution_df.copy()
        thresh = np.percentile(np.abs(df['nifty_return'].dropna()), percentile)
        extreme = df[np.abs(df['nifty_return']) >= thresh].copy()

        analyses = []
        contrib_cols = [c for c in df.columns if c.endswith('_contribution') and c != 'Other_contribution']

        for _, row in extreme.iterrows():
            impacts = {}
            for col in contrib_cols:
                sector = col.replace('_contribution', '')
                val = row[col]
                if pd.notna(val) and row['nifty_return'] != 0:
                    impacts[sector] = {
                        'contribution': val,
                        'impact_pct': abs(val / row['nifty_return']) * 100.0,
                        'sector_return': row.get(f'{sector}_return', np.nan),
                    }
            ordered = sorted(impacts.items(), key=lambda x: x[1]['impact_pct'], reverse=True)
            if ordered:
                top_sec, top_vals = ordered[0]
                top_contrib = top_vals['contribution']
                top_pct = top_vals['impact_pct']
            else:
                top_sec, top_contrib, top_pct = None, 0.0, 0.0

            analyses.append({
                'date': row['date'],
                'nifty_return': row['nifty_return'],
                'magnitude': abs(row['nifty_return']),
                'direction': 'Up' if row['nifty_return'] > 0 else 'Down',
                'top_driver': top_sec,
                'top_contribution': top_contrib,
                'top_impact_pct': top_pct,
            })

        self.driver_analysis = pd.DataFrame(analyses)
        return self.driver_analysis

    def attribution_summary_stats(self) -> dict:
        if self.contribution_df is None:
            self.calculate_sector_contributions()

        df = self.contribution_df.copy()
        stats = {
            'total_days': int(len(df)),
            'avg_daily_return': float(df['nifty_return'].mean() or 0),
            'return_volatility': float(df['nifty_return'].std() or 0),
        }
        var_n = df['nifty_return'].var() or 0
        var_e = df['explained_return'].var() or 0
        stats['explained_variance'] = float((var_e / var_n) if var_n else 0)

        sector_cols = [c for c in df.columns if c.endswith('_contribution')]
        sector_stats = {}
        for col in sector_cols:
            sec = col.replace('_contribution', '')
            s = df[col].dropna()
            sector_stats[sec] = {
                'avg_contribution': float(s.mean() or 0),
                'contribution_volatility': float(s.std() or 0),
                'positive_days': int((s > 0).sum()),
                'negative_days': int((s < 0).sum()),
                'max_positive_contribution': float((s.max() if len(s) else 0) or 0),
                'max_negative_contribution': float((s.min() if len(s) else 0) or 0),
                'total_absolute_contribution': float(s.abs().sum() or 0),
            }
        stats['sector_stats'] = sector_stats
        stats['volatility_ranking'] = sorted(
            [(k, v['contribution_volatility']) for k, v in sector_stats.items()],
            key=lambda x: x[1], reverse=True
        )

        self.summary_stats = stats
        return stats

    # ------------------------ Plots ------------------------
    def _savefig(self, figpath: str):
        plt.tight_layout()
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_timeseries(self):
        df = self.contribution_df.copy()
        figpath = os.path.join(self.cfg.plots_dir, 'nifty_sector_attribution_timeseries.png')
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        axes[0].plot(df['date'], df['nifty_return'] * 100, linewidth=1)
        axes[0].set_title('Daily NIFTY50 Returns (%)')
        axes[0].set_ylabel('Return (%)')
        axes[0].grid(alpha=0.3); axes[0].axhline(0, linestyle='--', alpha=0.5)

        contrib_cols = [c for c in df.columns if c.endswith('_contribution') and c != 'Other_contribution']
        labels = [c.replace('_contribution', '') for c in contrib_cols]
        vals = (df[contrib_cols].fillna(0) * 100).T.values
        axes[1].stackplot(df['date'], *vals, labels=labels, alpha=0.7)
        axes[1].plot(df['date'], df['nifty_return'] * 100, linewidth=2, label='Actual NIFTY')
        axes[1].set_title('Daily Sector Contributions to NIFTY Returns')
        axes[1].set_ylabel('Contribution (%)'); axes[1].legend(loc='upper left'); axes[1].grid(alpha=0.3)

        axes[2].plot(df['date'], df['explained_return'] * 100, label='Explained by Sectors')
        axes[2].plot(df['date'], df['unexplained_return'] * 100, label='Unexplained (Other)')
        axes[2].set_title('Explained vs Unexplained NIFTY Returns')
        axes[2].set_ylabel('Return (%)'); axes[2].set_xlabel('Date'); axes[2].legend(); axes[2].grid(alpha=0.3)

        self._savefig(figpath)

    def plot_sector_panels(self):
        df = self.contribution_df.copy()
        figpath = os.path.join(self.cfg.plots_dir, 'sector_contribution_analysis.png')
        contrib_cols = [c for c in df.columns if c.endswith('_contribution')]
        labels = [c.replace('_contribution', '') for c in contrib_cols]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        boxdata, boxlabels = [], []
        for col in contrib_cols:
            s = (df[col].dropna() * 100)
            if len(s): 
                boxdata.append(s); boxlabels.append(col.replace('_contribution', ''))
        if boxdata:
            axes[0,0].boxplot(boxdata, labels=boxlabels)
        axes[0,0].set_title('Distribution of Sector Contributions to NIFTY')
        axes[0,0].set_ylabel('Contribution (%)'); axes[0,0].tick_params(axis='x', rotation=45); axes[0,0].grid(alpha=0.3)

        avg_abs = [(df[c].abs().mean() * 100) for c in contrib_cols]
        axes[0,1].bar(labels, avg_abs, alpha=0.8)
        axes[0,1].set_title('Average Absolute Sector Contributions')
        axes[0,1].set_ylabel('Average |Contribution| (%)'); axes[0,1].tick_params(axis='x', rotation=45); axes[0,1].grid(alpha=0.3)

        ret_cols = [c for c in df.columns if c.endswith('_return') and c != 'nifty_return']
        retn_labels = [c.replace('_return', '') for c in ret_cols]
        vols = [(df[c].std() * 100) for c in ret_cols]
        axes[1,0].bar(retn_labels, vols, alpha=0.8)
        axes[1,0].set_title('Sector Return Volatility'); axes[1,0].set_ylabel('Volatility (% std)')
        axes[1,0].tick_params(axis='x', rotation=45); axes[1,0].grid(alpha=0.3)

        cum = df[contrib_cols].fillna(0).cumsum() * 100
        for col in contrib_cols:
            axes[1,1].plot(df['date'], cum[col], label=col.replace('_contribution', ''), alpha=0.9)
        axes[1,1].set_title('Cumulative Sector Contributions'); axes[1,1].set_ylabel('Cumulative (%)')
        axes[1,1].set_xlabel('Date'); axes[1,1].legend(loc='upper left'); axes[1,1].grid(alpha=0.3)

        self._savefig(figpath)

    def plot_extreme_days(self):
        if self.driver_analysis is None or self.driver_analysis.empty:
            return
        df = self.driver_analysis.copy()
        figpath = os.path.join(self.cfg.plots_dir, 'extreme_days_analysis.png')

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        top = df['top_driver'].value_counts()
        axes[0,0].bar(top.index.astype(str), top.values, alpha=0.8)
        axes[0,0].set_title('Most Frequent Top Drivers of Extreme Days')
        axes[0,0].set_ylabel('Frequency'); axes[0,0].tick_params(axis='x', rotation=45)

        axes[0,1].hist(df['top_impact_pct'], bins=15, alpha=0.8)
        axes[0,1].set_title('Distribution of Top Sector Impact %')
        axes[0,1].set_xlabel('Impact (%)'); axes[0,1].set_ylabel('Frequency')

        axes[1,0].scatter(df['nifty_return'] * 100, df['top_contribution'] * 100, alpha=0.8)
        axes[1,0].set_title('NIFTY Return vs Top Sector Contribution')
        axes[1,0].set_xlabel('NIFTY Return (%)'); axes[1,0].set_ylabel('Top Contribution (%)'); axes[1,0].grid(alpha=0.3)

        colors = ['red' if x < 0 else 'green' for x in df['nifty_return']]
        axes[1,1].scatter(df['date'], df['nifty_return'] * 100, c=colors, alpha=0.7, s=40)
        axes[1,1].set_title('Timeline of Extreme Movement Days')
        axes[1,1].set_xlabel('Date'); axes[1,1].set_ylabel('NIFTY Return (%)'); axes[1,1].grid(alpha=0.3)

        self._savefig(figpath)

    def plot_correlations(self):
        df = self.contribution_df.copy()
        figpath = os.path.join(self.cfg.plots_dir, 'sector_correlations.png')
        ret_cols = [c for c in df.columns if c.endswith('_return')]
        corr = df[ret_cols].corr()
        corr.index = [c.replace('_return', '') for c in corr.index]
        corr.columns = [c.replace('_return', '') for c in corr.columns]
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
        plt.title('Sector Return Correlations')
        self._savefig(figpath)


    # ------------------------ EDA (Inputs, Observations, Results) ------------------------
    def _ensure_eda_dirs(self):
        self.eda_dir = os.path.join(self.cfg.output_dir, 'eda_exports')
        os.makedirs(self.eda_dir, exist_ok=True)

    def eda_compute_and_export(self):
        """
        Generates EDA artifacts for Streamlit with 3 sections:
          1) Inputs     -> data coverage, nulls, sectors loaded
          2) Observations -> return distribution stats, rolling vol summary, correlations
          3) Results    -> extreme days & quick highlights

        Saves CSV/JSON to rq4_complete_results/eda_exports and a few PNGs to plots.
        """
        import json
        self._ensure_eda_dirs()

        # 1) INPUTS
        inputs_rows = []
        for sec, df in self.sector_data.items():
            row = {
                'sector': sec,
                'rows': int(len(df)),
                'date_min': str(df['date'].min()) if len(df) else None,
                'date_max': str(df['date'].max()) if len(df) else None,
                'null_close': int(df['close'].isna().sum()) if 'close' in df.columns else None,
                'null_volume': int(df['volume'].isna().sum()) if 'volume' in df.columns else None,
            }
            inputs_rows.append(row)
        pd.DataFrame(inputs_rows).to_csv(os.path.join(self.eda_dir, 'inputs_overview.csv'), index=False)

        # coverage vs NIFTY50
        cov = []
        if self.nifty_data is not None and len(self.nifty_data):
            base_dates = set(self.nifty_data['date'].dropna().astype(str))
            for sec, df in self.sector_data.items():
                sec_dates = set(df['date'].dropna().astype(str))
                cov.append({
                    'sector': sec,
                    'coverage_ratio_vs_nifty50': float(len(sec_dates & base_dates)) / max(1, float(len(base_dates))),
                    'missing_vs_nifty50_count': int(max(0, len(base_dates - sec_dates))),
                    'extra_vs_nifty50_count': int(max(0, len(sec_dates - base_dates))),
                })
        if cov:
            pd.DataFrame(cov).to_csv(os.path.join(self.eda_dir, 'inputs_coverage_vs_nifty50.csv'), index=False)

        # 2) OBSERVATIONS
        # Return distribution stats
        dist_rows = []
        for sec, df in self.sector_data.items():
            if 'daily_return' not in df.columns or df['daily_return'].dropna().empty:
                continue
            s = df['daily_return'].dropna()
            dist_rows.append({
                'sector': sec,
                'mean_ret': float(s.mean() or 0),
                'std_ret': float(s.std() or 0),
                'skew': float(s.skew() or 0),
                'kurtosis': float(s.kurtosis() or 0),
                'n': int(s.shape[0]),
            })
        dist_df = pd.DataFrame(dist_rows)
        if not dist_df.empty:
            dist_df.to_csv(os.path.join(self.eda_dir, 'observations_return_distribution_stats.csv'), index=False)

        # Volatility summary (5d, 20d)
        vol_rows = []
        for sec, df in self.sector_data.items():
            vol_rows.append({
                'sector': sec,
                'avg_vol_5d': float((df['volatility_5d'].mean() * 100) if 'volatility_5d' in df.columns and df['volatility_5d'].notna().any() else 0),
                'avg_vol_20d': float((df['volatility_20d'].mean() * 100) if 'volatility_20d' in df.columns and df['volatility_20d'].notna().any() else 0),
            })
        pd.DataFrame(vol_rows).to_csv(os.path.join(self.eda_dir, 'observations_volatility_summary.csv'), index=False)

        # Correlations (sector returns, not contributions)
        # Build a single DataFrame with aligned dates
        rets = []
        for sec, df in self.sector_data.items():
            if 'daily_return' in df.columns:
                tmp = df[['date', 'daily_return']].copy()
                tmp.columns = ['date', sec]
                rets.append(tmp)
        if rets:
            merged = rets[0]
            for nxt in rets[1:]:
                merged = pd.merge(merged, nxt, on='date', how='outer')
            merged = merged.sort_values('date')
            corr = merged.drop(columns=['date']).corr()
            corr.to_csv(os.path.join(self.eda_dir, 'observations_sector_return_corr.csv'))

        # 3) RESULTS
        # Extreme days for NIFTY50
        results = {}
        if self.nifty_data is not None and 'daily_return' in self.nifty_data.columns:
            nd = self.nifty_data[['date', 'daily_return']].dropna().copy()
            if not nd.empty:
                ups = nd.nlargest(5, 'daily_return')
                downs = nd.nsmallest(5, 'daily_return')
                ups.to_csv(os.path.join(self.eda_dir, 'results_top5_up_days.csv'), index=False)
                downs.to_csv(os.path.join(self.eda_dir, 'results_top5_down_days.csv'), index=False)
                results['top5_up'] = ups.to_dict(orient='records')
                results['top5_down'] = downs.to_dict(orient='records')

        # High-level highlights for narrative
        highlights = {}
        try:
            if not dist_df.empty:
                # Highest & lowest volatility sectors by std
                high_vol = dist_df.sort_values('std_ret', ascending=False).head(1)['sector'].iloc[0]
                low_vol = dist_df.sort_values('std_ret').head(1)['sector'].iloc[0]
                highlights['highest_vol_sector'] = high_vol
                highlights['lowest_vol_sector'] = low_vol
            if rets:
                # Most & least correlated pair (absolute)
                c = corr.abs().copy()
                np.fill_diagonal(c.values, np.nan)
                max_pair = np.unravel_index(np.nanargmax(c.values), c.shape)
                min_pair = np.unravel_index(np.nanargmin(c.values), c.shape)
                highlights['most_correlated_pair'] = (c.index[max_pair[0]], c.columns[max_pair[1]], float(c.values[max_pair]))
                highlights['least_correlated_pair'] = (c.index[min_pair[0]], c.columns[min_pair[1]], float(c.values[min_pair]))
            if self.summary_stats:
                highlights['explained_variance_pct'] = float(self.summary_stats.get('explained_variance', 0) * 100)
        except Exception:
            pass

        with open(os.path.join(self.eda_dir, 'results_highlights.json'), 'w') as f:
            json.dump({'highlights': highlights, 'results': results}, f, indent=2, default=str)

        # Simple EDA plots (reuse existing helpers)
        # Return histograms
        try:
            nsec = len(self.sector_data)
            if nsec:
                cols = 3
                rows_fig = int(np.ceil(nsec / cols))
                fig, axes = plt.subplots(rows_fig, cols, figsize=(5*cols, 3.5*rows_fig))
                axes = np.array(axes).reshape(rows_fig, cols)
                idx = 0
                for sec, df in self.sector_data.items():
                    ax = axes[idx // cols, idx % cols]
                    if 'daily_return' in df.columns and not df['daily_return'].dropna().empty:
                        ax.hist(df['daily_return'].dropna() * 100, bins=40, alpha=0.85)
                        ax.set_title(f'{sec} daily returns')
                        ax.set_xlabel('%'); ax.set_ylabel('Count')
                        ax.grid(alpha=0.3)
                    else:
                        ax.set_visible(False)
                    idx += 1
                for j in range(idx, rows_fig*cols):
                    axes[j // cols, j % cols].set_visible(False)
                self._savefig(os.path.join(self.cfg.plots_dir, 'eda_return_histograms.png'))
        except Exception:
            pass

        # Rolling vol overlay
        try:
            fig, ax = plt.subplots(figsize=(14, 6))
            for sec, df in self.sector_data.items():
                if 'volatility_20d' in df.columns and df['volatility_20d'].notna().any():
                    ax.plot(df['date'], df['volatility_20d'] * 100, label=sec, linewidth=1)
            ax.set_title('20D Rolling Volatility by Sector')
            ax.set_ylabel('Volatility (% std)'); ax.grid(alpha=0.3); ax.legend(loc='upper left', ncol=2, fontsize=8)
            self._savefig(os.path.join(self.cfg.plots_dir, 'eda_volatility_20d.png'))
        except Exception:
            pass

    # ------------------------ Exports ------------------------
    def save_artifacts(self):
        # CSVs for dashboard
        if self.contribution_df is not None:
            self.contribution_df.to_csv(os.path.join(self.cfg.exports_dir, 'nifty_sector_contributions.csv'), index=False)
        if self.driver_analysis is not None and not self.driver_analysis.empty:
            self.driver_analysis.to_csv(os.path.join(self.cfg.exports_dir, 'extreme_days_analysis.csv'), index=False)
        if self.summary_stats is not None:
            with open(os.path.join(self.cfg.exports_dir, 'summary_stats.json'), 'w') as f:
                json.dump(self.summary_stats, f, indent=2, default=str)

    # ------------------------ Pipeline ------------------------
    def run(self, file_paths: Optional[Dict[str, str]] = None):
        print('RQ4 pipeline starting…')
        self.load_sector_data(file_paths)
        # EDA (inputs, observations, results)
        self.eda_compute_and_export()
        self.calculate_sector_contributions()
        self.identify_key_drivers(percentile=90)
        self.attribution_summary_stats()

        # plots
        self.plot_timeseries()
        self.plot_sector_panels()
        self.plot_extreme_days()
        self.plot_correlations()

        # exports
        self.save_artifacts()

        print(f'Done. Artifacts at: {self.cfg.output_dir}')
        return {
            'contributions_path': os.path.join(self.cfg.exports_dir, 'nifty_sector_contributions.csv'),
            'drivers_path': os.path.join(self.cfg.exports_dir, 'extreme_days_analysis.csv'),
            'summary_path': os.path.join(self.cfg.exports_dir, 'summary_stats.json'),
            'plots_dir': self.cfg.plots_dir,
        }


if __name__ == '__main__':
    cfg = RQ4Config()  # customize paths if needed
    analyzer = NIFTYSectorAttributionAnalyzer(cfg)
    analyzer.run()  # expects CSVs under cfg.index_data_dir
