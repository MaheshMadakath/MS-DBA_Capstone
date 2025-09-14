
# 3_RQ4_Sector_Attribution.py
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Do NOT call st.set_page_config here; main app sets it.

RESULTS_DIR = Path('rq4_complete_results')
EXPORTS = RESULTS_DIR / 'dashboard_exports'
PLOTS   = RESULTS_DIR / 'plots'

@st.cache_data(show_spinner=False)
def _load_rq4_exports():
    def try_read_csv(p):
        try:
            if p.exists():
                return pd.read_csv(p)
        except Exception as e:
            st.warning(f'Failed to read {p.name}: {e}')
        return pd.DataFrame()

    contrib = try_read_csv(EXPORTS / 'nifty_sector_contributions.csv')
    drivers = try_read_csv(EXPORTS / 'extreme_days_analysis.csv')

    stats = {}
    stats_path = EXPORTS / 'summary_stats.json'
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
        except Exception as e:
            st.warning(f'Failed to parse summary_stats.json: {e}')

    for df, dcol in [(contrib, 'date'), (drivers, 'date')]:
        if not df.empty and dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol])

    return contrib, drivers, stats

def page():
    st.title('RQ4: NIFTY Sector Attribution 🔍')
    st.caption("Which sectors drive NIFTY's daily moves, and how much do they explain?")

    contrib, drivers, stats = _load_rq4_exports()

    if contrib.empty:
        st.info('Run rq4_v3.py to generate rq4_complete_results/ before using this page.')
        return

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    total_days = int(stats.get('total_days', len(contrib)))
    avg_ret = float(stats.get('avg_daily_return', contrib['nifty_return'].mean() if 'nifty_return' in contrib.columns else 0))
    vol = float(stats.get('return_volatility', contrib['nifty_return'].std() if 'nifty_return' in contrib.columns else 0))
    explained_var = float(stats.get('explained_variance', 0))

    c1.metric('Trading days', f"{total_days:,}")
    c2.metric('Avg daily return', f"{avg_ret*100:.2f}%")
    c3.metric('Return volatility', f"{vol*100:.2f}%")
    c4.metric('Explained variance', f"{explained_var*100:.1f}%")

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        '📈 Timeseries & Stacked Contributions',
        '🏭 Sector Panels',
        '⚠️ Extreme Days',
        '🔗 Sector Correlations',
        '📦 Raw Data', '📊 EDA'
    ])

    with tab1:
        st.subheader('Daily NIFTY return vs sector contributions')
        img = PLOTS / 'nifty_sector_attribution_timeseries.png'
        if img.exists():
            st.image(str(img))
        else:
            st.info('Plot not found; rendering quick Plotly charts instead.')
            ts = contrib[['date', 'nifty_return']].dropna()
            st.plotly_chart(px.line(ts, x='date', y='nifty_return', title='NIFTY daily return'),
                            use_container_width=True)

    with tab2:
        st.subheader('Sector contribution panels')
        img = PLOTS / 'sector_contribution_analysis.png'
        if img.exists():
            st.image(str(img))
        else:
            st.info('Plot not found; showing simple summaries.')
            cols = [c for c in contrib.columns if c.endswith('_contribution') and c != 'Other_contribution']
            if cols:
                tmp = (contrib[cols].abs().mean().sort_values(ascending=False) * 100.0)
                dfbar = tmp.reset_index()
                dfbar.columns = ['sector', 'avg_abs_contribution_pct']
                dfbar['sector'] = dfbar['sector'].str.replace('_contribution', '', regex=False)
                st.plotly_chart(px.bar(dfbar, x='sector', y='avg_abs_contribution_pct',
                                       title='Average |Contribution| (%)'), use_container_width=True)

    with tab3:
        st.subheader('Extreme movement days (≥ 90th percentile by |return|)')
        img = PLOTS / 'extreme_days_analysis.png'
        if img.exists():
            st.image(str(img))
        else:
            st.info('Plot not found; showing table.')
        if not drivers.empty:
            st.dataframe(drivers.sort_values('date', ascending=False), use_container_width=True)
            if 'top_driver' in drivers.columns:
                freq = drivers['top_driver'].value_counts().reset_index()
                freq.columns = ['sector', 'count']
                st.plotly_chart(px.bar(freq, x='sector', y='count', title='Most frequent top drivers'),
                                use_container_width=True)

    with tab4:
        st.subheader('Cross-sector return correlations')
        img = PLOTS / 'sector_correlations.png'
        if img.exists():
            st.image(str(img))
        else:
            ret_cols = [c for c in contrib.columns if c.endswith('_return')]
            if len(ret_cols) >= 2:
                corr = contrib[ret_cols].corr()
                corr.index = [c.replace('_return', '') for c in corr.index]
                corr.columns = [c.replace('_return', '') for c in corr.columns]
                st.dataframe(corr.style.background_gradient(cmap='RdBu_r'), use_container_width=True)
            else:
                st.info('Not enough return columns for correlation matrix.')


    with tab5:
        st.subheader('Data exports (for downloads)')
        colA, colB = st.columns(2)
        with colA:
            st.markdown('**nifty_sector_contributions.csv**')
            st.dataframe(contrib.head(100), use_container_width=True, height=300)
            st.download_button('Download full CSV', contrib.to_csv(index=False).encode(),
                               file_name='nifty_sector_contributions.csv', mime='text/csv', key='dl_contrib_csv_tab5')
        with colB:
            if not drivers.empty:
                st.markdown('**extreme_days_analysis.csv**')
                st.dataframe(drivers.head(100), use_container_width=True, height=300)
                st.download_button('Download full CSV', drivers.to_csv(index=False).encode(),
                                   file_name='extreme_days_analysis.csv', mime='text/csv', key='dl_drivers_csv_tab5')

    with tab6:
        st.subheader('Exploratory Data Analysis (EDA)')
        eda_dir = RESULTS_DIR / 'eda_exports'

        # Load EDA CSV/JSON
        def _read(p):
            import pandas as pd, json
            if p.suffix.lower() == '.csv' and p.exists():
                return pd.read_csv(p)
            if p.suffix.lower() == '.json' and p.exists():
                return json.loads(p.read_text())
            return None

        inputs_overview = _read(eda_dir / 'inputs_overview.csv')
        coverage_vs_nifty = _read(eda_dir / 'inputs_coverage_vs_nifty50.csv')
        dist_stats = _read(eda_dir / 'observations_return_distribution_stats.csv')
        vol_summ = _read(eda_dir / 'observations_volatility_summary.csv')
        corr_csv = _read(eda_dir / 'observations_sector_return_corr.csv')
        highlights_json = _read(eda_dir / 'results_highlights.json')

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('### Inputs')
            if inputs_overview is not None:
                st.dataframe(inputs_overview, use_container_width=True, height=260)
            if coverage_vs_nifty is not None:
                st.dataframe(coverage_vs_nifty, use_container_width=True, height=220)
        with c2:
            st.markdown('### Observations')
            if dist_stats is not None:
                st.dataframe(dist_stats, use_container_width=True, height=240)
            if vol_summ is not None:
                st.dataframe(vol_summ, use_container_width=True, height=240)

        # Plots if available
        st.markdown('### Visuals')
        pi, pv, pr = (PLOTS / 'eda_return_histograms.png'), (PLOTS / 'eda_volatility_20d.png'), (PLOTS / 'sector_correlations.png')
        row1 = st.columns(2)
        with row1[0]:
            if pi.exists(): st.image(str(pi), caption='Return histograms (per sector)')
        with row1[1]:
            if pv.exists(): st.image(str(pv), caption='20D rolling volatility overlay')

        st.markdown('### Correlations')
        if corr_csv is not None:
            st.dataframe(corr_csv, use_container_width=True, height=320)

        st.markdown('### Results / Takeaways')
        if highlights_json is not None:
            hl = highlights_json.get('highlights', {})
            rsl = highlights_json.get('results', {})
            bullets = []
            if 'explained_variance_pct' in hl:
                bullets.append(f"- **Explained variance** by sector mix: {hl['explained_variance_pct']:.1f}% (attribution model)")
            if 'highest_vol_sector' in hl and 'lowest_vol_sector' in hl:
                bullets.append(f"- **Volatility**: {hl['highest_vol_sector']} is highest; {hl['lowest_vol_sector']} is lowest (by daily-return std)")
            if 'most_correlated_pair' in hl:
                a, b, v = hl['most_correlated_pair']
                bullets.append(f"- **Most correlated pair**: {a} & {b} (|ρ|≈{v:.2f})")
            if 'least_correlated_pair' in hl:
                a, b, v = hl['least_correlated_pair']
                bullets.append(f"- **Least correlated pair**: {a} & {b} (|ρ|≈{v:.2f})")
            if bullets:
                st.markdown("\n".join(bullets))
        
        st.subheader('Data exports (for downloads)')
        colA, colB = st.columns(2)
        with colA:
            st.markdown('**nifty_sector_contributions.csv**')
            st.dataframe(contrib.head(100), use_container_width=True, height=300)
            st.download_button('Download full CSV', contrib.to_csv(index=False).encode(),
                               file_name='nifty_sector_contributions.csv', mime='text/csv', key='dl_contrib_csv_tab6')
        with colB:
            if not drivers.empty:
                st.markdown('**extreme_days_analysis.csv**')
                st.dataframe(drivers.head(100), use_container_width=True, height=300)
                st.download_button('Download full CSV', drivers.to_csv(index=False).encode(),
                                   file_name='extreme_days_analysis.csv', mime='text/csv', key='dl_drivers_csv_tab6')

if __name__ == '__main__':
    page()
