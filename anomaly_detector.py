import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class ExpenditureAnomalyDetector:
    """
    Comprehensive anomaly detection system for government expenditure data
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.scalers: dict[str, StandardScaler] = {}
        self.models: dict[str, IsolationForest] = {}

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['tv_date'] = pd.to_datetime(df['TV Date'], format='%d-%m-%Y', errors='coerce')
        df['bill_date'] = pd.to_datetime(df['Bill Date'], format='%d-%m-%Y', errors='coerce')
        df = df.sort_values(['Detail Head', 'tv_date'])
        df['deduction_amount'] = df['Gross Amount'] - df['Net Amount']
        df['deduction_rate'] = df['deduction_amount'] / df['Gross Amount'].clip(lower=1)
        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['day'] = df['tv_date'].dt.day
        df['month'] = df['tv_date'].dt.month
        df['weekday'] = df['tv_date'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_month_start'] = (df['day'] <= 7).astype(int)
        df['is_month_end'] = (df['day'] >= 24).astype(int)
        df['quarter'] = df['tv_date'].dt.quarter
        df['is_quarter_end'] = ((df['month'] % 3 == 0) & df['is_month_end']).astype(int)
        df['fiscal_month'] = ((df['month'] - 4) % 12) + 1
        df['fiscal_quarter'] = ((df['fiscal_month'] - 1) // 3) + 1
        df['is_fiscal_year_end'] = ((df['month'] == 3) & df['is_month_end']).astype(int)
        return df

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(['Detail Head', 'tv_date'])
        for lag in [1, 3, 7]:
            df[f'gross_lag_{lag}'] = df.groupby('Detail Head')['Gross Amount'].shift(lag)
            df[f'net_lag_{lag}'] = df.groupby('Detail Head')['Net Amount'].shift(lag)
        for window in [3, 7]:
            df[f'gross_roll_mean_{window}'] = df.groupby('Detail Head')['Gross Amount'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'gross_roll_std_{window}'] = df.groupby('Detail Head')['Gross Amount'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        df['prev_date'] = df.groupby('Detail Head')['tv_date'].shift(1)
        df['days_since_last'] = (df['tv_date'] - df['prev_date']).dt.days
        df['month_year'] = df['tv_date'].dt.to_period('M')
        df['cumsum_month'] = df.groupby(['Detail Head', 'month_year'])['Gross Amount'].cumsum()

        # Transaction frequency: count within past 7 days per Detail Head
        def _rolling_count_7d(s: pd.Series) -> pd.Series:
            s_sorted = s.sort_values()
            idx = s_sorted.index
            times = s_sorted.values
            counts = np.empty(len(times), dtype=int)
            start = 0
            for i in range(len(times)):
                while times[i] - times[start] > np.timedelta64(7, 'D'):
                    start += 1
                counts[i] = i - start + 1
            result = pd.Series(counts, index=idx)
            return result.reindex(s.index)

        df['trans_count_week'] = df.groupby('Detail Head')['tv_date'].transform(_rolling_count_7d)
        return df

    def create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        daily_totals = df.groupby('tv_date')['Gross Amount'].sum().reset_index()
        daily_totals.columns = ['tv_date', 'daily_total']
        df = df.merge(daily_totals, on='tv_date', how='left')
        df['prop_of_daily'] = df['Gross Amount'] / df['daily_total'].clip(lower=1)
        major_head_daily = df.groupby(['tv_date', 'Major Head'])['Gross Amount'].sum().reset_index()
        major_head_daily.columns = ['tv_date', 'Major Head', 'major_head_daily_total']
        df = df.merge(major_head_daily, on=['tv_date', 'Major Head'], how='left')
        df['detail_head_count_daily'] = df.groupby('tv_date')['Detail Head'].transform('count')
        df['spending_concentration'] = 1 / df['detail_head_count_daily'].clip(lower=1)
        return df

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def _zscore(s: pd.Series) -> pd.Series:
            std = s.std()
            denom = std if (pd.notna(std) and std > 1e-6) else 1e-6
            return (s - s.mean()) / denom

        df['gross_zscore'] = df.groupby('Detail Head')['Gross Amount'].transform(_zscore)
        df['deviation_from_mean'] = (df['Gross Amount'] - df['gross_roll_mean_7']) / df['gross_roll_std_7'].clip(lower=1)
        df['q1'] = df.groupby('Detail Head')['Gross Amount'].transform(lambda x: x.quantile(0.25))
        df['q3'] = df.groupby('Detail Head')['Gross Amount'].transform(lambda x: x.quantile(0.75))
        df['iqr'] = df['q3'] - df['q1']
        df['is_iqr_outlier'] = (
            (df['Gross Amount'] < (df['q1'] - 1.5 * df['iqr'])) |
            (df['Gross Amount'] > (df['q3'] + 1.5 * df['iqr']))
        ).astype(int)
        df['pct_change'] = df.groupby('Detail Head')['Gross Amount'].pct_change()
        df['abs_pct_change'] = df['pct_change'].abs()
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lag_cols = [col for col in df.columns if 'lag_' in col]
        for col in lag_cols:
            df[col] = df.groupby('Detail Head')[col].transform(lambda x: x.fillna(x.median()))
        roll_cols = [col for col in df.columns if 'roll_' in col]
        for col in roll_cols:
            df[col] = df.groupby('Detail Head')[col].transform(lambda x: x.ffill().bfill())
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df

    def select_features_for_model(self, df: pd.DataFrame) -> list[str]:
        feature_cols = [
            'Gross Amount', 'Net Amount', 'deduction_rate',
            'day', 'month', 'weekday', 'is_weekend', 'is_month_end',
            'is_quarter_end', 'fiscal_quarter', 'is_fiscal_year_end',
            'gross_lag_1', 'gross_lag_3', 'gross_lag_7',
            'net_lag_1', 'net_lag_3', 'net_lag_7',
            'gross_roll_mean_3', 'gross_roll_std_3',
            'gross_roll_mean_7', 'gross_roll_std_7',
            'days_since_last',
            'daily_total', 'prop_of_daily', 'major_head_daily_total',
            'spending_concentration',
            'gross_zscore', 'deviation_from_mean', 'is_iqr_outlier',
            'pct_change', 'abs_pct_change',
            'cumsum_month', 'trans_count_week'
        ]
        return [c for c in feature_cols if c in df.columns]

    def train_models(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        X_global = df[feature_cols].values
        self.scalers['global'] = StandardScaler()
        X_global_scaled = self.scalers['global'].fit_transform(X_global)
        self.models['global'] = IsolationForest(contamination=self.contamination, random_state=42, n_estimators=100)
        df['anomaly_global'] = self.models['global'].fit_predict(X_global_scaled)
        df['anomaly_score_global'] = self.models['global'].score_samples(X_global_scaled)

        df['anomaly_detail_head'] = 0
        df['anomaly_score_detail_head'] = 0.0
        for detail_head in df['Detail Head'].dropna().unique():
            mask = df['Detail Head'] == detail_head
            if mask.sum() < 10:
                continue
            X_dh = df.loc[mask, feature_cols].values
            scaler = StandardScaler()
            X_dh_scaled = scaler.fit_transform(X_dh)
            model_dh = IsolationForest(contamination=min(self.contamination, 0.5 / len(X_dh)), random_state=42, n_estimators=50)
            df.loc[mask, 'anomaly_detail_head'] = model_dh.fit_predict(X_dh_scaled)
            df.loc[mask, 'anomaly_score_detail_head'] = model_dh.score_samples(X_dh_scaled)
            self.scalers[detail_head] = scaler
            self.models[detail_head] = model_dh

        df['anomaly_temporal'] = np.where(
            (df['gross_zscore'].abs() > 3) |
            (df['deviation_from_mean'].abs() > 3) |
            (df['abs_pct_change'] > 5),
            -1, 1
        )
        return df

    def create_ensemble_score(self, df: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-6
        df['norm_score_global'] = (df['anomaly_score_global'] - df['anomaly_score_global'].min()) / (
            df['anomaly_score_global'].max() - df['anomaly_score_global'].min() + eps
        )
        df['norm_score_detail'] = (df['anomaly_score_detail_head'] - df['anomaly_score_detail_head'].min()) / (
            df['anomaly_score_detail_head'].max() - df['anomaly_score_detail_head'].min() + eps
        )
        df['ensemble_anomaly_score'] = (
            0.4 * df['norm_score_global'] +
            0.4 * df['norm_score_detail'] -
            0.2 * (df['anomaly_temporal'] == -1).astype(float)
        )
        threshold = df['ensemble_anomaly_score'].quantile(self.contamination)
        df['is_anomaly'] = (df['ensemble_anomaly_score'] <= threshold).astype(int)
        # Audit metrics: store threshold and selected quantiles for the score
        df['ensemble_threshold'] = threshold
        score_q = df['ensemble_anomaly_score'].quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        df['score_q01'] = float(score_q.loc[0.01])
        df['score_q05'] = float(score_q.loc[0.05])
        df['score_q10'] = float(score_q.loc[0.10])
        df['score_q25'] = float(score_q.loc[0.25])
        df['score_q50'] = float(score_q.loc[0.50])
        df['score_q75'] = float(score_q.loc[0.75])
        df['score_q90'] = float(score_q.loc[0.90])
        df['score_q95'] = float(score_q.loc[0.95])
        df['score_q99'] = float(score_q.loc[0.99])
        reasons = []
        for _, row in df.iterrows():
            reason = []
            if row['anomaly_global'] == -1:
                reason.append('Global outlier')
            if row['anomaly_detail_head'] == -1:
                reason.append(f"Unusual for {row['Detail Head']}")
            if row['anomaly_temporal'] == -1:
                reason.append('Temporal anomaly')
            if row['is_iqr_outlier'] == 1:
                reason.append('Statistical outlier')
            reasons.append('; '.join(reason) if reason else 'Normal')
        df['anomaly_reason'] = reasons
        return df

    def add_audit_details(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        # Quantiles of model scores (lower is more anomalous)
        df['score_global_quantile'] = df['anomaly_score_global'].rank(pct=True, ascending=True)
        df['score_detail_quantile'] = df.groupby('Detail Head')['anomaly_score_detail_head'].transform(
            lambda s: s.rank(pct=True, ascending=True)
        )

        # Temporal reason details
        temporal_details = []
        for _, row in df.iterrows():
            parts = []
            if abs(row.get('gross_zscore', 0.0)) > 3:
                parts.append(f"gross_zscore={row['gross_zscore']:.2f} (>3)")
            if abs(row.get('deviation_from_mean', 0.0)) > 3:
                parts.append(f"deviation_from_mean={row['deviation_from_mean']:.2f} (>3)")
            if row.get('abs_pct_change', 0.0) > 5:
                parts.append(f"abs_pct_change={row['abs_pct_change']:.2f} (>5)")
            temporal_details.append('; '.join(parts) if parts else '')
        df['temporal_reason_detail'] = temporal_details

        # Statistical reason (IQR) details
        iqr_details = []
        for _, row in df.iterrows():
            if int(row.get('is_iqr_outlier', 0)) == 1:
                q1 = row.get('q1', np.nan)
                q3 = row.get('q3', np.nan)
                iqr = row.get('iqr', np.nan)
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                val = row.get('Gross Amount', np.nan)
                side = 'above' if val > upper else 'below'
                iqr_details.append(
                    f"value={val:.2f} {side} fence; q1={q1:.2f}, q3={q3:.2f}, iqr={iqr:.2f}, lower={lower:.2f}, upper={upper:.2f}"
                )
            else:
                iqr_details.append('')
        df['statistical_reason_detail'] = iqr_details

        # Global IF details
        df['global_reason_detail'] = np.where(
            df['anomaly_global'] == -1,
            df.apply(lambda r: f"score={r['anomaly_score_global']:.6f}, quantile={r['score_global_quantile']:.4f}", axis=1),
            ''
        )

        # Detail-head IF details
        df['detail_reason_detail'] = np.where(
            df['anomaly_detail_head'] == -1,
            df.apply(lambda r: f"score={r['anomaly_score_detail_head']:.6f}, quantile_in_head={r['score_detail_quantile']:.4f}", axis=1),
            ''
        )

        # Top deviating features by standardized distance (exclude zscore feature itself)
        try:
            scaler = self.scalers.get('global')
            if scaler is not None and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                X = df[feature_cols].values.astype(float)
                mean = getattr(scaler, 'mean_', np.zeros(X.shape[1]))
                scale = getattr(scaler, 'scale_', np.ones(X.shape[1]))
                safe_scale = np.where(np.abs(scale) > 1e-6, scale, 1.0)
                Z = (X - mean) / safe_scale
                names = feature_cols
                exclude = set([c for c in names if 'zscore' in c])
                top_desc = []
                for i in range(Z.shape[0]):
                    if df.iloc[i]['is_anomaly'] != 1:
                        top_desc.append('')
                        continue
                    abs_z = np.abs(Z[i])
                    # Mask excluded features by setting their score to -inf
                    masked = abs_z.copy()
                    for j, n in enumerate(names):
                        if n in exclude:
                            masked[j] = -np.inf
                    top_idx = np.argsort(masked)[-3:][::-1]
                    items = []
                    for j in top_idx:
                        if masked[j] == -np.inf:
                            continue
                        items.append(f"{names[j]}: z={Z[i][j]:.2f}, val={X[i][j]:.4f}")
                    top_desc.append('; '.join(items))
                df['top_deviation_features'] = top_desc
        except Exception:
            df['top_deviation_features'] = ''

        # Mark whether a per-head model was trained for this head
        trained_heads = {k for k in self.models.keys() if k != 'global'}
        df['detail_model_trained'] = df['Detail Head'].isin(trained_heads).astype(int)

        return df

    def fit_predict(self, df: pd.DataFrame):
        print('Step 1: Preparing data...')
        df = self.prepare_data(df)
        print('Step 2: Creating temporal features...')
        df = self.create_temporal_features(df)
        print('Step 3: Creating lag features...')
        df = self.create_lag_features(df)
        print('Step 4: Creating aggregate features...')
        df = self.create_aggregate_features(df)
        print('Step 5: Creating statistical features...')
        df = self.create_statistical_features(df)
        print('Step 6: Handling missing values...')
        df = self.handle_missing_values(df)
        print('Step 7: Selecting features...')
        feature_cols = self.select_features_for_model(df)
        print(f'Using {len(feature_cols)} features for modeling')
        print('Step 8: Training models...')
        df = self.train_models(df, feature_cols)
        print('Step 9: Creating ensemble score...')
        df = self.create_ensemble_score(df)
        return df, feature_cols

    def get_anomaly_summary(self, df: pd.DataFrame) -> dict:
        anomalies = df[df['is_anomaly'] == 1]
        return {
            'total_transactions': int(len(df)),
            'total_anomalies': int(len(anomalies)),
            'anomaly_rate': float(len(anomalies) / len(df) * 100) if len(df) else 0.0,
            'anomalies_by_detail_head': anomalies['Detail Head'].value_counts().to_dict(),
            'anomaly_dates': anomalies['tv_date'].dt.date.unique().tolist(),
            'total_anomaly_amount': float(anomalies['Gross Amount'].sum()),
            'avg_anomaly_amount': float(anomalies['Gross Amount'].mean()) if len(anomalies) else 0.0,
            'top_anomalies': (
                anomalies.nsmallest(5, 'ensemble_anomaly_score')[
                    ['TV No.', 'tv_date', 'Detail Head', 'Gross Amount', 'anomaly_reason']
                ].to_dict('records')
                if len(anomalies) else []
            )
        }


def run_anomaly_detection(df: pd.DataFrame):
    detector = ExpenditureAnomalyDetector(contamination=0.02)  # Expect 10% anomalies
    df_with_anomalies, feature_cols = detector.fit_predict(df)
    # Add detailed audit info
    df_with_anomalies = detector.add_audit_details(df_with_anomalies, feature_cols)
    summary = detector.get_anomaly_summary(df_with_anomalies)

    # Prepare common output dirs and timestamp
    from pathlib import Path
    import datetime as _dt
    out_dir = Path('output')
    plots_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')

    print('\n' + '=' * 50)
    print('ANOMALY DETECTION RESULTS')
    print('=' * 50)
    print(f"Total Transactions: {summary['total_transactions']}")
    print(f"Anomalies Detected: {summary['total_anomalies']} ({summary['anomaly_rate']:.2f}%)")
    print('\nAnomalies by Detail Head:')
    for head, count in summary['anomalies_by_detail_head'].items():
        print(f'  - {head}: {count}')
    print(f"\nTotal Anomalous Amount: ₹{summary['total_anomaly_amount']:,.2f}")
    print(f"Average Anomalous Amount: ₹{summary['avg_anomaly_amount']:,.2f}")
    print('\nTop 5 Anomalies:')
    for i, anomaly in enumerate(summary['top_anomalies'], 1):
        print(f"{i}. TV {anomaly['TV No.']} on {anomaly['tv_date']}")
        print(f"   {anomaly['Detail Head']}: ₹{anomaly['Gross Amount']:,}")
        print(f"   Reason: {anomaly['anomaly_reason']}")

    # Print audit stats
    try:
        thr = float(df_with_anomalies['ensemble_threshold'].iloc[0])
        q01 = float(df_with_anomalies['score_q01'].iloc[0])
        q05 = float(df_with_anomalies['score_q05'].iloc[0])
        q10 = float(df_with_anomalies['score_q10'].iloc[0])
        q50 = float(df_with_anomalies['score_q50'].iloc[0])
        q90 = float(df_with_anomalies['score_q90'].iloc[0])
        q95 = float(df_with_anomalies['score_q95'].iloc[0])
        q99 = float(df_with_anomalies['score_q99'].iloc[0])
        print('\nAudit: Threshold and Score Quantiles')
        print(f"  Threshold (contam={detector.contamination:.2f}): {thr:.6f}")
        print(f"  q01={q01:.6f}, q05={q05:.6f}, q10={q10:.6f}, q50={q50:.6f}, q90={q90:.6f}, q95={q95:.6f}, q99={q99:.6f}")
    except Exception as _e:
        print(f"Warning: Failed to print audit stats: {_e}")

    # Export anomalies with detailed audit fields
    try:
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1].copy()
        export_cols = [
            'TV No.', 'TV Date', 'Bill Date', 'Major Head', 'Detail Head',
            'Gross Amount', 'Net Amount', 'deduction_amount', 'deduction_rate',
            'ensemble_anomaly_score', 'ensemble_threshold', 'is_anomaly', 'anomaly_reason',
            # component model outputs
            'anomaly_global', 'anomaly_score_global', 'norm_score_global',
            'anomaly_detail_head', 'anomaly_score_detail_head', 'norm_score_detail',
            'anomaly_temporal', 'is_iqr_outlier',
            # detailed reasons
            'temporal_reason_detail', 'statistical_reason_detail', 'global_reason_detail', 'detail_reason_detail',
            'score_global_quantile', 'score_detail_quantile', 'top_deviation_features', 'detail_model_trained',
            # audit quantiles
            'score_q01', 'score_q05', 'score_q10', 'score_q25', 'score_q50', 'score_q75', 'score_q90', 'score_q95', 'score_q99'
        ]
        export_cols = [c for c in export_cols if c in anomalies.columns]
        anomalies[export_cols].to_csv(out_dir / f'anomalies_{timestamp}.csv', index=False)
    except Exception as _e:
        print(f"Warning: Failed to export anomalies CSV: {_e}")

    # Export full scored dataframe
    try:
        full_export_path = out_dir / f'full_scored_{timestamp}.csv'
        df_with_anomalies.to_csv(full_export_path, index=False)
    except Exception as _e:
        print(f"Warning: Failed to export full dataframe CSV: {_e}")

    # Create seaborn plots for audit
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        # plots_dir and timestamp are already defined

        # 1) Score distribution with threshold
        plt.figure(figsize=(9, 5))
        ax = sns.histplot(data=df_with_anomalies, x='ensemble_anomaly_score', bins=50, kde=True, color='#4C78A8')
        thr = float(df_with_anomalies['ensemble_threshold'].iloc[0])
        plt.axvline(thr, color='red', linestyle='--', label=f'Threshold={thr:.4f}')
        plt.title('Ensemble Anomaly Score Distribution')
        plt.xlabel('ensemble_anomaly_score (lower = more anomalous)')
        plt.ylabel('count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f'score_distribution_{timestamp}.png', dpi=150)
        plt.close()

        # 2) Anomaly count by Detail Head
        plt.figure(figsize=(10, 6))
        by_head = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1]['Detail Head'].value_counts().head(20)
        sns.barplot(x=by_head.values, y=by_head.index, color='#72B7B2')
        plt.title('Anomalies by Detail Head (Top 20)')
        plt.xlabel('count')
        plt.ylabel('Detail Head')
        plt.tight_layout()
        plt.savefig(plots_dir / f'anomalies_by_detail_head_{timestamp}.png', dpi=150)
        plt.close()

        # 3) Time series of Gross Amount with anomalies highlighted
        plt.figure(figsize=(12, 6))
        temp_df = df_with_anomalies.sort_values('tv_date')
        sns.lineplot(x='tv_date', y='Gross Amount', data=temp_df, color='#9C755F', label='Gross Amount')
        anom_pts = temp_df[temp_df['is_anomaly'] == 1]
        if not anom_pts.empty:
            plt.scatter(anom_pts['tv_date'], anom_pts['Gross Amount'], color='red', s=12, label='Anomalies')
        plt.title('Gross Amount Over Time with Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Gross Amount')
        plt.tight_layout()
        plt.savefig(plots_dir / f'timeseries_anomalies_{timestamp}.png', dpi=150)
        plt.close()
    except Exception as _e:
        print(f"Warning: Failed to generate plots: {_e}")

    return df_with_anomalies, summary


if __name__ == '__main__':
    # Load your data
    df = pd.read_csv(r'D:\Office\ocr_jobs\SKIMS_expenditure.csv')
    # Run anomaly detection
    df_results, summary = run_anomaly_detection(df)
    print('Anomaly detection pipeline ready for execution!')