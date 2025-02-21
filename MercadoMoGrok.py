import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import pickle

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"trend_analyzer_{datetime.now().strftime('%Y-%m-%d')}.log"),
        logging.StreamHandler()
    ]
)

def plot_roc_curve_streamlit(y_true, y_score, model_name, ticker):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='r')
    ax.set_title(f"Curva ROC - {model_name} ({ticker})")
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
    ax.legend()
    st.pyplot(fig)

class TrendAnalyzer:
    def __init__(self, tickers: list, start_date: str, end_date: str, forecast_shift: int) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_shift = forecast_shift
        self.data = {}
        self.models = {}
        self.model_scores = {}
        self.prepared_data = {}
        self.report = ""
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_symbol(self, ticker: str) -> str:
        if "." in ticker or "=" in ticker:
            return ticker
        if ticker and ticker[-1].isdigit():
            return ticker + ".SA"
        return ticker

    def _check_data_continuity(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.index)
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        if len(date_range) != len(df):
            logging.warning("Dados incompletos detectados. Preenchendo com interpolação linear.")
            df = df.reindex(date_range).interpolate(method='linear')
        return df

    def fetch_data(self) -> None:
        def fetch_single(ticker):
            symbol = self._get_symbol(ticker)
            cache_file = os.path.join(self.cache_dir, f"{symbol}_{self.start_date}_{self.end_date}.pkl")
            if os.path.exists(cache_file):
                logging.info(f"Carregando cache para {ticker} ({symbol})...")
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
            else:
                logging.info(f"Buscando dados para {ticker} ({symbol})...")
                df = yf.download(symbol, start=self.start_date, end=self.end_date)
                if not isinstance(df, pd.DataFrame) or df.empty:
                    logging.warning(f"Dados inválidos retornados para {ticker}: {type(df)}")
                    return ticker, None
                if 'Close' not in df.columns:
                    if 'Adj Close' in df.columns:
                        df['Close'] = df['Adj Close']
                        logging.info(f"Usando 'Adj Close' como 'Close' para {ticker}.")
                    else:
                        logging.warning(f"Coluna 'Close' não encontrada para {ticker}.")
                        return ticker, None
                # Verifica se há dados suficientes para médias móveis
                if len(df) < 50:  # MA50 requer pelo menos 50 linhas
                    logging.warning(f"Dados insuficientes para {ticker}: apenas {len(df)} linhas.")
                    return ticker, None
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
            df = self._check_data_continuity(df)
            df = self._calculate_technical_indicators(df)
            if df is None:
                return ticker, None
            df.dropna(inplace=True)
            if len(df) < 50:  # Verifica novamente após dropna
                logging.warning(f"Dados insuficientes após limpeza para {ticker}: {len(df)} linhas.")
                return ticker, None
            return ticker, df

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_single, self.tickers))
        for ticker, df in results:
            if df is not None:
                self.data[ticker] = df
            else:
                st.warning(f"Falha ao carregar dados para {ticker}.")

    def _calculate_technical_indicators(self, df: pd.DataFrame, ma_windows=(20, 50)) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            logging.error(f"Esperado um DataFrame, recebido: {type(df)}")
            return None
        if 'Close' not in df.columns:
            logging.error("Coluna 'Close' não encontrada no DataFrame.")
            return None
        # Garante que o índice seja ordenado e único
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        try:
            for w in ma_windows:
                df[f'MA{w}'] = df['Close'].rolling(window=w, min_periods=1).mean()
            df['Daily_Return'] = df['Close'].pct_change()
            df['RSI'] = self._calculate_rsi(df)

            std_window = ma_windows[0]
            df[f'STD{std_window}'] = df['Close'].rolling(window=std_window, min_periods=1).std()
            df['BB_upper'] = df[f'MA{std_window}'] + (2 * df[f'STD{std_window}'])
            df['BB_lower'] = df[f'MA{std_window}'] - (2 * df[f'STD{std_window}'])

            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            return df
        except Exception as e:
            logging.error(f"Erro ao calcular indicadores técnicos: {e}")
            return None

    def _calculate_rsi(self, df: pd.DataFrame, periods: int = 14) -> pd.Series:
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/periods, min_periods=1).mean()
        avg_loss = loss.ewm(alpha=1/periods, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def prepare_data(self) -> None:
        for ticker, df in self.data.items():
            try:
                logging.info(f"Preparando dados para {ticker}...")
                features = ['MA20', 'MA50', 'Daily_Return', 'RSI', 'Volume',
                            'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']
                if not all(f in df.columns for f in features):
                    st.warning(f"Indicadores faltando para {ticker}.")
                    continue
                if len(df) <= self.forecast_shift:
                    st.warning(f"Dados insuficientes para {ticker} com previsão de {self.forecast_shift} dias.")
                    continue

                X_full = df[features].iloc[:-self.forecast_shift]
                y_full = np.where(df['Close'].shift(-self.forecast_shift).iloc[:-self.forecast_shift] > df['Close'].iloc[:-self.forecast_shift], 1, 0)

                n = len(X_full)
                cutoff = int(n * 0.8)
                X_train, X_test = X_full.iloc[:cutoff], X_full.iloc[cutoff:]
                y_train, y_test = y_full[:cutoff], y_full[cutoff:]

                class_dist = pd.Series(y_train).value_counts(normalize=True)
                logging.info(f"Distribuição de classes (treino) para {ticker}: {class_dist.to_dict()}")

                if min(class_dist) < 0.3:
                    smote = SMOTE(random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    logging.info(f"SMOTE aplicado para {ticker}.")

                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                self.prepared_data[ticker] = {
                    'X_train': X_train_scaled, 'X_test': X_test_scaled,
                    'y_train': y_train, 'y_test': y_test,
                    'scaler': scaler, 'feature_names': features
                }
            except Exception as e:
                st.error(f"Erro ao preparar dados para {ticker}: {e}")

    def train_models(self) -> None:
        for ticker, pdata in self.prepared_data.items():
            self.models[ticker] = {}
            self.model_scores[ticker] = {}
            X_train, y_train = pdata['X_train'], pdata['y_train']
            tscv = TimeSeriesSplit(n_splits=3)

            rf_params = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
            rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=tscv, n_jobs=-1)
            rf_grid.fit(X_train, y_train)
            best_rf = rf_grid.best_estimator_
            self.models[ticker]['RandomForest'] = best_rf
            self.model_scores[ticker]['RandomForest'] = accuracy_score(pdata['y_test'], best_rf.predict(pdata['X_test']))
            logging.info(f"RandomForest {ticker}: melhores params {rf_grid.best_params_}")

            xgb_params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
            xgb_grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), xgb_params, cv=tscv, n_jobs=-1)
            xgb_grid.fit(X_train, y_train)
            best_xgb = xgb_grid.best_estimator_
            self.models[ticker]['XGBoost'] = best_xgb
            self.model_scores[ticker]['XGBoost'] = accuracy_score(pdata['y_test'], best_xgb.predict(pdata['X_test']))
            logging.info(f"XGBoost {ticker}: melhores params {xgb_grid.best_params_}")

            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            lr_model.fit(X_train, y_train)
            self.models[ticker]['LogisticRegression'] = lr_model
            self.model_scores[ticker]['LogisticRegression'] = accuracy_score(pdata['y_test'], lr_model.predict(pdata['X_test']))

    def evaluate_models(self) -> None:
        for ticker, model_dict in self.models.items():
            pdata = self.prepared_data[ticker]
            X_test, y_test = pdata['X_test'], pdata['y_test']
            self.report += f"\n--- Resultados para {ticker} ---\n"
            for model_name, model in model_dict.items():
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                clf_report = classification_report(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                self.report += f"\nModelo: {model_name}\nAcurácia: {accuracy:.2f}\nRelatório:\n{clf_report}\nMatriz de Confusão:\n{cm}\n"
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                    auc_value = roc_auc_score(y_test, y_score)
                    self.report += f"AUC: {auc_value:.2f}\n"
                    plot_roc_curve_streamlit(y_test, y_score, model_name, ticker)
                if hasattr(model, 'feature_importances_'):
                    fi_df = pd.DataFrame({
                        'Feature': pdata['feature_names'],
                        'Importance': model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)
                    self.report += f"Importância das Features:\n{fi_df.to_string(index=False)}\n"

    def _weighted_ensemble_predict(self, ticker: str, X_scaled: np.ndarray) -> tuple[int, float]:
        model_dict = self.models[ticker]
        scores_dict = self.model_scores[ticker]
        total_weight = sum(scores_dict.values())
        prob_sum = 0.0
        for model_name, model in model_dict.items():
            weight = scores_dict[model_name]
            if hasattr(model, 'predict_proba'):
                p = model.predict_proba(X_scaled)[0][1]
            else:
                p = float(model.predict(X_scaled)[0])
            prob_sum += (p * weight)
        prob_weighted = prob_sum / total_weight if total_weight > 0 else 0.5
        pred_class = 1 if prob_weighted >= 0.5 else 0
        return pred_class, prob_weighted

    def predict_daily(self) -> pd.DataFrame:
        results = []
        for ticker, model_dict in self.models.items():
            try:
                df = self.data[ticker]
                features = self.prepared_data[ticker]['feature_names']
                latest_features = df[features].iloc[-1:]
                scaler = self.prepared_data[ticker]['scaler']
                X_scaled = scaler.transform(latest_features)
                pred_class, prob_weighted = self._weighted_ensemble_predict(ticker, X_scaled)
                trend = "Alta" if pred_class == 1 else "Baixa"
                suggestion = "Comprar" if pred_class == 1 else "Vender"
                row_dict = {
                    "Ticker": ticker,
                    "Predição_Ensemble": trend,
                    "Sugestão_Ensemble": suggestion,
                    "Probabilidade_Alta": f"{prob_weighted:.2f}"
                }
                current_price = df['Close'].iloc[-1]
                avg_return = df['Daily_Return'].mean()
                factor = (1 + avg_return) ** self.forecast_shift if pred_class == 1 else (1 - avg_return) ** self.forecast_shift
                row_dict["Preço_Atual"] = current_price
                row_dict["Valor_Alvo"] = current_price * factor
                results.append(row_dict)
            except Exception as e:
                st.error(f"Erro na previsão para {ticker}: {e}")
        pred_df = pd.DataFrame(results)
        self.report += "\n--- Previsões Diárias (Ensemble) ---\n" + pred_df.to_string(index=False)
        return pred_df

    def save_report(self) -> None:
        file_path = os.path.join(os.getcwd(), f"analysis_report_{datetime.now().strftime('%Y-%m-%d')}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.report)
        st.info(f"Relatório salvo em: {file_path}")

    def run_pipeline(self, run_fetch=True, run_prepare=True, run_train=True, run_evaluate=True, run_predict=True) -> pd.DataFrame:
        if run_fetch:
            self.fetch_data()
        if run_prepare:
            self.prepare_data()
        if run_train:
            self.train_models()
        if run_evaluate:
            self.evaluate_models()
        pred_df = self.predict_daily() if run_predict else None
        self.save_report()
        return pred_df

# ========== INTERFACE STREAMLIT ==========

st.title("Análise de Tendências - Versão Final")
st.info("Usa yfinance (educacional). Para trading profissional, considere Bloomberg ou Quandl.")

st.sidebar.header("Configurações")
tickers_default = ['PETR4', 'VALE3', 'ITUB4', 'NVDA', 'USDBRL=X']
custom_tickers = st.sidebar.text_input("Tickers (separados por vírgula)", value=", ".join(tickers_default))
start_date_input = st.sidebar.date_input("Data de Início", date.today() - timedelta(days=365))
end_date_input = st.sidebar.date_input("Data de Término", date.today())
forecast_shift = st.sidebar.number_input("Período de Previsão (dias)", min_value=1, value=30, step=1)

st.sidebar.header("Etapas")
run_fetch = st.sidebar.checkbox("Buscar Dados", value=True)
run_prepare = st.sidebar.checkbox("Preparar Dados", value=True)
run_train = st.sidebar.checkbox("Treinar Modelos", value=True)
run_evaluate = st.sidebar.checkbox("Avaliar Modelos", value=True)
run_predict = st.sidebar.checkbox("Previsões Diárias", value=True)

if st.sidebar.button("Executar Análise"):
    with st.spinner("Executando..."):
        tickers = [t.strip() for t in custom_tickers.split(",") if t.strip()]
        analyzer = TrendAnalyzer(tickers, str(start_date_input), str(end_date_input), forecast_shift)
        pred_df = analyzer.run_pipeline(run_fetch, run_prepare, run_train, run_evaluate, run_predict)
        
        st.success("Análise concluída!")
        st.subheader("Relatório Completo")
        st.text_area("Relatório", analyzer.report, height=400)

        if pred_df is not None and not pred_df.empty:
            st.subheader(f"Previsões para {forecast_shift} dias (Ensemble Ponderado)")
            st.dataframe(pred_df.style.format({"Preço_Atual": "{:.2f}", "Valor_Alvo": "{:.2f}"}))

            csv_data = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Baixar Previsões (CSV)",
                data=csv_data,
                file_name='previsoes.csv',
                mime='text/csv'
            )
            fig = px.bar(
                pred_df,
                x="Ticker", y="Valor_Alvo",
                color="Predição_Ensemble",
                hover_data=["Preço_Atual", "Probabilidade_Alta"],
                title=f"Previsões de Valor Alvo ({forecast_shift} dias)"
            )
            st.plotly_chart(fig)
