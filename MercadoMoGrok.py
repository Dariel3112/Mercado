import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import pickle

# Configuração do logging (sem mudanças)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"trend_analyzer_{datetime.now().strftime('%Y-%m-%d')}.log"),
        logging.StreamHandler()
    ]
)

# Classe TrendAnalyzer (sem mudanças, mantida como no código anterior)
class TrendAnalyzer:
    def __init__(self, tickers: list, start_date: str, end_date: str, forecast_shift: dict = None) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_shift = forecast_shift if forecast_shift else {ticker: 1 for ticker in tickers}
        self.data = {}
        self.models = {}
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
            logging.warning(f"Dados incompletos detectados. Preenchendo com interpolação.")
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
                if df.empty:
                    logging.warning(f"Nenhum dado retornado para {ticker}.")
                    return ticker, None
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
            df = self._check_data_continuity(df)
            df = self._calculate_technical_indicators(df)
            df.dropna(inplace=True)
            return ticker, df

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_single, self.tickers))
        for ticker, df in results:
            if df is not None:
                self.data[ticker] = df
            else:
                st.warning(f"Falha ao carregar dados para {ticker}.")

    def _calculate_technical_indicators(self, df: pd.DataFrame, ma_windows=(20, 50)) -> pd.DataFrame:
        for w in ma_windows:
            df[f'MA{w}'] = df['Close'].rolling(window=w).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df['RSI'] = self._calculate_rsi(df)
        std_window = ma_windows[0]
        df[f'STD{std_window}'] = df['Close'].rolling(window=std_window).std()
        df['BB_upper'] = df[f'MA{std_window}'] + (2 * df[f'STD{std_window}'])
        df['BB_lower'] = df[f'MA{std_window}'] - (2 * df[f'STD{std_window}'])
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        return df

    def _calculate_rsi(self, df: pd.DataFrame, periods: int = 14) -> pd.Series:
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/periods, min_periods=periods).mean()
        avg_loss = loss.ewm(alpha=1/periods, min_periods=periods).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def prepare_data(self) -> None:
        for ticker, df in self.data.items():
            try:
                logging.info(f"Preparando dados para {ticker}...")
                features = [f'MA{w}' for w in (20, 50)] + ['Daily_Return', 'RSI', 'Volume', 'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']
                if not all(f in df.columns for f in features):
                    st.warning(f"Indicadores faltando para {ticker}.")
                    continue
                shift_val = self.forecast_shift[ticker]
                X = df[features].iloc[:-shift_val]
                y = np.where(df['Close'].shift(-shift_val).iloc[:-shift_val] > df['Close'].iloc[:-shift_val], 1, 0)
                class_dist = pd.Series(y).value_counts(normalize=True)
                logging.info(f"Distribuição de classes para {ticker}: {class_dist.to_dict()}")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                if min(class_dist) < 0.3:
                    smote = SMOTE(random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    logging.info(f"SMOTE aplicado para {ticker}.")
                scaler = StandardScaler()
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
            X_train, y_train = pdata['X_train'], pdata['y_train']
            rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
            rf_model = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
            rf_model.fit(X_train, y_train)
            self.models[ticker]['RandomForest'] = rf_model.best_estimator_
            logging.info(f"RandomForest para {ticker}: Melhores parâmetros {rf_model.best_params_}")
            xgb_params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
            xgb_model = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), xgb_params, cv=5, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            self.models[ticker]['XGBoost'] = xgb_model.best_estimator_
            logging.info(f"XGBoost para {ticker}: Melhores parâmetros {xgb_model.best_params_}")
            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            lr_model.fit(X_train, y_train)
            self.models[ticker]['LogisticRegression'] = lr_model

    def evaluate_models(self) -> None:
        for ticker, model_dict in self.models.items():
            pdata = self.prepared_data[ticker]
            self.report += f"\n--- Resultados para {ticker} ---\n"
            for model_name, model in model_dict.items():
                y_pred = model.predict(pdata['X_test'])
                y_prob = model.predict_proba(pdata['X_test'])[:, 1] if hasattr(model, 'predict_proba') else None
                accuracy = accuracy_score(pdata['y_test'], y_pred)
                clf_report = classification_report(pdata['y_test'], y_pred)
                cm = confusion_matrix(pdata['y_test'], y_pred)
                self.report += f"\nModelo: {model_name}\nAcurácia: {accuracy:.2f}\nRelatório:\n{clf_report}\nMatriz de Confusão:\n{cm}\n"
                if y_prob is not None:
                    self.report += f"Probabilidade média de alta: {y_prob.mean():.2f}\n"
                if hasattr(model, 'feature_importances_'):
                    fi_df = pd.DataFrame({
                        'Feature': pdata['feature_names'],
                        'Importance': model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)
                    self.report += f"Importância das Features:\n{fi_df.to_string(index=False)}\n"

    def _majority_vote(self, values: list[str], tipo: str = "Tendencia") -> tuple[str, float]:
        positive_word = "Alta" if tipo == "Tendencia" else "Comprar"
        negative_word = "Baixa" if tipo == "Tendencia" else "Vender"
        count_pos = sum(1 for v in values if v == positive_word)
        total = len(values)
        confidence = count_pos / total if count_pos >= total / 2 else (total - count_pos) / total
        if count_pos == total:
            return f"Unânime {positive_word}", confidence
        elif count_pos > total / 2:
            return f"Maioria {positive_word} ({count_pos}x{total-count_pos})", confidence
        elif count_pos == 0:
            return f"Unânime {negative_word}", confidence
        else:
            return f"Maioria {negative_word} ({total-count_pos}x{count_pos})", confidence

    def predict_daily(self) -> pd.DataFrame:
        results = []
        for ticker, model_dict in self.models.items():
            try:
                df = self.data[ticker]
                features = self.prepared_data[ticker]['feature_names']
                latest_features = df[features].iloc[-1:]
                scaler = self.prepared_data[ticker]['scaler']
                latest_scaled = scaler.transform(latest_features)
                pred_dict = {"Ticker": ticker}
                current_price = df['Close'].iloc[-1]
                avg_daily_return = df['Daily_Return'].mean()
                shift_val = self.forecast_shift[ticker]
                tendencia_list, sugestao_list, prob_list = [], [], []
                for model_name, model in model_dict.items():
                    pred = model.predict(latest_scaled)[0]
                    prob = model.predict_proba(latest_scaled)[0][1] if hasattr(model, 'predict_proba') else 0.5
                    trend = "Alta" if pred == 1 else "Baixa"
                    suggestion = "Comprar" if pred == 1 else "Vender"
                    pred_dict[f"{model_name}_Tendencia"] = trend
                    pred_dict[f"{model_name}_Sugestao"] = suggestion
                    tendencia_list.append(trend)
                    sugestao_list.append(suggestion)
                    prob_list.append(prob if pred == 1 else 1 - prob)
                trend_result, trend_conf = self._majority_vote(tendencia_list, "Tendencia")
                sugg_result, _ = self._majority_vote(sugestao_list, "Sugestao")
                pred_dict["Tendencia_Comparacao"] = trend_result
                pred_dict["Sugestao_Comparacao"] = sugg_result
                pred_dict["Confiança"] = f"{trend_conf:.2f}"
                target_factor = (1 + avg_daily_return) ** shift_val if "Alta" in trend_result else (1 - avg_daily_return) ** shift_val
                pred_dict["Preço_Atual"] = current_price
                pred_dict["Valor_Alvo"] = current_price * target_factor
                results.append(pred_dict)
            except Exception as e:
                st.error(f"Erro na previsão para {ticker}: {e}")
        pred_df = pd.DataFrame(results)
        self.report += "\n--- Previsões Diárias ---\n" + pred_df.to_string()
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

# ========== INTERFACE STREAMLIT ATUALIZADA ==========

st.title("Análise de Tendências - Interface Interativa")
st.info("Nota: Este sistema usa yfinance, ideal para análises educacionais. Para trading profissional, considere fontes como Bloomberg ou Quandl.")

st.sidebar.header("Configurações")
tickers_default = ['PETR4', 'VALE3', 'ITUB4', 'NVDA', 'USDBRL=X']
custom_tickers = st.sidebar.text_input("Tickers (separados por vírgula)", value=", ".join(tickers_default))
start_date_input = st.sidebar.date_input("Data de Início", date.today() - timedelta(days=365))
end_date_input = st.sidebar.date_input("Data de Término", date.today())
forecast_shifts = st.sidebar.text_input(
    "Período de Previsão por Ticker (ex.: PETR4:1,VALE3:3, ou deixe vazio para 1 dia padrão)",
    value=""
)

st.sidebar.header("Etapas")
run_fetch = st.sidebar.checkbox("Buscar Dados", value=True)
run_prepare = st.sidebar.checkbox("Preparar Dados", value=True)
run_train = st.sidebar.checkbox("Treinar Modelos", value=True)
run_evaluate = st.sidebar.checkbox("Avaliar Modelos", value=True)
run_predict = st.sidebar.checkbox("Previsões Diárias", value=True)

if st.sidebar.button("Executar Análise"):
    with st.spinner("Executando..."):
        tickers = [t.strip() for t in custom_tickers.split(",") if t.strip()]
        forecast_dict = {}
        if forecast_shifts:
            try:
                for item in forecast_shifts.split(","):
                    if ":" in item:
                        ticker, days = item.split(":")
                        forecast_dict[ticker.strip()] = int(days.strip())
                    else:
                        st.warning(f"Formato inválido em '{item}'. Use 'TICKER:DIAS'. Ignorando este item.")
            except ValueError as e:
                st.error(f"Erro no formato de 'Período de Previsão': {e}. Use 'TICKER:DIAS' (ex.: PETR4:1,VALE3:3). Usando padrão de 1 dia.")
                forecast_dict = None
        analyzer = TrendAnalyzer(tickers, str(start_date_input), str(end_date_input), forecast_dict or None)
        pred_df = analyzer.run_pipeline(run_fetch, run_prepare, run_train, run_evaluate, run_predict)
        
        st.success("Análise concluída!")
        st.subheader("Relatório Completo")
        st.text_area("Relatório", analyzer.report, height=400)

        if pred_df is not None and not pred_df.empty:
            st.subheader("Previsões Diárias")
            st.dataframe(pred_df.style.format({"Preço_Atual": "{:.2f}", "Valor_Alvo": "{:.2f}"}))

            csv_data = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Baixar Previsões (CSV)", data=csv_data, file_name='previsoes.csv', mime='text/csv')

            fig = px.bar(pred_df, x="Ticker", y="Valor_Alvo", color="Tendencia_Comparacao",
                         hover_data=["Preço_Atual", "Confiança"], title="Previsões de Valor Alvo")
            st.plotly_chart(fig)
