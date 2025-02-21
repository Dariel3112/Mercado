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
from imblearn.over_sampling import SMOTE  # Para balanceamento
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

# ========== FUNÇÕES AUXILIARES ==========

def time_based_split(df, test_size=0.2):
    """
    Separa o DataFrame em treino e teste respeitando a ordem temporal.
    test_size = fração de dados no conjunto de teste.
    """
    n = len(df)
    cutoff = int(n * (1 - test_size))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def plot_roc_curve_streamlit(y_true, y_score, model_name, ticker):
    """
    Plota a curva ROC no Streamlit para um modelo/ticker específico.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
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
    def __init__(self, tickers: list, start_date: str, end_date: str, forecast_shift: dict = None) -> None:
        """
        :param tickers: Lista de tickers.
        :param start_date: Data de início ('YYYY-MM-DD').
        :param end_date: Data de término ('YYYY-MM-DD').
        :param forecast_shift: Dicionário {ticker: dias} ou inteiro padrão para todos.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_shift = forecast_shift if forecast_shift else {ticker: 1 for ticker in tickers}
        self.data = {}
        self.models = {}          # {ticker: {model_name: best_model}}
        self.model_scores = {}    # {ticker: {model_name: test_accuracy}}
        self.prepared_data = {}
        self.report = ""
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_symbol(self, ticker: str) -> str:
        """
        Retorna o símbolo adequado para yfinance.
        Se terminar com dígito, adiciona .SA (ações brasileiras).
        """
        if "." in ticker or "=" in ticker:
            return ticker
        if ticker and ticker[-1].isdigit():
            return ticker + ".SA"
        return ticker

    def _check_data_continuity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Verifica e corrige lacunas nos dados via interpolação (ou forward fill, se preferir).
        """
        df.index = pd.to_datetime(df.index)
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        if len(date_range) != len(df):
            logging.warning(f"Dados incompletos detectados. Preenchendo com interpolação linear.")
            df = df.reindex(date_range).interpolate(method='linear')
        return df

    def fetch_data(self) -> None:
        """
        Busca dados históricos em paralelo e armazena em cache.
        """
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
        """
        Calcula indicadores técnicos (MA, Bollinger, MACD, RSI etc.).
        """
        for w in ma_windows:
            df[f'MA{w}'] = df['Close'].rolling(window=w).mean()

        df['Daily_Return'] = df['Close'].pct_change()
        df['RSI'] = self._calculate_rsi(df)

        # Bollinger
        std_window = ma_windows[0]
        df[f'STD{std_window}'] = df['Close'].rolling(window=std_window).std()
        df['BB_upper'] = df[f'MA{std_window}'] + (2 * df[f'STD{std_window}'])
        df['BB_lower'] = df[f'MA{std_window}'] - (2 * df[f'STD{std_window}'])

        # MACD
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
        """
        Prepara dados, fazendo split temporal e balanceamento com SMOTE se necessário.
        """
        for ticker, df in self.data.items():
            try:
                logging.info(f"Preparando dados para {ticker}...")
                features = [
                    'MA20', 'MA50', 'Daily_Return', 'RSI', 'Volume',
                    'BB_upper', 'BB_lower', 'MACD', 'Signal_Line'
                ]
                if not all(f in df.columns for f in features):
                    st.warning(f"Indicadores faltando para {ticker}.")
                    continue

                shift_val = self.forecast_shift[ticker]
                X_full = df[features].iloc[:-shift_val]
                y_full = np.where(
                    df['Close'].shift(-shift_val).iloc[:-shift_val] > df['Close'].iloc[:-shift_val],
                    1, 0
                )

                # Divide em treino e teste mantendo a ordem temporal
                df_features = pd.DataFrame(X_full)
                df_features['Target'] = y_full

                train_df, test_df = time_based_split(df_features, test_size=0.2)
                X_train = train_df[features]
                y_train = train_df['Target']
                X_test = test_df[features]
                y_test = test_df['Target']

                # Verifica balanceamento
                class_dist = pd.Series(y_train).value_counts(normalize=True)
                logging.info(f"Distribuição de classes (treino) para {ticker}: {class_dist.to_dict()}")

                # Balanceamento com SMOTE se minoria < 30%
                if min(class_dist) < 0.3:
                    smote = SMOTE(random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    logging.info(f"SMOTE aplicado para {ticker}.")

                # Escalonamento robusto
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                self.prepared_data[ticker] = {
                    'X_train': X_train_scaled,
                    'X_test': X_test_scaled,
                    'y_train': y_train.values,
                    'y_test': y_test.values,
                    'scaler': scaler,
                    'feature_names': features
                }
            except Exception as e:
                st.error(f"Erro ao preparar dados para {ticker}: {e}")

    def train_models(self) -> None:
        """
        Treina modelos usando TimeSeriesSplit para GridSearchCV e salva o melhor estimador.
        """
        for ticker, pdata in self.prepared_data.items():
            self.models[ticker] = {}
            self.model_scores[ticker] = {}
            X_train, y_train = pdata['X_train'], pdata['y_train']

            # Usamos TimeSeriesSplit para validação
            tscv = TimeSeriesSplit(n_splits=3)

            # RandomForest com GridSearch
            rf_params = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
            rf_grid = GridSearchCV(
                RandomForestClassifier(random_state=42),
                rf_params, cv=tscv, n_jobs=-1
            )
            rf_grid.fit(X_train, y_train)
            best_rf = rf_grid.best_estimator_
            self.models[ticker]['RandomForest'] = best_rf
            logging.info(f"RandomForest {ticker}: melhores params {rf_grid.best_params_}")

            # XGBoost com GridSearch
            xgb_params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
            xgb_grid = GridSearchCV(
                XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                xgb_params, cv=tscv, n_jobs=-1
            )
            xgb_grid.fit(X_train, y_train)
            best_xgb = xgb_grid.best_estimator_
            self.models[ticker]['XGBoost'] = best_xgb
            logging.info(f"XGBoost {ticker}: melhores params {xgb_grid.best_params_}")

            # Logistic Regression (simples, sem GridSearch)
            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            lr_model.fit(X_train, y_train)
            self.models[ticker]['LogisticRegression'] = lr_model

    def evaluate_models(self) -> None:
        """
        Avalia modelos (accuracy, classification_report, AUC, etc.) e gera relatório.
        """
        for ticker, model_dict in self.models.items():
            pdata = self.prepared_data[ticker]
            X_test = pdata['X_test']
            y_test = pdata['y_test']

            self.report += f"\n--- Resultados para {ticker} ---\n"
            for model_name, model in model_dict.items():
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                clf_report = classification_report(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                # Guarda a acurácia para uso posterior (ensemble ponderado)
                self.model_scores[ticker][model_name] = accuracy

                self.report += f"\nModelo: {model_name}\n"
                self.report += f"Acurácia: {accuracy:.2f}\n"
                self.report += f"Relatório de Classificação:\n{clf_report}\n"
                self.report += f"Matriz de Confusão:\n{cm}\n"

                # AUC e curva ROC
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                    auc_value = roc_auc_score(y_test, y_score)
                    self.report += f"AUC: {auc_value:.2f}\n"

                    # Plot da curva ROC no Streamlit
                    plot_roc_curve_streamlit(y_test, y_score, model_name, ticker)

                # Importâncias
                if hasattr(model, 'feature_importances_'):
                    fi_df = pd.DataFrame({
                        'Feature': pdata['feature_names'],
                        'Importance': model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)
                    self.report += f"Importância das Features:\n{fi_df.to_string(index=False)}\n"

    def _weighted_ensemble_predict(self, ticker: str, X_scaled: np.ndarray):
        """
        Faz uma previsão ponderada pelas acurácias de cada modelo.
        Retorna (pred_classe, prob_media).
        """
        model_dict = self.models[ticker]
        scores_dict = self.model_scores[ticker]

        # Calcula soma dos pesos
        total_weight = sum(scores_dict.values())

        prob_sum = 0.0
        for model_name, model in model_dict.items():
            weight = scores_dict[model_name]
            if hasattr(model, 'predict_proba'):
                p = model.predict_proba(X_scaled)[0][1]  # probabilidade de "1"
            else:
                # Se não tiver predict_proba, faz predição binária e converte
                p = float(model.predict(X_scaled)[0])
            prob_sum += (p * weight)

        # Probabilidade média ponderada
        prob_weighted = prob_sum / total_weight
        pred_class = 1 if prob_weighted >= 0.5 else 0
        return pred_class, prob_weighted

    def predict_daily(self) -> pd.DataFrame:
        """
        Previsões diárias com ensemble ponderado.
        """
        results = []
        for ticker, model_dict in self.models.items():
            try:
                df = self.data[ticker]
                features = self.prepared_data[ticker]['feature_names']
                # Seleciona a última linha
                latest_features = df[features].iloc[-1:]
                scaler = self.prepared_data[ticker]['scaler']
                X_scaled = scaler.transform(latest_features)

                # Faz a predição ensemble
                pred_class, prob_weighted = self._weighted_ensemble_predict(ticker, X_scaled)
                trend = "Alta" if pred_class == 1 else "Baixa"
                suggestion = "Comprar" if pred_class == 1 else "Vender"

                # Monta dicionário de saída
                row_dict = {
                    "Ticker": ticker,
                    "Predição_Ensemble": trend,
                    "Sugestão_Ensemble": suggestion,
                    "Probabilidade_Alta": f"{prob_weighted:.2f}",
                }

                # Para fins de comparação, salvamos as previsões individuais
                for model_name, model in model_dict.items():
                    m_pred = model.predict(X_scaled)[0]
                    row_dict[f"{model_name}_Tendencia"] = "Alta" if m_pred == 1 else "Baixa"

                # Preço atual e Valor Alvo simples
                current_price = df['Close'].iloc[-1]
                avg_return = df['Daily_Return'].mean()
                shift_val = self.forecast_shift[ticker]
                factor = (1 + avg_return) ** shift_val if pred_class == 1 else (1 - avg_return) ** shift_val

                row_dict["Preço_Atual"] = current_price
                row_dict["Valor_Alvo"] = current_price * factor

                results.append(row_dict)
            except Exception as e:
                st.error(f"Erro na previsão para {ticker}: {e}")
        pred_df = pd.DataFrame(results)
        self.report += "\n--- Previsões Diárias (Ensemble Ponderado) ---\n"
        self.report += pred_df.to_string(index=False)
        return pred_df

    def save_report(self) -> None:
        """
        Salva o relatório em arquivo TXT.
        """
        file_path = os.path.join(os.getcwd(), f"analysis_report_{datetime.now().strftime('%Y-%m-%d')}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.report)
        st.info(f"Relatório salvo em: {file_path}")

    def run_pipeline(self, run_fetch=True, run_prepare=True, run_train=True,
                     run_evaluate=True, run_predict=True) -> pd.DataFrame:
        """
        Executa o pipeline completo.
        """
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

st.title("Análise de Tendências - Versão Aprimorada")
st.info("Exemplo com TimeSeriesSplit, RobustScaler, AUC, Ensemble Ponderado, etc.")

st.sidebar.header("Configurações")
tickers_default = ['PETR4', 'VALE3', 'ITUB4', 'NVDA', 'USDBRL=X']
custom_tickers = st.sidebar.text_input("Tickers (separados por vírgula)", value=", ".join(tickers_default))
start_date_input = st.sidebar.date_input("Data de Início", date.today() - timedelta(days=365))
end_date_input = st.sidebar.date_input("Data de Término", date.today())
forecast_shifts = st.sidebar.text_input("Previsão por Ticker (ex.: PETR4:1,VALE3:3)", value="")

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
            for item in forecast_shifts.split(","):
                try:
                    tk, days = item.split(":")
                    forecast_dict[tk.strip()] = int(days.strip())
                except:
                    pass  # Ignora entradas inválidas
        analyzer = TrendAnalyzer(tickers, str(start_date_input), str(end_date_input), forecast_dict or None)
        pred_df = analyzer.run_pipeline(run_fetch, run_prepare, run_train, run_evaluate, run_predict)
        
        st.success("Análise concluída!")
        st.subheader("Relatório Completo")
        st.text_area("Relatório", analyzer.report, height=400)

        if pred_df is not None and not pred_df.empty:
            st.subheader("Previsões Diárias (Ensemble Ponderado)")
            st.dataframe(pred_df.style.format({"Preço_Atual": "{:.2f}", "Valor_Alvo": "{:.2f}"}))

            # Download CSV
            csv_data = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Baixar Previsões (CSV)",
                data=csv_data,
                file_name='previsoes.csv',
                mime='text/csv'
            )

            # Gráfico interativo com Plotly
            fig = px.bar(
                pred_df,
                x="Ticker", y="Valor_Alvo",
                color="Predição_Ensemble",
                hover_data=["Preço_Atual", "Probabilidade_Alta"],
                title="Previsões de Valor Alvo (Ensemble)"
            )
            st.plotly_chart(fig)
