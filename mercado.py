# Mercado
# Relatorio-acoes
import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import streamlit as st

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrendAnalyzer:
    def __init__(self, tickers: list, start_date: str, end_date: str) -> None:
        """
        Inicializa o analisador de tendências.
        :param tickers: Lista de tickers. Para ações brasileiras sem sufixo, ".SA" será adicionado.
        :param start_date: Data de início (formato 'YYYY-MM-DD').
        :param end_date: Data de término (formato 'YYYY-MM-DD').
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        # Cada ticker terá um dicionário com os três modelos
        self.models = {}
        self.prepared_data = {}
        self.report = ""

    def _get_symbol(self, ticker: str) -> str:
        """
        Retorna o símbolo adequado para o yfinance.
        """
        if "." in ticker or "=" in ticker:
            return ticker
        return ticker + ".SA"

    def fetch_data(self) -> None:
        """
        Busca dados históricos para cada ticker e calcula indicadores técnicos.
        """
        for ticker in self.tickers:
            symbol = self._get_symbol(ticker)
            try:
                logging.info(f"Buscando dados para {ticker} ({symbol})...")
                df = yf.download(symbol, start=self.start_date, end=self.end_date)
                if df.empty:
                    logging.warning(f"Nenhum dado retornado para {ticker}.")
                    continue
                df = self._calculate_technical_indicators(df)
                df.dropna(inplace=True)
                self.data[ticker] = df
            except Exception as e:
                logging.error(f"Erro ao buscar dados para {ticker}: {e}")

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores técnicos: MA20, MA50, Retorno Diário, RSI, Bollinger Bands, MACD e Signal Line.
        """
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df['RSI'] = self._calculate_rsi(df)

        df['STD20'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['MA20'] + (2 * df['STD20'])
        df['BB_lower'] = df['MA20'] - (2 * df['STD20'])

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        return df

    def _calculate_rsi(self, df: pd.DataFrame, periods: int = 14) -> pd.Series:
        """
        Calcula o Índice de Força Relativa (RSI) utilizando método exponencial.
        """
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
        Prepara os dados para treinamento e teste, utilizando os indicadores técnicos como features.
        A variável alvo (y) indica se o fechamento do próximo período será de alta (1) ou baixa (0).
        """
        for ticker, df in self.data.items():
            try:
                logging.info(f"Preparando dados para {ticker}...")
                features = ['MA20', 'MA50', 'Daily_Return', 'RSI', 'Volume',
                            'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']
                if not all(feature in df.columns for feature in features):
                    logging.warning(f"Indicadores faltando para {ticker}.")
                    continue

                X = df[features].iloc[:-1]  # Exclui última linha sem target
                y = np.where(df['Close'].shift(-1).iloc[:-1] > df['Close'].iloc[:-1], 1, 0)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                self.prepared_data[ticker] = {
                    'X_train': X_train_scaled,
                    'X_test': X_test_scaled,
                    'y_train': y_train,
                    'y_test': y_test,
                    'scaler': scaler,
                    'feature_names': features
                }
            except Exception as e:
                logging.error(f"Erro ao preparar dados para {ticker}: {e}")

    def train_models(self) -> None:
        """
        Treina três modelos para cada ticker: RandomForest, XGBoost e Logistic Regression.
        """
        for ticker, pdata in self.prepared_data.items():
            self.models[ticker] = {}
            try:
                logging.info(f"Treinando RandomForest para {ticker}...")
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(pdata['X_train'], pdata['y_train'])
                self.models[ticker]['RandomForest'] = rf_model
            except Exception as e:
                logging.error(f"Erro ao treinar RandomForest para {ticker}: {e}")

            try:
                logging.info(f"Treinando XGBoost para {ticker}...")
                xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                xgb_model.fit(pdata['X_train'], pdata['y_train'])
                self.models[ticker]['XGBoost'] = xgb_model
            except Exception as e:
                logging.error(f"Erro ao treinar XGBoost para {ticker}: {e}")

            try:
                logging.info(f"Treinando Logistic Regression para {ticker}...")
                lr_model = LogisticRegression(max_iter=1000, random_state=42)
                lr_model.fit(pdata['X_train'], pdata['y_train'])
                self.models[ticker]['LogisticRegression'] = lr_model
            except Exception as e:
                logging.error(f"Erro ao treinar Logistic Regression para {ticker}: {e}")

    def evaluate_models(self) -> None:
        """
        Avalia os modelos treinados e gera um relatório comparativo.
        """
        for ticker, model_dict in self.models.items():
            pdata = self.prepared_data[ticker]
            self.report += f"\n--- Resultados para {ticker} ---\n"
            for model_name, model in model_dict.items():
                try:
                    y_pred = model.predict(pdata['X_test'])
                    accuracy = accuracy_score(pdata['y_test'], y_pred)
                    clf_report = classification_report(pdata['y_test'], y_pred)
                    cm = confusion_matrix(pdata['y_test'], y_pred)
                    
                    self.report += f"\nModelo: {model_name}\n"
                    self.report += f"Acurácia: {accuracy:.2f}\n"
                    self.report += "Relatório de Classificação:\n"
                    self.report += f"{clf_report}\n"
                    self.report += "Matriz de Confusão:\n"
                    self.report += f"{cm}\n"

                    if hasattr(model, 'feature_importances_'):
                        feature_importances = model.feature_importances_
                        fi_df = pd.DataFrame({
                            'Feature': pdata['feature_names'],
                            'Importance': feature_importances
                        }).sort_values(by='Importance', ascending=False)
                        self.report += "Importância das Features:\n"
                        self.report += f"{fi_df.to_string(index=False)}\n"
                except Exception as e:
                    self.report += f"Erro ao avaliar o modelo {model_name} para {ticker}: {e}\n"

    def predict_daily(self) -> None:
        """
        Realiza previsões diárias utilizando cada modelo e sugere operações de compra ou venda.
        """
        self.report += "\n--- Previsões Diárias e Sugestões ---\n"
        for ticker, model_dict in self.models.items():
            try:
                df = self.data[ticker]
                features = ['MA20', 'MA50', 'Daily_Return', 'RSI', 'Volume',
                            'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']
                latest_features = df[features].iloc[-1:]
                scaler = self.prepared_data[ticker]['scaler']
                latest_scaled = scaler.transform(latest_features)
                self.report += f"\nTicker: {ticker}\n"
                for model_name, model in model_dict.items():
                    prediction = model.predict(latest_scaled)
                    trend = "Alta" if prediction[0] == 1 else "Baixa"
                    suggestion = "Comprar" if prediction[0] == 1 else "Vender"
                    self.report += f"{model_name}: Tendência prevista: {trend}. Sugestão: {suggestion}.\n"
            except Exception as e:
                self.report += f"Erro na previsão para {ticker}: {e}\n"

    def save_report(self) -> None:
        """
        Salva o relatório de análise em um arquivo TXT com a data atual no nome.
        """
        file_path = os.path.join(os.getcwd(), f"analysis_report_{datetime.now().strftime('%Y-%m-%d')}.txt")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.report)
            logging.info(f"Relatório salvo em: {file_path}")
        except Exception as e:
            logging.error(f"Erro ao salvar relatório: {e}")

    def serve_report_streamlit(self) -> None:
        """
        Exibe o relatório utilizando Streamlit.
        """
        st.title("Relatório de Análise de Tendências")
        st.text_area("Relatório", self.report, height=600)

    def run_analysis(self) -> None:
        """
        Executa todo o pipeline: coleta de dados, preparação, treinamento, avaliação, previsão e salvamento do relatório.
        """
        self.fetch_data()
        self.prepare_data()
        self.train_models()
        self.evaluate_models()
        self.predict_daily()
        self.save_report()

if __name__ == "__main__":
    # Tickers do IBOVESPA (serão convertidos para ".SA" se necessário)
    tickers_ibovespa = [
        'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'BBAS3', 'ABEV3', 'WEGE3', 'RENT3',
        'EQTL3', 'RADL3', 'VIVT3', 'HAPV3', 'RAIL3', 'SUZB3', 'LREN3', 'GGBR4',
        'CSNA3', 'EMBR3', 'JBSS3', 'PRIO3', 'KLBN11', 'YDUQ3', 'CCRO3', 'CPFE3',
        'EGIE3', 'BRFS3', 'BRAP4', 'SANB11', 'UGPA3', 'CMIG4'
    ]
    
    # Tickers internacionais e do dólar frente ao real
    tickers_internacionais = [
        'NVDA', 'IVV', 'VOO', 'USDBRL=X'
    ]
    
    tickers = tickers_ibovespa + tickers_internacionais

    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    analyzer = TrendAnalyzer(tickers, start_date, end_date)
    analyzer.run_analysis()
    
    # Exibe o relatório via Streamlit
    analyzer.serve_report_streamlit()
