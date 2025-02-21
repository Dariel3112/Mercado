# Mercado
# Relatorio-acoes
import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
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
        :param tickers: Lista de tickers. Para ações brasileiras que terminam com dígito, ".SA" será adicionado.
        :param start_date: Data de início (formato 'YYYY-MM-DD').
        :param end_date: Data de término (formato 'YYYY-MM-DD').
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.models = {}
        self.prepared_data = {}
        self.report = ""

    def _get_symbol(self, ticker: str) -> str:
        """
        Retorna o símbolo adequado para o yfinance.
        Se o ticker já contiver '.' ou '=', assume que está completo.
        Se o ticker terminar com um dígito (ex.: 'PETR4'), adiciona '.SA'; caso contrário, retorna o ticker.
        """
        if "." in ticker or "=" in ticker:
            return ticker
        if ticker and ticker[-1].isdigit():
            return ticker + ".SA"
        return ticker

    def fetch_data(self) -> None:
        """
        Busca dados históricos para cada ticker e calcula indicadores técnicos.
        """
        for ticker in self.tickers:
            symbol = self._get_symbol(ticker)
            try:
                st.write(f"Buscando dados para {ticker} ({symbol})...")
                df = yf.download(symbol, start=self.start_date, end=self.end_date)
                if df.empty:
                    st.warning(f"Nenhum dado retornado para {ticker}.")
                    continue
                df = self._calculate_technical_indicators(df)
                df.dropna(inplace=True)
                self.data[ticker] = df
            except Exception as e:
                st.error(f"Erro ao buscar dados para {ticker}: {e}")

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
                st.write(f"Preparando dados para {ticker}...")
                features = ['MA20', 'MA50', 'Daily_Return', 'RSI', 'Volume',
                            'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']
                if not all(feature in df.columns for feature in features):
                    st.warning(f"Indicadores faltando para {ticker}.")
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
                st.error(f"Erro ao preparar dados para {ticker}: {e}")

    def train_models(self) -> None:
        """
        Treina três modelos para cada ticker: RandomForest, XGBoost e Logistic Regression.
        """
        for ticker, pdata in self.prepared_data.items():
            self.models[ticker] = {}
            try:
                st.write(f"Treinando RandomForest para {ticker}...")
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(pdata['X_train'], pdata['y_train'])
                self.models[ticker]['RandomForest'] = rf_model
            except Exception as e:
                st.error(f"Erro ao treinar RandomForest para {ticker}: {e}")

            try:
                st.write(f"Treinando XGBoost para {ticker}...")
                xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                xgb_model.fit(pdata['X_train'], pdata['y_train'])
                self.models[ticker]['XGBoost'] = xgb_model
            except Exception as e:
                st.error(f"Erro ao treinar XGBoost para {ticker}: {e}")

            try:
                st.write(f"Treinando Logistic Regression para {ticker}...")
                lr_model = LogisticRegression(max_iter=1000, random_state=42)
                lr_model.fit(pdata['X_train'], pdata['y_train'])
                self.models[ticker]['LogisticRegression'] = lr_model
            except Exception as e:
                st.error(f"Erro ao treinar Logistic Regression para {ticker}: {e}")

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

    def predict_daily(self) -> pd.DataFrame:
        """
        Realiza previsões diárias utilizando cada modelo e sugere operações de compra ou venda.
        Retorna um DataFrame com os resultados para facilitar a visualização e o download.
        """
        results = []
        for ticker, model_dict in self.models.items():
            try:
                df = self.data[ticker]
                features = ['MA20', 'MA50', 'Daily_Return', 'RSI', 'Volume',
                            'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']
                latest_features = df[features].iloc[-1:]
                scaler = self.prepared_data[ticker]['scaler']
                latest_scaled = scaler.transform(latest_features)
                pred_dict = {"Ticker": ticker}
                for model_name, model in model_dict.items():
                    prediction = model.predict(latest_scaled)
                    trend = "Alta" if prediction[0] == 1 else "Baixa"
                    suggestion = "Comprar" if prediction[0] == 1 else "Vender"
                    pred_dict[f"{model_name}_Tendencia"] = trend
                    pred_dict[f"{model_name}_Sugestao"] = suggestion
                results.append(pred_dict)
            except Exception as e:
                st.error(f"Erro na previsão para {ticker}: {e}")
        pred_df = pd.DataFrame(results)
        # Acrescenta os resultados ao relatório em formato texto, se desejado
        self.report += "\n--- Previsões Diárias e Sugestões ---\n"
        for index, row in pred_df.iterrows():
            self.report += f"\nTicker: {row['Ticker']}\n"
            for model in ['RandomForest', 'XGBoost', 'LogisticRegression']:
                self.report += f"{model}: Tendência prevista: {row.get(f'{model}_Tendencia', 'N/A')}. Sugestão: {row.get(f'{model}_Sugestao', 'N/A')}.\n"
        return pred_df

    def save_report(self) -> None:
        """
        Salva o relatório de análise em um arquivo TXT com a data atual no nome.
        """
        file_path = os.path.join(os.getcwd(), f"analysis_report_{datetime.now().strftime('%Y-%m-%d')}.txt")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.report)
            st.info(f"Relatório salvo em: {file_path}")
        except Exception as e:
            st.error(f"Erro ao salvar relatório: {e}")

    def run_pipeline(self, run_fetch=True, run_prepare=True, run_train=True, run_evaluate=True, run_predict=True) -> pd.DataFrame:
        """
        Executa o pipeline completo conforme as etapas selecionadas e retorna o DataFrame de previsões.
        """
        if run_fetch:
            self.fetch_data()
        if run_prepare:
            self.prepare_data()
        if run_train:
            self.train_models()
        if run_evaluate:
            self.evaluate_models()
        pred_df = None
        if run_predict:
            pred_df = self.predict_daily()
        self.save_report()
        return pred_df

# Interface do Streamlit
st.title("Análise de Tendências - Interface Interativa")

st.sidebar.header("Configurações")
# Tickers padrão (você pode editar esta lista)
tickers_default = [
    'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'BBAS3', 'ABEV3', 'WEGE3', 'RENT3',
    'EQTL3', 'RADL3', 'VIVT3', 'HAPV3', 'RAIL3', 'SUZB3', 'LREN3', 'GGBR4',
    'CSNA3', 'EMBR3', 'JBSS3', 'PRIO3', 'KLBN11', 'YDUQ3', 'CCRO3', 'CPFE3',
    'EGIE3', 'BRFS3', 'BRAP4', 'SANB11', 'UGPA3', 'CMIG4', 'NVDA', 'IVV', 'VOO', 'USDBRL=X'
]
custom_tickers = st.sidebar.text_input("Insira os tickers (separados por vírgula)", value=", ".join(tickers_default))
start_date_input = st.sidebar.date_input("Data de Início", date.today() - timedelta(days=365))
end_date_input = st.sidebar.date_input("Data de Término", date.today())

st.sidebar.header("Etapas a Executar")
run_fetch = st.sidebar.checkbox("Buscar Dados", value=True)
run_prepare = st.sidebar.checkbox("Preparar Dados", value=True)
run_train = st.sidebar.checkbox("Treinar Modelos", value=True)
run_evaluate = st.sidebar.checkbox("Avaliar Modelos", value=True)
run_predict = st.sidebar.checkbox("Previsões Diárias", value=True)

if st.sidebar.button("Executar Análise Completa"):
    with st.spinner("Executando análise..."):
        # Processa os tickers informados
        tickers = [t.strip() for t in custom_tickers.split(",") if t.strip()]
        analyzer = TrendAnalyzer(tickers, str(start_date_input), str(end_date_input))
        pred_df = analyzer.run_pipeline(run_fetch=run_fetch, run_prepare=run_prepare, run_train=run_train,
                                        run_evaluate=run_evaluate, run_predict=run_predict)
        st.success("Análise concluída!")
        st.subheader("Relatório Completo")
        st.text_area("Relatório", analyzer.report, height=400)
        
        if pred_df is not None and not pred_df.empty:
            st.subheader("Previsões Diárias e Sugestões")
            st.dataframe(pred_df)
            # Permite o download em CSV
            csv_data = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Baixar Previsões (CSV)",
                data=csv_data,
                file_name='previsoes.csv',
                mime='text/csv'
            )
        else:
            st.warning("Nenhuma previsão foi gerada.")
